"""Vision extraction — reads a chart image and returns a structured ChartExtraction.

Uses Anthropic's vision-capable models. The output is a typed ChartExtraction;
the verification step (vision/verify.py) cross-checks numerical claims against
yfinance ground truth before the agent uses them.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
from datetime import date
from pathlib import Path
from typing import Optional

import anthropic

from src.models.schemas import ChartExtraction, ChartType, PricePoint

logger = logging.getLogger(__name__)


# Note: this template is .format()-ted at call time with today's date so the
# vision model has a recency anchor. Without this, charts that show only
# month/day labels (e.g. "4/22, 4/23, 4/27" with no year visible) lead the
# model to fall back on training-data priors and label them with old years
# (commonly 2024). Injecting today's date eliminates that hallucination.
_EXTRACTION_PROMPT_TEMPLATE = """You are reading a financial chart for a downstream agentic system.

CONTEXT — Today's date is {today_iso}. The chart was almost certainly captured
recently. When the chart's x-axis shows month/day labels WITHOUT an explicit
year (e.g., "4/22", "Apr 22", "9:00 AM" intraday), interpret those dates as
belonging to {today_year} (or the most recent past occurrence of that
month/day relative to today). Do NOT default to older years like 2023 or 2024
unless the chart explicitly shows that year.

If the chart's timeframe selector or axis clearly spans multiple years
(e.g., a "5Y" view, or visible year labels like "2022", "2023"), use the
year information that is actually present.

Extract STRUCTURED, FACTUAL information from this chart. Be conservative —
when you're not sure, say so via lower extraction_confidence rather than
inventing values.

Return ONLY a JSON object with this exact shape:

{{
  "ticker": "<ticker symbol if visible on chart, else null>",
  "period_start_str": "<as it appears on chart, e.g., 'Sep 2024' or '2024-09-01'. If no year is visible on the chart, use {today_year} unless context indicates otherwise. Else null.>",
  "period_end_str": "<as it appears on chart, with same year inference rule>",
  "chart_type": "line | candlestick | bar | area | unknown",
  "extracted_points": [
    {{"date_str": "<as on chart>", "value": <number>, "kind": "close | high | low | open"}}
  ],
  "notable_features": [
    "<observation 1: descriptive only — patterns, volume, key levels. NEVER predictions.>",
    "<observation 2>",
    "..."
  ],
  "extraction_confidence": <0.0 to 1.0>
}}

Rules:
- DESCRIPTIVE ONLY. Do NOT predict where the price will go. Do NOT recommend
  buying or selling. Just describe what the chart shows.
- If the chart is unclear, lower extraction_confidence and extract fewer points.
- Pull at most 8 representative price points (start, end, notable highs/lows,
  inflection points). Quality over quantity.
- For dates, copy them as they appear on the chart axis — but if the chart
  shows no year, infer year as described in CONTEXT above. Do NOT silently
  default to an older year.
- For "notable_features", focus on FACTUAL observations: "Volume spiked on
  X date", "Price made a higher high on Y", "Pattern resembles ascending
  triangle". Do NOT include "this suggests..." or "likely to..." statements.
"""


def _build_extraction_prompt() -> str:
    """Build the extraction prompt with today's date injected."""
    today = date.today()
    return _EXTRACTION_PROMPT_TEMPLATE.format(
        today_iso=today.isoformat(),
        today_year=today.year,
    )


def _image_to_anthropic_block(image_path: str) -> dict:
    """Convert a local file path or base64 data URL to an Anthropic image block."""
    if image_path.startswith("data:"):
        # Already a data URL
        header, _, b64 = image_path.partition(",")
        mime = "image/png"
        m = re.match(r"data:([^;]+);base64", header)
        if m:
            mime = m.group(1)
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": mime, "data": b64},
        }

    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Chart image not found: {image_path}")

    suffix = p.suffix.lower().lstrip(".")
    mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "webp": "image/webp"}
    mime = mime_map.get(suffix, "image/png")
    data = base64.standard_b64encode(p.read_bytes()).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": mime, "data": data},
    }


def _safe_json(text: str) -> dict:
    """Tolerantly parse JSON from LLM output (may include code fences)."""
    if not text:
        return {}
    text = text.strip()
    if text.startswith("```"):
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(text[s:e + 1])
        except json.JSONDecodeError:
            pass
    return {}


def extract_chart(
    image_path: str,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ChartExtraction:
    """Read a chart image, return a structured ChartExtraction."""
    model_id = model or os.getenv("MODEL_VISION", "claude-sonnet-4-5-20250929")
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY required for chart extraction")

    client = anthropic.Anthropic(api_key=key)
    image_block = _image_to_anthropic_block(image_path)
    extraction_prompt = _build_extraction_prompt()

    resp = client.messages.create(
        model=model_id,
        max_tokens=2000,
        temperature=0.0,
        messages=[{
            "role": "user",
            "content": [image_block, {"type": "text", "text": extraction_prompt}],
        }],
    )
    raw = "".join(b.text for b in resp.content if hasattr(b, "text"))
    data = _safe_json(raw)

    # Tolerant shape normalization
    points = []
    for p in data.get("extracted_points", []) or []:
        try:
            points.append(PricePoint(
                date_str=str(p.get("date_str", "")).strip(),
                value=float(p.get("value", 0.0)),
                kind=str(p.get("kind", "close")).strip().lower() or "close",
            ))
        except Exception as exc:
            logger.warning("Skipping malformed price point %r: %s", p, exc)

    chart_type_raw = str(data.get("chart_type", "unknown")).strip().lower()
    try:
        chart_type = ChartType(chart_type_raw)
    except ValueError:
        chart_type = ChartType.UNKNOWN

    confidence_raw = data.get("extraction_confidence", 0.5)
    try:
        confidence = max(0.0, min(1.0, float(confidence_raw)))
    except (TypeError, ValueError):
        confidence = 0.5

    features = [str(f).strip() for f in (data.get("notable_features") or []) if str(f).strip()]

    return ChartExtraction(
        ticker=(str(data["ticker"]).strip().upper()
                if data.get("ticker") else None),
        period_start_str=data.get("period_start_str"),
        period_end_str=data.get("period_end_str"),
        chart_type=chart_type,
        extracted_points=points,
        notable_features=features,
        extraction_confidence=confidence,
    )
