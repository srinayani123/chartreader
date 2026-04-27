"""Baseline 2: Vision-only.

Single LLM call with the chart image and question. The LLM sees the chart
and reasons about it, but has NO access to:
  - yfinance ground truth (no verification)
  - news retrieval (no Tavily / pgvector)
  - peer comparison data
  - structured tools

This isolates the value of the retrieval + verification layers. If the crew
beats this baseline on faithfulness/groundedness but the vision-only baseline
beats the crew on latency/cost, that's the right tradeoff to surface in the
README.

Refusal handling is via prompt — the LLM is told to refuse advice/prediction
questions, but there's no programmatic refusal classifier. This isolates the
value of the regex + Haiku refusal layer.
"""
from __future__ import annotations

import logging
import os
import time

import anthropic

from src.models.schemas import GroundedAnswer
from src.vision.extract import _image_to_anthropic_block

logger = logging.getLogger(__name__)


_VISION_ONLY_PROMPT = """You are analyzing a financial chart. Use ONLY what \
you can see in the chart image to answer.

You do NOT have access to:
- Real-time market data
- News articles
- Peer or sector comparisons
- Any external information

Rules:
- Describe what the chart shows.
- Do NOT predict where the price will go next.
- Do NOT recommend buying or selling.
- If asked for prediction or advice, refuse and briefly explain why.
- Be honest about what you can and cannot see in the chart.
- Do NOT invent specific numbers (P/E ratios, market caps, analyst targets,
  earnings figures) that are not visible in the chart itself.
"""


def _detect_refusal(text: str) -> bool:
    """Detect a true refusal vs. a descriptive answer with hedging.

    Same logic as baseline_retrieval_only._detect_refusal — kept as a duplicate
    here to avoid cross-baseline imports.
    """
    if not text:
        return False
    lower = text.lower().strip()

    strong_openers = (
        "i can't and won't",
        "i cannot and will not",
        "i can't provide",
        "i cannot provide",
        "i won't provide",
        "i can't advise",
        "i cannot advise",
        "i won't advise",
        "i can't make",
        "i cannot make",
        "i won't make",
        "i can't recommend",
        "i cannot recommend",
        "i won't recommend",
        "i can't predict",
        "i cannot predict",
        "i don't make price predictions",
        "i don't provide investment advice",
        "i don't predict",
        "i refuse to",
        "i decline to",
    )

    head = lower[:250]
    has_strong_opener = any(opener in head for opener in strong_openers)
    if not has_strong_opener:
        return False

    descriptive_pivots = (
        "what i can tell you",
        "what i can do",
        "what i can describe",
        "what i can share",
        "what i can offer",
        "here's what",
        "here is what",
        "the chart shows",
        "the chart appears",
        "the chart depicts",
        "based on the chart",
        "looking at the chart",
        "what the chart shows",
        "## what",
        "**what",
        "# what",
    )
    return not any(p in lower for p in descriptive_pivots)


async def vision_only_analyze(image_path: str, question: str) -> GroundedAnswer:
    """Single-call vision baseline. Same vision model as the crew, no tools."""
    started = time.time()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    model = os.getenv("MODEL_VISION", "claude-sonnet-4-5-20250929")
    if not api_key:
        return GroundedAnswer(answer="No API key", elapsed_seconds=0)

    client = anthropic.Anthropic(api_key=api_key)
    image_block = _image_to_anthropic_block(image_path)

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=1500,
            temperature=0.2,
            system=_VISION_ONLY_PROMPT,
            messages=[{
                "role": "user",
                "content": [image_block, {"type": "text", "text": question}],
            }],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    except Exception as exc:
        logger.warning("Vision-only baseline call failed: %s", exc)
        text = f"Error: {exc}"

    refused = _detect_refusal(text)

    return GroundedAnswer(
        answer=text.strip(),
        refused=refused,
        refusal_reason=text.strip() if refused else None,
        elapsed_seconds=round(time.time() - started, 2),
    )
