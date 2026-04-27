"""Single-LLM baseline — naive approach for comparison.

Just gives the chart and question to one LLM call. No agent, no retrieval,
no verification. Used to demonstrate that the agentic system actually
beats a single shot on faithfulness / groundedness.
"""
from __future__ import annotations

import logging
import os
import time

import anthropic

from src.models.schemas import GroundedAnswer
from src.vision.extract import _image_to_anthropic_block

logger = logging.getLogger(__name__)


_BASELINE_PROMPT = """You are answering a question about a financial chart. \
Use only what you can see in the chart. Be concise.

Rules:
- Describe what the chart shows.
- Do NOT predict where the price will go next.
- Do NOT recommend buying or selling.
- If asked for a prediction or advice, refuse and explain why."""


async def baseline_analyze(image_path: str, question: str) -> GroundedAnswer:
    """One-shot single-call baseline."""
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
            system=_BASELINE_PROMPT,
            messages=[{
                "role": "user",
                "content": [image_block, {"type": "text", "text": question}],
            }],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    except Exception as exc:
        logger.warning("Baseline call failed: %s", exc)
        text = f"Error: {exc}"

    refused = (
        "i don't" in text.lower()
        or "cannot recommend" in text.lower()
        or "not able to predict" in text.lower()
    )

    return GroundedAnswer(
        answer=text.strip(),
        refused=refused,
        refusal_reason=text.strip() if refused else None,
        elapsed_seconds=round(time.time() - started, 2),
    )
