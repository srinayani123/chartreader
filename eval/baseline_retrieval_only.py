"""Baseline 3: Retrieval-only.

Pipeline:
  1. Vision LLM extracts a description of the chart (text only — what ticker,
     period, key features) but extracted prices are NOT verified against
     yfinance.
  2. Tavily retrieves news based on the extracted ticker and period.
  3. Final reasoning LLM call answers the question using the chart description
     + retrieved news, but without any structured ground-truth grounding.

This isolates the value of the deterministic verification step. If the crew
beats this baseline on visual_textual_consistency, that's evidence that
yfinance verification is doing real work (catching vision hallucinations).

Refusal: prompt-based + heuristic detection that distinguishes pure refusals
from descriptive answers with hedging language.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta

import anthropic

from src.data.news import search_news
from src.models.schemas import (
    Citation,
    GroundedAnswer,
    NewsArticle,
)
from src.vision.extract import extract_chart

logger = logging.getLogger(__name__)


_REASONING_PROMPT = """You are answering a question about a financial chart. \
You have access to:
- A description of the chart (what ticker, period, and key features the
  vision system extracted)
- A list of news articles retrieved for context

You do NOT have access to verified market data or peer comparisons.

Rules:
- Ground every factual claim in either the chart description or one of
  the news articles. Cite which article supports each claim.
- Do NOT predict where the price will go next.
- Do NOT recommend buying or selling.
- If asked for prediction or advice, refuse.
- Do NOT invent specific numbers (P/E, market cap, analyst targets) that
  are not in the chart description or news articles.
- Be honest about uncertainty — the chart's prices have NOT been verified
  against actual market data, so don't assert specific prices as fact;
  describe them as "the chart appears to show ~$X".
"""


def _detect_refusal(text: str) -> bool:
    """Detect a true refusal vs. a descriptive answer with hedging.

    A descriptive answer can legitimately contain phrases like "I can't predict
    where prices will go" while still answering the underlying question. We
    only flag refused=True when the answer opens with strong refusal language
    AND lacks substantive descriptive content.
    """
    if not text:
        return False

    lower = text.lower().strip()

    # Phrases that, when found at the START of an answer, suggest refusal
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

    # Pivot phrases indicate the answer refuses one part but provides
    # descriptive content for the rest — that's not a pure refusal.
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
    has_descriptive_pivot = any(p in lower for p in descriptive_pivots)

    return not has_descriptive_pivot


async def retrieval_only_analyze(image_path: str, question: str) -> GroundedAnswer:
    """Vision description + news retrieval, no verification, no structured tools."""
    started = time.time()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    model = os.getenv("MODEL_REASONING", "claude-sonnet-4-5-20250929")
    if not api_key:
        return GroundedAnswer(answer="No API key", elapsed_seconds=0)

    try:
        extraction = extract_chart(image_path)
    except Exception as exc:
        logger.warning("Retrieval-only extraction failed: %s", exc)
        return GroundedAnswer(
            answer=f"Vision extraction failed: {exc}",
            elapsed_seconds=round(time.time() - started, 2),
        )

    articles: list[NewsArticle] = []
    if extraction.ticker:
        try:
            from src.vision.verify import _parse_date_string
            start = _parse_date_string(
                extraction.period_start_str or ""
            ) if extraction.period_start_str else None
            end = _parse_date_string(
                extraction.period_end_str or ""
            ) if extraction.period_end_str else None
        except Exception:
            start = end = None

        if not start:
            start = date.today() - timedelta(days=180)
        if not end:
            end = date.today()

        try:
            query = f"{extraction.ticker} stock news {start.year}"
            articles = search_news(
                query, start_date=start, end_date=end, max_results=5
            )
        except Exception as exc:
            logger.warning("Retrieval-only news fetch failed: %s", exc)

    chart_desc = (
        f"Ticker: {extraction.ticker or 'unknown'}\n"
        f"Period: {extraction.period_start_str or 'unknown'} to "
        f"{extraction.period_end_str or 'unknown'}\n"
        f"Chart type: {extraction.chart_type.value}\n"
        f"Notable features: "
        + ("; ".join(extraction.notable_features) if extraction.notable_features
           else "none extracted")
    )

    news_section = "\n\n".join(
        f"[{i+1}] {a.title}\n  URL: {a.url}\n  Snippet: {a.snippet[:300]}"
        for i, a in enumerate(articles)
    ) or "(no news retrieved)"

    user_message = (
        f"Chart description:\n{chart_desc}\n\n"
        f"Retrieved news:\n{news_section}\n\n"
        f"Question: {question}"
    )

    client = anthropic.Anthropic(api_key=api_key)
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=1500,
            temperature=0.2,
            system=_REASONING_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    except Exception as exc:
        logger.warning("Retrieval-only reasoning call failed: %s", exc)
        text = f"Error: {exc}"

    refused = _detect_refusal(text)

    citations = [
        Citation(
            source_type="news",
            url=a.url,
            title=a.title,
            published_at=a.published_at,
            excerpt=a.snippet[:200],
        )
        for a in articles
    ]

    return GroundedAnswer(
        answer=text.strip(),
        refused=refused,
        refusal_reason=text.strip() if refused else None,
        citations=citations,
        n_news_retrieved=len(articles),
        elapsed_seconds=round(time.time() - started, 2),
    )
