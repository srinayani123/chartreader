"""Tools the ChartReader analyst agent calls.

Each tool is a typed Python function that Pydantic AI exposes to the LLM
via function calling. The LLM decides when to call which tool; the typed
inputs/outputs keep behavior strict.
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from typing import Optional

from pydantic_ai import RunContext

from src.data.news import search_news
from src.data.peers import compute_peer_comparison
from src.data.prices import compute_period_return, fetch_price_history
from src.models.schemas import (
    ChartContext,
    ChartExtraction,
    NewsArticle,
    PeerData,
    PriceVerification,
    RefusalCheck,
)
from src.refusal.classifier import check_refusal as _check_refusal_impl
from src.retrieval.cache import get_cached, put_cached
from src.retrieval.store import cache_articles, semantic_search
from src.vision.extract import extract_chart
from src.vision.verify import verify_extraction

logger = logging.getLogger(__name__)


def _parse_period_dates(extraction: ChartExtraction) -> tuple[Optional[date], Optional[date]]:
    """Best-effort parse of the chart's period dates from extraction.

    Note: `end` parsing inherits its year fallback from `start` if start
    parsed successfully. This handles the common case where the chart's
    x-axis labels start_str with a year ("Apr 22, 2026") but end_str is
    bare ("Apr 27") — without this, a bare end_str would fall back on
    today.year independently and could end up in a different year than start.
    """
    from src.vision.verify import _parse_date_string
    start = _parse_date_string(extraction.period_start_str or "") if extraction.period_start_str else None
    # Pass `start` as the year-fallback context for end parsing — preserves
    # year consistency between start and end when end_str has no year.
    end = _parse_date_string(extraction.period_end_str or "", period_start=start) if extraction.period_end_str else None
    return start, end


# ----------------------------------------------------------------------------
# Tool implementations — these are registered with the agent in analyst.py
# ----------------------------------------------------------------------------


async def check_refusal(ctx: RunContext[ChartContext]) -> RefusalCheck:
    """Check whether the user's question is a prediction or advice question
    that the agent must refuse. Call this FIRST before doing any other work.
    Returns a RefusalCheck — if should_refuse is true, stop and return a
    refusal answer immediately."""
    return _check_refusal_impl(ctx.deps.question)


async def extract_chart_data(ctx: RunContext[ChartContext]) -> ChartExtraction:
    """Read the chart image visually and return a structured ChartExtraction
    with extracted price points, ticker (if visible), period, chart type,
    and notable descriptive features. Always call this before any tool
    that needs chart data."""
    extraction = extract_chart(ctx.deps.image_path)
    # Mutate context so later tools can read what was extracted
    ctx.deps.extracted_data = extraction
    if extraction.ticker:
        ctx.deps.ticker = extraction.ticker
    start, end = _parse_period_dates(extraction)
    if start:
        ctx.deps.period_start = start
    if end:
        ctx.deps.period_end = end
    return extraction


async def verify_against_yfinance(
    ctx: RunContext[ChartContext],
    ticker: Optional[str] = None,
    period_start: Optional[str] = None,
    period_end: Optional[str] = None,
) -> PriceVerification:
    """Cross-check the chart's extracted prices against yfinance ground truth
    for the given ticker and period. If you don't pass ticker/dates, uses
    whatever extract_chart_data found. Returns a PriceVerification with
    MAE and reliability flag."""
    extraction = ctx.deps.extracted_data
    if extraction is None:
        # Run extraction first if the agent skipped that step
        extraction = await extract_chart_data(ctx)

    t = ticker or ctx.deps.ticker or extraction.ticker
    if not t:
        return PriceVerification(
            ticker="UNKNOWN",
            period_start=date.today() - timedelta(days=90),
            period_end=date.today(),
            n_points_failed=len(extraction.extracted_points),
            discrepancies=["No ticker available — cannot verify against yfinance."],
            is_reliable=False,
        )

    if period_start:
        start = date.fromisoformat(period_start)
    elif ctx.deps.period_start:
        start = ctx.deps.period_start
    else:
        start = date.today() - timedelta(days=90)

    if period_end:
        end = date.fromisoformat(period_end)
    elif ctx.deps.period_end:
        end = ctx.deps.period_end
    else:
        end = date.today()

    return verify_extraction(extraction, t, start, end)


async def get_news(
    ctx: RunContext[ChartContext],
    query: str,
    period_start: Optional[str] = None,
    period_end: Optional[str] = None,
    max_results: int = 5,
) -> list[NewsArticle]:
    """Retrieve relevant news for the query, optionally filtered to a date
    range. Use this to find what was happening in the news during the chart's
    period. Returns NewsArticle objects with URLs and snippets the agent must
    cite when using their content."""
    # Eval-mode short-circuit: if the cache has results, use them
    cached = get_cached(query, ctx.deps.ticker)
    if cached is not None:
        return cached

    start = date.fromisoformat(period_start) if period_start else None
    end = date.fromisoformat(period_end) if period_end else None
    articles = search_news(
        query, start_date=start, end_date=end, max_results=max_results
    )

    if articles:
        # Cache to file (eval reproducibility) AND pgvector (production semantic search)
        try:
            put_cached(query, articles, ticker=ctx.deps.ticker)
        except Exception as exc:
            logger.warning("File cache write failed: %s", exc)
        try:
            cache_articles(
                articles,
                ticker=ctx.deps.ticker,
                period_start=start,
                period_end=end,
            )
        except Exception as exc:
            logger.warning("pgvector cache write failed: %s", exc)
    return articles


async def get_peer_comparison(
    ctx: RunContext[ChartContext],
    base_ticker: Optional[str] = None,
    peer_tickers: Optional[list[str]] = None,
    sector_etf: Optional[str] = None,
    period_start: Optional[str] = None,
    period_end: Optional[str] = None,
) -> PeerData:
    """Compute period returns for the base ticker, its peers, and (optionally)
    a sector ETF. Returns a PeerData with returns and brief notes about
    relative performance. Use this when the question is comparative or asks
    about sector-wide moves."""
    t = base_ticker or ctx.deps.ticker
    if not t:
        return PeerData(
            base_ticker="UNKNOWN",
            peer_tickers=[],
            period_start=ctx.deps.period_start or date.today() - timedelta(days=90),
            period_end=ctx.deps.period_end or date.today(),
            notes=["No ticker available — cannot compute peer comparison."],
        )

    start = date.fromisoformat(period_start) if period_start else (
        ctx.deps.period_start or date.today() - timedelta(days=90)
    )
    end = date.fromisoformat(period_end) if period_end else (
        ctx.deps.period_end or date.today()
    )

    return compute_peer_comparison(
        t, peer_tickers or [], start, end, sector_etf=sector_etf
    )
