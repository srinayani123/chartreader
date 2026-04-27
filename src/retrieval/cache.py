"""JSON file cache for news — used during eval for reproducibility.

When running eval, we want the same news results every time (no Tavily API
flakiness or moving search results). This module checkpoints retrieved
news to disk under eval/test_set/news_cache/, keyed by query hash.

In production runs, the agent uses pgvector + live Tavily.
In eval runs, the agent uses this file cache.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.models.schemas import NewsArticle

logger = logging.getLogger(__name__)


def _cache_dir() -> Path:
    p = Path(os.getenv("EVAL_TEST_SET_PATH", "eval/test_set")) / "news_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _key(query: str, ticker: Optional[str] = None) -> str:
    """Stable hash of a query + ticker for filename safety."""
    raw = f"{query}|{ticker or ''}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def get_cached(query: str, ticker: Optional[str] = None) -> Optional[list[NewsArticle]]:
    """Return cached results if present, else None."""
    f = _cache_dir() / f"{_key(query, ticker)}.json"
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        out: list[NewsArticle] = []
        for r in data.get("results", []):
            published = r.get("published_at")
            if isinstance(published, str):
                try:
                    published = datetime.fromisoformat(published)
                except ValueError:
                    published = None
            out.append(NewsArticle(
                url=r.get("url", ""),
                title=r.get("title", ""),
                published_at=published,
                source=r.get("source"),
                snippet=r.get("snippet", ""),
                full_content=r.get("full_content"),
            ))
        return out
    except Exception as exc:
        logger.warning("Failed to read cache %s: %s", f, exc)
        return None


def put_cached(
    query: str,
    articles: list[NewsArticle],
    ticker: Optional[str] = None,
) -> None:
    """Persist results to the file cache for reproducible eval."""
    f = _cache_dir() / f"{_key(query, ticker)}.json"
    payload = {
        "query": query,
        "ticker": ticker,
        "cached_at": datetime.utcnow().isoformat(),
        "results": [
            {
                "url": a.url,
                "title": a.title,
                "published_at": a.published_at.isoformat() if a.published_at else None,
                "source": a.source,
                "snippet": a.snippet,
                "full_content": a.full_content,
            }
            for a in articles
        ],
    }
    f.write_text(json.dumps(payload, indent=2), encoding="utf-8")
