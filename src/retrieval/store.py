"""pgvector-backed news cache + semantic retrieval.

Stores news articles fetched via Tavily so the agent can:
  1. Avoid re-fetching the same query (cost savings)
  2. Run semantic retrieval over the cached corpus for evals
  3. Reproduce eval results without hitting Tavily's API every time

Uses Postgres + pgvector + sentence-transformers (local embedding model,
no API cost).
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime
from typing import Optional

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from src.models.schemas import NewsArticle

logger = logging.getLogger(__name__)


# 384-dim embeddings — small, fast, no GPU needed
_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (loads once, ~80MB)."""
    global _model
    if _model is None:
        logger.info("Loading embedding model %s (first run only)...", _EMBEDDING_MODEL_NAME)
        _model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    return _model


def _build_connection_string() -> str:
    return (
        f"host={os.getenv('PG_HOST', 'localhost')} "
        f"port={os.getenv('PG_PORT', '5432')} "
        f"dbname={os.getenv('PG_DATABASE', 'chartreader')} "
        f"user={os.getenv('PG_USER', 'postgres')} "
        f"password={os.getenv('PG_PASSWORD', 'chartreader')}"
    )


def init_schema() -> None:
    """Create the pgvector extension and news table if they don't exist.

    Idempotent — safe to call on every startup.
    """
    with psycopg.connect(_build_connection_string()) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS news_articles (
                    id SERIAL PRIMARY KEY,
                    url TEXT UNIQUE NOT NULL,
                    title TEXT,
                    published_at TIMESTAMP,
                    source TEXT,
                    snippet TEXT,
                    full_content TEXT,
                    ticker TEXT,
                    period_start DATE,
                    period_end DATE,
                    embedding vector({_EMBEDDING_DIM}),
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS news_articles_embedding_idx
                ON news_articles USING hnsw (embedding vector_cosine_ops);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS news_articles_ticker_period_idx
                ON news_articles (ticker, period_start, period_end);
            """)
        conn.commit()
    logger.info("pgvector schema initialized")


def cache_articles(
    articles: list[NewsArticle],
    *,
    ticker: Optional[str] = None,
    period_start: Optional[date] = None,
    period_end: Optional[date] = None,
) -> int:
    """Store articles in pgvector. Returns number newly inserted."""
    if not articles:
        return 0

    texts = [f"{a.title}\n\n{a.snippet}" for a in articles]
    embeddings = _get_model().encode(texts, show_progress_bar=False).tolist()

    inserted = 0
    with psycopg.connect(_build_connection_string()) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for article, emb in zip(articles, embeddings):
                try:
                    cur.execute("""
                        INSERT INTO news_articles
                        (url, title, published_at, source, snippet, full_content,
                         ticker, period_start, period_end, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (url) DO NOTHING
                    """, (
                        article.url, article.title, article.published_at,
                        article.source, article.snippet, article.full_content,
                        ticker, period_start, period_end, emb,
                    ))
                    if cur.rowcount > 0:
                        inserted += 1
                except Exception as exc:
                    logger.warning("Failed to cache %r: %s", article.url, exc)
        conn.commit()
    return inserted


def semantic_search(
    query: str,
    *,
    ticker: Optional[str] = None,
    k: int = 5,
) -> list[NewsArticle]:
    """Search the news cache by semantic similarity, optionally scoped to a ticker."""
    query_emb = _get_model().encode([query], show_progress_bar=False).tolist()[0]

    sql = """
        SELECT url, title, published_at, source, snippet, full_content
        FROM news_articles
        {where}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    params: list = []
    if ticker:
        sql = sql.format(where="WHERE ticker = %s")
        params.append(ticker)
    else:
        sql = sql.format(where="")
    params.extend([query_emb, k])

    out: list[NewsArticle] = []
    try:
        with psycopg.connect(_build_connection_string()) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(sql, params)
                for row in cur.fetchall():
                    url, title, published_at, source, snippet, full_content = row
                    out.append(NewsArticle(
                        url=url or "",
                        title=title or "",
                        published_at=published_at,
                        source=source,
                        snippet=snippet or "",
                        full_content=full_content,
                    ))
    except Exception as exc:
        logger.warning("Semantic search failed: %s", exc)
    return out


def count_articles() -> int:
    """Return total number of cached articles (for observability)."""
    try:
        with psycopg.connect(_build_connection_string()) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM news_articles")
                return cur.fetchone()[0]
    except Exception:
        return 0
