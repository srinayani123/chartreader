"""LangFuse observability integration (LangFuse SDK v4).

Wraps the agent's tool calls and LLM calls with traces so we can:
  - View end-to-end traces of single queries
  - Track token usage and cost over time
  - Surface failure modes in a dashboard

Falls back to no-op if LANGFUSE_* env vars are missing — the rest of the
system runs normally without observability.

Note: LangFuse v4 uses an OTel-based observation model. The v2 `client.trace()`
method and v3 `client.start_span()` method were both removed. v4 uses
`client.start_observation(as_type="span", ...)`. This module wraps that API
and keeps the existing `trace_query` / `trace_step` context-manager interface
so the rest of the codebase doesn't need to change.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)


_client = None


def get_client():
    """Lazy-init LangFuse v4 client; returns None if env vars missing."""
    global _client
    if _client is not None:
        return _client
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not (public_key and secret_key):
        return None
    try:
        from langfuse import Langfuse
        _client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        logger.info("LangFuse observability enabled")
        return _client
    except Exception as exc:
        logger.warning("LangFuse init failed: %s — continuing without tracing", exc)
        return None


@contextmanager
def trace_query(
    question: str, **metadata: Any
) -> Generator[Optional[Any], None, None]:
    """Context manager that creates a top-level trace for one ChartReader query.

    Usage:
        with trace_query(question="...", ticker="AAPL") as trace:
            ... agent runs ...
            if trace: trace.update(output={"answer": ...})

    On LangFuse v4 this creates a root span observation; the entire trace is
    constructed around it. We expose .update() so callers using the v2 API
    continue to work without modification.
    """
    client = get_client()
    if client is None:
        yield None
        return

    span = None
    try:
        # v4 API: start_observation(as_type="span") returns a LangfuseSpan.
        # The first observation in a request becomes the root of an OTel trace.
        span = client.start_observation(
            as_type="span",
            name="chartreader_query",
            input={"question": question},
            metadata=metadata or None,
        )
    except Exception as exc:
        logger.warning("Failed to start LangFuse observation: %s", exc)
        yield None
        return

    try:
        yield _SpanWrapper(span)
    finally:
        try:
            span.end()
        except Exception:
            pass
        try:
            client.flush()
        except Exception:
            pass


@contextmanager
def trace_step(
    parent: Optional[Any],
    name: str,
    **input_data: Any,
) -> Generator[Optional[Any], None, None]:
    """Sub-span for a single step (vision extraction, news retrieval, etc.).

    `parent` should be a `_SpanWrapper` returned by `trace_query`, or None.
    """
    if parent is None:
        yield None
        return
    sub = None
    try:
        # v4 child observations are created via the parent span's
        # start_observation method.
        sub = parent._span.start_observation(
            as_type="span",
            name=name,
            input=input_data or None,
        )
    except Exception as exc:
        logger.warning("Failed to start sub-observation %s: %s", name, exc)
        yield None
        return

    try:
        yield _SpanWrapper(sub)
    finally:
        try:
            sub.end()
        except Exception:
            pass


class _SpanWrapper:
    """Tiny adapter around a LangFuse v4 LangfuseSpan.

    Exposes the same `.update(...)` interface our existing code uses, so the
    agent and eval code don't need to know which LangFuse SDK version is
    installed.
    """

    __slots__ = ("_span",)

    def __init__(self, span):
        self._span = span

    def update(self, **kwargs: Any) -> None:
        """Update span attributes (input, output, metadata, etc.)."""
        try:
            self._span.update(**kwargs)
        except Exception as exc:
            logger.warning("Failed to update LangFuse span: %s", exc)

    # Backward-compat alias for the v2 `.span(...)` method
    def span(self, name: str, **input_data: Any):
        try:
            return _SpanWrapper(
                self._span.start_observation(
                    as_type="span",
                    name=name,
                    input=input_data or None,
                )
            )
        except Exception as exc:
            logger.warning("Failed to create child span %s: %s", name, exc)
            return None
        