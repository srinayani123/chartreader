"""ChartReader analyst agent — Pydantic AI assembly.

Brings together: typed deps (ChartContext), typed result (GroundedAnswer),
system prompt, and tool registrations.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

from pydantic_ai import Agent

from src.agents.prompts import ANALYST_SYSTEM_PROMPT
from src.agents.tools import (
    check_refusal,
    extract_chart_data,
    get_news,
    get_peer_comparison,
    verify_against_yfinance,
)
from src.models.schemas import ChartContext, GroundedAnswer
from src.observability.tracing import trace_query

logger = logging.getLogger(__name__)


def _build_agent() -> Agent[ChartContext, GroundedAnswer]:
    """Build the Pydantic AI agent with deps + tools registered."""
    model_name = os.getenv("MODEL_REASONING", "claude-sonnet-4-5-20250929")

    agent = Agent(
        f"anthropic:{model_name}",
        deps_type=ChartContext,
        output_type=GroundedAnswer,
        system_prompt=ANALYST_SYSTEM_PROMPT,
        retries=int(os.getenv("AGENT_RETRIES", "1")),
    )

    # Register tools — Pydantic AI introspects type hints to build the
    # function-calling schema for the LLM.
    agent.tool(check_refusal)
    agent.tool(extract_chart_data)
    agent.tool(verify_against_yfinance)
    agent.tool(get_news)
    agent.tool(get_peer_comparison)

    return agent


# Module-level singleton — agents are cheap to construct but we want to share
# them across requests in the MCP server.
_analyst: Optional[Agent[ChartContext, GroundedAnswer]] = None


def get_analyst() -> Agent[ChartContext, GroundedAnswer]:
    global _analyst
    if _analyst is None:
        _analyst = _build_agent()
    return _analyst


async def analyze_chart(image_path: str, question: str) -> GroundedAnswer:
    """Top-level entry point: analyze a chart and answer a question about it.

    Used by both the Streamlit UI and the MCP server.
    """
    started = time.time()
    ctx = ChartContext(image_path=image_path, question=question)

    with trace_query(question, image_path=image_path) as trace:
        analyst = get_analyst()
        try:
            result = await analyst.run(question, deps=ctx)
            answer = result.output
        except Exception as exc:
            logger.exception("Agent run failed: %s", exc)
            return GroundedAnswer(
                answer=f"ChartReader hit an error: {exc}",
                refused=False,
                confidence_notes=(
                    "An internal error occurred. This is not a refusal — "
                    "the system failed to complete the analysis."
                ),
                elapsed_seconds=round(time.time() - started, 2),
            )

        # Backfill telemetry
        answer.elapsed_seconds = round(time.time() - started, 2)
        if hasattr(result, "all_messages"):
            try:
                answer.n_tool_calls = sum(
                    1 for m in result.all_messages()
                    if any(
                        getattr(p, "part_kind", "") == "tool-call"
                        for p in getattr(m, "parts", [])
                    )
                )
            except Exception:
                pass

        if trace is not None:
            try:
                trace.update(
                    output={
                        "answer": answer.answer[:500],
                        "refused": answer.refused,
                        "n_citations": len(answer.citations),
                        "n_tool_calls": answer.n_tool_calls,
                    }
                )
            except Exception:
                pass

        return answer
