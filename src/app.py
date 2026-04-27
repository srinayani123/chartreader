"""Streamlit UI for ChartReader — standalone demo mode.

Two-column layout: left is the chart upload + question + run button, right is
the agent's answer with citations, visual claims, and confidence notes.

Run with: python -m streamlit run src/app.py
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.agents.analyst import analyze_chart
from src.retrieval.store import count_articles, init_schema

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="ChartReader", layout="wide")


def _escape_dollars(text: str) -> str:
    """Escape $ so Streamlit's markdown doesn't treat $...$ as LaTeX math.

    Without this, prices like '$110' through '$208' get parsed as math and
    rendered as run-together gibberish.
    """
    if not text:
        return text
    return text.replace("$", r"\$")


# Ensure pgvector schema exists on first run
@st.cache_resource
def _init_db():
    try:
        init_schema()
        return True
    except Exception as exc:
        st.warning(f"pgvector not reachable: {exc}. Semantic news cache disabled.")
        return False


def main() -> None:
    _ = _init_db()

    st.title("ChartReader")
    st.caption(
        "Multimodal chart-grounded financial Q&A. Upload a chart, ask a question, "
        "get a grounded answer with citations. Predictive and advice questions "
        "are explicitly refused."
    )

    with st.sidebar:
        st.subheader("System")
        try:
            n = count_articles()
            st.caption(f"News cache: {n} articles")
        except Exception:
            st.caption("News cache: unavailable")
        st.caption("LLM: Anthropic Claude")
        st.caption("Vector store: pgvector")

    col_in, col_out = st.columns([1, 1.5])

    with col_in:
        st.subheader("Chart + question")
        uploaded = st.file_uploader(
            "Chart image (PNG/JPG)", type=["png", "jpg", "jpeg", "webp"]
        )
        if uploaded is not None:
            # Streamlit 1.50+ replaced use_container_width with the `width` kwarg.
            # "stretch" matches the old default behavior.
            try:
                st.image(uploaded, caption="Chart to analyze", width="stretch")
            except TypeError:
                # Fallback for older Streamlit versions
                st.image(uploaded, caption="Chart to analyze")

        question = st.text_area(
            "Question about the chart",
            placeholder=(
                "e.g., What happened to AAPL in October 2024?\n"
                "      Why did the stock drop in Q3?\n"
                "      How did NVDA compare to the semis sector this period?"
            ),
            height=100,
        )

        run = st.button("Analyze", type="primary",
                        disabled=(uploaded is None or not question.strip()))

    with col_out:
        st.subheader("Answer")
        if not run:
            st.caption("Upload a chart, ask a question, click Analyze.")
            return

        # Persist the upload to a temp file so vision/extract.py can read it
        suffix = Path(uploaded.name).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        with st.spinner("Analyzing chart and gathering evidence..."):
            try:
                answer = asyncio.run(analyze_chart(tmp_path, question.strip()))
            except Exception as exc:
                st.error(f"Run failed: {exc}")
                return

        # Refusal path
        if answer.refused:
            st.warning("Refused.")
            st.markdown(_escape_dollars(answer.refusal_reason or answer.answer))
            return

        # Normal answer
        st.markdown(_escape_dollars(answer.answer))

        if answer.confidence_notes:
            with st.container(border=True):
                st.caption("Confidence notes")
                st.markdown(_escape_dollars(answer.confidence_notes))

        # Telemetry strip
        m1, m2, m3 = st.columns(3)
        m1.metric("Tool calls", answer.n_tool_calls)
        m2.metric("News articles", answer.n_news_retrieved)
        m3.metric("Latency (s)", answer.elapsed_seconds)

        if answer.citations:
            with st.expander(f"Citations ({len(answer.citations)})", expanded=False):
                for c in answer.citations:
                    title = c.title or c.url or c.source_type
                    if c.url:
                        st.markdown(f"- **[{title}]({c.url})** — {c.source_type}")
                    else:
                        st.markdown(f"- **{_escape_dollars(title)}** — {c.source_type}")
                    if c.excerpt:
                        st.caption(_escape_dollars(c.excerpt))

        if answer.visual_claims:
            with st.expander(
                f"Visual claims ({len(answer.visual_claims)})", expanded=False
            ):
                for v in answer.visual_claims:
                    st.markdown(
                        f"- {_escape_dollars(v.claim)}  *({v.grounded_in})*"
                    )

        if answer.retrieved_claims:
            with st.expander(
                f"Retrieved claims ({len(answer.retrieved_claims)})", expanded=False
            ):
                for r in answer.retrieved_claims:
                    src = r.citation.title or r.citation.url or r.citation.source_type
                    st.markdown(
                        f"- {_escape_dollars(r.claim)}  *(source: {_escape_dollars(src)})*"
                    )


if __name__ == "__main__":
    main()
    