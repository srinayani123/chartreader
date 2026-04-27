"""Tests for src/models/schemas.py — structural validation, no LLM calls."""
from datetime import date

import pytest
from pydantic import ValidationError

from src.models.schemas import (
    ChartContext,
    ChartExtraction,
    ChartType,
    Citation,
    GroundedAnswer,
    NewsArticle,
    PeerData,
    PricePoint,
    PriceVerification,
    RefusalCheck,
    RetrievedClaim,
    VisualClaim,
)


class TestChartContext:
    def test_minimal(self):
        c = ChartContext(image_path="/tmp/x.png", question="What happened?")
        assert c.image_path == "/tmp/x.png"
        assert c.question == "What happened?"
        assert c.extracted_data is None
        assert c.ticker is None

    def test_question_required(self):
        with pytest.raises(ValidationError):
            ChartContext(image_path="/tmp/x.png")


class TestChartExtraction:
    def test_default_values(self):
        e = ChartExtraction()
        assert e.ticker is None
        assert e.chart_type == ChartType.UNKNOWN
        assert e.extracted_points == []
        assert 0.0 <= e.extraction_confidence <= 1.0

    def test_confidence_clamped(self):
        # Pydantic enforces ge=0, le=1
        with pytest.raises(ValidationError):
            ChartExtraction(extraction_confidence=1.5)
        with pytest.raises(ValidationError):
            ChartExtraction(extraction_confidence=-0.1)

    def test_with_points(self):
        e = ChartExtraction(
            ticker="AAPL",
            chart_type=ChartType.LINE,
            extracted_points=[
                PricePoint(date_str="2024-10-15", value=232.5, kind="close"),
            ],
            notable_features=["Higher high in October"],
            extraction_confidence=0.8,
        )
        assert e.ticker == "AAPL"
        assert len(e.extracted_points) == 1
        assert e.extracted_points[0].value == 232.5


class TestPriceVerification:
    def test_reliable_default(self):
        v = PriceVerification(
            ticker="AAPL", period_start=date(2024, 9, 1),
            period_end=date(2024, 12, 31),
        )
        assert v.is_reliable is True

    def test_with_errors(self):
        v = PriceVerification(
            ticker="AAPL", period_start=date(2024, 9, 1),
            period_end=date(2024, 12, 31),
            n_points_failed=3,
            mean_absolute_error_pct=8.5,
            discrepancies=["Off by 8.5%"],
            is_reliable=False,
        )
        assert v.is_reliable is False
        assert v.mean_absolute_error_pct == 8.5


class TestRefusalCheck:
    def test_should_refuse(self):
        r = RefusalCheck(should_refuse=True, category="prediction",
                       reason="forward-looking")
        assert r.should_refuse is True
        assert r.category == "prediction"

    def test_default_ok(self):
        r = RefusalCheck(should_refuse=False)
        assert r.category == "ok"


class TestGroundedAnswer:
    def test_minimal_answer(self):
        a = GroundedAnswer(answer="The chart shows X.")
        assert a.refused is False
        assert a.citations == []

    def test_refused_answer(self):
        a = GroundedAnswer(
            answer="I don't make predictions.",
            refused=True,
            refusal_reason="Prediction question.",
        )
        assert a.refused is True
        assert a.refusal_reason == "Prediction question."

    def test_with_full_evidence(self):
        a = GroundedAnswer(
            answer="AAPL declined ~8% in October.",
            citations=[
                Citation(source_type="news", url="https://example.com/article",
                       title="iPhone sales miss"),
                Citation(source_type="yfinance"),
            ],
            visual_claims=[
                VisualClaim(claim="Price dropped from 233 to 215",
                           grounded_in="chart+yfinance",
                           numeric_values=[233.0, 215.0]),
            ],
            retrieved_claims=[
                RetrievedClaim(
                    claim="iPhone production cut",
                    citation=Citation(
                        source_type="news",
                        url="https://example.com/article",
                    ),
                ),
            ],
            confidence_notes="One news source unverified.",
            n_tool_calls=4,
            elapsed_seconds=12.3,
        )
        assert len(a.citations) == 2
        assert len(a.visual_claims) == 1
        assert a.n_tool_calls == 4


class TestNewsArticle:
    def test_minimal(self):
        n = NewsArticle(url="https://x.com", title="t", snippet="s")
        assert n.published_at is None
        assert n.full_content is None


class TestPeerData:
    def test_basic(self):
        p = PeerData(
            base_ticker="AAPL",
            peer_tickers=["MSFT", "GOOGL"],
            period_start=date(2024, 9, 1),
            period_end=date(2024, 12, 31),
            base_return_pct=7.5,
            peer_returns_pct={"MSFT": 5.2, "GOOGL": 3.1},
        )
        assert p.base_return_pct == 7.5
        assert p.peer_returns_pct["MSFT"] == 5.2
        assert p.sector_etf is None
