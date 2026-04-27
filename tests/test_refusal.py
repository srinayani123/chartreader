"""Tests for src/refusal/classifier.py.

Most refusal cases hit the cheap pattern prefilter and don't need an LLM call.
We test those deterministically. The LLM-classifier path is exercised
indirectly in the eval harness.
"""
import pytest

from src.refusal.classifier import check_refusal


class TestPatternRefusalRefuses:
    """Questions that match the auto-refuse patterns should be refused
    without any LLM call."""

    def test_should_i_buy(self):
        r = check_refusal("Should I buy AAPL?")
        assert r.should_refuse is True
        assert r.category == "advice"

    def test_should_i_sell(self):
        r = check_refusal("Should I sell my position?")
        assert r.should_refuse is True

    def test_should_i_hold(self):
        r = check_refusal("Should I hold this stock?")
        assert r.should_refuse is True

    def test_will_it_go_up(self):
        r = check_refusal("Will the stock go up next week?")
        assert r.should_refuse is True
        assert r.category == "prediction"

    def test_where_will_it(self):
        r = check_refusal("Where will the price be in 6 months?")
        assert r.should_refuse is True

    def test_price_target(self):
        r = check_refusal("What's the price target for NVDA?")
        assert r.should_refuse is True

    def test_predict(self):
        r = check_refusal("Predict the next move.")
        assert r.should_refuse is True

    def test_recommendation(self):
        r = check_refusal("What do you recommend?")
        assert r.should_refuse is True

    def test_refusal_reason_populated(self):
        r = check_refusal("Should I buy?")
        assert r.should_refuse is True
        assert r.reason  # non-empty


class TestPatternOkAllows:
    """Questions clearly descriptive should pass without LLM call."""

    def test_what_happened(self):
        r = check_refusal("What happened to AAPL in October?")
        assert r.should_refuse is False
        assert r.category == "ok"

    def test_why_did(self):
        r = check_refusal("Why did the stock drop in Q3?")
        assert r.should_refuse is False

    def test_compare(self):
        r = check_refusal("Compare AAPL to MSFT during 2024.")
        assert r.should_refuse is False

    def test_explain(self):
        r = check_refusal("Explain the volume spike in October.")
        assert r.should_refuse is False


class TestEdgeCases:
    def test_empty_question(self):
        r = check_refusal("")
        assert r.should_refuse is False

    def test_whitespace(self):
        r = check_refusal("   ")
        assert r.should_refuse is False

    def test_case_insensitive_refuse(self):
        r = check_refusal("SHOULD I BUY THIS?")
        assert r.should_refuse is True

    def test_case_insensitive_ok(self):
        r = check_refusal("WHAT HAPPENED HERE?")
        assert r.should_refuse is False
