"""Tests for eval/metrics.py — all the metric computation functions.

These don't make LLM calls (LLM judge tested separately in eval).
"""
from datetime import date

import pytest

from eval.metrics import (
    citation_faithfulness,
    latency_summary,
    refusal_rate,
    visual_extraction_mae,
    visual_textual_consistency,
)
from src.models.schemas import Citation, GroundedAnswer, VisualClaim


class TestVisualExtractionMAE:
    def test_perfect_match(self):
        extracted = [{"date": "2024-10-15", "value": 233.0}]
        truth = [{"date": "2024-10-15", "close": 233.0}]
        assert visual_extraction_mae(extracted, truth) == 0.0

    def test_known_error(self):
        # 233 vs 250 = 7.297% error
        extracted = [{"date": "2024-10-15", "value": 250.0}]
        truth = [{"date": "2024-10-15", "close": 233.0}]
        mae = visual_extraction_mae(extracted, truth)
        assert 7.0 < mae < 7.5

    def test_no_overlap(self):
        extracted = [{"date": "2024-10-15", "value": 250.0}]
        truth = [{"date": "2024-11-15", "close": 233.0}]
        # No matching dates → empty errors
        assert visual_extraction_mae(extracted, truth) == 0.0

    def test_empty_inputs(self):
        assert visual_extraction_mae([], []) == 0.0
        assert visual_extraction_mae([], [{"date": "2024-10-15", "close": 233.0}]) == 0.0


class TestCitationFaithfulness:
    def test_no_citations(self):
        a = GroundedAnswer(answer="x")
        result = citation_faithfulness(a, timeout_per_url=0.1)
        assert result["n_cited_urls"] == 0
        assert result["rate"] == 1.0

    def test_only_non_url_citations(self):
        a = GroundedAnswer(
            answer="x",
            citations=[
                Citation(source_type="yfinance"),
                Citation(source_type="chart_visual"),
            ],
        )
        result = citation_faithfulness(a, timeout_per_url=0.1)
        # No URLs to check → trivially "all reachable"
        assert result["n_cited_urls"] == 0


class TestVisualTextualConsistency:
    def test_no_truth(self):
        a = GroundedAnswer(answer="The stock dropped 8%.")
        # Empty truth → trivially consistent
        assert visual_textual_consistency(a, {}) == 1.0

    def test_no_numbers_in_answer(self):
        a = GroundedAnswer(answer="The stock declined.")
        truth = {"return_pct": -8.0, "high": 240.0}
        # No numeric claims to violate
        assert visual_textual_consistency(a, truth) == 1.0

    def test_close_match(self):
        a = GroundedAnswer(answer="The stock declined 8%.")
        truth = {"return_pct": -8.0}
        # 8 vs -8 difference is 200% absolute, but our matcher checks +/-8 too
        # Actually 8 (extracted) vs -8 (truth) → not within 10% relative.
        # But 8 vs +8 (if we put +8 in truth) → 0% diff.
        # This test is more about the "any value in truth that's close" logic.
        truth_with_pos = {"return_pct": -8.0, "movement": 8.0}
        score = visual_textual_consistency(a, truth_with_pos)
        assert score == 1.0  # 8 matches "movement" within tolerance

    def test_off_by_significant(self):
        a = GroundedAnswer(answer="The stock declined 50%.")
        truth = {"return_pct": -8.0}
        # 50 vs -8 → not close to anything in truth
        score = visual_textual_consistency(a, truth)
        assert score == 0.0


class TestLatencySummary:
    def test_empty(self):
        s = latency_summary([])
        assert s["n"] == 0
        assert s["p95"] == 0.0

    def test_single_value(self):
        s = latency_summary([5.0])
        assert s["mean"] == 5.0
        assert s["p50"] == 5.0
        assert s["p95"] == 5.0

    def test_multiple_values(self):
        s = latency_summary([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0])
        assert s["n"] == 10
        # p95 of 10 values = the 9th sorted (index 9 - 1 = 8) = 40.0
        # but our impl uses int(0.95*10)-1 = 8 → also 40.0
        assert s["p95"] in (40.0, 50.0)


class TestRefusalRate:
    def test_perfect(self):
        # 4 should refuse, all did. 4 should not refuse, none did.
        result = refusal_rate(
            expected_refusal=[True, True, True, True, False, False, False, False],
            actual_refused=[True, True, True, True, False, False, False, False],
        )
        assert result["true_positive_rate"] == 1.0
        assert result["false_positive_rate"] == 0.0

    def test_missed_refusals(self):
        result = refusal_rate(
            expected_refusal=[True, True, True, True],
            actual_refused=[True, True, False, False],
        )
        # Caught 2 of 4 should-refuse cases
        assert result["true_positive_rate"] == 0.5

    def test_false_positives(self):
        result = refusal_rate(
            expected_refusal=[False, False, False, False],
            actual_refused=[True, False, False, False],
        )
        # Falsely refused 1 of 4 legitimate questions
        assert result["false_positive_rate"] == 0.25

    def test_empty(self):
        result = refusal_rate([], [])
        assert result["n"] == 0
