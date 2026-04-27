"""Tests for src/vision/verify.py — date parsing and verification math.

All tests run without external calls (we mock yfinance).
"""
from datetime import date
from unittest.mock import patch

import pytest

from src.models.schemas import ChartExtraction, ChartType, PricePoint
from src.vision.verify import _parse_date_string, verify_extraction


class TestDateStringParsing:
    def test_iso_format(self):
        assert _parse_date_string("2024-10-15") == date(2024, 10, 15)

    def test_us_format(self):
        assert _parse_date_string("10/15/2024") == date(2024, 10, 15)

    def test_short_month_with_day(self):
        # No year — uses period_start as hint
        assert _parse_date_string(
            "Oct 15", period_start=date(2024, 9, 1)
        ) == date(2024, 10, 15)

    def test_short_month_with_year(self):
        assert _parse_date_string("Oct 15, 2024") == date(2024, 10, 15)

    def test_full_month(self):
        assert _parse_date_string("October 15, 2024") == date(2024, 10, 15)

    def test_year_only_month(self):
        assert _parse_date_string("Oct 2024") == date(2024, 10, 1)

    def test_unparseable_returns_none(self):
        assert _parse_date_string("garbage") is None
        assert _parse_date_string("") is None


class TestVerifyExtraction:
    def test_no_points_safe(self):
        e = ChartExtraction(extracted_points=[])
        v = verify_extraction(e, "AAPL", date(2024, 9, 1), date(2024, 12, 31))
        assert v.n_points_verified == 0
        # no points → no MAE → defaults; the `is_reliable` check requires
        # at least one verified point, so this is unreliable
        assert v.is_reliable is False

    @patch("src.vision.verify.fetch_close_on_date")
    def test_perfectly_accurate_extraction(self, mock_fetch):
        # Mock yfinance to return exactly what was extracted
        mock_fetch.return_value = 233.0

        e = ChartExtraction(
            extracted_points=[
                PricePoint(date_str="2024-10-15", value=233.0, kind="close"),
                PricePoint(date_str="2024-10-16", value=233.0, kind="close"),
            ],
        )
        v = verify_extraction(e, "AAPL", date(2024, 9, 1), date(2024, 12, 31))
        assert v.n_points_verified == 2
        assert v.mean_absolute_error_pct == 0.0
        assert v.is_reliable is True

    @patch("src.vision.verify.fetch_close_on_date")
    def test_off_by_significant_amount(self, mock_fetch):
        # yfinance says 233, vision said 250 → 7.3% off
        mock_fetch.return_value = 233.0

        e = ChartExtraction(
            extracted_points=[
                PricePoint(date_str="2024-10-15", value=250.0),
            ],
        )
        v = verify_extraction(
            e, "AAPL", date(2024, 9, 1), date(2024, 12, 31),
            reliability_threshold_pct=5.0,
        )
        assert v.n_points_verified == 1
        assert v.mean_absolute_error_pct > 5.0
        assert v.is_reliable is False
        assert any("off by" in d.lower() for d in v.discrepancies)

    @patch("src.vision.verify.fetch_close_on_date")
    def test_yfinance_unavailable(self, mock_fetch):
        # yfinance returns None → graceful failure
        mock_fetch.return_value = None

        e = ChartExtraction(
            extracted_points=[
                PricePoint(date_str="2024-10-15", value=233.0),
            ],
        )
        v = verify_extraction(e, "AAPL", date(2024, 9, 1), date(2024, 12, 31))
        assert v.n_points_failed == 1
        assert v.n_points_verified == 0
        assert v.is_reliable is False

    @patch("src.vision.verify.fetch_close_on_date")
    def test_outside_period_skipped(self, mock_fetch):
        mock_fetch.return_value = 233.0
        e = ChartExtraction(
            extracted_points=[
                PricePoint(date_str="2023-01-15", value=233.0),  # outside period
            ],
        )
        v = verify_extraction(e, "AAPL", date(2024, 9, 1), date(2024, 12, 31))
        # Skipped silently — no failure, no verification
        assert v.n_points_verified == 0
        assert v.n_points_failed == 0

    @patch("src.vision.verify.fetch_close_on_date")
    def test_unparseable_date_failure(self, mock_fetch):
        e = ChartExtraction(
            extracted_points=[
                PricePoint(date_str="garbage", value=233.0),
            ],
        )
        v = verify_extraction(e, "AAPL", date(2024, 9, 1), date(2024, 12, 31))
        assert v.n_points_failed == 1
        assert v.n_points_verified == 0
