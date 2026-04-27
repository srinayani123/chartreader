"""Tests for src/data/* — yfinance and peer-comparison logic.

We mock yfinance.Ticker so tests don't hit the network.
"""
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.peers import compute_peer_comparison
from src.data.prices import (
    compute_period_return,
    fetch_close_on_date,
    fetch_price_history,
)


def _mock_history_df(closes: list[tuple[str, float]]) -> pd.DataFrame:
    """Build a mock yfinance-shaped DataFrame from (date_str, close) tuples."""
    idx = pd.DatetimeIndex([d for d, _ in closes])
    return pd.DataFrame({
        "Open": [c for _, c in closes],
        "High": [c * 1.01 for _, c in closes],
        "Low": [c * 0.99 for _, c in closes],
        "Close": [c for _, c in closes],
        "Volume": [1_000_000] * len(closes),
    }, index=idx)


class TestFetchPriceHistory:
    @patch("src.data.prices.yf.Ticker")
    def test_basic_fetch(self, mock_ticker_class):
        df = _mock_history_df([("2024-09-30", 233.0), ("2024-10-01", 235.0)])
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        mock_ticker_class.return_value = mock_ticker

        result = fetch_price_history("AAPL", date(2024, 9, 30), date(2024, 10, 1))
        assert len(result) == 2
        assert result[0]["close"] == 233.0
        assert result[1]["close"] == 235.0

    @patch("src.data.prices.yf.Ticker")
    def test_empty_df_returns_empty(self, mock_ticker_class):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker

        result = fetch_price_history("FAKE", date(2024, 1, 1), date(2024, 1, 2))
        assert result == []

    @patch("src.data.prices.yf.Ticker")
    def test_exception_returns_empty(self, mock_ticker_class):
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = Exception("Network error")
        mock_ticker_class.return_value = mock_ticker

        result = fetch_price_history("AAPL", date(2024, 1, 1), date(2024, 1, 2))
        assert result == []


class TestComputePeriodReturn:
    @patch("src.data.prices.yf.Ticker")
    def test_positive_return(self, mock_ticker_class):
        df = _mock_history_df([
            ("2024-09-01", 100.0),
            ("2024-09-15", 105.0),
            ("2024-09-30", 110.0),
        ])
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        mock_ticker_class.return_value = mock_ticker

        r = compute_period_return("AAPL", date(2024, 9, 1), date(2024, 9, 30))
        assert r == 10.0  # (110 - 100) / 100 * 100

    @patch("src.data.prices.yf.Ticker")
    def test_negative_return(self, mock_ticker_class):
        df = _mock_history_df([("2024-09-01", 100.0), ("2024-09-30", 92.0)])
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        mock_ticker_class.return_value = mock_ticker

        r = compute_period_return("AAPL", date(2024, 9, 1), date(2024, 9, 30))
        assert r == -8.0

    @patch("src.data.prices.yf.Ticker")
    def test_insufficient_data(self, mock_ticker_class):
        df = _mock_history_df([("2024-09-01", 100.0)])  # only one point
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        mock_ticker_class.return_value = mock_ticker

        r = compute_period_return("AAPL", date(2024, 9, 1), date(2024, 9, 30))
        assert r is None


class TestPeerComparison:
    @patch("src.data.peers.compute_period_return")
    def test_basic_peer_comparison(self, mock_return):
        # AAPL up 10, MSFT up 5, GOOGL up 3
        def side_effect(ticker, *_):
            return {"AAPL": 10.0, "MSFT": 5.0, "GOOGL": 3.0}.get(ticker)
        mock_return.side_effect = side_effect

        result = compute_peer_comparison(
            "AAPL", ["MSFT", "GOOGL"],
            date(2024, 9, 1), date(2024, 12, 31),
        )
        assert result.base_return_pct == 10.0
        assert result.peer_returns_pct["MSFT"] == 5.0
        assert result.peer_returns_pct["GOOGL"] == 3.0
        assert any("outperformed 2/2 peers" in n for n in result.notes)

    @patch("src.data.peers.compute_period_return")
    def test_with_sector_etf(self, mock_return):
        def side_effect(ticker, *_):
            return {"AAPL": 10.0, "XLK": 7.0}.get(ticker)
        mock_return.side_effect = side_effect

        result = compute_peer_comparison(
            "AAPL", [], date(2024, 9, 1), date(2024, 12, 31),
            sector_etf="XLK",
        )
        assert result.sector_return_pct == 7.0
        assert any("XLK" in n for n in result.notes)

    @patch("src.data.peers.compute_period_return")
    def test_missing_base_return(self, mock_return):
        mock_return.return_value = None  # all return None

        result = compute_peer_comparison(
            "FAKE", ["MSFT"],
            date(2024, 9, 1), date(2024, 12, 31),
        )
        assert result.base_return_pct == 0.0
        assert any("Could not fetch" in n for n in result.notes)
