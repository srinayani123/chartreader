"""Verification — cross-checks vision-extracted values against yfinance ground truth.

The vision LLM might misread an axis or hallucinate a value. This module
deterministically compares each extracted price point to the actual close
on the corresponding date and produces a PriceVerification report.

If MAE exceeds a threshold, the agent is told the extraction is unreliable
and should weight it less when reasoning. Note: a failed verification is
NOT a sign the chart is fake — see rule 11 in the agent prompt for how the
agent should interpret partial/failed verifications.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import date, datetime, timedelta
from typing import Optional

from src.data.prices import fetch_close_on_date
from src.models.schemas import ChartExtraction, PriceVerification

logger = logging.getLogger(__name__)


# MAE > this fraction (default 5%) marks the extraction unreliable
DEFAULT_RELIABILITY_THRESHOLD_PCT = float(
    os.getenv("EXTRACTION_RELIABILITY_THRESHOLD_PCT", "5.0")
)


_DATE_PATTERNS = [
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%m/%d/%y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%b %Y",
    "%B %Y",
    "%Y-%m",
]


# Matches a trailing intraday time portion like " 9:00 AM", " 14:30",
# " 9:30:00 PM", etc. We strip these before date parsing because for
# end-of-day yfinance lookups we only need the date — the intraday time
# is irrelevant to the close-price comparison. Without this stripping,
# vision-extracted strings like "April 22 9:00 AM" (common on 1D and 5D
# Yahoo Finance charts) fail to parse and verification skips the point.
_TIME_SUFFIX_PATTERN = re.compile(
    r"\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\s*$"
)


def _strip_time_suffix(s: str) -> str:
    """Remove a trailing time portion (e.g. ' 9:00 AM', ' 14:30:00') from a
    date string, leaving just the date portion behind. Idempotent — calling
    on an already date-only string returns it unchanged.
    """
    return _TIME_SUFFIX_PATTERN.sub("", s).strip()


def _most_recent_past_occurrence(month: int, day: int, today: date) -> date:
    """Pick the year for a (month, day) such that the resulting date is the
    most recent past or current occurrence relative to `today`.

    Examples (assuming today is 2026-04-27):
      (4, 22) -> 2026-04-22  (this year, already passed)
      (4, 27) -> 2026-04-27  (today)
      (4, 28) -> 2025-04-28  (this year hasn't occurred yet, use last year)
      (12, 25) -> 2025-12-25 (last Christmas)
    """
    try:
        candidate = date(today.year, month, day)
    except ValueError:
        candidate = date(today.year - 1, month, day)
        return candidate
    if candidate > today:
        try:
            candidate = date(today.year - 1, month, day)
        except ValueError:
            pass
    return candidate


def _parse_date_string(s: str, period_start: Optional[date] = None) -> Optional[date]:
    """Parse a chart-axis date string into a date.

    Handles common chart date formats including:
      - "Oct 15", "Oct 2024", "October 15, 2024"
      - "2024-10-15", "10/15/24", "10/15/2024"
      - "4/22" (no year)
      - Any of the above with an intraday time suffix like
        " 9:00 AM" or " 14:30" — the time is stripped before parsing.

    Year inference order when no year is in the string:
      1. If period_start is given, use period_start.year.
      2. Otherwise, use the most recent past occurrence relative to today.
    """
    if not s:
        return None
    # Strip any intraday time suffix first — yfinance verification only
    # needs the date, and most chart axis labels include "9:00 AM" style
    # times on intraday timeframes (1D, 5D).
    s = _strip_time_suffix(s.strip())
    if not s:
        return None

    for fmt in _DATE_PATTERNS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue

    m = re.match(r"([A-Za-z]+)\s+(\d{1,2})(?:[, ]+(\d{2,4}))?", s)
    if m:
        month_str, day_str, year_str = m.group(1), m.group(2), m.group(3)
        try:
            month = datetime.strptime(month_str[:3], "%b").month
            day = int(day_str)
            if year_str:
                year = int(year_str)
                if year < 100:
                    year += 2000
                return date(year, month, day)
            elif period_start:
                return date(period_start.year, month, day)
            else:
                return _most_recent_past_occurrence(month, day, date.today())
        except (ValueError, AttributeError):
            pass

    m2 = re.match(r"^(\d{1,2})/(\d{1,2})$", s)
    if m2:
        try:
            month = int(m2.group(1))
            day = int(m2.group(2))
            if period_start:
                return date(period_start.year, month, day)
            return _most_recent_past_occurrence(month, day, date.today())
        except ValueError:
            pass

    return None


def _explain_missing_data(parsed_date: date, today: date) -> str:
    """Build a human-friendly explanation of why yfinance might have no
    data for a particular date. The agent uses this to pick the most
    likely reason in its final verification note (rule 11).

    The reasons are listed in rough order of likelihood — most charts that
    fail verification fail because (a) it's today during market hours, or
    (b) the date is on a weekend.
    """
    candidates = []
    if parsed_date == today:
        candidates.append(
            "today's market session may still be open — yfinance only "
            "publishes end-of-day closes after 4 PM US Eastern"
        )
    if parsed_date > today:
        days_ahead = (parsed_date - today).days
        if days_ahead <= 2:
            candidates.append(
                "this date is just past today — yfinance data hasn't "
                "been published yet"
            )
        else:
            candidates.append(
                "this date is in the future — the chart's x-axis may "
                "include padding that extends past today's data"
            )
    if parsed_date.weekday() >= 5:  # 5=Sat, 6=Sun
        candidates.append(
            "this date falls on a weekend — markets were closed and "
            "yfinance has no close"
        )
    if not candidates:
        candidates.append(
            "possible reasons include a market holiday, the ticker "
            "being delisted or absent from yfinance, or a date older "
            "than this ticker's available history"
        )
    return "; ".join(candidates)


def verify_extraction(
    extraction: ChartExtraction,
    ticker: str,
    period_start: date,
    period_end: date,
    *,
    reliability_threshold_pct: float = DEFAULT_RELIABILITY_THRESHOLD_PCT,
) -> PriceVerification:
    """Compare each extracted price point to yfinance ground truth.

    Verification is a BONUS check, not a gate. The agent must still describe
    the chart even when verification fails for some/all dates. See rule 11
    in the agent prompt.

    If period_end is well into the future, it's clipped to today. yfinance
    has no data past today and querying produces noisy "possibly delisted"
    warnings. The agent reports unverified dates honestly via the discrepancy
    messages this function emits.
    """
    today = date.today()
    if period_end > today:
        logger.info(
            "Clipping verification window from %s to %s (today) — "
            "yfinance has no future data",
            period_end, today,
        )
        period_end = today

    discrepancies: list[str] = []
    abs_pct_errors: list[float] = []
    n_verified = 0
    n_failed = 0

    for point in extraction.extracted_points:
        parsed_date = _parse_date_string(point.date_str, period_start)
        if parsed_date is None:
            n_failed += 1
            discrepancies.append(
                f"Could not parse date {point.date_str!r} for verification "
                f"(the chart's date format may not match expected patterns)."
            )
            continue
        if not (period_start <= parsed_date <= period_end):
            # Outside the verification window — skip silently
            continue

        actual = fetch_close_on_date(ticker, parsed_date)
        if actual is None:
            n_failed += 1
            reason = _explain_missing_data(parsed_date, today)
            discrepancies.append(
                f"yfinance returned no close for {ticker} on "
                f"{parsed_date.isoformat()} ({reason}). The chart-extracted "
                f"value of ${point.value:.2f} for this date could not be "
                f"independently verified, but the chart itself remains "
                f"primary evidence."
            )
            continue

        if actual == 0:
            continue
        diff_pct = abs(point.value - actual) / abs(actual) * 100.0
        abs_pct_errors.append(diff_pct)
        n_verified += 1

        if diff_pct > reliability_threshold_pct:
            discrepancies.append(
                f"On {parsed_date}: extracted ${point.value:.2f} vs actual "
                f"${actual:.2f} (off by {diff_pct:.1f}%)."
            )

    mae = round(sum(abs_pct_errors) / len(abs_pct_errors), 2) if abs_pct_errors else 0.0
    is_reliable = (
        mae <= reliability_threshold_pct
        and n_verified > 0
        and (n_verified / max(1, n_verified + n_failed)) >= 0.5
    )

    return PriceVerification(
        ticker=ticker,
        period_start=period_start,
        period_end=period_end,
        n_points_verified=n_verified,
        n_points_failed=n_failed,
        mean_absolute_error_pct=mae,
        discrepancies=discrepancies,
        is_reliable=is_reliable,
    )
