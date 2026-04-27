"""Dev helper: fetch real yfinance data, render a chart, save chart + ground truth JSON.

For TESTING/EVAL CONVENIENCE only. ChartReader's core demo is reading charts
the user uploads. This script gives you something to throw at the pipeline AND
auto-populates eval/test_set/ground_truth.json so you don't hand-copy numbers.

Usage:
    python scripts/fetch_yahoo_chart.py AAPL 6mo
    python scripts/fetch_yahoo_chart.py --batch AAPL:6mo NVDA:1y MSFT:1y

Periods: 1mo, 3mo, 6mo, 1y, 2y, 5y, ytd, max
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf

CHARTS_DIR = Path("eval/test_set/charts")
GROUND_TRUTH_PATH = Path("eval/test_set/ground_truth.json")


def fetch_and_render(ticker: str, period: str = "6mo", verbose: bool = True):
    """Fetch yfinance, render chart PNG, return (chart_id, ground_truth_dict)."""
    ticker = ticker.upper()

    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if df is None or df.empty:
        raise SystemExit(
            f"No data returned for {ticker} (period={period}). "
            f"Check the ticker symbol or try a different period."
        )

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    ax.plot(df.index, df["Close"], linewidth=1.8, color="#1f77b4")
    ax.fill_between(df.index, df["Close"], df["Close"].min(),
                    alpha=0.08, color="#1f77b4")

    # Tightly bound x-axis to actual data — prevents phantom future month labels
    ax.set_xlim(df.index[0], df.index[-1])

    period_label = {"1mo": "1 Month", "3mo": "3 Months", "6mo": "6 Months",
                    "1y": "1 Year", "2y": "2 Years", "5y": "5 Years",
                    "ytd": "YTD", "max": "Max"}.get(period, period)
    ax.set_title(f"{ticker} — Close Price ({period_label})", fontsize=14, pad=15)
    ax.set_ylabel("Price (USD)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)

    if len(df) > 90:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    else:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    chart_id = f"{ticker}_{period}"
    out_path = CHARTS_DIR / f"{chart_id}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    # Compute summary stats
    start_close = float(df["Close"].iloc[0])
    end_close = float(df["Close"].iloc[-1])
    high = float(df["Close"].max())
    low = float(df["Close"].min())
    pct = (end_close - start_close) / start_close * 100

    # Sample ~10 evenly-spaced points for price_history
    n_samples = min(10, len(df))
    step = max(1, len(df) // n_samples)
    sampled = df.iloc[::step].head(n_samples)
    price_history = [
        {"date": idx.date().isoformat(), "close": round(float(row["Close"]), 2)}
        for idx, row in sampled.iterrows()
    ]
    # Ensure final date is included
    last_date = df.index[-1].date().isoformat()
    if not price_history or price_history[-1]["date"] != last_date:
        price_history.append({
            "date": last_date,
            "close": round(float(df["Close"].iloc[-1]), 2),
        })

    truth = {
        "ticker": ticker,
        "period_start": df.index[0].date().isoformat(),
        "period_end": last_date,
        "summary": {
            "start_close": round(start_close, 2),
            "end_close": round(end_close, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "return_pct": round(pct, 2),
        },
        "price_history": price_history,
        "context_notes": [],
    }

    if verbose:
        print(f"\nSaved: {out_path}")
        print(f"  Period: {truth['period_start']} to {truth['period_end']}")
        print(f"  ${start_close:.2f} -> ${end_close:.2f} ({pct:+.2f}%)")
        print(f"  Range: ${low:.2f} - ${high:.2f}")
        print(f"  Sampled {len(price_history)} points for ground truth")

    return chart_id, truth


def update_ground_truth(chart_id: str, truth: dict) -> None:
    """Merge a chart's ground truth into ground_truth.json (idempotent)."""
    if GROUND_TRUTH_PATH.exists():
        try:
            current = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = {}
    else:
        current = {}

    # Drop the example placeholder if present
    current.pop("_example", None)
    current[chart_id] = truth

    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH_PATH.write_text(
        json.dumps(current, indent=2), encoding="utf-8"
    )


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage:")
        print("  python scripts/fetch_yahoo_chart.py AAPL 6mo")
        print("  python scripts/fetch_yahoo_chart.py --batch AAPL:6mo NVDA:1y")
        sys.exit(1)

    if args[0] == "--batch":
        pairs = args[1:]
        successes = 0
        for pair in pairs:
            if ":" not in pair:
                print(f"Skipping malformed pair: {pair} (use TICKER:PERIOD)")
                continue
            t, p = pair.split(":", 1)
            try:
                chart_id, truth = fetch_and_render(t, p, verbose=True)
                update_ground_truth(chart_id, truth)
                successes += 1
            except SystemExit as e:
                print(f"  ERROR for {pair}: {e}")
                continue
        print(f"\n{'=' * 50}")
        print(f"Updated {GROUND_TRUTH_PATH}")
        print(f"Successfully processed: {successes}/{len(pairs)}")
    else:
        ticker = args[0]
        period = args[1] if len(args) > 1 else "6mo"
        chart_id, truth = fetch_and_render(ticker, period, verbose=True)
        update_ground_truth(chart_id, truth)
        print(f"\nUpdated {GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    main()
    