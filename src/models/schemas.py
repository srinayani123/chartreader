"""Core typed schemas for ChartReader.

Every input, output, and tool argument is a Pydantic model.
This is what lets Pydantic AI enforce structure throughout the agentic pipeline.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ============================================================================
# Inputs
# ============================================================================


class ChartContext(BaseModel):
    """The full context the agent has when analyzing a chart.

    Passed as `deps` into the Pydantic AI agent. The agent's tools read from
    this and may also mutate it (e.g., extract step populates `extracted_data`).
    """

    image_path: str  # absolute path to chart image, OR base64 data URL
    question: str
    # Filled in progressively by the agent's tool calls
    extracted_data: Optional["ChartExtraction"] = None
    ticker: Optional[str] = None
    period_start: Optional[date] = None
    period_end: Optional[date] = None


# ============================================================================
# Vision extraction
# ============================================================================


class ChartType(str, Enum):
    LINE = "line"
    CANDLESTICK = "candlestick"
    BAR = "bar"
    AREA = "area"
    UNKNOWN = "unknown"


class PricePoint(BaseModel):
    """A single data point read from the chart."""

    date_str: str  # as appears on the chart (e.g., "2024-10-15" or "Oct 15")
    value: float
    kind: str = "close"  # "close" | "high" | "low" | "open" | "other"


class ChartExtraction(BaseModel):
    """What the vision LLM extracted from a chart image.

    These are CLAIMED values — they have not yet been verified against
    yfinance ground truth. The verification step compares these to the truth
    and produces a PriceVerification.
    """

    ticker: Optional[str] = None  # null if not visible on chart
    period_start_str: Optional[str] = None  # raw text from chart, e.g., "Sep 2024"
    period_end_str: Optional[str] = None
    chart_type: ChartType = ChartType.UNKNOWN
    extracted_points: list[PricePoint] = Field(default_factory=list)
    notable_features: list[str] = Field(
        default_factory=list,
        description=(
            "Descriptive observations about the chart — patterns, volume "
            "spikes, key levels. NOT predictions."
        ),
    )
    extraction_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="LLM's self-reported confidence in the extraction.",
    )


class PriceVerification(BaseModel):
    """Result of comparing extracted prices to yfinance ground truth."""

    ticker: str
    period_start: date
    period_end: date
    n_points_verified: int = 0
    n_points_failed: int = 0
    mean_absolute_error_pct: float = 0.0  # MAE as % of actual price
    discrepancies: list[str] = Field(
        default_factory=list,
        description="Human-readable notes on each significant discrepancy.",
    )
    is_reliable: bool = True  # False if MAE > threshold or extraction was sparse


# ============================================================================
# Retrieval
# ============================================================================


class NewsArticle(BaseModel):
    """A news article retrieved during the agent's investigation."""

    url: str
    title: str
    published_at: Optional[datetime] = None
    source: Optional[str] = None  # e.g., "Reuters", "Bloomberg"
    snippet: str  # short excerpt from the article
    full_content: Optional[str] = None  # populated if we fetched the full text


class PeerData(BaseModel):
    """Sector / peer comparison data from yfinance."""

    base_ticker: str
    peer_tickers: list[str] = Field(default_factory=list)
    period_start: date
    period_end: date
    base_return_pct: float = 0.0
    peer_returns_pct: dict[str, float] = Field(default_factory=dict)
    sector_etf: Optional[str] = None  # e.g., "XLK" for tech
    sector_return_pct: Optional[float] = None
    notes: list[str] = Field(default_factory=list)


# ============================================================================
# Refusal
# ============================================================================


class RefusalCheck(BaseModel):
    """Whether a question crosses into prediction/advice and should be refused."""

    should_refuse: bool
    category: str = "ok"  # "ok" | "prediction" | "advice" | "unclear"
    reason: str = ""  # short rationale, shown to user only on refusal


# ============================================================================
# Final answer
# ============================================================================


class Citation(BaseModel):
    """A source attribution for a claim in the answer."""

    source_type: str  # "news" | "yfinance" | "chart_visual" | "peer_data"
    url: Optional[str] = None
    title: Optional[str] = None
    published_at: Optional[datetime] = None
    excerpt: Optional[str] = None  # short relevant excerpt


class VisualClaim(BaseModel):
    """A claim the answer makes about what the chart shows."""

    claim: str  # e.g., "AAPL declined ~8% in October 2024"
    grounded_in: str = "chart"  # "chart" | "chart+yfinance"
    numeric_values: list[float] = Field(default_factory=list)


class RetrievedClaim(BaseModel):
    """A claim from external retrieved evidence (news, peer data, etc.)."""

    claim: str
    citation: Citation


class GroundedAnswer(BaseModel):
    """The agent's final output — grounded, cited, with explicit refusal handling.

    This is the type Pydantic AI returns from the agent.
    """

    answer: str
    refused: bool = False
    refusal_reason: Optional[str] = None

    citations: list[Citation] = Field(default_factory=list)
    visual_claims: list[VisualClaim] = Field(default_factory=list)
    retrieved_claims: list[RetrievedClaim] = Field(default_factory=list)

    # Honest disclosure
    confidence_notes: Optional[str] = Field(
        default=None,
        description=(
            "Plain-language note about anything the agent couldn't verify, "
            "uncertainty in the chart extraction, or claims it's flagging "
            "as less reliable."
        ),
    )

    # Telemetry — useful for eval and observability
    n_tool_calls: int = 0
    n_news_retrieved: int = 0
    elapsed_seconds: float = 0.0
