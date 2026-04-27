"""Refusal classifier — detects prediction/advice questions and refuses them.

ChartReader explicitly refuses:
  - Price predictions ("Will it go up?", "Where is it heading?")
  - Buy/sell advice ("Should I buy?", "Is it a good investment?")
  - Forward-looking targets ("What's the price target?", "Where will it be in 6 months?")

Uses a cheap Haiku call. Combines with a fast keyword-based prefilter to
avoid LLM costs on obviously-fine questions.

This is a metric we eval against: refusal rate on adversarial advice queries.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

import anthropic

from src.models.schemas import RefusalCheck

logger = logging.getLogger(__name__)


# Cheap pattern prefilter — clear advice/prediction phrasings get auto-refused
# without the Haiku call.
_AUTO_REFUSE_PATTERNS = [
    r"\bshould i (buy|sell|hold|invest|short)\b",
    r"\bwill (it|this|the (stock|price)) (go|rise|fall|drop|rally|crash)",
    r"\bwhere will (it|this|the (stock|price))\b",
    r"\bprice target\b",
    r"\bis (it|this|now) a good (buy|investment|time)\b",
    r"\bwhat (do you|would you) recommend\b",
    r"\bgive me (your|a) recommendation\b",
    r"\bpredict (the|this)\b",
]
_AUTO_REFUSE_RE = re.compile("|".join(_AUTO_REFUSE_PATTERNS), re.IGNORECASE)


# Cheap pattern allowlist — clear descriptive questions skip the Haiku call too
_AUTO_OK_PATTERNS = [
    r"\bwhat happened\b",
    r"\bwhy did\b",
    r"\bcompare\b",
    r"\bhow does .* compare\b",
    r"\bexplain\b",
    r"\bdescribe\b",
    r"\bsummari[zs]e\b",
]
_AUTO_OK_RE = re.compile("|".join(_AUTO_OK_PATTERNS), re.IGNORECASE)


_CLASSIFIER_PROMPT = """You classify a user's question about a financial chart \
into one of these categories. Output ONLY a JSON object.

Categories:
  "ok"          — descriptive, factual, or comparative. Asking what happened, \
why something happened in the past, comparing to peers, etc.
  "prediction"  — asks for forward-looking price movement: will it go up, \
where is it heading, forecast, etc.
  "advice"      — asks for a buy/sell/hold recommendation or investment advice.
  "unclear"     — ambiguous, could be either descriptive or forward-looking.

Output:
{ "category": "ok | prediction | advice | unclear",
  "should_refuse": true|false,
  "reason": "<one short sentence — only when refusing>" }

Rules:
- "should_refuse" is true ONLY when category is "prediction" or "advice".
- For "unclear", set should_refuse to false but keep reason short.
- The reason is shown to the user — keep it specific to their question."""


def _llm_check(question: str, *, model: str, api_key: str) -> RefusalCheck:
    """Run the Haiku classifier on a question."""
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=200,
        temperature=0.0,
        system=_CLASSIFIER_PROMPT,
        messages=[{"role": "user", "content": f"Question: {question}"}],
    )
    raw = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
    if raw.startswith("```"):
        nl = raw.find("\n")
        if nl != -1:
            raw = raw[nl + 1:]
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        s, e = raw.find("{"), raw.rfind("}")
        if s != -1 and e != -1:
            try:
                data = json.loads(raw[s:e + 1])
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    category = str(data.get("category", "unclear")).strip().lower()
    if category not in ("ok", "prediction", "advice", "unclear"):
        category = "unclear"
    should_refuse = bool(data.get("should_refuse", False))
    if category in ("prediction", "advice"):
        should_refuse = True
    return RefusalCheck(
        should_refuse=should_refuse,
        category=category,
        reason=str(data.get("reason", "")).strip(),
    )


def check_refusal(question: str) -> RefusalCheck:
    """Classify whether a question should be refused.

    Pipeline:
      1. Cheap pattern allowlist → "ok" without LLM call
      2. Cheap pattern refuse-list → refused without LLM call
      3. Otherwise: LLM classifier (Haiku)
    """
    q = question.strip()
    if not q:
        return RefusalCheck(should_refuse=False, category="ok", reason="")

    if _AUTO_REFUSE_RE.search(q):
        return RefusalCheck(
            should_refuse=True,
            category="advice" if re.search(r"\b(buy|sell|hold|invest|recommend)\b", q, re.I)
                     else "prediction",
            reason=(
                "I don't make price predictions or give buy/sell advice — "
                "chart-based prediction has weak empirical support, and this "
                "tool is designed to explain what charts show, not to forecast "
                "what they'll do next."
            ),
        )

    if _AUTO_OK_RE.search(q):
        return RefusalCheck(should_refuse=False, category="ok", reason="")

    # Defer to LLM for unclear cases
    model = os.getenv("MODEL_REFUSAL", "claude-haiku-4-5-20251001")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        # Fail open — assume OK if we can't run the classifier.
        # This is a reasonable choice: the agent's prompts also instruct it
        # to refuse advice, so this is defense-in-depth, not the only line.
        logger.warning("ANTHROPIC_API_KEY missing; refusal classifier skipped")
        return RefusalCheck(should_refuse=False, category="unclear",
                           reason="Classifier unavailable — proceeding cautiously")

    try:
        return _llm_check(q, model=model, api_key=api_key)
    except Exception as exc:
        logger.warning("Refusal classifier failed: %s", exc)
        return RefusalCheck(should_refuse=False, category="unclear",
                           reason="Classifier error — proceeding cautiously")
