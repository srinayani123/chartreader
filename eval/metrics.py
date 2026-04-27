"""Evaluation metrics for ChartReader.

Eight categories:
  1. Visual extraction MAE — how accurate is the chart→data step?
  2. Citation faithfulness — % cited URLs that exist + match claims
  3. Visual-textual consistency — % numerical claims match chart data
  4. Latency p95 + cost/query — production-mindedness
  5. Beats single-LLM baseline on faithfulness — agentic value
  6. Refusal rate on adversarial advice queries — safety property
  7. Judge stability — measures judge reliability via repeated calls
  8. Confidence interval — 95% CI on aggregate metrics for honest reporting

The judge is Claude Sonnet (configurable via MODEL_JUDGE).
"""
from __future__ import annotations

import json
import logging
import os
import re
import statistics
from typing import Optional

import anthropic
import httpx

from src.models.schemas import Citation, GroundedAnswer

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# 1. Visual extraction MAE
# ----------------------------------------------------------------------------


def visual_extraction_mae(
    extracted_points: list[dict],  # [{"date": "YYYY-MM-DD", "value": float}, ...]
    ground_truth_history: list[dict],  # [{"date": date, "close": float}, ...]
) -> float:
    """Mean absolute % error between extracted price points and yfinance closes."""
    if not extracted_points or not ground_truth_history:
        return 0.0
    truth_by_date = {str(g["date"]): g["close"] for g in ground_truth_history}
    errors = []
    for p in extracted_points:
        d = str(p.get("date", ""))
        v = p.get("value")
        if d in truth_by_date and isinstance(v, (int, float)) and truth_by_date[d] != 0:
            err = abs(v - truth_by_date[d]) / abs(truth_by_date[d]) * 100.0
            errors.append(err)
    return round(sum(errors) / len(errors), 2) if errors else 0.0


# ----------------------------------------------------------------------------
# 2. Citation faithfulness
# ----------------------------------------------------------------------------


def citation_faithfulness(
    answer: GroundedAnswer,
    *,
    timeout_per_url: float = 5.0,
) -> dict:
    """Verify cited URLs are reachable.

    Returns: {
      "n_cited_urls": int,
      "n_reachable": int,
      "n_unreachable": int,
      "rate": float (0..1),
    }

    For deeper faithfulness (does the URL's content actually contain the
    claim?), use llm_judged_citation_faithfulness which uses a Sonnet judge.
    """
    urls = [c.url for c in answer.citations if c.url]
    if not urls:
        return {"n_cited_urls": 0, "n_reachable": 0, "n_unreachable": 0, "rate": 1.0}

    reachable = 0
    for url in urls:
        try:
            resp = httpx.head(
                url, timeout=timeout_per_url, follow_redirects=True
            )
            if resp.status_code < 400:
                reachable += 1
        except Exception:
            pass

    return {
        "n_cited_urls": len(urls),
        "n_reachable": reachable,
        "n_unreachable": len(urls) - reachable,
        "rate": round(reachable / len(urls), 3),
    }


# ----------------------------------------------------------------------------
# 3. Visual-textual consistency
# ----------------------------------------------------------------------------


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?%?")


def _extract_numbers(text: str) -> list[float]:
    """Pull numeric values out of free text. Treat percentages as floats."""
    out = []
    for m in _NUMBER_RE.findall(text):
        try:
            out.append(float(m.rstrip("%")))
        except ValueError:
            continue
    return out


def visual_textual_consistency(
    answer: GroundedAnswer,
    ground_truth_summary: dict,
) -> float:
    """Fraction of numeric claims in the answer that match ground truth within tolerance.

    `ground_truth_summary` shape: {"return_pct": -8.2, "high": 240.0, "low": 215.5,
    "start_close": 233.0, "end_close": 231.0}.

    A "match" means the answer cites a number within 10% relative tolerance
    of one of the ground-truth values.
    """
    if not ground_truth_summary:
        return 1.0
    truth_values = [v for v in ground_truth_summary.values()
                    if isinstance(v, (int, float))]
    if not truth_values:
        return 1.0

    answer_numbers = _extract_numbers(answer.answer)
    if not answer_numbers:
        return 1.0  # no claims to violate

    def is_close(a: float, b: float) -> bool:
        if b == 0:
            return abs(a) < 0.01
        return abs(a - b) / abs(b) <= 0.10  # 10% tolerance

    matched = sum(
        1 for a in answer_numbers if any(is_close(a, t) for t in truth_values)
    )
    return round(matched / len(answer_numbers), 3)


# ----------------------------------------------------------------------------
# 4. Latency + cost
# ----------------------------------------------------------------------------


def latency_summary(elapsed_seconds_list: list[float]) -> dict:
    """Compute p50/p95 latency from a list of per-query elapsed times."""
    if not elapsed_seconds_list:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0, "n": 0}
    sorted_l = sorted(elapsed_seconds_list)
    return {
        "n": len(sorted_l),
        "mean": round(statistics.mean(sorted_l), 2),
        "p50": round(statistics.median(sorted_l), 2),
        "p95": round(sorted_l[max(0, int(len(sorted_l) * 0.95) - 1)], 2),
    }


# ----------------------------------------------------------------------------
# 5. LLM-judged faithfulness vs. baseline
# ----------------------------------------------------------------------------


_JUDGE_PROMPT = """You are evaluating a financial Q&A answer. Score on three \
metrics, each 0.0 to 1.0:

1. FAITHFULNESS — does the answer avoid claims unsupported by its citations?
   - 1.0: every factual claim corresponds to a cited source
   - 0.5: some claims are unsupported but plausible
   - 0.0: claims contradict or fabricate evidence

2. RELEVANCE — does the answer address the question asked?
   - 1.0: directly addresses every part of the question
   - 0.5: partially addresses
   - 0.0: off-topic or evasive

3. GROUNDEDNESS — does the answer use the chart's actual data + retrieved evidence?
   - 1.0: explicitly references chart data AND retrieved sources
   - 0.5: uses one but not the other
   - 0.0: ignores both, generates from priors

Output ONLY JSON:
{ "faithfulness": 0.0-1.0, "relevance": 0.0-1.0, "groundedness": 0.0-1.0,
  "notes": "<one short sentence on the lowest score>" }"""


def _safe_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1:
            try:
                return json.loads(text[s:e + 1])
            except json.JSONDecodeError:
                pass
    return {}


def llm_judge(question: str, answer: GroundedAnswer,
              *, model: Optional[str] = None,
              api_key: Optional[str] = None) -> dict:
    """Score one answer with the LLM judge."""
    model_id = model or os.getenv("MODEL_JUDGE", "claude-sonnet-4-5-20250929")
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return {"faithfulness": 0.0, "relevance": 0.0, "groundedness": 0.0,
                "notes": "No API key for judge"}

    client = anthropic.Anthropic(api_key=key)
    citations_text = "\n".join(
        f"  - {c.source_type}: {c.title or c.url or '(no title)'}"
        for c in answer.citations
    ) or "  (none)"
    user = (
        f"Question: {question}\n\n"
        f"Answer:\n{answer.answer}\n\n"
        f"Citations:\n{citations_text}"
    )
    try:
        resp = client.messages.create(
            model=model_id,
            max_tokens=400,
            temperature=0.0,
            system=_JUDGE_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        raw = "".join(b.text for b in resp.content if hasattr(b, "text"))
        d = _safe_json(raw)
        return {
            "faithfulness": float(d.get("faithfulness", 0.0)),
            "relevance": float(d.get("relevance", 0.0)),
            "groundedness": float(d.get("groundedness", 0.0)),
            "notes": str(d.get("notes", "")).strip(),
        }
    except Exception as exc:
        logger.warning("Judge failed: %s", exc)
        return {"faithfulness": 0.0, "relevance": 0.0, "groundedness": 0.0,
                "notes": f"Judge error: {exc}"}


# ----------------------------------------------------------------------------
# 6. Refusal rate
# ----------------------------------------------------------------------------


def refusal_rate(
    expected_refusal: list[bool],
    actual_refused: list[bool],
) -> dict:
    """Given parallel lists of (should-refuse, did-refuse) booleans, compute
    refusal accuracy on the cases that should be refused."""
    if not expected_refusal:
        return {"n": 0, "should_refuse": 0, "did_refuse": 0,
                "true_positive_rate": 0.0, "false_positive_rate": 0.0}

    n = len(expected_refusal)
    tp = sum(1 for e, a in zip(expected_refusal, actual_refused) if e and a)
    fn = sum(1 for e, a in zip(expected_refusal, actual_refused) if e and not a)
    fp = sum(1 for e, a in zip(expected_refusal, actual_refused) if not e and a)
    tn = sum(1 for e, a in zip(expected_refusal, actual_refused) if not e and not a)

    n_should = tp + fn
    n_should_not = fp + tn
    return {
        "n": n,
        "should_refuse": n_should,
        "did_refuse": tp + fp,
        "true_positive_rate": round(tp / n_should, 3) if n_should else 0.0,
        "false_positive_rate": round(fp / n_should_not, 3) if n_should_not else 0.0,
    }


# ----------------------------------------------------------------------------
# 7. Judge stability — re-run the judge N times on the same answer
# ----------------------------------------------------------------------------


def judge_stability(
    question: str,
    answer: GroundedAnswer,
    *,
    n_runs: int = 3,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    """Re-run the LLM judge n_runs times on the same (question, answer) pair.

    Used in the methodology section of the eval writeup to demonstrate that
    we measured judge variance rather than ignoring it. A high std means the
    judge gives inconsistent scores for that case; aggregating over many
    questions averages this out.

    Returns:
        {
          "n_runs": int,
          "faithfulness": {"mean", "std", "min", "max"},
          "relevance":    {...},
          "groundedness": {...},
        }
    """
    runs = [llm_judge(question, answer, model=model, api_key=api_key)
            for _ in range(n_runs)]
    if not runs:
        return {"n_runs": 0}

    def stat(key: str) -> dict:
        vals = [r.get(key, 0.0) for r in runs
                if isinstance(r.get(key), (int, float))]
        if not vals:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": round(statistics.mean(vals), 3),
            "std": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 3),
            "max": round(max(vals), 3),
        }

    return {
        "n_runs": n_runs,
        "faithfulness": stat("faithfulness"),
        "relevance": stat("relevance"),
        "groundedness": stat("groundedness"),
    }


# ----------------------------------------------------------------------------
# 8. Confidence interval for aggregate metrics
# ----------------------------------------------------------------------------


def confidence_interval_95(values: list[float]) -> dict:
    """Compute a 95% normal-approximation confidence interval.

    For sample size N >= 10, returns mean and the half-width of the 95% CI
    (mean ± half_width). Standard error = std / sqrt(n); half-width = 1.96 * SE.

    For very small N (< 10), the normal approximation is unreliable; we still
    report the half-width but flag low_n=True so the writeup can caveat
    accordingly.

    Returns:
        {
          "n": int,
          "mean": float,
          "std": float,
          "se": float,           # standard error
          "half_width": float,   # 1.96 * SE — the ± value in "mean ± X"
          "low_n": bool,         # True if N < 10
        }
    """
    if not values:
        return {"n": 0, "mean": 0.0, "half_width": 0.0, "low_n": True}
    n = len(values)
    mean = statistics.mean(values)
    if n < 2:
        return {"n": n, "mean": round(mean, 3), "std": 0.0, "se": 0.0,
                "half_width": 0.0, "low_n": True}
    std = statistics.stdev(values)
    se = std / (n ** 0.5)
    half_width = 1.96 * se
    return {
        "n": n,
        "mean": round(mean, 3),
        "std": round(std, 3),
        "se": round(se, 4),
        "half_width": round(half_width, 3),
        "low_n": n < 10,
    }
