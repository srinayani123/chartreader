"""Run the full ChartReader eval and produce a metrics report.

Compares 4 approaches:
  1. crew              — full Pydantic AI agent (vision + verify + retrieval + peers)
  2. baseline          — single-LLM call, vision + question, nothing else
  3. vision_only       — vision-capable LLM with refusal-by-prompt, no retrieval
  4. retrieval_only    — vision description + Tavily news, no yfinance verification

Run with:
  python -m eval.run_eval                  # full eval (all 40 questions)
  python -m eval.run_eval --smoke          # smoke test (3 questions, all 4 approaches)
  python -m eval.run_eval --smoke --n=5    # smoke test with N questions

Outputs:
  - Console summary
  - eval/results.json (full per-question detail)
  - eval/results_smoke.json (if --smoke)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make src importable when run as module
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from eval.baseline import baseline_analyze
from eval.baseline_retrieval_only import retrieval_only_analyze
from eval.baseline_vision_only import vision_only_analyze
from eval.metrics import (
    citation_faithfulness,
    latency_summary,
    llm_judge,
    refusal_rate,
    visual_extraction_mae,
    visual_textual_consistency,
)
from src.agents.analyst import analyze_chart
from src.models.schemas import GroundedAnswer

load_dotenv()
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


TEST_SET_PATH = Path(os.getenv("EVAL_TEST_SET_PATH", "eval/test_set"))


# Approach name → async function
APPROACHES = {
    "crew": analyze_chart,
    "baseline": baseline_analyze,
    "vision_only": vision_only_analyze,
    "retrieval_only": retrieval_only_analyze,
}


def _load_test_set() -> tuple[dict, list[dict]]:
    """Load ground_truth.json + questions.json."""
    gt_path = TEST_SET_PATH / "ground_truth.json"
    q_path = TEST_SET_PATH / "questions.json"
    if not gt_path.exists() or not q_path.exists():
        return {}, []
    ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
    questions = json.loads(q_path.read_text(encoding="utf-8"))
    return ground_truth, questions


def _resolve_chart_path(chart_id: str) -> Path | None:
    """Find chart file with any common extension."""
    base = TEST_SET_PATH / "charts"
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = base / f"{chart_id}{ext}"
        if p.exists():
            return p
    return None


async def _run_one(
    approach_name: str,
    qrec: dict,
    ground_truth: dict,
) -> dict:
    """Run a single (approach, question) pair."""
    chart_id = qrec["chart_id"]
    question = qrec["question"]
    chart_path = _resolve_chart_path(chart_id)
    if chart_path is None:
        return {
            "approach": approach_name,
            "chart_id": chart_id,
            "question": question,
            "error": f"chart image not found for {chart_id}",
        }

    chart_truth = ground_truth.get(chart_id, {})
    fn = APPROACHES[approach_name]

    try:
        answer: GroundedAnswer = await fn(str(chart_path), question)
    except Exception as exc:
        logger.exception("[%s] %s failed: %s", approach_name, chart_id, exc)
        return {
            "approach": approach_name,
            "chart_id": chart_id,
            "question": question,
            "error": str(exc),
        }

    record = {
        "approach": approach_name,
        "qid": qrec.get("id"),
        "chart_id": chart_id,
        "question": question,
        "category": qrec.get("category", "descriptive"),
        "expected_refusal": qrec.get("expected_refusal", False),
        "actual_refused": answer.refused,
        "answer_text": answer.answer,
        "n_citations": len(answer.citations),
        "n_tool_calls": answer.n_tool_calls,
        "elapsed_seconds": answer.elapsed_seconds,
    }

    # Metrics for non-refused answers only
    if not answer.refused:
        record["citation_faithfulness"] = citation_faithfulness(answer)
        record["visual_textual_consistency"] = visual_textual_consistency(
            answer, chart_truth.get("summary", {})
        )

    # LLM judge — skip for refusal-test cases (judge isn't designed for those)
    if not qrec.get("expected_refusal", False):
        record["judge"] = llm_judge(question, answer)

    return record


def _aggregate(records: list[dict], approach: str) -> dict:
    """Compute summary metrics across all records for one approach."""
    rs = [r for r in records if r.get("approach") == approach]
    if not rs:
        return {"approach": approach, "n": 0}

    non_error = [r for r in rs if "error" not in r]
    non_refused = [r for r in non_error if not r.get("actual_refused", False)]
    judged = [r for r in non_error if "judge" in r]

    expected_refusal = [r.get("expected_refusal", False) for r in rs]
    actual_refused = [r.get("actual_refused", False) for r in rs]

    elapsed = [r["elapsed_seconds"] for r in non_error if r.get("elapsed_seconds")]

    # Per-category breakdown
    by_category: dict[str, dict] = {}
    for r in non_error:
        cat = r.get("category", "uncategorized")
        d = by_category.setdefault(cat, {"n": 0, "elapsed": [], "judged": []})
        d["n"] += 1
        if r.get("elapsed_seconds"):
            d["elapsed"].append(r["elapsed_seconds"])
        if "judge" in r:
            d["judged"].append(r["judge"])

    cat_summary = {}
    for cat, d in by_category.items():
        cat_summary[cat] = {
            "n": d["n"],
            "mean_latency": (round(sum(d["elapsed"]) / len(d["elapsed"]), 2)
                             if d["elapsed"] else 0.0),
            "mean_judge_faithfulness": (
                round(sum(j["faithfulness"] for j in d["judged"]) / len(d["judged"]), 3)
                if d["judged"] else None
            ),
            "mean_judge_groundedness": (
                round(sum(j["groundedness"] for j in d["judged"]) / len(d["judged"]), 3)
                if d["judged"] else None
            ),
        }

    return {
        "approach": approach,
        "n_total": len(rs),
        "n_completed": len(non_error),
        "n_errored": len(rs) - len(non_error),
        "latency": latency_summary(elapsed),
        "refusal": refusal_rate(expected_refusal, actual_refused),
        "citation_faithfulness_mean": round(
            sum(r["citation_faithfulness"]["rate"] for r in non_refused
                if "citation_faithfulness" in r)
            / max(1, sum(1 for r in non_refused if "citation_faithfulness" in r)),
            3,
        ) if non_refused else 0.0,
        "visual_textual_consistency_mean": round(
            sum(r.get("visual_textual_consistency", 0.0) for r in non_refused)
            / max(1, len(non_refused)),
            3,
        ) if non_refused else 0.0,
        "judge_faithfulness_mean": round(
            sum(r["judge"]["faithfulness"] for r in judged) / max(1, len(judged)),
            3,
        ) if judged else 0.0,
        "judge_relevance_mean": round(
            sum(r["judge"]["relevance"] for r in judged) / max(1, len(judged)),
            3,
        ) if judged else 0.0,
        "judge_groundedness_mean": round(
            sum(r["judge"]["groundedness"] for r in judged) / max(1, len(judged)),
            3,
        ) if judged else 0.0,
        "by_category": cat_summary,
    }


async def run(
    smoke: bool = False,
    smoke_n: int = 3,
    approaches: list[str] | None = None,
) -> dict:
    """Run eval suite. Returns aggregated metrics."""
    ground_truth, questions = _load_test_set()
    if not questions:
        return {
            "error": (
                "No test set found at "
                f"{TEST_SET_PATH}. See eval/test_set/README.md."
            )
        }

    if approaches is None:
        approaches = list(APPROACHES.keys())

    if smoke:
        # Pick a diverse smoke set: 1 descriptive, 1 refusal, 1 adversarial if possible
        questions = _pick_smoke_questions(questions, smoke_n)
        print(f"[SMOKE] Running {len(questions)} questions × {len(approaches)} approaches")
    else:
        print(f"Running {len(questions)} questions × {len(approaches)} approaches")

    all_records: list[dict] = []
    for i, approach in enumerate(approaches):
        print(f"\n[{i + 1}/{len(approaches)}] Approach: {approach}")
        for j, q in enumerate(questions):
            print(f"  Q{j + 1}/{len(questions)}: {q['question'][:60]}...")
            r = await _run_one(approach, q, ground_truth)
            all_records.append(r)

    # Aggregate per approach
    summaries = {a: _aggregate(all_records, a) for a in approaches}

    # Deltas vs the simplest baseline (the naive "baseline" approach)
    deltas = {}
    if "baseline" in summaries and "crew" in summaries:
        crew_s = summaries["crew"]
        base_s = summaries["baseline"]
        deltas["crew_vs_baseline"] = {
            "judge_faithfulness": round(
                crew_s["judge_faithfulness_mean"]
                - base_s["judge_faithfulness_mean"], 3,
            ),
            "judge_groundedness": round(
                crew_s["judge_groundedness_mean"]
                - base_s["judge_groundedness_mean"], 3,
            ),
            "judge_relevance": round(
                crew_s["judge_relevance_mean"]
                - base_s["judge_relevance_mean"], 3,
            ),
        }

    report = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "smoke": smoke,
        "n_questions": len(questions),
        "approaches_compared": approaches,
        "summaries": summaries,
        "deltas": deltas,
        "per_question": all_records,
    }

    out_path = Path("eval/results_smoke.json" if smoke else "eval/results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nResults written to {out_path}")

    return report


def _pick_smoke_questions(questions: list[dict], n: int) -> list[dict]:
    """Pick a diverse smoke subset: descriptive + refusal + adversarial if available."""
    by_cat: dict[str, list[dict]] = {}
    for q in questions:
        by_cat.setdefault(q.get("category", "descriptive"), []).append(q)

    chosen: list[dict] = []
    # First one descriptive (the bread-and-butter case)
    if "descriptive" in by_cat:
        chosen.append(by_cat["descriptive"][0])
    # Then a refusal if available
    for refusal_cat in ("advice_refusal", "prediction_refusal"):
        if refusal_cat in by_cat and len(chosen) < n:
            chosen.append(by_cat[refusal_cat][0])
            break
    # Then an adversarial if room
    for adv_cat in ("adversarial_mixed", "adversarial_target",
                    "adversarial_period", "adversarial_wrongticker",
                    "adversarial_irrelevant"):
        if adv_cat in by_cat and len(chosen) < n:
            chosen.append(by_cat[adv_cat][0])
            break
    # Fill remaining slots with comparative or more descriptive
    if len(chosen) < n and "comparative" in by_cat:
        chosen.append(by_cat["comparative"][0])
    while len(chosen) < n and "descriptive" in by_cat and len(by_cat["descriptive"]) > len(
        [c for c in chosen if c.get("category") == "descriptive"]
    ):
        next_idx = len([c for c in chosen if c.get("category") == "descriptive"])
        chosen.append(by_cat["descriptive"][next_idx])
    return chosen[:n]


def _print_summary(report: dict) -> None:
    print("\n" + "=" * 78)
    print(" CHARTREADER EVAL SUMMARY" + ("  [SMOKE]" if report.get("smoke") else ""))
    print("=" * 78)
    print(f" Questions: {report['n_questions']}")
    print(f" Approaches: {', '.join(report['approaches_compared'])}")
    print(f" Run at:    {report.get('ran_at')}")
    print()

    # Compact comparison table
    summaries = report["summaries"]
    rows = ["approach", "n", "completed", "p95 lat", "judge_faith", "judge_grnd",
            "cit_faith", "vis_txt"]
    print(f" {'approach':16} {'comp':>6} {'p95':>7} {'faith':>7} {'grnd':>7} "
          f"{'rel':>7} {'cit':>7} {'vis':>7} {'refTPR':>7} {'refFPR':>7}")
    print(" " + "-" * 76)
    for a, s in summaries.items():
        print(f" {a:16} {s['n_completed']:>3}/{s['n_total']:<3} "
              f"{s['latency']['p95']:>7.1f} "
              f"{s['judge_faithfulness_mean']:>7.3f} "
              f"{s['judge_groundedness_mean']:>7.3f} "
              f"{s['judge_relevance_mean']:>7.3f} "
              f"{s['citation_faithfulness_mean']:>7.3f} "
              f"{s['visual_textual_consistency_mean']:>7.3f} "
              f"{s['refusal']['true_positive_rate']:>7.3f} "
              f"{s['refusal']['false_positive_rate']:>7.3f}")
    print()

    if report.get("deltas", {}).get("crew_vs_baseline"):
        d = report["deltas"]["crew_vs_baseline"]
        print(" Crew vs. Naive Baseline (positive = crew wins):")
        print(f"   Faithfulness delta:  {d['judge_faithfulness']:+.3f}")
        print(f"   Groundedness delta:  {d['judge_groundedness']:+.3f}")
        print(f"   Relevance delta:     {d['judge_relevance']:+.3f}")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Run a small smoke test (3 questions × 4 approaches)")
    parser.add_argument("--n", type=int, default=3,
                        help="Number of questions in smoke mode (default 3)")
    parser.add_argument("--approaches", type=str, default=None,
                        help="Comma-separated list of approaches to run "
                             "(default: all 4)")
    args = parser.parse_args()

    approaches = None
    if args.approaches:
        approaches = [a.strip() for a in args.approaches.split(",")]
        unknown = [a for a in approaches if a not in APPROACHES]
        if unknown:
            print(f"Unknown approaches: {unknown}. Valid: {list(APPROACHES.keys())}")
            sys.exit(1)

    report = asyncio.run(run(smoke=args.smoke, smoke_n=args.n, approaches=approaches))
    if "error" in report:
        print(report["error"])
        sys.exit(1)
    _print_summary(report)


if __name__ == "__main__":
    main()
