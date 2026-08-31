"""Summarize the 20-session grounded product-card disclosure experiment."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "be",
        "for",
        "from",
        "have",
        "i",
        "in",
        "is",
        "it",
        "item",
        "my",
        "of",
        "on",
        "or",
        "should",
        "the",
        "this",
        "to",
        "use",
        "with",
    }
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    turns = _load_jsonl(args.new_run / "turns.jsonl")
    sessions = _load_jsonl(args.new_run / "sessions.jsonl")
    baseline_root = _load_json(args.baseline_summary)
    baseline_sessions = cast(list[dict[str, object]], baseline_root["sessions"])
    selected_ids = {str(item["sample_id"]) for item in sessions}
    baseline = [item for item in baseline_sessions if str(item["sample_id"]) in selected_ids]
    if len(sessions) != len(selected_ids) or len(baseline) != len(selected_ids):
        raise ValueError("new and baseline runs do not cover the same unique samples")

    report = _build_report(turns=turns, sessions=sessions, baseline=baseline)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(_render_markdown(report), encoding="utf-8")
    return 0


def _build_report(
    *,
    turns: list[dict[str, object]],
    sessions: list[dict[str, object]],
    baseline: list[dict[str, object]],
) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in turns:
        evaluator = cast(dict[str, object], row["evaluator"])
        grouped[str(evaluator["sample_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item["turn"]))

    eligible = [
        row
        for row in turns
        if bool(cast(dict[str, object], row["evaluator"])["override_applied_before_scoring"])
    ]
    post_hit: list[dict[str, object]] = []
    retention_failures: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    for sample_id, rows in sorted(grouped.items()):
        final_rows.append(rows[-1])
        first_hit_index = next(
            (
                index
                for index, row in enumerate(rows)
                if bool(cast(dict[str, object], row["evaluator"])["scored_hit"])
            ),
            None,
        )
        if first_hit_index is None:
            continue
        observed = rows[first_hit_index:]
        post_hit.extend(observed)
        missing = [
            int(row["turn"])
            for row in observed
            if cast(dict[str, object], row["evaluator"])["target_rank"] is None
        ]
        if missing:
            retention_failures.append(
                {
                    "sample_id": sample_id,
                    "missing_turns": missing,
                    "diagnosis": _diagnose_retention(rows, missing),
                }
            )

    qu_failures = [
        {
            "sample_id": str(cast(dict[str, object], row["evaluator"])["sample_id"]),
            "turn": int(row["turn"]),
            "message": str(cast(dict[str, object], row["evaluator"])["simulator_user_message"]),
            "failure": row.get("failure"),
        }
        for row in turns
        if cast(dict[str, object], row["query_understanding"])["status"] != "success"
    ]
    pipeline_failures = [
        {
            "sample_id": str(cast(dict[str, object], row["evaluator"])["sample_id"]),
            "turn": int(row["turn"]),
            "message": str(cast(dict[str, object], row["evaluator"])["simulator_user_message"]),
            "failure": row.get("failure"),
        }
        for row in turns
        if row.get("failure") is not None
    ]
    fact_audit = _fact_coverage(turns)
    timing_values = [
        float(cast(dict[str, object], row["timings"])["total_agent_ms"]) for row in turns
    ]
    transparency = _transparency_by_scenario_and_turn(turns)
    new_by_id = {str(item["sample_id"]): item for item in sessions}
    recovered_baseline_misses = [
        {
            "sample_id": str(item["sample_id"]),
            "new_first_hit_turn": new_by_id[str(item["sample_id"])]["first_hit_turn"],
            "new_best_rank": new_by_id[str(item["sample_id"])]["best_rank"],
        }
        for item in sorted(baseline, key=lambda value: str(value["sample_id"]))
        if not bool(item["hit"]) and bool(new_by_id[str(item["sample_id"])]["hit"])
    ]

    return {
        "schema": "shopping-copilot/product-card-disclosure-run-analysis/v1",
        "comparison_note": (
            "Diagnostic A/B only: the baseline used legacy four-value disclosure and the old "
            "retrieval corpus, while the new run used grounded disclosures plus replacement "
            "cards in lexical, Dense, reranking, and DPP stages."
        ),
        "new_run": {
            "sample_count": len(sessions),
            "turn_count": len(turns),
            **_session_metrics(sessions),
            "eligible_turn_target_presence": _fraction(eligible, _has_target),
            "post_first_hit_retention": _fraction(post_hit, _has_target),
            "final_turn_target_presence": _fraction(final_rows, _has_target),
            "qu_success": {
                "count": len(turns) - len(qu_failures),
                "total": len(turns),
                "rate": (len(turns) - len(qu_failures)) / max(1, len(turns)),
            },
            "fact_lexical_coverage_lower_bound": fact_audit,
            "timing_ms_under_eight_session_contention": {
                "mean": statistics.mean(timing_values),
                "median": statistics.median(timing_values),
                "p95": _percentile(timing_values, 0.95),
                "minimum": min(timing_values),
                "maximum": max(timing_values),
            },
            "transparency_mean_by_scenario_and_turn": transparency,
        },
        "legacy_baseline_same_selection": {
            "sample_count": len(baseline),
            **_session_metrics(baseline),
            "missed_sample_ids": sorted(
                str(item["sample_id"]) for item in baseline if not bool(item["hit"])
            ),
        },
        "pipeline_failures": pipeline_failures,
        "qu_failures": qu_failures,
        "retention_failures": retention_failures,
        "recovered_baseline_misses": recovered_baseline_misses,
    }


def _session_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    hit_count = sum(bool(item["hit"]) for item in rows)
    hit_turns = [int(item["first_hit_turn"]) for item in rows if item["first_hit_turn"] is not None]
    return {
        "hit_count": hit_count,
        "hit_rate_at_10": hit_count / max(1, len(rows)),
        "mrr": sum(float(item["reciprocal_rank"]) for item in rows) / max(1, len(rows)),
        "mttc_among_hits": None if not hit_turns else statistics.mean(hit_turns),
    }


def _fraction(rows: list[dict[str, object]], predicate: Any) -> dict[str, object]:
    count = sum(bool(predicate(row)) for row in rows)
    return {"count": count, "total": len(rows), "rate": count / max(1, len(rows))}


def _has_target(row: dict[str, object]) -> bool:
    return cast(dict[str, object], row["evaluator"])["target_rank"] is not None


def _diagnose_retention(
    rows: list[dict[str, object]],
    missing_turns: list[int],
) -> dict[str, object]:
    details: list[dict[str, object]] = []
    for row in rows:
        if int(row["turn"]) not in missing_turns:
            continue
        evaluator = cast(dict[str, object], row["evaluator"])
        target = str(evaluator["target_parent_asin"])
        retrieval = cast(dict[str, object], row["retrieval"])
        ranking = cast(dict[str, object], row["ranking"])
        routes = cast(list[dict[str, object]], retrieval["routes"])
        details.append(
            {
                "turn": int(row["turn"]),
                "message": evaluator["simulator_user_message"],
                "route_ranks": {
                    str(route["route"]): _rank_in_hits(route.get("hits"), target)
                    for route in routes
                },
                "fused_rank": _rank_in_hits(retrieval.get("fused_candidates"), target),
                "cross_encoder_rank": _rank_in_hits(
                    _result_hits(ranking.get("cross_encoder")),
                    target,
                ),
                "dpp_rank": _rank_in_hits(
                    _result_hits(ranking.get("dpp")),
                    target,
                ),
            }
        )
    return {"turns": details}


def _rank_in_hits(value: object, target: str) -> int | None:
    if type(value) is not list:
        return None
    return next(
        (
            int(item["rank"])
            for item in value
            if type(item) is dict and item.get("parent_asin") == target
        ),
        None,
    )


def _result_hits(value: object) -> object:
    return value.get("hits", []) if type(value) is dict else []


def _fact_coverage(turns: list[dict[str, object]]) -> dict[str, object]:
    covered = 0
    total = 0
    misses: list[dict[str, object]] = []
    for row in turns:
        evaluator = cast(dict[str, object], row["evaluator"])
        facts = cast(list[dict[str, object]], evaluator["simulator_disclosed_facts_this_turn"])
        if not facts:
            continue
        resolved = cast(dict[str, object], row["query_understanding"]).get("resolved_turn")
        searchable = "" if resolved is None else json.dumps(resolved, ensure_ascii=False)
        searchable_tokens = set(_tokens(searchable))
        for fact in facts:
            total += 1
            expected = set(_tokens(str(fact["value"])))
            minimum = 1 if len(expected) <= 1 else math.ceil(len(expected) * 0.5)
            overlap = sorted(expected & searchable_tokens)
            if expected and len(overlap) >= minimum:
                covered += 1
            else:
                misses.append(
                    {
                        "sample_id": evaluator["sample_id"],
                        "turn": row["turn"],
                        "facet": fact["facet"],
                        "value": fact["value"],
                        "matched_tokens": overlap,
                        "qu_status": cast(dict[str, object], row["query_understanding"])["status"],
                    }
                )
    return {
        "count": covered,
        "total": total,
        "rate": covered / max(1, total),
        "method": (
            "At least half of non-stopword value tokens appear in the resolved QU payload; "
            "this is a reproducible lower bound, not a semantic correctness score."
        ),
        "misses": misses,
    }


def _tokens(value: str) -> list[str]:
    return [item for item in TOKEN.findall(value.casefold()) if item not in STOPWORDS]


def _transparency_by_scenario_and_turn(
    turns: list[dict[str, object]],
) -> dict[str, dict[str, float]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in turns:
        value = cast(dict[str, object], row["intent_transparency"])["applied_transparency"]
        if value is None:
            continue
        evaluator = cast(dict[str, object], row["evaluator"])
        grouped[(str(evaluator["scenario_type"]), int(row["turn"]))].append(float(value))
    result: dict[str, dict[str, float]] = defaultdict(dict)
    for (scenario, turn), values in sorted(grouped.items()):
        result[scenario][str(turn)] = statistics.mean(values)
    return dict(result)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def _render_markdown(report: dict[str, object]) -> str:
    new = cast(dict[str, object], report["new_run"])
    old = cast(dict[str, object], report["legacy_baseline_same_selection"])
    qu = cast(dict[str, object], new["qu_success"])
    facts = cast(dict[str, object], new["fact_lexical_coverage_lower_bound"])
    retention = cast(dict[str, object], new["post_first_hit_retention"])
    final = cast(dict[str, object], new["final_turn_target_presence"])
    failures = cast(list[dict[str, object]], report["pipeline_failures"])
    retention_failures = cast(list[dict[str, object]], report["retention_failures"])
    recovered = cast(list[dict[str, object]], report["recovered_baseline_misses"])
    lines = [
        f"# Grounded product-card disclosure: {new['sample_count']}-session run",
        "",
        "## Outcome",
        "",
        "| Metric | Legacy four-value run | Grounded product-card run |",
        "| --- | ---: | ---: |",
        f"| Sessions that ever hit Top 10 | {old['hit_count']}/{old['sample_count']} | {new['hit_count']}/{new['sample_count']} |",
        f"| Hit rate | {float(old['hit_rate_at_10']):.1%} | {float(new['hit_rate_at_10']):.1%} |",
        f"| MRR | {float(old['mrr']):.3f} | {float(new['mrr']):.3f} |",
        f"| Mean first-hit turn (hits only) | {float(old['mttc_among_hits']):.2f} | {float(new['mttc_among_hits']):.2f} |",
        "",
        f"This is a diagnostic A/B, not an isolated causal comparison. {report['comparison_note']}",
        "",
        "## Integrity checks",
        "",
        f"- QU accepted {qu['count']}/{qu['total']} turns ({float(qu['rate']):.1%}).",
        f"- Reproducible lexical fact-coverage lower bound: {facts['count']}/{facts['total']} ({float(facts['rate']):.1%}).",
        f"- Once first found, the target remained in Top 10 on {retention['count']}/{retention['total']} subsequent visible turns ({float(retention['rate']):.1%}).",
        f"- Target was present on the final scripted turn for {final['count']}/{final['total']} sessions ({float(final['rate']):.1%}).",
        f"- Recovered all {len(recovered)} samples that the same-selection legacy baseline missed.",
        "  - " + ", ".join(f"`{item['sample_id']}`" for item in recovered),
        "",
        "## Observed failures",
        "",
    ]
    if not failures:
        lines.append("No pipeline failures were recorded.")
    else:
        for failure in failures:
            detail = cast(dict[str, object], failure["failure"])
            error = cast(dict[str, object], detail["error"])
            lines.append(
                f"- `{failure['sample_id']}` turn {failure['turn']}: "
                f"{detail['stage']} — {error['message']}"
            )
    lines.extend(["", "## Target-retention failures", ""])
    if not retention_failures:
        lines.append("No target disappeared after its first hit.")
    else:
        for failure in retention_failures:
            lines.append(
                f"- `{failure['sample_id']}` lost the target on turns "
                + ", ".join(str(item) for item in failure["missing_turns"])
                + ". See `comparison.json` for route, fusion, BGE, and DPP ranks."
            )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `comparison.json`: machine-readable metrics and per-failure evidence",
            "- `turns.jsonl`: complete QU, Session Context, T_t, retrieval, ranking, and evaluator logs",
            "- `sessions.jsonl`: per-session outcomes",
            "",
        ]
    )
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        cast(dict[str, object], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())
