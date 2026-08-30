"""Rerank selected candidate policies with the pinned BGE + DPP stack.

Candidate generation has already completed in the candidate-recall sweep.  This
script scores the union of requested candidate pools once per query, then applies
the original 0.25 RRF prior blend and T-aware DPP independently to each policy.
The target ASIN is read only after target-blind ranking has completed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for source_path in (ROOT, SRC):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from shopping_copilot.retrieval import (  # noqa: E402
    GreedyDPPSelector,
    SentenceTransformerCrossEncoderScorer,
    VectorCandidate,
    load_dense_index,
    load_product_documents,
)

SCHEMA = "shopping-copilot/bge-dpp-candidate-policy-evaluation/v0"
BGE_MODEL = "BAAI/bge-reranker-v2-m3"
BGE_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
DEFAULT_POLICIES = ("single_k80", "single_k160")


def main() -> int:
    args = _parse_args()
    candidate_report = _load_object(args.candidate_report)
    turns = cast(list[dict[str, object]], candidate_report["turns"])
    policies = tuple(dict.fromkeys(args.policy))
    _validate_policies(turns, policies)
    if args.limit is not None:
        selected = sorted({str(item["sample_id"]) for item in turns})[: args.limit]
        allowed = frozenset(selected)
        turns = [item for item in turns if str(item["sample_id"]) in allowed]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "turns.jsonl"
    existing = _load_jsonl(log_path) if args.resume else []
    if log_path.exists() and not args.resume:
        raise SystemExit(f"output log already exists; use --resume: {log_path}")
    completed = {str(item["identity"]) for item in existing}
    pending = [item for item in turns if str(item["identity"]) not in completed]
    manifest = _mapping(candidate_report, "inputs")
    previous_summary_path = args.output_dir / "summary.json"
    previous_runtime: dict[str, object] = {}
    if args.resume and previous_summary_path.exists():
        previous_runtime = _mapping(_load_object(previous_summary_path), "runtime")
    scorer_model_id = f"{BGE_MODEL}@{BGE_REVISION}"
    initialization_seconds = 0.0
    started = time.perf_counter()
    new_records: list[dict[str, object]] = []
    if pending:
        source_turns = _source_turns(args.source_turn_log)
        print("loading dense index and product documents...", flush=True)
        initialized = time.perf_counter()
        index = load_dense_index(args.dense_index)
        documents = {
            item.parent_asin: _compact_document(item.text)
            for item in load_product_documents(
                args.catalog,
                expected_parent_asins=set(index.parent_asins),
            )
        }
        print("loading pinned BGE reranker...", flush=True)
        scorer = SentenceTransformerCrossEncoderScorer(
            BGE_MODEL,
            revision=BGE_REVISION,
            device=args.device,
            local_files_only=True,
            max_length=384,
        )
        scorer_model_id = scorer.model_id
        selector = GreedyDPPSelector(index=index)
        initialization_seconds = time.perf_counter() - initialized

        with log_path.open("a", encoding="utf-8", buffering=1) as stream:
            for ordinal, turn in enumerate(pending, start=1):
                identity = str(turn["identity"])
                pools = {policy: _candidate_pool(turn, policy) for policy in policies}
                union = tuple(
                    dict.fromkeys(
                        parent_asin for policy in policies for parent_asin in pools[policy][0]
                    )
                )
                score_started = time.perf_counter()
                raw_scores = scorer.score(
                    str(turn["q_sem"]),
                    [documents[parent_asin] for parent_asin in union],
                    batch_size=args.batch_size,
                )
                score_ms = (time.perf_counter() - score_started) * 1000.0
                by_asin = dict(zip(union, raw_scores, strict=True))
                ranked = {
                    policy: _rank_policy(
                        candidates=pools[policy][0],
                        prior_relevance=pools[policy][1],
                        raw_scores=by_asin,
                        selector=selector,
                        relevance_weight=float(turn["relevance_weight"]),
                    )
                    for policy in policies
                }
                target = str(turn["target_parent_asin"])
                variants = {
                    policy: {
                        **payload,
                        "target_bge_rank": _target_rank(
                            cast(list[dict[str, object]], payload["bge_ranking"]),
                            target,
                        ),
                        "target_dpp_rank": _target_rank(
                            cast(list[dict[str, object]], payload["dpp_slate"]),
                            target,
                        ),
                    }
                    for policy, payload in ranked.items()
                }
                expected = _saved_dpp(source_turns[identity])
                observed = tuple(
                    str(item["parent_asin"])
                    for item in cast(
                        list[dict[str, object]],
                        cast(dict[str, object], variants.get("single_k80", {})).get(
                            "dpp_slate", []
                        ),
                    )
                )
                parity_ok = "single_k80" not in policies or observed == expected
                record = {
                    "identity": identity,
                    "sample_id": turn["sample_id"],
                    "scenario_type": turn["scenario_type"],
                    "turn": turn["turn"],
                    "representative": turn["representative"],
                    "target_parent_asin": target,
                    "target_was_not_passed_to_ranking": True,
                    "q_sem": turn["q_sem"],
                    "relevance_weight": turn["relevance_weight"],
                    "scored_union_count": len(union),
                    "score_ms": score_ms,
                    "parity": {
                        "single_k80_matches_saved_bge_dpp": parity_ok,
                    },
                    "variants": variants,
                }
                new_records.append(record)
                stream.write(_json_line(record))
                stream.flush()
                if ordinal % 5 == 0 or ordinal == len(pending):
                    print(
                        f"BGE+DPP {ordinal}/{len(pending)} pairs={len(union)} "
                        f"score={score_ms:.1f}ms",
                        flush=True,
                    )
    else:
        print("all selected turns are complete; rebuilding summaries only", flush=True)

    records = [*existing, *new_records]
    parity_failures = sum(
        not bool(_mapping(record, "parity")["single_k80_matches_saved_bge_dpp"])
        for record in records
    )
    if "single_k80" in policies and parity_failures:
        raise RuntimeError(f"saved BGE+DPP parity failed on {parity_failures} turns")
    summary = _summarize(records, policies=policies)
    evaluation_seconds = (
        time.perf_counter() - started
        if pending
        else float(previous_runtime.get("evaluation_seconds", 0.0))
    )
    if not pending:
        initialization_seconds = float(previous_runtime.get("initialization_seconds", 0.0))
    report = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "candidate_report": str(args.candidate_report.resolve()),
            "source_turn_log": str(args.source_turn_log.resolve()),
            "catalog": str(args.catalog.resolve()),
            "dense_index": str(args.dense_index.resolve()),
            "catalog_semantic_release": manifest.get("semantic_release"),
        },
        "protocol": {
            "policies": list(policies),
            "cross_encoder": scorer_model_id,
            "cross_encoder_prior_weight": 0.25,
            "batch_size": args.batch_size,
            "target_visible_to_ranking": False,
            "turn_trajectory": "same fresh scored turns as the saved official run",
        },
        "runtime": {
            "device": args.device,
            "initialization_seconds": initialization_seconds,
            "evaluation_seconds": evaluation_seconds,
            "turn_count": len(records),
            "single_k80_parity_failure_count": parity_failures,
            "mean_score_ms": float(np.mean([float(item["score_ms"]) for item in records])),
        },
        "summary": summary,
        "paired_analysis": _paired_analysis(records, policies=policies),
    }
    _write_json(args.output_dir / "summary.json", report)
    (args.output_dir / "summary.md").write_text(
        _render_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    print(args.output_dir.resolve(), flush=True)
    return 0


def _candidate_pool(
    turn: dict[str, object],
    policy: str,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    variant = cast(dict[str, object], _mapping(turn, "variants")[policy])
    candidates = tuple(str(item) for item in cast(list[object], variant["candidates"]))
    relevance = tuple(
        float(item) for item in cast(list[object], variant["normalized_fusion_relevance"])
    )
    if len(candidates) != len(relevance):
        raise ValueError("candidate and relevance lengths differ")
    return candidates, relevance


def _rank_policy(
    *,
    candidates: tuple[str, ...],
    prior_relevance: tuple[float, ...],
    raw_scores: dict[str, float],
    selector: GreedyDPPSelector,
    relevance_weight: float,
) -> dict[str, object]:
    observed = tuple(float(raw_scores[parent_asin]) for parent_asin in candidates)
    normalized = _min_max(observed)
    blended = tuple(
        0.25 * prior + 0.75 * model
        for prior, model in zip(prior_relevance, normalized, strict=True)
    )
    order = sorted(
        range(len(candidates)),
        key=lambda index: (-blended[index], candidates[index]),
    )
    bge_ranking = [
        {
            "parent_asin": candidates[index],
            "rank": rank,
            "candidate_rank": index + 1,
            "raw_model_score": observed[index],
            "normalized_model_score": normalized[index],
            "prior_relevance": prior_relevance[index],
            "relevance": blended[index],
        }
        for rank, index in enumerate(order, start=1)
    ]
    reranked = tuple(
        VectorCandidate(
            parent_asin=str(item["parent_asin"]),
            candidate_rank=int(item["rank"]),
            relevance=float(item["relevance"]),
        )
        for item in bge_ranking
    )
    dpp = selector.select(
        reranked,
        top_k=10,
        relevance_weight=relevance_weight,
    )
    return {
        "candidate_count": len(candidates),
        "bge_ranking": bge_ranking,
        "dpp_slate": [
            {
                "parent_asin": item.parent_asin,
                "rank": item.rank,
                "candidate_rank": item.candidate_rank,
                "relevance": item.relevance,
                "maximum_similarity_to_selected": item.maximum_similarity_to_selected,
                "selection_score": item.selection_score,
            }
            for item in dpp.hits
        ],
    }


def _summarize(
    records: list[dict[str, object]],
    *,
    policies: tuple[str, ...],
) -> dict[str, object]:
    by_sample: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_sample[str(record["sample_id"])].append(record)
    return {policy: _policy_summary(by_sample, policy) for policy in policies}


def _policy_summary(
    by_sample: dict[str, list[dict[str, object]]],
    policy: str,
) -> dict[str, object]:
    sessions = []
    by_scenario: dict[str, list[dict[str, object]]] = defaultdict(list)
    for sample_id, rows in by_sample.items():
        ordered = sorted(rows, key=lambda item: int(item["turn"]))
        hit_turn = None
        hit_rank = None
        for row in ordered:
            variant = cast(dict[str, object], _mapping(row, "variants")[policy])
            rank = variant["target_dpp_rank"]
            if rank is not None:
                hit_turn = int(row["turn"])
                hit_rank = int(rank)
                break
        session = {
            "sample_id": sample_id,
            "scenario_type": ordered[0]["scenario_type"],
            "hit": hit_turn is not None,
            "first_hit_turn": hit_turn,
            "rank": hit_rank,
        }
        sessions.append(session)
        by_scenario[str(session["scenario_type"])].append(session)
    return {
        **_metrics(sessions),
        "scenario_metrics": {name: _metrics(items) for name, items in sorted(by_scenario.items())},
    }


def _paired_analysis(
    records: list[dict[str, object]],
    *,
    policies: tuple[str, ...],
) -> dict[str, object]:
    by_sample: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_sample[str(record["sample_id"])].append(record)

    outcomes: dict[str, dict[str, dict[str, object]]] = {}
    funnels = {
        policy: {
            "hit": 0,
            "candidate_absent": 0,
            "candidate_present_below_bge_top_10": 0,
            "bge_top_10_removed_by_dpp": 0,
        }
        for policy in policies
    }
    for sample_id, raw_rows in by_sample.items():
        rows = sorted(raw_rows, key=lambda item: int(item["turn"]))
        outcomes[sample_id] = {}
        for policy in policies:
            variants = [cast(dict[str, object], _mapping(row, "variants")[policy]) for row in rows]
            candidate_present = any(item["target_bge_rank"] is not None for item in variants)
            bge_top_10 = any(
                item["target_bge_rank"] is not None and int(item["target_bge_rank"]) <= 10
                for item in variants
            )
            hits = [
                (int(row["turn"]), int(variant["target_dpp_rank"]))
                for row, variant in zip(rows, variants, strict=True)
                if variant["target_dpp_rank"] is not None
            ]
            if hits:
                stage = "hit"
            elif bge_top_10:
                stage = "bge_top_10_removed_by_dpp"
            elif candidate_present:
                stage = "candidate_present_below_bge_top_10"
            else:
                stage = "candidate_absent"
            funnels[policy][stage] += 1
            outcomes[sample_id][policy] = {
                "hit": bool(hits),
                "first_hit": hits[0] if hits else None,
                "stage": stage,
            }

    result: dict[str, object] = {"failure_funnel": funnels}
    if len(policies) < 2:
        return result

    baseline, comparison = policies[:2]
    wins: list[dict[str, object]] = []
    losses: list[dict[str, object]] = []
    both = 0
    neither = 0
    for sample_id, policy_outcomes in outcomes.items():
        baseline_outcome = policy_outcomes[baseline]
        comparison_outcome = policy_outcomes[comparison]
        baseline_hit = bool(baseline_outcome["hit"])
        comparison_hit = bool(comparison_outcome["hit"])
        rows = sorted(by_sample[sample_id], key=lambda item: int(item["turn"]))
        scenario = str(rows[0]["scenario_type"])
        if baseline_hit and comparison_hit:
            both += 1
        elif not baseline_hit and not comparison_hit:
            neither += 1
        elif comparison_hit:
            hit_turn, hit_rank = cast(tuple[int, int], comparison_outcome["first_hit"])
            wins.append(
                {
                    "sample_id": sample_id,
                    "scenario_type": scenario,
                    "first_hit_turn": hit_turn,
                    "dpp_rank": hit_rank,
                }
            )
        else:
            hit_turn, hit_rank = cast(tuple[int, int], baseline_outcome["first_hit"])
            comparison_row = next(row for row in rows if int(row["turn"]) == hit_turn)
            comparison_variant = cast(
                dict[str, object], _mapping(comparison_row, "variants")[comparison]
            )
            losses.append(
                {
                    "sample_id": sample_id,
                    "scenario_type": scenario,
                    "baseline_hit_turn": hit_turn,
                    "baseline_dpp_rank": hit_rank,
                    "comparison_bge_rank_at_baseline_hit": comparison_variant["target_bge_rank"],
                    "comparison_dpp_rank_at_baseline_hit": comparison_variant["target_dpp_rank"],
                }
            )
    result["comparison"] = {
        "baseline": baseline,
        "comparison": comparison,
        "both_hit": both,
        "neither_hit": neither,
        "comparison_wins": wins,
        "comparison_losses": losses,
        "net_hit_sessions": len(wins) - len(losses),
    }
    return result


def _metrics(sessions: list[dict[str, object]]) -> dict[str, object]:
    count = len(sessions)
    hits = [item for item in sessions if bool(item["hit"])]
    reciprocal = sum(1.0 / int(item["rank"]) for item in hits) / count
    mttc = (
        sum(
            int(item["first_hit_turn"]) if item["first_hit_turn"] is not None else 11
            for item in sessions
        )
        / count
    )
    return {
        "sample_count": count,
        "hit_rate_at_10": len(hits) / count,
        "mrr": reciprocal,
        "mttc": mttc,
    }


def _source_turns(path: Path) -> dict[str, dict[str, object]]:
    result = {}
    for row in _load_jsonl(path):
        evaluator = _mapping(row, "evaluator")
        identity = f"{evaluator['sample_id']}/turn-{row['turn']}"
        result[identity] = row
    return result


def _saved_dpp(row: dict[str, object]) -> tuple[str, ...]:
    ranking = _mapping(row, "ranking")
    dpp = _mapping(ranking, "dpp")
    return tuple(str(item["parent_asin"]) for item in cast(list[dict[str, object]], dpp["hits"]))


def _target_rank(items: list[dict[str, object]], target: str) -> int | None:
    return next(
        (int(item["rank"]) for item in items if item["parent_asin"] == target),
        None,
    )


def _min_max(values: tuple[float, ...]) -> tuple[float, ...]:
    if not values:
        return ()
    if any(not math.isfinite(item) for item in values):
        raise ValueError("model scores must be finite")
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        return tuple(1.0 for _ in values)
    return tuple((item - minimum) / (maximum - minimum) for item in values)


def _compact_document(text: str) -> str:
    kept = []
    for line in text.splitlines():
        label = line.partition(":")[0]
        if label in {"title", "categories", "store", "features", "details"}:
            kept.append(line)
    return "\n".join(kept)[:2400]


def _validate_policies(turns: list[dict[str, object]], policies: tuple[str, ...]) -> None:
    if not policies:
        raise ValueError("at least one policy is required")
    available = set(_mapping(turns[0], "variants"))
    unknown = set(policies) - available
    if unknown:
        raise ValueError(f"unknown candidate policies: {sorted(unknown)}")


def _render_markdown(report: dict[str, object]) -> str:
    summary = _mapping(report, "summary")
    paired = _mapping(report, "paired_analysis")
    lines = [
        "# BGE + DPP candidate policy evaluation v0",
        "",
        "The same saved target-blind query trajectory is used for every candidate policy.",
        "",
        "| policy | Hit@10 | MRR | MTTC |",
        "| --- | ---: | ---: | ---: |",
    ]
    for policy, raw in summary.items():
        item = cast(dict[str, object], raw)
        lines.append(
            f"| {policy} | {float(item['hit_rate_at_10']):.3f} | "
            f"{float(item['mrr']):.3f} | {float(item['mttc']):.3f} |"
        )
    policy_names = list(summary)
    if len(policy_names) >= 2:
        baseline, comparison = policy_names[:2]
        baseline_summary = cast(dict[str, object], summary[baseline])
        comparison_summary = cast(dict[str, object], summary[comparison])
        baseline_scenarios = _mapping(baseline_summary, "scenario_metrics")
        comparison_scenarios = _mapping(comparison_summary, "scenario_metrics")
        lines.extend(
            [
                "",
                "## Scenario breakdown",
                "",
                f"| scenario | sessions | {baseline} Hit@10 | {comparison} Hit@10 | change |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for scenario, raw_baseline in baseline_scenarios.items():
            baseline_item = cast(dict[str, object], raw_baseline)
            comparison_item = cast(dict[str, object], comparison_scenarios[scenario])
            old = float(baseline_item["hit_rate_at_10"])
            new = float(comparison_item["hit_rate_at_10"])
            lines.append(
                f"| {scenario} | {int(baseline_item['sample_count'])} | {old:.3f} | "
                f"{new:.3f} | {new - old:+.3f} |"
            )

    funnels = _mapping(paired, "failure_funnel")
    lines.extend(
        [
            "",
            "## Session-level failure funnel",
            "",
            "Each session is assigned to the deepest stage reached at least once during its "
            "saved query trajectory.",
            "",
            "| policy | final hit | candidate absent | candidate present, below BGE Top 10 | "
            "BGE Top 10, removed by DPP |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for policy, raw_funnel in funnels.items():
        funnel = cast(dict[str, object], raw_funnel)
        lines.append(
            f"| {policy} | {int(funnel['hit'])} | {int(funnel['candidate_absent'])} | "
            f"{int(funnel['candidate_present_below_bge_top_10'])} | "
            f"{int(funnel['bge_top_10_removed_by_dpp'])} |"
        )

    raw_comparison = paired.get("comparison")
    if type(raw_comparison) is dict:
        comparison = cast(dict[str, object], raw_comparison)
        wins = cast(list[dict[str, object]], comparison["comparison_wins"])
        losses = cast(list[dict[str, object]], comparison["comparison_losses"])
        lines.extend(
            [
                "",
                "## Paired comparison",
                "",
                f"- Both policies hit: {int(comparison['both_hit'])} sessions.",
                f"- Neither policy hit: {int(comparison['neither_hit'])} sessions.",
                f"- Comparison wins: {len(wins)} sessions.",
                f"- Baseline wins: {len(losses)} sessions.",
                f"- Net effect: {int(comparison['net_hit_sessions']):+d} sessions.",
                "",
                "Comparison-policy wins:",
                "",
                "| sample | scenario | first hit turn | DPP rank |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for item in wins:
            lines.append(
                f"| {item['sample_id']} | {item['scenario_type']} | "
                f"{int(item['first_hit_turn'])} | {int(item['dpp_rank'])} |"
            )
        lines.extend(
            [
                "",
                "Comparison-policy regressions:",
                "",
                "| sample | scenario | baseline hit | comparison BGE rank |",
                "| --- | --- | --- | ---: |",
            ]
        )
        for item in losses:
            lines.append(
                f"| {item['sample_id']} | {item['scenario_type']} | turn "
                f"{int(item['baseline_hit_turn'])}, DPP rank "
                f"{int(item['baseline_dpp_rank'])} | "
                f"{int(item['comparison_bge_rank_at_baseline_hit'])} |"
            )

    runtime = _mapping(report, "runtime")
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Evaluated {int(runtime['turn_count'])} saved searchable turns.",
            "- Target ASINs were revealed only after target-blind ranking completed.",
            f"- Saved K=80 slate parity failures: "
            f"{int(runtime['single_k80_parity_failure_count'])}.",
            "",
            "`single_k80` must exactly reproduce the saved BGE+DPP slates before wider-pool "
            "results are accepted.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-report",
        type=Path,
        default=ROOT / "artifacts/retrieval/candidate-recall-sweep-v0.json",
    )
    parser.add_argument(
        "--source-turn-log",
        type=Path,
        default=ROOT / "artifacts/simulator/full-pipeline-other-200-v0/turns.jsonl",
    )
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.jsonl")
    parser.add_argument(
        "--dense-index",
        type=Path,
        default=ROOT / "artifacts/retrieval/dense-v0",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--policy", action="append", default=list(DEFAULT_POLICIES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/retrieval/bge-dpp-candidate-policy-v0",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("input must be a JSON object")
    return cast(dict[str, object], value)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [
        cast(dict[str, object], json.loads(line))
        for line in path.open(encoding="utf-8")
        if line.strip()
    ]


def _mapping(mapping: dict[str, object], key: str) -> dict[str, object]:
    value = mapping.get(key)
    if type(value) is not dict:
        raise ValueError(f"{key} must be an object")
    return cast(dict[str, object], value)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _json_line(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
