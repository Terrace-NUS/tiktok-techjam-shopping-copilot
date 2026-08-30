"""Replay saved simulator states through wider and atomic candidate retrieval.

The experiment never calls Query Understanding and never exposes a target ASIN to
retrieval.  It reads the target only after each target-blind candidate list has
been produced.  Three dense-query shapes are compared on the same hard mask:

* ``single``: the current composite ``q_sem`` dense route;
* ``atomic``: one dense route per positive goal/preference factor;
* ``hybrid``: composite and atomic dense routes together.

Lexical and facet routes are shared.  Route and fusion K are swept together so
the report can separate pre-fusion coverage from RRF truncation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for source_path in (ROOT, SRC):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from shopping_copilot.query_compiler import (  # noqa: E402
    COMPILED_QUERY_SCHEMA,
    CompilationTarget,
    CompiledDirectives,
    CompiledHardConstraint,
    CompiledQuery,
    CompiledRankingPreference,
    ConstraintPolicy,
    DiversityDirective,
    PreferenceCompilationTrace,
    RankingReason,
)
from shopping_copilot.retrieval import (  # noqa: E402
    FormalRetrievalPolicy,
    RecallStrategy,
    RouteHit,
    VectorDiversityPolicy,
    create_retrieval_controller,
    dense_route_observation,
)
from shopping_copilot.session_context import (  # noqa: E402
    Commitment,
    Operator,
    PreferenceSource,
    SemanticPolarity,
)

SCHEMA = "shopping-copilot/candidate-recall-sweep/v0"
DEFAULT_K_VALUES = (80, 160, 256, 400)
SHAPES = ("single", "atomic", "hybrid")


@dataclass(frozen=True, slots=True)
class ReplayCase:
    identity: str
    sample_id: str
    scenario_type: str
    turn: int
    target_parent_asin: str
    compiled: CompiledQuery
    atomic_factors: tuple[str, ...]
    logged_fused_top_80: tuple[str, ...]
    relevance_weight: float
    representative: bool


@dataclass(frozen=True, slots=True)
class NamedRoute:
    name: str
    hits: tuple[RouteHit, ...]


def main() -> int:
    args = _parse_args()
    k_values = tuple(sorted(set(args.k)))
    maximum_k = max(k_values)
    cases = _load_cases(args.turn_log)
    if args.limit is not None:
        selected_samples = sorted({item.sample_id for item in cases})[: args.limit]
        selected_set = frozenset(selected_samples)
        cases = tuple(item for item in cases if item.sample_id in selected_set)
    if not cases:
        raise SystemExit("no searchable scored turns were found")

    print(
        f"loading retrieval runtime for {len(cases)} turns / "
        f"{len({item.sample_id for item in cases})} sessions...",
        flush=True,
    )
    initialization_started = time.perf_counter()
    controller = create_retrieval_controller(
        index_path=args.dense_index,
        release_dir=args.semantic_release,
        catalog_path=args.catalog,
        device=args.device,
        local_files_only=True,
        policy=FormalRetrievalPolicy(
            route_k=maximum_k,
            fusion_k=maximum_k,
            final_k=10,
            recall_strategy=RecallStrategy.LEGACY_SINGLE_CENTER,
        ),
        diversity_policy=VectorDiversityPolicy(),
    )
    initialization_seconds = time.perf_counter() - initialization_started

    score_cache: dict[str, Any] = {}
    records: list[dict[str, object]] = []
    parity_failures = 0
    started = time.perf_counter()
    for ordinal, case in enumerate(cases, start=1):
        record = _evaluate_case(
            case,
            controller=controller,
            k_values=k_values,
            maximum_k=maximum_k,
            score_cache=score_cache,
        )
        if not cast(dict[str, object], record["parity"])["single_k80_matches_saved_run"]:
            parity_failures += 1
        records.append(record)
        if ordinal % 10 == 0 or ordinal == len(cases):
            print(
                f"candidate sweep {ordinal}/{len(cases)} cached_queries={len(score_cache)}",
                flush=True,
            )

    if 80 in k_values and parity_failures:
        raise RuntimeError(f"single K=80 parity failed on {parity_failures} turns")

    report = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "turn_log": str(args.turn_log.resolve()),
            "catalog": str(args.catalog.resolve()),
            "semantic_release": str(args.semantic_release.resolve()),
            "dense_index": str(args.dense_index.resolve()),
        },
        "protocol": {
            "k_values": list(k_values),
            "shapes": list(SHAPES),
            "target_visible_to_retrieval": False,
            "turn_selection": "fresh searchable scored turns only",
            "representative_turn": "last fresh searchable scored turn per session",
            "rrf_rank_constant": 60,
            "atomic_factor_policy": (
                "positive goal and active preferences; one dense route per unique factor"
            ),
        },
        "runtime": {
            "device": args.device,
            "initialization_seconds": initialization_seconds,
            "evaluation_seconds": time.perf_counter() - started,
            "turn_count": len(records),
            "session_count": len({item.sample_id for item in cases}),
            "unique_dense_query_count": len(score_cache),
            "single_k80_parity_failure_count": parity_failures,
        },
        "summary": _summarize(records, k_values=k_values),
        "turns": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    print(args.output.resolve(), flush=True)
    print(args.markdown.resolve(), flush=True)
    return 0


def _evaluate_case(
    case: ReplayCase,
    *,
    controller: Any,
    k_values: tuple[int, ...],
    maximum_k: int,
    score_cache: dict[str, Any],
) -> dict[str, object]:
    mask = controller.hard_mask_resolver.resolve(case.compiled)
    composite = _dense_hits(
        case.compiled.q_sem,
        controller=controller,
        mask=mask.eligible_mask,
        top_k=maximum_k,
        score_cache=score_cache,
    )
    lexical_observation = controller.lexical_route.observe(
        case.compiled.q_lex,
        eligible_parent_asins=mask.eligible_parent_asins,
    )
    lexical = tuple(
        RouteHit(
            parent_asin=item.parent_asin,
            rank=item.rank,
            raw_score=item.raw_bm25,
        )
        for item in lexical_observation.hits
    )
    facet_observation = controller.facet_route.search(
        case.compiled,
        eligible_parent_asins=mask.eligible_parent_asins,
        relaxed_constraints=mask.relaxed_constraints,
        top_k=maximum_k,
    )
    facet = facet_observation.hits
    atomic = tuple(
        (
            factor,
            _dense_hits(
                factor,
                controller=controller,
                mask=mask.eligible_mask,
                top_k=maximum_k,
                score_cache=score_cache,
            ),
        )
        for factor in case.atomic_factors
    )

    variants: dict[str, object] = {}
    for k in k_values:
        shared = (
            NamedRoute("lexical", lexical[:k]),
            NamedRoute("facet", facet[:k]),
        )
        route_sets = {
            "single": (NamedRoute("dense:composite", composite[:k]), *shared),
            "atomic": (
                *(
                    NamedRoute(f"dense:factor:{index}", hits[:k])
                    for index, (_, hits) in enumerate(atomic)
                ),
                *shared,
            ),
            "hybrid": (
                NamedRoute("dense:composite", composite[:k]),
                *(
                    NamedRoute(f"dense:factor:{index}", hits[:k])
                    for index, (_, hits) in enumerate(atomic)
                ),
                *shared,
            ),
        }
        for shape, routes in route_sets.items():
            fused, relevance, union = _fuse(routes, top_k=k)
            target_rank = next(
                (
                    rank
                    for rank, parent_asin in enumerate(fused, start=1)
                    if parent_asin == case.target_parent_asin
                ),
                None,
            )
            variants[f"{shape}_k{k}"] = {
                "candidate_count": len(fused),
                "pre_fusion_union_count": len(union),
                "target_in_pre_fusion_union": case.target_parent_asin in union,
                "target_fused_rank": target_rank,
                "candidates": fused,
                "normalized_fusion_relevance": relevance,
            }

    observed_single_80 = cast(dict[str, object] | None, variants.get("single_k80"))
    observed_candidates = (
        ()
        if observed_single_80 is None
        else tuple(cast(list[str], observed_single_80["candidates"]))
    )
    return {
        "identity": case.identity,
        "sample_id": case.sample_id,
        "scenario_type": case.scenario_type,
        "turn": case.turn,
        "representative": case.representative,
        "target_parent_asin": case.target_parent_asin,
        "target_was_not_passed_to_retrieval": True,
        "q_sem": case.compiled.q_sem,
        "q_lex": case.compiled.q_lex,
        "atomic_factors": list(case.atomic_factors),
        "eligible_count": len(mask.eligible_parent_asins),
        "hard_filter_relaxed": mask.hard_filter_relaxed,
        "relevance_weight": case.relevance_weight,
        "route_counts_at_maximum_k": {
            "dense_composite": len(composite),
            "dense_atomic": [len(hits) for _, hits in atomic],
            "lexical": len(lexical),
            "facet": len(facet),
        },
        "parity": {
            "single_k80_matches_saved_run": (
                True if 80 not in k_values else observed_candidates == case.logged_fused_top_80
            )
        },
        "variants": variants,
    }


def _dense_hits(
    text: str,
    *,
    controller: Any,
    mask: Any,
    top_k: int,
    score_cache: dict[str, Any],
) -> tuple[RouteHit, ...]:
    key = " ".join(text.split())
    scores = score_cache.get(key)
    if scores is None:
        scores = controller.retriever.score(key)
        score_cache[key] = scores
    ranked = controller.retriever.index.rank_scores(
        scores,
        top_k=top_k,
        eligible_mask=mask,
    )
    return dense_route_observation(ranked).hits


def _fuse(
    routes: tuple[NamedRoute, ...],
    *,
    top_k: int,
) -> tuple[list[str], list[float], frozenset[str]]:
    scores: dict[str, float] = {}
    union: set[str] = set()
    names: set[str] = set()
    for route in routes:
        if route.name in names:
            raise ValueError(f"duplicate route name: {route.name}")
        names.add(route.name)
        for expected_rank, hit in enumerate(route.hits, start=1):
            if hit.rank != expected_rank:
                raise ValueError("route ranks must be contiguous")
            union.add(hit.parent_asin)
            scores[hit.parent_asin] = scores.get(hit.parent_asin, 0.0) + 1.0 / (60 + hit.rank)
    ordered = sorted(scores, key=lambda parent_asin: (-scores[parent_asin], parent_asin))
    selected = ordered[:top_k]
    maximum = scores[selected[0]] if selected else 1.0
    relevance = [float(scores[parent_asin] / maximum) for parent_asin in selected]
    return selected, relevance, frozenset(union)


def _load_cases(path: Path) -> tuple[ReplayCase, ...]:
    raw_rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    eligible: list[dict[str, object]] = []
    by_sample: dict[str, list[dict[str, object]]] = defaultdict(list)
    for value in raw_rows:
        if type(value) is not dict:
            raise ValueError("turn log must contain JSON objects")
        row = cast(dict[str, object], value)
        evaluator = _mapping(row, "evaluator")
        understanding = _mapping(row, "query_understanding")
        compiled = row.get("compiled_query")
        if (
            not bool(evaluator.get("override_applied_before_scoring"))
            or understanding.get("status") != "success"
            or type(compiled) is not dict
            or not bool(cast(dict[str, object], compiled).get("search_ready"))
            or type(row.get("retrieval")) is not dict
        ):
            continue
        eligible.append(row)
        by_sample[str(evaluator["sample_id"])].append(row)

    representative_identities = {
        f"{sample_id}/turn-{max(int(row['turn']) for row in rows)}"
        for sample_id, rows in by_sample.items()
    }
    cases = []
    for row in eligible:
        evaluator = _mapping(row, "evaluator")
        understanding = _mapping(row, "query_understanding")
        resolved = _mapping(understanding, "resolved_turn")
        final_intent = _mapping(resolved, "final_intent")
        compiled_payload = _mapping(row, "compiled_query")
        retrieval = _mapping(row, "retrieval")
        sample_id = str(evaluator["sample_id"])
        turn = int(row["turn"])
        identity = f"{sample_id}/turn-{turn}"
        cases.append(
            ReplayCase(
                identity=identity,
                sample_id=sample_id,
                scenario_type=str(evaluator["scenario_type"]),
                turn=turn,
                target_parent_asin=str(evaluator["target_parent_asin"]),
                compiled=_decode_compiled(compiled_payload),
                atomic_factors=_atomic_factors(final_intent),
                logged_fused_top_80=tuple(
                    str(item["parent_asin"])
                    for item in cast(list[dict[str, object]], retrieval["fused_candidates"])
                ),
                relevance_weight=float(retrieval["relevance_weight"]),
                representative=identity in representative_identities,
            )
        )
    return tuple(cases)


def _decode_compiled(raw: dict[str, object]) -> CompiledQuery:
    hard_constraints = tuple(
        CompiledHardConstraint(
            preference_id=_text(item, "preference_id"),
            facet=_text(item, "facet"),
            operator=Operator(_text(item, "operator")),
            value=_preference_value(item.get("value")),
            policy=ConstraintPolicy(_text(item, "policy")),
        )
        for item in _object_list(raw, "hard_constraints")
    )
    ranking_preferences = tuple(
        CompiledRankingPreference(
            preference_id=_text(item, "preference_id"),
            facet=_optional_text(item.get("facet")),
            operator=(None if item.get("operator") is None else Operator(str(item["operator"]))),
            value=(None if item.get("value") is None else _preference_value(item.get("value"))),
            semantic_text=_optional_text(item.get("semantic_text")),
            semantic_polarity=(
                None
                if item.get("semantic_polarity") is None
                else SemanticPolarity(str(item["semantic_polarity"]))
            ),
            commitment=Commitment(_text(item, "commitment")),
            source=PreferenceSource(_text(item, "source")),
            reason=RankingReason(_text(item, "reason")),
        )
        for item in _object_list(raw, "ranking_preferences")
    )
    directives = _mapping(raw, "directives")
    trace = tuple(
        PreferenceCompilationTrace(
            preference_id=_text(item, "preference_id"),
            targets=tuple(CompilationTarget(str(value)) for value in item.get("targets", [])),
            reason=_text(item, "reason"),
        )
        for item in _object_list(raw, "trace")
    )
    return CompiledQuery(
        schema=COMPILED_QUERY_SCHEMA,
        compiler_version=_text(raw, "compiler_version"),
        catalog_id=_text(raw, "catalog_id"),
        catalog_semantic_release_id=_text(raw, "catalog_semantic_release_id"),
        category_graph_id=_text(raw, "category_graph_id"),
        intent_version=int(raw["intent_version"]),
        q_lex=str(raw["q_lex"]),
        q_sem=str(raw["q_sem"]),
        search_ready=bool(raw["search_ready"]),
        hard_constraints=hard_constraints,
        ranking_preferences=ranking_preferences,
        dont_care_facets=tuple(str(item) for item in raw.get("dont_care_facets", [])),
        directives=CompiledDirectives(
            diversity=DiversityDirective(_text(directives, "diversity")),
            comparison_requested=bool(directives["comparison_requested"]),
            explanation_requested=bool(directives["explanation_requested"]),
        ),
        requires_clarification=bool(raw["requires_clarification"]),
        clarification_reason=_optional_text(raw.get("clarification_reason")),
        trace=trace,
    )


def _atomic_factors(intent: dict[str, object]) -> tuple[str, ...]:
    factors: list[str] = []
    goal = intent.get("goal")
    if type(goal) is str and goal.strip():
        factors.append(f"Product goal: {goal.strip()}")
    for preference in _object_list(intent, "preferences"):
        operator = preference.get("operator")
        polarity = preference.get("semantic_polarity")
        if operator in {Operator.NEQ.value, Operator.NOT_IN.value} or (
            polarity == SemanticPolarity.NEGATIVE.value
        ):
            continue
        semantic_text = preference.get("semantic_text")
        evidence_text = preference.get("evidence_text")
        if type(semantic_text) is str and semantic_text.strip():
            text = semantic_text.strip()
        elif type(evidence_text) is str and evidence_text.strip():
            facet = preference.get("facet")
            prefix = (
                "Preference"
                if facet is None
                else str(facet).replace("system_product_category", "category").replace("_", " ")
            )
            text = f"{prefix}: {evidence_text.strip()}"
        else:
            value = preference.get("value")
            if value is None:
                continue
            facet = str(preference.get("facet") or "preference").replace("_", " ")
            rendered = (
                " or ".join(str(item) for item in value) if type(value) is list else str(value)
            )
            text = f"{facet}: {rendered}"
        factors.append(text)
    unique: list[str] = []
    seen: set[str] = set()
    for factor in factors:
        key = " ".join(factor.split()).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(" ".join(factor.split()))
    return tuple(unique)


def _summarize(
    records: list[dict[str, object]],
    *,
    k_values: tuple[int, ...],
) -> dict[str, object]:
    variant_names = [f"{shape}_k{k}" for shape in SHAPES for k in k_values]
    representative = [item for item in records if bool(item["representative"])]
    by_sample: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_sample[str(record["sample_id"])].append(record)
    return {
        "all_turns": {name: _row_summary(records, name) for name in variant_names},
        "representative_turns": {
            name: _row_summary(representative, name) for name in variant_names
        },
        "session_any_observed_turn": {
            name: _session_summary(by_sample, name) for name in variant_names
        },
    }


def _row_summary(rows: list[dict[str, object]], variant: str) -> dict[str, object]:
    ranks = []
    union_hits = 0
    for row in rows:
        item = cast(dict[str, object], cast(dict[str, object], row["variants"])[variant])
        rank = item["target_fused_rank"]
        if rank is not None:
            ranks.append(int(rank))
        union_hits += bool(item["target_in_pre_fusion_union"])
    count = len(rows)
    return {
        "count": count,
        "pre_fusion_union_recall": union_hits / count,
        "fused_candidate_recall": len(ranks) / count,
        "mean_reciprocal_candidate_rank": (
            0.0 if not ranks else sum(1.0 / rank for rank in ranks) / count
        ),
        "median_target_rank_when_present": (None if not ranks else _median(ranks)),
    }


def _session_summary(
    by_sample: dict[str, list[dict[str, object]]],
    variant: str,
) -> dict[str, object]:
    ranks = []
    union_hits = 0
    scenarios: dict[str, list[bool]] = defaultdict(list)
    for rows in by_sample.values():
        observed = [
            cast(dict[str, object], cast(dict[str, object], row["variants"])[variant])
            for row in rows
        ]
        present_ranks = [
            int(item["target_fused_rank"])
            for item in observed
            if item["target_fused_rank"] is not None
        ]
        hit = bool(present_ranks)
        ranks.append(min(present_ranks) if present_ranks else None)
        union_hits += any(bool(item["target_in_pre_fusion_union"]) for item in observed)
        scenarios[str(rows[0]["scenario_type"])].append(hit)
    count = len(ranks)
    return {
        "session_count": count,
        "pre_fusion_union_recall": union_hits / count,
        "fused_candidate_recall": sum(item is not None for item in ranks) / count,
        "scenario_fused_candidate_recall": {
            name: sum(values) / len(values) for name, values in sorted(scenarios.items())
        },
    }


def _render_markdown(report: dict[str, object]) -> str:
    summary = _mapping(report, "summary")
    session = _mapping(summary, "session_any_observed_turn")
    representative = _mapping(summary, "representative_turns")
    protocol = _mapping(report, "protocol")
    lines = [
        "# Candidate recall sweep v0",
        "",
        "This is a target-blind replay of saved real QU states. It does not call DeepSeek and does not change the runtime architecture.",
        "",
        "| shape | K | session-any fused recall | representative fused recall | representative union recall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for shape in SHAPES:
        for k in cast(list[int], protocol["k_values"]):
            name = f"{shape}_k{k}"
            session_item = cast(dict[str, object], session[name])
            representative_item = cast(dict[str, object], representative[name])
            lines.append(
                f"| {shape} | {k} | "
                f"{float(session_item['fused_candidate_recall']):.3f} | "
                f"{float(representative_item['fused_candidate_recall']):.3f} | "
                f"{float(representative_item['pre_fusion_union_recall']):.3f} |"
            )
    lines.extend(
        [
            "",
            "`single K=80` is checked for exact candidate-list parity with the saved formal run.",
            "Candidate recall is not Top-10 recommendation quality; cross-encoder and DPP evaluation follows after choosing a pool policy.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--turn-log",
        type=Path,
        default=ROOT / "artifacts/simulator/full-pipeline-other-200-v0/turns.jsonl",
    )
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.jsonl")
    parser.add_argument(
        "--semantic-release",
        type=Path,
        default=ROOT / "artifacts/catalog-semantic/release-v0",
    )
    parser.add_argument(
        "--dense-index",
        type=Path,
        default=ROOT / "artifacts/retrieval/dense-v0",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k", type=int, action="append", default=list(DEFAULT_K_VALUES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/retrieval/candidate-recall-sweep-v0.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=ROOT / "artifacts/retrieval/candidate-recall-sweep-v0.md",
    )
    args = parser.parse_args()
    if any(value < 1 for value in args.k):
        parser.error("--k values must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def _preference_value(value: object) -> Any:
    if type(value) is list:
        return tuple(cast(list[object], value))
    if type(value) not in (str, int, float, bool):
        raise ValueError("compiled preference value has an invalid shape")
    return value


def _object_list(mapping: dict[str, object], key: str) -> list[dict[str, object]]:
    value = mapping.get(key)
    if type(value) is not list or any(type(item) is not dict for item in value):
        raise ValueError(f"{key} must be a list of objects")
    return cast(list[dict[str, object]], value)


def _mapping(mapping: dict[str, object], key: str) -> dict[str, object]:
    value = mapping.get(key)
    if type(value) is not dict:
        raise ValueError(f"{key} must be an object")
    return cast(dict[str, object], value)


def _text(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)
    if type(value) is not str:
        raise ValueError(f"{key} must be a string")
    return value


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError("optional text must be a string or null")
    return value


def _median(values: list[int]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


if __name__ == "__main__":
    raise SystemExit(main())
