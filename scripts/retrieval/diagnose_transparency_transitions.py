"""Locate where C_t convergence first fails across saved QU-to-Probe transitions."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from shopping_copilot.catalog.semantic.release import (  # noqa: E402
    load_catalog_semantic_release,
)
from shopping_copilot.query_compiler import (  # noqa: E402
    COMPILED_QUERY_SCHEMA,
    QUERY_COMPILER_VERSION,
    CompiledDirectives,
    CompiledHardConstraint,
    CompiledQuery,
    ConstraintPolicy,
    DiversityDirective,
)
from shopping_copilot.retrieval import (  # noqa: E402
    HardMaskResolver,
    build_retrieval_evidence_index,
    create_compiled_probe_runner,
    load_bound_transparency_calibration,
)
from shopping_copilot.session_context import Operator  # noqa: E402

DEFAULT_INPUT = Path("artifacts/retrieval/qu-to-probe-simulator-other-16x4-v1.json")
DEFAULT_RELEASE = Path("artifacts/catalog-semantic/release-v0")
DEFAULT_DENSE_INDEX = Path("artifacts/retrieval/dense-v0")
DEFAULT_CALIBRATION = Path("config/retrieval/transparency-calibration-v1.json")


def main() -> int:
    args = _parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    calibration = load_bound_transparency_calibration(args.calibration)
    release = load_catalog_semantic_release(args.release)
    probe_runner = create_compiled_probe_runner(
        index_path=args.dense_index,
        release_dir=args.release,
        calibration=calibration,
        catalog_path=args.catalog,
        device=args.device,
        local_files_only=True,
        probe_k=calibration.probe_k,
        mode_threshold=calibration.mode_similarity_threshold,
    )
    catalog_source = args.catalog or args.release / "catalog.jsonl"
    evidence = build_retrieval_evidence_index(
        catalog_source,
        catalog_id=release.manifest.catalog_id,
        catalog_semantic_release_id=release.release_id,
        expected_parent_asins=set(probe_runner.dense_index.parent_asins),
    )
    resolver = HardMaskResolver(
        release=release,
        evidence_index=evidence,
        dense_index=probe_runner.dense_index,
    )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in source["turns"]:
        if item["status"] == "success":
            grouped[item["conversation_id"]].append(item)

    rows: list[dict[str, Any]] = []
    for conversation_id, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: item["turn"])
        if len(ordered) < 2:
            continue
        queries = [
            _restore_query(
                item["compiled"],
                runtime=source["runtime"],
                category_graph_id=release.category_registry.category_graph_id,
            )
            for item in ordered
        ]
        masks = [resolver.resolve(query) for query in queries]
        baseline = [
            probe_runner.run(
                query,
                eligible_mask=mask.eligible_mask,
                hard_filter_relaxed=mask.hard_filter_relaxed,
            )
            for query, mask in zip(queries, masks, strict=True)
        ]

        for index in range(1, len(ordered)):
            previous_item = ordered[index - 1]
            current_item = ordered[index]
            previous_query = queries[index - 1]
            current_query = queries[index]
            previous_mask = masks[index - 1]
            current_mask = masks[index]
            previous_run = baseline[index - 1]
            current_run = baseline[index]
            query_only = probe_runner.run(
                current_query,
                eligible_mask=previous_mask.eligible_mask,
                hard_filter_relaxed=previous_mask.hard_filter_relaxed,
            )
            mask_only_query = replace(
                previous_query,
                intent_version=current_query.intent_version,
                hard_constraints=current_query.hard_constraints,
            )
            mask_only = probe_runner.run(
                mask_only_query,
                eligible_mask=current_mask.eligible_mask,
                hard_filter_relaxed=current_mask.hard_filter_relaxed,
            )
            rows.append(
                _transition_payload(
                    conversation_id=conversation_id,
                    scenario=str(current_item.get("scenario_type")),
                    response_shape=str(current_item.get("response_shape")),
                    from_turn=int(previous_item["turn"]),
                    to_turn=int(current_item["turn"]),
                    previous=previous_run,
                    query_only=query_only,
                    mask_only=mask_only,
                    current=current_run,
                    previous_eligible=len(previous_mask.eligible_parent_asins),
                    current_eligible=len(current_mask.eligible_parent_asins),
                )
            )

    report = {
        "schema": "shopping-copilot/transparency-transition-diagnosis/v1",
        "source": str(args.input),
        "method": {
            "previous": "previous query + previous hard mask",
            "query_only": "current query + previous hard mask",
            "mask_only": "previous query + current hard mask",
            "current": "current query + current hard mask",
        },
        "summary": _summarize(rows),
        "transitions": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(_markdown(report), encoding="utf-8", newline="\n")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")
    print(f"wrote {markdown_path}")
    return 0


def _transition_payload(
    *,
    conversation_id: str,
    scenario: str,
    response_shape: str,
    from_turn: int,
    to_turn: int,
    previous: Any,
    query_only: Any,
    mask_only: Any,
    current: Any,
    previous_eligible: int,
    current_eligible: int,
) -> dict[str, Any]:
    previous_metrics = _run_metrics(previous)
    query_metrics = _run_metrics(query_only)
    mask_metrics = _run_metrics(mask_only)
    current_metrics = _run_metrics(current)
    previous_g = previous_metrics["mode_coherence"]
    current_g = current_metrics["mode_coherence"]
    full_delta = _delta(previous_g, current_g)
    query_delta = _delta(previous_g, query_metrics["mode_coherence"])
    mask_delta = _delta(previous_g, mask_metrics["mode_coherence"])
    listing_delta = _delta(
        previous_metrics["listing_coherence"],
        current_metrics["listing_coherence"],
    )
    return {
        "conversation_id": conversation_id,
        "scenario": scenario,
        "response_shape": response_shape,
        "from_turn": from_turn,
        "to_turn": to_turn,
        "previous_eligible": previous_eligible,
        "current_eligible": current_eligible,
        "eligible_ratio": (
            current_eligible / previous_eligible if previous_eligible > 0 else None
        ),
        "top_k_jaccard": _jaccard(previous.ranking.hits, current.ranking.hits),
        "previous": previous_metrics,
        "query_only": query_metrics,
        "mask_only": mask_metrics,
        "current": current_metrics,
        "deltas": {
            "full_mode": full_delta,
            "query_only_mode": query_delta,
            "mask_only_mode": mask_delta,
            "full_listing": listing_delta,
        },
        "signals": {
            "full_down": _negative(full_delta),
            "query_alone_down": _negative(query_delta),
            "mask_alone_down": _negative(mask_delta),
            "mode_reducer_reversal": _negative(full_delta)
            and listing_delta is not None
            and listing_delta >= 0.0,
            "low_top_k_overlap": _jaccard(previous.ranking.hits, current.ranking.hits) < 0.25,
        },
    }


def _run_metrics(run: Any) -> dict[str, Any]:
    semantic = run.snapshot.semantic
    return {
        "certainty": run.estimate.certainty,
        "mode_coherence": semantic.equal_mode_coherence.debiased_pairwise_cosine,
        "listing_coherence": semantic.raw_listing_coherence.debiased_pairwise_cosine,
        "mode_count": len(semantic.modes),
        "effective_mode_count": semantic.effective_mode_count,
        "largest_mode_share": semantic.largest_mode_share,
        "dense_count": len(semantic.hits),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    disclosures = [item for item in rows if item["response_shape"] == "attribute_disclosure"]
    comparable = [item for item in disclosures if item["deltas"]["full_mode"] is not None]
    down = [item for item in comparable if item["signals"]["full_down"]]
    return {
        "transition_count": len(rows),
        "attribute_disclosure_count": len(disclosures),
        "comparable_attribute_disclosure_count": len(comparable),
        "full_mode_direction": _directions(comparable, "full_mode"),
        "full_mode_mean_delta": _mean_delta(comparable, "full_mode"),
        "down_transition_diagnosis": {
            "count": len(down),
            "query_alone_down": sum(item["signals"]["query_alone_down"] for item in down),
            "mask_alone_down": sum(item["signals"]["mask_alone_down"] for item in down),
            "both_alone_down": sum(
                item["signals"]["query_alone_down"] and item["signals"]["mask_alone_down"]
                for item in down
            ),
            "neither_alone_down": sum(
                not item["signals"]["query_alone_down"]
                and not item["signals"]["mask_alone_down"]
                for item in down
            ),
            "mode_reducer_reversal": sum(
                item["signals"]["mode_reducer_reversal"] for item in down
            ),
            "low_top_k_overlap": sum(item["signals"]["low_top_k_overlap"] for item in down),
            "mean_top_k_jaccard": (
                statistics.fmean(item["top_k_jaccard"] for item in down) if down else None
            ),
            "mean_eligible_ratio": statistics.fmean(
                item["eligible_ratio"]
                for item in down
                if item["eligible_ratio"] is not None
            )
            if down
            else None,
        },
        "by_disclosure_turn": {
            str(turn): {
                "count": len(selected),
                "direction": _directions(selected, "full_mode"),
                "mean_delta": _mean_delta(selected, "full_mode"),
            }
            for turn in (2, 3)
            if (selected := [item for item in comparable if item["to_turn"] == turn])
        },
    }


def _directions(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    values = [item["deltas"][field] for item in rows if item["deltas"][field] is not None]
    return dict(
        Counter(
            "up" if value > 1e-12 else "down" if value < -1e-12 else "flat"
            for value in values
        )
    )


def _mean_delta(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [item["deltas"][field] for item in rows if item["deltas"][field] is not None]
    return statistics.fmean(values) if values else None


def _restore_query(
    payload: dict[str, Any],
    *,
    runtime: dict[str, Any],
    category_graph_id: str,
) -> CompiledQuery:
    constraints = []
    for item in payload["hard_constraints"]:
        value = item["value"]
        constraints.append(
            CompiledHardConstraint(
                preference_id=item["preference_id"],
                facet=item["facet"],
                operator=Operator(item["operator"]),
                value=tuple(value) if type(value) is list else value,
                policy=ConstraintPolicy(item["policy"]),
            )
        )
    return CompiledQuery(
        schema=COMPILED_QUERY_SCHEMA,
        compiler_version=QUERY_COMPILER_VERSION,
        catalog_id=runtime["catalog_id"],
        catalog_semantic_release_id=runtime["catalog_semantic_release_id"],
        category_graph_id=category_graph_id,
        intent_version=payload["intent_version"],
        q_lex=payload["q_lex"],
        q_sem=payload["q_sem"],
        search_ready=payload["search_ready"],
        hard_constraints=tuple(constraints),
        ranking_preferences=(),
        dont_care_facets=tuple(payload["dont_care_facets"]),
        directives=CompiledDirectives(
            diversity=DiversityDirective.AUTO,
            comparison_requested=False,
            explanation_requested=False,
        ),
        requires_clarification=False,
        clarification_reason=None,
        trace=(),
    )


def _delta(previous: float | None, current: float | None) -> float | None:
    if previous is None or current is None:
        return None
    return current - previous


def _negative(value: float | None) -> bool:
    return value is not None and value < -1e-12


def _jaccard(previous_hits: Any, current_hits: Any) -> float:
    previous = {item.parent_asin for item in previous_hits}
    current = {item.parent_asin for item in current_hits}
    union = previous | current
    return len(previous & current) / len(union) if union else 1.0


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    diagnosis = summary["down_transition_diagnosis"]
    lines = [
        "# C_t transition diagnosis",
        "",
        "The same saved QU states were replayed through a 2×2 counterfactual matrix.",
        "",
        "## Summary",
        "",
        f"- Comparable attribute disclosures: {summary['comparable_attribute_disclosure_count']}",
        f"- Full direction: `{summary['full_mode_direction']}`",
        f"- Full mean G_mode delta: `{summary['full_mode_mean_delta']:.6f}`",
        f"- Down transitions: {diagnosis['count']}",
        f"- Query alone also down: {diagnosis['query_alone_down']}",
        f"- Mask alone also down: {diagnosis['mask_alone_down']}",
        f"- Both alone down: {diagnosis['both_alone_down']}",
        f"- Neither alone down: {diagnosis['neither_alone_down']}",
        f"- Mode reducer reversed a non-decreasing listing signal: {diagnosis['mode_reducer_reversal']}",
        f"- Low Top-K overlap (<0.25): {diagnosis['low_top_k_overlap']}",
        f"- Mean Top-K Jaccard among down transitions: `{diagnosis['mean_top_k_jaccard']:.6f}`",
        f"- Mean eligible ratio among down transitions: `{diagnosis['mean_eligible_ratio']:.6f}`",
        "",
        "## Every attribute-disclosure transition",
        "",
        "| Task / transition | scenario | Δ full | Δ query-only | Δ mask-only | Δ listing | Top-K Jaccard | eligible ratio |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["transitions"]:
        if item["response_shape"] != "attribute_disclosure":
            continue
        delta = item["deltas"]
        lines.append(
            f"| `{item['conversation_id']} {item['from_turn']}→{item['to_turn']}` | "
            f"{item['scenario']} | {_fmt(delta['full_mode'])} | "
            f"{_fmt(delta['query_only_mode'])} | {_fmt(delta['mask_only_mode'])} | "
            f"{_fmt(delta['full_listing'])} | {_fmt(item['top_k_jaccard'])} | "
            f"{_fmt(item['eligible_ratio'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:+.4f}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/retrieval/transparency-transition-diagnosis-v1.json"),
    )
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--dense-index", type=Path, default=DEFAULT_DENSE_INDEX)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
