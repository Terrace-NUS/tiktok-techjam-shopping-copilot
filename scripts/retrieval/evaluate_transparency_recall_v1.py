"""Compare legacy Top-K recall with transparency-aware multi-center recall."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for source in (ROOT, SRC):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from scripts.retrieval.evaluate_multi_route_v0 import (  # noqa: E402
    CASES,
    _compile_case,
    _load_metadata,
)
from shopping_copilot.catalog.semantic.release import (  # noqa: E402
    load_catalog_semantic_release,
)
from shopping_copilot.retrieval import (  # noqa: E402
    FormalRetrievalPolicy,
    RecallStrategy,
    RetrievalController,
    create_retrieval_controller,
)


def main() -> int:
    args = _parse_args()
    initialization_started = time.perf_counter()
    controller = create_retrieval_controller(
        index_path=args.dense_index,
        release_dir=args.semantic_release,
        catalog_path=args.catalog,
        device=args.device,
        local_files_only=True,
        policy=FormalRetrievalPolicy(),
    )
    initialization_ms = _elapsed_ms(initialization_started)
    legacy = RetrievalController(
        retriever=controller.retriever,
        lexical_route=controller.lexical_route,
        facet_route=controller.facet_route,
        hard_mask_resolver=controller.hard_mask_resolver,
        policy=FormalRetrievalPolicy(
            fusion_k=80,
            recall_strategy=RecallStrategy.LEGACY_SINGLE_CENTER,
        ),
        diversity_policy=controller.diversity_policy,
    )
    release = load_catalog_semantic_release(args.semantic_release)
    metadata = _load_metadata(args.catalog)

    cases: list[dict[str, Any]] = []
    for case in CASES:
        query = _compile_case(case, release)
        modern = controller.search(query, transparency=case.transparency_anchor)
        baseline = legacy.search(query, transparency=case.transparency_anchor)
        cases.append(
            {
                "case_id": case.case_id,
                "intent_shape": case.intent_shape,
                "q_lex": case.q_lex,
                "q_sem": case.q_sem,
                "transparency": case.transparency_anchor,
                "legacy": _result_log(baseline, controller, metadata),
                "multi_center": _result_log(modern, controller, metadata),
            }
        )

    broad_case = CASES[0]
    broad_query = _compile_case(broad_case, release)
    causal_sweep = []
    for transparency in (0.0, 0.25, 0.5, 0.75, 1.0):
        result = controller.search(broad_query, transparency=transparency)
        causal_sweep.append(
            {
                "transparency": transparency,
                **_result_log(result, controller, metadata),
            }
        )

    payload = {
        "schema": "shopping-copilot/transparency-recall-evaluation/v1",
        "purpose": "candidate_recall_and_warm_latency_comparison",
        "initialization_ms": initialization_ms,
        "bindings": {
            "catalog_id": controller.retriever.index.manifest.catalog_id,
            "release_id": controller.retriever.index.manifest.catalog_semantic_release_id,
            "dense_index_id": controller.retriever.index.index_id,
            "product_count": controller.retriever.index.manifest.product_count,
            "embedding_model": controller.retriever.index.manifest.embedding.model_id,
            "device": args.device,
        },
        "cases": cases,
        "causal_sweep": {
            "case_id": broad_case.case_id,
            "note": "The compiled query is fixed; only T changes.",
            "runs": causal_sweep,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(_markdown_ascii(payload), encoding="utf-8")
    print(args.output_json)
    print(args.output_markdown)
    return 0


def _result_log(
    result: Any,
    controller: RetrievalController,
    metadata: dict[str, dict[str, object]],
) -> dict[str, Any]:
    candidate_ids = tuple(item.parent_asin for item in result.fused_candidates)
    rows = [controller.retriever.index.row_index(item) for item in candidate_ids]
    vectors = controller.retriever.index.vectors[rows]
    pairwise = _mean_off_diagonal(vectors @ vectors.T)
    dense_route = next(item for item in result.routes if item.route.value == "dense")
    trace = result.recall_trace
    if trace is None:
        dense_query_similarities = [item.raw_score for item in dense_route.hits]
        directions: list[dict[str, object]] = []
        requested_directions = 1
        actual_directions = 1 if dense_route.hits else 0
        direction_counts: dict[str, int] = {}
    else:
        dense_query_similarities = [item.query_similarity for item in trace.dense_candidates]
        directions = [
            {
                **asdict(item),
                "title": metadata[item.center_parent_asin]["title"],
                "reporting_group": metadata[item.center_parent_asin]["reporting_group"],
                "leaf_category": metadata[item.center_parent_asin]["leaf_category"],
            }
            for item in trace.directions
        ]
        requested_directions = trace.requested_direction_count
        actual_directions = trace.actual_direction_count
        direction_counts = {}
        for candidate in trace.dense_candidates:
            direction_counts[candidate.direction_id] = (
                direction_counts.get(candidate.direction_id, 0) + 1
            )
    reporting_groups = {str(metadata[item]["reporting_group"]) for item in candidate_ids}
    leaf_categories = {str(metadata[item]["leaf_category"]) for item in candidate_ids}
    return {
        "candidate_count": len(candidate_ids),
        "mean_pairwise_candidate_cosine": pairwise,
        "unique_reporting_groups": len(reporting_groups),
        "unique_leaf_categories": len(leaf_categories),
        "mean_dense_query_similarity": _mean(dense_query_similarities),
        "minimum_dense_query_similarity": (
            None if not dense_query_similarities else min(dense_query_similarities)
        ),
        "requested_directions": requested_directions,
        "actual_directions": actual_directions,
        "selected_dense_count_by_direction": direction_counts,
        "directions": directions,
        "route_counts": {item.route.value: len(item.hits) for item in result.routes},
        "timings_ms": asdict(result.timings),
    }


def _mean_off_diagonal(matrix: np.ndarray) -> float | None:
    count = matrix.shape[0]
    if count < 2:
        return None
    return float((float(matrix.sum()) - float(np.trace(matrix))) / (count * (count - 1)))


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Transparency-aware Multi-center Recall v1",
        "",
        f"一次性初始化耗时：`{payload['initialization_ms']:.1f} ms`。表中为初始化后的单次查询耗时。",
        "",
        "| Case | T | Policy | Pool | Directions | Pair cosine ↓ | Groups ↑ | Mean q-sim ↑ | Retrieval ms | Recall planning ms |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in payload["cases"]:
        for key, label in (("legacy", "Legacy"), ("multi_center", "Multi-center")):
            item = case[key]
            lines.append(
                f"| {case['case_id']} | {case['transparency']:.2f} | {label} | "
                f"{item['candidate_count']} | {item['actual_directions']} | "
                f"{_number(item['mean_pairwise_candidate_cosine'])} | "
                f"{item['unique_reporting_groups']} | "
                f"{_number(item['mean_dense_query_similarity'])} | "
                f"{item['timings_ms']['total_ms']:.1f} | "
                f"{item['timings_ms']['recall_planning_ms']:.1f} |"
            )
    lines.extend(
        [
            "",
            "## 固定查询，只改变 T",
            "",
            "| T | Directions | Pair cosine ↓ | Groups ↑ | Mean q-sim ↑ | Retrieval ms |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in payload["causal_sweep"]["runs"]:
        lines.append(
            f"| {item['transparency']:.2f} | {item['actual_directions']} | "
            f"{_number(item['mean_pairwise_candidate_cosine'])} | "
            f"{item['unique_reporting_groups']} | "
            f"{_number(item['mean_dense_query_similarity'])} | "
            f"{item['timings_ms']['total_ms']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _elapsed_ms(started: float) -> float:
    return float((time.perf_counter() - started) * 1_000.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dense-index",
        type=Path,
        default=ROOT / "artifacts/retrieval/dense-v0",
    )
    parser.add_argument(
        "--semantic-release",
        type=Path,
        default=ROOT / "artifacts/catalog-semantic/release-v0",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=ROOT / "artifacts/catalog-semantic/release-v0/catalog.jsonl",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "artifacts/retrieval/transparency-recall-evaluation-v1.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=ROOT / "artifacts/retrieval/transparency-recall-evaluation-v1.md",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _markdown_ascii(payload: dict[str, Any]) -> str:
    lines = [
        "# Transparency-aware Multi-center Recall v1",
        "",
        f"One-time initialization: `{payload['initialization_ms']:.1f} ms`. "
        "The table reports warm per-query latency after initialization.",
        "",
        "| Case | T | Policy | Pool | Directions | Pair cosine (lower is broader) | Groups | Mean q-sim | Total ms | Recall planning ms |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in payload["cases"]:
        for key, label in (("legacy", "Legacy"), ("multi_center", "Multi-center")):
            item = case[key]
            lines.append(
                f"| {case['case_id']} | {case['transparency']:.2f} | {label} | "
                f"{item['candidate_count']} | {item['actual_directions']} | "
                f"{_number(item['mean_pairwise_candidate_cosine'])} | "
                f"{item['unique_reporting_groups']} | "
                f"{_number(item['mean_dense_query_similarity'])} | "
                f"{item['timings_ms']['total_ms']:.1f} | "
                f"{item['timings_ms']['recall_planning_ms']:.1f} |"
            )
    lines.extend(
        [
            "",
            "## Fixed query; only T changes",
            "",
            "| T | Directions | Pair cosine | Groups | Mean q-sim | Total ms |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in payload["causal_sweep"]["runs"]:
        lines.append(
            f"| {item['transparency']:.2f} | {item['actual_directions']} | "
            f"{_number(item['mean_pairwise_candidate_cosine'])} | "
            f"{item['unique_reporting_groups']} | "
            f"{_number(item['mean_dense_query_similarity'])} | "
            f"{item['timings_ms']['total_ms']:.1f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
