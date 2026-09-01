"""Analyze query-to-catalog cosine thresholds over saved QU turns.

This is an offline experiment. It reuses committed ``q_sem`` values and hard
masks from a saved QU-to-Probe run, scores the complete dense catalog on CUDA
when available, and reports how absolute and query-relative thresholds change
the supported candidate population. It does not mutate the catalog or any
runtime index.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from shopping_copilot.catalog.semantic.release import (
    VerifiedCatalogSemanticRelease,
    load_catalog_semantic_release,
)
from shopping_copilot.query_compiler import (
    COMPILED_QUERY_SCHEMA,
    QUERY_COMPILER_VERSION,
    CompiledDirectives,
    CompiledHardConstraint,
    CompiledQuery,
    ConstraintPolicy,
    DiversityDirective,
)
from shopping_copilot.retrieval import (
    HardMaskResolver,
    SentenceTransformerTextEmbedder,
    build_retrieval_evidence_index,
    load_dense_index,
)
from shopping_copilot.retrieval.coherence import (
    compute_catalog_mean,
    compute_probe_coherence,
)
from shopping_copilot.retrieval.dense import DenseIndex
from shopping_copilot.session_context import Operator

REPORT_SCHEMA = "shopping-copilot/semantic-threshold-analysis/v0"
DEFAULT_EVALUATION = Path("artifacts/retrieval/qu-to-probe-full-v1.json")
DEFAULT_DENSE_INDEX = Path("artifacts/retrieval/dense-v0")
DEFAULT_RELEASE = Path("artifacts/catalog-semantic/release-v0")
DEFAULT_OUTPUT = Path("artifacts/retrieval/semantic-threshold-analysis-v0.json")
DEFAULT_MARKDOWN = Path("artifacts/retrieval/semantic-threshold-analysis-v0.md")

ABSOLUTE_THRESHOLDS = tuple(round(0.20 + index * 0.025, 3) for index in range(27))
TOP5_DELTAS = tuple(round(0.025 + index * 0.025, 3) for index in range(14))
TURN_QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.999, 1.0)
PAIRWISE_KEEP_DELTA = 0.1
PAIRWISE_SAMPLE_MAX = 1_536
PAIR_SAMPLE_MAX = 100_000
PAIRWISE_QUANTILES = (0.5, 0.9, 0.95, 0.99, 0.999, 1.0)
MERGE_THRESHOLDS = (0.85, 0.875, 0.9, 0.925, 0.94, 0.95, 0.96, 0.975, 0.99)
FULL_MERGE_KEEP_DELTAS = (0.075, 0.1)
FULL_MERGE_THRESHOLDS = (0.875, 0.9, 0.925, 0.94, 0.95)
METRIC_MERGE_THRESHOLD = 0.94
KERNEL_TEMPERATURES = (0.025, 0.05, 0.075, 0.1, 0.15, 0.2)
LOGDET_BETAS = (10.0, 50.0, 100.0, 500.0)
KNN_VALUES = (3, 5, 10)
METRIC_PAIR_SAMPLE_MAX = 100_000
EXPECTED_TRANSITION_TAGS = (
    "expected_narrower",
    "expected_broader",
    "expected_stable",
    "expected_override",
)
STABLE_RELATIVE_TOLERANCE = 0.10


@dataclass(frozen=True, slots=True)
class SavedTurn:
    suite_id: str
    cohort: str
    conversation_id: str
    turn: int
    scenario_type: str
    tags: tuple[str, ...]
    user_message: str
    q_sem: str
    intent_version: int
    preference_count: int
    hard_constraint_count: int
    reported_eligible_count: int
    compiled: dict[str, object]

    @property
    def identity(self) -> str:
        return f"{self.suite_id}/{self.conversation_id}/turn-{self.turn}"


def main() -> int:
    args = _parse_args()
    started = time.perf_counter()
    payload = _load_json(args.evaluation)
    turns = _load_turns(payload, cohort=args.cohort, max_turn=args.max_turn)
    if args.limit is not None:
        turns = turns[: args.limit]
    if not turns:
        raise ValueError("the selected evaluation contains no successful searchable turns")

    release = load_catalog_semantic_release(args.release)
    dense = load_dense_index(
        args.dense_index,
        expected_catalog_id=release.manifest.catalog_id,
        expected_release_id=release.release_id,
    )
    device = _resolve_device(args.device)
    query_vectors, encode_seconds = _encode_queries(
        turns,
        dense=dense,
        device=device,
    )
    if args.ignore_hard_mask:
        eligible_masks = [
            np.ones(dense.manifest.product_count, dtype=np.bool_) for _ in turns
        ]
        mask_seconds = 0.0
    else:
        eligible_masks, mask_seconds = _resolve_masks(
            turns,
            release=release,
            dense=dense,
            release_dir=args.release,
        )
    score_rows, score_seconds = _score_catalog(
        query_vectors,
        dense=dense,
        device=device,
    )
    pairwise_rows, pairwise_seconds = _sample_pairwise_geometry(
        turns,
        score_rows=score_rows,
        eligible_masks=eligible_masks,
        dense=dense,
        device=device,
    )
    merge_evaluation: dict[str, object] | None = None
    merge_seconds = 0.0
    if args.run_merge:
        merge_evaluation, merge_seconds = _evaluate_full_merging(
            turns,
            score_rows=score_rows,
            eligible_masks=eligible_masks,
            dense=dense,
            device=device,
            max_candidates=args.merge_max_candidates,
            run_metric_suite=args.run_metrics,
        )

    turn_rows = [
        _analyze_turn(
            turn,
            scores=score_rows[index],
            eligible=eligible_masks[index],
            pairwise=pairwise_rows[index],
            validate_reported_eligible=not args.ignore_hard_mask,
        )
        for index, turn in enumerate(turns)
    ]
    report = {
        "schema": REPORT_SCHEMA,
        "inputs": {
            "evaluation": str(args.evaluation.resolve()),
            "dense_index": str(args.dense_index.resolve()),
            "semantic_release": str(args.release.resolve()),
            "source_evaluation_schema": payload.get("schema"),
            "max_turn": args.max_turn,
            "hard_mask_mode": "ignored" if args.ignore_hard_mask else "resolved",
            "dense_index_id": dense.index_id,
            "catalog_id": dense.manifest.catalog_id,
            "catalog_semantic_release_id": dense.manifest.catalog_semantic_release_id,
        },
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "query_count": len(turns),
            "product_count": dense.manifest.product_count,
            "embedding_dimension": dense.manifest.embedding.dimension,
            "query_encode_seconds": encode_seconds,
            "hard_mask_seconds": mask_seconds,
            "full_catalog_score_seconds": score_seconds,
            "pairwise_sample_seconds": pairwise_seconds,
            "full_merge_seconds": merge_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "thresholds": {
            "absolute": list(ABSOLUTE_THRESHOLDS),
            "top5_mean_deltas": list(TOP5_DELTAS),
            "relative_definition": "keep score >= eligible_top5_mean - delta",
            "pairwise_keep_delta": PAIRWISE_KEEP_DELTA,
            "pairwise_candidate_sample_max": PAIRWISE_SAMPLE_MAX,
            "pair_sample_max": PAIR_SAMPLE_MAX,
            "merge_thresholds": list(MERGE_THRESHOLDS),
            "run_metric_suite": args.run_metrics,
        },
        "summary": _summarize(turn_rows),
        "full_merge_evaluation": merge_evaluation,
        "turns": turn_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_render_markdown(report), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--dense-index", type=Path, default=DEFAULT_DENSE_INDEX)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cohort", choices=("all", "natural", "simulator"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--max-turn",
        type=int,
        help="ignore later turns in each conversation (for example, 3 for simulator runs)",
    )
    parser.add_argument("--run-merge", action="store_true")
    parser.add_argument("--run-metrics", action="store_true")
    parser.add_argument(
        "--ignore-hard-mask",
        action="store_true",
        help="measure the semantic support space over all products without facet filtering",
    )
    parser.add_argument("--merge-max-candidates", type=int, default=20_000)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.max_turn is not None and args.max_turn <= 0:
        parser.error("--max-turn must be positive")
    if args.merge_max_candidates <= 1:
        parser.error("--merge-max-candidates must exceed one")
    if args.run_metrics and not args.run_merge:
        parser.error("--run-metrics requires --run-merge")
    return args


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("evaluation must be a JSON object")
    return cast(dict[str, object], value)


def _load_turns(
    payload: dict[str, object],
    *,
    cohort: str,
    max_turn: int | None,
) -> list[SavedTurn]:
    raw_turns = payload.get("turns")
    if type(raw_turns) is not list:
        raise ValueError("evaluation.turns must be an array")
    turns: list[SavedTurn] = []
    for raw in cast(list[object], raw_turns):
        if type(raw) is not dict:
            raise ValueError("evaluation turn must be an object")
        item = cast(dict[str, object], raw)
        if item.get("status") != "success":
            continue
        observed_cohort = _string(item, "cohort")
        if cohort != "all" and observed_cohort != cohort:
            continue
        turn_number = _integer(item, "turn")
        if max_turn is not None and turn_number > max_turn:
            continue
        compiled = _mapping(item, "compiled")
        if not bool(compiled.get("search_ready")):
            continue
        q_sem = _string(compiled, "q_sem")
        final_intent = _mapping(item, "final_intent")
        preferences = final_intent.get("preferences")
        hard_constraints = compiled.get("hard_constraints")
        mask = _mapping(item, "mask")
        if type(preferences) is not list or type(hard_constraints) is not list:
            raise ValueError("saved turn has malformed preferences or hard constraints")
        turns.append(
            SavedTurn(
                suite_id=_string(item, "suite_id"),
                cohort=observed_cohort,
                conversation_id=_string(item, "conversation_id"),
                turn=turn_number,
                scenario_type=_optional_string(item, "scenario_type", default="unspecified"),
                tags=_string_tuple(item, "tags"),
                user_message=_string(item, "user_message"),
                q_sem=q_sem,
                intent_version=_integer(compiled, "intent_version"),
                preference_count=len(preferences),
                hard_constraint_count=len(hard_constraints),
                reported_eligible_count=_integer(mask, "eligible_count"),
                compiled=compiled,
            )
        )
    return turns


def _resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _encode_queries(
    turns: Sequence[SavedTurn],
    *,
    dense: DenseIndex,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    embedder = SentenceTransformerTextEmbedder(
        dense.manifest.embedding,
        device=str(device),
        local_files_only=True,
    )
    vectors = np.stack([embedder.encode_query(turn.q_sem) for turn in turns])
    return np.asarray(vectors, dtype=np.float32), time.perf_counter() - started


def _resolve_masks(
    turns: Sequence[SavedTurn],
    *,
    release: VerifiedCatalogSemanticRelease,
    dense: DenseIndex,
    release_dir: Path,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    if not any(turn.hard_constraint_count for turn in turns):
        masks = np.ones((len(turns), dense.manifest.product_count), dtype=np.bool_)
        return masks, time.perf_counter() - started
    evidence = build_retrieval_evidence_index(
        release_dir / "catalog.jsonl",
        catalog_id=release.manifest.catalog_id,
        catalog_semantic_release_id=release.release_id,
        expected_parent_asins=set(dense.parent_asins),
    )
    resolver = HardMaskResolver(release=release, evidence_index=evidence, dense_index=dense)
    masks: list[np.ndarray] = []
    for turn in turns:
        if turn.hard_constraint_count:
            resolution = resolver.resolve(
                _restore_compiled_query(turn, release=release, dense=dense)
            )
            observed = len(resolution.eligible_parent_asins)
            if observed != turn.reported_eligible_count:
                raise ValueError(
                    f"{turn.identity}: reproduced eligible_count={observed}, "
                    f"saved={turn.reported_eligible_count}"
                )
            masks.append(np.array(resolution.eligible_mask.values, dtype=np.bool_, copy=True))
        else:
            if turn.reported_eligible_count != dense.manifest.product_count:
                raise ValueError(
                    f"{turn.identity}: no constraints but saved mask is not all eligible"
                )
            masks.append(np.ones(dense.manifest.product_count, dtype=np.bool_))
    return np.stack(masks), time.perf_counter() - started


def _restore_compiled_query(
    turn: SavedTurn,
    *,
    release: VerifiedCatalogSemanticRelease,
    dense: DenseIndex,
) -> CompiledQuery:
    raw_constraints = turn.compiled.get("hard_constraints")
    assert type(raw_constraints) is list
    constraints = []
    for raw in cast(list[object], raw_constraints):
        if type(raw) is not dict:
            raise ValueError(f"{turn.identity}: malformed hard constraint")
        item = cast(dict[str, object], raw)
        value = item.get("value")
        if type(value) is list:
            value = tuple(cast(list[object], value))
        constraints.append(
            CompiledHardConstraint(
                preference_id=_string(item, "preference_id"),
                facet=_string(item, "facet"),
                operator=Operator(_string(item, "operator")),
                value=cast(Any, value),
                policy=ConstraintPolicy(_string(item, "policy")),
            )
        )
    return CompiledQuery(
        schema=COMPILED_QUERY_SCHEMA,
        compiler_version=QUERY_COMPILER_VERSION,
        catalog_id=dense.manifest.catalog_id,
        catalog_semantic_release_id=dense.manifest.catalog_semantic_release_id,
        category_graph_id=release.category_registry.category_graph_id,
        intent_version=turn.intent_version,
        q_lex=str(turn.compiled.get("q_lex", "")),
        q_sem=turn.q_sem,
        search_ready=True,
        hard_constraints=tuple(constraints),
        ranking_preferences=(),
        dont_care_facets=tuple(cast(list[str], turn.compiled.get("dont_care_facets", []))),
        directives=CompiledDirectives(
            diversity=DiversityDirective.AUTO,
            comparison_requested=False,
            explanation_requested=False,
        ),
        requires_clarification=False,
        clarification_reason=None,
        trace=(),
    )


def _score_catalog(
    query_vectors: np.ndarray,
    *,
    dense: DenseIndex,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    catalog = torch.from_numpy(np.array(dense.vectors, copy=True)).to(device)
    queries = torch.from_numpy(query_vectors).to(device)
    with torch.inference_mode():
        scores = queries @ catalog.T
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = scores.cpu().numpy().astype(np.float32, copy=False)
    return result, time.perf_counter() - started


def _sample_pairwise_geometry(
    turns: Sequence[SavedTurn],
    *,
    score_rows: np.ndarray,
    eligible_masks: np.ndarray,
    dense: DenseIndex,
    device: torch.device,
) -> tuple[list[dict[str, object]], float]:
    """Measure pair and nearest-neighbor similarity in one plausible keep region."""

    started = time.perf_counter()
    catalog = torch.from_numpy(np.array(dense.vectors, copy=True)).to(device)
    results: list[dict[str, object]] = []
    with torch.inference_mode():
        for turn_index, _turn in enumerate(turns):
            scores = score_rows[turn_index]
            eligible = eligible_masks[turn_index]
            eligible_scores = scores[eligible]
            sorted_scores = np.sort(eligible_scores)[::-1]
            top5_mean = float(np.mean(sorted_scores[: min(5, sorted_scores.size)]))
            cutoff = top5_mean - PAIRWISE_KEEP_DELTA
            candidate_rows = np.flatnonzero(eligible & (scores >= cutoff))
            order = np.lexsort((candidate_rows, -scores[candidate_rows]))
            candidate_rows = candidate_rows[order]
            support_count = int(candidate_rows.size)
            if support_count > PAIRWISE_SAMPLE_MAX:
                positions = np.linspace(
                    0,
                    support_count - 1,
                    num=PAIRWISE_SAMPLE_MAX,
                    dtype=np.int64,
                )
                sampled_rows = candidate_rows[positions]
            else:
                sampled_rows = candidate_rows
            sampled_count = int(sampled_rows.size)
            if sampled_count < 2:
                results.append(
                    {
                        "available": False,
                        "reason": "insufficient_candidates",
                        "keep_delta": PAIRWISE_KEEP_DELTA,
                        "score_cutoff": cutoff,
                        "support_count": support_count,
                        "sampled_candidate_count": sampled_count,
                    }
                )
                continue

            row_indices = torch.from_numpy(sampled_rows).to(device=device, dtype=torch.long)
            candidate_vectors = catalog.index_select(0, row_indices)
            similarities = candidate_vectors @ candidate_vectors.T
            triangular = torch.triu_indices(
                sampled_count,
                sampled_count,
                offset=1,
                device=device,
            )
            pair_total = int(triangular.shape[1])
            if pair_total > PAIR_SAMPLE_MAX:
                sampled_positions = (
                    torch.linspace(
                        0,
                        pair_total - 1,
                        steps=PAIR_SAMPLE_MAX,
                        device=device,
                    )
                    .round()
                    .to(dtype=torch.long)
                )
                triangular = triangular.index_select(1, sampled_positions)
            pair_values = similarities[triangular[0], triangular[1]]
            similarities.fill_diagonal_(-torch.inf)
            nearest = torch.max(similarities, dim=1).values
            pair_numpy = pair_values.cpu().numpy()
            nearest_numpy = nearest.cpu().numpy()
            results.append(
                {
                    "available": True,
                    "reason": None,
                    "keep_delta": PAIRWISE_KEEP_DELTA,
                    "score_cutoff": cutoff,
                    "support_count": support_count,
                    "sampled_candidate_count": sampled_count,
                    "sampled_pair_count": int(pair_numpy.size),
                    "pair_similarity_quantiles": {
                        _quantile_key(quantile): float(value)
                        for quantile, value in zip(
                            PAIRWISE_QUANTILES,
                            np.quantile(pair_numpy, PAIRWISE_QUANTILES),
                            strict=True,
                        )
                    },
                    "nearest_neighbor_quantiles": {
                        _quantile_key(quantile): float(value)
                        for quantile, value in zip(
                            PAIRWISE_QUANTILES,
                            np.quantile(nearest_numpy, PAIRWISE_QUANTILES),
                            strict=True,
                        )
                    },
                    "threshold_fractions": {
                        _key(threshold): {
                            "sampled_pair_fraction": float(
                                np.mean(pair_numpy >= threshold, dtype=np.float64)
                            ),
                            "candidate_with_neighbor_fraction": float(
                                np.mean(nearest_numpy >= threshold, dtype=np.float64)
                            ),
                        }
                        for threshold in MERGE_THRESHOLDS
                    },
                }
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return results, time.perf_counter() - started


def _evaluate_full_merging(
    turns: Sequence[SavedTurn],
    *,
    score_rows: np.ndarray,
    eligible_masks: np.ndarray,
    dense: DenseIndex,
    device: torch.device,
    max_candidates: int,
    run_metric_suite: bool,
) -> tuple[dict[str, object], float]:
    """Run exact GPU pair matrices and query-ordered greedy semantic merging."""

    started = time.perf_counter()
    catalog_numpy = np.asarray(dense.vectors, dtype=np.float32)
    catalog_mean = compute_catalog_mean(catalog_numpy)
    catalog = torch.from_numpy(np.array(catalog_numpy, copy=True)).to(device)
    turn_results: list[dict[str, object]] = []

    with torch.inference_mode():
        for turn_index, turn in enumerate(turns):
            scores = score_rows[turn_index]
            eligible = eligible_masks[turn_index]
            eligible_scores = scores[eligible]
            sorted_scores = np.sort(eligible_scores)[::-1]
            top5_mean = float(np.mean(sorted_scores[: min(5, sorted_scores.size)]))
            widest_delta = max(FULL_MERGE_KEEP_DELTAS)
            widest_cutoff = top5_mean - widest_delta
            candidate_rows = np.flatnonzero(eligible & (scores >= widest_cutoff))
            order = np.lexsort((candidate_rows, -scores[candidate_rows]))
            candidate_rows = candidate_rows[order]
            if candidate_rows.size > max_candidates:
                raise ValueError(
                    f"{turn.identity}: merge candidate count {candidate_rows.size} exceeds "
                    f"the explicit safety limit {max_candidates}"
                )
            row_indices = torch.from_numpy(candidate_rows).to(device=device, dtype=torch.long)
            candidate_vectors_gpu = catalog.index_select(0, row_indices)
            pairwise_gpu = candidate_vectors_gpu @ candidate_vectors_gpu.T
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            pairwise = pairwise_gpu.cpu().numpy()
            candidate_vectors = catalog_numpy[candidate_rows]
            configurations: dict[str, object] = {}

            for keep_delta in FULL_MERGE_KEEP_DELTAS:
                cutoff = top5_mean - keep_delta
                retained_count = int(np.count_nonzero(scores[candidate_rows] >= cutoff))
                retained_vectors = candidate_vectors[:retained_count]
                raw_coherence = _coherence_payload(
                    compute_probe_coherence(retained_vectors, catalog_mean)
                )
                raw_metrics = (
                    _dispersion_metric_suite(
                        retained_vectors,
                        pairwise_gpu[:retained_count, :retained_count],
                    )
                    if run_metric_suite
                    else None
                )
                configurations[_merge_key(keep_delta, None)] = {
                    "keep_delta": keep_delta,
                    "merge_threshold": None,
                    "score_cutoff": cutoff,
                    "candidate_count": retained_count,
                    "representative_count": retained_count,
                    "reduction_fraction": 0.0,
                    "cluster_size": {
                        "min": 1,
                        "median": 1,
                        "mean": 1.0,
                        "p90": 1,
                        "max": 1,
                    },
                    "coherence": raw_coherence,
                    "metric_suite": raw_metrics,
                }
                similarity_view = pairwise[:retained_count, :retained_count]
                for merge_threshold in FULL_MERGE_THRESHOLDS:
                    centroids, cluster_sizes = _greedy_merge(
                        retained_vectors,
                        similarity_view,
                        threshold=merge_threshold,
                    )
                    representative_count = int(centroids.shape[0])
                    merged_metrics = None
                    if run_metric_suite and math.isclose(
                        merge_threshold,
                        METRIC_MERGE_THRESHOLD,
                        abs_tol=1e-12,
                    ):
                        centroid_tensor = torch.from_numpy(
                            np.asarray(centroids, dtype=np.float32)
                        ).to(device)
                        centroid_pairwise = centroid_tensor @ centroid_tensor.T
                        merged_metrics = _dispersion_metric_suite(
                            centroids,
                            centroid_pairwise,
                        )
                        del centroid_pairwise, centroid_tensor
                    configurations[_merge_key(keep_delta, merge_threshold)] = {
                        "keep_delta": keep_delta,
                        "merge_threshold": merge_threshold,
                        "score_cutoff": cutoff,
                        "candidate_count": retained_count,
                        "representative_count": representative_count,
                        "reduction_fraction": 1.0 - representative_count / retained_count,
                        "cluster_size": _distribution(cluster_sizes.tolist()),
                        "coherence": _coherence_payload(
                            compute_probe_coherence(centroids, catalog_mean)
                        ),
                        "metric_suite": merged_metrics,
                    }

            turn_results.append(
                {
                    "identity": turn.identity,
                    "cohort": turn.cohort,
                    "conversation_id": turn.conversation_id,
                    "turn": turn.turn,
                    "tags": list(turn.tags),
                    "user_message": turn.user_message,
                    "q_sem": turn.q_sem,
                    "eligible_count": int(np.count_nonzero(eligible)),
                    "eligible_top5_mean": top5_mean,
                    "configurations": configurations,
                }
            )
            del pairwise, pairwise_gpu, candidate_vectors_gpu
            if (turn_index + 1) % 10 == 0 or turn_index + 1 == len(turns):
                print(f"merged {turn_index + 1}/{len(turns)} turns")

    return (
        {
            "algorithm": {
                "keep_definition": "score >= eligible_top5_mean - keep_delta",
                "keep_deltas": list(FULL_MERGE_KEEP_DELTAS),
                "merge_thresholds": list(FULL_MERGE_THRESHOLDS),
                "merge_order": "descending query similarity, stable catalog row tie-break",
                "merge_rule": "one leader absorbs currently-unassigned products at or above threshold",
                "cluster_representation": "L2-normalized mean product vector",
                "cluster_weighting": "one equal vote per cluster centroid",
                "max_candidates": max_candidates,
                "metric_suite": (
                    {
                        "raw_and_merge_threshold": METRIC_MERGE_THRESHOLD,
                        "kernel_temperatures": list(KERNEL_TEMPERATURES),
                        "logdet_betas": list(LOGDET_BETAS),
                        "knn_values": list(KNN_VALUES),
                    }
                    if run_metric_suite
                    else None
                ),
            },
            "summary": _summarize_full_merge(turn_results),
            "turns": turn_results,
        },
        time.perf_counter() - started,
    )


def _greedy_merge(
    vectors: np.ndarray,
    similarities: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    count = int(vectors.shape[0])
    if similarities.shape != (count, count):
        raise ValueError("merge similarity matrix differs from candidate vectors")
    unassigned = np.ones(count, dtype=np.bool_)
    labels = np.full(count, -1, dtype=np.int32)
    cluster_count = 0
    for anchor in range(count):
        if not unassigned[anchor]:
            continue
        members = unassigned & (similarities[anchor] >= threshold)
        if not members[anchor]:
            raise ValueError("merge matrix lost its self similarity")
        labels[members] = cluster_count
        unassigned[members] = False
        cluster_count += 1
    if np.any(labels < 0):
        raise ValueError("merge left candidates unassigned")
    cluster_sizes = np.bincount(labels, minlength=cluster_count).astype(np.int64, copy=False)
    sums = np.zeros((cluster_count, vectors.shape[1]), dtype=np.float64)
    np.add.at(sums, labels, vectors)
    centroids = sums / cluster_sizes[:, np.newaxis]
    norms = np.linalg.norm(centroids, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("merge produced an invalid centroid")
    centroids /= norms[:, np.newaxis]
    return centroids, cluster_sizes


def _dispersion_metric_suite(
    vectors: np.ndarray,
    similarities: torch.Tensor,
) -> dict[str, object]:
    """Evaluate several high-dimensional spread estimators on one fixed point set."""

    count, dimension = vectors.shape
    if similarities.shape != (count, count):
        raise ValueError("metric similarity matrix differs from candidate vectors")
    if count == 0:
        raise ValueError("metric suite requires at least one vector")

    pairwise_metrics: dict[str, object]
    if count < 2:
        pairwise_metrics = {"available": False, "reason": "insufficient_candidates"}
    else:
        pair_values = _sample_metric_pairs(similarities)
        pair_values = torch.clamp(pair_values, min=-1.0, max=1.0)
        distances = torch.sqrt(torch.clamp(2.0 - 2.0 * pair_values, min=0.0))
        pair_numpy = pair_values.cpu().numpy()
        distance_numpy = distances.cpu().numpy()
        pairwise_metrics = {
            "available": True,
            "sample_count": int(pair_numpy.size),
            "median_cosine": float(np.median(pair_numpy)),
            "mean_cosine": float(np.mean(pair_numpy, dtype=np.float64)),
            "median_angular_distance": float(np.median(distance_numpy)),
            "p90_angular_distance": float(np.quantile(distance_numpy, 0.9)),
        }

    kernel_effective_number: dict[str, float] = {}
    for temperature in KERNEL_TEMPERATURES:
        kernel_square_sum = 0.0
        row_block = 2_048
        for start in range(0, count, row_block):
            block = similarities[start : start + row_block]
            square_kernel = torch.exp((2.0 / temperature) * (block - 1.0))
            kernel_square_sum += float(torch.sum(square_kernel, dtype=torch.float64).item())
        kernel_effective_number[_key(temperature)] = float(count * count / kernel_square_sum)

    matrix = np.asarray(vectors, dtype=np.float64)
    centered = matrix - np.mean(matrix, axis=0, dtype=np.float64)
    covariance = centered.T @ centered / count
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    trace = float(np.sum(eigenvalues, dtype=np.float64))
    spectrum: dict[str, object]
    logdet: dict[str, float | None]
    if not math.isfinite(trace) or trace <= 0.0:
        spectrum = {"available": False, "reason": "zero_covariance"}
        logdet = {_key(beta): None for beta in LOGDET_BETAS}
    else:
        positive = eigenvalues[eigenvalues > 0.0]
        probabilities = positive / trace
        spectrum = {
            "available": True,
            "total_variance": trace,
            "stable_rank": trace / float(np.max(positive)),
            "renyi2_effective_rank": trace
            * trace
            / float(np.sum(positive * positive, dtype=np.float64)),
            "shannon_effective_rank": float(
                np.exp(-np.sum(probabilities * np.log(probabilities), dtype=np.float64))
            ),
        }
        logdet = {
            _key(beta): float(np.sum(np.log1p(beta * positive), dtype=np.float64))
            for beta in LOGDET_BETAS
        }

    knn: dict[str, object] = {}
    if count < 2:
        knn = {_key(float(k)): None for k in KNN_VALUES}
    else:
        maximum_k = min(max(KNN_VALUES), count - 1)
        diagonal = similarities.diagonal()
        saved_diagonal = diagonal.clone()
        diagonal.fill_(-torch.inf)
        nearest = torch.topk(similarities, k=maximum_k, dim=1, largest=True).values
        diagonal.copy_(saved_diagonal)
        psi_n = float(
            torch.special.digamma(
                torch.tensor(float(count), device=similarities.device, dtype=torch.float64)
            ).item()
        )
        for k in KNN_VALUES:
            if k >= count:
                knn[_key(float(k))] = None
                continue
            kth_cosine = torch.clamp(nearest[:, k - 1], min=-1.0, max=1.0)
            kth_distance = torch.sqrt(torch.clamp(2.0 - 2.0 * kth_cosine, min=1e-12))
            mean_log_distance = float(torch.mean(torch.log(kth_distance)).item())
            psi_k = float(
                torch.special.digamma(
                    torch.tensor(float(k), device=similarities.device, dtype=torch.float64)
                ).item()
            )
            knn[_key(float(k))] = {
                "mean_kth_neighbor_distance": float(torch.mean(kth_distance).item()),
                "median_kth_neighbor_distance": float(torch.median(kth_distance).item()),
                "kl_entropy_proxy_without_unit_ball_constant": psi_n
                - psi_k
                + dimension * mean_log_distance,
            }

    return {
        "vector_count": count,
        "dimension": dimension,
        "pairwise": pairwise_metrics,
        "kernel_effective_number": kernel_effective_number,
        "covariance_spectrum": spectrum,
        "regularized_logdet": logdet,
        "knn": knn,
    }


def _sample_metric_pairs(similarities: torch.Tensor) -> torch.Tensor:
    count = int(similarities.shape[0])
    pair_total = count * (count - 1) // 2
    if pair_total <= METRIC_PAIR_SAMPLE_MAX:
        triangular = torch.triu_indices(count, count, offset=1, device=similarities.device)
        return similarities[triangular[0], triangular[1]]
    positions = torch.arange(
        METRIC_PAIR_SAMPLE_MAX,
        device=similarities.device,
        dtype=torch.long,
    )
    left = (positions * 104_729 + 17) % count
    right = (positions * 130_363 + 31) % count
    right = torch.where(right == left, (right + 1) % count, right)
    return similarities[left, right]


def _coherence_payload(result: object) -> dict[str, object]:
    from shopping_copilot.retrieval.coherence import ProbeCoherence

    assert type(result) is ProbeCoherence
    return {
        "available": result.available,
        "reason": result.reason,
        "n": result.n,
        "resultant_length": result.resultant_length,
        "debiased_pairwise_cosine": result.debiased_pairwise_cosine,
    }


def _merge_key(keep_delta: float, merge_threshold: float | None) -> str:
    merge = "raw" if merge_threshold is None else f"m{merge_threshold:.3f}"
    return f"d{keep_delta:.3f}_{merge}"


def _analyze_turn(
    turn: SavedTurn,
    *,
    scores: np.ndarray,
    eligible: np.ndarray,
    pairwise: dict[str, object],
    validate_reported_eligible: bool,
) -> dict[str, object]:
    eligible_scores = scores[eligible]
    if validate_reported_eligible and eligible_scores.size != turn.reported_eligible_count:
        raise ValueError(f"{turn.identity}: eligible score count mismatch")
    sorted_desc = np.sort(eligible_scores)[::-1]
    top5_mean = float(np.mean(sorted_desc[: min(5, sorted_desc.size)], dtype=np.float64))
    quantiles = np.quantile(eligible_scores, TURN_QUANTILES)
    absolute_counts = {
        _key(threshold): int(np.count_nonzero(eligible_scores >= threshold))
        for threshold in ABSOLUTE_THRESHOLDS
    }
    relative_counts = {
        _key(delta): int(np.count_nonzero(eligible_scores >= top5_mean - delta))
        for delta in TOP5_DELTAS
    }
    return {
        "identity": turn.identity,
        "suite_id": turn.suite_id,
        "cohort": turn.cohort,
        "conversation_id": turn.conversation_id,
        "turn": turn.turn,
        "scenario_type": turn.scenario_type,
        "tags": list(turn.tags),
        "user_message": turn.user_message,
        "q_sem": turn.q_sem,
        "preference_count": turn.preference_count,
        "hard_constraint_count": turn.hard_constraint_count,
        "eligible_count": int(eligible_scores.size),
        "score_quantiles": {
            _quantile_key(quantile): float(value)
            for quantile, value in zip(TURN_QUANTILES, quantiles, strict=True)
        },
        "eligible_top5_mean": top5_mean,
        "absolute_candidate_counts": absolute_counts,
        "top5_delta_candidate_counts": relative_counts,
        "pairwise_sample": pairwise,
    }


def _summarize(turn_rows: Sequence[dict[str, object]]) -> dict[str, object]:
    cohorts = sorted({str(row["cohort"]) for row in turn_rows})
    return {
        "all": _summarize_group(turn_rows),
        "by_cohort": {
            cohort: _summarize_group([row for row in turn_rows if row["cohort"] == cohort])
            for cohort in cohorts
        },
        "clarity_pairs": _clarity_pairs(turn_rows),
        "transitions": _transition_summary(turn_rows),
        "pairwise_sample": _summarize_pairwise(turn_rows),
    }


def _summarize_group(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    return {
        "turn_count": len(rows),
        "eligible_count": _distribution([int(row["eligible_count"]) for row in rows]),
        "eligible_top5_mean": _distribution([float(row["eligible_top5_mean"]) for row in rows]),
        "score_quantile_medians": {
            _quantile_key(quantile): statistics.median(
                float(cast(dict[str, object], row["score_quantiles"])[_quantile_key(quantile)])
                for row in rows
            )
            for quantile in TURN_QUANTILES
        },
        "absolute_thresholds": {
            _key(threshold): _candidate_count_summary(
                rows,
                field="absolute_candidate_counts",
                parameter=threshold,
            )
            for threshold in ABSOLUTE_THRESHOLDS
        },
        "top5_mean_deltas": {
            _key(delta): _candidate_count_summary(
                rows,
                field="top5_delta_candidate_counts",
                parameter=delta,
            )
            for delta in TOP5_DELTAS
        },
    }


def _candidate_count_summary(
    rows: Sequence[dict[str, object]],
    *,
    field: str,
    parameter: float,
) -> dict[str, object]:
    counts = [int(cast(dict[str, object], row[field])[_key(parameter)]) for row in rows]
    return {
        **_distribution(counts),
        "empty_count": sum(count == 0 for count in counts),
        "le_512_count": sum(count <= 512 for count in counts),
        "le_2048_count": sum(count <= 2048 for count in counts),
        "gt_4096_count": sum(count > 4096 for count in counts),
        "gt_10000_count": sum(count > 10_000 for count in counts),
    }


def _summarize_pairwise(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    available = [
        cast(dict[str, object], row["pairwise_sample"])
        for row in rows
        if bool(cast(dict[str, object], row["pairwise_sample"])["available"])
    ]
    if not available:
        return {"available_turn_count": 0}
    return {
        "available_turn_count": len(available),
        "keep_delta": PAIRWISE_KEEP_DELTA,
        "support_count": _distribution([int(item["support_count"]) for item in available]),
        "sampled_candidate_count": _distribution(
            [int(item["sampled_candidate_count"]) for item in available]
        ),
        "median_turn_pair_similarity_quantiles": {
            _quantile_key(quantile): statistics.median(
                float(
                    cast(dict[str, object], item["pair_similarity_quantiles"])[
                        _quantile_key(quantile)
                    ]
                )
                for item in available
            )
            for quantile in PAIRWISE_QUANTILES
        },
        "median_turn_nearest_neighbor_quantiles": {
            _quantile_key(quantile): statistics.median(
                float(
                    cast(dict[str, object], item["nearest_neighbor_quantiles"])[
                        _quantile_key(quantile)
                    ]
                )
                for item in available
            )
            for quantile in PAIRWISE_QUANTILES
        },
        "merge_thresholds": {
            _key(threshold): {
                "sampled_pair_fraction": _distribution(
                    [
                        float(
                            cast(
                                dict[str, object],
                                cast(dict[str, object], item["threshold_fractions"])[
                                    _key(threshold)
                                ],
                            )["sampled_pair_fraction"]
                        )
                        for item in available
                    ]
                ),
                "candidate_with_neighbor_fraction": _distribution(
                    [
                        float(
                            cast(
                                dict[str, object],
                                cast(dict[str, object], item["threshold_fractions"])[
                                    _key(threshold)
                                ],
                            )["candidate_with_neighbor_fraction"]
                        )
                        for item in available
                    ]
                ),
            }
            for threshold in MERGE_THRESHOLDS
        },
    }


def _summarize_full_merge(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot summarize empty merge results")
    first_configurations = cast(dict[str, object], rows[0]["configurations"])
    config_keys = tuple(first_configurations)
    configurations: dict[str, object] = {}
    for config_key in config_keys:
        items = [
            cast(
                dict[str, object],
                cast(dict[str, object], row["configurations"])[config_key],
            )
            for row in rows
        ]
        coherence_values = [
            float(value)
            for item in items
            if (value := cast(dict[str, object], item["coherence"])["debiased_pairwise_cosine"])
            is not None
        ]
        configurations[config_key] = {
            "keep_delta": items[0]["keep_delta"],
            "merge_threshold": items[0]["merge_threshold"],
            "candidate_count": _distribution([int(item["candidate_count"]) for item in items]),
            "representative_count": _distribution(
                [int(item["representative_count"]) for item in items]
            ),
            "reduction_fraction": _distribution(
                [float(item["reduction_fraction"]) for item in items]
            ),
            "coherence_available_count": len(coherence_values),
            "coherence": (_distribution(coherence_values) if coherence_values else None),
        }

    clarity_pairs: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row["conversation_id"]).startswith("c0"):
            grouped[str(row["conversation_id"])].append(row)
    for conversation_id, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: int(item["turn"]))
        if len(ordered) != 2:
            continue
        deltas: dict[str, object] = {}
        for config_key in config_keys:
            vague = cast(
                dict[str, object],
                cast(dict[str, object], ordered[0]["configurations"])[config_key],
            )
            specific = cast(
                dict[str, object],
                cast(dict[str, object], ordered[1]["configurations"])[config_key],
            )
            vague_coherence = cast(dict[str, object], vague["coherence"])[
                "debiased_pairwise_cosine"
            ]
            specific_coherence = cast(dict[str, object], specific["coherence"])[
                "debiased_pairwise_cosine"
            ]
            deltas[config_key] = {
                "vague_candidate_count": vague["candidate_count"],
                "specific_candidate_count": specific["candidate_count"],
                "vague_representative_count": vague["representative_count"],
                "specific_representative_count": specific["representative_count"],
                "vague_coherence": vague_coherence,
                "specific_coherence": specific_coherence,
                "coherence_delta": (
                    None
                    if vague_coherence is None or specific_coherence is None
                    else float(specific_coherence) - float(vague_coherence)
                ),
            }
        clarity_pairs.append(
            {
                "conversation_id": conversation_id,
                "vague_message": ordered[0]["user_message"],
                "specific_message": ordered[1]["user_message"],
                "configurations": deltas,
            }
        )

    for config_key in config_keys:
        clarity_deltas = [
            cast(
                dict[str, object],
                cast(dict[str, object], pair["configurations"])[config_key],
            )["coherence_delta"]
            for pair in clarity_pairs
        ]
        available_deltas = [float(value) for value in clarity_deltas if value is not None]
        cast(dict[str, object], configurations[config_key])["clarity_pairs"] = {
            "available_count": len(available_deltas),
            "specific_higher_count": sum(value > 0.0 for value in available_deltas),
            "specific_lower_count": sum(value < 0.0 for value in available_deltas),
            "delta": _distribution(available_deltas) if available_deltas else None,
        }

    return {
        "turn_count": len(rows),
        "configurations": configurations,
        "clarity_pairs": clarity_pairs,
        "metric_comparison": _summarize_metric_comparison(rows),
    }


def _summarize_metric_comparison(rows: Sequence[dict[str, object]]) -> dict[str, object] | None:
    first_configurations = cast(dict[str, object], rows[0]["configurations"])
    metric_config_keys = [
        key
        for key, raw in first_configurations.items()
        if cast(dict[str, object], raw).get("metric_suite") is not None
    ]
    if not metric_config_keys:
        return None

    clarity_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    simulator_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    expected_groups: dict[str, dict[str, list[dict[str, object]]]] = {
        tag: defaultdict(list) for tag in EXPECTED_TRANSITION_TAGS
    }
    for row in rows:
        conversation_id = str(row["conversation_id"])
        if conversation_id.startswith("c0"):
            clarity_groups[conversation_id].append(row)
        if row["cohort"] == "simulator":
            simulator_groups[conversation_id].append(row)
        if row["cohort"] == "natural":
            tags = _row_tags(row)
            matched = [tag for tag in EXPECTED_TRANSITION_TAGS if tag in tags]
            if len(matched) > 1:
                raise ValueError(f"{conversation_id} has conflicting expected-transition tags")
            if matched:
                expected_groups[matched[0]][conversation_id].append(row)

    result: dict[str, object] = {}
    for config_key in metric_config_keys:
        per_turn: list[dict[str, tuple[float, str]]] = []
        for row in rows:
            config = cast(
                dict[str, object],
                cast(dict[str, object], row["configurations"])[config_key],
            )
            per_turn.append(_flatten_metric_values(config))
        metric_ids = sorted(set.union(*(set(item) for item in per_turn)))
        metric_rows: dict[str, object] = {}
        for metric_id in metric_ids:
            observations = [item[metric_id] for item in per_turn if metric_id in item]
            values = [item[0] for item in observations]
            direction = observations[0][1]
            clarity_details = _metric_pair_details(
                clarity_groups,
                config_key=config_key,
                metric_id=metric_id,
                direction=direction,
            )
            simulator_details = _metric_pair_details(
                simulator_groups,
                config_key=config_key,
                metric_id=metric_id,
                direction=direction,
                first_last=True,
            )
            expected_details = {
                tag: _metric_expected_transition_details(
                    groups,
                    config_key=config_key,
                    metric_id=metric_id,
                    direction=direction,
                    expectation=tag,
                )
                for tag, groups in expected_groups.items()
            }
            metric_rows[metric_id] = {
                "clear_direction": direction,
                "distribution": _distribution(values),
                "available_turn_count": len(values),
                "clarity": clarity_details,
                "simulator_first_to_last_observation": simulator_details,
                "natural_expected_transitions": expected_details,
            }
        result[config_key] = {"metrics": metric_rows}
    return result


def _metric_pair_details(
    groups: dict[str, list[dict[str, object]]],
    *,
    config_key: str,
    metric_id: str,
    direction: str,
    first_last: bool = False,
) -> dict[str, object]:
    pairs: list[dict[str, object]] = []
    for conversation_id, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: int(row["turn"]))
        if len(ordered) < 2:
            continue
        if not first_last and len(ordered) != 2:
            continue
        left = ordered[0]
        right = ordered[-1]
        left_config = cast(
            dict[str, object],
            cast(dict[str, object], left["configurations"])[config_key],
        )
        right_config = cast(
            dict[str, object],
            cast(dict[str, object], right["configurations"])[config_key],
        )
        left_metrics = _flatten_metric_values(left_config)
        right_metrics = _flatten_metric_values(right_config)
        if metric_id not in left_metrics or metric_id not in right_metrics:
            continue
        left_value = left_metrics[metric_id][0]
        right_value = right_metrics[metric_id][0]
        delta = right_value - left_value
        clearer = delta > 0.0 if direction == "higher" else delta < 0.0
        pairs.append(
            {
                "conversation_id": conversation_id,
                "first": left_value,
                "last": right_value,
                "delta": delta,
                "clearer": clearer,
            }
        )
    return {
        "pair_count": len(pairs),
        "clearer_count": sum(bool(pair["clearer"]) for pair in pairs),
        "not_clearer_count": sum(not bool(pair["clearer"]) for pair in pairs),
        "pairs": pairs,
    }


def _metric_expected_transition_details(
    groups: dict[str, list[dict[str, object]]],
    *,
    config_key: str,
    metric_id: str,
    direction: str,
    expectation: str,
) -> dict[str, object]:
    pairs: list[dict[str, object]] = []
    for conversation_id, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: int(row["turn"]))
        if len(ordered) != 2:
            continue
        left_metrics = _flatten_metric_values(
            cast(
                dict[str, object],
                cast(dict[str, object], ordered[0]["configurations"])[config_key],
            )
        )
        right_metrics = _flatten_metric_values(
            cast(
                dict[str, object],
                cast(dict[str, object], ordered[1]["configurations"])[config_key],
            )
        )
        if metric_id not in left_metrics or metric_id not in right_metrics:
            continue
        first = left_metrics[metric_id][0]
        last = right_metrics[metric_id][0]
        delta = last - first
        relative_change = abs(delta) / max(abs(first), abs(last), 1e-12)
        if expectation == "expected_narrower":
            matched: bool | None = delta > 0.0 if direction == "higher" else delta < 0.0
        elif expectation == "expected_broader":
            matched = delta < 0.0 if direction == "higher" else delta > 0.0
        elif expectation == "expected_stable":
            matched = relative_change <= STABLE_RELATIVE_TOLERANCE
        elif expectation == "expected_override":
            matched = None
        else:
            raise ValueError(f"unsupported expectation: {expectation}")
        pairs.append(
            {
                "conversation_id": conversation_id,
                "first": first,
                "last": last,
                "delta": delta,
                "relative_change": relative_change,
                "matched": matched,
            }
        )
    scored = [pair for pair in pairs if pair["matched"] is not None]
    return {
        "pair_count": len(pairs),
        "scored_pair_count": len(scored),
        "matched_count": sum(bool(pair["matched"]) for pair in scored),
        "not_matched_count": sum(not bool(pair["matched"]) for pair in scored),
        "stable_relative_tolerance": (
            STABLE_RELATIVE_TOLERANCE if expectation == "expected_stable" else None
        ),
        "pairs": pairs,
    }


def _flatten_metric_values(config: dict[str, object]) -> dict[str, tuple[float, str]]:
    values: dict[str, tuple[float, str]] = {
        "representative_count": (float(config["representative_count"]), "lower")
    }
    coherence = cast(dict[str, object], config["coherence"])["debiased_pairwise_cosine"]
    if coherence is not None:
        values["coherence"] = (float(coherence), "higher")
    suite = config.get("metric_suite")
    if type(suite) is not dict:
        return values
    metrics = cast(dict[str, object], suite)
    pairwise = cast(dict[str, object], metrics["pairwise"])
    if bool(pairwise["available"]):
        values["pairwise.median_angular_distance"] = (
            float(pairwise["median_angular_distance"]),
            "lower",
        )
        values["pairwise.p90_angular_distance"] = (
            float(pairwise["p90_angular_distance"]),
            "lower",
        )
    for temperature, value in cast(dict[str, object], metrics["kernel_effective_number"]).items():
        values[f"kernel_effective_number.tau_{temperature}"] = (float(value), "lower")
    spectrum = cast(dict[str, object], metrics["covariance_spectrum"])
    if bool(spectrum["available"]):
        for name in (
            "total_variance",
            "stable_rank",
            "renyi2_effective_rank",
            "shannon_effective_rank",
        ):
            values[f"covariance.{name}"] = (float(spectrum[name]), "lower")
    for beta, value in cast(dict[str, object], metrics["regularized_logdet"]).items():
        if value is not None:
            values[f"logdet.beta_{beta}"] = (float(value), "lower")
    for k, raw in cast(dict[str, object], metrics["knn"]).items():
        if type(raw) is not dict:
            continue
        item = cast(dict[str, object], raw)
        values[f"knn.k_{k}.median_distance"] = (
            float(item["median_kth_neighbor_distance"]),
            "lower",
        )
        values[f"knn.k_{k}.entropy_proxy"] = (
            float(item["kl_entropy_proxy_without_unit_ball_constant"]),
            "lower",
        )
    return values


def _clarity_pairs(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row["conversation_id"]).startswith("c0"):
            grouped[str(row["conversation_id"])].append(row)
    result = []
    for conversation_id, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: int(item["turn"]))
        if len(ordered) != 2:
            continue
        result.append(
            {
                "conversation_id": conversation_id,
                "turns": [
                    {
                        "turn": item["turn"],
                        "user_message": item["user_message"],
                        "eligible_count": item["eligible_count"],
                        "absolute_candidate_counts": item["absolute_candidate_counts"],
                        "top5_delta_candidate_counts": item["top5_delta_candidate_counts"],
                    }
                    for item in ordered
                ],
            }
        )
    return result


def _transition_summary(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["suite_id"]), str(row["conversation_id"]))].append(row)
    pairs = []
    for items in grouped.values():
        ordered = sorted(items, key=lambda item: int(item["turn"]))
        pairs.extend(zip(ordered, ordered[1:], strict=False))
    return {
        "pair_count": len(pairs),
        "absolute_thresholds": {
            _key(threshold): _directions(pairs, field="absolute_candidate_counts", value=threshold)
            for threshold in ABSOLUTE_THRESHOLDS
        },
        "top5_mean_deltas": {
            _key(delta): _directions(pairs, field="top5_delta_candidate_counts", value=delta)
            for delta in TOP5_DELTAS
        },
    }


def _directions(
    pairs: Sequence[tuple[dict[str, object], dict[str, object]]],
    *,
    field: str,
    value: float,
) -> dict[str, int]:
    result = {"decrease": 0, "same": 0, "increase": 0}
    for previous, current in pairs:
        left = int(cast(dict[str, object], previous[field])[_key(value)])
        right = int(cast(dict[str, object], current[field])[_key(value)])
        direction = "decrease" if right < left else "increase" if right > left else "same"
        result[direction] += 1
    return result


def _distribution(values: Sequence[int | float]) -> dict[str, int | float]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": _number(float(np.min(array))),
        "p10": _number(float(np.quantile(array, 0.1))),
        "median": _number(float(np.median(array))),
        "mean": float(np.mean(array)),
        "p90": _number(float(np.quantile(array, 0.9))),
        "max": _number(float(np.max(array))),
    }


def _number(value: float) -> int | float:
    rounded = round(value)
    return int(rounded) if math.isclose(value, rounded, abs_tol=1e-12) else value


def _key(value: float) -> str:
    return f"{value:.3f}"


def _quantile_key(value: float) -> str:
    return f"p{value * 100:g}"


def _mapping(mapping: dict[str, object], key: str) -> dict[str, object]:
    value = mapping.get(key)
    if type(value) is not dict:
        raise ValueError(f"{key} must be an object")
    return cast(dict[str, object], value)


def _string(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)
    if type(value) is not str:
        raise ValueError(f"{key} must be a string")
    return value


def _optional_string(mapping: dict[str, object], key: str, *, default: str) -> str:
    value = mapping.get(key)
    if value is None:
        return default
    if type(value) is not str:
        raise ValueError(f"{key} must be a string or null")
    return value


def _string_tuple(mapping: dict[str, object], key: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if value is None:
        return ()
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError(f"{key} must be an array of strings")
    return tuple(cast(list[str], value))


def _row_tags(row: dict[str, object]) -> frozenset[str]:
    value = row.get("tags")
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError("row.tags must be an array of strings")
    return frozenset(cast(list[str], value))


def _integer(mapping: dict[str, object], key: str) -> int:
    value = mapping.get(key)
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    return value


def _render_markdown(report: dict[str, object]) -> str:
    runtime = cast(dict[str, object], report["runtime"])
    summary = cast(dict[str, object], report["summary"])
    all_rows = cast(dict[str, object], summary["all"])
    absolute = cast(dict[str, object], all_rows["absolute_thresholds"])
    relative = cast(dict[str, object], all_rows["top5_mean_deltas"])
    pairwise = cast(dict[str, object], summary["pairwise_sample"])
    full_merge = report.get("full_merge_evaluation")
    lines = [
        "# Semantic support threshold analysis v0",
        "",
        "This report measures query-to-catalog score distributions before choosing a keep threshold.",
        "It uses saved accepted QU state, reproduces the existing hard mask, and scores all 50,000",
        "catalog vectors. No threshold in this report is a frozen runtime decision.",
        "",
        "## Runtime",
        "",
        f"- Device: `{runtime['device']}` ({runtime['gpu_name']})",
        f"- Turns: `{runtime['query_count']}`",
        f"- Products: `{runtime['product_count']}`",
        f"- Full score matrix time: `{float(runtime['full_catalog_score_seconds']):.4f}s`",
        "",
        "## Absolute cosine sweep",
        "",
        "| threshold | empty | median candidates | p90 candidates | >4,096 | >10,000 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for threshold in ABSOLUTE_THRESHOLDS:
        item = cast(dict[str, object], absolute[_key(threshold)])
        lines.append(
            f"| {threshold:.3f} | {item['empty_count']} | {item['median']} | "
            f"{item['p90']} | {item['gt_4096_count']} | {item['gt_10000_count']} |"
        )
    lines.extend(
        [
            "",
            "## Relative sweep: eligible Top-5 mean minus delta",
            "",
            "| delta | empty | median candidates | p90 candidates | >4,096 | >10,000 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for delta in TOP5_DELTAS:
        item = cast(dict[str, object], relative[_key(delta)])
        lines.append(
            f"| {delta:.3f} | {item['empty_count']} | {item['median']} | "
            f"{item['p90']} | {item['gt_4096_count']} | {item['gt_10000_count']} |"
        )
    lines.extend(
        [
            "",
            f"## Pairwise sample at Top-5 mean minus {PAIRWISE_KEEP_DELTA:.3f}",
            "",
            "Each turn samples at most 1,536 candidates across the complete retained score range.",
            "Pair fractions use at most 100,000 deterministic upper-triangle pairs; nearest-neighbor",
            "fractions use the exact matrix within that candidate sample.",
            "",
            "| merge threshold | median pair fraction | median candidates with a neighbor | p90 candidates with a neighbor |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    merge_rows = cast(dict[str, object], pairwise["merge_thresholds"])
    for threshold in MERGE_THRESHOLDS:
        item = cast(dict[str, object], merge_rows[_key(threshold)])
        pair_fraction = cast(dict[str, object], item["sampled_pair_fraction"])
        neighbor_fraction = cast(dict[str, object], item["candidate_with_neighbor_fraction"])
        lines.append(
            f"| {threshold:.3f} | {float(pair_fraction['median']):.6f} | "
            f"{float(neighbor_fraction['median']):.6f} | "
            f"{float(neighbor_fraction['p90']):.6f} |"
        )
    if type(full_merge) is dict:
        merge_summary = cast(dict[str, object], cast(dict[str, object], full_merge)["summary"])
        configurations = cast(dict[str, object], merge_summary["configurations"])
        lines.extend(
            [
                "",
                "## Full retained-set greedy merge",
                "",
                "| config | median candidates | median representatives | median reduction | median coherence | clarity specific higher |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for config_key, raw_item in configurations.items():
            item = cast(dict[str, object], raw_item)
            coherence = cast(dict[str, object] | None, item["coherence"])
            clarity = cast(dict[str, object], item["clarity_pairs"])
            coherence_median = "n/a" if coherence is None else f"{float(coherence['median']):.6f}"
            lines.append(
                f"| `{config_key}` | "
                f"{cast(dict[str, object], item['candidate_count'])['median']} | "
                f"{cast(dict[str, object], item['representative_count'])['median']} | "
                f"{float(cast(dict[str, object], item['reduction_fraction'])['median']):.6f} | "
                f"{coherence_median} | "
                f"{clarity['specific_higher_count']}/{clarity['available_count']} |"
            )
        metric_comparison = merge_summary.get("metric_comparison")
        if type(metric_comparison) is dict:
            primary_key = _merge_key(0.1, None)
            primary = cast(dict[str, object], metric_comparison[primary_key])
            metric_rows = cast(dict[str, object], primary["metrics"])
            lines.extend(
                [
                    "",
                    f"## Dispersion metric comparison on `{primary_key}`",
                    "",
                    "| metric | clear direction | expected narrower | expected broader | expected stable | simulator first-to-last clearer |",
                    "| --- | --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for metric_id, raw_metric in metric_rows.items():
                metric = cast(dict[str, object], raw_metric)
                simulator = cast(dict[str, object], metric["simulator_first_to_last_observation"])
                expected = cast(dict[str, object], metric["natural_expected_transitions"])
                narrower = cast(dict[str, object], expected["expected_narrower"])
                broader = cast(dict[str, object], expected["expected_broader"])
                stable = cast(dict[str, object], expected["expected_stable"])
                lines.append(
                    f"| `{metric_id}` | {metric['clear_direction']} | "
                    f"{narrower['matched_count']}/{narrower['scored_pair_count']} | "
                    f"{broader['matched_count']}/{broader['scored_pair_count']} | "
                    f"{stable['matched_count']}/{stable['scored_pair_count']} | "
                    f"{simulator['clearer_count']}/{simulator['pair_count']} |"
                )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This stage only identifies computationally and semantically plausible keep-threshold",
            "regions. Merge-threshold and post-merge dispersion sweeps must be run on the observed",
            "candidate-size region; they should not be inferred from Top-K behavior.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
