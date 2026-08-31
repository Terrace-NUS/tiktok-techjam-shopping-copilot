#!/usr/bin/env python3
"""Build reproducible public-likeness benchmark gradients from the frozen catalog.

The builder never changes ``catalog.jsonl``. It treats the 200 public targets as
examples of the organizer's target-selection distribution, scores every other
product using visible metadata, and emits nested Top-1k/2k/4k suites plus a cold
control. Scenario, difficulty-label, and aggregate-profile proportions are copied
exactly from the public set.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for source_path in (ROOT, ROOT / "src"):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from evaluator.local_evaluator import coarse_category, searchable_text  # noqa: E402

SCHEMA = "shopping-copilot/public-like-benchmark/v1"
SEED = 20260831
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

FEATURE_NAMES = (
    "log_reviews",
    "average_rating",
    "price_known",
    "log_price",
    "log_feature_count",
    "log_searchable_tokens",
    "log_title_tokens",
    "category_depth",
    "release_year_known",
    "release_year",
)

# Popularity is the clearest public-selection signal.  The remaining weights keep
# the matched suite close in information completeness without letting long catalog
# prose dominate the distance.
FEATURE_WEIGHTS = np.asarray(
    (3.0, 0.35, 1.25, 0.65, 0.80, 0.80, 0.30, 0.25, 0.40, 0.25),
    dtype=np.float32,
)
COARSE_CATEGORY_PENALTY = 1.50
LEAF_CATEGORY_PENALTY = 0.45
MATCHES_PER_PUBLIC_TARGET = 4
NEIGHBOR_LIST_SIZE = 2_048


@dataclass(frozen=True, slots=True)
class ProductFeatures:
    parent_asin: str
    values: tuple[float, ...]
    coarse_category: str
    leaf_category: str
    review_count: int
    price_known: bool
    feature_count: int
    searchable_tokens: int


def main() -> int:
    args = _parse_args()
    products = _load_jsonl(args.catalog)
    public_sessions = _load_jsonl(args.public_set)
    by_asin = {str(row["parent_asin"]): row for row in products}
    public_ids = [
        str(cast(dict[str, object], row["ground_truth"])["parent_asin"])
        for row in public_sessions
    ]
    if len(products) != 50_000 or len(by_asin) != 50_000:
        raise ValueError("expected exactly 50,000 unique catalog products")
    if len(public_sessions) != 200 or len(set(public_ids)) != 200:
        raise ValueError("expected exactly 200 sessions with unique public targets")
    if missing := sorted(set(public_ids) - set(by_asin)):
        raise ValueError(f"public targets missing from catalog: {missing}")

    print("deriving visible metadata features...", flush=True)
    raw_rows = [_raw_features(row) for row in products]
    row_by_asin = {row.parent_asin: row for row in raw_rows}
    public_rows = [row_by_asin[parent_asin] for parent_asin in public_ids]
    public_id_set = set(public_ids)
    candidate_rows = [row for row in raw_rows if row.parent_asin not in public_id_set]

    public_matrix = np.asarray([row.values for row in public_rows], dtype=np.float32)
    candidate_matrix = np.asarray([row.values for row in candidate_rows], dtype=np.float32)
    public_matrix, candidate_matrix, scaling = _robust_scale(public_matrix, candidate_matrix)

    print("matching four unique catalog targets to every public target...", flush=True)
    distances = _distance_matrix(
        public_rows=public_rows,
        candidate_rows=candidate_rows,
        public_matrix=public_matrix,
        candidate_matrix=candidate_matrix,
    )
    nearest_public_cost = np.partition(distances, kth=4, axis=0)[:5].mean(axis=0)
    public_neighbor_distances = _distance_matrix(
        public_rows=public_rows,
        candidate_rows=public_rows,
        public_matrix=public_matrix,
        candidate_matrix=public_matrix,
    )
    np.fill_diagonal(public_neighbor_distances, np.inf)
    public_leave_one_out_cost = np.partition(public_neighbor_distances, kth=4, axis=0)[:5].mean(
        axis=0
    )
    all_likeness = _descending_quantiles(
        np.concatenate((public_leave_one_out_cost, nearest_public_cost))
    )
    public_likeness = all_likeness[: len(public_rows)]
    candidate_likeness = all_likeness[len(public_rows) :]
    ranked_candidate_indices = sorted(
        range(len(candidate_rows)),
        key=lambda index: (
            float(nearest_public_cost[index]),
            candidate_rows[index].parent_asin,
        ),
    )

    gradient_sizes = (1_000, 2_000, 4_000)
    gradient_rows: dict[int, list[ProductFeatures]] = {}
    gradient_sessions: dict[int, list[dict[str, object]]] = {}
    for total_size in gradient_sizes:
        added_count = total_size - len(public_sessions)
        rows = [candidate_rows[index] for index in ranked_candidate_indices[:added_count]]
        gradient_rows[total_size] = rows
        gradient_sessions[total_size] = _sessions_for_targets(
            public_sessions,
            rows,
            prefix=f"public_like_top_{total_size}",
            include_public=True,
        )

    cold_rows = [candidate_rows[index] for index in ranked_candidate_indices[-1_000:]]
    cold_sessions = _sessions_for_targets(
        public_sessions,
        cold_rows,
        prefix="cold_control",
        include_public=False,
    )

    # Keep a per-public nearest-neighbor match set as an audit view.  The actual
    # gradient benchmark below is selected globally by the five-neighbor likeness
    # score, as requested, and does not use these assignments.
    assignments = _assign_unique_matches(distances)
    selected_indices = [candidate_index for row in assignments for candidate_index in row]
    if len(selected_indices) != 800 or len(set(selected_indices)) != 800:
        raise AssertionError("matcher must produce exactly 800 unique targets")

    pseudo_sessions: list[dict[str, object]] = []
    match_manifest: list[dict[str, object]] = []
    sequence = 0
    for public_index, candidate_indices in enumerate(assignments):
        source = public_sessions[public_index]
        source_target = public_ids[public_index]
        for replica, candidate_index in enumerate(candidate_indices, start=1):
            sequence += 1
            candidate = candidate_rows[candidate_index]
            pseudo_sessions.append(
                _clone_session(
                    source,
                    sample_id=f"public_like_{sequence:04d}",
                    target=candidate.parent_asin,
                )
            )
            match_manifest.append(
                {
                    "sample_id": f"public_like_{sequence:04d}",
                    "target_parent_asin": candidate.parent_asin,
                    "source_public_sample_id": source["sample_id"],
                    "source_public_target": source_target,
                    "replica": replica,
                    "match_cost": float(distances[public_index, candidate_index]),
                    "same_coarse_category": (
                        candidate.coarse_category == public_rows[public_index].coarse_category
                    ),
                    "same_leaf_category": (
                        candidate.leaf_category == public_rows[public_index].leaf_category
                    ),
                }
            )

    # A target-free control is useful when a ranking change improves the learned
    # public prior but damages ordinary catalog retrieval.
    selected_asins = {candidate_rows[index].parent_asin for index in selected_indices}
    control_pool = [
        row
        for row in candidate_rows
        if row.parent_asin not in selected_asins
    ]
    control_pool.sort(key=lambda row: _stable_key(row.parent_asin, args.seed))
    control_rows = control_pool[:200]
    control_sessions = [
        _clone_session(
            public_sessions[index],
            sample_id=f"uniform_control_{index + 1:04d}",
            target=row.parent_asin,
        )
        for index, row in enumerate(control_rows)
    ]

    selected_rows = [candidate_rows[index] for index in selected_indices]
    analysis = _analysis(
        public_rows=public_rows,
        selected_rows=selected_rows,
        control_rows=control_rows,
        all_rows=raw_rows,
        pseudo_sessions=pseudo_sessions,
        match_manifest=match_manifest,
        scaling=scaling,
        gradient_rows=gradient_rows,
        cold_rows=cold_rows,
        gradient_sessions=gradient_sessions,
        cold_sessions=cold_sessions,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    for total_size in gradient_sizes:
        _write_jsonl(
            args.output / f"public-like-top-{total_size}.jsonl",
            gradient_sessions[total_size],
        )
    _write_jsonl(args.output / "cold-control-1000.jsonl", cold_sessions)
    _write_jsonl(args.output / "matched-pseudo-private-800.jsonl", pseudo_sessions)
    _write_jsonl(
        args.output / "matched-expanded-public-like-1000.jsonl",
        [*public_sessions, *pseudo_sessions],
    )
    _write_jsonl(args.output / "uniform-control-200.jsonl", control_sessions)
    _write_jsonl(args.output / "target-matches.jsonl", match_manifest)
    _write_json(args.output / "analysis.json", analysis)
    (args.output / "report.md").write_text(_render_report(analysis), encoding="utf-8")

    # Quantile scores are diagnostic selection priors, not probabilities.  Keeping
    # them in a sidecar makes it impossible to accidentally mutate the catalog.
    prior_rows = [
        {
            "parent_asin": row.parent_asin,
            "public_likeness_quantile": float(score),
            "nearest_public_cost": float(cost),
        }
        for row, score, cost in zip(
            candidate_rows,
            candidate_likeness,
            nearest_public_cost,
            strict=True,
        )
    ]
    prior_rows.extend(
        {
            "parent_asin": parent_asin,
            "public_likeness_quantile": float(score),
            "nearest_public_cost": float(cost),
        }
        for parent_asin, score, cost in zip(
            public_ids,
            public_likeness,
            public_leave_one_out_cost,
            strict=True,
        )
    )
    prior_rows.sort(key=lambda row: cast(str, row["parent_asin"]))
    _write_jsonl(args.output / "catalog-selection-prior.jsonl", prior_rows)

    print(f"wrote public-like benchmark to {args.output.resolve()}", flush=True)
    return 0


def _raw_features(product: dict[str, object]) -> ProductFeatures:
    price = _number(product.get("price"))
    details = product.get("details")
    details_dict = details if type(details) is dict else {}
    release_year = _release_year(cast(dict[object, object], details_dict))
    feature_count = len(product.get("features") or []) if type(product.get("features")) is list else 0
    searchable_tokens = len(TOKEN_RE.findall(searchable_text(product)))
    title_tokens = len(TOKEN_RE.findall(str(product.get("title") or "")))
    categories = product.get("categories") if type(product.get("categories")) is list else []
    values = (
        math.log1p(max(0.0, _number(product.get("rating_number")) or 0.0)),
        _number(product.get("average_rating")) or 0.0,
        float(price is not None),
        math.nan if price is None else math.log1p(max(0.0, price)),
        math.log1p(feature_count),
        math.log1p(searchable_tokens),
        math.log1p(title_tokens),
        float(len(categories)),
        float(release_year is not None),
        math.nan if release_year is None else float(release_year),
    )
    return ProductFeatures(
        parent_asin=str(product["parent_asin"]),
        values=values,
        coarse_category=coarse_category([str(item) for item in categories]),
        leaf_category=(str(categories[-1]).casefold() if categories else "clothing item"),
        review_count=int(_number(product.get("rating_number")) or 0),
        price_known=price is not None,
        feature_count=feature_count,
        searchable_tokens=searchable_tokens,
    )


def _robust_scale(
    public: np.ndarray,
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    combined = np.concatenate((public, candidates), axis=0).astype(np.float64)
    fill_values: list[float] = []
    centers: list[float] = []
    scales: list[float] = []
    for column in range(combined.shape[1]):
        values = combined[:, column]
        finite = np.isfinite(values)
        fill = float(np.median(values[finite])) if finite.any() else 0.0
        values[~finite] = fill
        center = float(np.median(values))
        q25, q75 = np.quantile(values, (0.25, 0.75))
        scale = float(q75 - q25)
        if scale < 1e-6:
            scale = float(np.std(values))
        if scale < 1e-6:
            scale = 1.0
        fill_values.append(fill)
        centers.append(center)
        scales.append(scale)
    scaled = (combined - np.asarray(centers)) / np.asarray(scales)
    public_scaled = scaled[: len(public)].astype(np.float32)
    candidate_scaled = scaled[len(public) :].astype(np.float32)
    return public_scaled, candidate_scaled, {
        "feature_names": list(FEATURE_NAMES),
        "weights": [float(value) for value in FEATURE_WEIGHTS],
        "fill_values": fill_values,
        "centers": centers,
        "scales": scales,
        "coarse_category_penalty": COARSE_CATEGORY_PENALTY,
        "leaf_category_penalty": LEAF_CATEGORY_PENALTY,
    }


def _distance_matrix(
    *,
    public_rows: list[ProductFeatures],
    candidate_rows: list[ProductFeatures],
    public_matrix: np.ndarray,
    candidate_matrix: np.ndarray,
) -> np.ndarray:
    result = np.empty((len(public_rows), len(candidate_rows)), dtype=np.float32)
    for public_index, public_row in enumerate(public_rows):
        delta = candidate_matrix - public_matrix[public_index]
        costs = np.sum(delta * delta * FEATURE_WEIGHTS, axis=1)
        costs += np.asarray(
            [
                0.0
                if row.coarse_category == public_row.coarse_category
                else COARSE_CATEGORY_PENALTY
                for row in candidate_rows
            ],
            dtype=np.float32,
        )
        costs += np.asarray(
            [
                0.0
                if row.leaf_category == public_row.leaf_category
                else LEAF_CATEGORY_PENALTY
                for row in candidate_rows
            ],
            dtype=np.float32,
        )
        result[public_index] = costs
    return result


def _assign_unique_matches(distances: np.ndarray) -> list[list[int]]:
    neighbor_count = min(NEIGHBOR_LIST_SIZE, distances.shape[1])
    neighbor_lists: list[np.ndarray] = []
    hardness: list[float] = []
    for row in distances:
        indices = np.argpartition(row, neighbor_count - 1)[:neighbor_count]
        indices = indices[np.argsort(row[indices], kind="stable")]
        neighbor_lists.append(indices)
        hardness.append(float(row[indices[min(31, len(indices) - 1)]]))

    # Hard-to-match public targets choose first.  Each round gives every source one
    # match before any source receives its next match.
    source_order = sorted(range(len(neighbor_lists)), key=lambda item: (-hardness[item], item))
    assigned: list[list[int]] = [[] for _ in neighbor_lists]
    used: set[int] = set()
    for _ in range(MATCHES_PER_PUBLIC_TARGET):
        for source in source_order:
            match = next(
                (int(candidate) for candidate in neighbor_lists[source] if int(candidate) not in used),
                None,
            )
            if match is None:
                # This should never happen with a 2,048-neighbor list and 800
                # assignments, but the full row is a safe deterministic fallback.
                match = next(
                    int(candidate)
                    for candidate in np.argsort(distances[source], kind="stable")
                    if int(candidate) not in used
                )
            assigned[source].append(match)
            used.add(match)
    return assigned


def _clone_session(
    source: dict[str, object],
    *,
    sample_id: str,
    target: str,
) -> dict[str, object]:
    clone = copy.deepcopy(source)
    clone["sample_id"] = sample_id
    clone["ground_truth"] = {"parent_asin": target}
    clone.pop("intent_card", None)
    clone.pop("behavior", None)
    return clone


def _sessions_for_targets(
    public_sessions: list[dict[str, object]],
    targets: list[ProductFeatures],
    *,
    prefix: str,
    include_public: bool,
) -> list[dict[str, object]]:
    if len(targets) % len(public_sessions) != 0:
        raise ValueError("gradient additions must be a whole multiple of the public 200")
    result = copy.deepcopy(public_sessions) if include_public else []
    for index, target in enumerate(targets):
        template = public_sessions[index % len(public_sessions)]
        result.append(
            _clone_session(
                template,
                sample_id=f"{prefix}_{index + 1:04d}",
                target=target.parent_asin,
            )
        )
    return result


def _analysis(
    *,
    public_rows: list[ProductFeatures],
    selected_rows: list[ProductFeatures],
    control_rows: list[ProductFeatures],
    all_rows: list[ProductFeatures],
    pseudo_sessions: list[dict[str, object]],
    match_manifest: list[dict[str, object]],
    scaling: dict[str, object],
    gradient_rows: dict[int, list[ProductFeatures]],
    cold_rows: list[ProductFeatures],
    gradient_sessions: dict[int, list[dict[str, object]]],
    cold_sessions: list[dict[str, object]],
) -> dict[str, object]:
    cohorts = {
        "public_200": _cohort_summary(public_rows),
        "matched_pseudo_private_800": _cohort_summary(selected_rows),
        **{
            f"public_like_top_{size}": _cohort_summary([*public_rows, *rows])
            for size, rows in gradient_rows.items()
        },
        "cold_control_1000": _cohort_summary(cold_rows),
        "uniform_control_200": _cohort_summary(control_rows),
        "catalog_50000": _cohort_summary(all_rows),
    }
    public_categories = Counter(row.coarse_category for row in public_rows)
    pseudo_categories = Counter(row.coarse_category for row in selected_rows)
    category_tv = _total_variation(public_categories, pseudo_categories)
    return {
        "schema": SCHEMA,
        "construction": {
            "catalog_products": len(all_rows),
            "public_targets": len(public_rows),
            "pseudo_private_targets": len(selected_rows),
            "uniform_control_targets": len(control_rows),
            "target_uniqueness": len({row.parent_asin for row in selected_rows}),
            "public_overlap": len(
                {row.parent_asin for row in public_rows}
                & {row.parent_asin for row in selected_rows}
            ),
            "matching": (
                "four unique nearest visible-metadata neighbors per public target, "
                "with coarse/leaf category penalties"
            ),
            "scenario_counts": dict(
                sorted(Counter(str(row["scenario_type"]) for row in pseudo_sessions).items())
            ),
            "seed": SEED,
            "gradient": {
                str(size): {
                    "total_sessions": len(gradient_sessions[size]),
                    "public_sessions": len(public_rows),
                    "new_unique_targets": len(rows),
                    "scenario_counts": dict(
                        sorted(
                            Counter(
                                str(row["scenario_type"])
                                for row in gradient_sessions[size]
                            ).items()
                        )
                    ),
                }
                for size, rows in gradient_rows.items()
            },
            "cold_control": {
                "total_sessions": len(cold_sessions),
                "public_overlap": len(
                    {row.parent_asin for row in public_rows}
                    & {row.parent_asin for row in cold_rows}
                ),
                "scenario_counts": dict(
                    sorted(Counter(str(row["scenario_type"]) for row in cold_sessions).items())
                ),
            },
        },
        "cohorts": cohorts,
        "matching_quality": {
            "mean_cost": float(np.mean([row["match_cost"] for row in match_manifest])),
            "median_cost": float(np.median([row["match_cost"] for row in match_manifest])),
            "p90_cost": float(np.quantile([row["match_cost"] for row in match_manifest], 0.9)),
            "same_coarse_category_rate": float(
                np.mean([row["same_coarse_category"] for row in match_manifest])
            ),
            "same_leaf_category_rate": float(
                np.mean([row["same_leaf_category"] for row in match_manifest])
            ),
            "coarse_category_total_variation": category_tv,
        },
        "scaling": scaling,
        "limitations": [
            "The private 800 are unavailable; public-like is a hypothesis, not a reconstruction.",
            "Unique non-public inventory cannot reproduce the public set's hottest review-count tail.",
            "The copied difficulty_bucket preserves label proportions but is not recomputed for new targets.",
            "The selection-prior quantile is a diagnostic ranking feature, not a calibrated probability.",
            "Top-N tiers deliberately test public-distribution overfitting and are not unbiased private-set estimates.",
        ],
    }


def _cohort_summary(rows: list[ProductFeatures]) -> dict[str, object]:
    reviews = np.asarray([row.review_count for row in rows], dtype=np.float64)
    return {
        "count": len(rows),
        "review_mean": float(reviews.mean()),
        "review_median": float(np.median(reviews)),
        "review_p25": float(np.quantile(reviews, 0.25)),
        "review_p75": float(np.quantile(reviews, 0.75)),
        "review_ge_1000_rate": float(np.mean(reviews >= 1_000)),
        "review_ge_5000_rate": float(np.mean(reviews >= 5_000)),
        "review_ge_10000_rate": float(np.mean(reviews >= 10_000)),
        "price_known_rate": float(np.mean([row.price_known for row in rows])),
        "feature_count_mean": float(np.mean([row.feature_count for row in rows])),
        "searchable_tokens_mean": float(np.mean([row.searchable_tokens for row in rows])),
        "unique_coarse_categories": len({row.coarse_category for row in rows}),
    }


def _total_variation(left: Counter[str], right: Counter[str]) -> float:
    left_n = sum(left.values())
    right_n = sum(right.values())
    values = set(left) | set(right)
    return 0.5 * sum(abs(left[value] / left_n - right[value] / right_n) for value in values)


def _descending_quantiles(costs: np.ndarray) -> np.ndarray:
    order = np.argsort(costs, kind="stable")
    result = np.empty(len(costs), dtype=np.float32)
    if len(costs) == 1:
        result[0] = 1.0
        return result
    result[order] = np.linspace(1.0, 0.0, len(costs), dtype=np.float32)
    return result


def _render_report(payload: dict[str, object]) -> str:
    construction = cast(dict[str, object], payload["construction"])
    gradient = cast(dict[str, dict[str, object]], construction["gradient"])
    cold_control = cast(dict[str, object], construction["cold_control"])
    cohorts = cast(dict[str, dict[str, object]], payload["cohorts"])
    matching = cast(dict[str, object], payload["matching_quality"])
    lines = [
        "# Public-likeness gradient benchmark v1",
        "",
        "## What this is",
        "",
        "This is an internal stress set, not organizer data. Every non-public product is scored by its visible-metadata distance to the public 200. Nested Top-1k/2k/4k suites test progressively weaker public similarity, while a cold control tests the opposite tail.",
        "",
        "- Nesting: **Top-1k is a subset of Top-2k, which is a subset of Top-4k**",
        "- All three Top suites contain the unchanged public 200",
        f"- Top-1k scenario counts: `{json.dumps(gradient['1000']['scenario_counts'], sort_keys=True)}`",
        f"- Cold-control public overlap: **{cold_control['public_overlap']}**",
        "",
        "## Distribution comparison",
        "",
        "| Cohort | n | Mean reviews | Median reviews | >=1k | >=5k | >=10k | Price known | Mean features | Mean searchable tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "public_200",
        "public_like_top_1000",
        "public_like_top_2000",
        "public_like_top_4000",
        "cold_control_1000",
        "uniform_control_200",
        "catalog_50000",
    ):
        row = cohorts[name]
        lines.append(
            f"| {name} | {row['count']} | {float(cast(float, row['review_mean'])):,.1f} | "
            f"{float(cast(float, row['review_median'])):,.1f} | "
            f"{100 * float(cast(float, row['review_ge_1000_rate'])):.1f}% | "
            f"{100 * float(cast(float, row['review_ge_5000_rate'])):.1f}% | "
            f"{100 * float(cast(float, row['review_ge_10000_rate'])):.1f}% | "
            f"{100 * float(cast(float, row['price_known_rate'])):.1f}% | "
            f"{float(cast(float, row['feature_count_mean'])):.2f} | "
            f"{float(cast(float, row['searchable_tokens_mean'])):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Secondary nearest-match audit",
            "",
            f"- Same coarse category: **{100 * float(cast(float, matching['same_coarse_category_rate'])):.1f}%**",
            f"- Same leaf category: **{100 * float(cast(float, matching['same_leaf_category_rate'])):.1f}%**",
            f"- Public/pseudo coarse-category total variation: **{float(cast(float, matching['coarse_category_total_variation'])):.4f}**",
            f"- Median weighted match cost: **{float(cast(float, matching['median_cost'])):.4f}**",
            "",
            "## Files",
            "",
            "- `public-like-top-1000.jsonl`: public 200 plus the 800 most public-like non-public targets",
            "- `public-like-top-2000.jsonl`: public 200 plus the 1,800 most public-like non-public targets",
            "- `public-like-top-4000.jsonl`: public 200 plus the 3,800 most public-like non-public targets",
            "- `cold-control-1000.jsonl`: the 1,000 least public-like non-public targets",
            "- `matched-pseudo-private-800.jsonl`: four-neighbor-per-public audit set",
            "- `matched-expanded-public-like-1000.jsonl`: public 200 plus the matched audit set",
            "- `uniform-control-200.jsonl`: uniform anti-overfit control",
            "- `target-matches.jsonl`: auditable public-to-pseudo matching provenance",
            "- `catalog-selection-prior.jsonl`: target-likeness diagnostic sidecar",
            "- `analysis.json`: full construction metadata and statistics",
            "",
            "## Boundary",
            "",
            "The hidden 800 are unavailable, so this suite tests one explicit hypothesis: organizer targets resemble the visible public targets. Improvements should also be checked on the uniform control before using the selection prior in the toy ranker.",
            "",
        ]
    )
    return "\n".join(lines)


def _number(value: object) -> float | None:
    if type(value) in (int, float):
        number = float(cast(int | float, value))
        if math.isfinite(number):
            return number
    if type(value) is str:
        try:
            number = float(cast(str, value).replace("$", "").replace(",", "").strip())
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _release_year(details: dict[object, object]) -> int | None:
    for key, value in details.items():
        if "date first available" not in str(key).casefold():
            continue
        if match := YEAR_RE.search(str(value)):
            return int(match.group(1))
    return None


def _stable_key(parent_asin: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{parent_asin}".encode()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.jsonl")
    parser.add_argument("--public-set", type=Path, default=ROOT / "data/public_set.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/benchmark/public-like-gradient-v1",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
