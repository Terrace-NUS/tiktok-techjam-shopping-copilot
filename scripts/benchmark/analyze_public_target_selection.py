#!/usr/bin/env python3
"""Compare the 200 public target products with the remaining frozen catalog."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for source_path in (ROOT, SRC):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from evaluator.local_evaluator import (  # noqa: E402
    coarse_category,
    intent_card,
    searchable_text,
)
from shopping_copilot.application.toy_simulator.catalog import (  # noqa: E402
    CatalogIndex,
    normalize_phrase,
)

REPORT_SCHEMA = "shopping-copilot/public-target-selection-analysis/v1"
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
PRICE_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?")
BOOTSTRAP_DRAWS = 4_000
RANDOM_SEED = 20260831

METRIC_LABELS = {
    "rating_number": "review count",
    "log_rating_number": "log(1 + review count)",
    "average_rating": "average rating",
    "price_known": "price present",
    "price": "price when present",
    "store_known": "store present",
    "store_frequency": "products from same store",
    "title_tokens": "title token count",
    "feature_count": "feature bullet count",
    "detail_count": "detail field count",
    "description_count": "description block count",
    "searchable_tokens": "all searchable token count",
    "category_depth": "category path depth",
    "coarse_category_frequency": "coarse-category catalog size",
    "leaf_category_frequency": "leaf-category catalog size",
    "intent_constraint_count": "simulator intent-card fact count",
    "first_constraint_posting": "products matching first disclosed fact",
    "rarest_constraint_posting": "products matching rarest disclosed fact",
    "intent_signature_collision": "products sharing complete intent card",
    "release_year_known": "Date First Available present",
    "release_year": "Date First Available year",
}

PREDICTOR_METRICS = (
    "log_rating_number",
    "average_rating",
    "price_known",
    "price",
    "store_frequency",
    "title_tokens",
    "feature_count",
    "detail_count",
    "description_count",
    "searchable_tokens",
    "category_depth",
    "coarse_category_frequency",
    "leaf_category_frequency",
    "intent_constraint_count",
    "first_constraint_posting",
    "rarest_constraint_posting",
    "intent_signature_collision",
    "release_year_known",
    "release_year",
)


def main() -> int:
    args = _parse_args()
    products = _load_jsonl(args.catalog)
    samples = _load_jsonl(args.public_set)
    by_asin = {str(item["parent_asin"]): item for item in products}
    public_ids = {
        str(cast(dict[str, object], item["ground_truth"])["parent_asin"]) for item in samples
    }
    if len(products) != 50_000 or len(by_asin) != 50_000:
        raise ValueError("expected exactly 50,000 unique catalog products")
    if len(samples) != 200 or len(public_ids) != 200:
        raise ValueError("expected exactly 200 unique public targets")
    missing = sorted(public_ids - set(by_asin))
    if missing:
        raise ValueError(f"public targets missing from catalog: {missing}")

    print("building toy retrieval evidence index...", flush=True)
    retrieval_index = CatalogIndex(args.catalog)
    print("deriving catalog features and simulator intent cards...", flush=True)
    features, categories, leaf_categories = _feature_rows(products, retrieval_index)
    public_mask = np.asarray(
        [str(product["parent_asin"]) in public_ids for product in products],
        dtype=bool,
    )

    comparisons = {
        name: _compare_metric(
            values=np.asarray([row[name] for row in features], dtype=np.float64),
            public_mask=public_mask,
            categories=categories,
            seed=RANDOM_SEED + offset,
        )
        for offset, name in enumerate(METRIC_LABELS)
    }
    category_analysis = _categorical_comparison(
        categories,
        public_mask,
        seed=RANDOM_SEED + 100,
    )
    leaf_analysis = _categorical_comparison(
        leaf_categories,
        public_mask,
        seed=RANDOM_SEED + 101,
    )
    scenario_analysis = _scenario_summary(samples, by_asin, features, products)
    popularity_analysis = _popularity_selection_analysis(
        features=features,
        public_mask=public_mask,
    )
    predictability = _selection_predictability(
        features=features,
        categories=categories,
        public_mask=public_mask,
    )

    payload: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "inputs": {
            "catalog": str(args.catalog.resolve()),
            "public_set": str(args.public_set.resolve()),
            "catalog_product_count": len(products),
            "public_target_count": int(public_mask.sum()),
            "non_public_catalog_count": int((~public_mask).sum()),
        },
        "scope_note": (
            "The private 800 targets are unavailable. This compares the known public 200 "
            "against the other 49,800 catalog products; it cannot distinguish a public/private "
            "split effect from the upstream 1,000-session selection mechanism."
        ),
        "official_sampling_evidence": {
            "source": "README.md Data Source",
            "statement": (
                "Sessions are sampled deterministically from the official Clothing 5-core "
                "leave-last-out split and joined to the frozen catalog."
            ),
            "interpretation": (
                "Targets originate from held-out purchase interactions, not a uniform draw "
                "over catalog product IDs."
            ),
        },
        "numeric_comparisons": comparisons,
        "coarse_category_comparison": category_analysis,
        "leaf_category_comparison": leaf_analysis,
        "popularity_selection_analysis": popularity_analysis,
        "scenario_summary": scenario_analysis,
        "selection_predictability": predictability,
        "method": {
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_sample_size": "number of finite public values for each metric",
            "category_adjustment": (
                "Expected non-public mean after weighting each coarse category by the public "
                "category mix."
            ),
            "intent_card": "official evaluator.intent_card()",
            "posting_sizes": "current default toy specialist CatalogIndex",
            "random_seed": RANDOM_SEED,
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    _write_json(args.output / "analysis.json", payload)
    (args.output / "report.md").write_text(
        _render_report(payload),
        encoding="utf-8",
    )
    print(f"wrote analysis to {args.output.resolve()}", flush=True)
    return 0


def _feature_rows(
    products: list[dict[str, object]],
    retrieval_index: CatalogIndex,
) -> tuple[list[dict[str, float]], list[str], list[str]]:
    store_counts = Counter(_store(product) for product in products if _store(product))
    categories = [_coarse_category(product) for product in products]
    category_counts = Counter(categories)
    leaf_categories = [_leaf_category(product) for product in products]
    leaf_counts = Counter(leaf_categories)

    cards = [intent_card(product) for product in products]
    signatures = [
        _intent_signature(category, card) for category, card in zip(categories, cards, strict=True)
    ]
    signature_counts = Counter(signatures)
    posting_size_cache: dict[str, int] = {}

    rows: list[dict[str, float]] = []
    for product, category, leaf, card, signature in zip(
        products,
        categories,
        leaf_categories,
        cards,
        signatures,
        strict=True,
    ):
        features = _list(product.get("features"))
        descriptions = _list(product.get("description"))
        details = _dict(product.get("details"))
        constraints = [
            *[str(item) for item in _list(card.get("hard_constraints"))],
            *[str(item) for item in _list(card.get("soft_preferences"))],
        ]
        posting_sizes = []
        for constraint in constraints:
            key = normalize_phrase(constraint)
            if key not in posting_size_cache:
                posting_size_cache[key] = len(retrieval_index.postings_for_constraint(constraint))
            posting_sizes.append(posting_size_cache[key])
        first_posting = posting_sizes[0] if posting_sizes else retrieval_index.size
        rarest_posting = min(posting_sizes) if posting_sizes else retrieval_index.size
        price = _price(product.get("price"))
        year = _release_year(details)
        store = _store(product)
        searchable = searchable_text(product)
        rating_number = float(product.get("rating_number") or 0)
        rows.append(
            {
                "rating_number": rating_number,
                "log_rating_number": math.log1p(rating_number),
                "average_rating": float(product.get("average_rating") or 0.0),
                "price_known": float(price is not None),
                "price": math.nan if price is None else price,
                "store_known": float(store is not None),
                "store_frequency": float(store_counts.get(store, 0)),
                "title_tokens": float(len(TOKEN_RE.findall(str(product.get("title") or "")))),
                "feature_count": float(len(features)),
                "detail_count": float(len(details)),
                "description_count": float(len(descriptions)),
                "searchable_tokens": float(len(TOKEN_RE.findall(searchable))),
                "category_depth": float(len(_list(product.get("categories")))),
                "coarse_category_frequency": float(category_counts[category]),
                "leaf_category_frequency": float(leaf_counts[leaf]),
                "intent_constraint_count": float(len(constraints)),
                "first_constraint_posting": float(first_posting),
                "rarest_constraint_posting": float(rarest_posting),
                "intent_signature_collision": float(signature_counts[signature]),
                "release_year_known": float(year is not None),
                "release_year": math.nan if year is None else float(year),
            }
        )
    return rows, categories, leaf_categories


def _compare_metric(
    *,
    values: np.ndarray,
    public_mask: np.ndarray,
    categories: list[str],
    seed: int,
) -> dict[str, object]:
    public = values[public_mask]
    rest = values[~public_mask]
    public = public[np.isfinite(public)]
    rest = rest[np.isfinite(rest)]
    if not len(public) or not len(rest):
        raise ValueError("metric has an empty cohort")

    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(rest), size=(BOOTSTRAP_DRAWS, len(public)))
    bootstrap_means = rest[sample_indices].mean(axis=1)
    public_mean = float(public.mean())
    rest_mean = float(rest.mean())
    lower_fraction = float(np.mean(bootstrap_means <= public_mean))
    empirical_two_sided = min(1.0, 2.0 * min(lower_fraction, 1.0 - lower_fraction))
    bootstrap_std = float(bootstrap_means.std(ddof=1))
    pooled_variance = (
        (len(public) - 1) * float(public.var(ddof=1)) + (len(rest) - 1) * float(rest.var(ddof=1))
    ) / max(1, len(public) + len(rest) - 2)
    standardized_difference = (
        0.0 if pooled_variance <= 0.0 else (public_mean - rest_mean) / math.sqrt(pooled_variance)
    )

    public_category_counts = Counter(
        category for category, selected in zip(categories, public_mask, strict=True) if selected
    )
    rest_by_category: dict[str, list[float]] = defaultdict(list)
    for value, category, selected in zip(values, categories, public_mask, strict=True):
        if not selected and math.isfinite(float(value)):
            rest_by_category[category].append(float(value))
    matched_total = 0.0
    matched_weight = 0
    for category, count in public_category_counts.items():
        cohort = rest_by_category.get(category, [])
        if cohort:
            matched_total += count * float(np.mean(cohort))
            matched_weight += count
    category_matched_mean = matched_total / matched_weight if matched_weight else math.nan

    return {
        "label": "",
        "public": _summary(public),
        "non_public": _summary(rest),
        "mean_difference": public_mean - rest_mean,
        "mean_ratio": None if rest_mean == 0.0 else public_mean / rest_mean,
        "standardized_mean_difference": standardized_difference,
        "uniform_random_200": {
            "mean": float(bootstrap_means.mean()),
            "p05": float(np.quantile(bootstrap_means, 0.05)),
            "p95": float(np.quantile(bootstrap_means, 0.95)),
            "public_percentile": 100.0 * lower_fraction,
            "z_score": (
                None if bootstrap_std == 0.0 else (public_mean - rest_mean) / bootstrap_std
            ),
            "empirical_two_sided_p": empirical_two_sided,
        },
        "category_matched_non_public_mean": category_matched_mean,
        "category_adjusted_difference": public_mean - category_matched_mean,
    }


def _summary(values: np.ndarray) -> dict[str, object]:
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _categorical_comparison(
    values: list[str],
    public_mask: np.ndarray,
    *,
    seed: int,
) -> dict[str, object]:
    public = Counter(value for value, selected in zip(values, public_mask, strict=True) if selected)
    rest = Counter(
        value for value, selected in zip(values, public_mask, strict=True) if not selected
    )
    public_n = int(public_mask.sum())
    rest_n = len(values) - public_n
    all_values = set(public) | set(rest)
    total_variation = 0.5 * sum(
        abs(public[value] / public_n - rest[value] / rest_n) for value in all_values
    )
    rest_values = np.asarray(
        [value for value, selected in zip(values, public_mask, strict=True) if not selected],
        dtype=object,
    )
    rng = np.random.default_rng(seed)
    bootstrap_tv = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = Counter(rest_values[rng.integers(0, len(rest_values), size=public_n)].tolist())
        bootstrap_tv.append(
            0.5
            * sum(
                abs(sampled[value] / public_n - rest[value] / rest_n)
                for value in set(sampled) | set(rest)
            )
        )
    bootstrap_tv_array = np.asarray(bootstrap_tv, dtype=np.float64)
    rows = []
    for value in all_values:
        public_rate = public[value] / public_n
        rest_rate = rest[value] / rest_n
        rows.append(
            {
                "value": value,
                "public_count": public[value],
                "public_rate": public_rate,
                "non_public_count": rest[value],
                "non_public_rate": rest_rate,
                "expected_public_count_under_uniform": public_n * rest_rate,
                "enrichment": None if rest_rate == 0.0 else public_rate / rest_rate,
            }
        )
    rows.sort(
        key=lambda row: (
            -cast(float, row["public_count"]),
            -abs(math.log(max(cast(float | None, row["enrichment"]) or 1.0, 1e-12))),
            cast(str, row["value"]),
        )
    )
    enriched = sorted(
        (row for row in rows if cast(int, row["public_count"]) >= 2),
        key=lambda row: -(cast(float | None, row["enrichment"]) or 0.0),
    )[:20]
    depleted = sorted(
        (row for row in rows if cast(float, row["expected_public_count_under_uniform"]) >= 1.0),
        key=lambda row: cast(float | None, row["enrichment"]) or 0.0,
    )[:20]
    return {
        "unique_public_values": len(public),
        "unique_non_public_values": len(rest),
        "total_variation_distance": total_variation,
        "uniform_random_200": {
            "mean": float(bootstrap_tv_array.mean()),
            "p05": float(np.quantile(bootstrap_tv_array, 0.05)),
            "p95": float(np.quantile(bootstrap_tv_array, 0.95)),
            "public_percentile": float(100.0 * np.mean(bootstrap_tv_array <= total_variation)),
        },
        "largest_public_categories": rows[:25],
        "most_enriched_with_at_least_two_public_targets": enriched,
        "most_depleted_with_expected_public_count_at_least_one": depleted,
    }


def _scenario_summary(
    samples: list[dict[str, object]],
    by_asin: dict[str, dict[str, object]],
    features: list[dict[str, float]],
    products: list[dict[str, object]],
) -> dict[str, object]:
    feature_by_asin = {
        str(product["parent_asin"]): row for product, row in zip(products, features, strict=True)
    }
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for sample in samples:
        target = str(cast(dict[str, object], sample["ground_truth"])["parent_asin"])
        if target not in by_asin:
            raise ValueError(f"unknown target {target}")
        grouped[str(sample["scenario_type"])].append(feature_by_asin[target])
    return {
        scenario: {
            "sample_count": len(rows),
            "mean_review_count": float(np.mean([row["rating_number"] for row in rows])),
            "price_known_rate": float(np.mean([row["price_known"] for row in rows])),
            "mean_first_constraint_posting": float(
                np.mean([row["first_constraint_posting"] for row in rows])
            ),
            "mean_intent_signature_collision": float(
                np.mean([row["intent_signature_collision"] for row in rows])
            ),
        }
        for scenario, rows in sorted(grouped.items())
    }


def _popularity_selection_analysis(
    *,
    features: list[dict[str, float]],
    public_mask: np.ndarray,
) -> dict[str, object]:
    from scipy.stats import hypergeom

    review_counts = np.asarray([row["rating_number"] for row in features], dtype=np.float64)
    public_n = int(public_mask.sum())
    catalog_n = len(features)
    bands = (
        ("0-4", 0, 5),
        ("5-9", 5, 10),
        ("10-49", 10, 50),
        ("50-249", 50, 250),
        ("250-999", 250, 1_000),
        ("1,000-4,999", 1_000, 5_000),
        ("5,000-9,999", 5_000, 10_000),
        ("10,000+", 10_000, math.inf),
    )
    band_rows = []
    for label, lower, upper in bands:
        mask = (review_counts >= lower) & (review_counts < upper)
        catalog_count = int(mask.sum())
        public_count = int((mask & public_mask).sum())
        catalog_rate = catalog_count / catalog_n
        public_rate = public_count / public_n
        band_rows.append(
            {
                "band": label,
                "catalog_count": catalog_count,
                "public_count": public_count,
                "catalog_rate": catalog_rate,
                "public_rate": public_rate,
                "public_target_rate_within_band": (
                    0.0 if catalog_count == 0 else public_count / catalog_count
                ),
                "enrichment_over_uniform_product_sampling": (
                    None if catalog_rate == 0.0 else public_rate / catalog_rate
                ),
            }
        )

    split_bounds = []
    for threshold in (3_000, 4_000, 5_000, 7_500, 10_000, 20_000, 50_000):
        eligible = review_counts >= threshold
        catalog_count = int(eligible.sum())
        public_count = int((eligible & public_mask).sum())
        if catalog_count > 1_000:
            continue
        upper_tail = float(hypergeom.sf(public_count - 1, 1_000, catalog_count, 200))
        split_bounds.append(
            {
                "review_threshold": threshold,
                "catalog_count": catalog_count,
                "public_count": public_count,
                "public_rate": public_count / 200,
                "expected_public_count_if_all_eligible_entered_pool_and_split_was_random": (
                    200 * catalog_count / 1_000
                ),
                "hypergeometric_upper_tail_probability": upper_tail,
                "maximum_private_count_if_targets_are_unique_and_disjoint": (
                    catalog_count - public_count
                ),
                "maximum_private_rate_if_targets_are_unique_and_disjoint": (
                    catalog_count - public_count
                )
                / 800,
            }
        )
    return {
        "bands": band_rows,
        "conditional_1000_unique_target_split_test": {
            "assumption": (
                "Exactly 1,000 distinct catalog targets were selected, public and private "
                "targets are disjoint, and the public 200 were then drawn uniformly from that pool."
            ),
            "method": (
                "For each threshold with at most 1,000 eligible catalog products, assume every "
                "eligible product entered the 1,000-target pool. This maximizes the probability "
                "of observing the public count, so the reported upper-tail probability is a "
                "conservative upper bound."
            ),
            "bounds": split_bounds,
        },
    }


def _selection_predictability(
    *,
    features: list[dict[str, float]],
    categories: list[str],
    public_mask: np.ndarray,
) -> dict[str, object]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"available": False}

    top_categories = [item for item, _ in Counter(categories).most_common(30)]
    feature_names = [*PREDICTOR_METRICS, *[f"category={item}" for item in top_categories]]
    numeric = np.asarray(
        [[row[name] for name in PREDICTOR_METRICS] for row in features],
        dtype=np.float64,
    )
    for column in range(numeric.shape[1]):
        values = numeric[:, column]
        finite = np.isfinite(values)
        replacement = float(np.median(values[finite])) if finite.any() else 0.0
        values[~finite] = replacement
    category_matrix = np.asarray(
        [[float(category == item) for item in top_categories] for category in categories],
        dtype=np.float64,
    )
    matrix = np.concatenate((numeric, category_matrix), axis=1)
    labels = public_mask.astype(np.int64)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    variants = {
        "numeric_only": numeric,
        "category_only": category_matrix,
        "combined": matrix,
    }
    variant_results: dict[str, dict[str, object]] = {}
    for name, variant in variants.items():
        fold_auc = []
        for train, test in splitter.split(variant, labels):
            fold_scaler = StandardScaler()
            train_matrix = fold_scaler.fit_transform(variant[train])
            test_matrix = fold_scaler.transform(variant[test])
            model = LogisticRegression(
                class_weight="balanced",
                max_iter=2_000,
                random_state=RANDOM_SEED,
            )
            model.fit(train_matrix, labels[train])
            fold_auc.append(
                float(
                    roc_auc_score(
                        labels[test],
                        model.predict_proba(test_matrix)[:, 1],
                    )
                )
            )
        variant_results[name] = {
            "fold_auc": fold_auc,
            "mean_auc": float(np.mean(fold_auc)),
            "std_auc": float(np.std(fold_auc, ddof=1)),
        }

    scaler = StandardScaler()
    matrix = scaler.fit_transform(matrix)
    final_model = LogisticRegression(
        class_weight="balanced",
        max_iter=2_000,
        random_state=RANDOM_SEED,
    )
    final_model.fit(matrix, labels)
    coefficients = [
        {"feature": name, "coefficient": float(value)}
        for name, value in zip(feature_names, final_model.coef_[0], strict=True)
    ]
    coefficients.sort(key=lambda item: -abs(cast(float, item["coefficient"])))
    return {
        "available": True,
        "model": "class-balanced logistic regression",
        "cross_validation": "5-fold stratified, shuffled",
        "variants": variant_results,
        "fold_auc": variant_results["combined"]["fold_auc"],
        "mean_auc": variant_results["combined"]["mean_auc"],
        "std_auc": variant_results["combined"]["std_auc"],
        "top_standardized_coefficients": coefficients[:20],
        "interpretation": (
            "AUC materially above 0.5 means public-target membership is predictable from "
            "visible catalog metadata and therefore is not consistent with a uniform product draw."
        ),
    }


def _render_report(payload: dict[str, object]) -> str:
    inputs = cast(dict[str, object], payload["inputs"])
    comparisons = cast(dict[str, dict[str, object]], payload["numeric_comparisons"])
    category = cast(dict[str, object], payload["coarse_category_comparison"])
    popularity = cast(dict[str, object], payload["popularity_selection_analysis"])
    conditional_split = cast(
        dict[str, object],
        popularity["conditional_1000_unique_target_split_test"],
    )
    predictability = cast(dict[str, object], payload["selection_predictability"])
    category_bootstrap = cast(dict[str, object], category["uniform_random_200"])
    predictor_variants = cast(
        dict[str, dict[str, object]],
        predictability.get("variants", {}),
    )
    lines = [
        "# Public 200 target-selection analysis",
        "",
        "## Scope",
        "",
        f"- Catalog products: **{inputs['catalog_product_count']:,}**",
        f"- Known public targets: **{inputs['public_target_count']}**",
        f"- Other catalog products: **{inputs['non_public_catalog_count']:,}**",
        "- The private 800 targets are not available. This report cannot directly compare public and private targets.",
        "- Official documentation says sessions come from the Clothing 5-core leave-last-out split; targets are held-out purchase interactions rather than uniformly sampled product IDs.",
        "",
        "## Numeric differences",
        "",
        "| Metric | Public mean | Other mean | Ratio | Uniform-200 5–95% | Public percentile | Category-adjusted difference |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in METRIC_LABELS:
        row = comparisons[name]
        public = cast(dict[str, object], row["public"])
        rest = cast(dict[str, object], row["non_public"])
        bootstrap = cast(dict[str, object], row["uniform_random_200"])
        ratio = row["mean_ratio"]
        lines.append(
            f"| {METRIC_LABELS[name]} | {_number(public['mean'])} | {_number(rest['mean'])} | "
            f"{'—' if ratio is None else _number(ratio)} | {_number(bootstrap['p05'])}–{_number(bootstrap['p95'])} | "
            f"{_number(bootstrap['public_percentile'])}% | {_number(row['category_adjusted_difference'])} |"
        )

    lines.extend(
        [
            "",
            "## Review-count selection gradient",
            "",
            "| Review-count band | Catalog products | Public targets | Public share | Target probability within band | Enrichment over uniform product draw |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in cast(list[dict[str, object]], popularity["bands"]):
        lines.append(
            f"| {item['band']} | {item['catalog_count']} | {item['public_count']} | "
            f"{_number(100 * cast(float, item['public_rate']))}% | "
            f"{_number(100 * cast(float, item['public_target_rate_within_band']))}% | "
            f"{_number(item['enrichment_over_uniform_product_sampling'])} |"
        )

    lines.extend(
        [
            "",
            "### Conditional check of a random 200/800 split",
            "",
            f"Assumption: {conditional_split['assumption']}",
            "",
            "| Minimum reviews | Eligible in entire catalog | Observed public | Expected public under most favorable random split | Conservative upper-tail probability | Maximum possible private share |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in cast(list[dict[str, object]], conditional_split["bounds"]):
        lines.append(
            f"| {item['review_threshold']:,} | {item['catalog_count']} | {item['public_count']} | "
            f"{_number(item['expected_public_count_if_all_eligible_entered_pool_and_split_was_random'])} | "
            f"{float(cast(float, item['hypergeometric_upper_tail_probability'])):.3e} | "
            f"{_number(100 * cast(float, item['maximum_private_rate_if_targets_are_unique_and_disjoint']))}% |"
        )

    lines.extend(
        [
            "",
            "## Coarse-category shift",
            "",
            f"Total-variation distance: **{_number(category['total_variation_distance'])}**. "
            f"A uniform random 200-product sample has a 5–95% range of "
            f"**{_number(category_bootstrap['p05'])}–{_number(category_bootstrap['p95'])}**; "
            f"the public set is at the **{_number(category_bootstrap['public_percentile'])}th percentile**.",
            "",
            "| Category | Public | Expected under uniform draw | Enrichment | Other catalog count |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in cast(list[dict[str, object]], category["largest_public_categories"]):
        enrichment = item["enrichment"]
        lines.append(
            f"| {item['value']} | {item['public_count']} | "
            f"{_number(item['expected_public_count_under_uniform'])} | "
            f"{'—' if enrichment is None else _number(enrichment)} | {item['non_public_count']} |"
        )

    lines.extend(
        [
            "",
            "## Can visible metadata predict public membership?",
            "",
            f"- Numeric metadata only AUC: **{_number(predictor_variants.get('numeric_only', {}).get('mean_auc'))}**",
            f"- Coarse category only AUC: **{_number(predictor_variants.get('category_only', {}).get('mean_auc'))}**",
            f"- Combined 5-fold AUC: **{_number(predictability.get('mean_auc'))}** "
            f"± **{_number(predictability.get('std_auc'))}**",
            "- An AUC near 0.5 would be consistent with a uniform product draw. A materially higher AUC means the public targets occupy a visibly different part of catalog space.",
            "",
            "Top standardized predictors:",
            "",
            "| Feature | Coefficient toward public target |",
            "|---|---:|",
        ]
    )
    for item in cast(
        list[dict[str, object]],
        predictability.get("top_standardized_coefficients", []),
    ):
        lines.append(f"| {item['feature']} | {_number(item['coefficient'])} |")

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This analysis can establish that the public targets differ from a uniform catalog sample. It cannot establish that the public 200 differ from the private 800, because those products are hidden. If the organizer split one already-selected 1,000-session pool randomly, the same upstream purchase-weighting bias should appear in both sets even when individual products do not overlap.",
            "",
        ]
    )
    return "\n".join(lines)


def _intent_signature(category: str, card: dict[str, object]) -> tuple[str, ...]:
    values = [
        *[str(item) for item in _list(card.get("hard_constraints"))],
        *[str(item) for item in _list(card.get("soft_preferences"))],
    ]
    return (normalize_phrase(category), *[normalize_phrase(item) for item in values])


def _coarse_category(product: dict[str, object]) -> str:
    values = [str(item) for item in _list(product.get("categories"))]
    return normalize_phrase(coarse_category(values)) or "clothing item"


def _leaf_category(product: dict[str, object]) -> str:
    values = [normalize_phrase(item) for item in _list(product.get("categories"))]
    return next((item for item in reversed(values) if item), "unknown")


def _store(product: dict[str, object]) -> str | None:
    value = product.get("store")
    return value.strip().casefold() if type(value) is str and value.strip() else None


def _release_year(details: dict[str, object]) -> int | None:
    for key, value in details.items():
        if "date first available" not in key.casefold():
            continue
        match = YEAR_RE.search(str(value))
        if match:
            return int(match.group(1))
    return None


def _price(value: object) -> float | None:
    if type(value) in (int, float):
        numeric = float(cast(int | float, value))
        return numeric if math.isfinite(numeric) else None
    if type(value) is str:
        match = PRICE_RE.search(value.replace(",", ""))
        if match:
            return float(match.group(0))
    return None


def _list(value: object) -> list[object]:
    return cast(list[object], value) if type(value) is list else []


def _dict(value: object) -> dict[str, object]:
    return cast(dict[str, object], value) if type(value) is dict else {}


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, 1):
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            if type(value) is not dict:
                raise ValueError(f"{path}:{line_number} must be an object")
            rows.append(cast(dict[str, object], value))
    return rows


def _number(value: object) -> str:
    if value is None:
        return "—"
    numeric = float(cast(float | int, value))
    if abs(numeric) >= 1_000:
        return f"{numeric:,.1f}"
    if abs(numeric) >= 10:
        return f"{numeric:.2f}"
    return f"{numeric:.4f}"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.jsonl")
    parser.add_argument("--public-set", type=Path, default=ROOT / "data/public_set.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/benchmark/public-target-selection-v1",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
