"""Evaluate first-turn BM25 and dense target recall on the public simulator.

The script deliberately keeps simulator-only labels on the evaluation side.
Dense receives the first user-visible message as temporary ``q_sem``; BM25 uses
the official reset profile and message API. The dense route is injected through
a small factory contract so this evaluator does not choose a model or own a
production index implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

CUTOFFS = (10, 40, 100)
MAX_CUTOFF = max(CUTOFFS)
PROBE_CUTOFFS = (20, 40, 80)
SCHEMA = "shopping-copilot/first-turn-retrieval-evaluation/v1"

# Support both ``python -m scripts.retrieval.evaluate_first_turn`` and direct
# execution from a repository checkout.  Imports remain function-local so this
# path setup does not hide an installed-package import failure.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from shopping_copilot.retrieval.errors import RetrievalError  # noqa: E402


class DenseRetriever(Protocol):
    """Minimal adapter expected from the injected production dense route."""

    def search_with_scores(self, q_sem: str, *, top_k: int) -> object:
        """Return one bound production ranking with reusable scores."""


@dataclass(frozen=True, slots=True)
class FirstTurnCase:
    """Simulator-produced message plus evaluation-only metadata."""

    ordinal: int
    scenario: str
    user_message: str
    user_profile: dict[str, object]
    target_parent_asin: str


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Target ranks from two independent first-turn retrieval calls."""

    scenario: str
    bm25_rank: int | None
    dense_rank: int | None
    bm25_parent_asins: tuple[str, ...]
    dense_parent_asins: tuple[str, ...]
    probe_g: tuple[float | None, ...]


@dataclass(frozen=True, slots=True)
class NormalizedRanking:
    """Catalog-valid unique IDs plus normalization diagnostics."""

    parent_asins: tuple[str, ...]
    invalid_count: int
    duplicate_count: int


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/catalog.jsonl"),
        help="frozen catalog JSONL (default: data/catalog.jsonl)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/public_set.jsonl"),
        help="public session JSONL (default: data/public_set.jsonl)",
    )
    parser.add_argument(
        "--dense-factory",
        required=True,
        metavar="MODULE:CALLABLE",
        help=(
            "factory called with catalog_path, index_path, and release_dir; "
            "the returned object must implement search_with_scores(q_sem, top_k=...)"
        ),
    )
    parser.add_argument(
        "--dense-index",
        type=Path,
        default=None,
        help="optional prebuilt index path passed unchanged to the dense factory",
    )
    parser.add_argument(
        "--semantic-release",
        type=Path,
        default=Path("artifacts/catalog-semantic/release-v0"),
        help="active Catalog Semantic release used to bind the Dense index",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write JSON here instead of stdout",
    )
    parser.add_argument(
        "--specificity-chains",
        type=int,
        default=100,
        help="number of target-derived broad/medium/narrow shadow chains (default: 100)",
    )
    return parser


def _load_factory(spec: str) -> Callable[..., object]:
    module_name, separator, attribute_name = spec.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("--dense-factory must use MODULE:CALLABLE syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise TypeError(f"dense factory is not callable: {spec}")
    return cast(Callable[..., object], factory)


def _build_dense_retriever(
    factory_spec: str,
    *,
    catalog_path: Path,
    index_path: Path | None,
    release_dir: Path,
) -> DenseRetriever:
    factory = _load_factory(factory_spec)
    retriever = factory(
        catalog_path=catalog_path,
        index_path=index_path,
        release_dir=release_dir,
    )
    if not callable(getattr(retriever, "search_with_scores", None)):
        raise TypeError("dense factory result must provide search_with_scores")
    return cast(DenseRetriever, retriever)


def _first_turn_cases(
    samples: Sequence[dict],
    *,
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> tuple[FirstTurnCase, ...]:
    from evaluator.local_evaluator import (
        coarse_category,
        initial_message,
        materialize_hidden_fields,
    )

    cases: list[FirstTurnCase] = []
    for ordinal, sample in enumerate(samples, start=1):
        ground_truth = sample.get("ground_truth")
        if not isinstance(ground_truth, dict):
            raise ValueError(f"sample {ordinal} has no ground_truth object")
        target = str(ground_truth.get("parent_asin", "")).strip()
        if not target or target not in products:
            raise ValueError(f"sample {ordinal} has an unknown target parent_asin")

        intent_card, behavior = materialize_hidden_fields(sample, products)
        effective_sample = {
            **sample,
            "intent_card": intent_card,
            "behavior": behavior,
        }
        user_message = initial_message(
            effective_sample,
            coarse_category(categories.get(target, [])),
            set(),
        )
        scenario = str(sample.get("scenario_type", "")).strip()
        profile = sample.get("user_profile")
        if not scenario or not isinstance(profile, dict):
            raise ValueError(f"sample {ordinal} has malformed scenario/profile metadata")
        cases.append(
            FirstTurnCase(
                ordinal=ordinal,
                scenario=scenario,
                user_message=user_message,
                user_profile=profile,
                target_parent_asin=target,
            )
        )
    return tuple(cases)


def _hit_payload(result: object) -> object:
    if isinstance(result, Mapping):
        if "hits" in result:
            return result["hits"]
        if "recommendations" in result:
            return result["recommendations"]
    hits = getattr(result, "hits", None)
    return hits if hits is not None else result


def _parent_asin(item: object) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, Mapping):
        return str(item.get("parent_asin", "")).strip()
    return str(getattr(item, "parent_asin", "")).strip()


def _normalize_ranking(
    payload: object,
    *,
    catalog_ids: set[str],
    limit: int,
) -> NormalizedRanking:
    payload = _hit_payload(payload)
    if isinstance(payload, (str, bytes)) or not isinstance(payload, Iterable):
        raise TypeError("retrieval result must be an iterable of hits")

    parent_asins: list[str] = []
    seen: set[str] = set()
    invalid_count = 0
    duplicate_count = 0
    for item in payload:
        parent_asin = _parent_asin(item)
        if not parent_asin or parent_asin not in catalog_ids:
            invalid_count += 1
            continue
        if parent_asin in seen:
            duplicate_count += 1
            continue
        seen.add(parent_asin)
        parent_asins.append(parent_asin)
        if len(parent_asins) >= limit:
            break
    return NormalizedRanking(
        parent_asins=tuple(parent_asins),
        invalid_count=invalid_count,
        duplicate_count=duplicate_count,
    )


def _target_rank(target: str, ranking: Sequence[str]) -> int | None:
    try:
        return ranking.index(target) + 1
    except ValueError:
        return None


def _route_metrics(results: Sequence[CaseResult], route: str) -> dict[str, object]:
    ranks = [getattr(item, f"{route}_rank") for item in results]
    return {
        "recall_at": {
            str(cutoff): _ratio(
                sum(rank is not None and rank <= cutoff for rank in ranks), len(ranks)
            )
            for cutoff in CUTOFFS
        },
        "mrr_at_10": _mean([0.0 if rank is None or rank > 10 else 1.0 / rank for rank in ranks]),
    }


def _summary(results: Sequence[CaseResult]) -> dict[str, object]:
    contribution: dict[str, object] = {}
    union_recall: dict[str, float] = {}
    pool_shape: dict[str, object] = {}
    for cutoff in CUTOFFS:
        both = 0
        dense_only = 0
        bm25_only = 0
        neither = 0
        overlap_sizes: list[float] = []
        union_sizes: list[float] = []
        for item in results:
            bm25_hit = item.bm25_rank is not None and item.bm25_rank <= cutoff
            dense_hit = item.dense_rank is not None and item.dense_rank <= cutoff
            if bm25_hit and dense_hit:
                both += 1
            elif dense_hit:
                dense_only += 1
            elif bm25_hit:
                bm25_only += 1
            else:
                neither += 1
            bm25_ids = set(item.bm25_parent_asins[:cutoff])
            dense_ids = set(item.dense_parent_asins[:cutoff])
            overlap_sizes.append(float(len(bm25_ids & dense_ids)))
            union_sizes.append(float(len(bm25_ids | dense_ids)))
        contribution[str(cutoff)] = {
            "both": both,
            "dense_only": dense_only,
            "bm25_only": bm25_only,
            "neither": neither,
        }
        union_recall[str(cutoff)] = _ratio(both + dense_only + bm25_only, len(results))
        pool_shape[str(cutoff)] = {
            "mean_overlap_size": _mean(overlap_sizes),
            "mean_union_size": _mean(union_sizes),
            "maximum_union_budget": cutoff * 2,
        }

    return {
        "sample_count": len(results),
        "bm25": _route_metrics(results, "bm25"),
        "dense": _route_metrics(results, "dense"),
        "candidate_union": {
            "definition": (
                "target appears in either route's first K results; this consumes up to 2K "
                "candidates and is not a fused ranking"
            ),
            "recall_at": union_recall,
            "pool_shape_at": pool_shape,
        },
        "target_contribution_counts": contribution,
        "shadow_probe": _probe_summary(results),
    }


def _probe_summary(results: Sequence[CaseResult]) -> dict[str, object]:
    by_k: dict[str, object] = {}
    for position, cutoff in enumerate(PROBE_CUTOFFS):
        values = [item.probe_g[position] for item in results]
        available = [value for value in values if value is not None]
        by_k[str(cutoff)] = {
            "available_count": len(available),
            "unavailable_count": len(values) - len(available),
            "raw_g": _distribution(available),
        }
    correlations = {}
    for left, right in zip(PROBE_CUTOFFS[:-1], PROBE_CUTOFFS[1:], strict=True):
        left_position = PROBE_CUTOFFS.index(left)
        right_position = PROBE_CUTOFFS.index(right)
        pairs = [
            (item.probe_g[left_position], item.probe_g[right_position])
            for item in results
            if item.probe_g[left_position] is not None and item.probe_g[right_position] is not None
        ]
        correlations[f"{left}_vs_{right}"] = _spearman(
            [float(left_value) for left_value, _ in pairs],
            [float(right_value) for _, right_value in pairs],
        )
    return {
        "definition": (
            "uncalibrated mean-centered debiased pairwise cosine; "
            "logged only and never used by retrieval"
        ),
        "by_probe_k": by_k,
        "spearman": correlations,
        "route_contribution_at_40": _probe_by_route_contribution(results, cutoff=40),
    }


def _distribution(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "min": round(ordered[0], 6),
        "p10": round(_quantile(ordered, 0.10), 6),
        "median": round(statistics.median(ordered), 6),
        "mean": round(statistics.fmean(ordered), 6),
        "p90": round(_quantile(ordered, 0.90), 6),
        "max": round(ordered[-1], 6),
    }


def _latency_distribution(values: Sequence[float]) -> dict[str, float] | None:
    """Summarize locally observed wall-clock latency in milliseconds."""

    if not values:
        return None
    ordered = sorted(values)
    return {
        "min": round(ordered[0], 3),
        "median": round(statistics.median(ordered), 3),
        "mean": round(statistics.fmean(ordered), 3),
        "p95": round(_quantile(ordered, 0.95), 3),
        "p99": round(_quantile(ordered, 0.99), 3),
        "max": round(ordered[-1], 3),
    }


def _latency_series(values: Sequence[float]) -> dict[str, object]:
    return {
        "sample_count": len(values),
        "first": None if not values else round(values[0], 3),
        "all": _latency_distribution(values),
        "warm_excluding_first": _latency_distribution(values[1:]),
    }


def _quantile(ordered: Sequence[float], probability: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = statistics.fmean(left_ranks)
    right_mean = statistics.fmean(right_ranks)
    covariance = sum(
        (left_rank - left_mean) * (right_rank - right_mean)
        for left_rank, right_rank in zip(left_ranks, right_ranks, strict=True)
    )
    left_scale = math.sqrt(sum((rank - left_mean) ** 2 for rank in left_ranks))
    right_scale = math.sqrt(sum((rank - right_mean) ** 2 for rank in right_ranks))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return round(covariance / (left_scale * right_scale), 6)


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def _probe_by_route_contribution(
    results: Sequence[CaseResult], *, cutoff: int
) -> dict[str, object]:
    groups: defaultdict[str, list[float]] = defaultdict(list)
    probe_position = PROBE_CUTOFFS.index(40)
    for item in results:
        value = item.probe_g[probe_position]
        if value is None:
            continue
        bm25_hit = item.bm25_rank is not None and item.bm25_rank <= cutoff
        dense_hit = item.dense_rank is not None and item.dense_rank <= cutoff
        if bm25_hit and dense_hit:
            group = "both"
        elif dense_hit:
            group = "dense_only"
        elif bm25_hit:
            group = "bm25_only"
        else:
            group = "neither"
        groups[group].append(value)
    return {
        group: {"count": len(values), "mean_raw_g": round(statistics.fmean(values), 6)}
        for group, values in sorted(groups.items())
    }


def _specificity_chain_diagnostic(
    cases: Sequence[FirstTurnCase],
    *,
    products: Mapping[str, dict],
    dense_retriever: object,
    probe: object,
    limit: int,
) -> dict[str, object] | None:
    search_with_scores = getattr(dense_retriever, "search_with_scores", None)
    if limit <= 0 or not callable(search_with_scores):
        return None
    from shopping_copilot.retrieval import DenseSearchResult, FixedDenseProbe

    if not isinstance(probe, FixedDenseProbe):
        return None
    level_values: dict[str, list[float]] = {"broad": [], "medium": [], "narrow": []}
    broad_medium = 0
    medium_narrow = 0
    full_chain = 0
    evaluated = 0
    for case in _specificity_case_order(cases, products=products):
        if evaluated >= limit:
            break
        product = products[case.target_parent_asin]
        queries = _specificity_queries(product)
        if queries is None:
            continue
        values: list[float] = []
        for query in queries:
            result = search_with_scores(query, top_k=40)
            if not isinstance(result, DenseSearchResult):
                raise TypeError("search_with_scores must return DenseSearchResult")
            observation = probe.observe(result, probe_k=40)
            value = observation.coherence.debiased_pairwise_cosine
            if value is None:
                values = []
                break
            values.append(value)
        if len(values) != 3:
            continue
        for level, value in zip(("broad", "medium", "narrow"), values, strict=True):
            level_values[level].append(value)
        evaluated += 1
        if values[0] < values[1]:
            broad_medium += 1
        if values[1] < values[2]:
            medium_narrow += 1
        if values[0] < values[1] < values[2]:
            full_chain += 1
    if evaluated == 0:
        return None
    return {
        "definition": (
            "target-derived synthetic falsification check using broad category, full "
            "category path, and the path plus two explicit product constraints; queries "
            "never enter runtime or calibration"
        ),
        "selection": "stable SHA-256 order, round-robin stratified by scenario and broad category",
        "evaluated_count": evaluated,
        "mean_raw_g": {
            level: round(statistics.fmean(values), 6) for level, values in level_values.items()
        },
        "ordering_rate": {
            "broad_lt_medium": _ratio(broad_medium, evaluated),
            "medium_lt_narrow": _ratio(medium_narrow, evaluated),
            "broad_lt_medium_lt_narrow": _ratio(full_chain, evaluated),
        },
    }


def _specificity_queries(product: Mapping[str, object]) -> tuple[str, str, str] | None:
    from evaluator.local_evaluator import intent_card

    categories = _meaningful_categories(product)
    if len(categories) < 2:
        return None
    broad = categories[0]
    medium = " ".join(dict.fromkeys(categories))
    card = intent_card(dict(product))
    constraints = [
        str(value).strip() for value in card.get("hard_constraints", []) if str(value).strip()
    ][:2]
    if not constraints:
        return None
    narrow = medium + ". Requirements: " + "; ".join(constraints)
    if len({broad.casefold(), medium.casefold(), narrow.casefold()}) != 3:
        return None
    return broad, medium, narrow


def _specificity_case_order(
    cases: Sequence[FirstTurnCase],
    *,
    products: Mapping[str, dict],
) -> tuple[FirstTurnCase, ...]:
    buckets: defaultdict[tuple[str, str], list[FirstTurnCase]] = defaultdict(list)
    for case in cases:
        categories = _meaningful_categories(products[case.target_parent_asin])
        broad = "" if not categories else categories[0].casefold()
        buckets[(case.scenario, broad)].append(case)
    for bucket in buckets.values():
        bucket.sort(
            key=lambda case: hashlib.sha256(
                (case.target_parent_asin + "\0" + case.user_message).encode("utf-8")
            ).digest()
        )
    ordered: list[FirstTurnCase] = []
    depth = 0
    while True:
        appended = False
        for key in sorted(buckets):
            bucket = buckets[key]
            if depth < len(bucket):
                ordered.append(bucket[depth])
                appended = True
        if not appended:
            return tuple(ordered)
        depth += 1


def _meaningful_categories(product: Mapping[str, object]) -> list[str]:
    raw_categories = product.get("categories")
    if type(raw_categories) is not list:
        return []
    excluded = {
        "clothing",
        "clothing shoes & jewelry",
        "clothing, shoes & jewelry",
    }
    return [
        str(value).strip()
        for value in raw_categories
        if type(value) is str
        and str(value).strip()
        and str(value).strip().casefold() not in excluded
    ]


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else round(numerator / denominator, 6)


def _mean(values: Sequence[float]) -> float:
    return 0.0 if not values else round(sum(values) / len(values), 6)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def evaluate_first_turn(
    *,
    catalog_path: Path,
    dataset_path: Path,
    dense_factory: str,
    dense_index: Path | None,
    semantic_release: Path,
    specificity_chains: int = 100,
) -> dict[str, object]:
    """Run independent first-turn BM25 and dense component retrieval."""

    from evaluator.local_evaluator import catalog_index, load_jsonl
    from starter.agent import Agent as Bm25Agent

    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    cases = _first_turn_cases(samples, categories=categories, products=products)
    dense_initialization_started = time.perf_counter()
    dense = _build_dense_retriever(
        dense_factory,
        catalog_path=catalog_path,
        index_path=dense_index,
        release_dir=semantic_release,
    )
    dense_initialization_ms = (time.perf_counter() - dense_initialization_started) * 1000.0
    bm25_initialization_started = time.perf_counter()
    bm25 = Bm25Agent(catalog_path)
    bm25_initialization_ms = (time.perf_counter() - bm25_initialization_started) * 1000.0
    dense_index_object = getattr(dense, "index", None)
    from shopping_copilot.retrieval import DenseIndex, DenseSearchResult, FixedDenseProbe

    if not isinstance(dense_index_object, DenseIndex):
        raise TypeError("dense factory result must expose its verified DenseIndex")
    probe_initialization_started = time.perf_counter()
    probe = FixedDenseProbe(dense_index_object)
    probe_initialization_ms = (time.perf_counter() - probe_initialization_started) * 1000.0

    results: list[CaseResult] = []
    normalization = {
        "bm25_invalid": 0,
        "bm25_duplicates": 0,
        "dense_invalid": 0,
        "dense_duplicates": 0,
    }
    query_latency_ms: dict[str, list[float]] = {
        "bm25_reset": [],
        "bm25": [],
        "dense_retrieval": [],
        "shadow_probe": [],
    }
    for case in cases:
        # Neither route receives sample_id, scenario_type, difficulty, or target.
        session_id = f"first_turn_{case.ordinal:04d}"
        bm25_reset_started = time.perf_counter()
        bm25.reset(session_id, case.user_profile)
        query_latency_ms["bm25_reset"].append((time.perf_counter() - bm25_reset_started) * 1000.0)
        bm25_started = time.perf_counter()
        bm25_response = bm25.respond(session_id, case.user_message, 1, MAX_CUTOFF)
        query_latency_ms["bm25"].append((time.perf_counter() - bm25_started) * 1000.0)
        bm25_ranking = _normalize_ranking(
            bm25_response.get("recommendations"),
            catalog_ids=catalog_ids,
            limit=MAX_CUTOFF,
        )
        dense_started = time.perf_counter()
        dense_result = dense.search_with_scores(case.user_message, top_k=MAX_CUTOFF)
        if not isinstance(dense_result, DenseSearchResult):
            raise TypeError("search_with_scores must return DenseSearchResult")
        query_latency_ms["dense_retrieval"].append((time.perf_counter() - dense_started) * 1000.0)
        dense_response = dense_result.hits
        probe_started = time.perf_counter()
        observed_values: list[float | None] = []
        for probe_k in PROBE_CUTOFFS:
            observation = probe.observe(dense_result, probe_k=probe_k)
            observed_values.append(observation.coherence.debiased_pairwise_cosine)
        query_latency_ms["shadow_probe"].append((time.perf_counter() - probe_started) * 1000.0)
        probe_g = tuple(observed_values)
        dense_ranking = _normalize_ranking(
            dense_response,
            catalog_ids=catalog_ids,
            limit=MAX_CUTOFF,
        )
        normalization["bm25_invalid"] += bm25_ranking.invalid_count
        normalization["bm25_duplicates"] += bm25_ranking.duplicate_count
        normalization["dense_invalid"] += dense_ranking.invalid_count
        normalization["dense_duplicates"] += dense_ranking.duplicate_count
        results.append(
            CaseResult(
                scenario=case.scenario,
                bm25_rank=_target_rank(case.target_parent_asin, bm25_ranking.parent_asins),
                dense_rank=_target_rank(case.target_parent_asin, dense_ranking.parent_asins),
                bm25_parent_asins=bm25_ranking.parent_asins,
                dense_parent_asins=dense_ranking.parent_asins,
                probe_g=probe_g,
            )
        )

    by_scenario: defaultdict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        by_scenario[result.scenario].append(result)

    specificity = _specificity_chain_diagnostic(
        cases,
        products=products,
        dense_retriever=dense,
        probe=probe,
        limit=specificity_chains,
    )
    return {
        "schema": SCHEMA,
        "catalog": {
            "path": str(catalog_path),
            "sha256": _sha256(catalog_path),
            "product_count": len(catalog_ids),
        },
        "dataset": {
            "path": str(dataset_path),
            "sha256": _sha256(dataset_path),
        },
        "dense": {
            "factory": dense_factory,
            "index_path": None if dense_index is None else str(dense_index),
            "semantic_release": str(semantic_release),
        },
        "cutoffs": list(CUTOFFS),
        "notes": [
            "Dense receives only the simulator-produced first user_message as temporary q_sem.",
            "BM25 receives the official reset user_profile and the first user_message.",
            "Labels and scenario metadata are used only after retrieval for evaluation/grouping.",
            "Intent-override first-turn recall is component diagnostics, not an official eligible hit.",
            "Candidate union is unordered membership coverage, not Top-K fusion quality.",
            "Probe G is shadow-only and does not alter route depth, scores, or ranking.",
            "Latency is local wall-clock component timing; initialization excludes dataset loading.",
        ],
        "latency_ms": {
            "initialization": {
                "bm25": round(bm25_initialization_ms, 3),
                "dense": round(dense_initialization_ms, 3),
                "shadow_probe": round(probe_initialization_ms, 3),
            },
            "per_query": {
                name: _latency_series(values) for name, values in query_latency_ms.items()
            },
        },
        "normalization": normalization,
        "component_all_first_turn": _summary(results),
        "officially_eligible_first_turn": _summary(
            [result for result in results if result.scenario != "intent_override"]
        ),
        "scenario_breakdown": {
            scenario: _summary(group) for scenario, group in sorted(by_scenario.items())
        },
        "shadow_specificity_chain": specificity,
    }


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        report = evaluate_first_turn(
            catalog_path=args.catalog,
            dataset_path=args.dataset,
            dense_factory=args.dense_factory,
            dense_index=args.dense_index,
            semantic_release=args.semantic_release,
            specificity_chains=args.specificity_chains,
        )
    except (ImportError, OSError, RetrievalError, TypeError, ValueError) as error:
        parser.error(str(error))

    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
