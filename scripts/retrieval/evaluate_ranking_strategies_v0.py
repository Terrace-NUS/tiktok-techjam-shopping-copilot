"""Compare fusion, cross-encoder, MMR, DPP, and latent-xQuAD ranking strategies."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluator.local_evaluator import (  # noqa: E402
    coarse_category,
    initial_message,
    materialize_hidden_fields,
)
from scripts.retrieval.evaluate_multi_route_v0 import (  # noqa: E402
    CASES,
    _compile_case,
    _load_metadata,
)
from shopping_copilot.catalog.semantic.release import (  # noqa: E402
    load_catalog_semantic_release,
)
from shopping_copilot.retrieval import (  # noqa: E402
    CrossEncoderRelevanceReranker,
    GreedyDPPSelector,
    LatentAspectXQuADSelector,
    ReciprocalRankFusion,
    RelativeScoreFusion,
    RouteObservation,
    SentenceTransformerCrossEncoderScorer,
    VectorCandidate,
    create_retrieval_controller,
    dense_route_observation,
    lexical_route_observation,
    load_product_documents,
    normalized_fusion_relevance,
)

QWEN_MODEL = "Qwen/Qwen3-Reranker-0.6B"
QWEN_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
BGE_MODEL = "BAAI/bge-reranker-v2-m3"
BGE_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
QWEN_INSTRUCTION = (
    "Given a shopping request, judge whether the product satisfies the current intent "
    "and is useful to recommend. Rank exact constraints and semantic use-case fit above "
    "generic keyword overlap."
)
PUBLIC_SCENARIOS = frozenset({"buying", "browsing"})


@dataclass(frozen=True, slots=True)
class RankingCase:
    case_id: str
    source: str
    scenario: str | None
    query: str
    transparency: float
    routes: tuple[RouteObservation, ...]
    target_parent_asin: str | None
    eligible_count: int


@dataclass(frozen=True, slots=True)
class SelectedHit:
    parent_asin: str
    rank: int
    candidate_rank: int
    relevance: float
    maximum_similarity_to_selected: float
    selection_score: float
    latent_aspect: int | None = None


def main() -> int:
    args = _parse_args()
    controller = create_retrieval_controller(
        index_path=args.dense_index,
        release_dir=args.semantic_release,
        catalog_path=args.catalog,
        device=args.device,
        local_files_only=True,
    )
    release = load_catalog_semantic_release(args.semantic_release)
    metadata = _load_metadata(args.catalog)
    documents = {
        item.parent_asin: _compact_document(item.text)
        for item in load_product_documents(
            args.catalog,
            expected_parent_asins=set(controller.retriever.index.parent_asins),
        )
    }
    cases = [*_natural_cases(controller, release)]
    if args.public_limit > 0:
        cases.extend(
            _public_cases(
                controller,
                args.public_set,
                args.catalog,
                limit=args.public_limit,
            )
        )

    print(f"loaded {len(cases)} ranking cases", flush=True)
    qwen = CrossEncoderRelevanceReranker(
        scorer=SentenceTransformerCrossEncoderScorer(
            QWEN_MODEL,
            revision=QWEN_REVISION,
            device=args.device,
            local_files_only=args.local_models_only,
            max_length=args.max_length,
            instruction=QWEN_INSTRUCTION,
        )
    )
    bge = CrossEncoderRelevanceReranker(
        scorer=SentenceTransformerCrossEncoderScorer(
            BGE_MODEL,
            revision=BGE_REVISION,
            device=args.device,
            local_files_only=args.local_models_only,
            max_length=args.max_length,
        )
    )
    print("loaded Qwen3 and BGE cross-encoders", flush=True)

    case_payloads: list[dict[str, object]] = []
    model_latency = {"qwen_ms": [], "bge_ms": []}
    started = time.perf_counter()
    for ordinal, case in enumerate(cases, start=1):
        payload, timings = _evaluate_case(
            case,
            controller=controller,
            documents=documents,
            metadata=metadata,
            qwen=qwen,
            bge=bge,
            candidate_k=args.candidate_k,
            final_k=args.final_k,
            batch_size=args.batch_size,
        )
        case_payloads.append(payload)
        model_latency["qwen_ms"].append(timings["qwen_ms"])
        model_latency["bge_ms"].append(timings["bge_ms"])
        print(
            f"[{ordinal}/{len(cases)}] {case.source}:{case.case_id} "
            f"qwen={timings['qwen_ms']:.1f}ms bge={timings['bge_ms']:.1f}ms",
            flush=True,
        )

    payload = {
        "schema": "shopping-copilot/ranking-strategy-evaluation/v0",
        "purpose": "same_pool_ranking_strategy_comparison",
        "guardrails": [
            "Every strategy receives the same RRF-bounded candidate pool per query.",
            "Hard eligibility is resolved before ranking for natural formal cases.",
            "The ranking stack never receives public target labels; labels are evaluation-only.",
            "Public evaluation includes only Buying and Browsing simulator scenarios.",
            "Public low/high T values are counterfactual sweeps, not scenario-derived runtime estimates.",
        ],
        "bindings": {
            "catalog_id": controller.retriever.index.manifest.catalog_id,
            "release_id": controller.retriever.index.manifest.catalog_semantic_release_id,
            "dense_index_id": controller.retriever.index.index_id,
            "embedding_model": controller.retriever.index.manifest.embedding.model_id,
            "qwen_model": qwen.scorer.model_id,
            "bge_model": bge.scorer.model_id,
        },
        "parameters": {
            "candidate_k": args.candidate_k,
            "final_k": args.final_k,
            "batch_size": args.batch_size,
            "cross_encoder_max_length": args.max_length,
            "cross_encoder_prior_weight": 0.25,
            "public_t_sweep": [0.2, 0.8],
        },
        "latency": {
            "wall_seconds": time.perf_counter() - started,
            "qwen": _latency_summary(model_latency["qwen_ms"]),
            "bge": _latency_summary(model_latency["bge_ms"]),
        },
        "public_summary": _aggregate_public(case_payloads),
        "cases": case_payloads,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(_render_markdown(payload), encoding="utf-8")
    print(args.output_json, flush=True)
    print(args.output_markdown, flush=True)
    return 0


def _natural_cases(controller: Any, release: object) -> tuple[RankingCase, ...]:
    result: list[RankingCase] = []
    for case in CASES:
        query = _compile_case(case, release)
        retrieved = controller.search(query, transparency=case.transparency_anchor)
        result.append(
            RankingCase(
                case_id=case.case_id,
                source="natural",
                scenario=case.intent_shape,
                query=case.q_sem,
                transparency=case.transparency_anchor,
                routes=retrieved.routes,
                target_parent_asin=None,
                eligible_count=len(retrieved.hard_mask.eligible_parent_asins),
            )
        )
    return tuple(result)


def _public_cases(
    controller: Any,
    public_set: Path,
    catalog: Path,
    *,
    limit: int,
) -> tuple[RankingCase, ...]:
    with catalog.open("r", encoding="utf-8") as stream:
        products = {str(row["parent_asin"]): row for row in map(json.loads, stream)}
    with public_set.open("r", encoding="utf-8") as stream:
        samples = [
            row for row in map(json.loads, stream) if row["scenario_type"] in PUBLIC_SCENARIOS
        ]
    result: list[RankingCase] = []
    for sample in samples[:limit]:
        target = str(sample["ground_truth"]["parent_asin"])
        intent_card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": intent_card, "behavior": behavior}
        message = initial_message(
            effective,
            coarse_category(products[target].get("categories", [])),
            set(),
        )
        dense = controller.retriever.search_with_scores(message, top_k=controller.policy.route_k)
        lexical = controller.lexical_route.observe(message)
        result.append(
            RankingCase(
                case_id=str(sample["sample_id"]),
                source="public_simulator",
                scenario=str(sample["scenario_type"]),
                query=message,
                transparency=0.5,
                routes=(dense_route_observation(dense), lexical_route_observation(lexical)),
                target_parent_asin=target,
                eligible_count=len(controller.retriever.index.parent_asins),
            )
        )
    return tuple(result)


def _evaluate_case(
    case: RankingCase,
    *,
    controller: Any,
    documents: dict[str, str],
    metadata: dict[str, dict[str, object]],
    qwen: CrossEncoderRelevanceReranker,
    bge: CrossEncoderRelevanceReranker,
    candidate_k: int,
    final_k: int,
    batch_size: int,
) -> tuple[dict[str, object], dict[str, float]]:
    rrf_candidates, rrf_fused = _fusion_candidates(
        ReciprocalRankFusion(rank_constant=60),
        case.routes,
        candidate_k=candidate_k,
    )
    pool = frozenset(item.parent_asin for item in rrf_candidates)
    score_candidates, _ = _fusion_candidates(
        RelativeScoreFusion(agreement_power=0.0),
        case.routes,
        candidate_k=sum(len(item.hits) for item in case.routes),
        restrict_to=pool,
    )
    combmnz_candidates, _ = _fusion_candidates(
        RelativeScoreFusion(agreement_power=1.0),
        case.routes,
        candidate_k=sum(len(item.hits) for item in case.routes),
        restrict_to=pool,
    )

    qwen_started = time.perf_counter()
    qwen_result = qwen.rerank(
        case.query,
        rrf_candidates,
        documents=documents,
        prior_weight=0.25,
        batch_size=batch_size,
    )
    qwen_ms = (time.perf_counter() - qwen_started) * 1000.0
    bge_started = time.perf_counter()
    bge_result = bge.rerank(
        case.query,
        rrf_candidates,
        documents=documents,
        prior_weight=0.25,
        batch_size=batch_size,
    )
    bge_ms = (time.perf_counter() - bge_started) * 1000.0
    qwen_scores = {hit.parent_asin: hit.normalized_model_score for hit in qwen_result.hits}
    bge_scores = {hit.parent_asin: hit.normalized_model_score for hit in bge_result.hits}
    rrf_scores = {item.parent_asin: item.relevance for item in rrf_candidates}
    contributions = {item.parent_asin: item.contributions for item in rrf_fused}

    variants: list[dict[str, object]] = []
    for name, candidates in (
        ("rrf_topk", rrf_candidates),
        ("relative_score_topk", score_candidates),
        ("combmnz_topk", combmnz_candidates),
        ("qwen_topk", qwen_result.candidates),
        ("bge_topk", bge_result.candidates),
    ):
        variants.append(
            _variant_payload(
                name,
                _top_k(controller.retriever.index, candidates, final_k),
                transparency=None,
                metadata=metadata,
                rrf_scores=rrf_scores,
                qwen_scores=qwen_scores,
                bge_scores=bge_scores,
                contributions=contributions,
                target=case.target_parent_asin,
                index=controller.retriever.index,
            )
        )

    sweeps = (case.transparency, "observed") if case.source == "natural" else (0.2, "low")
    sweep_values = (sweeps,) if case.source == "natural" else ((0.2, "low"), (0.8, "high"))
    for transparency, suffix in sweep_values:
        weight = controller.diversity_policy.relevance_weight(float(transparency))
        for prefix, candidates in (
            ("rrf", rrf_candidates),
            ("qwen", qwen_result.candidates),
        ):
            variants.append(
                _variant_payload(
                    f"{prefix}_mmr_{suffix}",
                    _mmr(controller, candidates, final_k, weight),
                    transparency=float(transparency),
                    metadata=metadata,
                    rrf_scores=rrf_scores,
                    qwen_scores=qwen_scores,
                    bge_scores=bge_scores,
                    contributions=contributions,
                    target=case.target_parent_asin,
                    index=controller.retriever.index,
                )
            )
        for name, selected in (
            (
                f"qwen_dpp_{suffix}",
                GreedyDPPSelector(index=controller.retriever.index).select(
                    qwen_result.candidates,
                    top_k=final_k,
                    relevance_weight=weight,
                ),
            ),
            (
                f"bge_dpp_{suffix}",
                GreedyDPPSelector(index=controller.retriever.index).select(
                    bge_result.candidates,
                    top_k=final_k,
                    relevance_weight=weight,
                ),
            ),
            (
                f"qwen_xquad_{suffix}",
                LatentAspectXQuADSelector(index=controller.retriever.index).select(
                    qwen_result.candidates,
                    top_k=final_k,
                    relevance_weight=weight,
                ),
            ),
        ):
            variants.append(
                _variant_payload(
                    name,
                    tuple(SelectedHit(**asdict(hit)) for hit in selected.hits),
                    transparency=float(transparency),
                    metadata=metadata,
                    rrf_scores=rrf_scores,
                    qwen_scores=qwen_scores,
                    bge_scores=bge_scores,
                    contributions=contributions,
                    target=case.target_parent_asin,
                    index=controller.retriever.index,
                )
            )

    return (
        {
            "case_id": case.case_id,
            "source": case.source,
            "scenario": case.scenario,
            "query": case.query,
            "transparency": case.transparency if case.source == "natural" else None,
            "eligible_count": case.eligible_count,
            "target_parent_asin": case.target_parent_asin,
            "candidate_count": len(rrf_candidates),
            "target_in_candidate_pool": (
                None if case.target_parent_asin is None else case.target_parent_asin in pool
            ),
            "target_candidate_rank": (
                None
                if case.target_parent_asin is None
                else next(
                    (
                        item.candidate_rank
                        for item in rrf_candidates
                        if item.parent_asin == case.target_parent_asin
                    ),
                    None,
                )
            ),
            "routes": [
                {
                    "route": route.route.value,
                    "available": route.available,
                    "reason": route.reason,
                    "hit_count": len(route.hits),
                }
                for route in case.routes
            ],
            "variants": variants,
        },
        {"qwen_ms": qwen_ms, "bge_ms": bge_ms},
    )


def _fusion_candidates(
    fusion: Any,
    routes: tuple[RouteObservation, ...],
    *,
    candidate_k: int,
    restrict_to: frozenset[str] | None = None,
) -> tuple[tuple[VectorCandidate, ...], tuple[Any, ...]]:
    fused = fusion.fuse(routes, top_k=candidate_k)
    if restrict_to is not None:
        fused = tuple(item for item in fused if item.parent_asin in restrict_to)
    relevance = normalized_fusion_relevance(fused)
    return (
        tuple(
            VectorCandidate(
                parent_asin=item.parent_asin,
                candidate_rank=rank,
                relevance=item_relevance,
            )
            for rank, (item, item_relevance) in enumerate(
                zip(fused, relevance, strict=True), start=1
            )
        ),
        fused,
    )


def _top_k(
    index: Any, candidates: tuple[VectorCandidate, ...], top_k: int
) -> tuple[SelectedHit, ...]:
    selected: list[SelectedHit] = []
    rows: list[int] = []
    for rank, candidate in enumerate(candidates[:top_k], start=1):
        row = index.row_index(candidate.parent_asin)
        maximum = (
            0.0 if not rows else float(max(0.0, np.max(index.vectors[rows] @ index.vectors[row])))
        )
        rows.append(row)
        selected.append(
            SelectedHit(
                parent_asin=candidate.parent_asin,
                rank=rank,
                candidate_rank=candidate.candidate_rank,
                relevance=candidate.relevance,
                maximum_similarity_to_selected=maximum,
                selection_score=candidate.relevance,
            )
        )
    return tuple(selected)


def _mmr(
    controller: Any,
    candidates: tuple[VectorCandidate, ...],
    top_k: int,
    weight: float,
) -> tuple[SelectedHit, ...]:
    result = controller.reranker.rerank_candidates(
        candidates,
        top_k=top_k,
        relevance_weight=float(weight),
    )
    return tuple(
        SelectedHit(
            parent_asin=hit.parent_asin,
            rank=hit.rank,
            candidate_rank=hit.candidate_rank,
            relevance=hit.relevance,
            maximum_similarity_to_selected=hit.maximum_similarity_to_selected,
            selection_score=hit.mmr_score,
        )
        for hit in result.hits
    )


def _variant_payload(
    name: str,
    hits: tuple[SelectedHit, ...],
    *,
    transparency: float | None,
    metadata: dict[str, dict[str, object]],
    rrf_scores: dict[str, float],
    qwen_scores: dict[str, float],
    bge_scores: dict[str, float],
    contributions: dict[str, tuple[Any, ...]],
    target: str | None,
    index: Any,
) -> dict[str, object]:
    products = []
    for hit in hits:
        products.append(
            {
                **asdict(hit),
                "rrf_relevance": rrf_scores[hit.parent_asin],
                "qwen_score": qwen_scores.get(hit.parent_asin),
                "bge_score": bge_scores.get(hit.parent_asin),
                "route_contributions": [
                    {
                        "route": contribution.route.value,
                        "rank": contribution.route_rank,
                        "raw_score": contribution.raw_score,
                    }
                    for contribution in contributions[hit.parent_asin]
                ],
                **metadata[hit.parent_asin],
            }
        )
    target_rank = next(
        (int(item["rank"]) for item in products if item["parent_asin"] == target),
        None,
    )
    metrics = _list_metrics(index, products)
    metrics["target_rank"] = target_rank
    metrics["reciprocal_rank_at_10"] = (
        None if target is None else (0.0 if target_rank is None else 1.0 / target_rank)
    )
    return {
        "name": name,
        "transparency": transparency,
        "metrics": metrics,
        "products": products,
    }


def _list_metrics(index: Any, products: list[dict[str, object]]) -> dict[str, object]:
    if not products:
        return {
            "mean_pairwise_product_cosine": None,
            "intra_list_diversity": None,
            "unique_reporting_groups": 0,
            "unique_leaf_categories": 0,
            "mean_rrf_relevance": None,
            "mean_qwen_score": None,
            "mean_bge_score": None,
        }
    rows = [index.row_index(str(item["parent_asin"])) for item in products]
    similarity = index.vectors[rows] @ index.vectors[rows].T
    upper = similarity[np.triu_indices(len(rows), k=1)]
    mean_similarity = float(np.mean(upper)) if upper.size else 1.0
    return {
        "mean_pairwise_product_cosine": mean_similarity,
        "intra_list_diversity": 1.0 - mean_similarity,
        "unique_reporting_groups": len({str(item["reporting_group"]) for item in products}),
        "reporting_groups": sorted({str(item["reporting_group"]) for item in products}),
        "unique_leaf_categories": len({str(item["leaf_category"]) for item in products}),
        "mean_rrf_relevance": float(np.mean([float(item["rrf_relevance"]) for item in products])),
        "mean_qwen_score": _optional_mean(products, "qwen_score"),
        "mean_bge_score": _optional_mean(products, "bge_score"),
        "maximum_candidate_rank": max(int(item["candidate_rank"]) for item in products),
    }


def _optional_mean(products: list[dict[str, object]], key: str) -> float | None:
    values = [float(item[key]) for item in products if item.get(key) is not None]
    return None if not values else float(np.mean(values))


def _aggregate_public(cases: list[dict[str, object]]) -> dict[str, object]:
    public = [case for case in cases if case["source"] == "public_simulator"]
    if not public:
        return {"case_count": 0, "variants": {}}
    variant_names = [variant["name"] for variant in public[0]["variants"]]
    summaries: dict[str, object] = {}
    for name in variant_names:
        selected = [
            next(variant for variant in case["variants"] if variant["name"] == name)
            for case in public
        ]
        metrics = [variant["metrics"] for variant in selected]
        ranks = [item["target_rank"] for item in metrics]
        recalled_pairs = [
            (case, variant)
            for case, variant in zip(public, selected, strict=True)
            if bool(case["target_in_candidate_pool"])
        ]
        recalled_metrics = [variant["metrics"] for _, variant in recalled_pairs]
        summaries[name] = {
            "hit_at_10": sum(rank is not None for rank in ranks) / len(ranks),
            "mrr_at_10": float(np.mean([float(item["reciprocal_rank_at_10"]) for item in metrics])),
            "conditional_hit_at_10": (
                sum(item["target_rank"] is not None for item in recalled_metrics)
                / len(recalled_metrics)
                if recalled_metrics
                else 0.0
            ),
            "conditional_mrr_at_10": (
                float(np.mean([float(item["reciprocal_rank_at_10"]) for item in recalled_metrics]))
                if recalled_metrics
                else 0.0
            ),
            "mean_pairwise_product_cosine": float(
                np.mean([float(item["mean_pairwise_product_cosine"]) for item in metrics])
            ),
            "mean_unique_reporting_groups": float(
                np.mean([int(item["unique_reporting_groups"]) for item in metrics])
            ),
            "mean_qwen_score": _optional_metric_mean(metrics, "mean_qwen_score"),
            "mean_bge_score": _optional_metric_mean(metrics, "mean_bge_score"),
            "by_scenario": {
                scenario: _scenario_metrics(public, name, scenario)
                for scenario in sorted(PUBLIC_SCENARIOS)
            },
        }
    return {
        "case_count": len(public),
        "candidate_recall": sum(bool(case["target_in_candidate_pool"]) for case in public)
        / len(public),
        "variants": summaries,
    }


def _optional_metric_mean(metrics: list[dict[str, object]], key: str) -> float | None:
    values = [float(item[key]) for item in metrics if item.get(key) is not None]
    return None if not values else float(np.mean(values))


def _scenario_metrics(
    cases: list[dict[str, object]],
    variant_name: str,
    scenario: str,
) -> dict[str, float]:
    selected = [case for case in cases if case["scenario"] == scenario]
    metrics = [
        next(variant for variant in case["variants"] if variant["name"] == variant_name)["metrics"]
        for case in selected
    ]
    recalled_metrics = [
        next(variant for variant in case["variants"] if variant["name"] == variant_name)["metrics"]
        for case in selected
        if bool(case["target_in_candidate_pool"])
    ]
    return {
        "case_count": float(len(selected)),
        "hit_at_10": sum(item["target_rank"] is not None for item in metrics) / len(metrics),
        "mrr_at_10": float(np.mean([float(item["reciprocal_rank_at_10"]) for item in metrics])),
        "conditional_hit_at_10": (
            sum(item["target_rank"] is not None for item in recalled_metrics)
            / len(recalled_metrics)
            if recalled_metrics
            else 0.0
        ),
        "conditional_mrr_at_10": (
            float(np.mean([float(item["reciprocal_rank_at_10"]) for item in recalled_metrics]))
            if recalled_metrics
            else 0.0
        ),
    }


def _latency_summary(values: list[float]) -> dict[str, float]:
    observed = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(observed)),
        "median_ms": float(np.median(observed)),
        "p95_ms": float(np.quantile(observed, 0.95)),
        "max_ms": float(np.max(observed)),
    }


def _compact_document(text: str) -> str:
    kept = []
    for line in text.splitlines():
        label = line.partition(":")[0]
        if label in {"title", "categories", "store", "features", "details"}:
            kept.append(line)
    return "\n".join(kept)[:2400]


def _render_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Ranking Strategy v0：统一候选池实测",
        "",
        "所有方案收到同一个 RRF Top-K 候选池；hard mask 在排序之前完成。",
        "公开 simulator 只统计 Buying / Browsing，并对每个请求都分别扫低 T 与高 T，",
        "因此没有用二元标签冒充线上 Intent Transparency。",
        "",
        "## Public simulator 汇总",
        "",
        "| 方案 | Hit@10 | MRR@10 | 条件 Hit@10 | 条件 MRR@10 | 商品相似度↓ | 大类数↑ | Qwen 分↑ | BGE 分↑ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    summary = payload["public_summary"]
    for name, metrics in summary.get("variants", {}).items():
        lines.append(
            f"| {name} | {metrics['hit_at_10']:.3f} | {metrics['mrr_at_10']:.3f} | "
            f"{metrics['conditional_hit_at_10']:.3f} | "
            f"{metrics['conditional_mrr_at_10']:.3f} | "
            f"{metrics['mean_pairwise_product_cosine']:.3f} | "
            f"{metrics['mean_unique_reporting_groups']:.2f} | "
            f"{_format_optional(metrics['mean_qwen_score'])} | "
            f"{_format_optional(metrics['mean_bge_score'])} |"
        )
    lines.extend(
        [
            "",
            f"公共案例数：{summary.get('case_count', 0)}；"
            f"统一候选池 target recall：{summary.get('candidate_recall', 0.0):.3f}。",
            "",
            "## Natural story cases",
            "",
        ]
    )
    for case in payload["cases"]:
        if case["source"] != "natural":
            continue
        lines.extend(
            [
                f"### {case['case_id']}",
                "",
                f"请求：{case['query']}",
                "",
                "| 方案 | 商品相似度↓ | 大类数↑ | Qwen 分↑ | BGE 分↑ |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for variant in case["variants"]:
            metrics = variant["metrics"]
            lines.append(
                f"| {variant['name']} | {metrics['mean_pairwise_product_cosine']:.3f} | "
                f"{metrics['unique_reporting_groups']} | "
                f"{_format_optional(metrics['mean_qwen_score'])} | "
                f"{_format_optional(metrics['mean_bge_score'])} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _format_optional(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-index", type=Path, default=ROOT / "artifacts/retrieval/dense-v0")
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
    parser.add_argument("--public-set", type=Path, default=ROOT / "data/public_set.jsonl")
    parser.add_argument("--public-limit", type=int, default=160)
    parser.add_argument("--candidate-k", type=int, default=80)
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--local-models-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "artifacts/retrieval/ranking-strategy-v0.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=ROOT / "artifacts/retrieval/ranking-strategy-v0.md",
    )
    args = parser.parse_args()
    for name in ("public_limit", "candidate_k", "final_k", "batch_size", "max_length"):
        value = getattr(args, name)
        if type(value) is not int or value < (0 if name == "public_limit" else 1):
            parser.error(f"--{name.replace('_', '-')} is invalid")
    if args.final_k > args.candidate_k:
        parser.error("--final-k must not exceed --candidate-k")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
