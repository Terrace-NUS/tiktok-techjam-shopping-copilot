"""Evaluate formal Dense/Lexical/Facet retrieval on the bound 50k catalog."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shopping_copilot.catalog.semantic.release import (  # noqa: E402
    load_catalog_semantic_release,
)
from shopping_copilot.catalog.semantic.runtime import (  # noqa: E402
    SYSTEM_PRODUCT_CATEGORY_FACET_ID,
)
from shopping_copilot.query_compiler import (  # noqa: E402
    COMPILED_QUERY_SCHEMA,
    QUERY_COMPILER_VERSION,
    CompiledDirectives,
    CompiledHardConstraint,
    CompiledQuery,
    CompiledRankingPreference,
    ConstraintPolicy,
    DiversityDirective,
    RankingReason,
)
from shopping_copilot.retrieval import (  # noqa: E402
    FormalRetrievalPolicy,
    RetrievalRoute,
    RouteObservation,
    VectorCandidate,
    create_retrieval_controller,
    normalized_fusion_relevance,
)
from shopping_copilot.session_context import (  # noqa: E402
    Commitment,
    Operator,
    PreferenceSource,
)


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    intent_shape: str
    user_utterance: str
    q_lex: str
    q_sem: str
    transparency_anchor: float
    preferences: tuple[tuple[str, Operator, str], ...] = ()
    category_scope_label: str | None = None
    hard_inclusions: tuple[tuple[str, str], ...] = ()
    hard_exclusions: tuple[tuple[str, str], ...] = ()


CASES = (
    EvaluationCase(
        case_id="broad_hokkaido_winter",
        intent_shape="broad",
        user_utterance="我想去北海道，帮我找找有什么能买的。",
        q_lex="Hokkaido winter cold snow trip",
        q_sem=(
            "Products that could be useful to wear or bring for a winter trip to "
            "Hokkaido with cold weather and snow; open to different product types."
        ),
        transparency_anchor=0.10,
    ),
    EvaluationCase(
        case_id="broad_summer_wedding",
        intent_shape="broad",
        user_utterance="随便看看夏季婚礼穿什么。",
        q_lex="summer wedding outfit elegant ceremony",
        q_sem=(
            "Something suitable to wear to a summer wedding; elegant and lightweight, "
            "but the product type and final look are still open."
        ),
        transparency_anchor=0.20,
        preferences=(
            ("use_case", Operator.EQ, "ceremony"),
            ("feature", Operator.EQ, "lightweight"),
        ),
    ),
    EvaluationCase(
        case_id="broad_new_office_job",
        intent_shape="broad",
        user_utterance="我刚开始一份办公室工作，想看看有什么合适的。",
        q_lex="office work professional polished",
        q_sem=(
            "Something useful to wear or carry for a new office job, polished and "
            "professional; not sure what product type yet."
        ),
        transparency_anchor=0.25,
        preferences=(
            ("style", Operator.EQ, "professional"),
            ("use_case", Operator.EQ, "work"),
        ),
    ),
    EvaluationCase(
        case_id="mid_red_wedding_accessory",
        intent_shape="mid",
        user_utterance="想找红色的婚礼配饰，材质别太厚重。",
        q_lex="red wedding accessory lightweight",
        q_sem="A red lightweight accessory suitable for a wedding or ceremony.",
        transparency_anchor=0.55,
        preferences=(
            ("color", Operator.EQ, "red"),
            ("use_case", Operator.EQ, "wedding"),
            ("feature", Operator.EQ, "lightweight"),
        ),
    ),
    EvaluationCase(
        case_id="focused_black_snow_boots",
        intent_shape="focused",
        user_utterance="男士黑色防水保暖雪地靴，10 码。",
        q_lex="men black waterproof insulated snow boots size 10",
        q_sem="Men's black waterproof insulated snow boots in size 10.",
        transparency_anchor=0.90,
        preferences=(("feature", Operator.EQ, "insulated"),),
        category_scope_label="General footwear",
        hard_inclusions=(
            ("gender", "men"),
            ("color", "black"),
            ("feature", "waterproof"),
            ("size", "10"),
        ),
    ),
    EvaluationCase(
        case_id="focused_no_black_leather_heels",
        intent_shape="focused",
        user_utterance="女士红色皮质包头高跟鞋，不要黑色。",
        q_lex="women red leather closed toe high heel shoes",
        q_sem="Women's red leather closed-toe high heel shoes; exclude black products.",
        transparency_anchor=0.90,
        category_scope_label="General footwear",
        hard_inclusions=(
            ("gender", "women"),
            ("color", "red"),
            ("material", "leather"),
            ("style", "closed toe"),
        ),
        hard_exclusions=(("color", "black"),),
    ),
)

ABLATIONS = (
    ("dense_only", frozenset({RetrievalRoute.DENSE})),
    (
        "dense_lexical",
        frozenset({RetrievalRoute.DENSE, RetrievalRoute.LEXICAL}),
    ),
    (
        "dense_lexical_facet",
        frozenset(
            {
                RetrievalRoute.DENSE,
                RetrievalRoute.LEXICAL,
                RetrievalRoute.FACET,
            }
        ),
    ),
)


def main() -> int:
    args = _parse_args()
    policy = FormalRetrievalPolicy()
    controller = create_retrieval_controller(
        index_path=args.dense_index,
        release_dir=args.semantic_release,
        catalog_path=args.catalog,
        device=args.device,
        local_files_only=True,
        policy=policy,
    )
    release = load_catalog_semantic_release(args.semantic_release)
    metadata = _load_metadata(args.catalog)

    cases: list[dict[str, object]] = []
    for case in CASES:
        query = _compile_case(case, release)
        full = controller.search(query, transparency=case.transparency_anchor)
        variants = [
            _run_ablation(
                controller,
                full.routes,
                enabled_routes=enabled,
                name=name,
                transparency=case.transparency_anchor,
                metadata=metadata,
            )
            for name, enabled in ABLATIONS
        ]
        cases.append(
            {
                **asdict(case),
                "preferences": [
                    {"facet": facet, "operator": operator.value, "value": value}
                    for facet, operator, value in case.preferences
                ],
                "hard_inclusions": [
                    {"facet": facet, "value": value} for facet, value in case.hard_inclusions
                ],
                "hard_exclusions": [
                    {"facet": facet, "value": value} for facet, value in case.hard_exclusions
                ],
                "compiled_query": _query_log(query),
                "eligible_count": len(full.hard_mask.eligible_parent_asins),
                "hard_mask_trace": [asdict(item) for item in full.hard_mask.trace],
                "route_summary": [
                    {
                        "route": route.route.value,
                        "available": route.available,
                        "reason": route.reason,
                        "hit_count": len(route.hits),
                    }
                    for route in full.routes
                ],
                "variants": variants,
            }
        )

    payload = {
        "schema": "shopping-copilot/multi-route-evaluation/v0",
        "purpose": "architecture_ablation_on_real_50k_catalog",
        "transparency_note": (
            "T values are fixed experimental anchors; this experiment isolates retrieval "
            "behavior and does not claim to re-estimate T."
        ),
        "algorithm": {
            "flow": "hard mask -> route Top-80 -> RRF Top-80 -> T-aware vector MMR Top-10",
            "route_k": policy.route_k,
            "fusion_k": policy.fusion_k,
            "final_k": policy.final_k,
            "rrf_rank_constant": policy.rrf_rank_constant,
            "ablation_variants": [name for name, _ in ABLATIONS],
        },
        "bindings": {
            "catalog_id": controller.retriever.index.manifest.catalog_id,
            "catalog_semantic_release_id": (
                controller.retriever.index.manifest.catalog_semantic_release_id
            ),
            "dense_index_id": controller.retriever.index.index_id,
            "product_count": controller.retriever.index.manifest.product_count,
            "embedding_model": controller.retriever.index.manifest.embedding.model_id,
        },
        "cases": cases,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(_render_markdown(payload), encoding="utf-8")
    print(args.output_json)
    print(args.output_markdown)
    return 0


def _compile_case(case: EvaluationCase, release: object) -> CompiledQuery:
    preferences = tuple(
        CompiledRankingPreference(
            preference_id=f"{case.case_id}/preference/{index}",
            facet=facet,
            operator=operator,
            value=value,
            semantic_text=None,
            semantic_polarity=None,
            commitment=Commitment.SOFT,
            source=PreferenceSource.USER_EXPLICIT,
            reason=RankingReason.SOFT_COMMITMENT,
        )
        for index, (facet, operator, value) in enumerate(case.preferences, start=1)
    )
    category = ()
    if case.category_scope_label is not None:
        scope_id = next(
            item.id
            for item in release.category_registry.scopes
            if item.label == case.category_scope_label
        )
        category = (
            CompiledHardConstraint(
                preference_id=f"{case.case_id}/category",
                facet=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
                operator=Operator.EQ,
                value=scope_id,
                policy=ConstraintPolicy.VERIFIED_CATEGORY,
            ),
        )
    inclusions = tuple(
        CompiledHardConstraint(
            preference_id=f"{case.case_id}/inclusion/{index}",
            facet=facet,
            operator=Operator.EQ,
            value=value,
            policy=ConstraintPolicy.CLOSED_WORLD_RETRIEVAL_EVIDENCE,
        )
        for index, (facet, value) in enumerate(case.hard_inclusions, start=1)
    )
    exclusions = tuple(
        CompiledHardConstraint(
            preference_id=f"{case.case_id}/exclusion/{index}",
            facet=facet,
            operator=Operator.NEQ,
            value=value,
            policy=ConstraintPolicy.CLOSED_WORLD_RETRIEVAL_EVIDENCE,
        )
        for index, (facet, value) in enumerate(case.hard_exclusions, start=1)
    )
    return CompiledQuery(
        schema=COMPILED_QUERY_SCHEMA,
        compiler_version=QUERY_COMPILER_VERSION,
        catalog_id=release.manifest.catalog_id,
        catalog_semantic_release_id=release.release_id,
        category_graph_id=release.category_registry.category_graph_id,
        intent_version=1,
        q_lex=case.q_lex,
        q_sem=case.q_sem,
        search_ready=True,
        hard_constraints=(*category, *inclusions, *exclusions),
        ranking_preferences=preferences,
        dont_care_facets=(),
        directives=CompiledDirectives(
            diversity=DiversityDirective.AUTO,
            comparison_requested=False,
            explanation_requested=False,
        ),
        requires_clarification=False,
        clarification_reason=None,
        trace=(),
    )


def _run_ablation(
    controller: object,
    routes: tuple[RouteObservation, ...],
    *,
    enabled_routes: frozenset[RetrievalRoute],
    name: str,
    transparency: float,
    metadata: dict[str, dict[str, object]],
) -> dict[str, object]:
    selected_routes = tuple(item for item in routes if item.route in enabled_routes)
    fused = controller.fusion.fuse(selected_routes, top_k=controller.policy.fusion_k)
    relevance = normalized_fusion_relevance(fused)
    candidates = tuple(
        VectorCandidate(
            parent_asin=item.parent_asin,
            candidate_rank=item.rank,
            relevance=item_relevance,
        )
        for item, item_relevance in zip(fused, relevance, strict=True)
    )
    relevance_weight = controller.diversity_policy.relevance_weight(transparency)
    diversified = controller.reranker.rerank_candidates(
        candidates,
        top_k=controller.policy.final_k,
        relevance_weight=relevance_weight,
    )
    fused_by_asin = {item.parent_asin: item for item in fused}
    products = []
    for hit in diversified.hits:
        fused_item = fused_by_asin[hit.parent_asin]
        products.append(
            {
                **asdict(hit),
                "fusion_score": fused_item.fusion_score,
                "route_contributions": [
                    {
                        "route": item.route.value,
                        "rank": item.route_rank,
                        "raw_score": item.raw_score,
                    }
                    for item in fused_item.contributions
                ],
                **metadata[hit.parent_asin],
            }
        )
    return {
        "name": name,
        "enabled_routes": sorted(item.value for item in enabled_routes),
        "active_routes": [item.route.value for item in selected_routes if item.available],
        "fused_candidate_count": len(fused),
        "relevance_weight": relevance_weight,
        "metrics": _metrics(controller.retriever.index, products),
        "products": products,
    }


def _metrics(index: object, products: list[dict[str, object]]) -> dict[str, object]:
    if not products:
        return {
            "mean_pairwise_product_cosine": None,
            "unique_reporting_groups": 0,
            "reporting_groups": [],
            "unique_leaf_categories": 0,
            "multi_route_products": 0,
        }
    rows = [index.row_index(str(item["parent_asin"])) for item in products]
    vectors = index.vectors[rows]
    similarity = vectors @ vectors.T
    upper = similarity[np.triu_indices(len(rows), k=1)]
    return {
        "mean_pairwise_product_cosine": float(np.mean(upper)),
        "unique_reporting_groups": len({str(item["reporting_group"]) for item in products}),
        "reporting_groups": sorted({str(item["reporting_group"]) for item in products}),
        "unique_leaf_categories": len({str(item["leaf_category"]) for item in products}),
        "multi_route_products": sum(len(item["route_contributions"]) >= 2 for item in products),
        "mean_fused_relevance": float(np.mean([float(item["relevance"]) for item in products])),
        "maximum_fused_rank": max(int(item["candidate_rank"]) for item in products),
    }


def _query_log(query: CompiledQuery) -> dict[str, object]:
    return {
        "q_lex": query.q_lex,
        "q_sem": query.q_sem,
        "hard_constraints": [asdict(item) for item in query.hard_constraints],
        "ranking_preferences": [asdict(item) for item in query.ranking_preferences],
        "directives": asdict(query.directives),
    }


def _load_metadata(path: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line)
            categories = [str(value) for value in raw["categories"]]
            result[str(raw["parent_asin"])] = {
                "title": str(raw["title"]),
                "categories": categories,
                "leaf_category": categories[-1] if categories else "",
                "reporting_group": _reporting_group(categories),
            }
    return result


def _reporting_group(categories: list[str]) -> str:
    lowered = [item.casefold() for item in categories]
    for marker, group in (
        ("jewelry", "jewelry"),
        ("shoes", "footwear"),
        ("boot shop", "footwear"),
        ("watches", "watches"),
        ("handbags & wallets", "handbags_wallets"),
        ("luggage", "luggage"),
        ("clothing", "clothing"),
        ("accessories", "accessories"),
    ):
        if marker in lowered:
            return group
    return categories[1].casefold() if len(categories) > 1 else "other"


def _render_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Multi-route Retrieval v0: 50k 实测",
        "",
        "流程：共享 hard mask → 每路 Top-80 → RRF 融合 Top-80 → T-aware 向量 MMR Top-10。",
        "这里的 T 是用于隔离检索行为的实验锚点，不是重新计算出来的线上 T。",
        "",
    ]
    for case in payload["cases"]:
        lines.extend(
            [
                f"## {case['case_id']}",
                "",
                f"- 用户：{case['user_utterance']}",
                f"- `q_lex`：{case['q_lex']}",
                f"- `q_sem`：{case['q_sem']}",
                f"- T 锚点：{case['transparency_anchor']:.2f}",
                f"- hard mask 后商品数：{case['eligible_count']}",
                "",
                "| 方案 | 实际生效路线 | 向量分散度↓ | 大类数 | 叶子类数 | 多路共同命中 |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for variant in case["variants"]:
            metrics = variant["metrics"]
            pair = metrics["mean_pairwise_product_cosine"]
            pair_text = "n/a" if pair is None else f"{pair:.3f}"
            lines.append(
                f"| {variant['name']} | {', '.join(variant['active_routes'])} | "
                f"{pair_text} | {metrics['unique_reporting_groups']} | "
                f"{metrics['unique_leaf_categories']} | {metrics['multi_route_products']} |"
            )
        lines.extend(["", "三路最终 Top-10：", ""])
        final = case["variants"][-1]
        for product in final["products"]:
            routes = ", ".join(item["route"] for item in product["route_contributions"])
            lines.append(
                f"{product['rank']}. `{product['parent_asin']}` "
                f"{product['title']} — {product['reporting_group']} — {routes}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _json_default(value: object) -> object:
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"cannot encode {type(value).__name__}")


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
        default=ROOT / "artifacts/retrieval/multi-route-v0.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=ROOT / "artifacts/retrieval/multi-route-v0.md",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
