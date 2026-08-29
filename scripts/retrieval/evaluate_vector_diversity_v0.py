"""Evaluate category-blind vector MMR on broad and focused shopping requests."""

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

from shopping_copilot.retrieval import (  # noqa: E402
    VectorDiversityPolicy,
    VectorMMRReranker,
    create_dense_retriever,
)


@dataclass(frozen=True, slots=True)
class PromptCase:
    case_id: str
    intent_shape: str
    user_utterance: str
    q_sem: str


PROMPTS = (
    PromptCase(
        case_id="vague_hokkaido_literal",
        intent_shape="vague",
        user_utterance="我想去北海道，帮我找找有什么能买的。",
        q_sem=(
            "products that could be useful to wear or bring for a trip to Hokkaido, Japan; "
            "open to different kinds of products"
        ),
    ),
    PromptCase(
        case_id="vague_hokkaido_contextualized",
        intent_shape="vague",
        user_utterance="我想去北海道，帮我找找有什么能买的。",
        q_sem=(
            "products that could be useful to wear or bring for a winter trip to Hokkaido, "
            "Japan, with cold weather and snow; open to different kinds of products"
        ),
    ),
    PromptCase(
        case_id="vague_summer_wedding",
        intent_shape="vague",
        user_utterance="随便看看夏季婚礼穿什么。",
        q_sem="something suitable to wear to a summer wedding; still figuring out the look",
    ),
    PromptCase(
        case_id="vague_beach_trip",
        intent_shape="vague",
        user_utterance="我要去海边度假，看看有什么可以买的。",
        q_sem="useful things to wear or bring for a beach vacation; still deciding what to buy",
    ),
    PromptCase(
        case_id="vague_new_office_job",
        intent_shape="vague",
        user_utterance="我刚开始一份办公室工作，想看看有什么合适的。",
        q_sem=(
            "something useful for starting a new office job and looking polished and "
            "professional; not sure what kind of product yet"
        ),
    ),
    PromptCase(
        case_id="vague_elegant_gift",
        intent_shape="vague",
        user_utterance="想给喜欢优雅风格的人买个礼物，还没想好买什么。",
        q_sem="an elegant fashion gift; not sure what kind of product yet",
    ),
    PromptCase(
        case_id="focused_snow_boots",
        intent_shape="focused",
        user_utterance="男士黑色防水保暖雪地靴，10 码。",
        q_sem="men's black waterproof insulated snow boots, size 10",
    ),
    PromptCase(
        case_id="focused_red_heels",
        intent_shape="focused",
        user_utterance="女士红色皮质包头高跟鞋。",
        q_sem="women's red leather closed-toe high heel shoes",
    ),
    PromptCase(
        case_id="focused_pearl_studs",
        intent_shape="focused",
        user_utterance="小号纯银珍珠耳钉。",
        q_sem="small pearl stud earrings in sterling silver",
    ),
)


def main() -> int:
    args = _parse_args()
    retriever = create_dense_retriever(
        index_path=args.dense_index,
        release_dir=args.semantic_release,
        catalog_path=args.catalog,
        device=args.device,
        local_files_only=True,
    )
    metadata = _load_metadata(args.catalog)
    reranker = VectorMMRReranker(index=retriever.index)
    policy = VectorDiversityPolicy()
    candidate_windows = (80, 500, 2_000)
    transparency_anchors = (0.10, 0.50, 0.90)

    cases: list[dict[str, object]] = []
    for prompt in PROMPTS:
        scores = retriever.score(prompt.q_sem)
        dense_products = [
            {
                "parent_asin": hit.parent_asin,
                "rank": hit.rank,
                "dense_rank": hit.rank,
                "relevance": hit.score,
                "maximum_similarity_to_selected": 0.0,
                "mmr_score": hit.score,
                **metadata[hit.parent_asin],
            }
            for hit in retriever.index.select_top_k(scores, top_k=10)
        ]
        variants: list[dict[str, object]] = []
        for candidate_k in candidate_windows:
            for transparency in transparency_anchors:
                relevance_weight = policy.relevance_weight(transparency)
                result = reranker.rerank(
                    scores,
                    candidate_k=candidate_k,
                    top_k=10,
                    relevance_weight=relevance_weight,
                )
                products = [
                    {
                        **asdict(hit),
                        **metadata[hit.parent_asin],
                    }
                    for hit in result.hits
                ]
                variants.append(
                    {
                        "candidate_k": candidate_k,
                        "transparency_anchor": transparency,
                        "relevance_weight": relevance_weight,
                        "metrics": _metrics(retriever.index, products),
                        "products": products,
                    }
                )
        cases.append(
            {
                **asdict(prompt),
                "dense_baseline": {
                    "metrics": _metrics(retriever.index, dense_products),
                    "products": dense_products,
                },
                "variants": variants,
            }
        )

    payload = {
        "schema": "shopping-copilot/vector-diversity-evaluation/v0",
        "algorithm": {
            "name": "category_blind_cosine_mmr",
            "candidate_windows": list(candidate_windows),
            "transparency_anchors": list(transparency_anchors),
            "policy": asdict(policy),
            "category_usage": "reporting_only",
        },
        "bindings": {
            "dense_index_id": retriever.index.index_id,
            "catalog_id": retriever.index.manifest.catalog_id,
            "catalog_semantic_release_id": (retriever.index.manifest.catalog_semantic_release_id),
            "embedding_model": retriever.index.manifest.embedding.model_id,
            "embedding_revision": retriever.index.manifest.embedding.model_revision,
        },
        "cases": cases,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(_render_markdown(payload), encoding="utf-8")
    return 0


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
        default=ROOT / "artifacts/retrieval/vector-diversity-v0.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=ROOT / "artifacts/retrieval/vector-diversity-v0.md",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


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
    if "jewelry" in lowered:
        return "jewelry"
    if "shoes" in lowered or "boot shop" in lowered:
        return "footwear"
    if "watches" in lowered:
        return "watches"
    if "handbags & wallets" in lowered:
        return "handbags_wallets"
    if "luggage" in lowered:
        return "luggage"
    if "clothing" in lowered:
        return "clothing"
    if "accessories" in lowered:
        return "accessories"
    return categories[1].casefold() if len(categories) > 1 else "other"


def _metrics(index: object, products: list[dict[str, object]]) -> dict[str, object]:
    dense_index = index
    rows = [dense_index.row_index(str(item["parent_asin"])) for item in products]
    vectors = dense_index.vectors[rows]
    similarity = vectors @ vectors.T
    upper = similarity[np.triu_indices(len(rows), k=1)]
    return {
        "mean_query_relevance": float(np.mean([float(item["relevance"]) for item in products])),
        "minimum_query_relevance": float(min(float(item["relevance"]) for item in products)),
        "mean_pairwise_product_cosine": float(np.mean(upper)),
        "maximum_dense_rank": max(int(item["dense_rank"]) for item in products),
        "unique_reporting_groups": len({str(item["reporting_group"]) for item in products}),
        "reporting_groups": sorted({str(item["reporting_group"]) for item in products}),
        "unique_leaf_categories": len({str(item["leaf_category"]) for item in products}),
    }


def _render_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Category-blind Vector Diversity v0",
        "",
        "Category and facet metadata are used only to audit the output. Selection uses dense",
        "query-product cosine and product-product cosine MMR exclusively.",
        "",
    ]
    for case in payload["cases"]:
        lines.extend(
            [
                f"## {case['case_id']}",
                "",
                f"- User: {case['user_utterance']}",
                f"- `q_sem`: {case['q_sem']}",
                "",
                "| candidate K | T anchor | rel weight | mean rel | pair cosine | groups | leaves | max dense rank |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        baseline = case["dense_baseline"]["metrics"]
        lines.append(
            "| baseline | — | 1.00 | {mean_query_relevance:.3f} | "
            "{mean_pairwise_product_cosine:.3f} | {unique_reporting_groups} | "
            "{unique_leaf_categories} | {maximum_dense_rank} |".format(**baseline)
        )
        for variant in case["variants"]:
            metric = variant["metrics"]
            lines.append(
                "| {candidate_k} | {transparency_anchor:.2f} | {relevance_weight:.2f} | "
                "{mean_query_relevance:.3f} | {mean_pairwise_product_cosine:.3f} | "
                "{unique_reporting_groups} | {unique_leaf_categories} | "
                "{maximum_dense_rank} |".format(**variant, **metric)
            )
        lines.extend(["", "### Wide candidate window examples", ""])
        for variant in case["variants"]:
            if variant["candidate_k"] != 2_000:
                continue
            lines.append(
                f"#### T={variant['transparency_anchor']:.2f}, "
                f"relevance weight={variant['relevance_weight']:.2f}"
            )
            lines.append("")
            for product in variant["products"]:
                lines.append(
                    f"{product['rank']}. `{product['parent_asin']}` "
                    f"[{product['reporting_group']} / {product['leaf_category']}] "
                    f"rel={product['relevance']:.3f}, dense-rank={product['dense_rank']}: "
                    f"{product['title']}"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
