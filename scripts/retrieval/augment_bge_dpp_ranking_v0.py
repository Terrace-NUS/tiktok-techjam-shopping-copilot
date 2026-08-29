"""Add BGE+DPP variants to an existing same-pool ranking evaluation artifact."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.retrieval.evaluate_ranking_strategies_v0 import (  # noqa: E402
    BGE_MODEL,
    BGE_REVISION,
    SelectedHit,
    _aggregate_public,
    _compact_document,
    _fusion_candidates,
    _load_metadata,
    _natural_cases,
    _public_cases,
    _render_markdown,
    _variant_payload,
)
from shopping_copilot.catalog.semantic.release import (  # noqa: E402
    load_catalog_semantic_release,
)
from shopping_copilot.retrieval import (  # noqa: E402
    CrossEncoderRelevanceReranker,
    GreedyDPPSelector,
    ReciprocalRankFusion,
    SentenceTransformerCrossEncoderScorer,
    create_retrieval_controller,
    load_product_documents,
)


def main() -> int:
    args = _parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("schema") != "shopping-copilot/ranking-strategy-evaluation/v0":
        raise ValueError("input is not a ranking-strategy v0 evaluation")

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
    public_count = sum(case["source"] == "public_simulator" for case in payload["cases"])
    cases = [
        *_natural_cases(controller, release),
        *_public_cases(
            controller,
            args.public_set,
            args.catalog,
            limit=public_count,
        ),
    ]
    stored = {str(case["case_id"]): case for case in payload["cases"]}
    if {case.case_id for case in cases} != set(stored):
        raise ValueError("reconstructed cases do not match the source artifact")

    bge = CrossEncoderRelevanceReranker(
        scorer=SentenceTransformerCrossEncoderScorer(
            BGE_MODEL,
            revision=BGE_REVISION,
            device=args.device,
            local_files_only=args.local_models_only,
            max_length=int(payload["parameters"]["cross_encoder_max_length"]),
        )
    )
    selector = GreedyDPPSelector(index=controller.retriever.index)
    latencies: list[float] = []
    started = time.perf_counter()
    for ordinal, case in enumerate(cases, start=1):
        candidate_k = int(payload["parameters"]["candidate_k"])
        final_k = int(payload["parameters"]["final_k"])
        batch_size = int(payload["parameters"]["batch_size"])
        rrf_candidates, rrf_fused = _fusion_candidates(
            ReciprocalRankFusion(rank_constant=60),
            case.routes,
            candidate_k=candidate_k,
        )
        model_started = time.perf_counter()
        bge_result = bge.rerank(
            case.query,
            rrf_candidates,
            documents=documents,
            prior_weight=float(payload["parameters"]["cross_encoder_prior_weight"]),
            batch_size=batch_size,
        )
        latency = (time.perf_counter() - model_started) * 1000.0
        latencies.append(latency)
        rrf_scores = {item.parent_asin: item.relevance for item in rrf_candidates}
        bge_scores = {hit.parent_asin: hit.normalized_model_score for hit in bge_result.hits}
        contributions = {item.parent_asin: item.contributions for item in rrf_fused}
        sweep_values = (
            ((case.transparency, "observed"),)
            if case.source == "natural"
            else ((0.2, "low"), (0.8, "high"))
        )
        additions = []
        for transparency, suffix in sweep_values:
            weight = controller.diversity_policy.relevance_weight(float(transparency))
            selected = selector.select(
                bge_result.candidates,
                top_k=final_k,
                relevance_weight=weight,
            )
            additions.append(
                _variant_payload(
                    f"bge_dpp_{suffix}",
                    tuple(SelectedHit(**asdict(hit)) for hit in selected.hits),
                    transparency=float(transparency),
                    metadata=metadata,
                    rrf_scores=rrf_scores,
                    qwen_scores={},
                    bge_scores=bge_scores,
                    contributions=contributions,
                    target=case.target_parent_asin,
                    index=controller.retriever.index,
                )
            )
        case_payload = stored[case.case_id]
        case_payload["variants"] = [
            variant
            for variant in case_payload["variants"]
            if not str(variant["name"]).startswith("bge_dpp_")
        ] + additions
        print(
            f"[{ordinal}/{len(cases)}] {case.source}:{case.case_id} bge={latency:.1f}ms",
            flush=True,
        )

    payload["public_summary"] = _aggregate_public(payload["cases"])
    payload["bge_dpp_augmentation"] = {
        "model": bge.scorer.model_id,
        "case_count": len(cases),
        "wall_seconds": time.perf_counter() - started,
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "max_ms": float(np.max(latencies)),
        "qwen_scores_in_added_variants": "not_recomputed",
    }
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(_render_markdown(payload), encoding="utf-8")
    print(args.output_json, flush=True)
    print(args.output_markdown, flush=True)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "artifacts/retrieval/ranking-strategy-v0.json",
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--local-models-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
