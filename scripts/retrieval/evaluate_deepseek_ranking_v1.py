"""Run BGE shortlist -> DeepSeek quality ranking on real 50k catalog cases."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for source in (ROOT, SRC):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from scripts.retrieval.evaluate_multi_route_v0 import (  # noqa: E402
    CASES,
    EvaluationCase,
    _compile_case,
    _load_metadata,
)
from shopping_copilot.catalog.semantic.release import (  # noqa: E402
    load_catalog_semantic_release,
)
from shopping_copilot.query_compiler import CompiledQuery  # noqa: E402
from shopping_copilot.retrieval import (  # noqa: E402
    CrossEncoderRelevanceReranker,
    SentenceTransformerCrossEncoderScorer,
    VectorCandidate,
    create_retrieval_controller,
    load_product_documents,
    normalized_fusion_relevance,
)
from shopping_copilot.retrieval.deepseek_ranking import (  # noqa: E402
    DeepSeekQualityPipeline,
    DeepSeekQualityRanker,
    DeepSeekRankingConfig,
    DeepSeekRankingProvider,
    FinalQualitySlate,
    QualityPipelineResult,
    RankingUserProfile,
    TransparencyAwareDPPFinalizer,
)
from shopping_copilot.session_context import (  # noqa: E402
    Commitment,
    IntentState,
    Operator,
    Preference,
    PreferenceSource,
)

BGE_MODEL = "BAAI/bge-reranker-v2-m3"
BGE_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
DEFAULT_CASE_IDS = (
    "broad_hokkaido_winter",
    "mid_red_wedding_accessory",
    "focused_black_snow_boots",
)


def main() -> int:
    args = _parse_args()
    selected_cases = _select_cases(args.case)
    api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError("DeepSeek API key file is empty")

    initialized = time.perf_counter()
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
        item.parent_asin: item.text
        for item in load_product_documents(
            args.catalog,
            expected_parent_asins=set(controller.retriever.index.parent_asins),
        )
    }
    bge = CrossEncoderRelevanceReranker(
        scorer=SentenceTransformerCrossEncoderScorer(
            BGE_MODEL,
            revision=BGE_REVISION,
            device=args.device,
            local_files_only=True,
            max_length=args.max_length,
        )
    )
    provider = DeepSeekRankingProvider(
        api_key=api_key,
        config=DeepSeekRankingConfig(
            timeout_seconds=args.timeout_seconds,
            max_tokens=args.max_tokens,
        ),
    )
    pipeline = DeepSeekQualityPipeline(
        index=controller.retriever.index,
        bge_reranker=bge,
        deepseek_ranker=DeepSeekQualityRanker(provider=provider),
        shortlist_k=args.shortlist_k,
        protected_per_direction=args.protected_per_direction,
    )
    finalizer = TransparencyAwareDPPFinalizer(index=controller.retriever.index)
    initialization_ms = _elapsed_ms(initialized)

    case_logs: list[dict[str, object]] = []
    for ordinal, case in enumerate(selected_cases, start=1):
        compiled = _compile_case(case, release)
        intent = _intent_from_case(case)
        profile = _evaluation_profile(case)
        retrieval_started = time.perf_counter()
        recalled = controller.search(compiled, transparency=case.transparency_anchor)
        retrieval_ms = _elapsed_ms(retrieval_started)
        relevance = normalized_fusion_relevance(recalled.fused_candidates)
        candidates = tuple(
            VectorCandidate(
                parent_asin=item.parent_asin,
                candidate_rank=item.rank,
                relevance=score,
            )
            for item, score in zip(recalled.fused_candidates, relevance, strict=True)
        )
        routes = {
            item.parent_asin: tuple(
                contribution.route.value for contribution in item.contributions
            )
            for item in recalled.fused_candidates
        }
        ranked = pipeline.rank(
            request_id=f"deepseek-ranking-v1/{case.case_id}",
            intent=intent,
            compiled_query=compiled,
            candidates=candidates,
            documents=documents,
            recall_trace=recalled.recall_trace,
            routes=routes,
            user_profile=profile,
            bge_batch_size=args.batch_size,
        )
        final_started = time.perf_counter()
        final_slate = finalizer.select(
            ranked.quality_ranking,
            transparency=case.transparency_anchor,
            top_k=10,
            directive=compiled.directives.diversity,
        )
        final_slate_ms = _elapsed_ms(final_started)
        case_log = _case_log(
            case,
            intent=intent,
            compiled=compiled,
            profile=profile,
            retrieval_ms=retrieval_ms,
            candidate_count=len(candidates),
            ranked=ranked,
            final_slate=final_slate,
            final_slate_ms=final_slate_ms,
            metadata=metadata,
        )
        case_logs.append(case_log)
        print(
            f"[{ordinal}/{len(selected_cases)}] {case.case_id}: "
            f"candidates={len(candidates)} shortlist={len(ranked.shortlist.cards)} "
            f"mode={ranked.quality_ranking.mode.value} "
            f"BGE={ranked.timings.bge_ms:.1f}ms "
            f"DeepSeek={ranked.timings.deepseek_ms:.1f}ms",
            flush=True,
        )

    payload = {
        "schema": "shopping-copilot/deepseek-quality-ranking-evaluation/v1",
        "purpose": "real_catalog_contract_and_behavior_smoke_test",
        "architecture": (
            "T-aware 300-candidate recall -> BGE direction-protected shortlist -> "
            "one DeepSeek native tool call -> 0.8 DeepSeek + 0.2 BGE quality"
        ),
        "guardrails": [
            "Current Session Context is authoritative over optional user_profile.",
            "DeepSeek judges individual fit and cannot see T, BGE score, route, or order.",
            "DeepSeek does not choose diversity or the final slate.",
            "Invalid DeepSeek output gets one repair attempt, then BGE fallback.",
        ],
        "bindings": {
            "catalog_id": controller.retriever.index.manifest.catalog_id,
            "dense_index_id": controller.retriever.index.index_id,
            "product_count": controller.retriever.index.manifest.product_count,
            "embedding_model": controller.retriever.index.manifest.embedding.model_id,
            "bge_model": bge.scorer.model_id,
            "deepseek_model": "deepseek-v4-flash",
        },
        "parameters": {
            "shortlist_k": args.shortlist_k,
            "protected_per_direction": args.protected_per_direction,
            "deepseek_weight": 0.8,
            "bge_weight": 0.2,
        },
        "initialization_ms": initialization_ms,
        "cases": case_logs,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(_markdown(payload), encoding="utf-8")
    print(args.output_json.resolve(), flush=True)
    print(args.output_markdown.resolve(), flush=True)
    return 0


def _intent_from_case(case: EvaluationCase) -> IntentState:
    inputs: list[tuple[str, Operator, str, Commitment]] = [
        (facet, operator, value, Commitment.SOFT)
        for facet, operator, value in case.preferences
    ]
    inputs.extend(
        (facet, Operator.EQ, value, Commitment.HARD)
        for facet, value in case.hard_inclusions
    )
    inputs.extend(
        (facet, Operator.NEQ, value, Commitment.HARD)
        for facet, value in case.hard_exclusions
    )
    preferences = tuple(
        Preference(
            id=f"p_1_1_{index}",
            facet=facet,
            operator=operator,
            value=value,
            semantic_text=None,
            semantic_polarity=None,
            commitment=commitment,
            source=PreferenceSource.USER_EXPLICIT,
            source_turn=1,
            evidence_text=f"{facet} {operator.value} {value}",
            interpretation_confidence=1.0,
        )
        for index, (facet, operator, value, commitment) in enumerate(inputs)
    )
    return IntentState(
        goal=case.q_sem,
        preferences=preferences,
        dont_care_facets=frozenset(),
        version=1,
    )


def _evaluation_profile(case: EvaluationCase) -> RankingUserProfile | None:
    if case.case_id == "mid_red_wedding_accessory":
        return RankingUserProfile(
            schema="shopping-copilot/user-profile/demo-v0",
            version=1,
            payload={
                "stable_preferences": {"favorite_colors": ["blue"]},
                "evaluation_note": (
                    "Deliberate conflict: the current session explicitly requests red."
                ),
            },
        )
    if case.case_id == "broad_hokkaido_winter":
        return RankingUserProfile(
            schema="shopping-copilot/user-profile/demo-v0",
            version=1,
            payload={"stable_preferences": {"shopping_style": ["practical", "comfortable"]}},
        )
    return None


def _case_log(
    case: EvaluationCase,
    *,
    intent: IntentState,
    compiled: CompiledQuery,
    profile: RankingUserProfile | None,
    retrieval_ms: float,
    candidate_count: int,
    ranked: QualityPipelineResult,
    final_slate: FinalQualitySlate,
    final_slate_ms: float,
    metadata: dict[str, dict[str, object]],
) -> dict[str, object]:
    quality = ranked.quality_ranking
    shortlist_by_asin = {item.parent_asin: item for item in ranked.shortlist.cards}
    top_products = []
    for hit in quality.hits[:10]:
        card = shortlist_by_asin[hit.parent_asin]
        top_products.append(
            {
                **asdict(hit),
                "verdict": None if hit.verdict is None else hit.verdict.value,
                "direction_id": card.direction_id,
                "routes": list(card.routes),
                "metadata": metadata[hit.parent_asin],
            }
        )
    verdict_counts: dict[str, int] = {}
    for hit in quality.hits:
        key = "none" if hit.verdict is None else hit.verdict.value
        verdict_counts[key] = verdict_counts.get(key, 0) + 1
    quality_by_asin = {item.parent_asin: item for item in quality.hits}
    final_products = []
    for hit in final_slate.result.hits:
        quality_hit = quality_by_asin[hit.parent_asin]
        final_products.append(
            {
                **asdict(hit),
                "quality_rank": quality_hit.rank,
                "deepseek_fit": quality_hit.deepseek_fit,
                "verdict": (
                    None if quality_hit.verdict is None else quality_hit.verdict.value
                ),
                "reason": quality_hit.reason,
                "metadata": metadata[hit.parent_asin],
            }
        )
    return {
        "case_id": case.case_id,
        "intent_shape": case.intent_shape,
        "transparency": case.transparency_anchor,
        "session_context": _intent_payload(intent),
        "compiled_query": asdict(compiled),
        "user_profile": None if profile is None else profile.as_payload(),
        "candidate_count": candidate_count,
        "shortlist": [asdict(item) for item in ranked.shortlist.cards],
        "result": {
            "mode": quality.mode.value,
            "attempts": quality.attempts,
            "fallback_reason": quality.fallback_reason,
            "verdict_counts": verdict_counts,
            "trace": [asdict(item) for item in quality.traces],
            "all_quality_hits": [
                {
                    **asdict(item),
                    "verdict": None if item.verdict is None else item.verdict.value,
                }
                for item in quality.hits
            ],
            "top_10": top_products,
            "final_slate": {
                "transparency": final_slate.transparency,
                "relevance_weight": final_slate.relevance_weight,
                "method": final_slate.result.method,
                "products": final_products,
            },
        },
        "timings_ms": {
            "retrieval": retrieval_ms,
            **asdict(ranked.timings),
            "final_slate_ms": final_slate_ms,
        },
    }


def _intent_payload(intent: IntentState) -> dict[str, object]:
    return {
        "version": intent.version,
        "goal": intent.goal,
        "preferences": [
            {
                "id": item.id,
                "facet": item.facet,
                "operator": None if item.operator is None else item.operator.value,
                "value": item.value,
                "commitment": item.commitment.value,
                "source": item.source.value,
                "evidence_text": item.evidence_text,
            }
            for item in intent.preferences
        ],
        "dont_care_facets": sorted(intent.dont_care_facets),
    }


def _markdown(payload: dict[str, object]) -> str:
    cases = cast(list[dict[str, object]], payload["cases"])
    lines = [
        "# DeepSeek Quality Ranking v1 - Real Catalog Smoke Test",
        "",
        "BGE reduces the recall pool and protects semantic directions. DeepSeek then "
        "judges every surviving product independently. This is not the final diverse slate.",
        "",
        "| Case | T | Candidates | Shortlist | Mode | Attempts | Strong | Possible | Weak | BGE ms | DeepSeek ms |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in cases:
        result = cast(dict[str, object], case["result"])
        counts = cast(dict[str, int], result["verdict_counts"])
        timings = cast(dict[str, float], case["timings_ms"])
        shortlist = cast(list[dict[str, object]], case["shortlist"])
        lines.append(
            f"| {case['case_id']} | {case['transparency']:.2f} | "
            f"{case['candidate_count']} | {len(shortlist)} | "
            f"{result['mode']} | {result['attempts']} | "
            f"{counts.get('strong_match', 0)} | {counts.get('possible_match', 0)} | "
            f"{counts.get('weak_match', 0)} | {timings['bge_ms']:.1f} | "
            f"{timings['deepseek_ms']:.1f} |"
        )
        lines.extend(["", f"## {case['case_id']}", ""])
        final_slate = cast(dict[str, object], result["final_slate"])
        final_products = cast(list[dict[str, object]], final_slate["products"])
        lines.append(
            f"Final DPP relevance weight: `{final_slate['relevance_weight']:.3f}`."
        )
        lines.append("")
        for hit in final_products[:5]:
            metadata = cast(dict[str, object], hit["metadata"])
            title = metadata.get("title", "")
            lines.append(
                f"- #{hit['rank']} `{hit['parent_asin']}` ({hit['relevance']:.3f}, "
                f"{hit['verdict']}): {title} — {hit['reason']}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _select_cases(case_ids: list[str]) -> tuple[EvaluationCase, ...]:
    requested = tuple(case_ids) if case_ids else DEFAULT_CASE_IDS
    by_id = {item.case_id: item for item in CASES}
    unknown = set(requested) - set(by_id)
    if unknown:
        raise ValueError(f"unknown case IDs: {sorted(unknown)}")
    return tuple(by_id[item] for item in requested)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1_000.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--api-key-file", type=Path, default=ROOT / "dpskapi")
    parser.add_argument(
        "--dense-index", type=Path, default=ROOT / "artifacts/retrieval/dense-v0"
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
        default=ROOT / "artifacts/retrieval/deepseek-ranking-evaluation-v1.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=ROOT / "artifacts/retrieval/deepseek-ranking-evaluation-v1.md",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shortlist-k", type=int, default=48)
    parser.add_argument("--protected-per-direction", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
