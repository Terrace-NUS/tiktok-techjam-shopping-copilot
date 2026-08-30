"""End-to-end BGE shortlist and DeepSeek quality-ranking boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter

from shopping_copilot.query_compiler import CompiledQuery
from shopping_copilot.session_context import IntentState

from ..dense import DenseIndex
from ..ranking import CrossEncoderRankingResult, CrossEncoderRelevanceReranker
from ..transparency_recall import TransparencyRecallTrace
from ..vector_diversity import VectorCandidate
from .models import (
    DeepSeekRankingRequest,
    QualityRankingResult,
    RankingShortlist,
    RankingUserProfile,
)
from .service import DeepSeekQualityRanker
from .shortlist import DirectionAwareShortlister


@dataclass(frozen=True, slots=True, kw_only=True)
class QualityPipelineTimings:
    bge_ms: float
    shortlist_ms: float
    deepseek_ms: float
    total_ms: float


@dataclass(frozen=True, slots=True, kw_only=True)
class QualityPipelineResult:
    bge_ranking: CrossEncoderRankingResult
    shortlist: RankingShortlist
    quality_ranking: QualityRankingResult
    timings: QualityPipelineTimings


class DeepSeekQualityPipeline:
    """Turn the retrieval pool into individually scored ranking candidates."""

    def __init__(
        self,
        *,
        index: DenseIndex,
        bge_reranker: CrossEncoderRelevanceReranker,
        deepseek_ranker: DeepSeekQualityRanker,
        shortlist_k: int = 48,
        protected_per_direction: int = 6,
    ) -> None:
        self._bge_reranker = bge_reranker
        self._deepseek_ranker = deepseek_ranker
        self._shortlister = DirectionAwareShortlister(
            index=index,
            top_k=shortlist_k,
            protected_per_direction=protected_per_direction,
        )

    def rank(
        self,
        *,
        request_id: str,
        intent: IntentState,
        compiled_query: CompiledQuery,
        candidates: tuple[VectorCandidate, ...],
        documents: Mapping[str, str],
        recall_trace: TransparencyRecallTrace | None,
        routes: Mapping[str, tuple[str, ...]] | None = None,
        user_profile: RankingUserProfile | None = None,
        bge_prior_weight: float = 0.25,
        bge_batch_size: int = 16,
    ) -> QualityPipelineResult:
        total_started = perf_counter()

        bge_started = perf_counter()
        bge_ranking = self._bge_reranker.rerank(
            compiled_query.q_sem,
            candidates,
            documents=documents,
            prior_weight=bge_prior_weight,
            batch_size=bge_batch_size,
        )
        bge_ms = _elapsed_ms(bge_started)

        shortlist_started = perf_counter()
        shortlist = self._shortlister.select(
            bge_ranking,
            documents=documents,
            recall_trace=recall_trace,
            routes=routes,
        )
        shortlist_ms = _elapsed_ms(shortlist_started)

        deepseek_started = perf_counter()
        quality_ranking = self._deepseek_ranker.rank(
            DeepSeekRankingRequest(
                request_id=request_id,
                intent=intent,
                compiled_query=compiled_query,
                shortlist=shortlist,
                user_profile=user_profile,
            )
        )
        deepseek_ms = _elapsed_ms(deepseek_started)
        return QualityPipelineResult(
            bge_ranking=bge_ranking,
            shortlist=shortlist,
            quality_ranking=quality_ranking,
            timings=QualityPipelineTimings(
                bge_ms=bge_ms,
                shortlist_ms=shortlist_ms,
                deepseek_ms=deepseek_ms,
                total_ms=_elapsed_ms(total_started),
            ),
        )


def _elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1_000.0
