"""Real-world ranking orchestration with explicit, auditable fallbacks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from shopping_copilot.query_compiler import CompiledQuery
from shopping_copilot.retrieval.controller import FormalRetrievalHit, FormalRetrievalResult
from shopping_copilot.retrieval.deepseek_ranking import (
    DeepSeekQualityPipeline,
    DeepSeekRankingTrace,
    FinalQualitySlate,
    QualityPipelineResult,
    QualityRankingMode,
    RankingUserProfile,
    TransparencyAwareDPPFinalizer,
)
from shopping_copilot.retrieval.fusion import normalized_fusion_relevance
from shopping_copilot.retrieval.ranking import (
    CrossEncoderRankingResult,
    CrossEncoderRelevanceReranker,
    GreedyDPPSelector,
    VectorSlateResult,
)
from shopping_copilot.retrieval.vector_diversity import VectorCandidate
from shopping_copilot.session_context import IntentState


@dataclass(frozen=True, slots=True, kw_only=True)
class RankingFailure:
    """A serializable failure boundary rather than an opaque exception object."""

    stage: str
    error_type: str
    message: str


@dataclass(frozen=True, slots=True, kw_only=True)
class RealWorldRankingResult:
    """One final recommendation slate plus every ranking stage used to produce it."""

    mode: str
    recommendations: tuple[str, ...]
    quality_pipeline: QualityPipelineResult | None
    quality_slate: FinalQualitySlate | None
    fallback_cross_encoder: CrossEncoderRankingResult | None
    fallback_slate: VectorSlateResult | None
    formal_mmr_fallback_hits: tuple[FormalRetrievalHit, ...]
    quality_failure: RankingFailure | None
    fallback_failure: RankingFailure | None

    @property
    def prompt_tokens(self) -> int:
        return sum(trace.prompt_tokens or 0 for trace in self._deepseek_traces())

    @property
    def completion_tokens(self) -> int:
        return sum(trace.completion_tokens or 0 for trace in self._deepseek_traces())

    def _deepseek_traces(self) -> tuple[DeepSeekRankingTrace, ...]:
        if self.quality_pipeline is None:
            return ()
        return self.quality_pipeline.quality_ranking.traces


class RealWorldRankingCoordinator:
    """Run quality ranking, then degrade through BGE and formal retrieval safely."""

    def __init__(
        self,
        *,
        documents: Mapping[str, str],
        fallback_selector: GreedyDPPSelector,
        quality_pipeline: DeepSeekQualityPipeline | None = None,
        quality_finalizer: TransparencyAwareDPPFinalizer | None = None,
        fallback_reranker: CrossEncoderRelevanceReranker | None = None,
    ) -> None:
        if not isinstance(documents, Mapping):
            raise TypeError("documents must be a mapping")
        if (quality_pipeline is None) != (quality_finalizer is None):
            raise ValueError("quality_pipeline and quality_finalizer must be configured together")
        self._documents = documents
        self._fallback_selector = fallback_selector
        self._quality_pipeline = quality_pipeline
        self._quality_finalizer = quality_finalizer
        self._fallback_reranker = fallback_reranker

    def rank(
        self,
        *,
        request_id: str,
        intent: IntentState,
        compiled_query: CompiledQuery,
        retrieval: FormalRetrievalResult,
        top_k: int,
        user_profile: RankingUserProfile | None = None,
    ) -> RealWorldRankingResult:
        """Produce a final Top-K without ever reintroducing hard-mask rejects."""

        if type(request_id) is not str or not request_id.strip():
            raise ValueError("request_id must be non-empty")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be positive")

        candidates = _vector_candidates(retrieval)
        routes = _candidate_routes(retrieval)
        quality_failure: RankingFailure | None = None
        if (
            self._quality_pipeline is not None
            and self._quality_finalizer is not None
            and candidates
        ):
            try:
                quality = self._quality_pipeline.rank(
                    request_id=request_id,
                    intent=intent,
                    compiled_query=compiled_query,
                    candidates=candidates,
                    documents=self._documents,
                    recall_trace=retrieval.recall_trace,
                    routes=routes,
                    user_profile=user_profile,
                    bge_prior_weight=0.25,
                    bge_batch_size=16,
                )
                quality_slate = self._quality_finalizer.select(
                    quality.quality_ranking,
                    transparency=float(retrieval.transparency),
                    top_k=top_k,
                    directive=compiled_query.directives.diversity,
                )
                mode = (
                    "deepseek_quality_dpp"
                    if quality.quality_ranking.mode is QualityRankingMode.DEEPSEEK
                    else "deepseek_bge_fallback_dpp"
                )
                return RealWorldRankingResult(
                    mode=mode,
                    recommendations=tuple(
                        hit.parent_asin for hit in quality_slate.result.hits
                    ),
                    quality_pipeline=quality,
                    quality_slate=quality_slate,
                    fallback_cross_encoder=None,
                    fallback_slate=None,
                    formal_mmr_fallback_hits=retrieval.hits,
                    quality_failure=None,
                    fallback_failure=None,
                )
            except Exception as error:
                quality_failure = _failure("quality_pipeline", error)

        if self._fallback_reranker is not None and candidates:
            try:
                cross_encoder = self._fallback_reranker.rerank(
                    compiled_query.q_sem,
                    candidates,
                    documents=self._documents,
                    prior_weight=0.25,
                    batch_size=32,
                )
                fallback_slate = self._fallback_selector.select(
                    cross_encoder.candidates,
                    top_k=top_k,
                    relevance_weight=float(retrieval.relevance_weight),
                )
                return RealWorldRankingResult(
                    mode=("bge_dpp" if quality_failure is None else "bge_dpp_after_failure"),
                    recommendations=tuple(hit.parent_asin for hit in fallback_slate.hits),
                    quality_pipeline=None,
                    quality_slate=None,
                    fallback_cross_encoder=cross_encoder,
                    fallback_slate=fallback_slate,
                    formal_mmr_fallback_hits=retrieval.hits,
                    quality_failure=quality_failure,
                    fallback_failure=None,
                )
            except Exception as error:
                fallback_failure = _failure("bge_dpp", error)
        else:
            fallback_failure = None

        recommendations = tuple(hit.parent_asin for hit in retrieval.hits[:top_k])
        return RealWorldRankingResult(
            mode=(
                "formal_mmr"
                if quality_failure is None and fallback_failure is None
                else "formal_mmr_fallback"
            ),
            recommendations=recommendations,
            quality_pipeline=None,
            quality_slate=None,
            fallback_cross_encoder=None,
            fallback_slate=None,
            formal_mmr_fallback_hits=retrieval.hits,
            quality_failure=quality_failure,
            fallback_failure=fallback_failure,
        )


def _vector_candidates(retrieval: FormalRetrievalResult) -> tuple[VectorCandidate, ...]:
    fused = retrieval.fused_candidates
    if not fused:
        return ()
    relevance = normalized_fusion_relevance(fused)
    return tuple(
        VectorCandidate(
            parent_asin=item.parent_asin,
            candidate_rank=item.rank,
            relevance=item_relevance,
        )
        for item, item_relevance in zip(fused, relevance, strict=True)
    )


def _candidate_routes(retrieval: FormalRetrievalResult) -> dict[str, tuple[str, ...]]:
    return {
        item.parent_asin: tuple(contribution.route.value for contribution in item.contributions)
        for item in retrieval.fused_candidates
    }


def _failure(stage: str, error: Exception) -> RankingFailure:
    return RankingFailure(
        stage=stage,
        error_type=type(error).__name__,
        message=str(error),
    )
