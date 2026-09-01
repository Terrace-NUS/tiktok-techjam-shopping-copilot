from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from shopping_copilot.application.quality_ranking import ApertureRankingCoordinator
from shopping_copilot.query_compiler import DiversityDirective
from shopping_copilot.retrieval import (
    FusedCandidate,
    RetrievalRoute,
    RouteContribution,
    VectorCandidate,
)
from shopping_copilot.retrieval.deepseek_ranking import (
    DeepSeekRankingTrace,
    QualityRankingMode,
)


class FakeQualityPipeline:
    def __init__(self, *, mode: QualityRankingMode, failure: Exception | None = None) -> None:
        self.mode = mode
        self.failure = failure
        self.calls: list[dict[str, object]] = []

    def rank(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.failure is not None:
            raise self.failure
        return SimpleNamespace(
            quality_ranking=SimpleNamespace(
                mode=self.mode,
                traces=(
                    DeepSeekRankingTrace(
                        response_id="rank-1",
                        model="deepseek-v4-flash",
                        prompt_tokens=120,
                        completion_tokens=30,
                        total_tokens=150,
                    ),
                ),
            )
        )


class FakeQualityFinalizer:
    def __init__(self, recommendations: tuple[str, ...]) -> None:
        self.recommendations = recommendations
        self.calls: list[dict[str, object]] = []

    def select(self, ranking: object, **kwargs: object) -> object:
        self.calls.append({"ranking": ranking, **kwargs})
        return SimpleNamespace(
            result=SimpleNamespace(
                hits=tuple(
                    SimpleNamespace(parent_asin=parent_asin) for parent_asin in self.recommendations
                )
            )
        )


class FailingFallbackReranker:
    def rerank(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("the BGE fallback should not run")


class FakeFallbackReranker:
    def rerank(self, *args: object, **kwargs: object) -> object:
        return SimpleNamespace(
            candidates=(
                VectorCandidate(parent_asin="B", candidate_rank=1, relevance=0.9),
                VectorCandidate(parent_asin="A", candidate_rank=2, relevance=0.7),
            )
        )


class FakeSelector:
    def select(self, *args: object, **kwargs: object) -> object:
        return SimpleNamespace(
            hits=(
                SimpleNamespace(parent_asin="B"),
                SimpleNamespace(parent_asin="A"),
            )
        )


def test_quality_ranking_preserves_routes_and_uses_t_aware_finalizer() -> None:
    pipeline = FakeQualityPipeline(mode=QualityRankingMode.DEEPSEEK)
    finalizer = FakeQualityFinalizer(("C", "A"))
    coordinator = ApertureRankingCoordinator(
        documents={"A": "A", "B": "B", "C": "C"},
        quality_pipeline=cast(Any, pipeline),
        quality_finalizer=cast(Any, finalizer),
        fallback_reranker=cast(Any, FailingFallbackReranker()),
        fallback_selector=cast(Any, FakeSelector()),
    )

    result = coordinator.rank(
        request_id="session:turn:1:intent:1",
        intent=cast(Any, SimpleNamespace()),
        compiled_query=cast(Any, _compiled()),
        retrieval=cast(Any, _retrieval()),
        top_k=2,
    )

    assert result.mode == "deepseek_quality_dpp"
    assert result.recommendations == ("C", "A")
    assert result.prompt_tokens == 120
    assert result.completion_tokens == 30
    call = pipeline.calls[0]
    assert [item.parent_asin for item in cast(tuple[VectorCandidate, ...], call["candidates"])] == [
        "A",
        "B",
        "C",
    ]
    assert cast(dict[str, tuple[str, ...]], call["routes"]) == {
        "A": ("dense", "lexical"),
        "B": ("facet",),
        "C": ("dense",),
    }
    assert finalizer.calls[0]["transparency"] == 0.2
    assert finalizer.calls[0]["directive"] is DiversityDirective.INCREASE


def test_provider_level_bge_fallback_still_gets_t_aware_dpp() -> None:
    pipeline = FakeQualityPipeline(mode=QualityRankingMode.BGE_FALLBACK)
    finalizer = FakeQualityFinalizer(("B", "C"))
    coordinator = ApertureRankingCoordinator(
        documents={"A": "A", "B": "B", "C": "C"},
        quality_pipeline=cast(Any, pipeline),
        quality_finalizer=cast(Any, finalizer),
        fallback_reranker=cast(Any, FailingFallbackReranker()),
        fallback_selector=cast(Any, FakeSelector()),
    )

    result = coordinator.rank(
        request_id="session:turn:1:intent:1",
        intent=cast(Any, SimpleNamespace()),
        compiled_query=cast(Any, _compiled()),
        retrieval=cast(Any, _retrieval()),
        top_k=2,
    )

    assert result.mode == "deepseek_bge_fallback_dpp"
    assert result.recommendations == ("B", "C")
    assert result.quality_failure is None


def test_unexpected_quality_failure_degrades_to_old_bge_dpp() -> None:
    pipeline = FakeQualityPipeline(
        mode=QualityRankingMode.DEEPSEEK,
        failure=RuntimeError("broken ranking prompt"),
    )
    coordinator = ApertureRankingCoordinator(
        documents={"A": "A", "B": "B", "C": "C"},
        quality_pipeline=cast(Any, pipeline),
        quality_finalizer=cast(Any, FakeQualityFinalizer(("C",))),
        fallback_reranker=cast(Any, FakeFallbackReranker()),
        fallback_selector=cast(Any, FakeSelector()),
    )

    result = coordinator.rank(
        request_id="session:turn:1:intent:1",
        intent=cast(Any, SimpleNamespace()),
        compiled_query=cast(Any, _compiled()),
        retrieval=cast(Any, _retrieval()),
        top_k=2,
    )

    assert result.mode == "bge_dpp_after_failure"
    assert result.recommendations == ("B", "A")
    assert result.quality_failure is not None
    assert result.quality_failure.stage == "quality_pipeline"
    assert result.quality_failure.error_type == "RuntimeError"
    assert result.fallback_failure is None


def _compiled() -> SimpleNamespace:
    return SimpleNamespace(
        q_sem="summer hiking shoes",
        directives=SimpleNamespace(diversity=DiversityDirective.INCREASE),
    )


def _retrieval() -> SimpleNamespace:
    return SimpleNamespace(
        transparency=0.2,
        relevance_weight=0.42,
        recall_trace=object(),
        fused_candidates=(
            FusedCandidate(
                parent_asin="A",
                rank=1,
                fusion_score=0.9,
                contributions=(
                    RouteContribution(
                        route=RetrievalRoute.DENSE,
                        route_rank=1,
                        raw_score=0.8,
                    ),
                    RouteContribution(
                        route=RetrievalRoute.LEXICAL,
                        route_rank=2,
                        raw_score=-1.0,
                    ),
                ),
            ),
            FusedCandidate(
                parent_asin="B",
                rank=2,
                fusion_score=0.6,
                contributions=(
                    RouteContribution(
                        route=RetrievalRoute.FACET,
                        route_rank=1,
                        raw_score=1.0,
                    ),
                ),
            ),
            FusedCandidate(
                parent_asin="C",
                rank=3,
                fusion_score=0.3,
                contributions=(
                    RouteContribution(
                        route=RetrievalRoute.DENSE,
                        route_rank=3,
                        raw_score=0.4,
                    ),
                ),
            ),
        ),
        hits=(
            SimpleNamespace(parent_asin="A"),
            SimpleNamespace(parent_asin="B"),
            SimpleNamespace(parent_asin="C"),
        ),
    )
