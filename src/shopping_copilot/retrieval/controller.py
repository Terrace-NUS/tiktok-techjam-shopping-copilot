"""Formal multi-route retrieval controller."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from time import perf_counter

from shopping_copilot.query_compiler import CompiledQuery, DiversityDirective

from .dense import DenseRetriever, DenseScoreSnapshot
from .errors import CompiledQueryNotSearchableError
from .fusion import (
    FusedCandidate,
    ReciprocalRankFusion,
    RouteContribution,
    normalized_fusion_relevance,
)
from .hard_mask import HardMaskResolver, ResolvedHardMask
from .lexical import LexicalProbe
from .routing import (
    FacetRoute,
    RetrievalRoute,
    RouteHit,
    RouteObservation,
    dense_route_observation,
    lexical_route_observation,
)
from .transparency_recall import (
    DirectionalDenseCandidate,
    TransparencyAwareDenseRecall,
    TransparencyRecallPolicy,
    TransparencyRecallTrace,
)
from .vector_diversity import VectorCandidate, VectorDiversityPolicy, VectorMMRReranker


class RecallStrategy(str, Enum):
    """Candidate-generation policy selected for one controller instance."""

    LEGACY_SINGLE_CENTER = "legacy_single_center"
    TRANSPARENCY_MULTI_CENTER = "transparency_multi_center"


@dataclass(frozen=True, slots=True, kw_only=True)
class FormalRetrievalPolicy:
    """Small policy surface for the hackathon retrieval story."""

    route_k: int = 80
    fusion_k: int = 300
    final_k: int = 10
    rrf_rank_constant: int = 60
    directive_adjustment: float = 0.15
    recall_strategy: RecallStrategy = RecallStrategy.TRANSPARENCY_MULTI_CENTER

    def __post_init__(self) -> None:
        for name, value in (
            ("route_k", self.route_k),
            ("fusion_k", self.fusion_k),
            ("final_k", self.final_k),
            ("rrf_rank_constant", self.rrf_rank_constant),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.final_k > self.fusion_k:
            raise ValueError("final_k must not exceed fusion_k")
        if type(self.recall_strategy) is not RecallStrategy:
            raise TypeError("recall_strategy must be a RecallStrategy")
        if (
            type(self.directive_adjustment) is not float
            or not math.isfinite(self.directive_adjustment)
            or not 0.0 <= self.directive_adjustment <= 1.0
        ):
            raise ValueError("directive_adjustment must be a finite float in [0, 1]")


@dataclass(frozen=True, slots=True, kw_only=True)
class FormalRetrievalHit:
    """One final result with fusion and diversity evidence intact."""

    parent_asin: str
    rank: int
    fused_rank: int
    fusion_score: float
    normalized_relevance: float
    maximum_similarity_to_selected: float
    mmr_score: float
    contributions: tuple[RouteContribution, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class FormalRetrievalResult:
    """Complete audit trail for one formal retrieval run."""

    transparency: float
    relevance_weight: float
    hard_mask: ResolvedHardMask
    routes: tuple[RouteObservation, ...]
    fused_candidates: tuple[FusedCandidate, ...]
    hits: tuple[FormalRetrievalHit, ...]
    recall_trace: TransparencyRecallTrace | None
    timings: FormalRetrievalTimings


@dataclass(frozen=True, slots=True, kw_only=True)
class FormalRetrievalTimings:
    """Wall-clock retrieval stages in milliseconds."""

    hard_mask_ms: float
    dense_score_ms: float
    recall_planning_ms: float
    lexical_ms: float
    facet_ms: float
    fusion_ms: float
    ranking_ms: float
    total_ms: float


class RetrievalController:
    """Run hard mask -> three routes -> RRF -> T-aware vector MMR."""

    def __init__(
        self,
        *,
        retriever: DenseRetriever,
        lexical_route: LexicalProbe,
        facet_route: FacetRoute,
        hard_mask_resolver: HardMaskResolver,
        policy: FormalRetrievalPolicy | None = None,
        diversity_policy: VectorDiversityPolicy | None = None,
        transparency_recall_policy: TransparencyRecallPolicy | None = None,
    ) -> None:
        if type(retriever) is not DenseRetriever:
            raise TypeError("retriever must be an exact DenseRetriever")
        if type(lexical_route) is not LexicalProbe:
            raise TypeError("lexical_route must be an exact LexicalProbe")
        if type(facet_route) is not FacetRoute:
            raise TypeError("facet_route must be an exact FacetRoute")
        if type(hard_mask_resolver) is not HardMaskResolver:
            raise TypeError("hard_mask_resolver must be an exact HardMaskResolver")
        resolved_policy = FormalRetrievalPolicy() if policy is None else policy
        resolved_diversity_policy = (
            VectorDiversityPolicy() if diversity_policy is None else diversity_policy
        )
        if type(resolved_policy) is not FormalRetrievalPolicy:
            raise TypeError("policy must be an exact FormalRetrievalPolicy")
        if type(resolved_diversity_policy) is not VectorDiversityPolicy:
            raise TypeError("diversity_policy must be an exact VectorDiversityPolicy")
        resolved_recall_policy = (
            TransparencyRecallPolicy()
            if transparency_recall_policy is None
            else transparency_recall_policy
        )
        if type(resolved_recall_policy) is not TransparencyRecallPolicy:
            raise TypeError(
                "transparency_recall_policy must be an exact TransparencyRecallPolicy"
            )
        expected = retriever.index.parent_asins
        if lexical_route.parent_asins != frozenset(expected):
            raise ValueError("lexical route and dense index contain different products")
        if facet_route.parent_asins != expected:
            raise ValueError("facet route and dense index contain different products")
        if lexical_route.probe_k != resolved_policy.route_k:
            raise ValueError("lexical route Top-K differs from formal retrieval policy")
        maximum_lexical_budget = (
            resolved_recall_policy.lexical_budget_at_low_transparency
            + resolved_recall_policy.lexical_budget_range
        )
        if (
            resolved_policy.recall_strategy is RecallStrategy.TRANSPARENCY_MULTI_CENTER
            and lexical_route.probe_k < maximum_lexical_budget
        ):
            raise ValueError("lexical route Top-K cannot satisfy transparency recall budgets")
        if (
            resolved_policy.recall_strategy is RecallStrategy.TRANSPARENCY_MULTI_CENTER
            and resolved_policy.fusion_k < resolved_recall_policy.candidate_pool_k
        ):
            raise ValueError("fusion_k cannot truncate the transparency-aware candidate pool")

        self.retriever = retriever
        self.lexical_route = lexical_route
        self.facet_route = facet_route
        self.hard_mask_resolver = hard_mask_resolver
        self.policy = resolved_policy
        self.diversity_policy = resolved_diversity_policy
        self.transparency_recall_policy = resolved_recall_policy
        self.fusion = ReciprocalRankFusion(rank_constant=resolved_policy.rrf_rank_constant)
        self.reranker = VectorMMRReranker(index=retriever.index)
        self.transparency_recall = TransparencyAwareDenseRecall(
            index=retriever.index,
            policy=resolved_recall_policy,
        )

    def search(
        self,
        query: CompiledQuery,
        *,
        transparency: float,
    ) -> FormalRetrievalResult:
        """Execute all available routes under one eligibility and diversity policy."""

        if type(query) is not CompiledQuery:
            raise TypeError("query must be an exact CompiledQuery")
        if not query.search_ready or not query.q_sem.strip():
            raise CompiledQueryNotSearchableError("compiled query is not search-ready")
        if type(transparency) is not float or not math.isfinite(transparency):
            raise ValueError("transparency must be a finite float in [0, 1]")
        if not 0.0 <= transparency <= 1.0:
            raise ValueError("transparency must be a finite float in [0, 1]")

        total_started = perf_counter()
        hard_mask_started = perf_counter()
        hard_mask = self.hard_mask_resolver.resolve(query)
        hard_mask_ms = _elapsed_ms(hard_mask_started)

        dense_score_started = perf_counter()
        dense_scores = self.retriever.score(query.q_sem)
        dense_score_ms = _elapsed_ms(dense_score_started)

        recall_trace: TransparencyRecallTrace | None = None
        if self.policy.recall_strategy is RecallStrategy.TRANSPARENCY_MULTI_CENTER:
            (
                routes,
                recall_trace,
                recall_planning_ms,
                lexical_ms,
                facet_ms,
            ) = self._transparency_aware_routes(
                query,
                hard_mask=hard_mask,
                dense_scores=dense_scores,
                transparency=transparency,
            )
        else:
            recall_started = perf_counter()
            dense = self.retriever.index.rank_scores(
                dense_scores,
                top_k=self.policy.route_k,
                eligible_mask=hard_mask.eligible_mask,
            )
            recall_planning_ms = _elapsed_ms(recall_started)
            lexical_started = perf_counter()
            lexical = self.lexical_route.observe(
                query.q_lex,
                eligible_parent_asins=hard_mask.eligible_parent_asins,
            )
            lexical_ms = _elapsed_ms(lexical_started)
            facet_started = perf_counter()
            facet = self.facet_route.search(
                query,
                eligible_parent_asins=hard_mask.eligible_parent_asins,
                relaxed_constraints=hard_mask.relaxed_constraints,
                top_k=self.policy.route_k,
            )
            facet_ms = _elapsed_ms(facet_started)
            routes = (
                dense_route_observation(dense),
                lexical_route_observation(lexical),
                facet,
            )

        fusion_started = perf_counter()
        fused = self.fusion.fuse(routes, top_k=self.policy.fusion_k)
        fusion_ms = _elapsed_ms(fusion_started)
        relevance = normalized_fusion_relevance(fused)
        candidates = tuple(
            VectorCandidate(
                parent_asin=item.parent_asin,
                candidate_rank=item.rank,
                relevance=item_relevance,
            )
            for item, item_relevance in zip(fused, relevance, strict=True)
        )
        relevance_weight = self._relevance_weight(
            transparency,
            query.directives.diversity,
        )
        ranking_started = perf_counter()
        diversified = self.reranker.rerank_candidates(
            candidates,
            top_k=self.policy.final_k,
            relevance_weight=relevance_weight,
        )
        ranking_ms = _elapsed_ms(ranking_started)
        by_asin = {item.parent_asin: item for item in fused}
        hits = tuple(
            FormalRetrievalHit(
                parent_asin=hit.parent_asin,
                rank=hit.rank,
                fused_rank=hit.candidate_rank,
                fusion_score=by_asin[hit.parent_asin].fusion_score,
                normalized_relevance=hit.relevance,
                maximum_similarity_to_selected=hit.maximum_similarity_to_selected,
                mmr_score=hit.mmr_score,
                contributions=by_asin[hit.parent_asin].contributions,
            )
            for hit in diversified.hits
        )
        return FormalRetrievalResult(
            transparency=transparency,
            relevance_weight=relevance_weight,
            hard_mask=hard_mask,
            routes=routes,
            fused_candidates=fused,
            hits=hits,
            recall_trace=recall_trace,
            timings=FormalRetrievalTimings(
                hard_mask_ms=hard_mask_ms,
                dense_score_ms=dense_score_ms,
                recall_planning_ms=recall_planning_ms,
                lexical_ms=lexical_ms,
                facet_ms=facet_ms,
                fusion_ms=fusion_ms,
                ranking_ms=ranking_ms,
                total_ms=_elapsed_ms(total_started),
            ),
        )

    def _transparency_aware_routes(
        self,
        query: CompiledQuery,
        *,
        hard_mask: ResolvedHardMask,
        dense_scores: DenseScoreSnapshot,
        transparency: float,
    ) -> tuple[
        tuple[RouteObservation, ...],
        TransparencyRecallTrace,
        float,
        float,
        float,
    ]:
        recall_started = perf_counter()
        dense_recall = self.transparency_recall.recall(
            dense_scores,
            eligible_mask=hard_mask.eligible_mask,
            transparency=transparency,
        )
        budgets = self.transparency_recall_policy.budgets(transparency)
        recall_planning_ms = _elapsed_ms(recall_started)

        lexical_started = perf_counter()
        lexical = _truncate_route(
            lexical_route_observation(
                self.lexical_route.observe(
                    query.q_lex,
                    eligible_parent_asins=hard_mask.eligible_parent_asins,
                )
            ),
            top_k=budgets.lexical,
        )
        lexical_ms = _elapsed_ms(lexical_started)

        facet_started = perf_counter()
        facet = self.facet_route.search(
            query,
            eligible_parent_asins=hard_mask.eligible_parent_asins,
            relaxed_constraints=hard_mask.relaxed_constraints,
            top_k=budgets.facet,
        )
        facet_ms = _elapsed_ms(facet_started)

        selected_dense = list(dense_recall.candidates[: budgets.dense])
        union = {
            *(item.parent_asin for item in selected_dense),
            *(item.parent_asin for item in lexical.hits),
            *(item.parent_asin for item in facet.hits),
        }
        dense_refill_count = 0
        for candidate in dense_recall.candidates[budgets.dense :]:
            if len(union) >= self.transparency_recall_policy.candidate_pool_k:
                break
            if candidate.parent_asin in union:
                continue
            selected_dense.append(candidate)
            union.add(candidate.parent_asin)
            dense_refill_count += 1

        dense = _directional_dense_route(tuple(selected_dense))
        routes = (dense, lexical, facet)
        trace = TransparencyRecallTrace(
            policy_id=dense_recall.policy_id,
            transparency=transparency,
            candidate_pool_k=self.transparency_recall_policy.candidate_pool_k,
            candidate_pool_count=len(union),
            requested_direction_count=dense_recall.requested_direction_count,
            actual_direction_count=len(dense_recall.directions),
            frontier_requested_k=dense_recall.frontier_requested_k,
            frontier_count=dense_recall.frontier_count,
            budgets=budgets,
            actual_dense_count=len(dense.hits),
            actual_lexical_count=len(lexical.hits),
            actual_facet_count=len(facet.hits),
            dense_refill_count=dense_refill_count,
            directions=dense_recall.directions,
            dense_candidates=tuple(selected_dense),
            planner_timings=dense_recall.timings,
        )
        return routes, trace, recall_planning_ms, lexical_ms, facet_ms

    def _relevance_weight(
        self,
        transparency: float,
        directive: DiversityDirective,
    ) -> float:
        weight = self.diversity_policy.relevance_weight(transparency)
        if directive is DiversityDirective.INCREASE:
            weight -= self.policy.directive_adjustment
        elif directive is DiversityDirective.DECREASE:
            weight += self.policy.directive_adjustment
        return float(min(1.0, max(0.0, weight)))


def _directional_dense_route(
    candidates: tuple[DirectionalDenseCandidate, ...],
) -> RouteObservation:
    return RouteObservation(
        route=RetrievalRoute.DENSE,
        requested_top_k=len(candidates),
        available=bool(candidates),
        reason=None if candidates else "no_eligible_documents",
        hits=tuple(
            RouteHit(
                parent_asin=candidate.parent_asin,
                rank=rank,
                raw_score=candidate.combined_score,
            )
            for rank, candidate in enumerate(candidates, start=1)
        ),
    )


def _truncate_route(observation: RouteObservation, *, top_k: int) -> RouteObservation:
    hits = tuple(
        RouteHit(
            parent_asin=hit.parent_asin,
            rank=rank,
            raw_score=hit.raw_score,
            evidence_ids=hit.evidence_ids,
        )
        for rank, hit in enumerate(observation.hits[:top_k], start=1)
    )
    return RouteObservation(
        route=observation.route,
        requested_top_k=top_k,
        available=observation.available,
        reason=observation.reason,
        hits=hits,
    )


def _elapsed_ms(started: float) -> float:
    return float((perf_counter() - started) * 1_000.0)
