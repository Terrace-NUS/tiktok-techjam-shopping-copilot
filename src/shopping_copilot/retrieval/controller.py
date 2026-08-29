"""Formal multi-route retrieval controller."""

from __future__ import annotations

import math
from dataclasses import dataclass

from shopping_copilot.query_compiler import CompiledQuery, DiversityDirective

from .dense import DenseRetriever
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
    RouteObservation,
    dense_route_observation,
    lexical_route_observation,
)
from .vector_diversity import VectorCandidate, VectorDiversityPolicy, VectorMMRReranker


@dataclass(frozen=True, slots=True, kw_only=True)
class FormalRetrievalPolicy:
    """Small policy surface for the hackathon retrieval story."""

    route_k: int = 80
    fusion_k: int = 80
    final_k: int = 10
    rrf_rank_constant: int = 60
    directive_adjustment: float = 0.15

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
        expected = retriever.index.parent_asins
        if lexical_route.parent_asins != frozenset(expected):
            raise ValueError("lexical route and dense index contain different products")
        if facet_route.parent_asins != expected:
            raise ValueError("facet route and dense index contain different products")
        if lexical_route.probe_k != resolved_policy.route_k:
            raise ValueError("lexical route Top-K differs from formal retrieval policy")

        self.retriever = retriever
        self.lexical_route = lexical_route
        self.facet_route = facet_route
        self.hard_mask_resolver = hard_mask_resolver
        self.policy = resolved_policy
        self.diversity_policy = resolved_diversity_policy
        self.fusion = ReciprocalRankFusion(rank_constant=resolved_policy.rrf_rank_constant)
        self.reranker = VectorMMRReranker(index=retriever.index)

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

        hard_mask = self.hard_mask_resolver.resolve(query)
        dense = self.retriever.search_with_scores(
            query.q_sem,
            top_k=self.policy.route_k,
            eligible_mask=hard_mask.eligible_mask,
        )
        lexical = self.lexical_route.observe(
            query.q_lex,
            eligible_parent_asins=hard_mask.eligible_parent_asins,
        )
        facet = self.facet_route.search(
            query,
            eligible_parent_asins=hard_mask.eligible_parent_asins,
            relaxed_constraints=hard_mask.relaxed_constraints,
            top_k=self.policy.route_k,
        )
        routes = (
            dense_route_observation(dense),
            lexical_route_observation(lexical),
            facet,
        )
        fused = self.fusion.fuse(routes, top_k=self.policy.fusion_k)
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
        diversified = self.reranker.rerank_candidates(
            candidates,
            top_k=self.policy.final_k,
            relevance_weight=relevance_weight,
        )
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
        )

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
