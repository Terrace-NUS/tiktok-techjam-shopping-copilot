"""Rank-only fusion for heterogeneous retrieval routes."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .routing import RetrievalRoute, RouteObservation


@dataclass(frozen=True, slots=True, kw_only=True)
class RouteContribution:
    """One route's auditable contribution to a fused candidate."""

    route: RetrievalRoute
    route_rank: int
    raw_score: float


@dataclass(frozen=True, slots=True, kw_only=True)
class FusedCandidate:
    """One candidate after reciprocal-rank fusion."""

    parent_asin: str
    rank: int
    fusion_score: float
    contributions: tuple[RouteContribution, ...]


class ReciprocalRankFusion:
    """Fuse available route rankings without comparing incompatible raw scores."""

    def __init__(self, *, rank_constant: int = 60) -> None:
        if type(rank_constant) is not int or rank_constant <= 0:
            raise ValueError("rank_constant must be a positive integer")
        self.rank_constant = rank_constant

    def fuse(
        self,
        observations: tuple[RouteObservation, ...],
        *,
        top_k: int,
    ) -> tuple[FusedCandidate, ...]:
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        active = tuple(item for item in observations if item.available)
        seen_routes = [item.route for item in observations]
        if len(seen_routes) != len(set(seen_routes)):
            raise ValueError("observations contains duplicate routes")

        contributions: dict[str, list[RouteContribution]] = {}
        scores: dict[str, float] = {}
        for observation in active:
            for expected_rank, hit in enumerate(observation.hits, start=1):
                if hit.rank != expected_rank:
                    raise ValueError("route ranks must be contiguous")
                contribution = 1.0 / (self.rank_constant + hit.rank)
                scores[hit.parent_asin] = scores.get(hit.parent_asin, 0.0) + contribution
                contributions.setdefault(hit.parent_asin, []).append(
                    RouteContribution(
                        route=observation.route,
                        route_rank=hit.rank,
                        raw_score=hit.raw_score,
                    )
                )

        ordered = sorted(scores, key=lambda item: (-scores[item], item))[:top_k]
        return tuple(
            FusedCandidate(
                parent_asin=parent_asin,
                rank=rank,
                fusion_score=float(scores[parent_asin]),
                contributions=tuple(
                    sorted(
                        contributions[parent_asin],
                        key=lambda item: item.route.value,
                    )
                ),
            )
            for rank, parent_asin in enumerate(ordered, start=1)
        )


def normalized_fusion_relevance(candidates: tuple[FusedCandidate, ...]) -> tuple[float, ...]:
    """Scale positive RRF scores to a stable [0, 1] relevance range."""

    if not candidates:
        return ()
    maximum = candidates[0].fusion_score
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("fusion scores must be positive and finite")
    relevance = tuple(float(item.fusion_score / maximum) for item in candidates)
    if any(not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in relevance):
        raise ValueError("normalized fusion relevance is invalid")
    return relevance
