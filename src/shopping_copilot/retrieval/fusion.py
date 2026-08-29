"""Rank-only fusion for heterogeneous retrieval routes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .routing import RetrievalRoute, RouteObservation


@dataclass(frozen=True, slots=True, kw_only=True)
class RouteContribution:
    """One route's auditable contribution to a fused candidate."""

    route: RetrievalRoute
    route_rank: int
    raw_score: float


@dataclass(frozen=True, slots=True, kw_only=True)
class FusedCandidate:
    """One candidate after a deterministic multi-route fusion stage."""

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


class RelativeScoreFusion:
    """Fuse route-local min-max scores while preserving score orientation.

    Dense and facet scores are larger-is-better, while SQLite FTS5 BM25 is
    smaller-is-better.  Each available route is normalized independently before
    addition, so heterogeneous raw score units are never compared directly.
    ``agreement_power=1`` yields a CombMNZ-style consensus bonus.
    """

    def __init__(
        self,
        *,
        route_weights: Mapping[RetrievalRoute, float] | None = None,
        agreement_power: float = 0.0,
    ) -> None:
        weights = (
            {route: 1.0 for route in RetrievalRoute}
            if route_weights is None
            else dict(route_weights)
        )
        if set(weights) != set(RetrievalRoute):
            raise ValueError("route_weights must define every retrieval route")
        for route, weight in weights.items():
            if type(route) is not RetrievalRoute:
                raise TypeError("route_weights contains an invalid route")
            if type(weight) is not float or not math.isfinite(weight) or weight < 0.0:
                raise ValueError("route weights must be finite non-negative floats")
        if not any(weight > 0.0 for weight in weights.values()):
            raise ValueError("at least one route weight must be positive")
        if (
            type(agreement_power) is not float
            or not math.isfinite(agreement_power)
            or agreement_power < 0.0
        ):
            raise ValueError("agreement_power must be a finite non-negative float")
        self.route_weights = MappingProxyType(weights)
        self.agreement_power = agreement_power

    def fuse(
        self,
        observations: tuple[RouteObservation, ...],
        *,
        top_k: int,
    ) -> tuple[FusedCandidate, ...]:
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        seen_routes = [item.route for item in observations]
        if len(seen_routes) != len(set(seen_routes)):
            raise ValueError("observations contains duplicate routes")

        contributions: dict[str, list[RouteContribution]] = {}
        normalized_scores: dict[str, float] = {}
        for observation in observations:
            if not observation.available or not observation.hits:
                continue
            route_values: list[float] = []
            for expected_rank, hit in enumerate(observation.hits, start=1):
                if hit.rank != expected_rank:
                    raise ValueError("route ranks must be contiguous")
                oriented = (
                    -hit.raw_score if observation.route is RetrievalRoute.LEXICAL else hit.raw_score
                )
                route_values.append(oriented)
            minimum = min(route_values)
            maximum = max(route_values)
            span = maximum - minimum
            weight = self.route_weights[observation.route]
            for hit, oriented in zip(observation.hits, route_values, strict=True):
                relative = 1.0 if span == 0.0 else (oriented - minimum) / span
                normalized_scores[hit.parent_asin] = (
                    normalized_scores.get(hit.parent_asin, 0.0) + weight * relative
                )
                contributions.setdefault(hit.parent_asin, []).append(
                    RouteContribution(
                        route=observation.route,
                        route_rank=hit.rank,
                        raw_score=hit.raw_score,
                    )
                )

        scores = {
            parent_asin: score * len(contributions[parent_asin]) ** self.agreement_power
            for parent_asin, score in normalized_scores.items()
        }
        ordered = sorted(scores, key=lambda item: (-scores[item], item))[:top_k]
        return tuple(
            FusedCandidate(
                parent_asin=parent_asin,
                rank=rank,
                fusion_score=float(scores[parent_asin]),
                contributions=tuple(
                    sorted(contributions[parent_asin], key=lambda item: item.route.value)
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
