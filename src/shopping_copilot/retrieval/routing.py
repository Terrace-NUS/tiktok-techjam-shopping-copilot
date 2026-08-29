"""Auditable candidate routes for formal multi-route retrieval."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal, cast

from shopping_copilot.query_compiler import (
    CompiledHardConstraint,
    CompiledQuery,
    CompiledRankingPreference,
)
from shopping_copilot.session_context import Operator
from shopping_copilot.session_context.models import ScalarValue

from .dense import DenseSearchResult
from .errors import CompiledQueryBindingError
from .evidence import SUPPORTED_FACETS, RetrievalEvidenceIndex
from .lexical import LexicalProbeObservation


class RetrievalRoute(str, Enum):
    """Stable names for the three formal candidate generators."""

    DENSE = "dense"
    LEXICAL = "lexical"
    FACET = "facet"


@dataclass(frozen=True, slots=True, kw_only=True)
class RouteHit:
    """One route-local result before rank fusion."""

    parent_asin: str
    rank: int
    raw_score: float
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_route_hit(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class RouteObservation:
    """A complete, possibly unavailable, observation from one route."""

    route: RetrievalRoute
    requested_top_k: int
    available: bool
    reason: str | None
    hits: tuple[RouteHit, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class _FacetCondition:
    evidence_id: str
    facet: str
    values: tuple[str, ...]
    excludes: bool


class FacetRoute:
    """Rank products using only structured preferences verified by catalog evidence.

    This route does not infer new facets. It consumes the exact structured values
    emitted by Query Understanding. Positive matches create candidates; soft negative
    matches only penalize them. Hard exclusions have already been removed by the shared
    hard mask.
    """

    def __init__(self, *, evidence_index: RetrievalEvidenceIndex) -> None:
        if type(evidence_index) is not RetrievalEvidenceIndex:
            raise TypeError("evidence_index must be a RetrievalEvidenceIndex")
        self.evidence_index = evidence_index

    @property
    def parent_asins(self) -> tuple[str, ...]:
        return self.evidence_index.parent_asins

    def search(
        self,
        query: CompiledQuery,
        *,
        eligible_parent_asins: tuple[str, ...],
        relaxed_constraints: tuple[CompiledHardConstraint, ...] = (),
        top_k: int,
    ) -> RouteObservation:
        """Apply eligibility before ranking structured evidence matches."""

        if type(query) is not CompiledQuery:
            raise TypeError("query must be an exact CompiledQuery")
        if (
            query.catalog_id != self.evidence_index.catalog_id
            or query.catalog_semantic_release_id != self.evidence_index.catalog_semantic_release_id
        ):
            raise CompiledQueryBindingError("compiled query differs from facet-route bindings")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        eligible = frozenset(eligible_parent_asins)
        if not eligible:
            return _unavailable_facet(top_k, "no_eligible_documents")
        unknown = eligible.difference(self.evidence_index.parent_asins)
        if unknown:
            raise KeyError(f"unknown eligible parent_asin: {min(unknown)}")

        conditions = tuple(
            item
            for item in (
                *(_condition_from_ranking(item) for item in query.ranking_preferences),
                *(_condition_from_hard(item) for item in relaxed_constraints),
            )
            if item is not None
        )
        positive = tuple(item for item in conditions if not item.excludes)
        negative = tuple(item for item in conditions if item.excludes)
        if not positive:
            return _unavailable_facet(top_k, "no_positive_structured_evidence")

        positive_matches: dict[str, list[str]] = {}
        for condition in positive:
            for parent_asin in self._matches(condition).intersection(eligible):
                positive_matches.setdefault(parent_asin, []).append(condition.evidence_id)
        if not positive_matches:
            return _unavailable_facet(top_k, "no_matches")

        negative_matches: dict[str, list[str]] = {}
        for condition in negative:
            for parent_asin in self._matches(condition).intersection(positive_matches):
                negative_matches.setdefault(parent_asin, []).append(condition.evidence_id)

        positive_count = float(len(positive))
        negative_count = float(max(1, len(negative)))
        scored = []
        for parent_asin, matched_positive in positive_matches.items():
            matched_negative = negative_matches.get(parent_asin, [])
            score = len(matched_positive) / positive_count
            if negative:
                score -= 0.35 * len(matched_negative) / negative_count
            scored.append(
                (
                    parent_asin,
                    float(score),
                    tuple(sorted((*matched_positive, *matched_negative))),
                )
            )
        scored.sort(key=lambda item: (-item[1], item[0]))
        hits = tuple(
            RouteHit(
                parent_asin=parent_asin,
                rank=rank,
                raw_score=score,
                evidence_ids=evidence_ids,
            )
            for rank, (parent_asin, score, evidence_ids) in enumerate(scored[:top_k], start=1)
        )
        return RouteObservation(
            route=RetrievalRoute.FACET,
            requested_top_k=top_k,
            available=True,
            reason=None,
            hits=hits,
        )

    def _matches(self, condition: _FacetCondition) -> frozenset[str]:
        matches: set[str] = set()
        for value in condition.values:
            matches.update(self.evidence_index.match(condition.facet, value))
        return frozenset(matches)


def dense_route_observation(result: DenseSearchResult) -> RouteObservation:
    """Adapt a mask-aware dense result into the common route contract."""

    return RouteObservation(
        route=RetrievalRoute.DENSE,
        requested_top_k=result.requested_top_k,
        available=bool(result.hits),
        reason=None if result.hits else "no_eligible_documents",
        hits=tuple(
            RouteHit(parent_asin=hit.parent_asin, rank=hit.rank, raw_score=hit.score)
            for hit in result.hits
        ),
    )


def lexical_route_observation(observation: LexicalProbeObservation) -> RouteObservation:
    """Adapt an eligibility-aware FTS observation into the common route contract."""

    return RouteObservation(
        route=RetrievalRoute.LEXICAL,
        requested_top_k=observation.probe_k,
        available=observation.available,
        reason=observation.reason,
        hits=tuple(
            RouteHit(
                parent_asin=hit.parent_asin,
                rank=hit.rank,
                raw_score=hit.raw_bm25,
            )
            for hit in observation.hits
        ),
    )


def _condition_from_ranking(
    preference: CompiledRankingPreference,
) -> _FacetCondition | None:
    if type(preference) is not CompiledRankingPreference:
        raise TypeError("ranking_preferences contains an invalid value")
    return _condition(
        evidence_id=preference.preference_id,
        facet=preference.facet,
        operator=preference.operator,
        value=preference.value,
    )


def _condition_from_hard(constraint: CompiledHardConstraint) -> _FacetCondition | None:
    if type(constraint) is not CompiledHardConstraint:
        raise TypeError("relaxed_constraints contains an invalid value")
    return _condition(
        evidence_id=constraint.preference_id,
        facet=constraint.facet,
        operator=constraint.operator,
        value=constraint.value,
    )


def _condition(
    *,
    evidence_id: str,
    facet: str | None,
    operator: Operator | None,
    value: object,
) -> _FacetCondition | None:
    if facet not in SUPPORTED_FACETS:
        return None
    if operator not in {Operator.EQ, Operator.IN, Operator.NEQ, Operator.NOT_IN}:
        return None
    values = _string_values(operator, value)
    if not values:
        return None
    return _FacetCondition(
        evidence_id=evidence_id,
        facet=facet,
        values=values,
        excludes=operator in {Operator.NEQ, Operator.NOT_IN},
    )


def _string_values(operator: Operator, value: object) -> tuple[str, ...]:
    values: tuple[ScalarValue, ...]
    if operator in {Operator.IN, Operator.NOT_IN}:
        if type(value) is not tuple:
            return ()
        values = cast(tuple[ScalarValue, ...], value)
    else:
        if type(value) is not str:
            return ()
        values = (value,)
    if any(type(item) is not str or not item.strip() for item in values):
        return ()
    return tuple(cast(str, item).strip() for item in values)


def _unavailable_facet(
    top_k: int,
    reason: Literal[
        "no_eligible_documents",
        "no_positive_structured_evidence",
        "no_matches",
    ],
) -> RouteObservation:
    return RouteObservation(
        route=RetrievalRoute.FACET,
        requested_top_k=top_k,
        available=False,
        reason=reason,
        hits=(),
    )


def _validate_route_hit(hit: RouteHit) -> None:
    if type(hit.parent_asin) is not str or not hit.parent_asin.strip():
        raise ValueError("parent_asin must be a non-empty string")
    if type(hit.rank) is not int or hit.rank <= 0:
        raise ValueError("rank must be positive")
    if type(hit.raw_score) is not float or not math.isfinite(hit.raw_score):
        raise ValueError("raw_score must be finite")
