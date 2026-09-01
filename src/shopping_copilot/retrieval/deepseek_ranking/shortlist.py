"""Direction-protected BGE shortlist construction."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..dense import DenseIndex
from ..ranking import CrossEncoderRankingResult
from ..transparency_recall import TransparencyRecallTrace
from .models import RankingCandidateCard, RankingShortlist

DEFAULT_SHORTLIST_K = 48
DEFAULT_PROTECTED_PER_DIRECTION = 6

_FIELD_LIMITS = {
    "title": 320,
    "categories": 260,
    "store": 120,
    "features": 480,
    "details": 480,
    "description": 480,
}


class DirectionAwareShortlister:
    """Keep top BGE products while reserving evidence from each recall direction."""

    def __init__(
        self,
        *,
        index: DenseIndex,
        top_k: int = DEFAULT_SHORTLIST_K,
        protected_per_direction: int = DEFAULT_PROTECTED_PER_DIRECTION,
    ) -> None:
        if type(index) is not DenseIndex:
            raise TypeError("index must be an exact DenseIndex")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be positive")
        if type(protected_per_direction) is not int or protected_per_direction < 0:
            raise ValueError("protected_per_direction must be non-negative")
        self.index = index
        self.top_k = top_k
        self.protected_per_direction = protected_per_direction

    def select(
        self,
        ranking: CrossEncoderRankingResult,
        *,
        documents: Mapping[str, str],
        recall_trace: TransparencyRecallTrace | None,
        routes: Mapping[str, tuple[str, ...]] | None = None,
    ) -> RankingShortlist:
        if type(ranking) is not CrossEncoderRankingResult:
            raise TypeError("ranking must be an exact CrossEncoderRankingResult")
        if not isinstance(documents, Mapping):
            raise TypeError("documents must be a mapping")
        route_mapping = {} if routes is None else routes
        if not isinstance(route_mapping, Mapping):
            raise TypeError("routes must be a mapping or None")

        direction_by_asin = self._assign_directions(ranking, recall_trace)
        selected: set[str] = set()
        if recall_trace is not None and self.protected_per_direction:
            for direction in recall_trace.directions:
                protected = 0
                for hit in ranking.hits:
                    if direction_by_asin.get(hit.parent_asin) != direction.direction_id:
                        continue
                    selected.add(hit.parent_asin)
                    protected += 1
                    if protected == self.protected_per_direction or len(selected) == self.top_k:
                        break
                if len(selected) == self.top_k:
                    break

        for hit in ranking.hits:
            if len(selected) == self.top_k:
                break
            selected.add(hit.parent_asin)

        retained = tuple(hit for hit in ranking.hits if hit.parent_asin in selected)
        cards: list[RankingCandidateCard] = []
        for shortlist_rank, hit in enumerate(retained, start=1):
            try:
                document = documents[hit.parent_asin]
            except KeyError as error:
                raise KeyError(f"missing product document: {hit.parent_asin}") from error
            if type(document) is not str or not document.strip():
                raise ValueError("product documents must contain non-empty strings")
            candidate_routes = route_mapping.get(hit.parent_asin, ())
            if type(candidate_routes) is not tuple:
                raise TypeError("route mappings must contain tuples")
            cards.append(
                RankingCandidateCard(
                    parent_asin=hit.parent_asin,
                    shortlist_rank=shortlist_rank,
                    original_candidate_rank=hit.candidate_rank,
                    bge_relevance=hit.relevance,
                    normalized_bge_score=hit.normalized_model_score,
                    direction_id=direction_by_asin.get(hit.parent_asin),
                    routes=candidate_routes,
                    product_text=compact_product_text(document),
                )
            )
        return RankingShortlist(
            model_id=ranking.model_id,
            requested_top_k=self.top_k,
            protected_per_direction=self.protected_per_direction,
            cards=tuple(cards),
        )

    def _assign_directions(
        self,
        ranking: CrossEncoderRankingResult,
        trace: TransparencyRecallTrace | None,
    ) -> dict[str, str]:
        if trace is None or not trace.directions or not ranking.hits:
            return {}
        center_rows = np.fromiter(
            (self.index.row_index(direction.center_parent_asin) for direction in trace.directions),
            dtype=np.int64,
            count=len(trace.directions),
        )
        candidate_rows = np.fromiter(
            (self.index.row_index(hit.parent_asin) for hit in ranking.hits),
            dtype=np.int64,
            count=len(ranking.hits),
        )
        similarities = self.index.vectors[candidate_rows] @ self.index.vectors[center_rows].T
        chosen = np.argmax(similarities, axis=1)
        return {
            hit.parent_asin: trace.directions[int(direction_index)].direction_id
            for hit, direction_index in zip(ranking.hits, chosen, strict=True)
        }


def compact_product_text(document: str) -> str:
    """Bound every catalog field without dropping the description wholesale."""

    if type(document) is not str or not document.strip():
        raise ValueError("document must be a non-empty string")
    rendered: list[str] = []
    for raw_line in document.splitlines():
        field, separator, value = raw_line.partition(":")
        normalized_field = field.strip().casefold()
        limit = _FIELD_LIMITS.get(normalized_field, 240)
        text = " ".join(value.split()) if separator else " ".join(raw_line.split())
        if len(text) > limit:
            text = text[:limit].rstrip() + "..."
        if text:
            rendered.append(f"{field.strip()}: {text}" if separator else text)
    result = "\n".join(rendered)
    if not result:
        raise ValueError("document contains no model-visible text")
    return result
