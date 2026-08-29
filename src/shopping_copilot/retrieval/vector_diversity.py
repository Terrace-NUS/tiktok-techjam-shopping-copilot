"""Category-blind vector diversification for dense retrieval candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .dense import DenseEligibilityMask, DenseIndex, DenseScoreSnapshot


@dataclass(frozen=True, slots=True, kw_only=True)
class VectorDiversityPolicy:
    """Map Intent Transparency to an MMR relevance/diversity trade-off."""

    minimum_relevance_weight: float = 0.30
    maximum_relevance_weight: float = 0.90

    def __post_init__(self) -> None:
        _require_unit_interval(
            self.minimum_relevance_weight,
            name="minimum_relevance_weight",
        )
        _require_unit_interval(
            self.maximum_relevance_weight,
            name="maximum_relevance_weight",
        )
        if self.minimum_relevance_weight > self.maximum_relevance_weight:
            raise ValueError("minimum_relevance_weight must not exceed maximum_relevance_weight")

    def relevance_weight(self, transparency: float) -> float:
        """Return the continuous MMR weight for one valid ``T_t`` value."""

        _require_unit_interval(transparency, name="transparency")
        return self.minimum_relevance_weight + transparency * (
            self.maximum_relevance_weight - self.minimum_relevance_weight
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class VectorDiversityHit:
    """One item selected by category-blind maximal marginal relevance."""

    parent_asin: str
    rank: int
    dense_rank: int
    relevance: float
    maximum_similarity_to_selected: float
    mmr_score: float

    def __post_init__(self) -> None:
        if type(self.parent_asin) is not str or not self.parent_asin.strip():
            raise ValueError("parent_asin must be a non-empty string")
        if self.parent_asin != self.parent_asin.strip():
            raise ValueError("parent_asin must be trimmed")
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be positive")
        if type(self.dense_rank) is not int or self.dense_rank <= 0:
            raise ValueError("dense_rank must be positive")
        for name, value in (
            ("relevance", self.relevance),
            ("maximum_similarity_to_selected", self.maximum_similarity_to_selected),
            ("mmr_score", self.mmr_score),
        ):
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite float")


@dataclass(frozen=True, slots=True, kw_only=True)
class VectorDiversityResult:
    """An auditable MMR result produced from one bound dense score snapshot."""

    index_id: str
    candidate_k: int
    requested_top_k: int
    relevance_weight: float
    hits: tuple[VectorDiversityHit, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class VectorCandidate:
    """One externally ranked candidate with relevance normalized to [0, 1]."""

    parent_asin: str
    candidate_rank: int
    relevance: float


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateVectorDiversityHit:
    """One category-blind MMR selection from an arbitrary candidate ranking."""

    parent_asin: str
    rank: int
    candidate_rank: int
    relevance: float
    maximum_similarity_to_selected: float
    mmr_score: float


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateVectorDiversityResult:
    """MMR output over candidates supplied by a fusion stage."""

    index_id: str
    candidate_count: int
    requested_top_k: int
    relevance_weight: float
    hits: tuple[CandidateVectorDiversityHit, ...]


class VectorMMRReranker:
    """Rerank a broad dense candidate window using product-vector novelty only."""

    def __init__(self, *, index: DenseIndex) -> None:
        if type(index) is not DenseIndex:
            raise TypeError("index must be an exact DenseIndex")
        self.index = index

    def rerank(
        self,
        scores: DenseScoreSnapshot,
        *,
        candidate_k: int,
        top_k: int,
        relevance_weight: float,
        eligible_mask: DenseEligibilityMask | None = None,
    ) -> VectorDiversityResult:
        """Select ``top_k`` results with cosine MMR from the dense candidate window.

        Both query relevance and candidate redundancy use the same normalized dense
        vectors. Categories, facets, brands, and hand-written diversity buckets are not
        inspected. A non-positive product-product cosine is treated as zero redundancy.
        """

        if type(candidate_k) is not int or candidate_k <= 0:
            raise ValueError("candidate_k must be a positive integer")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if top_k > candidate_k:
            raise ValueError("top_k must not exceed candidate_k")
        _require_unit_interval(relevance_weight, name="relevance_weight")

        candidate_result = self.index.rank_scores(
            scores,
            top_k=candidate_k,
            eligible_mask=eligible_mask,
        )
        candidates = candidate_result.hits
        output_size = min(top_k, len(candidates))
        if output_size == 0:
            return VectorDiversityResult(
                index_id=self.index.index_id,
                candidate_k=candidate_k,
                requested_top_k=top_k,
                relevance_weight=relevance_weight,
                hits=(),
            )

        rows = np.fromiter(
            (self.index.row_index(hit.parent_asin) for hit in candidates),
            dtype=np.int64,
            count=len(candidates),
        )
        vectors = self.index.vectors[rows]
        relevance = np.fromiter(
            (hit.score for hit in candidates),
            dtype=np.float32,
            count=len(candidates),
        )
        maximum_similarity = np.zeros(len(candidates), dtype=np.float32)
        available = np.ones(len(candidates), dtype=np.bool_)
        selected: list[VectorDiversityHit] = []

        for rank in range(1, output_size + 1):
            mmr_scores = (
                relevance_weight * relevance - (1.0 - relevance_weight) * maximum_similarity
            )
            mmr_scores[~available] = -np.inf
            selected_index = int(np.argmax(mmr_scores))
            candidate = candidates[selected_index]
            selected.append(
                VectorDiversityHit(
                    parent_asin=candidate.parent_asin,
                    rank=rank,
                    dense_rank=candidate.rank,
                    relevance=float(relevance[selected_index]),
                    maximum_similarity_to_selected=float(maximum_similarity[selected_index]),
                    mmr_score=float(mmr_scores[selected_index]),
                )
            )
            available[selected_index] = False
            similarities = vectors @ vectors[selected_index]
            non_negative = np.maximum(similarities, 0.0)
            maximum_similarity = np.maximum(maximum_similarity, non_negative)

        return VectorDiversityResult(
            index_id=self.index.index_id,
            candidate_k=candidate_k,
            requested_top_k=top_k,
            relevance_weight=relevance_weight,
            hits=tuple(selected),
        )

    def rerank_candidates(
        self,
        candidates: tuple[VectorCandidate, ...],
        *,
        top_k: int,
        relevance_weight: float,
    ) -> CandidateVectorDiversityResult:
        """Apply vector-only redundancy control to an externally fused ranking."""

        if type(candidates) is not tuple:
            raise TypeError("candidates must be a tuple")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        _require_unit_interval(relevance_weight, name="relevance_weight")
        parent_asins: set[str] = set()
        for expected_rank, candidate in enumerate(candidates, start=1):
            if type(candidate) is not VectorCandidate:
                raise TypeError("candidates must contain exact VectorCandidate values")
            if candidate.candidate_rank != expected_rank:
                raise ValueError("candidate ranks must be contiguous")
            if candidate.parent_asin in parent_asins:
                raise ValueError("candidates must contain unique products")
            parent_asins.add(candidate.parent_asin)
            _require_unit_interval(candidate.relevance, name="candidate relevance")

        output_size = min(top_k, len(candidates))
        if output_size == 0:
            return CandidateVectorDiversityResult(
                index_id=self.index.index_id,
                candidate_count=0,
                requested_top_k=top_k,
                relevance_weight=relevance_weight,
                hits=(),
            )

        rows = np.fromiter(
            (self.index.row_index(item.parent_asin) for item in candidates),
            dtype=np.int64,
            count=len(candidates),
        )
        vectors = self.index.vectors[rows]
        relevance = np.fromiter(
            (item.relevance for item in candidates),
            dtype=np.float32,
            count=len(candidates),
        )
        maximum_similarity = np.zeros(len(candidates), dtype=np.float32)
        available = np.ones(len(candidates), dtype=np.bool_)
        selected: list[CandidateVectorDiversityHit] = []

        for rank in range(1, output_size + 1):
            mmr_scores = (
                relevance_weight * relevance - (1.0 - relevance_weight) * maximum_similarity
            )
            mmr_scores[~available] = -np.inf
            selected_index = int(np.argmax(mmr_scores))
            candidate = candidates[selected_index]
            selected.append(
                CandidateVectorDiversityHit(
                    parent_asin=candidate.parent_asin,
                    rank=rank,
                    candidate_rank=candidate.candidate_rank,
                    relevance=float(relevance[selected_index]),
                    maximum_similarity_to_selected=float(maximum_similarity[selected_index]),
                    mmr_score=float(mmr_scores[selected_index]),
                )
            )
            available[selected_index] = False
            similarities = vectors @ vectors[selected_index]
            maximum_similarity = np.maximum(
                maximum_similarity,
                np.maximum(similarities, 0.0),
            )

        return CandidateVectorDiversityResult(
            index_id=self.index.index_id,
            candidate_count=len(candidates),
            requested_top_k=top_k,
            relevance_weight=relevance_weight,
            hits=tuple(selected),
        )


def _require_unit_interval(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite float in [0, 1]")
    return value
