"""Transparency-aware multi-center dense recall over one bound catalog index.

The module deliberately stops at candidate generation.  It reuses the query-to-
catalog score vector produced for Probe, discovers several catalog-grounded semantic
directions, and fills a dense candidate budget round-robin across those directions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from .dense import DenseEligibilityMask, DenseIndex, DenseScoreSnapshot

TRANSPARENCY_RECALL_POLICY_ID = "transparency_multi_center_recall_v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class TransparencyRecallPolicy:
    """Small, testable policy surface for the first multi-center experiment."""

    candidate_pool_k: int = 300
    frontier_k: int = 2_000
    maximum_directions: int = 6
    minimum_normalized_center_relevance: float = 0.35
    maximum_center_similarity: float = 0.90
    dense_budget_at_high_transparency: int = 150
    dense_budget_range: int = 60
    lexical_budget_at_low_transparency: int = 45
    lexical_budget_range: int = 30

    def __post_init__(self) -> None:
        for name, value in (
            ("candidate_pool_k", self.candidate_pool_k),
            ("frontier_k", self.frontier_k),
            ("maximum_directions", self.maximum_directions),
            (
                "dense_budget_at_high_transparency",
                self.dense_budget_at_high_transparency,
            ),
            ("lexical_budget_at_low_transparency", self.lexical_budget_at_low_transparency),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.dense_budget_range) is not int or self.dense_budget_range < 0:
            raise ValueError("dense_budget_range must be a non-negative integer")
        if type(self.lexical_budget_range) is not int or self.lexical_budget_range < 0:
            raise ValueError("lexical_budget_range must be a non-negative integer")
        for probability_name, probability in (
            (
                "minimum_normalized_center_relevance",
                self.minimum_normalized_center_relevance,
            ),
            ("maximum_center_similarity", self.maximum_center_similarity),
        ):
            _require_probability(probability, name=probability_name)
        if self.frontier_k < self.maximum_directions:
            raise ValueError("frontier_k must cover maximum_directions")
        low_dense = self.dense_budget_at_high_transparency + self.dense_budget_range
        high_lexical = self.lexical_budget_at_low_transparency + self.lexical_budget_range
        if low_dense + self.lexical_budget_at_low_transparency >= self.candidate_pool_k:
            raise ValueError("low-transparency route budgets leave no facet budget")
        if self.dense_budget_at_high_transparency + high_lexical >= self.candidate_pool_k:
            raise ValueError("high-transparency route budgets leave no facet budget")

    def requested_direction_count(self, transparency: float) -> int:
        """Map T to one through ``maximum_directions`` with half-up rounding."""

        _require_probability(transparency, name="transparency")
        span = self.maximum_directions - 1
        return 1 + int(math.floor((1.0 - transparency) * span + 0.5))

    def budgets(self, transparency: float) -> RecallBudgets:
        """Keep the total pool fixed while shifting budget between route types."""

        _require_probability(transparency, name="transparency")
        dense = self.dense_budget_at_high_transparency + int(
            math.floor(self.dense_budget_range * (1.0 - transparency) + 0.5)
        )
        lexical = self.lexical_budget_at_low_transparency + int(
            math.floor(self.lexical_budget_range * transparency + 0.5)
        )
        facet = self.candidate_pool_k - dense - lexical
        return RecallBudgets(dense=dense, lexical=lexical, facet=facet)


@dataclass(frozen=True, slots=True, kw_only=True)
class RecallBudgets:
    """Planned candidate counts before cross-route overlap and dense refill."""

    dense: int
    lexical: int
    facet: int

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in self.as_tuple()):
            raise ValueError("recall budgets must be positive integers")

    @property
    def total(self) -> int:
        return sum(self.as_tuple())

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.dense, self.lexical, self.facet)


@dataclass(frozen=True, slots=True, kw_only=True)
class RecallDirection:
    """One catalog product used as the observable center of a semantic direction."""

    direction_id: str
    center_parent_asin: str
    query_similarity: float
    maximum_similarity_to_previous_centers: float | None


@dataclass(frozen=True, slots=True, kw_only=True)
class DirectionalDenseCandidate:
    """One candidate and the direction that admitted it into dense recall."""

    parent_asin: str
    direction_id: str
    direction_rank: int
    query_similarity: float
    center_similarity: float
    combined_score: float


@dataclass(frozen=True, slots=True, kw_only=True)
class MultiCenterRecallTimings:
    """Wall-clock stages internal to multi-center planning, in milliseconds."""

    frontier_ms: float
    center_selection_ms: float
    direction_expansion_ms: float
    total_ms: float


@dataclass(frozen=True, slots=True, kw_only=True)
class MultiCenterDenseRecall:
    """Dense recall reserve plus enough evidence to explain how it was produced."""

    policy_id: str
    transparency: float
    requested_direction_count: int
    frontier_requested_k: int
    frontier_count: int
    directions: tuple[RecallDirection, ...]
    candidates: tuple[DirectionalDenseCandidate, ...]
    timings: MultiCenterRecallTimings


@dataclass(frozen=True, slots=True, kw_only=True)
class TransparencyRecallTrace:
    """Final recall audit after lexical/facet overlap and dense refill."""

    policy_id: str
    transparency: float
    candidate_pool_k: int
    candidate_pool_count: int
    requested_direction_count: int
    actual_direction_count: int
    frontier_requested_k: int
    frontier_count: int
    budgets: RecallBudgets
    actual_dense_count: int
    actual_lexical_count: int
    actual_facet_count: int
    dense_refill_count: int
    directions: tuple[RecallDirection, ...]
    dense_candidates: tuple[DirectionalDenseCandidate, ...]
    planner_timings: MultiCenterRecallTimings


class TransparencyAwareDenseRecall:
    """Discover catalog-grounded directions and recall evenly around each one."""

    def __init__(
        self,
        *,
        index: DenseIndex,
        policy: TransparencyRecallPolicy | None = None,
    ) -> None:
        if type(index) is not DenseIndex:
            raise TypeError("index must be an exact DenseIndex")
        resolved_policy = TransparencyRecallPolicy() if policy is None else policy
        if type(resolved_policy) is not TransparencyRecallPolicy:
            raise TypeError("policy must be an exact TransparencyRecallPolicy")
        self.index = index
        self.policy = resolved_policy

    def recall(
        self,
        scores: DenseScoreSnapshot,
        *,
        eligible_mask: DenseEligibilityMask,
        transparency: float,
    ) -> MultiCenterDenseRecall:
        """Reuse full-catalog scores; no query or document is embedded here."""

        _require_probability(transparency, name="transparency")
        total_started = perf_counter()

        frontier_started = perf_counter()
        frontier = self.index.rank_scores(
            scores,
            top_k=self.policy.frontier_k,
            eligible_mask=eligible_mask,
        ).hits
        frontier_ms = _elapsed_ms(frontier_started)
        requested = self.policy.requested_direction_count(transparency)
        if not frontier:
            total_ms = _elapsed_ms(total_started)
            return MultiCenterDenseRecall(
                policy_id=TRANSPARENCY_RECALL_POLICY_ID,
                transparency=transparency,
                requested_direction_count=requested,
                frontier_requested_k=self.policy.frontier_k,
                frontier_count=0,
                directions=(),
                candidates=(),
                timings=MultiCenterRecallTimings(
                    frontier_ms=frontier_ms,
                    center_selection_ms=0.0,
                    direction_expansion_ms=0.0,
                    total_ms=total_ms,
                ),
            )

        center_started = perf_counter()
        frontier_rows = np.fromiter(
            (self.index.row_index(hit.parent_asin) for hit in frontier),
            dtype=np.int64,
            count=len(frontier),
        )
        frontier_vectors = self.index.vectors[frontier_rows]
        frontier_scores = np.asarray(
            [hit.score for hit in frontier],
            dtype=np.float32,
        )
        normalized_relevance = _min_max(frontier_scores)
        center_offsets, previous_similarities = self._select_centers(
            frontier_vectors,
            normalized_relevance,
            requested_count=requested,
            transparency=transparency,
        )
        directions = tuple(
            RecallDirection(
                direction_id=f"direction_{position}",
                center_parent_asin=frontier[offset].parent_asin,
                query_similarity=float(frontier_scores[offset]),
                maximum_similarity_to_previous_centers=previous_similarity,
            )
            for position, (offset, previous_similarity) in enumerate(
                zip(center_offsets, previous_similarities, strict=True),
                start=1,
            )
        )
        center_selection_ms = _elapsed_ms(center_started)

        expansion_started = perf_counter()
        candidates = self._expand_directions(
            scores,
            eligible_mask=eligible_mask,
            frontier_rows=frontier_rows,
            center_offsets=center_offsets,
            directions=directions,
            transparency=transparency,
        )
        direction_expansion_ms = _elapsed_ms(expansion_started)
        return MultiCenterDenseRecall(
            policy_id=TRANSPARENCY_RECALL_POLICY_ID,
            transparency=transparency,
            requested_direction_count=requested,
            frontier_requested_k=self.policy.frontier_k,
            frontier_count=len(frontier),
            directions=directions,
            candidates=candidates,
            timings=MultiCenterRecallTimings(
                frontier_ms=frontier_ms,
                center_selection_ms=center_selection_ms,
                direction_expansion_ms=direction_expansion_ms,
                total_ms=_elapsed_ms(total_started),
            ),
        )

    def _select_centers(
        self,
        frontier_vectors: NDArray[np.float32],
        normalized_relevance: NDArray[np.float32],
        *,
        requested_count: int,
        transparency: float,
    ) -> tuple[tuple[int, ...], tuple[float | None, ...]]:
        centers = [0]
        previous_similarities: list[float | None] = [None]
        maximum_similarity = np.asarray(
            frontier_vectors @ frontier_vectors[0],
            dtype=np.float32,
        )
        relevance_weight = 0.40 + 0.50 * transparency

        while len(centers) < min(requested_count, len(frontier_vectors)):
            allowed = normalized_relevance >= self.policy.minimum_normalized_center_relevance
            allowed &= maximum_similarity <= self.policy.maximum_center_similarity
            allowed[np.asarray(centers, dtype=np.int64)] = False
            if not bool(np.any(allowed)):
                break

            novelty = _min_max(np.asarray(1.0 - maximum_similarity, dtype=np.float32))
            center_scores = relevance_weight * normalized_relevance + (
                1.0 - relevance_weight
            ) * novelty
            center_scores = np.where(allowed, center_scores, -np.inf)
            selected = int(np.argmax(center_scores))
            if not math.isfinite(float(center_scores[selected])):
                break
            previous_similarities.append(float(maximum_similarity[selected]))
            centers.append(selected)
            selected_similarity = np.asarray(
                frontier_vectors @ frontier_vectors[selected],
                dtype=np.float32,
            )
            maximum_similarity = np.maximum(maximum_similarity, selected_similarity)

        return tuple(centers), tuple(previous_similarities)

    def _expand_directions(
        self,
        scores: DenseScoreSnapshot,
        *,
        eligible_mask: DenseEligibilityMask,
        frontier_rows: NDArray[np.int64],
        center_offsets: tuple[int, ...],
        directions: tuple[RecallDirection, ...],
        transparency: float,
    ) -> tuple[DirectionalDenseCandidate, ...]:
        if not directions:
            return ()
        center_rows = frontier_rows[np.asarray(center_offsets, dtype=np.int64)]
        center_vectors = self.index.vectors[center_rows]
        center_similarities = np.asarray(
            self.index.vectors @ center_vectors.T,
            dtype=np.float32,
        )
        # A broad query protects relevance while several directions are explored.
        # Once T is high, the selected center represents the narrow intent region,
        # so retrieval should deepen around that region instead of re-broadening
        # toward the original query vector.
        alpha = 0.75 - 0.35 * transparency
        combined = alpha * scores.values[:, np.newaxis] + (1.0 - alpha) * center_similarities
        combined[~eligible_mask.values, :] = -np.inf

        orders: list[NDArray[np.int64]] = []
        eligible_rows = np.asarray(np.flatnonzero(eligible_mask.values), dtype=np.int64)
        asin_sort_keys = np.asarray(self.index.parent_asins, dtype=np.str_)[eligible_rows]
        for direction_index in range(len(directions)):
            direction_scores = combined[:, direction_index]
            order = np.lexsort(
                (
                    asin_sort_keys,
                    -direction_scores[eligible_rows],
                )
            )
            orders.append(eligible_rows[order])

        selected: list[DirectionalDenseCandidate] = []
        selected_asins: set[str] = set()
        pointers = [0] * len(directions)
        direction_ranks = [0] * len(directions)
        while len(selected) < self.policy.candidate_pool_k:
            added = False
            for direction_index, direction in enumerate(directions):
                order = orders[direction_index]
                while pointers[direction_index] < len(order):
                    row = int(order[pointers[direction_index]])
                    pointers[direction_index] += 1
                    direction_ranks[direction_index] += 1
                    parent_asin = self.index.parent_asins[row]
                    if parent_asin in selected_asins:
                        continue
                    selected_asins.add(parent_asin)
                    selected.append(
                        DirectionalDenseCandidate(
                            parent_asin=parent_asin,
                            direction_id=direction.direction_id,
                            direction_rank=direction_ranks[direction_index],
                            query_similarity=float(scores.values[row]),
                            center_similarity=float(center_similarities[row, direction_index]),
                            combined_score=float(combined[row, direction_index]),
                        )
                    )
                    added = True
                    break
                if len(selected) == self.policy.candidate_pool_k:
                    break
            if not added:
                break
        return tuple(selected)


def _min_max(values: NDArray[np.float32]) -> NDArray[np.float32]:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    span = maximum - minimum
    if span <= np.finfo(np.float32).eps:
        return np.ones_like(values, dtype=np.float32)
    return np.asarray((values - minimum) / span, dtype=np.float32)


def _elapsed_ms(started: float) -> float:
    return float((perf_counter() - started) * 1_000.0)


def _require_probability(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite float in [0, 1]")
    return value
