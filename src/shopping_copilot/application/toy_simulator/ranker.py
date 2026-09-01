from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .catalog import CatalogIndex
from .state import SessionState


@dataclass(slots=True)
class RankedResult:
    pids: list[int]
    strict_match: bool
    matched_constraint_count: int
    candidate_count: int


@dataclass(frozen=True, slots=True)
class AmbiguityPrior:
    """Bounded catalog prior allowed to reorder only near-equal evidence blocks."""

    scores: Sequence[float]
    strength: float
    evidence_window: float
    reorder_depth: int | None = None

    def validate(self, catalog_size: int) -> None:
        if len(self.scores) != catalog_size:
            raise ValueError("ambiguity prior must contain one score per catalog product")
        if not math.isfinite(self.strength) or self.strength <= 0.0:
            raise ValueError("ambiguity prior strength must be positive and finite")
        if not math.isfinite(self.evidence_window) or self.evidence_window < 0.0:
            raise ValueError("ambiguity prior evidence_window must be finite and non-negative")
        if self.reorder_depth is not None and self.reorder_depth < 1:
            raise ValueError("ambiguity prior reorder_depth must be positive when provided")
        if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in self.scores):
            raise ValueError("ambiguity prior scores must be finite values in [0, 1]")


class ProductRanker:
    def __init__(
        self,
        catalog: CatalogIndex,
        *,
        ambiguity_prior: AmbiguityPrior | None = None,
    ) -> None:
        self.catalog = catalog
        if ambiguity_prior is not None:
            ambiguity_prior.validate(catalog.size)
        self.ambiguity_prior = ambiguity_prior

    def rank(self, state: SessionState) -> RankedResult:
        scores: dict[int, float] = defaultdict(float)
        constraint_hits: dict[int, int] = defaultdict(int)
        positional_hits: dict[int, int] = defaultdict(int)
        filters: list[set[int]] = []
        union_candidates: set[int] = set()

        category_postings: tuple[int, ...] = ()
        if state.category:
            category_postings = self.catalog.category_candidates(state.category)
            if category_postings:
                category_set = set(category_postings)
                filters.append(category_set)
                union_candidates.update(category_set)
                for pid in category_postings:
                    scores[pid] += 1.5

        matched_constraints = 0
        for constraint in state.active_constraints:
            postings = self.catalog.postings_for_constraint(constraint.text)
            if not postings:
                continue
            matched_constraints += 1
            posting_set = set(postings)
            filters.append(posting_set)
            union_candidates.update(posting_set)
            weight = 4.0 * self.catalog.idf(len(postings))
            # Long, exact catalog phrases are much more diagnostic than isolated
            # attribute tokens, while IDF still controls generic marketing text.
            length_bonus = min(2.0, len(constraint.normalized) / 60.0)
            for pid in postings:
                scores[pid] += weight + length_bonus
                constraint_hits[pid] += 1
            for position, positional_postings in enumerate(
                self.catalog.positional_phrase_candidates(constraint.text)
            ):
                position_weight = 2.5 * (4 - position)
                for pid in positional_postings:
                    scores[pid] += position_weight
                    positional_hits[pid] += 1

        fts_query = state.query_text or " ".join(state.messages[-2:])
        fts_pids = self.catalog.fts_search(fts_query, limit=750)
        for rank, pid in enumerate(fts_pids, start=1):
            union_candidates.add(pid)
            scores[pid] += 3.0 / math.sqrt(rank)

        strict_match = False
        if filters:
            strict_candidates = set.intersection(*filters)
            if strict_candidates:
                pool = strict_candidates
                strict_match = True
            else:
                # A single parser or normalization mismatch must not permanently
                # remove the target. Fall back to a scored union.
                pool = union_candidates
        elif union_candidates:
            pool = union_candidates
        else:
            pool = set(self.catalog.all_pids)

        if not pool:
            pool = set(fts_pids) or set(self.catalog.all_pids)

        ranked = sorted(
            pool,
            key=lambda pid: (
                -scores.get(pid, 0.0),
                -math.log1p(self.catalog.rating_count_by_pid[pid]),
                self.catalog.asins[pid],
            ),
        )
        if self.ambiguity_prior is not None:
            ranked = self._apply_ambiguity_prior(
                ranked,
                scores=scores,
                constraint_hits=constraint_hits,
                positional_hits=positional_hits,
            )
        return RankedResult(
            pids=ranked,
            strict_match=strict_match,
            matched_constraint_count=matched_constraints,
            candidate_count=len(ranked),
        )

    def _apply_ambiguity_prior(
        self,
        ranked: list[int],
        *,
        scores: dict[int, float],
        constraint_hits: dict[int, int],
        positional_hits: dict[int, int],
    ) -> list[int]:
        prior = self.ambiguity_prior
        if prior is None or len(ranked) < 2:
            return ranked

        limit = (
            len(ranked) if prior.reorder_depth is None else min(prior.reorder_depth, len(ranked))
        )
        reorderable = ranked[:limit]
        result: list[int] = []
        start = 0
        while start < len(reorderable):
            leader = reorderable[start]
            leader_score = scores.get(leader, 0.0)
            signature = (constraint_hits.get(leader, 0), positional_hits.get(leader, 0))
            end = start + 1
            while end < len(reorderable):
                candidate = reorderable[end]
                candidate_signature = (
                    constraint_hits.get(candidate, 0),
                    positional_hits.get(candidate, 0),
                )
                if candidate_signature != signature:
                    break
                if leader_score - scores.get(candidate, 0.0) > prior.evidence_window:
                    break
                end += 1

            block = reorderable[start:end]
            block.sort(
                key=lambda pid: (
                    -(scores.get(pid, 0.0) + prior.strength * prior.scores[pid]),
                    -scores.get(pid, 0.0),
                    -math.log1p(self.catalog.rating_count_by_pid[pid]),
                    self.catalog.asins[pid],
                )
            )
            result.extend(block)
            start = end
        result.extend(ranked[limit:])
        return result
