"""Deterministic semantic modes over an existing dense Probe ranking.

This module is intentionally an observer.  It reuses the hits and index rows
already selected by :class:`DenseSearchResult`; it never embeds a query,
recomputes scores, or changes eligibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .coherence import ProbeCoherence, compute_catalog_mean, compute_probe_coherence
from .dense import DenseIndex, DenseSearchResult
from .models import DenseHit

Float64Vector = NDArray[np.float64]

DEFAULT_MODE_SIMILARITY_THRESHOLD = 0.94
MAX_REPRESENTATIVE_IDS = 3
DUPLICATE_CONCENTRATION_SHARE_THRESHOLD = 0.5
_MODE_ID_WIDTH = 4


@dataclass(frozen=True, slots=True, kw_only=True)
class SemanticModeMembership:
    """The deterministic mode assignment for one dense-ranked listing."""

    parent_asin: str
    dense_rank: int
    mode_id: str
    similarity_to_leader: float


@dataclass(frozen=True, slots=True, kw_only=True)
class SemanticMode:
    """One near-duplicate mode and its equal-weight geometry."""

    id: str
    leader_id: str
    size: int
    best_score: float
    representative_ids: tuple[str, ...]
    centroid: Float64Vector = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        observed = np.asarray(self.centroid)
        if observed.ndim != 1 or observed.dtype != np.float64:
            raise TypeError("SemanticMode.centroid must be a float64 vector")
        if not np.isfinite(observed).all():
            raise ValueError("SemanticMode.centroid must be finite")
        norm = float(np.linalg.norm(observed))
        if not math.isfinite(norm) or not math.isclose(norm, 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("SemanticMode.centroid must be L2-normalized")
        owned = np.array(observed, dtype=np.float64, order="C", copy=True)
        owned.setflags(write=False)
        object.__setattr__(self, "centroid", owned)


@dataclass(frozen=True, slots=True, kw_only=True)
class SemanticModeObservation:
    """Mode assignments plus listing- and equal-mode coherence evidence."""

    probe_k: int
    threshold: float
    hits: tuple[DenseHit, ...]
    memberships: tuple[SemanticModeMembership, ...]
    modes: tuple[SemanticMode, ...]
    largest_mode_share: float
    effective_mode_count: float
    raw_listing_coherence: ProbeCoherence
    equal_mode_coherence: ProbeCoherence
    duplicate_concentration_warning: bool


@dataclass(slots=True)
class _ModeBuilder:
    id: str
    leader: DenseHit
    leader_vector: Float64Vector
    members: list[tuple[DenseHit, Float64Vector]]


class SemanticModeProbe:
    """Group an existing dense ranking by cosine similarity to fixed leaders."""

    def __init__(self, index: DenseIndex) -> None:
        self.index = index
        self.catalog_mean = compute_catalog_mean(index.vectors)

    def observe(
        self,
        result: DenseSearchResult,
        *,
        probe_k: int,
        threshold: float = DEFAULT_MODE_SIMILARITY_THRESHOLD,
    ) -> SemanticModeObservation:
        """Observe up to ``probe_k`` already-ranked hits without rescoring them."""

        if type(probe_k) is not int or probe_k <= 0:
            raise ValueError("probe_k must be a positive integer")
        if type(threshold) is not float or not math.isfinite(threshold):
            raise ValueError("threshold must be a finite float")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between zero and one")

        self.index._require_search_result(result)
        if probe_k > result.requested_top_k:
            raise ValueError("probe_k cannot exceed the ranking depth")

        hits = result.hits[:probe_k]
        vectors = tuple(self._vector_for_hit(hit) for hit in hits)
        builders: list[_ModeBuilder] = []
        memberships: list[SemanticModeMembership] = []

        for hit, vector in zip(hits, vectors, strict=True):
            selected_index: int | None = None
            selected_similarity = -1.0
            for mode_index, mode in enumerate(builders):
                similarity = _rounded_cosine(vector, mode.leader_vector)
                if similarity >= threshold and similarity > selected_similarity:
                    selected_index = mode_index
                    selected_similarity = similarity

            if selected_index is None:
                mode_id = _mode_id(len(builders))
                builders.append(
                    _ModeBuilder(
                        id=mode_id,
                        leader=hit,
                        leader_vector=vector,
                        members=[(hit, vector)],
                    )
                )
                selected_similarity = 1.0
            else:
                mode = builders[selected_index]
                mode_id = mode.id
                mode.members.append((hit, vector))

            memberships.append(
                SemanticModeMembership(
                    parent_asin=hit.parent_asin,
                    dense_rank=hit.rank,
                    mode_id=mode_id,
                    similarity_to_leader=selected_similarity,
                )
            )

        modes = tuple(_finish_mode(builder) for builder in builders)
        raw_listing_coherence = compute_probe_coherence(
            _matrix(vectors, dimension=self.index.manifest.embedding.dimension),
            self.catalog_mean,
        )
        equal_mode_coherence = _equal_mode_coherence(
            modes,
            catalog_mean=self.catalog_mean,
            dimension=self.index.manifest.embedding.dimension,
        )
        largest_mode_share = _largest_mode_share(modes, observed_count=len(hits))

        return SemanticModeObservation(
            probe_k=probe_k,
            threshold=threshold,
            hits=hits,
            memberships=tuple(memberships),
            modes=modes,
            largest_mode_share=largest_mode_share,
            effective_mode_count=_effective_mode_count(modes, observed_count=len(hits)),
            raw_listing_coherence=raw_listing_coherence,
            equal_mode_coherence=equal_mode_coherence,
            duplicate_concentration_warning=_duplicate_concentration_warning(
                modes,
                largest_mode_share=largest_mode_share,
            ),
        )

    def _vector_for_hit(self, hit: DenseHit) -> Float64Vector:
        row = self.index.row_index(hit.parent_asin)
        vector = np.asarray(self.index.vectors[row], dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        return vector / norm


def _rounded_cosine(left: Float64Vector, right: Float64Vector) -> float:
    cosine = float(np.dot(left, right))
    bounded = min(1.0, max(-1.0, cosine))
    return float(round(bounded, 6))


def _mode_id(zero_based_index: int) -> str:
    return f"mode_{zero_based_index + 1:0{_MODE_ID_WIDTH}d}"


def _finish_mode(builder: _ModeBuilder) -> SemanticMode:
    member_vectors = np.stack([vector for _, vector in builder.members]).astype(
        np.float64,
        copy=False,
    )
    mean = np.mean(member_vectors, axis=0, dtype=np.float64)
    centroid = cast(Float64Vector, mean / np.linalg.norm(mean))
    member_hits = tuple(hit for hit, _ in builder.members)
    return SemanticMode(
        id=builder.id,
        leader_id=builder.leader.parent_asin,
        size=len(builder.members),
        best_score=max(hit.score for hit in member_hits),
        representative_ids=tuple(hit.parent_asin for hit in member_hits[:MAX_REPRESENTATIVE_IDS]),
        centroid=centroid,
    )


def _matrix(vectors: tuple[Float64Vector, ...], *, dimension: int) -> NDArray[np.float64]:
    if not vectors:
        return np.empty((0, dimension), dtype=np.float64)
    return np.asarray(vectors, dtype=np.float64)


def _equal_mode_coherence(
    modes: tuple[SemanticMode, ...],
    *,
    catalog_mean: Float64Vector,
    dimension: int,
) -> ProbeCoherence:
    if len(modes) < 2:
        return ProbeCoherence(
            n=len(modes),
            resultant_length=None,
            debiased_pairwise_cosine=None,
            available=False,
            reason="insufficient_candidates",
        )
    centroids = _matrix(tuple(mode.centroid for mode in modes), dimension=dimension)
    return compute_probe_coherence(centroids, catalog_mean)


def _largest_mode_share(
    modes: tuple[SemanticMode, ...],
    *,
    observed_count: int,
) -> float:
    if observed_count == 0:
        return 0.0
    return max(mode.size for mode in modes) / observed_count


def _effective_mode_count(
    modes: tuple[SemanticMode, ...],
    *,
    observed_count: int,
) -> float:
    if observed_count == 0:
        return 0.0
    shares = (mode.size / observed_count for mode in modes)
    entropy = -math.fsum(share * math.log(share) for share in shares)
    return float(math.exp(entropy))


def _duplicate_concentration_warning(
    modes: tuple[SemanticMode, ...],
    *,
    largest_mode_share: float,
) -> bool:
    return (
        any(mode.size > 1 for mode in modes)
        and largest_mode_share >= DUPLICATE_CONCENTRATION_SHARE_THRESHOLD
    )
