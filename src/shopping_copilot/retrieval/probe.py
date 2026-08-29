"""Fixed dense Probe that observes result geometry without controlling retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coherence import ProbeCoherence, compute_catalog_mean, compute_probe_coherence
from .dense import DenseIndex, DenseSearchResult
from .models import DenseHit


@dataclass(frozen=True, slots=True, kw_only=True)
class DenseProbeObservation:
    """Top-K membership and its uncalibrated semantic coherence."""

    probe_k: int
    hits: tuple[DenseHit, ...]
    coherence: ProbeCoherence


class FixedDenseProbe:
    """A C-independent Probe over the same score vector as the dense route."""

    def __init__(self, index: DenseIndex) -> None:
        self.index = index
        self.catalog_mean = compute_catalog_mean(index.vectors)

    def observe(
        self,
        result: DenseSearchResult,
        *,
        probe_k: int = 40,
    ) -> DenseProbeObservation:
        if type(probe_k) is not int or probe_k <= 0:
            raise ValueError("probe_k must be a positive integer")
        self.index._require_search_result(result)
        if probe_k > result.requested_top_k:
            raise ValueError("probe_k cannot exceed the ranking depth")
        hits = result.hits[:probe_k]
        if hits:
            rows = [self.index.row_index(hit.parent_asin) for hit in hits]
            candidates = np.asarray(self.index.vectors[rows], dtype=np.float64)
        else:
            candidates = np.empty(
                (0, self.index.manifest.embedding.dimension),
                dtype=np.float64,
            )
        coherence = compute_probe_coherence(candidates, self.catalog_mean)
        return DenseProbeObservation(
            probe_k=probe_k,
            hits=hits,
            coherence=coherence,
        )
