"""Uncalibrated semantic coherence for a fixed retrieval probe.

The statistic in this module is deliberately narrow: it measures the geometry
of a supplied candidate bag.  It does not turn that geometry into an intent
clarity probability and it does not choose or alter the probe candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
UnavailableReason: TypeAlias = Literal[
    "empty_candidates",
    "insufficient_candidates",
    "invalid_candidate_shape",
    "invalid_catalog_mean",
    "dimension_mismatch",
    "nonfinite_candidate",
    "nonfinite_catalog_mean",
    "zero_candidate",
    "zero_centered_candidate",
    "nonfinite_result",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class ProbeCoherence:
    """Raw coherence evidence produced by a fixed semantic probe."""

    n: int
    resultant_length: float | None
    debiased_pairwise_cosine: float | None
    available: bool
    reason: UnavailableReason | None


def compute_catalog_mean(catalog_embeddings: npt.ArrayLike) -> FloatArray:
    """Return the float64 mean vector for a valid catalog embedding matrix.

    Catalog construction is an offline prerequisite, so malformed catalog
    inputs raise ``ValueError`` instead of producing a misleading mean.
    """

    try:
        embeddings = np.asarray(catalog_embeddings, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("catalog_embeddings must be a numeric matrix") from error

    if embeddings.ndim != 2 or embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError("catalog_embeddings must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("catalog_embeddings must contain only finite values")

    with np.errstate(over="ignore", invalid="ignore"):
        norms = np.linalg.norm(embeddings, axis=1)
    if not np.all(np.isfinite(norms)):
        raise ValueError("catalog_embeddings produced a non-finite norm")
    if np.any(norms == 0.0):
        raise ValueError("catalog_embeddings must not contain zero vectors")

    with np.errstate(over="ignore", invalid="ignore"):
        catalog_mean = np.asarray(
            np.mean(embeddings, axis=0, dtype=np.float64),
            dtype=np.float64,
        )
    if not np.all(np.isfinite(catalog_mean)):
        raise ValueError("catalog mean must contain only finite values")
    return catalog_mean


def compute_probe_coherence(
    candidate_embeddings: npt.ArrayLike,
    catalog_mean: npt.ArrayLike,
) -> ProbeCoherence:
    """Compute mean-centered resultant length and raw debiased cosine.

    Each candidate is mean-centered with ``catalog_mean`` and then L2
    normalized.  For ``n >= 2``, the returned raw statistic is

    ``G = (n * R**2 - 1) / (n - 1)``

    where ``R`` is the norm of the mean normalized vector.  No calibration or
    clipping is applied.  Invalid evidence fails closed as unavailable.
    """

    try:
        candidates = np.asarray(candidate_embeddings, dtype=np.float64)
    except (TypeError, ValueError):
        return _unavailable(n=0, reason="invalid_candidate_shape")

    if candidates.ndim != 2 or candidates.shape[1] == 0:
        return _unavailable(n=0, reason="invalid_candidate_shape")

    n = int(candidates.shape[0])
    if n == 0:
        return _unavailable(n=0, reason="empty_candidates")

    try:
        mean = np.asarray(catalog_mean, dtype=np.float64)
    except (TypeError, ValueError):
        return _unavailable(n=n, reason="invalid_catalog_mean")

    if mean.ndim != 1 or mean.shape[0] == 0:
        return _unavailable(n=n, reason="invalid_catalog_mean")
    if candidates.shape[1] != mean.shape[0]:
        return _unavailable(n=n, reason="dimension_mismatch")
    if not np.all(np.isfinite(mean)):
        return _unavailable(n=n, reason="nonfinite_catalog_mean")
    if not np.all(np.isfinite(candidates)):
        return _unavailable(n=n, reason="nonfinite_candidate")

    with np.errstate(over="ignore", invalid="ignore"):
        raw_norms = np.linalg.norm(candidates, axis=1)
    if not np.all(np.isfinite(raw_norms)):
        return _unavailable(n=n, reason="nonfinite_candidate")
    if np.any(raw_norms == 0.0):
        return _unavailable(n=n, reason="zero_candidate")

    with np.errstate(over="ignore", invalid="ignore"):
        centered = candidates - mean
        centered_norms = np.linalg.norm(centered, axis=1)
    if not np.all(np.isfinite(centered)) or not np.all(np.isfinite(centered_norms)):
        return _unavailable(n=n, reason="nonfinite_candidate")
    if np.any(centered_norms == 0.0):
        return _unavailable(n=n, reason="zero_centered_candidate")

    if n == 1:
        return _unavailable(n=1, reason="insufficient_candidates")

    normalized = centered / centered_norms[:, np.newaxis]
    mean_direction = np.mean(normalized, axis=0, dtype=np.float64)
    resultant_length = float(np.linalg.norm(mean_direction))
    debiased_pairwise_cosine = float((n * resultant_length * resultant_length - 1.0) / (n - 1))

    if not np.isfinite(resultant_length) or not np.isfinite(debiased_pairwise_cosine):
        return _unavailable(n=n, reason="nonfinite_result")

    return ProbeCoherence(
        n=n,
        resultant_length=resultant_length,
        debiased_pairwise_cosine=debiased_pairwise_cosine,
        available=True,
        reason=None,
    )


def _unavailable(*, n: int, reason: UnavailableReason) -> ProbeCoherence:
    return ProbeCoherence(
        n=n,
        resultant_length=None,
        debiased_pairwise_cosine=None,
        available=False,
        reason=reason,
    )
