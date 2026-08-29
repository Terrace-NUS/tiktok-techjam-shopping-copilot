"""Tests for fixed-probe semantic coherence."""

from __future__ import annotations

import numpy as np
import pytest

from shopping_copilot.retrieval.coherence import (
    compute_catalog_mean,
    compute_probe_coherence,
)


def test_compute_catalog_mean_uses_float64_values() -> None:
    catalog = np.array([[1, 2], [3, 6]], dtype=np.int64)

    result = compute_catalog_mean(catalog)

    assert result.dtype == np.float64
    np.testing.assert_allclose(result, np.array([2.0, 4.0]))


@pytest.mark.parametrize(
    "catalog",
    [
        np.empty((0, 2)),
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[1.0, np.nan], [0.0, 1.0]]),
        np.array([[1.0, np.inf], [0.0, 1.0]]),
    ],
)
def test_compute_catalog_mean_rejects_unusable_catalog_vectors(catalog: np.ndarray) -> None:
    with pytest.raises(ValueError):
        compute_catalog_mean(catalog)


def test_identical_directions_have_maximal_raw_coherence() -> None:
    candidates = np.array([[2.0, 0.0], [5.0, 0.0], [1.0, 0.0]])

    result = compute_probe_coherence(candidates, np.zeros(2))

    assert result.available is True
    assert result.reason is None
    assert result.n == 3
    assert result.resultant_length == pytest.approx(1.0)
    assert result.debiased_pairwise_cosine == pytest.approx(1.0)


def test_orthogonal_directions_have_zero_debiased_pairwise_cosine() -> None:
    candidates = np.eye(3, dtype=np.float64)

    result = compute_probe_coherence(candidates, np.zeros(3))

    assert result.available is True
    assert result.resultant_length == pytest.approx(1.0 / np.sqrt(3.0))
    assert result.debiased_pairwise_cosine == pytest.approx(0.0, abs=1e-15)


def test_opposite_directions_have_minimal_raw_coherence() -> None:
    candidates = np.array([[1.0, 0.0], [-1.0, 0.0]])

    result = compute_probe_coherence(candidates, np.zeros(2))

    assert result.available is True
    assert result.resultant_length == pytest.approx(0.0)
    assert result.debiased_pairwise_cosine == pytest.approx(-1.0)


def test_catalog_mean_is_subtracted_before_normalization() -> None:
    candidates = np.array([[2.0, 1.0], [1.0, 2.0]])

    result = compute_probe_coherence(candidates, np.array([1.0, 1.0]))

    assert result.available is True
    assert result.resultant_length == pytest.approx(1.0 / np.sqrt(2.0))
    assert result.debiased_pairwise_cosine == pytest.approx(0.0, abs=1e-15)


def test_empty_probe_is_unavailable() -> None:
    result = compute_probe_coherence(np.empty((0, 2)), np.zeros(2))

    assert result.n == 0
    assert result.available is False
    assert result.reason == "empty_candidates"
    assert result.resultant_length is None
    assert result.debiased_pairwise_cosine is None


def test_single_candidate_is_unavailable_instead_of_maximally_clear() -> None:
    result = compute_probe_coherence(np.array([[1.0, 0.0]]), np.zeros(2))

    assert result.n == 1
    assert result.available is False
    assert result.reason == "insufficient_candidates"
    assert result.resultant_length is None
    assert result.debiased_pairwise_cosine is None


@pytest.mark.parametrize(
    ("candidates", "catalog_mean", "reason"),
    [
        (np.array([[1.0, 0.0], [np.nan, 1.0]]), np.zeros(2), "nonfinite_candidate"),
        (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([np.inf, 0.0]), "nonfinite_catalog_mean"),
        (np.array([[1.0, 0.0], [0.0, 0.0]]), np.zeros(2), "zero_candidate"),
        (
            np.array([[1.0, 1.0], [2.0, 1.0]]),
            np.array([1.0, 1.0]),
            "zero_centered_candidate",
        ),
    ],
)
def test_invalid_numeric_evidence_fails_closed(
    candidates: np.ndarray,
    catalog_mean: np.ndarray,
    reason: str,
) -> None:
    result = compute_probe_coherence(candidates, catalog_mean)

    assert result.available is False
    assert result.reason == reason
    assert result.resultant_length is None
    assert result.debiased_pairwise_cosine is None


def test_dimension_mismatch_fails_closed() -> None:
    result = compute_probe_coherence(np.eye(2), np.zeros(3))

    assert result.n == 2
    assert result.available is False
    assert result.reason == "dimension_mismatch"
