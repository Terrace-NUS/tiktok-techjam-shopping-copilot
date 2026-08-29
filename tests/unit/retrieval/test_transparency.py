from __future__ import annotations

from dataclasses import replace

import pytest

from shopping_copilot.retrieval.transparency import (
    CONTROLLER_FALLBACK,
    DiagnosticStatus,
    TransparencyCalibration,
    TransparencyEstimator,
    TransparencyEvidence,
    estimate_transparency,
    project_search_belief,
)
from shopping_copilot.session_context import FacetRegistry, ProbeQuality, validate_search_belief


def _calibration(*, approved: bool = True) -> TransparencyCalibration:
    return TransparencyCalibration(
        policy_id="mode_coherence_calibration_v1",
        low_anchor=0.2,
        high_anchor=0.6,
        approved=approved,
    )


def _evidence(**changes: object) -> TransparencyEvidence:
    value = TransparencyEvidence(
        probe_id="probe-17",
        intent_version=7,
        probe_k=4,
        eligible_count=100,
        dense_hits=("A", "B", "C", "D"),
        lexical_hits=("C", "D", "E", "F"),
        listing_coherence=0.35,
        mode_coherence=0.4,
        mode_count=3,
        largest_mode_share=0.5,
        effective_mode_count=2.5,
        duplicate_warning=False,
        lexical_available=True,
        lexical_token_coverage=1.0,
        lexical_mean_normalized_idf=0.75,
        hard_filter_relaxed=False,
    )
    return replace(value, **changes)


def test_eligible_count_does_not_change_certainty_for_the_same_geometry() -> None:
    estimator = TransparencyEstimator(_calibration())

    small = estimator.estimate(_evidence(eligible_count=4))
    large = estimator.estimate(_evidence(eligible_count=50_000))

    assert small.certainty == pytest.approx(0.5)
    assert large.certainty == small.certainty
    assert small.diagnostics.eligible_count == 4
    assert large.diagnostics.eligible_count == 50_000


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (-0.5, 0.0),
        (0.2, 0.0),
        (0.4, 0.5),
        (0.6, 1.0),
        (0.9, 1.0),
    ],
)
def test_calibration_clips_only_the_linear_mode_coherence_mapping(
    raw: float,
    expected: float,
) -> None:
    estimate = estimate_transparency(_evidence(mode_coherence=raw), _calibration())

    assert estimate.certainty == pytest.approx(expected)
    assert estimate.raw_mode_coherence == raw


def test_lexical_and_overlap_observations_change_diagnostics_not_certainty() -> None:
    baseline = estimate_transparency(_evidence(), _calibration())
    unavailable_lexical = estimate_transparency(
        _evidence(
            lexical_hits=(),
            lexical_available=False,
            lexical_token_coverage=None,
            lexical_mean_normalized_idf=None,
        ),
        _calibration(),
    )

    assert unavailable_lexical.certainty == baseline.certainty
    assert baseline.diagnostics.route_overlap_count == 2
    assert baseline.diagnostics.route_overlap == pytest.approx(0.5)
    assert unavailable_lexical.diagnostics.route_overlap is None
    assert unavailable_lexical.diagnostics.status is DiagnosticStatus.DEGRADED
    assert "lexical_unavailable" in unavailable_lexical.diagnostics.reason_codes


def test_underfilled_probe_keeps_geometry_score_but_marks_diagnostics_degraded() -> None:
    estimate = estimate_transparency(
        _evidence(dense_hits=("A", "B", "C")),
        _calibration(),
    )

    assert estimate.certainty == pytest.approx(0.5)
    assert estimate.controller_fallback == CONTROLLER_FALLBACK == 0.5
    assert estimate.diagnostics.status is DiagnosticStatus.DEGRADED
    assert estimate.diagnostics.fill_ratio == pytest.approx(0.75)
    assert "dense_probe_underfilled" in estimate.diagnostics.reason_codes

    belief = project_search_belief(estimate)
    assert belief.certainty == pytest.approx(0.5)
    assert belief.certainty_evidence.quality_status is ProbeQuality.VALID
    assert belief.certainty_evidence.raw_concentration == pytest.approx(0.4)


def test_empty_and_missing_mode_observations_are_unavailable() -> None:
    empty = estimate_transparency(
        _evidence(
            eligible_count=0,
            dense_hits=(),
            lexical_hits=(),
            listing_coherence=None,
            mode_coherence=None,
            mode_count=0,
            largest_mode_share=0.0,
            effective_mode_count=0.0,
        ),
        _calibration(),
    )
    one_mode = estimate_transparency(
        _evidence(
            mode_coherence=None,
            mode_count=1,
            largest_mode_share=1.0,
            effective_mode_count=1.0,
        ),
        _calibration(),
    )

    assert empty.certainty is None
    assert empty.diagnostics.status is DiagnosticStatus.UNAVAILABLE
    assert empty.diagnostics.reason_codes == tuple(sorted(empty.diagnostics.reason_codes))
    assert "dense_probe_empty" in empty.diagnostics.reason_codes
    assert "eligible_catalog_empty" in empty.diagnostics.reason_codes
    assert one_mode.certainty is None
    assert "insufficient_modes" in one_mode.diagnostics.reason_codes


def test_unapproved_calibration_never_emits_measured_certainty() -> None:
    estimate = estimate_transparency(_evidence(), _calibration(approved=False))

    assert estimate.certainty is None
    assert estimate.controller_fallback == 0.5
    assert estimate.raw_mode_coherence == pytest.approx(0.4)
    assert estimate.diagnostics.status is DiagnosticStatus.UNAVAILABLE
    assert "calibration_unapproved" in estimate.diagnostics.reason_codes

    belief = project_search_belief(estimate)
    assert belief.certainty_evidence.quality_status is ProbeQuality.INSUFFICIENT


def test_projection_obeys_session_context_quality_and_raw_display_contract() -> None:
    estimate = estimate_transparency(
        _evidence(
            mode_coherence=-0.1,
            duplicate_warning=True,
            hard_filter_relaxed=True,
        ),
        TransparencyCalibration(
            policy_id="signed_mode_v1",
            low_anchor=-0.2,
            high_anchor=0.2,
        ),
    )

    assert estimate.certainty == pytest.approx(0.25)
    assert estimate.diagnostics.status is DiagnosticStatus.DEGRADED
    belief = project_search_belief(estimate)

    assert belief.based_on_intent_version == 7
    assert belief.certainty == pytest.approx(0.25)
    assert belief.certainty_method == "signed_mode_v1"
    assert belief.certainty_evidence.probe_id == "probe-17"
    assert belief.certainty_evidence.probe_size == 4
    assert belief.certainty_evidence.raw_concentration == 0.0
    assert belief.certainty_evidence.quality_status is ProbeQuality.VALID
    assert belief.certainty_evidence.quality_reasons == ()
    assert belief.candidate_modes == ()
    assert belief.facet_stats == ()
    validate_search_belief(belief, FacetRegistry(specs=()))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"policy_id": "Mode-v1"},
        {"low_anchor": float("nan")},
        {"low_anchor": 0.6, "high_anchor": 0.6},
        {"low_anchor": 0.7, "high_anchor": 0.6},
        {"high_anchor": 1.1},
        {"approved": 1},
    ],
)
def test_calibration_validation_is_strict(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "policy_id": "mode_calibration_v1",
        "low_anchor": 0.2,
        "high_anchor": 0.6,
        "approved": True,
    }
    values.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        TransparencyCalibration(**values)  # type: ignore[arg-type]


def test_evidence_rejects_unbound_or_internally_inconsistent_observations() -> None:
    with pytest.raises(ValueError, match="unavailable lexical evidence"):
        _evidence(
            lexical_available=False,
            lexical_hits=("A",),
            lexical_token_coverage=None,
            lexical_mean_normalized_idf=None,
        )
    with pytest.raises(ValueError, match="mode_count"):
        _evidence(mode_count=5)
    with pytest.raises(ValueError, match="unique"):
        _evidence(dense_hits=("A", "A", "B", "C"))
