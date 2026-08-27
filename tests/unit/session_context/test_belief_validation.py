"""Tests for canonical SearchBelief validation."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import cast

import pytest

import shopping_copilot.session_context as session_context
from shopping_copilot.session_context.errors import ErrorCode, SessionContextError
from shopping_copilot.session_context.models import (
    CandidateMode,
    CertaintyEvidence,
    FacetStats,
    ProbeQuality,
    ScalarValue,
    SearchBelief,
    ValueMass,
)
from shopping_copilot.session_context.registry import (
    CATEGORICAL_OPERATORS,
    NUMERIC_OPERATORS,
    FacetKind,
    FacetRegistry,
    FacetSpec,
    canonical_number,
    canonical_text,
)
from shopping_copilot.session_context.validation import validate_search_belief


def _identity_scalar(value: ScalarValue) -> ScalarValue:
    return value


@pytest.fixture
def registry() -> FacetRegistry:
    return FacetRegistry(
        specs=(
            FacetSpec(
                id="color",
                kind=FacetKind.CATEGORICAL,
                operators=CATEGORICAL_OPERATORS,
                normalizer=canonical_text,
            ),
            FacetSpec(
                id="budget",
                kind=FacetKind.NUMERIC,
                operators=NUMERIC_OPERATORS,
                normalizer=canonical_number,
            ),
        )
    )


def _evidence(**changes: object) -> CertaintyEvidence:
    value = CertaintyEvidence(
        probe_id="probe-1",
        probe_size=10,
        raw_concentration=0.8,
        quality_status=ProbeQuality.VALID,
        quality_reasons=(),
    )
    return replace(value, **changes)


def _mode(**changes: object) -> CandidateMode:
    value = CandidateMode(
        id="mode-a",
        label="primary mode",
        mass=1.0,
        representative_ids=("sku-1", "sku-2"),
    )
    return replace(value, **changes)


def _color_stats(**changes: object) -> FacetStats:
    value = FacetStats(
        facet="color",
        entropy=0.0,
        coverage=1.0,
        top_values=(ValueMass(value="black", mass=1.0),),
    )
    return replace(value, **changes)


def _belief(**changes: object) -> SearchBelief:
    value = SearchBelief(
        based_on_intent_version=0,
        certainty=0.8,
        certainty_method="bods_v1",
        certainty_evidence=_evidence(),
        candidate_modes=(_mode(),),
        facet_stats=(_color_stats(),),
    )
    return replace(value, **changes)


def _assert_code(
    expected: ErrorCode,
    belief: object,
    registry: FacetRegistry,
) -> SessionContextError:
    with pytest.raises(SessionContextError) as caught:
        validate_search_belief(cast(SearchBelief, belief), registry)
    assert caught.value.code is expected
    return caught.value


def test_validate_search_belief_is_exported_from_the_public_package() -> None:
    assert session_context.validate_search_belief is validate_search_belief


@pytest.mark.parametrize("certainty", [0, 0.0, 1, 1.0])
def test_valid_quality_accepts_probability_endpoints(
    registry: FacetRegistry,
    certainty: int | float,
) -> None:
    validate_search_belief(
        _belief(
            certainty=certainty,
            certainty_evidence=_evidence(raw_concentration=certainty),
        ),
        registry,
    )


@pytest.mark.parametrize("status", [ProbeQuality.LOW_QUALITY, ProbeQuality.INSUFFICIENT])
@pytest.mark.parametrize("raw_concentration", [None, 0.25])
def test_unavailable_quality_accepts_optional_raw_concentration(
    registry: FacetRegistry,
    status: ProbeQuality,
    raw_concentration: float | None,
) -> None:
    validate_search_belief(
        _belief(
            certainty=None,
            certainty_evidence=_evidence(
                probe_size=0,
                raw_concentration=raw_concentration,
                quality_status=status,
                quality_reasons=("insufficient_candidates",),
            ),
        ),
        registry,
    )


@pytest.mark.parametrize(
    "belief",
    [
        object(),
        _belief(based_on_intent_version=True),
        _belief(based_on_intent_version=-1),
        _belief(certainty_method=""),
        _belief(certainty_method="BODS_v1"),
        _belief(certainty_method="bods-v1"),
        _belief(certainty_method="bods v1"),
        _belief(certainty_evidence=object()),
    ],
)
def test_belief_and_top_level_fields_require_exact_canonical_shapes(
    registry: FacetRegistry,
    belief: object,
) -> None:
    _assert_code(ErrorCode.INVALID_PROBE_EVIDENCE, belief, registry)


@pytest.mark.parametrize(
    "evidence",
    [
        _evidence(probe_id="  "),
        _evidence(probe_size=True),
        _evidence(probe_size=-1),
        _evidence(quality_status="valid"),
        _evidence(quality_reasons=[]),
        _evidence(probe_size=0),
        _evidence(quality_reasons=("unexpected_reason",)),
        _evidence(
            quality_status=ProbeQuality.INSUFFICIENT,
            raw_concentration=None,
            quality_reasons=(),
        ),
    ],
)
def test_probe_evidence_shape_and_quality_requirements_are_enforced(
    registry: FacetRegistry,
    evidence: CertaintyEvidence,
) -> None:
    _assert_code(
        ErrorCode.INVALID_PROBE_EVIDENCE,
        _belief(certainty_evidence=evidence),
        registry,
    )


@pytest.mark.parametrize(
    "reasons",
    [
        ("",),
        ("Timeout",),
        ("low-quality",),
        ("low quality",),
        ("timeout", "timeout"),
        ("timeout", "low_quality"),
    ],
)
def test_non_valid_quality_reasons_are_identifiers_unique_and_sorted(
    registry: FacetRegistry,
    reasons: tuple[str, ...],
) -> None:
    belief = _belief(
        certainty=None,
        certainty_evidence=_evidence(
            raw_concentration=None,
            quality_status=ProbeQuality.LOW_QUALITY,
            quality_reasons=reasons,
        ),
    )
    _assert_code(ErrorCode.INVALID_PROBE_EVIDENCE, belief, registry)


@pytest.mark.parametrize(
    "belief",
    [
        _belief(certainty=None),
        _belief(certainty_evidence=_evidence(raw_concentration=None)),
        _belief(
            certainty=0.4,
            certainty_evidence=_evidence(
                raw_concentration=None,
                quality_status=ProbeQuality.LOW_QUALITY,
                quality_reasons=("low_recall",),
            ),
        ),
    ],
)
def test_certainty_presence_follows_the_quality_truth_table(
    registry: FacetRegistry,
    belief: SearchBelief,
) -> None:
    _assert_code(ErrorCode.CERTAINTY_QUALITY_MISMATCH, belief, registry)


@pytest.mark.parametrize("value", [True, math.nan, math.inf, -math.inf, -0.01, 1.01])
def test_certainty_and_raw_concentration_reject_invalid_probabilities(
    registry: FacetRegistry,
    value: object,
) -> None:
    _assert_code(ErrorCode.INVALID_MASS_DISTRIBUTION, _belief(certainty=value), registry)
    _assert_code(
        ErrorCode.INVALID_MASS_DISTRIBUTION,
        _belief(certainty_evidence=_evidence(raw_concentration=value)),
        registry,
    )


@pytest.mark.parametrize(
    "candidate_modes",
    [
        [_mode()],
        (object(),),
        (_mode(id=""),),
        (_mode(label="  "),),
        (_mode(representative_ids=()),),
        (_mode(representative_ids=["sku-1"]),),
        (_mode(representative_ids=("",)),),
        (_mode(representative_ids=("sku-1", "sku-1")),),
    ],
)
def test_candidate_mode_shapes_are_exact_and_nonempty(
    registry: FacetRegistry,
    candidate_modes: object,
) -> None:
    _assert_code(
        ErrorCode.INVALID_PROBE_EVIDENCE,
        _belief(candidate_modes=candidate_modes),
        registry,
    )


@pytest.mark.parametrize("mass", [True, math.nan, math.inf, -math.inf, -0.1, 0, 1.1])
def test_candidate_mode_mass_is_finite_positive_probability(
    registry: FacetRegistry,
    mass: object,
) -> None:
    _assert_code(
        ErrorCode.INVALID_MASS_DISTRIBUTION,
        _belief(candidate_modes=(_mode(mass=mass),)),
        registry,
    )


def test_duplicate_candidate_mode_ids_have_a_specific_error(
    registry: FacetRegistry,
) -> None:
    modes = (_mode(mass=0.6), _mode(label="duplicate", mass=0.4))
    error = _assert_code(
        ErrorCode.DUPLICATE_MODE_ID,
        _belief(candidate_modes=modes),
        registry,
    )
    assert error.path == ("candidate_modes", 1, "id")


def test_candidate_modes_use_descending_mass_then_id_order(
    registry: FacetRegistry,
) -> None:
    canonical = (
        _mode(id="mode-a", mass=0.5),
        _mode(id="mode-b", label="secondary", mass=0.5, representative_ids=("sku-3",)),
    )
    validate_search_belief(_belief(candidate_modes=canonical), registry)
    _assert_code(
        ErrorCode.NON_CANONICAL_VALUE,
        _belief(candidate_modes=tuple(reversed(canonical))),
        registry,
    )


def test_candidate_mode_total_uses_the_frozen_mass_tolerance(
    registry: FacetRegistry,
) -> None:
    within_tolerance = (
        _mode(id="mode-a", mass=0.5000000005),
        _mode(id="mode-b", label="secondary", mass=0.5, representative_ids=("sku-3",)),
    )
    validate_search_belief(_belief(candidate_modes=within_tolerance), registry)

    above_tolerance = (
        _mode(id="mode-a", mass=0.500000002),
        _mode(id="mode-b", label="secondary", mass=0.5, representative_ids=("sku-3",)),
    )
    _assert_code(
        ErrorCode.INVALID_MASS_DISTRIBUTION,
        _belief(candidate_modes=above_tolerance),
        registry,
    )


@pytest.mark.parametrize(
    "facet_stats",
    [
        [_color_stats()],
        (object(),),
        (_color_stats(facet=""),),
        (_color_stats(top_values=[]),),
        (_color_stats(top_values=()),),
        (_color_stats(top_values=(object(),)),),
    ],
)
def test_facet_stat_shapes_are_exact_and_nonempty(
    registry: FacetRegistry,
    facet_stats: object,
) -> None:
    _assert_code(
        ErrorCode.INVALID_PROBE_EVIDENCE,
        _belief(facet_stats=facet_stats),
        registry,
    )


@pytest.mark.parametrize("field", ["entropy", "coverage"])
@pytest.mark.parametrize("value", [True, math.nan, math.inf, -math.inf, -0.1, 1.1])
def test_facet_probabilities_are_finite_and_bounded(
    registry: FacetRegistry,
    field: str,
    value: object,
) -> None:
    _assert_code(
        ErrorCode.INVALID_MASS_DISTRIBUTION,
        _belief(facet_stats=(_color_stats(**{field: value}),)),
        registry,
    )


def test_emitted_facet_requires_positive_coverage(registry: FacetRegistry) -> None:
    _assert_code(
        ErrorCode.INVALID_MASS_DISTRIBUTION,
        _belief(facet_stats=(_color_stats(coverage=0),)),
        registry,
    )


def test_unknown_and_duplicate_facet_stats_have_specific_errors(
    registry: FacetRegistry,
) -> None:
    _assert_code(
        ErrorCode.UNKNOWN_FACET,
        _belief(facet_stats=(_color_stats(facet="colour"),)),
        registry,
    )
    _assert_code(
        ErrorCode.DUPLICATE_FACET_STATS,
        _belief(facet_stats=(_color_stats(), _color_stats())),
        registry,
    )


def test_facet_stats_are_sorted_by_facet_id(registry: FacetRegistry) -> None:
    budget = FacetStats(
        facet="budget",
        entropy=0,
        coverage=1,
        top_values=(ValueMass(value=100, mass=1),),
    )
    color = _color_stats()
    validate_search_belief(_belief(facet_stats=(budget, color)), registry)
    _assert_code(
        ErrorCode.NON_CANONICAL_VALUE,
        _belief(facet_stats=(color, budget)),
        registry,
    )


@pytest.mark.parametrize("mass", [True, math.nan, math.inf, -math.inf, -0.1, 0, 1.1])
def test_top_value_mass_is_finite_positive_probability(
    registry: FacetRegistry,
    mass: object,
) -> None:
    stats = _color_stats(top_values=(ValueMass(value="black", mass=mass),))
    _assert_code(
        ErrorCode.INVALID_MASS_DISTRIBUTION,
        _belief(facet_stats=(stats,)),
        registry,
    )


def test_top_value_total_uses_the_frozen_mass_tolerance(registry: FacetRegistry) -> None:
    within_tolerance = _color_stats(
        entropy=0.5,
        top_values=(
            ValueMass(value="black", mass=0.5000000005),
            ValueMass(value="blue", mass=0.5),
        ),
    )
    validate_search_belief(_belief(facet_stats=(within_tolerance,)), registry)

    above_tolerance = replace(
        within_tolerance,
        top_values=(
            ValueMass(value="black", mass=0.500000002),
            ValueMass(value="blue", mass=0.5),
        ),
    )
    _assert_code(
        ErrorCode.INVALID_MASS_DISTRIBUTION,
        _belief(facet_stats=(above_tolerance,)),
        registry,
    )


def test_duplicate_top_values_have_a_specific_error(registry: FacetRegistry) -> None:
    stats = _color_stats(
        entropy=0.5,
        top_values=(
            ValueMass(value="black", mass=0.6),
            ValueMass(value="black", mass=0.4),
        ),
    )
    error = _assert_code(
        ErrorCode.DUPLICATE_FACET_VALUE,
        _belief(facet_stats=(stats,)),
        registry,
    )
    assert error.path == ("facet_stats", 0, "top_values", 1, "value")


def test_top_values_use_descending_mass_then_scalar_wire_key(
    registry: FacetRegistry,
) -> None:
    canonical = _color_stats(
        entropy=0.5,
        top_values=(
            ValueMass(value="black", mass=0.5),
            ValueMass(value="blue", mass=0.5),
        ),
    )
    validate_search_belief(_belief(facet_stats=(canonical,)), registry)
    noncanonical = replace(canonical, top_values=tuple(reversed(canonical.top_values)))
    _assert_code(
        ErrorCode.NON_CANONICAL_VALUE,
        _belief(facet_stats=(noncanonical,)),
        registry,
    )


def test_scalar_wire_key_uses_the_frozen_type_rank() -> None:
    mixed_registry = FacetRegistry(
        specs=(
            FacetSpec(
                id="mixed",
                kind=FacetKind.CATEGORICAL,
                operators=CATEGORICAL_OPERATORS,
                normalizer=_identity_scalar,
            ),
        )
    )
    values = (
        ValueMass(value=True, mass=0.2),
        ValueMass(value=1, mass=0.2),
        ValueMass(value=1.5, mass=0.2),
        ValueMass(value="1", mass=0.2),
    )
    stats = FacetStats(facet="mixed", entropy=0.5, coverage=1, top_values=values)
    validate_search_belief(_belief(facet_stats=(stats,)), mixed_registry)

    swapped = replace(stats, top_values=(values[1], values[0], *values[2:]))
    _assert_code(
        ErrorCode.NON_CANONICAL_VALUE,
        _belief(facet_stats=(swapped,)),
        mixed_registry,
    )


def test_scalar_wire_key_handles_arbitrary_size_integer(registry: FacetRegistry) -> None:
    stats = FacetStats(
        facet="budget",
        entropy=0,
        coverage=1,
        top_values=(ValueMass(value=10**4_999, mass=1),),
    )

    validate_search_belief(_belief(facet_stats=(stats,)), registry)


@pytest.mark.parametrize(
    "stats",
    [
        _color_stats(top_values=(ValueMass(value=" Black ", mass=1),)),
        FacetStats(
            facet="budget",
            entropy=0,
            coverage=1,
            top_values=(ValueMass(value=100.0, mass=1),),
        ),
        FacetStats(
            facet="budget",
            entropy=0,
            coverage=1,
            top_values=(ValueMass(value=True, mass=1),),
        ),
    ],
)
def test_top_values_must_match_the_facet_normalizer_with_exact_type(
    registry: FacetRegistry,
    stats: FacetStats,
) -> None:
    _assert_code(
        ErrorCode.NON_CANONICAL_VALUE,
        _belief(facet_stats=(stats,)),
        registry,
    )
