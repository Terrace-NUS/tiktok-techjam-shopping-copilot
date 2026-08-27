"""Tests for immutable session-context leaf values."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from enum import Enum
from typing import cast

import pytest

from shopping_copilot.session_context.errors import ErrorCode, SessionContextError
from shopping_copilot.session_context.models import (
    MASS_TOLERANCE,
    CandidateMode,
    CertaintyEvidence,
    Commitment,
    FacetStats,
    FeedbackSignal,
    IntentState,
    Operator,
    Preference,
    PreferenceDraft,
    PreferenceSource,
    ProbeQuality,
    ProfilePrior,
    SearchBelief,
    SemanticPolarity,
    ValueMass,
)
from shopping_copilot.session_context.validation import validate_profile_prior

ENUM_WIRE_VALUES: tuple[tuple[type[Enum], tuple[str, ...]], ...] = (
    (Operator, ("eq", "neq", "in", "not_in", "lt", "le", "gt", "ge")),
    (SemanticPolarity, ("positive", "negative")),
    (Commitment, ("hard", "soft")),
    (
        PreferenceSource,
        ("user_explicit", "behavioral_feedback", "system_inferred"),
    ),
    (ProbeQuality, ("valid", "low_quality", "insufficient")),
    (
        FeedbackSignal,
        ("positive", "negative", "selected", "rejected", "comparative"),
    ),
)


def _profile(**overrides: object) -> ProfilePrior:
    values: dict[str, object] = {
        "purchase_frequency": "monthly",
        "average_prior_rating": 4.5,
        "rating_style": "balanced",
        "preference_tags": ("durable", "minimal"),
        "summary": "Prefers durable products.",
    }
    values.update(overrides)
    return ProfilePrior(**values)  # type: ignore[arg-type]


def _preference() -> Preference:
    return Preference(
        id="p_1_0_0",
        facet="color",
        operator=Operator.IN,
        value=("black", "blue"),
        semantic_text="black or blue",
        semantic_polarity=SemanticPolarity.POSITIVE,
        commitment=Commitment.HARD,
        source=PreferenceSource.USER_EXPLICIT,
        source_turn=1,
        evidence_text="I want black or blue.",
        interpretation_confidence=1.0,
    )


def _search_belief() -> SearchBelief:
    evidence = CertaintyEvidence(
        probe_id="probe-1",
        probe_size=4,
        raw_concentration=0.75,
        quality_status=ProbeQuality.VALID,
        quality_reasons=(),
    )
    top_value = ValueMass(value="black", mass=0.75)
    facet_stats = FacetStats(
        facet="color",
        entropy=0.5,
        coverage=1.0,
        top_values=(top_value,),
    )
    candidate_mode = CandidateMode(
        id="mode-1",
        label="dark colors",
        mass=0.75,
        representative_ids=("sku-1", "sku-2"),
    )
    return SearchBelief(
        based_on_intent_version=1,
        certainty=0.75,
        certainty_method="bods_v1",
        certainty_evidence=evidence,
        candidate_modes=(candidate_mode,),
        facet_stats=(facet_stats,),
    )


def _model_instances() -> tuple[object, ...]:
    profile = _profile()
    draft = PreferenceDraft(
        facet=None,
        operator=None,
        value=None,
        semantic_text="easy to carry",
        semantic_polarity=SemanticPolarity.POSITIVE,
        commitment=Commitment.SOFT,
        source=PreferenceSource.SYSTEM_INFERRED,
        source_turn=1,
        evidence_text="Something easy to carry.",
        interpretation_confidence=0.6,
    )
    preference = _preference()
    intent = IntentState(
        goal="buy a backpack",
        preferences=(preference,),
        dont_care_facets=frozenset({"brand"}),
        version=1,
    )
    belief = _search_belief()
    return (
        profile,
        draft,
        preference,
        intent,
        belief.certainty_evidence,
        belief.facet_stats[0].top_values[0],
        belief.facet_stats[0],
        belief.candidate_modes[0],
        belief,
    )


@pytest.mark.parametrize(
    ("enum_type", "wire_values"),
    ENUM_WIRE_VALUES,
    ids=lambda value: value.__name__ if isinstance(value, type) else None,
)
def test_enums_have_exact_stable_wire_values(
    enum_type: type[Enum],
    wire_values: tuple[str, ...],
) -> None:
    assert tuple(member.value for member in enum_type) == wire_values
    assert tuple(enum_type(value) for value in wire_values) == tuple(enum_type)

    with pytest.raises(ValueError):
        enum_type("unknown_v1_value")


def test_mass_tolerance_is_the_frozen_v1_value() -> None:
    assert MASS_TOLERANCE == 1e-9


@pytest.mark.parametrize(
    "instance",
    _model_instances(),
    ids=lambda instance: type(instance).__name__,
)
def test_model_values_are_frozen_and_slotted(instance: object) -> None:
    first_field = fields(instance)[0]

    assert type(instance).__dataclass_params__.frozen is True
    assert not hasattr(instance, "__dict__")
    with pytest.raises(FrozenInstanceError):
        setattr(instance, first_field.name, object())


def test_model_collection_fields_use_immutable_domain_types() -> None:
    profile = _profile()
    preference = _preference()
    intent = IntentState(
        goal=None,
        preferences=(preference,),
        dont_care_facets=frozenset({"brand"}),
        version=0,
    )
    belief = _search_belief()

    assert type(profile.preference_tags) is tuple
    assert type(preference.value) is tuple
    assert type(intent.preferences) is tuple
    assert type(intent.dont_care_facets) is frozenset
    assert type(belief.certainty_evidence.quality_reasons) is tuple
    assert type(belief.candidate_modes) is tuple
    assert type(belief.candidate_modes[0].representative_ids) is tuple
    assert type(belief.facet_stats) is tuple
    assert type(belief.facet_stats[0].top_values) is tuple


def test_initial_intent_has_the_frozen_empty_shape() -> None:
    intent = IntentState(
        goal=None,
        preferences=(),
        dont_care_facets=frozenset(),
        version=0,
    )

    assert intent == IntentState(
        goal=None,
        preferences=(),
        dont_care_facets=frozenset(),
        version=0,
    )


def test_preference_draft_can_represent_an_ungrounded_interpretation() -> None:
    draft = PreferenceDraft(
        facet=None,
        operator=None,
        value=None,
        semantic_text=None,
        semantic_polarity=None,
        commitment=Commitment.SOFT,
        source=PreferenceSource.SYSTEM_INFERRED,
        source_turn=1,
        evidence_text="Unparsed model output",
        interpretation_confidence=0.0,
    )

    assert draft.facet is None
    assert draft.operator is None
    assert draft.semantic_text is None


@pytest.mark.parametrize("rating", [None, 4, 4.5])
def test_profile_accepts_nullable_or_finite_numeric_rating(rating: float | None) -> None:
    validate_profile_prior(_profile(average_prior_rating=rating))


def test_profile_validation_does_not_narrow_official_text_values() -> None:
    validate_profile_prior(
        _profile(
            purchase_frequency="",
            rating_style="",
            preference_tags=(),
            summary="",
        )
    )


@pytest.mark.parametrize("field_name", ["purchase_frequency", "rating_style", "summary"])
def test_profile_rejects_non_string_text_fields(field_name: str) -> None:
    profile = replace(_profile(), **{field_name: None})

    with pytest.raises(SessionContextError) as caught:
        validate_profile_prior(profile)

    assert caught.value.code is ErrorCode.INVALID_PROFILE
    assert caught.value.path == (field_name,)


@pytest.mark.parametrize("rating", [True, "4.5", float("nan"), float("inf"), float("-inf")])
def test_profile_rejects_invalid_rating_values(rating: object) -> None:
    with pytest.raises(SessionContextError) as caught:
        validate_profile_prior(_profile(average_prior_rating=rating))

    assert caught.value.code is ErrorCode.INVALID_PROFILE
    assert caught.value.path == ("average_prior_rating",)


@pytest.mark.parametrize("tags", [["durable"], ("durable", 3), None])
def test_profile_rejects_non_tuple_or_non_string_tags(tags: object) -> None:
    with pytest.raises(SessionContextError) as caught:
        validate_profile_prior(_profile(preference_tags=tags))

    assert caught.value.code is ErrorCode.INVALID_PROFILE
    assert caught.value.path == ("preference_tags",)


def test_profile_validator_rejects_a_non_profile_value() -> None:
    with pytest.raises(SessionContextError) as caught:
        validate_profile_prior(cast(ProfilePrior, object()))

    assert caught.value.code is ErrorCode.INVALID_PROFILE
    assert caught.value.path == ()
