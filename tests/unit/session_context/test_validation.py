from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any

import pytest

from shopping_copilot.session_context import (
    CATEGORICAL_OPERATORS,
    NUMERIC_OPERATORS,
    AddPreference,
    ClearFacet,
    Commitment,
    ErrorCode,
    FacetKind,
    FacetRegistry,
    FacetSpec,
    IntentState,
    Operator,
    Preference,
    PreferenceDraft,
    PreferenceSource,
    RemovePreference,
    ReplaceFacet,
    SemanticPolarity,
    SessionContextError,
    SetDontCare,
    StateUpdateBatch,
    SwitchGoal,
    canonical_number,
    canonical_text,
    validate_intent_state,
    validate_preference,
    validate_state_update_batch,
)


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


def preference(**changes: Any) -> Preference:
    values: dict[str, Any] = {
        "facet": "color",
        "operator": Operator.EQ,
        "value": "red",
        "semantic_text": None,
        "semantic_polarity": None,
        "commitment": Commitment.HARD,
        "source": PreferenceSource.USER_EXPLICIT,
        "source_turn": 1,
        "evidence_text": "red please",
        "interpretation_confidence": 1.0,
        "id": "p_1_0_0",
    }
    values.update(changes)
    return Preference(**values)


def semantic_preference(**changes: Any) -> Preference:
    values: dict[str, Any] = {
        "facet": None,
        "operator": None,
        "value": None,
        "semantic_text": "easy to repair",
        "semantic_polarity": SemanticPolarity.POSITIVE,
    }
    values.update(changes)
    return preference(**values)


def assert_code(expected: ErrorCode, callable_: Any) -> SessionContextError:
    with pytest.raises(SessionContextError) as captured:
        callable_()
    assert captured.value.code is expected
    return captured.value


@pytest.mark.parametrize(
    "candidate",
    [
        preference(),
        preference(operator=Operator.IN, value=("blue", "red")),
        preference(facet="budget", operator=Operator.GE, value=10),
        semantic_preference(),
        preference(
            semantic_text="not red",
            semantic_polarity=SemanticPolarity.NEGATIVE,
            operator=Operator.NEQ,
        ),
        preference(interpretation_confidence=0),
    ],
)
def test_valid_committed_preferences(
    registry: FacetRegistry,
    candidate: Preference,
) -> None:
    validate_preference(candidate, registry)


def test_draft_dto_can_carry_input_that_has_not_reached_grounding() -> None:
    draft = PreferenceDraft(
        facet="color",
        operator=Operator.IN,
        value=(" Red ", "BLUE"),
        semantic_text=None,
        semantic_polarity=None,
        commitment=Commitment.HARD,
        source=PreferenceSource.USER_EXPLICIT,
        source_turn=1,
        evidence_text="colors",
        interpretation_confidence=0.8,
    )

    assert draft.value == (" Red ", "BLUE")


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (preference(operator=None), ErrorCode.INVALID_REPRESENTATION),
        (
            semantic_preference(semantic_polarity=None),
            ErrorCode.INVALID_REPRESENTATION,
        ),
        (
            preference(
                facet=None,
                operator=None,
                value=None,
                semantic_text=None,
                semantic_polarity=None,
            ),
            ErrorCode.INVALID_REPRESENTATION,
        ),
        (preference(id=""), ErrorCode.INVALID_REPRESENTATION),
        (preference(id="p_01_0_0"), ErrorCode.NON_CANONICAL_VALUE),
        (preference(evidence_text="  "), ErrorCode.INVALID_REPRESENTATION),
        (preference(source_turn=0), ErrorCode.INVALID_SOURCE_TURN),
        (preference(interpretation_confidence=True), ErrorCode.INVALID_CONFIDENCE),
        (preference(interpretation_confidence=10**5000), ErrorCode.INVALID_CONFIDENCE),
        (preference(interpretation_confidence=math.inf), ErrorCode.INVALID_CONFIDENCE),
        (
            preference(
                source=PreferenceSource.SYSTEM_INFERRED,
                commitment=Commitment.HARD,
            ),
            ErrorCode.INVALID_COMMITMENT_FOR_SOURCE,
        ),
        (
            preference(
                semantic_text="not red",
                semantic_polarity=SemanticPolarity.NEGATIVE,
            ),
            ErrorCode.INVALID_REPRESENTATION,
        ),
        (preference(operator=Operator.IN, value="red"), ErrorCode.INVALID_OPERATOR_VALUE),
        (preference(operator=Operator.IN, value=()), ErrorCode.INVALID_OPERATOR_VALUE),
        (
            preference(operator=Operator.IN, value=("red", 1)),
            ErrorCode.INVALID_OPERATOR_VALUE,
        ),
        (
            preference(facet="budget", operator=Operator.GE, value=True),
            ErrorCode.INVALID_OPERATOR_VALUE,
        ),
        (
            preference(facet="budget", operator=Operator.GE, value=math.nan),
            ErrorCode.INVALID_OPERATOR_VALUE,
        ),
        (preference(value=" Red "), ErrorCode.NON_CANONICAL_VALUE),
        (
            preference(operator=Operator.IN, value=("red", "blue")),
            ErrorCode.NON_CANONICAL_VALUE,
        ),
        (
            preference(operator=Operator.IN, value=("red", "red")),
            ErrorCode.NON_CANONICAL_VALUE,
        ),
        (
            preference(facet="budget", operator=Operator.GE, value=10.0),
            ErrorCode.NON_CANONICAL_VALUE,
        ),
        (preference(facet="other"), ErrorCode.UNKNOWN_FACET),
        (
            preference(facet="color", operator=Operator.LT, value=10),
            ErrorCode.INVALID_OPERATOR_FOR_FACET,
        ),
        (
            preference(facet="color", operator=Operator.LT, value="cheap"),
            ErrorCode.INVALID_OPERATOR_FOR_FACET,
        ),
        (
            preference(facet="unknown", operator=Operator.IN, value=()),
            ErrorCode.UNKNOWN_FACET,
        ),
    ],
)
def test_invalid_committed_preferences_have_stable_codes(
    registry: FacetRegistry,
    candidate: Preference,
    expected: ErrorCode,
) -> None:
    assert_code(expected, lambda: validate_preference(candidate, registry))


def test_initial_and_supported_intent_states_are_valid(registry: FacetRegistry) -> None:
    states = (
        IntentState(goal=None, preferences=(), dont_care_facets=frozenset(), version=0),
        IntentState(
            goal="find shoes",
            preferences=(),
            dont_care_facets=frozenset({"color"}),
            version=1,
        ),
        IntentState(
            goal=None,
            preferences=(semantic_preference(),),
            dont_care_facets=frozenset(),
            version=0,
        ),
        IntentState(
            goal=None,
            preferences=(preference(operator=Operator.NEQ),),
            dont_care_facets=frozenset(),
            version=0,
        ),
        IntentState(
            goal=None,
            preferences=(
                preference(facet="budget", operator=Operator.GE, value=10),
                preference(
                    id="p_1_0_1",
                    facet="budget",
                    operator=Operator.LE,
                    value=10,
                ),
            ),
            dont_care_facets=frozenset(),
            version=0,
        ),
    )
    for state in states:
        validate_intent_state(state, registry)


@pytest.mark.parametrize(
    ("preferences", "dont_care", "goal", "version", "expected"),
    [
        ((), frozenset(), "  ", 0, ErrorCode.INVALID_GOAL),
        ((), frozenset(), None, -1, ErrorCode.NON_CANONICAL_VALUE),
        ((), frozenset({"unknown"}), None, 0, ErrorCode.UNKNOWN_FACET),
        (
            (preference(), preference()),
            frozenset(),
            None,
            0,
            ErrorCode.DUPLICATE_PREFERENCE_ID,
        ),
        (
            (
                preference(),
                preference(source_turn=2, evidence_text="same meaning, new evidence"),
            ),
            frozenset(),
            None,
            0,
            ErrorCode.DUPLICATE_PREFERENCE_ID,
        ),
        (
            (preference(), preference(id="p_1_0_0", value="blue")),
            frozenset(),
            None,
            0,
            ErrorCode.PREFERENCE_ID_CONFLICT,
        ),
        (
            (preference(), preference(id="p_1_0_1", source_turn=2, evidence_text="again")),
            frozenset(),
            None,
            0,
            ErrorCode.DUPLICATE_PREFERENCE_SEMANTICS,
        ),
        (
            (preference(id="p_2_0_0"), preference(id="p_1_0_0", value="blue")),
            frozenset(),
            None,
            0,
            ErrorCode.NON_CANONICAL_VALUE,
        ),
        (
            (preference(),),
            frozenset({"color"}),
            None,
            0,
            ErrorCode.DONT_CARE_CONFLICT,
        ),
    ],
)
def test_invalid_intent_identity_and_shape(
    registry: FacetRegistry,
    preferences: tuple[Preference, ...],
    dont_care: frozenset[str],
    goal: str | None,
    version: int,
    expected: ErrorCode,
) -> None:
    state = IntentState(
        goal=goal,
        preferences=preferences,
        dont_care_facets=dont_care,
        version=version,
    )
    assert_code(expected, lambda: validate_intent_state(state, registry))


@pytest.mark.parametrize(
    ("preferences", "expected"),
    [
        (
            (
                preference(),
                preference(id="p_1_0_1", operator=Operator.IN, value=("blue", "red")),
            ),
            ErrorCode.MULTIPLE_POSITIVE_SELECTOR,
        ),
        (
            (
                preference(operator=Operator.NEQ),
                preference(id="p_1_0_1", operator=Operator.NOT_IN, value=("blue",)),
            ),
            ErrorCode.MULTIPLE_NEGATIVE_SELECTOR,
        ),
        (
            (
                preference(),
                preference(
                    id="p_1_0_1",
                    value="blue",
                    commitment=Commitment.SOFT,
                ),
            ),
            ErrorCode.EMPTY_CATEGORICAL_DOMAIN,
        ),
        (
            (
                preference(),
                preference(id="p_1_0_1", operator=Operator.NEQ),
            ),
            ErrorCode.EMPTY_CATEGORICAL_DOMAIN,
        ),
        (
            (
                preference(facet="budget", operator=Operator.GT, value=10),
                preference(
                    id="p_1_0_1",
                    facet="budget",
                    operator=Operator.LE,
                    value=10,
                ),
            ),
            ErrorCode.EMPTY_NUMERIC_INTERSECTION,
        ),
        (
            (
                preference(facet="budget", operator=Operator.GE, value=10),
                preference(
                    id="p_1_0_1",
                    facet="budget",
                    operator=Operator.LE,
                    value=9,
                    commitment=Commitment.SOFT,
                ),
            ),
            ErrorCode.EMPTY_NUMERIC_INTERSECTION,
        ),
        (
            (
                preference(facet="budget", operator=Operator.GE, value=10),
                preference(
                    id="p_1_0_1",
                    facet="budget",
                    operator=Operator.GT,
                    value=20,
                ),
            ),
            ErrorCode.NON_CANONICAL_VALUE,
        ),
    ],
)
def test_canonical_facet_state_invariants(
    registry: FacetRegistry,
    preferences: tuple[Preference, ...],
    expected: ErrorCode,
) -> None:
    state = IntentState(
        goal=None,
        preferences=preferences,
        dont_care_facets=frozenset(),
        version=0,
    )
    assert_code(expected, lambda: validate_intent_state(state, registry))


def test_all_six_operation_shapes_and_empty_remove_are_locally_valid(
    registry: FacetRegistry,
) -> None:
    operations = (
        AddPreference(preference=preference()),
        ReplaceFacet(facet="color", preferences=(preference(),)),
        RemovePreference(preference_ids=()),
        ClearFacet(facet="color"),
        SetDontCare(facet="color"),
        SwitchGoal(new_goal="new goal"),
    )
    for operation in operations:
        validate_state_update_batch(
            StateUpdateBatch(turn=1, base_intent_version=0, operations=(operation,)),
            registry,
        )
    validate_state_update_batch(
        StateUpdateBatch(
            turn=1,
            base_intent_version=0,
            operations=(SwitchGoal(new_goal="new goal"), AddPreference(preference=preference())),
        ),
        registry,
    )


@pytest.mark.parametrize(
    ("batch", "expected"),
    [
        (StateUpdateBatch(turn=1, base_intent_version=0, operations=()), ErrorCode.EMPTY_BATCH),
        (
            StateUpdateBatch(
                turn=0,
                base_intent_version=0,
                operations=(ClearFacet(facet="color"),),
            ),
            ErrorCode.TURN_OUT_OF_ORDER,
        ),
        (
            StateUpdateBatch(
                turn=1,
                base_intent_version=-1,
                operations=(ClearFacet(facet="color"),),
            ),
            ErrorCode.STALE_BASE_VERSION,
        ),
        (
            StateUpdateBatch(
                turn=1,
                base_intent_version=0,
                operations=(SwitchGoal(new_goal="a"), SwitchGoal(new_goal="b")),
            ),
            ErrorCode.MULTIPLE_GOAL_SWITCH,
        ),
        (
            StateUpdateBatch(
                turn=1,
                base_intent_version=0,
                operations=(ClearFacet(facet="color"), SwitchGoal(new_goal="a")),
            ),
            ErrorCode.INVALID_OPERATION_ORDER,
        ),
        (
            StateUpdateBatch(
                turn=1,
                base_intent_version=0,
                operations=(ReplaceFacet(facet="color", preferences=()),),
            ),
            ErrorCode.EMPTY_REPLACEMENT,
        ),
        (
            StateUpdateBatch(
                turn=1,
                base_intent_version=0,
                operations=(
                    ReplaceFacet(
                        facet="color",
                        preferences=(semantic_preference(),),
                    ),
                ),
            ),
            ErrorCode.FACET_MISMATCH,
        ),
        (
            StateUpdateBatch(
                turn=1,
                base_intent_version=0,
                operations=(RemovePreference(preference_ids=("p_2_0_0", "p_1_0_0")),),
            ),
            ErrorCode.NON_CANONICAL_VALUE,
        ),
        (
            StateUpdateBatch(
                turn=1,
                base_intent_version=0,
                operations=(SwitchGoal(new_goal="  "),),
            ),
            ErrorCode.INVALID_GOAL,
        ),
        (
            StateUpdateBatch(
                turn=1,
                base_intent_version=0,
                operations=(ClearFacet(facet="other"),),
            ),
            ErrorCode.UNKNOWN_FACET,
        ),
    ],
)
def test_invalid_batch_and_operation_shapes(
    registry: FacetRegistry,
    batch: StateUpdateBatch,
    expected: ErrorCode,
) -> None:
    assert_code(expected, lambda: validate_state_update_batch(batch, registry))


def test_nested_operation_error_has_stable_path_and_index(registry: FacetRegistry) -> None:
    batch = StateUpdateBatch(
        turn=1,
        base_intent_version=0,
        operations=(AddPreference(preference=preference(source_turn=0)),),
    )
    error = assert_code(
        ErrorCode.INVALID_SOURCE_TURN,
        lambda: validate_state_update_batch(batch, registry),
    )
    assert error.operation_index == 0
    assert error.path == ("operations", 0, "preference", "source_turn")


def test_replacement_state_error_retains_its_collection_path(
    registry: FacetRegistry,
) -> None:
    batch = StateUpdateBatch(
        turn=1,
        base_intent_version=0,
        operations=(
            ReplaceFacet(
                facet="color",
                preferences=(
                    preference(),
                    preference(
                        id="p_1_0_1",
                        operator=Operator.IN,
                        value=("blue", "red"),
                    ),
                ),
            ),
        ),
    )

    error = assert_code(
        ErrorCode.MULTIPLE_POSITIVE_SELECTOR,
        lambda: validate_state_update_batch(batch, registry),
    )
    assert error.path == ("operations", 0, "preferences")


def test_arbitrarily_large_canonical_id_is_compared_without_integer_parsing(
    registry: FacetRegistry,
) -> None:
    candidate = preference(id=f"p_{'9' * 5000}_0_0")

    validate_preference(candidate, registry)
    validate_intent_state(
        IntentState(
            goal=None,
            preferences=(candidate,),
            dont_care_facets=frozenset(),
            version=0,
        ),
        registry,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ExtendedPreference(Preference):
    extras: list[str]


@dataclass(frozen=True, slots=True, kw_only=True)
class ExtendedClearFacet(ClearFacet):
    extras: list[str]


def test_domain_subclasses_cannot_bypass_closed_immutable_boundaries(
    registry: FacetRegistry,
) -> None:
    base = preference()
    extended_preference = ExtendedPreference(
        **{field.name: getattr(base, field.name) for field in fields(base)},
        extras=[],
    )
    assert_code(
        ErrorCode.INVALID_REPRESENTATION,
        lambda: validate_preference(extended_preference, registry),
    )

    batch = StateUpdateBatch(
        turn=1,
        base_intent_version=0,
        operations=(ExtendedClearFacet(facet="color", extras=[]),),
    )
    assert_code(
        ErrorCode.NON_CANONICAL_VALUE,
        lambda: validate_state_update_batch(batch, registry),
    )


def test_mutable_operation_collection_is_noncanonical(registry: FacetRegistry) -> None:
    valid = StateUpdateBatch(
        turn=1,
        base_intent_version=0,
        operations=(ClearFacet(facet="color"),),
    )
    invalid = replace(valid, operations=[ClearFacet(facet="color")])
    assert_code(
        ErrorCode.NON_CANONICAL_VALUE,
        lambda: validate_state_update_batch(invalid, registry),
    )
