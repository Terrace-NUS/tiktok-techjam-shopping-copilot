"""Contract tests for pure, ordered, atomic intent reduction."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from shopping_copilot.session_context.errors import ErrorCode, SessionContextError
from shopping_copilot.session_context.models import (
    Commitment,
    IntentState,
    Operator,
    Preference,
    PreferenceSource,
    PreferenceValue,
)
from shopping_copilot.session_context.operations import (
    AddPreference,
    ClearFacet,
    RemovePreference,
    ReplaceFacet,
    SetDontCare,
    StateOperation,
    StateUpdateBatch,
    SwitchGoal,
)
from shopping_copilot.session_context.reducer import reduce_intent
from shopping_copilot.session_context.registry import (
    CATEGORICAL_OPERATORS,
    NUMERIC_OPERATORS,
    FacetKind,
    FacetRegistry,
    FacetSpec,
    canonical_number,
    canonical_text,
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


def preference(
    *,
    id: str = "p_1_0_0",
    facet: str | None = "color",
    operator: Operator | None = Operator.EQ,
    value: PreferenceValue | None = "blue",
    commitment: Commitment = Commitment.HARD,
    source_turn: int = 1,
    evidence_text: str | None = None,
    interpretation_confidence: float = 1.0,
) -> Preference:
    return Preference(
        id=id,
        facet=facet,
        operator=operator,
        value=value,
        semantic_text=None,
        semantic_polarity=None,
        commitment=commitment,
        source=PreferenceSource.USER_EXPLICIT,
        source_turn=source_turn,
        evidence_text=evidence_text or f"evidence for {id}",
        interpretation_confidence=interpretation_confidence,
    )


def intent(
    *preferences: Preference,
    goal: str | None = None,
    dont_care: frozenset[str] = frozenset(),
    version: int = 0,
) -> IntentState:
    return IntentState(
        goal=goal,
        preferences=preferences,
        dont_care_facets=dont_care,
        version=version,
    )


def batch(
    *operations: StateOperation,
    turn: int = 1,
    base_version: int = 0,
) -> StateUpdateBatch:
    return StateUpdateBatch(
        turn=turn,
        base_intent_version=base_version,
        operations=operations,
    )


def assert_reducer_error(
    expected: ErrorCode,
    current: IntentState,
    update: StateUpdateBatch,
    registry: FacetRegistry,
    *,
    operation_index: int | None = None,
) -> SessionContextError:
    before = (
        current.goal,
        current.preferences,
        current.dont_care_facets,
        current.version,
    )

    with pytest.raises(SessionContextError) as caught:
        reduce_intent(current, update, registry)

    error = caught.value
    assert error.code is expected
    assert error.operation_index == operation_index
    assert (
        current.goal,
        current.preferences,
        current.dont_care_facets,
        current.version,
    ) == before
    assert current.preferences is before[1]
    assert current.dont_care_facets is before[2]
    return error


def test_add_preference_adds_constraint_and_removes_dont_care(
    registry: FacetRegistry,
) -> None:
    current = intent(dont_care=frozenset({"color"}))
    added = preference()

    result = reduce_intent(
        current,
        batch(AddPreference(preference=added)),
        registry,
    )

    assert result == intent(added, version=1)
    assert result.preferences[0] is added
    assert current == intent(dont_care=frozenset({"color"}))


def test_replace_facet_replaces_only_that_facet_and_removes_dont_care(
    registry: FacetRegistry,
) -> None:
    budget = preference(
        id="p_1_0_0",
        facet="budget",
        operator=Operator.GE,
        value=20,
    )
    current = intent(budget, dont_care=frozenset({"color"}))
    replacement = preference(
        id="p_2_0_0",
        value="red",
        source_turn=2,
    )

    result = reduce_intent(
        current,
        batch(
            ReplaceFacet(facet="color", preferences=(replacement,)),
            turn=2,
        ),
        registry,
    )

    assert result == intent(budget, replacement, version=1)
    assert result.preferences[0] is budget
    assert result.preferences[1] is replacement


def test_remove_preference_removes_only_requested_active_ids(
    registry: FacetRegistry,
) -> None:
    color = preference(id="p_1_0_0")
    budget = preference(
        id="p_1_0_1",
        facet="budget",
        operator=Operator.LE,
        value=100,
    )
    current = intent(color, budget)

    result = reduce_intent(
        current,
        batch(RemovePreference(preference_ids=(color.id,)), turn=2),
        registry,
    )

    assert result == intent(budget, version=1)
    assert result.preferences[0] is budget


def test_clear_facet_returns_only_that_facet_to_unset(
    registry: FacetRegistry,
) -> None:
    color = preference(id="p_1_0_0")
    budget = preference(
        id="p_1_0_1",
        facet="budget",
        operator=Operator.GE,
        value=20,
    )
    current = intent(color, budget)

    result = reduce_intent(
        current,
        batch(ClearFacet(facet="color"), turn=2),
        registry,
    )

    assert result == intent(budget, version=1)


def test_set_dont_care_removes_active_facet_preferences_and_sets_marker(
    registry: FacetRegistry,
) -> None:
    color = preference(id="p_1_0_0")
    budget = preference(
        id="p_1_0_1",
        facet="budget",
        operator=Operator.LE,
        value=100,
    )
    current = intent(color, budget)

    result = reduce_intent(
        current,
        batch(SetDontCare(facet="color"), turn=2),
        registry,
    )

    assert result == intent(budget, dont_care=frozenset({"color"}), version=1)


def test_switch_goal_carries_only_prebatch_preferences_and_accepts_new_constraints(
    registry: FacetRegistry,
) -> None:
    carried = preference(
        id="p_1_0_0",
        evidence_text="original evidence",
        interpretation_confidence=0.8,
    )
    current = intent(
        carried,
        goal="old goal",
        dont_care=frozenset({"budget"}),
    )
    new_budget = preference(
        id="p_2_1_0",
        facet="budget",
        operator=Operator.LE,
        value=80,
        source_turn=2,
    )

    result = reduce_intent(
        current,
        batch(
            SwitchGoal(new_goal="new goal", carry_preference_ids=(carried.id,)),
            AddPreference(preference=new_budget),
            turn=2,
        ),
        registry,
    )

    assert result.goal == "new goal"
    assert result.preferences == (carried, new_budget)
    assert result.preferences[0] is carried
    assert result.preferences[0].evidence_text == "original evidence"
    assert result.preferences[0].source_turn == 1
    assert result.dont_care_facets == frozenset()
    assert result.version == 1


def _no_op_cases() -> tuple[tuple[str, IntentState, StateOperation], ...]:
    active = preference()
    return (
        (
            "add_reassertion",
            intent(active),
            AddPreference(
                preference=preference(
                    source_turn=2,
                    evidence_text="new evidence that must be ignored",
                )
            ),
        ),
        (
            "replace_same_facet",
            intent(active),
            ReplaceFacet(facet="color", preferences=(active,)),
        ),
        ("remove_empty", intent(active), RemovePreference(preference_ids=())),
        (
            "clear_absent_facet",
            intent(
                preference(
                    facet="budget",
                    operator=Operator.LE,
                    value=100,
                )
            ),
            ClearFacet(facet="color"),
        ),
        (
            "set_existing_dont_care",
            intent(dont_care=frozenset({"color"})),
            SetDontCare(facet="color"),
        ),
        (
            "switch_same_empty_goal",
            intent(goal="same goal"),
            SwitchGoal(new_goal="same goal"),
        ),
    )


@pytest.mark.parametrize(
    ("current", "operation"),
    [(current, operation) for _, current, operation in _no_op_cases()],
    ids=[name for name, _, _ in _no_op_cases()],
)
def test_all_six_operations_have_logical_no_op_cases(
    registry: FacetRegistry,
    current: IntentState,
    operation: StateOperation,
) -> None:
    result = reduce_intent(
        current,
        batch(operation, turn=2),
        registry,
    )

    assert result is current
    assert result.version == current.version


def _failure_cases() -> tuple[tuple[str, IntentState, StateUpdateBatch, ErrorCode], ...]:
    active = preference()
    conflicting = preference(id="p_2_0_0", value="red", source_turn=2)
    return (
        (
            "add_conflicting_selector",
            intent(active),
            batch(AddPreference(preference=conflicting), turn=2),
            ErrorCode.MULTIPLE_POSITIVE_SELECTOR,
        ),
        (
            "replace_empty",
            intent(active),
            batch(ReplaceFacet(facet="color", preferences=()), turn=2),
            ErrorCode.EMPTY_REPLACEMENT,
        ),
        (
            "remove_unknown",
            intent(active),
            batch(RemovePreference(preference_ids=("p_9_0_0",)), turn=2),
            ErrorCode.UNKNOWN_PREFERENCE_ID,
        ),
        (
            "clear_unknown_facet",
            intent(active),
            batch(ClearFacet(facet="unknown"), turn=2),
            ErrorCode.UNKNOWN_FACET,
        ),
        (
            "set_unknown_facet",
            intent(active),
            batch(SetDontCare(facet="unknown"), turn=2),
            ErrorCode.UNKNOWN_FACET,
        ),
        (
            "switch_unknown_carry",
            intent(active, goal="old"),
            batch(
                SwitchGoal(new_goal="new", carry_preference_ids=("p_9_0_0",)),
                turn=2,
            ),
            ErrorCode.INVALID_CARRY_ID,
        ),
    )


@pytest.mark.parametrize(
    ("current", "update", "expected"),
    [(current, update, expected) for _, current, update, expected in _failure_cases()],
    ids=[name for name, _, _, _ in _failure_cases()],
)
def test_all_six_operations_have_failure_cases(
    registry: FacetRegistry,
    current: IntentState,
    update: StateUpdateBatch,
    expected: ErrorCode,
) -> None:
    assert_reducer_error(
        expected,
        current,
        update,
        registry,
        operation_index=0,
    )


def test_active_logical_reassertion_reuses_the_exact_old_preference(
    registry: FacetRegistry,
) -> None:
    original = preference(
        evidence_text="first evidence",
        interpretation_confidence=0.75,
    )
    current = intent(original)
    reassertion = preference(
        source_turn=2,
        evidence_text="second evidence",
        interpretation_confidence=0.75,
    )

    result = reduce_intent(
        current,
        batch(AddPreference(preference=reassertion), turn=2),
        registry,
    )

    assert result is current
    assert result.preferences[0] is original
    assert result.preferences[0].evidence_text == "first evidence"
    assert result.preferences[0].source_turn == 1


def test_same_active_id_with_different_payload_is_a_conflict(
    registry: FacetRegistry,
) -> None:
    original = preference()
    current = intent(original)
    conflict = preference(
        id=original.id,
        value="red",
        source_turn=2,
    )

    error = assert_reducer_error(
        ErrorCode.PREFERENCE_ID_CONFLICT,
        current,
        batch(AddPreference(preference=conflict), turn=2),
        registry,
        operation_index=0,
    )
    assert error.path == ("operations", 0, "preference", "id")


def test_removed_id_cannot_be_readded_later_in_the_same_batch(
    registry: FacetRegistry,
) -> None:
    original = preference()
    current = intent(original)
    readd = preference(
        id=original.id,
        source_turn=2,
        evidence_text="attempted recycled ID",
    )

    error = assert_reducer_error(
        ErrorCode.DUPLICATE_PREFERENCE_ID,
        current,
        batch(
            RemovePreference(preference_ids=(original.id,)),
            AddPreference(preference=readd),
            turn=2,
        ),
        registry,
        operation_index=1,
    )
    assert error.path == ("operations", 1, "preference", "id")


def test_new_id_with_existing_semantics_is_rejected(
    registry: FacetRegistry,
) -> None:
    original = preference()
    duplicate = preference(id="p_2_0_0", source_turn=2)

    assert_reducer_error(
        ErrorCode.DUPLICATE_PREFERENCE_SEMANTICS,
        intent(original),
        batch(AddPreference(preference=duplicate), turn=2),
        registry,
        operation_index=0,
    )


def test_new_add_id_uses_exact_turn_and_operation_coordinates(
    registry: FacetRegistry,
) -> None:
    candidate = preference(
        id="p_7_1_0",
        source_turn=7,
    )

    result = reduce_intent(
        intent(),
        batch(
            ClearFacet(facet="color"),
            AddPreference(preference=candidate),
            turn=7,
        ),
        registry,
    )

    assert result.preferences == (candidate,)


def test_replace_ids_use_exact_preference_indexes(
    registry: FacetRegistry,
) -> None:
    positive = preference(id="p_7_0_0", source_turn=7)
    negative = preference(
        id="p_7_0_1",
        operator=Operator.NEQ,
        value="red",
        source_turn=7,
    )

    result = reduce_intent(
        intent(),
        batch(
            ReplaceFacet(facet="color", preferences=(positive, negative)),
            turn=7,
        ),
        registry,
    )

    assert result.preferences == (positive, negative)


@pytest.mark.parametrize(
    "preference_id",
    ["p_3_0_0", "p_2_1_0", "p_2_0_1"],
    ids=["wrong_turn", "wrong_operation", "wrong_preference_index"],
)
def test_new_add_rejects_inexact_id_coordinates(
    registry: FacetRegistry,
    preference_id: str,
) -> None:
    candidate = preference(
        id=preference_id,
        source_turn=2,
    )

    assert_reducer_error(
        ErrorCode.NON_CANONICAL_VALUE,
        intent(),
        batch(AddPreference(preference=candidate), turn=2),
        registry,
        operation_index=0,
    )


def test_replace_rejects_an_inexact_preference_index(
    registry: FacetRegistry,
) -> None:
    positive = preference(id="p_2_0_0", source_turn=2)
    negative = preference(
        id="p_2_0_2",
        operator=Operator.NEQ,
        value="red",
        source_turn=2,
    )

    assert_reducer_error(
        ErrorCode.NON_CANONICAL_VALUE,
        intent(),
        batch(
            ReplaceFacet(facet="color", preferences=(positive, negative)),
            turn=2,
        ),
        registry,
        operation_index=0,
    )


def test_new_preference_source_turn_must_match_batch_turn(
    registry: FacetRegistry,
) -> None:
    candidate = preference(id="p_2_0_0", source_turn=1)

    error = assert_reducer_error(
        ErrorCode.INVALID_SOURCE_TURN,
        intent(),
        batch(AddPreference(preference=candidate), turn=2),
        registry,
        operation_index=0,
    )
    assert error.path == ("operations", 0, "preference", "source_turn")


@pytest.mark.parametrize(
    ("existing_operator", "existing_value", "candidate_operator", "candidate_value", "replace"),
    [
        (Operator.GE, 10, Operator.GE, 20, True),
        (Operator.GE, 20, Operator.GE, 10, False),
        (Operator.GE, 10, Operator.GT, 10, True),
        (Operator.GT, 10, Operator.GE, 10, False),
        (Operator.LE, 20, Operator.LE, 10, True),
        (Operator.LE, 10, Operator.LE, 20, False),
        (Operator.LE, 10, Operator.LT, 10, True),
        (Operator.LT, 10, Operator.LE, 10, False),
    ],
    ids=(
        "stronger_lower_value",
        "weaker_lower_value",
        "strict_lower_tie_wins",
        "inclusive_lower_tie_loses",
        "stronger_upper_value",
        "weaker_upper_value",
        "strict_upper_tie_wins",
        "inclusive_upper_tie_loses",
    ),
)
def test_numeric_add_keeps_only_the_strongest_bound(
    registry: FacetRegistry,
    existing_operator: Operator,
    existing_value: int,
    candidate_operator: Operator,
    candidate_value: int,
    replace: bool,
) -> None:
    existing = preference(
        facet="budget",
        operator=existing_operator,
        value=existing_value,
    )
    candidate = preference(
        id="p_2_0_0",
        facet="budget",
        operator=candidate_operator,
        value=candidate_value,
        source_turn=2,
    )
    current = intent(existing)

    result = reduce_intent(
        current,
        batch(AddPreference(preference=candidate), turn=2),
        registry,
    )

    if replace:
        assert result.preferences == (candidate,)
        assert result.preferences[0] is candidate
        assert result.version == 1
    else:
        assert result is current
        assert result.preferences == (existing,)


def test_equal_inclusive_numeric_endpoints_form_a_valid_singleton(
    registry: FacetRegistry,
) -> None:
    lower = preference(
        id="p_1_0_0",
        facet="budget",
        operator=Operator.GE,
        value=10,
    )
    upper = preference(
        id="p_1_1_0",
        facet="budget",
        operator=Operator.LE,
        value=10,
    )

    result = reduce_intent(
        intent(),
        batch(
            AddPreference(preference=lower),
            AddPreference(preference=upper),
        ),
        registry,
    )

    assert result.preferences == (lower, upper)
    assert result.version == 1


@pytest.mark.parametrize(
    ("lower_operator", "upper_operator"),
    [(Operator.GE, Operator.LT), (Operator.GT, Operator.LE)],
)
def test_equal_endpoint_interval_is_rejected_when_either_bound_is_strict(
    registry: FacetRegistry,
    lower_operator: Operator,
    upper_operator: Operator,
) -> None:
    lower = preference(
        id="p_1_0_0",
        facet="budget",
        operator=lower_operator,
        value=10,
    )
    upper = preference(
        id="p_1_1_0",
        facet="budget",
        operator=upper_operator,
        value=10,
    )

    assert_reducer_error(
        ErrorCode.EMPTY_NUMERIC_INTERSECTION,
        intent(),
        batch(
            AddPreference(preference=lower),
            AddPreference(preference=upper),
        ),
        registry,
        operation_index=1,
    )


def test_categorical_hard_and_soft_selectors_must_keep_a_common_value(
    registry: FacetRegistry,
) -> None:
    hard = preference(value="blue")
    overlapping_soft = preference(
        id="p_2_0_0",
        operator=Operator.IN,
        value=("blue", "red"),
        commitment=Commitment.SOFT,
        source_turn=2,
    )

    result = reduce_intent(
        intent(hard),
        batch(AddPreference(preference=overlapping_soft), turn=2),
        registry,
    )
    assert result.preferences == (hard, overlapping_soft)

    disjoint_soft = preference(
        id="p_2_0_0",
        value="red",
        commitment=Commitment.SOFT,
        source_turn=2,
    )
    assert_reducer_error(
        ErrorCode.EMPTY_CATEGORICAL_DOMAIN,
        intent(hard),
        batch(AddPreference(preference=disjoint_soft), turn=2),
        registry,
        operation_index=0,
    )


def test_categorical_negative_cannot_exclude_the_only_positive_value(
    registry: FacetRegistry,
) -> None:
    positive = preference(value="blue")
    exclusion = preference(
        id="p_2_0_0",
        operator=Operator.NEQ,
        value="blue",
        source_turn=2,
    )

    assert_reducer_error(
        ErrorCode.EMPTY_CATEGORICAL_DOMAIN,
        intent(positive),
        batch(AddPreference(preference=exclusion), turn=2),
        registry,
        operation_index=0,
    )


def test_later_replace_cannot_repair_an_invalid_intermediate_categorical_state(
    registry: FacetRegistry,
) -> None:
    blue = preference(id="p_1_0_0", value="blue")
    red_addition = preference(
        id="p_1_1_0",
        value="red",
    )
    repaired_red = preference(
        id="p_1_2_0",
        value="red",
    )
    current = intent()

    error = assert_reducer_error(
        ErrorCode.MULTIPLE_POSITIVE_SELECTOR,
        current,
        batch(
            AddPreference(preference=blue),
            AddPreference(preference=red_addition),
            ReplaceFacet(facet="color", preferences=(repaired_red,)),
        ),
        registry,
        operation_index=1,
    )
    assert error.path[:2] == ("operations", 1)
    assert current == intent()


def test_later_remove_cannot_repair_an_invalid_intermediate_numeric_interval(
    registry: FacetRegistry,
) -> None:
    lower = preference(
        facet="budget",
        operator=Operator.GE,
        value=10,
    )
    current = intent(lower)
    conflicting_upper = preference(
        id="p_2_0_0",
        facet="budget",
        operator=Operator.LT,
        value=10,
        source_turn=2,
    )

    assert_reducer_error(
        ErrorCode.EMPTY_NUMERIC_INTERSECTION,
        current,
        batch(
            AddPreference(preference=conflicting_upper),
            RemovePreference(preference_ids=(lower.id,)),
            turn=2,
        ),
        registry,
        operation_index=0,
    )


def test_valid_first_operation_is_rolled_back_when_second_operation_fails(
    registry: FacetRegistry,
) -> None:
    color = preference(id="p_1_0_0")
    budget = preference(
        id="p_1_1_0",
        facet="budget",
        operator=Operator.LE,
        value=0,
    )
    current = intent()
    original_preferences = current.preferences

    assert_reducer_error(
        ErrorCode.EMPTY_NUMERIC_INTERSECTION,
        current,
        batch(
            AddPreference(preference=color),
            AddPreference(preference=budget),
            AddPreference(
                preference=preference(
                    id="p_1_2_0",
                    facet="budget",
                    operator=Operator.GE,
                    value=1,
                )
            ),
        ),
        registry,
        operation_index=2,
    )
    assert current.preferences is original_preferences
    assert current.preferences == ()


def test_switch_cannot_carry_an_id_that_will_only_be_added_later(
    registry: FacetRegistry,
) -> None:
    future = preference(
        id="p_2_1_0",
        facet="budget",
        operator=Operator.LE,
        value=100,
        source_turn=2,
    )

    assert_reducer_error(
        ErrorCode.INVALID_CARRY_ID,
        intent(goal="old goal"),
        batch(
            SwitchGoal(new_goal="new goal", carry_preference_ids=(future.id,)),
            AddPreference(preference=future),
            turn=2,
        ),
        registry,
        operation_index=0,
    )


@pytest.mark.parametrize(
    ("operations", "expected", "operation_index"),
    [
        (
            (ClearFacet(facet="color"), SwitchGoal(new_goal="new")),
            ErrorCode.INVALID_OPERATION_ORDER,
            1,
        ),
        (
            (SwitchGoal(new_goal="first"), SwitchGoal(new_goal="second")),
            ErrorCode.MULTIPLE_GOAL_SWITCH,
            None,
        ),
    ],
)
def test_switch_goal_ordering_is_rejected_before_reduction(
    registry: FacetRegistry,
    operations: tuple[StateOperation, ...],
    expected: ErrorCode,
    operation_index: int | None,
) -> None:
    assert_reducer_error(
        expected,
        intent(goal="old"),
        batch(*operations, turn=2),
        registry,
        operation_index=operation_index,
    )


def test_stale_base_version_is_rejected_before_any_operation(
    registry: FacetRegistry,
) -> None:
    current = intent(goal="old", version=7)

    error = assert_reducer_error(
        ErrorCode.STALE_BASE_VERSION,
        current,
        batch(SwitchGoal(new_goal="new"), turn=8, base_version=6),
        registry,
    )

    assert error.path == ("base_intent_version",)
    assert error.details == (("actual", 6), ("expected", 7))


def test_multi_operation_batch_increments_version_exactly_once(
    registry: FacetRegistry,
) -> None:
    color = preference(id="p_5_1_0", source_turn=5)
    budget = preference(
        id="p_5_2_0",
        facet="budget",
        operator=Operator.GE,
        value=10,
        source_turn=5,
    )
    current = intent(goal="old", version=11)

    result = reduce_intent(
        current,
        batch(
            SwitchGoal(new_goal="new"),
            AddPreference(preference=color),
            AddPreference(preference=budget),
            turn=5,
            base_version=11,
        ),
        registry,
    )

    assert result.version == 12
    assert result.goal == "new"
    assert result.preferences == (color, budget)


def test_batch_with_transient_changes_but_same_final_state_is_a_no_op(
    registry: FacetRegistry,
) -> None:
    current = intent()

    result = reduce_intent(
        current,
        batch(
            SetDontCare(facet="color"),
            ClearFacet(facet="color"),
        ),
        registry,
    )

    assert result is current
    assert result.version == 0


def test_success_returns_new_state_without_mutating_or_copying_current_values(
    registry: FacetRegistry,
) -> None:
    original = preference()
    current = intent(original, goal="shopping", version=4)
    before_preferences = current.preferences
    added = preference(
        id="p_3_0_0",
        facet="budget",
        operator=Operator.LE,
        value=100,
        source_turn=3,
    )

    result = reduce_intent(
        current,
        batch(
            AddPreference(preference=added),
            turn=3,
            base_version=4,
        ),
        registry,
    )

    assert result is not current
    assert result.preferences[0] is original
    assert current.preferences is before_preferences
    assert current == intent(original, goal="shopping", version=4)


def test_reduction_is_deterministic_for_the_same_inputs(
    registry: FacetRegistry,
) -> None:
    current = intent()
    candidate = preference()
    update = batch(AddPreference(preference=candidate))

    first = reduce_intent(current, update, registry)
    second = reduce_intent(current, update, registry)

    assert first == second
    assert first.preferences[0] is candidate
    assert second.preferences[0] is candidate


def test_rejected_batch_returns_the_same_error_deterministically(
    registry: FacetRegistry,
) -> None:
    current = intent(preference())
    update = batch(
        AddPreference(
            preference=preference(
                id="p_2_0_0",
                value="red",
                source_turn=2,
            )
        ),
        turn=2,
    )

    def reduce_once() -> SessionContextError:
        with pytest.raises(SessionContextError) as caught:
            reduce_intent(current, update, registry)
        return caught.value

    first = reduce_once()
    second = reduce_once()
    assert (first.code, first.path, first.operation_index, first.details) == (
        second.code,
        second.path,
        second.operation_index,
        second.details,
    )


def test_arbitrarily_large_turn_does_not_use_cpython_int_string_conversion(
    registry: FacetRegistry,
) -> None:
    digit_count = 5_000
    huge_turn = 10 ** (digit_count - 1)
    huge_turn_digits = "1" + "0" * (digit_count - 1)
    candidate = preference(
        id=f"p_{huge_turn_digits}_0_0",
        source_turn=huge_turn,
    )

    result = reduce_intent(
        intent(),
        batch(
            AddPreference(preference=candidate),
            turn=huge_turn,
        ),
        registry,
    )

    assert result.preferences[0] is candidate
    assert result.version == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda current: current.preferences,
        lambda current: current.dont_care_facets,
    ],
    ids=("preferences", "dont_care_facets"),
)
def test_failed_reduction_keeps_current_collection_objects(
    registry: FacetRegistry,
    mutate: Callable[[IntentState], object],
) -> None:
    current = intent(preference())
    before = mutate(current)
    conflicting = preference(
        id="p_2_0_0",
        value="red",
        source_turn=2,
    )

    assert_reducer_error(
        ErrorCode.MULTIPLE_POSITIVE_SELECTOR,
        current,
        batch(AddPreference(preference=conflicting), turn=2),
        registry,
        operation_index=0,
    )
    assert mutate(current) is before
