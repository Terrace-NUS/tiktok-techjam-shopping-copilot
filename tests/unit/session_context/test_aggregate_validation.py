"""Contract tests for replay-aware aggregate and transition validation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import cast

import pytest

from shopping_copilot.session_context.aggregate_validation import (
    validate_session_context,
    validate_session_transition,
    validate_turn_record,
)
from shopping_copilot.session_context.aggregates import (
    InteractionContext,
    ProductFeedback,
    SessionContext,
    SessionState,
    TurnRecord,
)
from shopping_copilot.session_context.errors import ErrorCode, SessionContextError
from shopping_copilot.session_context.models import (
    CandidateMode,
    CertaintyEvidence,
    Commitment,
    FeedbackSignal,
    IntentState,
    Operator,
    Preference,
    PreferenceSource,
    ProbeQuality,
    ProfilePrior,
    SearchBelief,
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


def _preference(
    *,
    preference_id: str = "p_1_0_0",
    value: str = "blue",
    source_turn: int = 1,
    evidence_text: str = "blue please",
) -> Preference:
    return Preference(
        id=preference_id,
        facet="color",
        operator=Operator.EQ,
        value=value,
        semantic_text=None,
        semantic_polarity=None,
        commitment=Commitment.HARD,
        source=PreferenceSource.USER_EXPLICIT,
        source_turn=source_turn,
        evidence_text=evidence_text,
        interpretation_confidence=1.0,
    )


def _intent(
    *preferences: Preference,
    goal: str | None = None,
    version: int = 0,
) -> IntentState:
    return IntentState(
        goal=goal,
        preferences=preferences,
        dont_care_facets=frozenset(),
        version=version,
    )


def _batch(
    *operations: StateOperation,
    turn: int = 1,
    base_version: int = 0,
) -> StateUpdateBatch:
    return StateUpdateBatch(
        turn=turn,
        base_intent_version=base_version,
        operations=operations,
    )


def _record(
    *,
    turn: int = 1,
    before: int = 0,
    update: StateUpdateBatch | None = None,
    after: int = 0,
    user_message: str = "",
    assistant_message: str = "",
    question: str | None = None,
    question_key: str | None = None,
    ask_attribute: str | None = None,
    shown: tuple[str, ...] = (),
    feedback: tuple[ProductFeedback, ...] = (),
    probe_id: str | None = None,
) -> TurnRecord:
    return TurnRecord(
        turn=turn,
        user_message=user_message,
        intent_version_before=before,
        accepted_update=update,
        intent_version_after=after,
        assistant_message=assistant_message,
        question=question,
        question_key=question_key,
        ask_attribute=ask_attribute,
        shown_product_ids=shown,
        feedback=feedback,
        search_belief_probe_id=probe_id,
    )


def _belief(*, version: int, probe_id: str = "probe-1") -> SearchBelief:
    return SearchBelief(
        based_on_intent_version=version,
        certainty=0.8,
        certainty_method="bods_v1",
        certainty_evidence=CertaintyEvidence(
            probe_id=probe_id,
            probe_size=1,
            raw_concentration=0.8,
            quality_status=ProbeQuality.VALID,
            quality_reasons=(),
        ),
        candidate_modes=(
            CandidateMode(
                id="mode-1",
                label="primary",
                mass=1.0,
                representative_ids=("sku-1",),
            ),
        ),
        facet_stats=(),
    )


def _profile(*, summary: str = "profile") -> ProfilePrior:
    return ProfilePrior(
        purchase_frequency="monthly",
        average_prior_rating=None,
        rating_style="balanced",
        preference_tags=(),
        summary=summary,
    )


def _context(
    *,
    intent: IntentState | None = None,
    turns: tuple[TurnRecord, ...] = (),
    belief: SearchBelief | None = None,
    session_id: str = "session-1",
    profile: ProfilePrior | None = None,
) -> SessionContext:
    return SessionContext(
        session_id=session_id,
        profile=profile,
        state=SessionState(
            intent=_intent() if intent is None else intent,
            interaction=InteractionContext(turns=turns),
            search_belief=belief,
        ),
    )


def _assert_code(expected: ErrorCode, call: Callable[[], None]) -> SessionContextError:
    with pytest.raises(SessionContextError) as caught:
        call()
    assert caught.value.code is expected
    return caught.value


def _one_add_history(
    registry: FacetRegistry,
) -> tuple[Preference, StateUpdateBatch, TurnRecord, IntentState]:
    preference = _preference()
    update = _batch(AddPreference(preference=preference))
    result = reduce_intent(_intent(), update, registry)
    record = _record(update=update, after=result.version)
    return preference, update, record, result


@pytest.mark.parametrize(
    "ask_attribute",
    [
        "category",
        "material",
        "color",
        "size",
        "style",
        "brand",
        "budget",
        "feature",
        "use_case",
        "other",
    ],
)
def test_turn_record_accepts_empty_messages_and_all_official_question_attributes(
    registry: FacetRegistry,
    ask_attribute: str,
) -> None:
    validate_turn_record(
        _record(
            question="Which one?",
            question_key="next_question",
            ask_attribute=ask_attribute,
        ),
        InteractionContext(turns=()),
        registry,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"question": "question"},
        {"question": " ", "question_key": "key", "ask_attribute": "color"},
        {"question": "question", "question_key": " ", "ask_attribute": "color"},
        {"question": "question", "question_key": "key", "ask_attribute": "price"},
    ],
)
def test_question_triple_is_all_or_none_nonblank_and_protocol_bounded(
    registry: FacetRegistry,
    changes: dict[str, object],
) -> None:
    invalid = replace(_record(), **changes)
    _assert_code(
        ErrorCode.INVALID_QUESTION_FIELDS,
        lambda: validate_turn_record(invalid, InteractionContext(turns=()), registry),
    )


def test_feedback_can_only_reference_products_shown_on_strictly_earlier_turns(
    registry: FacetRegistry,
) -> None:
    prior = _record(shown=("sku-a", "sku-b"))
    interaction = InteractionContext(turns=(prior,))
    valid = ProductFeedback(
        product_ids=("sku-a",),
        signal=FeedbackSignal.COMPARATIVE,
        compared_to_ids=("sku-b",),
        evidence_text="A is better than B.",
    )
    validate_turn_record(
        _record(turn=2, shown=("sku-c",), feedback=(valid,)),
        interaction,
        registry,
    )

    current_turn_reference = replace(valid, product_ids=("sku-c",))
    error = _assert_code(
        ErrorCode.INVALID_FEEDBACK_REFERENCE,
        lambda: validate_turn_record(
            _record(turn=2, shown=("sku-c",), feedback=(current_turn_reference,)),
            interaction,
            registry,
        ),
    )
    assert error.path == ("feedback", 0, "product_ids", 0)


@pytest.mark.parametrize(
    "feedback",
    [
        ProductFeedback(
            product_ids=(),
            signal=FeedbackSignal.POSITIVE,
            compared_to_ids=(),
            evidence_text="good",
        ),
        ProductFeedback(
            product_ids=("sku-a", "sku-a"),
            signal=FeedbackSignal.POSITIVE,
            compared_to_ids=(),
            evidence_text="good",
        ),
        ProductFeedback(
            product_ids=("sku-a",),
            signal=FeedbackSignal.COMPARATIVE,
            compared_to_ids=(),
            evidence_text="better",
        ),
        ProductFeedback(
            product_ids=("sku-a",),
            signal=FeedbackSignal.COMPARATIVE,
            compared_to_ids=("sku-a",),
            evidence_text="better",
        ),
        ProductFeedback(
            product_ids=("sku-a",),
            signal=FeedbackSignal.POSITIVE,
            compared_to_ids=("sku-b",),
            evidence_text="good",
        ),
        ProductFeedback(
            product_ids=("sku-a",),
            signal=FeedbackSignal.POSITIVE,
            compared_to_ids=(),
            evidence_text=" ",
        ),
    ],
)
def test_feedback_shape_and_nonblank_evidence_are_enforced(
    registry: FacetRegistry,
    feedback: ProductFeedback,
) -> None:
    prior = InteractionContext(turns=(_record(shown=("sku-a", "sku-b")),))
    _assert_code(
        ErrorCode.INVALID_FEEDBACK,
        lambda: validate_turn_record(_record(turn=2, feedback=(feedback,)), prior, registry),
    )


def test_shown_products_are_nonblank_unique_and_order_preserving(
    registry: FacetRegistry,
) -> None:
    validate_turn_record(
        _record(shown=("sku-b", "sku-a")),
        InteractionContext(turns=()),
        registry,
    )
    for shown in (("",), ("sku-a", "sku-a")):
        _assert_code(
            ErrorCode.INVALID_TURN_RECORD,
            lambda shown=shown: validate_turn_record(
                _record(shown=shown),
                InteractionContext(turns=()),
                registry,
            ),
        )


def test_turn_sequence_and_version_metadata_are_local_invariants(
    registry: FacetRegistry,
) -> None:
    prior = InteractionContext(turns=(_record(),))
    _assert_code(
        ErrorCode.INVALID_TURN_SEQUENCE,
        lambda: validate_turn_record(_record(turn=3), prior, registry),
    )
    _assert_code(
        ErrorCode.TURN_RECORD_VERSION_MISMATCH,
        lambda: validate_turn_record(
            _record(turn=2, before=0, after=1),
            prior,
            registry,
        ),
    )

    wrong_turn_update = _batch(ClearFacet(facet="color"), turn=3)
    _assert_code(
        ErrorCode.INVALID_TURN_RECORD,
        lambda: validate_turn_record(
            _record(turn=2, update=wrong_turn_update),
            prior,
            registry,
        ),
    )


def test_probe_reference_is_locally_shape_checked_only(registry: FacetRegistry) -> None:
    validate_turn_record(
        _record(probe_id="historical-probe"),
        InteractionContext(turns=()),
        registry,
    )
    _assert_code(
        ErrorCode.INVALID_TURN_RECORD,
        lambda: validate_turn_record(
            _record(probe_id=" "),
            InteractionContext(turns=()),
            registry,
        ),
    )


def test_empty_initial_session_context_is_valid(registry: FacetRegistry) -> None:
    validate_session_context(_context(profile=_profile()), registry)


@pytest.mark.parametrize("session_id", ["", " ", " session", "session "])
def test_session_id_is_nonblank_and_has_no_outer_whitespace(
    registry: FacetRegistry,
    session_id: str,
) -> None:
    _assert_code(
        ErrorCode.INVALID_SESSION_ID,
        lambda: validate_session_context(_context(session_id=session_id), registry),
    )


def test_complete_context_replays_real_and_logical_noop_batches(
    registry: FacetRegistry,
) -> None:
    original, first_update, first_record, after_first = _one_add_history(registry)
    reassertion = replace(original, source_turn=2, evidence_text="still blue")
    second_update = _batch(
        AddPreference(preference=reassertion),
        turn=2,
        base_version=after_first.version,
    )
    after_second = reduce_intent(after_first, second_update, registry)
    second_record = _record(
        turn=2,
        before=after_first.version,
        update=second_update,
        after=after_second.version,
    )
    context = _context(
        intent=after_second,
        turns=(first_record, second_record),
    )

    validate_session_context(context, registry)
    assert context.state.intent.preferences[0] is original
    assert first_update.turn == 1


def test_context_rejects_turn_version_and_final_replay_divergence(
    registry: FacetRegistry,
) -> None:
    _, update, record, result = _one_add_history(registry)
    wrong_version_record = replace(record, intent_version_after=0)
    _assert_code(
        ErrorCode.TURN_RECORD_VERSION_MISMATCH,
        lambda: validate_session_context(
            _context(intent=result, turns=(wrong_version_record,)),
            registry,
        ),
    )

    valid_version_record = replace(record, intent_version_after=1, accepted_update=update)
    _assert_code(
        ErrorCode.INVALID_SESSION_TRANSITION,
        lambda: validate_session_context(
            _context(intent=_intent(), turns=(valid_version_record,)),
            registry,
        ),
    )


def test_context_replay_comparison_is_numeric_type_sensitive(
    registry: FacetRegistry,
) -> None:
    _, _, record, replayed = _one_add_history(registry)
    rewritten = replace(
        replayed.preferences[0],
        interpretation_confidence=1,
    )
    tampered = replace(replayed, preferences=(rewritten,))

    error = _assert_code(
        ErrorCode.INVALID_SESSION_TRANSITION,
        lambda: validate_session_context(_context(intent=tampered, turns=(record,)), registry),
    )
    assert error.path == ("state", "intent")


def test_active_belief_is_valid_and_attached_to_the_active_intent_version(
    registry: FacetRegistry,
) -> None:
    validate_session_context(_context(belief=_belief(version=0)), registry)
    error = _assert_code(
        ErrorCode.STALE_SEARCH_BELIEF,
        lambda: validate_session_context(_context(belief=_belief(version=1)), registry),
    )
    assert error.path == ("state", "search_belief", "based_on_intent_version")


@pytest.mark.parametrize(
    ("replacement_value", "expected"),
    [
        ("blue", ErrorCode.DUPLICATE_PREFERENCE_ID),
        ("red", ErrorCode.PREFERENCE_ID_CONFLICT),
    ],
)
def test_context_rejects_recycling_an_inactive_historical_id(
    registry: FacetRegistry,
    replacement_value: str,
    expected: ErrorCode,
) -> None:
    original, first_update, first_record, after_first = _one_add_history(registry)
    remove = _batch(
        RemovePreference(preference_ids=(original.id,)),
        turn=2,
        base_version=after_first.version,
    )
    after_remove = reduce_intent(after_first, remove, registry)
    remove_record = _record(turn=2, before=1, update=remove, after=2)
    recycled = _preference(
        preference_id=original.id,
        value=replacement_value,
        source_turn=3,
        evidence_text="recycled",
    )
    recycle = _batch(
        AddPreference(preference=recycled),
        turn=3,
        base_version=after_remove.version,
    )
    recycle_record = _record(turn=3, before=2, update=recycle, after=3)
    context = _context(
        intent=after_remove,
        turns=(first_record, remove_record, recycle_record),
    )

    error = _assert_code(expected, lambda: validate_session_context(context, registry))
    assert error.path == (
        "state",
        "interaction",
        "turns",
        2,
        "accepted_update",
        "operations",
        0,
        "preference",
        "id",
    )
    assert error.operation_index == 0
    assert first_update.turn == 1


def test_removed_semantics_may_return_under_a_genuinely_new_id(
    registry: FacetRegistry,
) -> None:
    original, _, first_record, after_first = _one_add_history(registry)
    remove = _batch(
        RemovePreference(preference_ids=(original.id,)),
        turn=2,
        base_version=1,
    )
    after_remove = reduce_intent(after_first, remove, registry)
    remove_record = _record(turn=2, before=1, update=remove, after=2)
    returning = _preference(preference_id="p_3_0_0", source_turn=3)
    add_again = _batch(AddPreference(preference=returning), turn=3, base_version=2)
    final = reduce_intent(after_remove, add_again, registry)
    add_record = _record(turn=3, before=2, update=add_again, after=3)

    validate_session_context(
        _context(intent=final, turns=(first_record, remove_record, add_record)),
        registry,
    )


@pytest.mark.parametrize(
    "deactivation",
    [
        RemovePreference(preference_ids=("p_1_0_0",)),
        ClearFacet(facet="color"),
        SetDontCare(facet="color"),
        ReplaceFacet(
            facet="color",
            preferences=(
                _preference(
                    preference_id="p_2_0_0",
                    value="red",
                    source_turn=2,
                ),
            ),
        ),
        SwitchGoal(new_goal="new goal"),
    ],
    ids=("remove", "clear", "dont_care", "replace", "switch_without_carry"),
)
def test_context_rejects_same_batch_reuse_after_id_becomes_inactive(
    registry: FacetRegistry,
    deactivation: StateOperation,
) -> None:
    original, _, first_record, after_first = _one_add_history(registry)
    reassertion = _preference(source_turn=2, evidence_text="later evidence")
    invalid_update = _batch(
        deactivation,
        AddPreference(preference=reassertion),
        turn=2,
        base_version=1,
    )
    invalid_record = _record(
        turn=2,
        before=1,
        update=invalid_update,
        after=after_first.version,
    )

    error = _assert_code(
        ErrorCode.DUPLICATE_PREFERENCE_ID,
        lambda: validate_session_context(
            _context(intent=after_first, turns=(first_record, invalid_record)),
            registry,
        ),
    )
    assert error.path[-4:] == ("operations", 1, "preference", "id")
    assert error.operation_index == 1
    assert original.id == reassertion.id


def test_context_allows_same_batch_reassertion_of_a_carried_active_id(
    registry: FacetRegistry,
) -> None:
    original, _, first_record, after_first = _one_add_history(registry)
    reassertion = _preference(source_turn=2, evidence_text="later evidence")
    update = _batch(
        SwitchGoal(new_goal="new goal", carry_preference_ids=(original.id,)),
        AddPreference(preference=reassertion),
        turn=2,
        base_version=1,
    )
    final = reduce_intent(after_first, update, registry)
    second_record = _record(turn=2, before=1, update=update, after=2)

    validate_session_context(
        _context(intent=final, turns=(first_record, second_record)),
        registry,
    )
    assert final.preferences == (original,)


def test_transition_accepts_reducer_result_and_binds_a_new_probe(
    registry: FacetRegistry,
) -> None:
    previous = _context()
    preference = _preference()
    update = _batch(AddPreference(preference=preference))
    next_intent = reduce_intent(previous.state.intent, update, registry)
    belief = _belief(version=next_intent.version, probe_id="probe-new")
    appended = _record(
        update=update,
        after=next_intent.version,
        probe_id="probe-new",
    )
    next_context = _context(intent=next_intent, turns=(appended,), belief=belief)

    validate_session_transition(previous, next_context, 1, registry)


def test_transition_allows_preserve_refresh_or_clear_belief_on_a_noop(
    registry: FacetRegistry,
) -> None:
    previous_belief = _belief(version=0, probe_id="probe-old")
    previous = _context(belief=previous_belief)
    update = _batch(ClearFacet(facet="color"))
    intent = reduce_intent(previous.state.intent, update, registry)

    preserved_record = _record(update=update)
    validate_session_transition(
        previous,
        _context(intent=intent, turns=(preserved_record,), belief=previous_belief),
        1,
        registry,
    )

    refreshed = _belief(version=0, probe_id="probe-new")
    refreshed_record = replace(preserved_record, search_belief_probe_id="probe-new")
    validate_session_transition(
        previous,
        _context(intent=intent, turns=(refreshed_record,), belief=refreshed),
        1,
        registry,
    )

    validate_session_transition(
        previous,
        _context(intent=intent, turns=(preserved_record,), belief=None),
        1,
        registry,
    )


def test_transition_rejects_probe_reference_not_matching_belief_change(
    registry: FacetRegistry,
) -> None:
    previous = _context(belief=_belief(version=0, probe_id="probe-old"))
    update = _batch(ClearFacet(facet="color"))
    refreshed = _belief(version=0, probe_id="probe-new")
    missing_reference = _context(
        turns=(_record(update=update),),
        belief=refreshed,
    )
    error = _assert_code(
        ErrorCode.INVALID_PROBE_EVIDENCE,
        lambda: validate_session_transition(previous, missing_reference, 1, registry),
    )
    assert error.path[-1] == "search_belief_probe_id"

    stale_reference = _context(
        turns=(_record(update=update, probe_id="probe-old"),),
        belief=previous.state.search_belief,
    )
    _assert_code(
        ErrorCode.INVALID_PROBE_EVIDENCE,
        lambda: validate_session_transition(previous, stale_reference, 1, registry),
    )


def test_transition_treats_numeric_type_change_as_a_new_belief(
    registry: FacetRegistry,
) -> None:
    previous_belief = replace(_belief(version=0), certainty=1)
    refreshed = replace(previous_belief, certainty=1.0)
    previous = _context(belief=previous_belief)
    missing_reference = _context(turns=(_record(),), belief=refreshed)

    _assert_code(
        ErrorCode.INVALID_PROBE_EVIDENCE,
        lambda: validate_session_transition(previous, missing_reference, 1, registry),
    )


@pytest.mark.parametrize("attack", ["session_id", "profile", "history"])
def test_transition_rejects_identity_profile_and_history_prefix_attacks(
    registry: FacetRegistry,
    attack: str,
) -> None:
    previous_turn = _record(shown=("sku-old",))
    previous = _context(turns=(previous_turn,))
    appended = _record(turn=2)
    next_context = _context(turns=(previous_turn, appended))
    if attack == "session_id":
        next_context = replace(next_context, session_id="session-2")
    elif attack == "profile":
        next_context = replace(next_context, profile=_profile(summary="changed"))
    else:
        rewritten = replace(previous_turn, assistant_message="rewritten")
        next_context = _context(turns=(rewritten, appended))

    _assert_code(
        ErrorCode.INVALID_SESSION_TRANSITION,
        lambda: validate_session_transition(previous, next_context, 2, registry),
    )


def test_transition_profile_and_history_prefix_comparisons_are_type_sensitive(
    registry: FacetRegistry,
) -> None:
    previous_profile = replace(_profile(), average_prior_rating=1)
    rewritten_profile = replace(previous_profile, average_prior_rating=1.0)
    profile_error = _assert_code(
        ErrorCode.INVALID_SESSION_TRANSITION,
        lambda: validate_session_transition(
            _context(profile=previous_profile),
            _context(turns=(_record(),), profile=rewritten_profile),
            1,
            registry,
        ),
    )
    assert profile_error.path == ("profile",)

    signed_zero_profile = replace(_profile(), average_prior_rating=-0.0)
    rewritten_zero_profile = replace(signed_zero_profile, average_prior_rating=0.0)
    signed_zero_error = _assert_code(
        ErrorCode.INVALID_SESSION_TRANSITION,
        lambda: validate_session_transition(
            _context(profile=signed_zero_profile),
            _context(turns=(_record(),), profile=rewritten_zero_profile),
            1,
            registry,
        ),
    )
    assert signed_zero_error.path == ("profile",)

    _, original_update, original_record, original_intent = _one_add_history(registry)
    previous = _context(intent=original_intent, turns=(original_record,))
    rewritten_preference = replace(
        original_intent.preferences[0],
        interpretation_confidence=1,
    )
    rewritten_update = replace(
        original_update,
        operations=(AddPreference(preference=rewritten_preference),),
    )
    rewritten_intent = reduce_intent(_intent(), rewritten_update, registry)
    rewritten_record = replace(original_record, accepted_update=rewritten_update)
    appended = _record(turn=2, before=1, after=1)
    attacked = _context(
        intent=rewritten_intent,
        turns=(rewritten_record, appended),
    )

    history_error = _assert_code(
        ErrorCode.INVALID_SESSION_TRANSITION,
        lambda: validate_session_transition(previous, attacked, 2, registry),
    )
    assert history_error.path == ("state", "interaction", "turns")


def test_transition_rejects_wrong_expected_turn_and_direct_intent_substitution(
    registry: FacetRegistry,
) -> None:
    previous = _context()
    appended = _record()
    next_context = _context(turns=(appended,))
    _assert_code(
        ErrorCode.TURN_OUT_OF_ORDER,
        lambda: validate_session_transition(previous, next_context, 2, registry),
    )

    substituted = _context(intent=_intent(goal="injected", version=1), turns=(appended,))
    _assert_code(
        ErrorCode.INVALID_SESSION_TRANSITION,
        lambda: validate_session_transition(previous, substituted, 1, registry),
    )


def test_exact_aggregate_collection_boundaries_are_enforced(
    registry: FacetRegistry,
) -> None:
    mutable_turns = replace(
        InteractionContext(turns=()),
        turns=cast(tuple[TurnRecord, ...], []),
    )
    invalid = replace(_context().state, interaction=mutable_turns)
    context = replace(_context(), state=invalid)
    _assert_code(
        ErrorCode.INVALID_TURN_SEQUENCE,
        lambda: validate_session_context(context, registry),
    )
