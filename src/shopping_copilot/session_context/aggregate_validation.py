"""Replay-aware validation for interaction and session aggregates."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, NoReturn, cast

from .aggregates import (
    InteractionContext,
    ProductFeedback,
    SessionContext,
    SessionState,
    TurnRecord,
)
from .errors import ErrorCode, ErrorPathSegment, SessionContextError
from .models import FeedbackSignal, IntentState, Preference, ScalarValue
from .operations import AddPreference, ReplaceFacet, StateUpdateBatch
from .reducer import reduce_intent
from .registry import FacetRegistry
from .validation import (
    _logical_preference_key,
    validate_intent_state,
    validate_profile_prior,
    validate_search_belief,
    validate_state_update_batch,
)

_ASK_ATTRIBUTES = frozenset(
    {
        "brand",
        "budget",
        "category",
        "color",
        "feature",
        "material",
        "other",
        "size",
        "style",
        "use_case",
    }
)


def validate_turn_record(
    record: TurnRecord,
    prior_interaction: InteractionContext,
    registry: FacetRegistry,
) -> None:
    """Validate one turn using only local fields and earlier shown products."""

    _validate_prior_interaction_shape(prior_interaction)
    if type(record) is not TurnRecord:
        _fail(ErrorCode.INVALID_TURN_RECORD)

    if type(record.turn) is not int or record.turn < 1:
        _fail(ErrorCode.INVALID_TURN_RECORD, path=("turn",))
    expected_turn = len(prior_interaction.turns) + 1
    if record.turn != expected_turn:
        _fail(
            ErrorCode.INVALID_TURN_SEQUENCE,
            path=("turn",),
            details=(("actual", record.turn), ("expected", expected_turn)),
        )

    for field_name in ("user_message", "assistant_message"):
        if type(getattr(record, field_name)) is not str:
            _fail(ErrorCode.INVALID_TURN_RECORD, path=(field_name,))

    for field_name in ("intent_version_before", "intent_version_after"):
        value = getattr(record, field_name)
        if type(value) is not int or value < 0:
            _fail(ErrorCode.INVALID_TURN_RECORD, path=(field_name,))

    update = record.accepted_update
    if update is None:
        if record.intent_version_after != record.intent_version_before:
            _fail(
                ErrorCode.TURN_RECORD_VERSION_MISMATCH,
                path=("intent_version_after",),
            )
    else:
        if type(update) is not StateUpdateBatch:
            _fail(ErrorCode.INVALID_TURN_RECORD, path=("accepted_update",))
        try:
            validate_state_update_batch(update, registry)
        except SessionContextError as error:
            raise _prefixed(error, ("accepted_update",)) from error
        if update.turn != record.turn:
            _fail(
                ErrorCode.INVALID_TURN_RECORD,
                path=("accepted_update", "turn"),
            )
        if update.base_intent_version != record.intent_version_before:
            _fail(
                ErrorCode.TURN_RECORD_VERSION_MISMATCH,
                path=("accepted_update", "base_intent_version"),
            )
        if record.intent_version_after not in (
            record.intent_version_before,
            record.intent_version_before + 1,
        ):
            _fail(
                ErrorCode.TURN_RECORD_VERSION_MISMATCH,
                path=("intent_version_after",),
            )

    _validate_question_fields(record)
    _validate_shown_product_ids(record.shown_product_ids)

    if type(record.feedback) is not tuple:
        _fail(ErrorCode.INVALID_FEEDBACK, path=("feedback",))
    previously_shown = _previously_shown_product_ids(prior_interaction)
    for index, feedback in enumerate(record.feedback):
        try:
            _validate_feedback(feedback, previously_shown)
        except SessionContextError as error:
            raise _prefixed(error, ("feedback", index)) from error

    probe_id = record.search_belief_probe_id
    if probe_id is not None and (type(probe_id) is not str or not probe_id.strip()):
        _fail(ErrorCode.INVALID_TURN_RECORD, path=("search_belief_probe_id",))


def validate_session_context(
    context: SessionContext,
    registry: FacetRegistry,
) -> None:
    """Validate a complete snapshot and replay its accepted intent batches."""

    if type(context) is not SessionContext:
        _fail(ErrorCode.INVALID_SESSION_TRANSITION)
    if (
        type(context.session_id) is not str
        or not context.session_id
        or context.session_id != context.session_id.strip()
    ):
        _fail(ErrorCode.INVALID_SESSION_ID, path=("session_id",))

    if context.profile is not None:
        try:
            validate_profile_prior(context.profile)
        except SessionContextError as error:
            raise _prefixed(error, ("profile",)) from error

    state = context.state
    if type(state) is not SessionState:
        _fail(ErrorCode.INVALID_SESSION_TRANSITION, path=("state",))
    try:
        validate_intent_state(state.intent, registry)
    except SessionContextError as error:
        raise _prefixed(error, ("state", "intent")) from error

    interaction = state.interaction
    if type(interaction) is not InteractionContext or type(interaction.turns) is not tuple:
        _fail(ErrorCode.INVALID_TURN_SEQUENCE, path=("state", "interaction"))

    replayed = _initial_intent()
    prior = InteractionContext(turns=())
    lifetime_by_id: dict[str, Preference] = {}
    for index, record in enumerate(interaction.turns):
        record_path: tuple[ErrorPathSegment, ...] = (
            "state",
            "interaction",
            "turns",
            index,
        )
        try:
            validate_turn_record(record, prior, registry)
        except SessionContextError as error:
            raise _prefixed(error, record_path) from error

        if record.intent_version_before != replayed.version:
            _fail(
                ErrorCode.TURN_RECORD_VERSION_MISMATCH,
                path=record_path + ("intent_version_before",),
            )

        update = record.accepted_update
        if update is not None:
            update_path = record_path + ("accepted_update",)
            try:
                _validate_lifetime_ids(update, replayed, lifetime_by_id, registry)
                replayed = reduce_intent(replayed, update, registry)
            except SessionContextError as error:
                raise _prefixed(error, update_path) from error

        if record.intent_version_after != replayed.version:
            _fail(
                ErrorCode.TURN_RECORD_VERSION_MISMATCH,
                path=record_path + ("intent_version_after",),
            )
        prior = InteractionContext(turns=prior.turns + (record,))

    if not _exact_domain_equal(replayed, state.intent):
        _fail(
            ErrorCode.INVALID_SESSION_TRANSITION,
            path=("state", "intent"),
        )

    belief = state.search_belief
    if belief is not None:
        try:
            validate_search_belief(belief, registry)
        except SessionContextError as error:
            raise _prefixed(error, ("state", "search_belief")) from error
        if belief.based_on_intent_version != state.intent.version:
            _fail(
                ErrorCode.STALE_SEARCH_BELIEF,
                path=("state", "search_belief", "based_on_intent_version"),
                details=(
                    ("actual", belief.based_on_intent_version),
                    ("expected", state.intent.version),
                ),
            )


def validate_session_transition(
    previous: SessionContext,
    next_context: SessionContext,
    expected_turn: int,
    registry: FacetRegistry,
) -> None:
    """Validate one copy-on-write session commit against its captured snapshot."""

    validate_session_context(previous, registry)
    if type(next_context) is not SessionContext:
        _fail(ErrorCode.INVALID_SESSION_TRANSITION)
    if next_context.session_id != previous.session_id:
        _fail(ErrorCode.INVALID_SESSION_TRANSITION, path=("session_id",))
    if not _exact_domain_equal(next_context.profile, previous.profile):
        _fail(ErrorCode.INVALID_SESSION_TRANSITION, path=("profile",))
    if type(next_context.state) is not SessionState:
        _fail(ErrorCode.INVALID_SESSION_TRANSITION, path=("state",))
    if type(next_context.state.interaction) is not InteractionContext:
        _fail(ErrorCode.INVALID_SESSION_TRANSITION, path=("state", "interaction"))

    previous_turns = previous.state.interaction.turns
    next_turns = next_context.state.interaction.turns
    if type(next_turns) is not tuple:
        _fail(ErrorCode.INVALID_SESSION_TRANSITION, path=("state", "interaction", "turns"))
    if len(next_turns) != len(previous_turns) + 1 or not _exact_domain_equal(
        next_turns[:-1], previous_turns
    ):
        _fail(ErrorCode.INVALID_SESSION_TRANSITION, path=("state", "interaction", "turns"))

    if type(expected_turn) is not int or expected_turn < 1:
        _fail(ErrorCode.TURN_OUT_OF_ORDER, path=("turn",))
    contiguous_turn = len(previous_turns) + 1
    appended = next_turns[-1]
    if type(appended) is not TurnRecord:
        _fail(
            ErrorCode.INVALID_SESSION_TRANSITION,
            path=("state", "interaction", "turns", len(previous_turns)),
        )
    if expected_turn != contiguous_turn or appended.turn != expected_turn:
        _fail(
            ErrorCode.TURN_OUT_OF_ORDER,
            path=("state", "interaction", "turns", len(previous_turns), "turn"),
            details=(("actual", appended.turn), ("expected", expected_turn)),
        )

    validate_session_context(next_context, registry)

    record_path: tuple[ErrorPathSegment, ...] = (
        "state",
        "interaction",
        "turns",
        len(previous_turns),
    )
    if appended.intent_version_before != previous.state.intent.version:
        _fail(
            ErrorCode.TURN_RECORD_VERSION_MISMATCH,
            path=record_path + ("intent_version_before",),
        )
    if appended.intent_version_after != next_context.state.intent.version:
        _fail(
            ErrorCode.TURN_RECORD_VERSION_MISMATCH,
            path=record_path + ("intent_version_after",),
        )

    update = appended.accepted_update
    if update is None:
        expected_intent = previous.state.intent
    else:
        try:
            expected_intent = reduce_intent(previous.state.intent, update, registry)
        except SessionContextError as error:
            raise _prefixed(error, record_path + ("accepted_update",)) from error
    if not _exact_domain_equal(expected_intent, next_context.state.intent):
        _fail(ErrorCode.INVALID_SESSION_TRANSITION, path=("state", "intent"))

    previous_belief = previous.state.search_belief
    next_belief = next_context.state.search_belief
    if (
        not _exact_domain_equal(expected_intent, previous.state.intent)
        and previous_belief is not None
        and _exact_domain_equal(next_belief, previous_belief)
    ):
        _fail(ErrorCode.STALE_SEARCH_BELIEF, path=("state", "search_belief"))

    expected_probe_id = (
        next_belief.certainty_evidence.probe_id
        if next_belief is not None and not _exact_domain_equal(next_belief, previous_belief)
        else None
    )
    if appended.search_belief_probe_id != expected_probe_id:
        _fail(
            ErrorCode.INVALID_PROBE_EVIDENCE,
            path=record_path + ("search_belief_probe_id",),
        )


def _validate_prior_interaction_shape(interaction: InteractionContext) -> None:
    if type(interaction) is not InteractionContext or type(interaction.turns) is not tuple:
        _fail(ErrorCode.INVALID_TURN_SEQUENCE)
    for index, record in enumerate(interaction.turns):
        if type(record) is not TurnRecord or type(record.shown_product_ids) is not tuple:
            _fail(ErrorCode.INVALID_TURN_SEQUENCE, path=("turns", index))
        if not all(type(product_id) is str for product_id in record.shown_product_ids):
            _fail(
                ErrorCode.INVALID_TURN_SEQUENCE,
                path=("turns", index, "shown_product_ids"),
            )


def _validate_question_fields(record: TurnRecord) -> None:
    fields = (
        ("question", record.question),
        ("question_key", record.question_key),
        ("ask_attribute", record.ask_attribute),
    )
    present = tuple(value is not None for _, value in fields)
    if any(present) and not all(present):
        _fail(ErrorCode.INVALID_QUESTION_FIELDS)
    if not any(present):
        return
    for field_name, value in fields:
        if type(value) is not str or not value.strip():
            _fail(ErrorCode.INVALID_QUESTION_FIELDS, path=(field_name,))
    if record.ask_attribute not in _ASK_ATTRIBUTES:
        _fail(ErrorCode.INVALID_QUESTION_FIELDS, path=("ask_attribute",))


def _validate_shown_product_ids(product_ids: tuple[str, ...]) -> None:
    if type(product_ids) is not tuple:
        _fail(ErrorCode.INVALID_TURN_RECORD, path=("shown_product_ids",))
    if not all(type(product_id) is str and bool(product_id.strip()) for product_id in product_ids):
        _fail(ErrorCode.INVALID_TURN_RECORD, path=("shown_product_ids",))
    if len(set(product_ids)) != len(product_ids):
        _fail(ErrorCode.INVALID_TURN_RECORD, path=("shown_product_ids",))


def _validate_feedback(feedback: ProductFeedback, previously_shown: set[str]) -> None:
    if type(feedback) is not ProductFeedback:
        _fail(ErrorCode.INVALID_FEEDBACK)
    _validate_feedback_ids(feedback.product_ids, field_name="product_ids", non_empty=True)
    _validate_feedback_ids(
        feedback.compared_to_ids,
        field_name="compared_to_ids",
        non_empty=False,
    )
    if type(feedback.signal) is not FeedbackSignal:
        _fail(ErrorCode.INVALID_FEEDBACK, path=("signal",))
    if type(feedback.evidence_text) is not str or not feedback.evidence_text.strip():
        _fail(ErrorCode.INVALID_FEEDBACK, path=("evidence_text",))

    compared = feedback.compared_to_ids
    if feedback.signal is FeedbackSignal.COMPARATIVE:
        if not compared or set(feedback.product_ids).intersection(compared):
            _fail(ErrorCode.INVALID_FEEDBACK, path=("compared_to_ids",))
    elif compared:
        _fail(ErrorCode.INVALID_FEEDBACK, path=("compared_to_ids",))

    for field_name, product_ids in (
        ("product_ids", feedback.product_ids),
        ("compared_to_ids", feedback.compared_to_ids),
    ):
        for index, product_id in enumerate(product_ids):
            if product_id not in previously_shown:
                _fail(
                    ErrorCode.INVALID_FEEDBACK_REFERENCE,
                    path=(field_name, index),
                    details=(("product_id", product_id),),
                )


def _validate_feedback_ids(
    product_ids: tuple[str, ...],
    *,
    field_name: str,
    non_empty: bool,
) -> None:
    if type(product_ids) is not tuple or (non_empty and not product_ids):
        _fail(ErrorCode.INVALID_FEEDBACK, path=(field_name,))
    if not all(type(product_id) is str and bool(product_id.strip()) for product_id in product_ids):
        _fail(ErrorCode.INVALID_FEEDBACK, path=(field_name,))
    if len(set(product_ids)) != len(product_ids):
        _fail(ErrorCode.INVALID_FEEDBACK, path=(field_name,))


def _previously_shown_product_ids(interaction: InteractionContext) -> set[str]:
    return {product_id for record in interaction.turns for product_id in record.shown_product_ids}


def _validate_lifetime_ids(
    update: StateUpdateBatch,
    current: IntentState,
    lifetime_by_id: dict[str, Preference],
    registry: FacetRegistry,
) -> None:
    active_ids = {preference.id for preference in current.preferences}
    for operation_index, operation in enumerate(update.operations):
        candidates: tuple[tuple[tuple[ErrorPathSegment, ...], Preference], ...]
        if type(operation) is AddPreference:
            candidates = ((("operations", operation_index, "preference"), operation.preference),)
        elif type(operation) is ReplaceFacet:
            candidates = tuple(
                (
                    ("operations", operation_index, "preferences", preference_index),
                    preference,
                )
                for preference_index, preference in enumerate(operation.preferences)
            )
        else:
            candidates = ()

        for base_path, candidate in candidates:
            existing = lifetime_by_id.get(candidate.id)
            if existing is None:
                lifetime_by_id[candidate.id] = candidate
                continue
            same_logical = _logical_preference_key(existing) == _logical_preference_key(candidate)
            if candidate.id in active_ids and same_logical:
                continue
            _fail(
                ErrorCode.DUPLICATE_PREFERENCE_ID
                if same_logical
                else ErrorCode.PREFERENCE_ID_CONFLICT,
                path=base_path + ("id",),
                operation_index=operation_index,
                details=(("id", candidate.id),),
            )

        prefix = StateUpdateBatch(
            turn=update.turn,
            base_intent_version=update.base_intent_version,
            operations=update.operations[: operation_index + 1],
        )
        active_ids = {
            preference.id for preference in reduce_intent(current, prefix, registry).preferences
        }


def _initial_intent() -> IntentState:
    return IntentState(
        goal=None,
        preferences=(),
        dont_care_facets=frozenset(),
        version=0,
    )


def _exact_domain_equal(left: object, right: object) -> bool:
    """Compare frozen domain values without Python's bool/int/float coercion."""

    if type(left) is not type(right):
        return False
    if is_dataclass(left) and not isinstance(left, type):
        dataclass_fields = fields(cast(Any, left))
        return all(
            _exact_domain_equal(getattr(left, field.name), getattr(right, field.name))
            for field in dataclass_fields
        )
    if type(left) is tuple:
        left_items = cast(tuple[object, ...], left)
        right_items = cast(tuple[object, ...], right)
        return len(left_items) == len(right_items) and all(
            _exact_domain_equal(left_item, right_item)
            for left_item, right_item in zip(left_items, right_items, strict=True)
        )
    if type(left) is frozenset:
        unmatched = list(cast(frozenset[object], right))
        for left_item in cast(frozenset[object], left):
            match = next(
                (
                    index
                    for index, right_item in enumerate(unmatched)
                    if _exact_domain_equal(left_item, right_item)
                ),
                None,
            )
            if match is None:
                return False
            unmatched.pop(match)
        return not unmatched
    if type(left) is float:
        return left.hex() == cast(float, right).hex()
    return left == right


def _prefixed(
    error: SessionContextError,
    prefix: tuple[ErrorPathSegment, ...],
) -> SessionContextError:
    return SessionContextError(
        code=error.code,
        path=prefix + error.path,
        operation_index=error.operation_index,
        details=error.details,
    )


def _fail(
    code: ErrorCode,
    *,
    path: tuple[ErrorPathSegment, ...] = (),
    operation_index: int | None = None,
    details: tuple[tuple[str, ScalarValue], ...] = (),
) -> NoReturn:
    raise SessionContextError(
        code=code,
        path=path,
        operation_index=operation_index,
        details=details,
    )
