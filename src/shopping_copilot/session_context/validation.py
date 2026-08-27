"""State-independent validation for committed session-context values."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import NoReturn, TypeAlias, cast

from .errors import ErrorCode, ErrorPathSegment, SessionContextError
from .models import (
    Commitment,
    IntentState,
    Operator,
    Preference,
    PreferenceDraft,
    PreferenceSource,
    PreferenceValue,
    ProfilePrior,
    ScalarValue,
    SemanticPolarity,
)
from .operations import (
    AddPreference,
    ClearFacet,
    RemovePreference,
    ReplaceFacet,
    SetDontCare,
    StateOperation,
    StateUpdateBatch,
    SwitchGoal,
)
from .registry import FacetKind, FacetRegistry

_PREFERENCE_ID_PATTERN = re.compile(r"^p_([1-9][0-9]*)_(0|[1-9][0-9]*)_(0|[1-9][0-9]*)$")
_POSITIVE_OPERATORS = frozenset({Operator.EQ, Operator.IN})
_NEGATIVE_OPERATORS = frozenset({Operator.NEQ, Operator.NOT_IN})
_LOWER_OPERATORS = frozenset({Operator.GT, Operator.GE})
_UPPER_OPERATORS = frozenset({Operator.LT, Operator.LE})

_NumericIdSegment: TypeAlias = tuple[int, str]
_PreferenceIdKey: TypeAlias = tuple[_NumericIdSegment, _NumericIdSegment, _NumericIdSegment]


def validate_profile_prior(profile: ProfilePrior) -> None:
    """Validate the exact official profile shape without narrowing its values."""

    if type(profile) is not ProfilePrior:
        _fail(ErrorCode.INVALID_PROFILE)
    for field_name in ("purchase_frequency", "rating_style", "summary"):
        if type(getattr(profile, field_name)) is not str:
            _fail(ErrorCode.INVALID_PROFILE, path=(field_name,))
    rating = profile.average_prior_rating
    if rating is not None and (
        type(rating) not in (int, float) or (type(rating) is float and not math.isfinite(rating))
    ):
        _fail(ErrorCode.INVALID_PROFILE, path=("average_prior_rating",))
    if type(profile.preference_tags) is not tuple or not all(
        type(tag) is str for tag in profile.preference_tags
    ):
        _fail(ErrorCode.INVALID_PROFILE, path=("preference_tags",))


def validate_preference(preference: Preference, registry: FacetRegistry) -> None:
    """Validate one committed, grounded, canonical preference."""

    if type(preference) is not Preference:
        _fail(ErrorCode.INVALID_REPRESENTATION)
    _validate_preference_fields(preference)
    _preference_id_key(preference.id, path=("id",))

    if preference.facet is None:
        return

    facet = preference.facet
    operator = preference.operator
    value = preference.value
    if not isinstance(operator, Operator) or value is None:
        _fail(ErrorCode.INVALID_REPRESENTATION)
    spec = registry.require(facet, path=("facet",))
    if operator not in spec.operators:
        _fail(
            ErrorCode.INVALID_OPERATOR_FOR_FACET,
            path=("operator",),
            details=(("facet", facet),),
        )
    _validate_operator_value_shape(operator, value)
    _validate_structured_semantic_polarity(preference)
    normalized = registry.normalize_value(facet, operator, value, path=("value",))
    if not _canonical_equal(value, normalized):
        _fail(ErrorCode.NON_CANONICAL_VALUE, path=("value",))


def validate_intent_state(intent: IntentState, registry: FacetRegistry) -> None:
    """Validate every canonical invariant independent of prior state or history."""

    if type(intent) is not IntentState:
        _fail(ErrorCode.NON_CANONICAL_VALUE)
    if intent.goal is not None and (type(intent.goal) is not str or not intent.goal.strip()):
        _fail(ErrorCode.INVALID_GOAL, path=("goal",))
    if type(intent.version) is not int or intent.version < 0:
        _fail(ErrorCode.NON_CANONICAL_VALUE, path=("version",))
    if type(intent.preferences) is not tuple:
        _fail(ErrorCode.NON_CANONICAL_VALUE, path=("preferences",))

    id_keys: list[_PreferenceIdKey] = []
    by_id: dict[str, Preference] = {}
    by_semantics: dict[object, Preference] = {}
    for index, preference in enumerate(intent.preferences):
        try:
            validate_preference(preference, registry)
        except SessionContextError as error:
            raise _prefixed(error, ("preferences", index)) from error

        id_keys.append(_preference_id_key(preference.id))
        prior_id = by_id.get(preference.id)
        if prior_id is not None:
            code = (
                ErrorCode.DUPLICATE_PREFERENCE_ID
                if _logical_preference_key(preference) == _logical_preference_key(prior_id)
                else ErrorCode.PREFERENCE_ID_CONFLICT
            )
            _fail(code, path=("preferences", index, "id"), details=(("id", preference.id),))
        by_id[preference.id] = preference

        logical_key = _logical_preference_key(preference)
        prior_semantics = by_semantics.get(logical_key)
        if prior_semantics is not None and prior_semantics.id != preference.id:
            _fail(
                ErrorCode.DUPLICATE_PREFERENCE_SEMANTICS,
                path=("preferences", index),
                details=(("id", preference.id),),
            )
        by_semantics[logical_key] = preference

    if id_keys != sorted(id_keys):
        _fail(ErrorCode.NON_CANONICAL_VALUE, path=("preferences",))

    if type(intent.dont_care_facets) is not frozenset or not all(
        type(facet) is str for facet in intent.dont_care_facets
    ):
        _fail(ErrorCode.NON_CANONICAL_VALUE, path=("dont_care_facets",))
    for facet in sorted(intent.dont_care_facets):
        registry.require(facet, path=("dont_care_facets", facet))

    _validate_categorical_state(intent.preferences, registry)
    _validate_numeric_state(intent.preferences, registry)

    active_facets = {
        preference.facet for preference in intent.preferences if preference.facet is not None
    }
    conflicts = sorted(active_facets.intersection(intent.dont_care_facets))
    if conflicts:
        _fail(
            ErrorCode.DONT_CARE_CONFLICT,
            path=("dont_care_facets",),
            details=(("facet", conflicts[0]),),
        )


def validate_state_update_batch(
    batch: StateUpdateBatch,
    registry: FacetRegistry,
) -> None:
    """Validate batch and operation shape without consulting current state."""

    if type(batch) is not StateUpdateBatch:
        _fail(ErrorCode.NON_CANONICAL_VALUE)
    if type(batch.turn) is not int or batch.turn < 1:
        _fail(ErrorCode.TURN_OUT_OF_ORDER, path=("turn",))
    if type(batch.base_intent_version) is not int or batch.base_intent_version < 0:
        _fail(ErrorCode.STALE_BASE_VERSION, path=("base_intent_version",))
    if type(batch.operations) is not tuple:
        _fail(ErrorCode.NON_CANONICAL_VALUE, path=("operations",))
    if not batch.operations:
        _fail(ErrorCode.EMPTY_BATCH, path=("operations",))
    if not all(_is_operation(operation) for operation in batch.operations):
        _fail(ErrorCode.NON_CANONICAL_VALUE, path=("operations",))

    switch_indexes = [
        index for index, operation in enumerate(batch.operations) if type(operation) is SwitchGoal
    ]
    if len(switch_indexes) > 1:
        _fail(ErrorCode.MULTIPLE_GOAL_SWITCH, path=("operations",))
    if switch_indexes and switch_indexes[0] != 0:
        _fail(
            ErrorCode.INVALID_OPERATION_ORDER,
            path=("operations", switch_indexes[0]),
            operation_index=switch_indexes[0],
        )

    for index, operation in enumerate(batch.operations):
        try:
            _validate_operation(operation, registry)
        except SessionContextError as error:
            raise _prefixed(error, ("operations", index), operation_index=index) from error


def _validate_preference_fields(preference: PreferenceDraft) -> None:
    structured_presence = (
        preference.facet is not None,
        preference.operator is not None,
        preference.value is not None,
    )
    if any(structured_presence) and not all(structured_presence):
        _fail(ErrorCode.INVALID_REPRESENTATION)

    semantic_presence = (
        preference.semantic_text is not None,
        preference.semantic_polarity is not None,
    )
    if any(semantic_presence) and not all(semantic_presence):
        _fail(ErrorCode.INVALID_REPRESENTATION)
    if not any(structured_presence) and not any(semantic_presence):
        _fail(ErrorCode.INVALID_REPRESENTATION)

    if preference.facet is not None and (
        type(preference.facet) is not str or not preference.facet.strip()
    ):
        _fail(ErrorCode.INVALID_REPRESENTATION, path=("facet",))
    if preference.operator is not None and not isinstance(preference.operator, Operator):
        _fail(ErrorCode.INVALID_OPERATOR_VALUE, path=("operator",))
    if preference.semantic_text is not None and (
        type(preference.semantic_text) is not str or not preference.semantic_text.strip()
    ):
        _fail(ErrorCode.INVALID_REPRESENTATION, path=("semantic_text",))
    if preference.semantic_polarity is not None and not isinstance(
        preference.semantic_polarity, SemanticPolarity
    ):
        _fail(ErrorCode.INVALID_REPRESENTATION, path=("semantic_polarity",))
    if not isinstance(preference.commitment, Commitment) or not isinstance(
        preference.source, PreferenceSource
    ):
        _fail(ErrorCode.INVALID_REPRESENTATION)
    if type(preference.evidence_text) is not str or not preference.evidence_text.strip():
        _fail(ErrorCode.INVALID_REPRESENTATION, path=("evidence_text",))
    if type(preference.source_turn) is not int or preference.source_turn < 1:
        _fail(ErrorCode.INVALID_SOURCE_TURN, path=("source_turn",))
    confidence = preference.interpretation_confidence
    if type(confidence) not in (int, float):
        _fail(ErrorCode.INVALID_CONFIDENCE, path=("interpretation_confidence",))
    if type(confidence) is float and not math.isfinite(confidence):
        _fail(ErrorCode.INVALID_CONFIDENCE, path=("interpretation_confidence",))
    if not 0 <= confidence <= 1:
        _fail(ErrorCode.INVALID_CONFIDENCE, path=("interpretation_confidence",))
    if (
        preference.source
        in (PreferenceSource.BEHAVIORAL_FEEDBACK, PreferenceSource.SYSTEM_INFERRED)
        and preference.commitment is Commitment.HARD
    ):
        _fail(ErrorCode.INVALID_COMMITMENT_FOR_SOURCE, path=("commitment",))


def _validate_operator_value_shape(
    operator: Operator,
    value: PreferenceValue | None,
) -> None:
    if operator in (Operator.IN, Operator.NOT_IN):
        if type(value) is not tuple or not value:
            _fail(ErrorCode.INVALID_OPERATOR_VALUE, path=("value",))
        if not all(_is_finite_scalar(item) for item in value):
            _fail(ErrorCode.INVALID_OPERATOR_VALUE, path=("value",))
        if len({type(item) for item in value}) != 1:
            _fail(ErrorCode.INVALID_OPERATOR_VALUE, path=("value",))
        return
    if operator in (Operator.LT, Operator.LE, Operator.GT, Operator.GE):
        if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
            _fail(ErrorCode.INVALID_OPERATOR_VALUE, path=("value",))
        return
    if not _is_finite_scalar(value) or type(value) is tuple:
        _fail(ErrorCode.INVALID_OPERATOR_VALUE, path=("value",))


def _validate_structured_semantic_polarity(preference: Preference) -> None:
    operator = preference.operator
    polarity = preference.semantic_polarity
    if operator in _POSITIVE_OPERATORS and polarity is SemanticPolarity.NEGATIVE:
        _fail(ErrorCode.INVALID_REPRESENTATION, path=("semantic_polarity",))
    if operator in _NEGATIVE_OPERATORS and polarity is SemanticPolarity.POSITIVE:
        _fail(ErrorCode.INVALID_REPRESENTATION, path=("semantic_polarity",))


def _validate_categorical_state(
    preferences: tuple[Preference, ...],
    registry: FacetRegistry,
) -> None:
    grouped: dict[tuple[str, Commitment], dict[str, Preference]] = {}
    by_facet: dict[str, list[Preference]] = {}
    for preference in preferences:
        if preference.facet is None:
            continue
        spec = registry.get(preference.facet)
        if spec is None or spec.kind is not FacetKind.CATEGORICAL:
            continue
        operator = preference.operator
        if not isinstance(operator, Operator):
            continue
        selector = "positive" if operator in _POSITIVE_OPERATORS else "negative"
        key = (preference.facet, preference.commitment)
        selectors = grouped.setdefault(key, {})
        if selector in selectors:
            _fail(
                ErrorCode.MULTIPLE_POSITIVE_SELECTOR
                if selector == "positive"
                else ErrorCode.MULTIPLE_NEGATIVE_SELECTOR,
                path=("preferences",),
                details=(
                    ("commitment", preference.commitment.value),
                    ("facet", preference.facet),
                ),
            )
        selectors[selector] = preference
        by_facet.setdefault(preference.facet, []).append(preference)

    for facet in sorted(by_facet):
        positives = [
            _selector_values(preference)
            for preference in by_facet[facet]
            if preference.operator in _POSITIVE_OPERATORS
        ]
        if not positives:
            continue
        common = set.intersection(*positives)
        excluded: set[object] = set()
        for preference in by_facet[facet]:
            if preference.operator in _NEGATIVE_OPERATORS:
                excluded.update(_selector_values(preference))
        if not common.difference(excluded):
            _fail(
                ErrorCode.EMPTY_CATEGORICAL_DOMAIN,
                path=("preferences",),
                details=(("facet", facet),),
            )


def _validate_numeric_state(
    preferences: tuple[Preference, ...],
    registry: FacetRegistry,
) -> None:
    grouped: dict[tuple[str, Commitment], dict[str, Preference]] = {}
    by_facet: dict[str, list[Preference]] = {}
    for preference in preferences:
        if preference.facet is None:
            continue
        spec = registry.get(preference.facet)
        if spec is None or spec.kind is not FacetKind.NUMERIC:
            continue
        operator = preference.operator
        if not isinstance(operator, Operator):
            continue
        direction = "lower" if operator in _LOWER_OPERATORS else "upper"
        key = (preference.facet, preference.commitment)
        bounds = grouped.setdefault(key, {})
        if direction in bounds:
            _fail(
                ErrorCode.NON_CANONICAL_VALUE,
                path=("preferences",),
                details=(
                    ("commitment", preference.commitment.value),
                    ("facet", preference.facet),
                ),
            )
        bounds[direction] = preference
        by_facet.setdefault(preference.facet, []).append(preference)

    for facet in sorted(by_facet):
        lower: Preference | None = None
        upper: Preference | None = None
        for preference in by_facet[facet]:
            if preference.operator in _LOWER_OPERATORS:
                lower = _stronger_lower(lower, preference)
            else:
                upper = _stronger_upper(upper, preference)
        if lower is not None and upper is not None and _interval_is_empty(lower, upper):
            _fail(
                ErrorCode.EMPTY_NUMERIC_INTERSECTION,
                path=("preferences",),
                details=(("facet", facet),),
            )


def _validate_operation(operation: StateOperation, registry: FacetRegistry) -> None:
    expected_op: str
    if type(operation) is AddPreference:
        add = operation
        expected_op = "add_preference"
        _validate_discriminator(add.op, expected_op)
        try:
            validate_preference(add.preference, registry)
        except SessionContextError as error:
            raise _prefixed(error, ("preference",)) from error
        return
    if type(operation) is ReplaceFacet:
        replacement = operation
        expected_op = "replace_facet"
        _validate_discriminator(replacement.op, expected_op)
        if type(replacement.preferences) is not tuple:
            _fail(ErrorCode.NON_CANONICAL_VALUE, path=("preferences",))
        if not replacement.preferences:
            _fail(ErrorCode.EMPTY_REPLACEMENT, path=("preferences",))
        registry.require(replacement.facet, path=("facet",))
        for index, preference in enumerate(replacement.preferences):
            try:
                validate_preference(preference, registry)
            except SessionContextError as error:
                raise _prefixed(error, ("preferences", index)) from error
            if preference.facet != replacement.facet:
                _fail(ErrorCode.FACET_MISMATCH, path=("preferences", index, "facet"))
        _validate_replacement_state(replacement.preferences, registry)
        return
    if type(operation) is RemovePreference:
        removal = operation
        expected_op = "remove_preference"
        _validate_discriminator(removal.op, expected_op)
        _validate_id_tuple(removal.preference_ids, path=("preference_ids",))
        return
    if type(operation) is ClearFacet:
        clear = operation
        expected_op = "clear_facet"
        _validate_discriminator(clear.op, expected_op)
        registry.require(clear.facet, path=("facet",))
        return
    if type(operation) is SetDontCare:
        set_dont_care = operation
        expected_op = "set_dont_care"
        _validate_discriminator(set_dont_care.op, expected_op)
        registry.require(set_dont_care.facet, path=("facet",))
        return
    if type(operation) is SwitchGoal:
        switch = operation
        expected_op = "switch_goal"
        _validate_discriminator(switch.op, expected_op)
        if type(switch.new_goal) is not str or not switch.new_goal.strip():
            _fail(ErrorCode.INVALID_GOAL, path=("new_goal",))
        _validate_id_tuple(
            switch.carry_preference_ids,
            path=("carry_preference_ids",),
        )
        return
    _fail(ErrorCode.NON_CANONICAL_VALUE)


def _validate_replacement_state(
    preferences: tuple[Preference, ...],
    registry: FacetRegistry,
) -> None:
    validate_intent_state(
        IntentState(
            goal=None,
            preferences=preferences,
            dont_care_facets=frozenset(),
            version=0,
        ),
        registry,
    )


def _validate_id_tuple(
    preference_ids: tuple[str, ...],
    *,
    path: tuple[ErrorPathSegment, ...],
) -> None:
    if type(preference_ids) is not tuple:
        _fail(ErrorCode.NON_CANONICAL_VALUE, path=path)
    keys: list[_PreferenceIdKey] = []
    for index, preference_id in enumerate(preference_ids):
        keys.append(_preference_id_key(preference_id, path=path + (index,)))
    if len(set(preference_ids)) != len(preference_ids) or keys != sorted(keys):
        _fail(ErrorCode.NON_CANONICAL_VALUE, path=path)


def _preference_id_key(
    preference_id: object,
    *,
    path: tuple[ErrorPathSegment, ...] = (),
) -> _PreferenceIdKey:
    if type(preference_id) is not str or not preference_id.strip():
        _fail(ErrorCode.INVALID_REPRESENTATION, path=path)
    match = _PREFERENCE_ID_PATTERN.fullmatch(preference_id)
    if match is None:
        _fail(ErrorCode.NON_CANONICAL_VALUE, path=path)
    turn, operation, preference = match.groups()
    return (
        (len(turn), turn),
        (len(operation), operation),
        (len(preference), preference),
    )


def _logical_preference_key(preference: Preference) -> object:
    return (
        preference.facet,
        preference.operator,
        _typed_value(preference.value),
        preference.semantic_text,
        preference.semantic_polarity,
        preference.commitment,
        preference.source,
        _typed_value(preference.interpretation_confidence),
    )


def _typed_value(value: PreferenceValue | None) -> object:
    if type(value) is tuple:
        return tuple(_typed_value(item) for item in value)
    return type(value), value


def _canonical_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is tuple:
        left_tuple = cast(tuple[object, ...], left)
        right_tuple = cast(tuple[object, ...], right)
        return len(left_tuple) == len(right_tuple) and all(
            _canonical_equal(left_item, right_item)
            for left_item, right_item in zip(left_tuple, right_tuple, strict=True)
        )
    return left == right


def _selector_values(preference: Preference) -> set[object]:
    value = preference.value
    values: Iterable[ScalarValue]
    if type(value) is tuple:
        values = value
    else:
        assert value is not None
        values = (cast(ScalarValue, value),)
    return {_typed_value(item) for item in values}


def _stronger_lower(current: Preference | None, candidate: Preference) -> Preference:
    if current is None:
        return candidate
    current_value = _numeric_value(current)
    candidate_value = _numeric_value(candidate)
    if candidate_value > current_value or (
        candidate_value == current_value and candidate.operator is Operator.GT
    ):
        return candidate
    return current


def _stronger_upper(current: Preference | None, candidate: Preference) -> Preference:
    if current is None:
        return candidate
    current_value = _numeric_value(current)
    candidate_value = _numeric_value(candidate)
    if candidate_value < current_value or (
        candidate_value == current_value and candidate.operator is Operator.LT
    ):
        return candidate
    return current


def _interval_is_empty(lower: Preference, upper: Preference) -> bool:
    lower_value = _numeric_value(lower)
    upper_value = _numeric_value(upper)
    if lower_value > upper_value:
        return True
    if lower_value < upper_value:
        return False
    return lower.operator is Operator.GT or upper.operator is Operator.LT


def _numeric_value(preference: Preference) -> int | float:
    value = preference.value
    assert type(value) in (int, float)
    return cast(int | float, value)


def _is_operation(operation: object) -> bool:
    return type(operation) in (
        AddPreference,
        ReplaceFacet,
        RemovePreference,
        ClearFacet,
        SetDontCare,
        SwitchGoal,
    )


def _validate_discriminator(actual: object, expected: str) -> None:
    if actual != expected:
        _fail(ErrorCode.NON_CANONICAL_VALUE, path=("op",))


def _is_finite_scalar(value: object) -> bool:
    return type(value) in (str, int, float, bool) and not (
        type(value) is float and not math.isfinite(value)
    )


def _prefixed(
    error: SessionContextError,
    prefix: tuple[ErrorPathSegment, ...],
    *,
    operation_index: int | None = None,
) -> SessionContextError:
    return SessionContextError(
        code=error.code,
        path=prefix + error.path,
        operation_index=(error.operation_index if operation_index is None else operation_index),
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
