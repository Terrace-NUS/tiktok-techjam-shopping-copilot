"""Pure ordered reduction for canonical intent updates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

from .errors import ErrorCode, SessionContextError
from .models import IntentState, Operator, Preference
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
from .validation import (
    _logical_preference_key,
    _preference_id_key,
    _preference_id_parts,
    validate_intent_state,
    validate_state_update_batch,
)


@dataclass(frozen=True, slots=True)
class _WorkingIntent:
    goal: str | None
    preferences: tuple[Preference, ...]
    dont_care_facets: frozenset[str]


@dataclass(slots=True)
class _ReductionIdentity:
    by_id: dict[str, Preference]
    by_semantics: dict[object, Preference]


def reduce_intent(
    current: IntentState,
    batch: StateUpdateBatch,
    registry: FacetRegistry,
) -> IntentState:
    """Apply one validated batch atomically and return a canonical snapshot."""

    validate_intent_state(current, registry)
    validate_state_update_batch(batch, registry)
    if batch.base_intent_version != current.version:
        raise SessionContextError(
            code=ErrorCode.STALE_BASE_VERSION,
            path=("base_intent_version",),
            details=(
                ("actual", batch.base_intent_version),
                ("expected", current.version),
            ),
        )

    pre_batch_by_id = {preference.id: preference for preference in current.preferences}
    identity = _ReductionIdentity(
        by_id=dict(pre_batch_by_id),
        by_semantics={
            _logical_preference_key(preference): preference for preference in current.preferences
        },
    )
    working = _WorkingIntent(
        goal=current.goal,
        preferences=current.preferences,
        dont_care_facets=current.dont_care_facets,
    )

    for operation_index, operation in enumerate(batch.operations):
        try:
            working = _apply_operation(
                working,
                operation,
                batch=batch,
                operation_index=operation_index,
                registry=registry,
                identity=identity,
                pre_batch_by_id=pre_batch_by_id,
            )
            working = _canonicalize_preferences(working)
            validate_intent_state(
                _snapshot(working, version=current.version),
                registry,
            )
        except SessionContextError as error:
            if error.operation_index is not None:
                raise
            raise _at_operation(error, operation_index) from error

    candidate = _snapshot(working, version=current.version)
    if candidate == current:
        return current
    result = _snapshot(working, version=current.version + 1)
    validate_intent_state(result, registry)
    return result


def _apply_operation(
    working: _WorkingIntent,
    operation: StateOperation,
    *,
    batch: StateUpdateBatch,
    operation_index: int,
    registry: FacetRegistry,
    identity: _ReductionIdentity,
    pre_batch_by_id: dict[str, Preference],
) -> _WorkingIntent:
    if type(operation) is AddPreference:
        return _add_preference(
            working,
            operation.preference,
            batch=batch,
            operation_index=operation_index,
            registry=registry,
            identity=identity,
        )
    if type(operation) is ReplaceFacet:
        return _replace_facet(
            working,
            operation,
            batch=batch,
            operation_index=operation_index,
            identity=identity,
        )
    if type(operation) is RemovePreference:
        return _remove_preferences(working, operation)
    if type(operation) is ClearFacet:
        return _clear_facet(working, operation.facet)
    if type(operation) is SetDontCare:
        cleared = _clear_facet(working, operation.facet)
        return replace(
            cleared,
            dont_care_facets=cleared.dont_care_facets.union({operation.facet}),
        )
    if type(operation) is SwitchGoal:
        return _switch_goal(working, operation, pre_batch_by_id)
    raise AssertionError("batch validation admitted an unknown operation")


def _add_preference(
    working: _WorkingIntent,
    candidate: Preference,
    *,
    batch: StateUpdateBatch,
    operation_index: int,
    registry: FacetRegistry,
    identity: _ReductionIdentity,
) -> _WorkingIntent:
    active_by_id = {active.id: active for active in working.preferences}
    preference = _resolve_identity(
        candidate,
        batch=batch,
        operation_index=operation_index,
        preference_index=0,
        base_path=("preference",),
        active_by_id=active_by_id,
        identity=identity,
    )
    if preference.id in active_by_id:
        return working

    facet = preference.facet
    dont_care_facets = working.dont_care_facets
    if facet is not None:
        dont_care_facets = dont_care_facets.difference({facet})
        spec = registry.require(facet)
        if spec.kind is FacetKind.NUMERIC:
            existing = _matching_numeric_bound(working.preferences, preference)
            if existing is not None:
                if _is_stronger_numeric_bound(preference, existing):
                    preferences = tuple(
                        preference if active is existing else active
                        for active in working.preferences
                    )
                    return replace(
                        working,
                        preferences=preferences,
                        dont_care_facets=dont_care_facets,
                    )
                return replace(working, dont_care_facets=dont_care_facets)
    return replace(
        working,
        preferences=working.preferences + (preference,),
        dont_care_facets=dont_care_facets,
    )


def _replace_facet(
    working: _WorkingIntent,
    operation: ReplaceFacet,
    *,
    batch: StateUpdateBatch,
    operation_index: int,
    identity: _ReductionIdentity,
) -> _WorkingIntent:
    active_by_id = {active.id: active for active in working.preferences}
    resolved = tuple(
        _resolve_identity(
            preference,
            batch=batch,
            operation_index=operation_index,
            preference_index=preference_index,
            base_path=("preferences", preference_index),
            active_by_id=active_by_id,
            identity=identity,
        )
        for preference_index, preference in enumerate(operation.preferences)
    )
    remaining = tuple(
        preference for preference in working.preferences if preference.facet != operation.facet
    )
    return replace(
        working,
        preferences=remaining + resolved,
        dont_care_facets=working.dont_care_facets.difference({operation.facet}),
    )


def _remove_preferences(
    working: _WorkingIntent,
    operation: RemovePreference,
) -> _WorkingIntent:
    active_ids = {preference.id for preference in working.preferences}
    for index, preference_id in enumerate(operation.preference_ids):
        if preference_id not in active_ids:
            raise SessionContextError(
                code=ErrorCode.UNKNOWN_PREFERENCE_ID,
                path=("preference_ids", index),
                details=(("id", preference_id),),
            )
    removed = set(operation.preference_ids)
    return replace(
        working,
        preferences=tuple(
            preference for preference in working.preferences if preference.id not in removed
        ),
    )


def _clear_facet(working: _WorkingIntent, facet: str) -> _WorkingIntent:
    return replace(
        working,
        preferences=tuple(
            preference for preference in working.preferences if preference.facet != facet
        ),
        dont_care_facets=working.dont_care_facets.difference({facet}),
    )


def _switch_goal(
    working: _WorkingIntent,
    operation: SwitchGoal,
    pre_batch_by_id: dict[str, Preference],
) -> _WorkingIntent:
    carried: list[Preference] = []
    for index, preference_id in enumerate(operation.carry_preference_ids):
        preference = pre_batch_by_id.get(preference_id)
        if preference is None:
            raise SessionContextError(
                code=ErrorCode.INVALID_CARRY_ID,
                path=("carry_preference_ids", index),
                details=(("id", preference_id),),
            )
        carried.append(preference)
    return _WorkingIntent(
        goal=operation.new_goal,
        preferences=tuple(carried),
        dont_care_facets=frozenset(),
    )


def _resolve_identity(
    candidate: Preference,
    *,
    batch: StateUpdateBatch,
    operation_index: int,
    preference_index: int,
    base_path: tuple[str | int, ...],
    active_by_id: dict[str, Preference],
    identity: _ReductionIdentity,
) -> Preference:
    logical_key = _logical_preference_key(candidate)
    existing_id = identity.by_id.get(candidate.id)
    if existing_id is not None:
        if _logical_preference_key(existing_id) != logical_key:
            raise SessionContextError(
                code=ErrorCode.PREFERENCE_ID_CONFLICT,
                path=base_path + ("id",),
                details=(("id", candidate.id),),
            )
        if candidate.id not in active_by_id:
            raise SessionContextError(
                code=ErrorCode.DUPLICATE_PREFERENCE_ID,
                path=base_path + ("id",),
                details=(("id", candidate.id),),
            )
        return existing_id

    existing_semantics = identity.by_semantics.get(logical_key)
    if existing_semantics is not None:
        raise SessionContextError(
            code=ErrorCode.DUPLICATE_PREFERENCE_SEMANTICS,
            path=base_path,
            details=(
                ("existing_id", existing_semantics.id),
                ("id", candidate.id),
            ),
        )

    id_turn, id_operation, id_preference = _preference_id_parts(candidate.id)
    if (
        id_turn != _decimal_digits(batch.turn)
        or id_operation != str(operation_index)
        or id_preference != str(preference_index)
    ):
        raise SessionContextError(
            code=ErrorCode.NON_CANONICAL_VALUE,
            path=base_path + ("id",),
        )
    if candidate.source_turn != batch.turn:
        raise SessionContextError(
            code=ErrorCode.INVALID_SOURCE_TURN,
            path=base_path + ("source_turn",),
            details=(
                ("actual", candidate.source_turn),
                ("expected", batch.turn),
            ),
        )

    identity.by_id[candidate.id] = candidate
    identity.by_semantics[logical_key] = candidate
    return candidate


def _matching_numeric_bound(
    preferences: tuple[Preference, ...],
    candidate: Preference,
) -> Preference | None:
    candidate_direction = _numeric_direction(candidate.operator)
    return next(
        (
            preference
            for preference in preferences
            if preference.facet == candidate.facet
            and preference.commitment is candidate.commitment
            and _numeric_direction(preference.operator) == candidate_direction
        ),
        None,
    )


def _numeric_direction(operator: Operator | None) -> str | None:
    if operator in (Operator.GT, Operator.GE):
        return "lower"
    if operator in (Operator.LT, Operator.LE):
        return "upper"
    return None


def _is_stronger_numeric_bound(candidate: Preference, existing: Preference) -> bool:
    candidate_value = _numeric_value(candidate)
    existing_value = _numeric_value(existing)
    if candidate.operator in (Operator.GT, Operator.GE):
        return candidate_value > existing_value or (
            candidate_value == existing_value
            and candidate.operator is Operator.GT
            and existing.operator is Operator.GE
        )
    return candidate_value < existing_value or (
        candidate_value == existing_value
        and candidate.operator is Operator.LT
        and existing.operator is Operator.LE
    )


def _numeric_value(preference: Preference) -> int | float:
    value = preference.value
    assert type(value) in (int, float)
    return cast(int | float, value)


def _decimal_digits(value: int) -> str:
    """Format an arbitrary-size non-negative int without CPython's digit limit."""

    if value == 0:
        return "0"
    chunks: list[int] = []
    while value:
        value, remainder = divmod(value, 1_000_000_000)
        chunks.append(remainder)
    return str(chunks[-1]) + "".join(f"{chunk:09d}" for chunk in reversed(chunks[:-1]))


def _canonicalize_preferences(working: _WorkingIntent) -> _WorkingIntent:
    return replace(
        working,
        preferences=tuple(
            sorted(working.preferences, key=lambda preference: _preference_id_key(preference.id))
        ),
    )


def _snapshot(working: _WorkingIntent, *, version: int) -> IntentState:
    return IntentState(
        goal=working.goal,
        preferences=tuple(working.preferences),
        dont_care_facets=frozenset(working.dont_care_facets),
        version=version,
    )


def _at_operation(
    error: SessionContextError,
    operation_index: int,
) -> SessionContextError:
    return SessionContextError(
        code=error.code,
        path=("operations", operation_index) + error.path,
        operation_index=operation_index,
        details=error.details,
    )
