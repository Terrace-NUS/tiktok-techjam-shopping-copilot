"""Native function-call schema and strict decoder for DeepSeek tool arguments."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from enum import Enum
from typing import NoReturn, TypeAlias, TypedDict, TypeVar, cast

from shopping_copilot.session_context import FeedbackSignal, SemanticPolarity

from .errors import QueryUnderstandingError, QueryUnderstandingErrorCode
from .models import (
    BehavioralDirectives,
    ClarificationNeed,
    DiversityMode,
    FeedbackFrame,
    GoalAction,
    GoalFrame,
    PreferenceBasis,
    PreferenceRelation,
    PreferenceStrength,
    PricePreferenceFrame,
    ReconciledIntentFrame,
    SemanticPreferenceFrame,
    StructuredPreferenceFrame,
    UnderstandingDisposition,
)

TOOL_NAME = "reconcile_session_intent"

JsonObject: TypeAlias = dict[str, object]
EnumT = TypeVar("EnumT", bound=Enum)
DecodedT = TypeVar("DecodedT")


class _PreferenceMetadata(TypedDict):
    strength: PreferenceStrength
    basis: PreferenceBasis
    meaning: str
    evidence: str
    confidence: float


def reconcile_session_intent_tool(*, strict: bool) -> dict[str, object]:
    """Return the one model-facing mutation tool in DeepSeek's native shape."""

    function: dict[str, object] = {
        "name": TOOL_NAME,
        "description": (
            "Return the complete intended state after the latest shopping turn. "
            "Keep every old active_N ref that still applies; omission deletes it."
        ),
        "parameters": _parameters_schema(),
    }
    if strict:
        function["strict"] = True
    return {"type": "function", "function": function}


def decode_reconciled_intent(arguments: str) -> ReconciledIntentFrame:
    """Decode one tool argument string, rejecting ambiguity and coercion."""

    if type(arguments) is not str:
        _fail(QueryUnderstandingErrorCode.INVALID_TOOL_CALL)
    try:
        raw = json.loads(
            arguments,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise QueryUnderstandingError(
            code=QueryUnderstandingErrorCode.INVALID_FRAME,
        ) from error
    root = _require_object(
        raw,
        path=(),
        keys={
            "base_intent_version",
            "disposition",
            "goal",
            "keep_active_refs",
            "new_preferences",
            "dont_care_facets",
            "feedback",
            "directives",
            "clarification",
            "summary",
        },
    )
    structured, price, semantic = _decode_preference_groups(root["new_preferences"])
    try:
        frame = ReconciledIntentFrame(
            base_intent_version=_require_int(
                root["base_intent_version"], path=("base_intent_version",), minimum=0
            ),
            disposition=_enum(
                UnderstandingDisposition,
                root["disposition"],
                path=("disposition",),
            ),
            goal=_decode_goal(root["goal"]),
            keep_active_refs=_string_tuple(
                root["keep_active_refs"], path=("keep_active_refs",), unique=True
            ),
            structured_preferences=structured,
            price_preferences=price,
            semantic_preferences=semantic,
            dont_care_facets=_string_tuple(
                root["dont_care_facets"], path=("dont_care_facets",), unique=True
            ),
            feedback=_object_tuple(root["feedback"], path=("feedback",), decoder=_decode_feedback),
            directives=_decode_directives(root["directives"]),
            clarification=_decode_clarification(root["clarification"]),
            summary=_nonempty_string(root["summary"], path=("summary",)),
        )
    except QueryUnderstandingError:
        raise
    except (TypeError, ValueError) as error:
        raise QueryUnderstandingError(
            code=QueryUnderstandingErrorCode.INVALID_FRAME,
        ) from error
    if (frame.disposition is UnderstandingDisposition.NEEDS_CLARIFICATION) != (
        frame.clarification.needed
    ):
        _fail(QueryUnderstandingErrorCode.INVALID_FRAME, path=("clarification", "needed"))
    return frame


def _decode_preference_groups(
    value: object,
) -> tuple[
    tuple[StructuredPreferenceFrame, ...],
    tuple[PricePreferenceFrame, ...],
    tuple[SemanticPreferenceFrame, ...],
]:
    item = _require_object(
        value,
        path=("new_preferences",),
        keys={"structured", "price", "semantic"},
    )
    return (
        _object_tuple(
            item["structured"],
            path=("new_preferences", "structured"),
            decoder=_decode_structured_preference,
        ),
        _object_tuple(
            item["price"],
            path=("new_preferences", "price"),
            decoder=_decode_price_preference,
        ),
        _object_tuple(
            item["semantic"],
            path=("new_preferences", "semantic"),
            decoder=_decode_semantic_preference,
        ),
    )


def _decode_goal(value: object) -> GoalFrame:
    item = _require_object(value, path=("goal",), keys={"action", "value"})
    action = _enum(GoalAction, item["action"], path=("goal", "action"))
    goal_value = _optional_string(item["value"], path=("goal", "value"))
    try:
        return GoalFrame(action=action, value=goal_value)
    except (TypeError, ValueError) as error:
        raise QueryUnderstandingError(
            code=QueryUnderstandingErrorCode.INVALID_FRAME,
            path=("goal",),
        ) from error


_CATEGORICAL_RELATIONS = (
    PreferenceRelation.EQ,
    PreferenceRelation.NEQ,
    PreferenceRelation.IN,
    PreferenceRelation.NOT_IN,
)
_PRICE_RELATIONS = (
    PreferenceRelation.LT,
    PreferenceRelation.LE,
    PreferenceRelation.GT,
    PreferenceRelation.GE,
)
_STRUCTURED_RELATIONS = (*_CATEGORICAL_RELATIONS, *_PRICE_RELATIONS)


def _decode_structured_preference(
    value: object,
    path: tuple[str | int, ...],
) -> StructuredPreferenceFrame:
    item = _require_object(
        value,
        path=path,
        keys={
            "facet",
            "relation",
            "values",
            "strength",
            "basis",
            "meaning",
            "evidence",
            "confidence",
        },
    )
    relation = _enum(PreferenceRelation, item["relation"], path=path + ("relation",))
    if relation not in _STRUCTURED_RELATIONS:
        _fail(
            QueryUnderstandingErrorCode.INVALID_FRAME,
            path=path + ("relation",),
            details=(("reason", "structured_relation_required"),),
        )
    values = _string_tuple(item["values"], path=path + ("values",), unique=True)
    if not values:
        _fail(
            QueryUnderstandingErrorCode.INVALID_FRAME,
            path=path + ("values",),
            details=(("reason", "structured_values_required"),),
        )
    if relation is PreferenceRelation.EQ and len(values) > 1:
        relation = PreferenceRelation.IN
    elif relation is PreferenceRelation.NEQ and len(values) > 1:
        relation = PreferenceRelation.NOT_IN
    try:
        return StructuredPreferenceFrame(
            facet=_nonempty_string(item["facet"], path=path + ("facet",)),
            relation=relation,
            values=values,
            **_preference_metadata(item, path=path),
        )
    except (TypeError, ValueError) as error:
        raise QueryUnderstandingError(
            code=QueryUnderstandingErrorCode.INVALID_FRAME,
            path=path,
            details=(("reason", "invalid_structured_preference"),),
        ) from error


def _decode_price_preference(
    value: object,
    path: tuple[str | int, ...],
) -> PricePreferenceFrame:
    item = _require_object(
        value,
        path=path,
        keys={
            "relation",
            "value_usd",
            "strength",
            "basis",
            "meaning",
            "evidence",
            "confidence",
        },
    )
    relation = _enum(PreferenceRelation, item["relation"], path=path + ("relation",))
    if relation not in _PRICE_RELATIONS:
        _fail(
            QueryUnderstandingErrorCode.INVALID_FRAME,
            path=path + ("relation",),
            details=(("reason", "price_relation_required"),),
        )
    try:
        return PricePreferenceFrame(
            relation=relation,
            value_usd=_nonempty_string(item["value_usd"], path=path + ("value_usd",)),
            **_preference_metadata(item, path=path),
        )
    except (TypeError, ValueError) as error:
        raise QueryUnderstandingError(
            code=QueryUnderstandingErrorCode.INVALID_FRAME,
            path=path,
            details=(("reason", "invalid_price_preference"),),
        ) from error


def _decode_semantic_preference(
    value: object,
    path: tuple[str | int, ...],
) -> SemanticPreferenceFrame:
    item = _require_object(
        value,
        path=path,
        keys={
            "polarity",
            "strength",
            "basis",
            "meaning",
            "evidence",
            "confidence",
        },
    )
    try:
        return SemanticPreferenceFrame(
            polarity=_enum(SemanticPolarity, item["polarity"], path=path + ("polarity",)),
            **_preference_metadata(item, path=path),
        )
    except (TypeError, ValueError) as error:
        raise QueryUnderstandingError(
            code=QueryUnderstandingErrorCode.INVALID_FRAME,
            path=path,
            details=(("reason", "invalid_semantic_preference"),),
        ) from error


def _preference_metadata(
    item: JsonObject,
    *,
    path: tuple[str | int, ...],
) -> _PreferenceMetadata:
    return {
        "strength": _enum(
            PreferenceStrength,
            item["strength"],
            path=path + ("strength",),
        ),
        "basis": _enum(PreferenceBasis, item["basis"], path=path + ("basis",)),
        "meaning": _nonempty_string(item["meaning"], path=path + ("meaning",)),
        "evidence": _nonempty_string(item["evidence"], path=path + ("evidence",)),
        "confidence": _probability(item["confidence"], path=path + ("confidence",)),
    }


def _decode_feedback(value: object, path: tuple[str | int, ...]) -> FeedbackFrame:
    item = _require_object(
        value,
        path=path,
        keys={"target_refs", "signal", "compared_to_refs", "evidence"},
    )
    targets = _string_tuple(item["target_refs"], path=path + ("target_refs",), unique=True)
    if not targets:
        _fail(QueryUnderstandingErrorCode.INVALID_FRAME, path=path + ("target_refs",))
    return FeedbackFrame(
        target_refs=targets,
        signal=_enum(FeedbackSignal, item["signal"], path=path + ("signal",)),
        compared_to_refs=_string_tuple(
            item["compared_to_refs"], path=path + ("compared_to_refs",), unique=True
        ),
        evidence=_nonempty_string(item["evidence"], path=path + ("evidence",)),
    )


def _decode_directives(value: object) -> BehavioralDirectives:
    path = ("directives",)
    item = _require_object(
        value,
        path=path,
        keys={"diversity", "comparison_requested", "explanation_requested"},
    )
    return BehavioralDirectives(
        diversity=_enum(DiversityMode, item["diversity"], path=path + ("diversity",)),
        comparison_requested=_boolean(
            item["comparison_requested"], path=path + ("comparison_requested",)
        ),
        explanation_requested=_boolean(
            item["explanation_requested"], path=path + ("explanation_requested",)
        ),
    )


def _decode_clarification(value: object) -> ClarificationNeed:
    path = ("clarification",)
    item = _require_object(
        value,
        path=path,
        keys={"needed", "reason", "alternatives"},
    )
    needed = _boolean(item["needed"], path=path + ("needed",))
    reason = _optional_string(item["reason"], path=path + ("reason",))
    alternatives = _string_tuple(item["alternatives"], path=path + ("alternatives",), unique=True)
    if needed and reason is None:
        _fail(QueryUnderstandingErrorCode.INVALID_FRAME, path=path + ("reason",))
    if not needed and (reason is not None or alternatives):
        _fail(QueryUnderstandingErrorCode.INVALID_FRAME, path=path)
    return ClarificationNeed(needed=needed, reason=reason, alternatives=alternatives)


def _parameters_schema() -> dict[str, object]:
    nullable_string = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    string_array = {"type": "array", "items": {"type": "string"}}
    preference_metadata = {
        "strength": {
            "type": "string",
            "enum": [item.value for item in PreferenceStrength],
        },
        "basis": {"type": "string", "enum": [item.value for item in PreferenceBasis]},
        "meaning": {"type": "string"},
        "evidence": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }
    structured_preference = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "facet": {"type": "string"},
            "relation": {
                "type": "string",
                "enum": [item.value for item in _STRUCTURED_RELATIONS],
            },
            "values": string_array,
            **preference_metadata,
        },
        "required": [
            "facet",
            "relation",
            "values",
            "strength",
            "basis",
            "meaning",
            "evidence",
            "confidence",
        ],
    }
    price_preference = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "relation": {
                "type": "string",
                "enum": [item.value for item in _PRICE_RELATIONS],
            },
            "value_usd": {"type": "string"},
            **preference_metadata,
        },
        "required": [
            "relation",
            "value_usd",
            "strength",
            "basis",
            "meaning",
            "evidence",
            "confidence",
        ],
    }
    semantic_preference = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "polarity": {
                "type": "string",
                "enum": [item.value for item in SemanticPolarity],
            },
            **preference_metadata,
        },
        "required": [
            "polarity",
            "strength",
            "basis",
            "meaning",
            "evidence",
            "confidence",
        ],
    }
    feedback = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "target_refs": string_array,
            "signal": {"type": "string", "enum": [item.value for item in FeedbackSignal]},
            "compared_to_refs": string_array,
            "evidence": {"type": "string"},
        },
        "required": ["target_refs", "signal", "compared_to_refs", "evidence"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "base_intent_version": {"type": "integer", "minimum": 0},
            "disposition": {
                "type": "string",
                "enum": [item.value for item in UnderstandingDisposition],
            },
            "goal": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "action": {"type": "string", "enum": [item.value for item in GoalAction]},
                    "value": nullable_string,
                },
                "required": ["action", "value"],
            },
            "keep_active_refs": string_array,
            "new_preferences": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "structured": {
                        "type": "array",
                        "items": structured_preference,
                    },
                    "price": {"type": "array", "items": price_preference},
                    "semantic": {
                        "type": "array",
                        "items": semantic_preference,
                    },
                },
                "required": ["structured", "price", "semantic"],
            },
            "dont_care_facets": string_array,
            "feedback": {"type": "array", "items": feedback},
            "directives": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "diversity": {
                        "type": "string",
                        "enum": [item.value for item in DiversityMode],
                    },
                    "comparison_requested": {"type": "boolean"},
                    "explanation_requested": {"type": "boolean"},
                },
                "required": [
                    "diversity",
                    "comparison_requested",
                    "explanation_requested",
                ],
            },
            "clarification": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "needed": {"type": "boolean"},
                    "reason": nullable_string,
                    "alternatives": string_array,
                },
                "required": ["needed", "reason", "alternatives"],
            },
            "summary": {"type": "string"},
        },
        "required": [
            "base_intent_version",
            "disposition",
            "goal",
            "keep_active_refs",
            "new_preferences",
            "dont_care_facets",
            "feedback",
            "directives",
            "clarification",
            "summary",
        ],
    }


def _unique_object(pairs: list[tuple[str, object]]) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def _require_object(
    value: object,
    *,
    path: tuple[str | int, ...],
    keys: set[str],
) -> JsonObject:
    if type(value) is not dict:
        _fail(
            QueryUnderstandingErrorCode.INVALID_FRAME,
            path=path,
            details=(("reason", "expected_object"),),
        )
    item = cast(JsonObject, value)
    if set(item) != keys:
        missing = ",".join(sorted(keys.difference(item)))
        unexpected = ",".join(sorted(set(item).difference(keys)))
        details: list[tuple[str, str]] = [("reason", "object_shape")]
        if missing:
            details.append(("missing", missing))
        if unexpected:
            details.append(("unexpected", unexpected))
        _fail(
            QueryUnderstandingErrorCode.INVALID_FRAME,
            path=path,
            details=tuple(details),
        )
    return item


def _object_tuple(
    value: object,
    *,
    path: tuple[str | int, ...],
    decoder: Callable[[object, tuple[str | int, ...]], DecodedT],
) -> tuple[DecodedT, ...]:
    if type(value) is not list:
        _fail(QueryUnderstandingErrorCode.INVALID_FRAME, path=path)
    return tuple(decoder(item, path + (index,)) for index, item in enumerate(value))


def _string_tuple(
    value: object,
    *,
    path: tuple[str | int, ...],
    unique: bool,
) -> tuple[str, ...]:
    if type(value) is not list:
        _fail(QueryUnderstandingErrorCode.INVALID_FRAME, path=path)
    result = tuple(_nonempty_string(item, path=path + (index,)) for index, item in enumerate(value))
    if unique and len(set(result)) != len(result):
        _fail(
            QueryUnderstandingErrorCode.INVALID_FRAME,
            path=path,
            details=(("reason", "duplicate_value"),),
        )
    return result


def _nonempty_string(value: object, *, path: tuple[str | int, ...]) -> str:
    if type(value) is not str or not value.strip():
        _fail(QueryUnderstandingErrorCode.INVALID_FRAME, path=path)
    return value


def _optional_string(value: object, *, path: tuple[str | int, ...]) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, path=path)


def _require_int(value: object, *, path: tuple[str | int, ...], minimum: int) -> int:
    if type(value) is not int or value < minimum:
        _fail(QueryUnderstandingErrorCode.INVALID_FRAME, path=path)
    return value


def _number(value: object, *, path: tuple[str | int, ...]) -> float:
    if type(value) not in (int, float):
        _fail(QueryUnderstandingErrorCode.INVALID_FRAME, path=path)
    number = float(cast(int | float, value))
    if not math.isfinite(number):
        _fail(QueryUnderstandingErrorCode.INVALID_FRAME, path=path)
    return number


def _probability(value: object, *, path: tuple[str | int, ...]) -> float:
    number = _number(value, path=path)
    if not 0 <= number <= 1:
        _fail(
            QueryUnderstandingErrorCode.INVALID_FRAME,
            path=path,
            details=(("reason", "probability_out_of_range"),),
        )
    return number


def _boolean(value: object, *, path: tuple[str | int, ...]) -> bool:
    if type(value) is not bool:
        _fail(QueryUnderstandingErrorCode.INVALID_FRAME, path=path)
    return value


def _enum(enum_type: type[EnumT], value: object, *, path: tuple[str | int, ...]) -> EnumT:
    if type(value) is not str:
        _fail(QueryUnderstandingErrorCode.INVALID_FRAME, path=path)
    try:
        return enum_type(value)
    except ValueError as error:
        raise QueryUnderstandingError(
            code=QueryUnderstandingErrorCode.INVALID_FRAME,
            path=path,
        ) from error


def _fail(
    code: QueryUnderstandingErrorCode,
    *,
    path: tuple[str | int, ...] = (),
    details: tuple[tuple[str, str | int | float | bool], ...] = (),
) -> NoReturn:
    raise QueryUnderstandingError(code=code, path=path, details=details)
