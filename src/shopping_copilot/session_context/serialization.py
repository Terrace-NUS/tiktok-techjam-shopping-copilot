"""Deterministic, versioned JSON snapshots for session context."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Final, NoReturn, TypeVar, cast

from .aggregates import (
    InteractionContext,
    ProductFeedback,
    SessionContext,
    SessionState,
    TurnRecord,
)
from .errors import ErrorCode, ErrorPathSegment, SessionContextError
from .models import (
    CandidateMode,
    CertaintyEvidence,
    Commitment,
    FacetStats,
    FeedbackSignal,
    IntentState,
    Operator,
    Preference,
    PreferenceSource,
    PreferenceValue,
    ProbeQuality,
    ProfilePrior,
    ScalarValue,
    SearchBelief,
    SemanticPolarity,
    ValueMass,
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
from .registry import FacetRegistry

SCHEMA_ID: Final = "shopping-copilot/session-context/v1"

_EnumT = TypeVar("_EnumT", bound=Enum)
_LoadT = TypeVar("_LoadT")


@dataclass(frozen=True, slots=True)
class _ObjectPairs:
    """Intermediate JSON object retaining duplicate keys and their location."""

    values: tuple[tuple[str, object], ...]


def encode_snapshot(context: SessionContext, registry: FacetRegistry) -> bytes:
    """Validate and encode one snapshot as canonical compact UTF-8 JSON."""

    try:
        _validate_complete_context(context, registry)
        envelope: dict[str, object] = {
            "schema": SCHEMA_ID,
            "payload": _encode_session_context(context),
        }
        text = json.dumps(
            envelope,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return text.encode("utf-8")
    except SessionContextError:
        raise
    except (TypeError, ValueError, OverflowError, UnicodeEncodeError, RecursionError):
        _invalid_snapshot()


def decode_snapshot(data: bytes, registry: FacetRegistry) -> SessionContext:
    """Decode, reconstruct, validate, and replay one untrusted snapshot."""

    if type(data) is not bytes:
        _invalid_snapshot()
    try:
        source = data.decode("utf-8")
        parsed: object = json.loads(source, object_pairs_hook=_collect_object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, OverflowError, RecursionError):
        _invalid_snapshot()
    try:
        materialized = _materialize_json(parsed, path=())
    except (ValueError, OverflowError, RecursionError):
        # JSON lexical/materialization failures have no trusted DTO field path.
        _invalid_snapshot()

    envelope = _load_object(materialized, ("schema", "payload"), path=())
    schema = _load_string(envelope["schema"], path=("schema",))
    if schema != SCHEMA_ID:
        raise SessionContextError(
            code=ErrorCode.UNKNOWN_SCHEMA_VERSION,
            path=("schema",),
        )

    context = _load_session_context(envelope["payload"], path=("payload",))
    try:
        _validate_complete_context(context, registry)
    except SessionContextError as error:
        raise _prefixed(error, ("payload",)) from error
    return context


def _validate_complete_context(context: SessionContext, registry: FacetRegistry) -> None:
    # Local import keeps the codec below the reducer-aware aggregate boundary.
    from .aggregate_validation import validate_session_context

    validate_session_context(context, registry)


def _encode_session_context(context: SessionContext) -> dict[str, object]:
    return {
        "session_id": context.session_id,
        "profile": None if context.profile is None else _encode_profile(context.profile),
        "state": _encode_session_state(context.state),
    }


def _encode_profile(profile: ProfilePrior) -> dict[str, object]:
    return {
        "purchase_frequency": profile.purchase_frequency,
        "average_prior_rating": profile.average_prior_rating,
        "rating_style": profile.rating_style,
        "preference_tags": list(profile.preference_tags),
        "summary": profile.summary,
    }


def _encode_session_state(state: SessionState) -> dict[str, object]:
    return {
        "intent": _encode_intent(state.intent),
        "interaction": _encode_interaction(state.interaction),
        "search_belief": (
            None if state.search_belief is None else _encode_search_belief(state.search_belief)
        ),
    }


def _encode_intent(intent: IntentState) -> dict[str, object]:
    return {
        "goal": intent.goal,
        "preferences": [_encode_preference(preference) for preference in intent.preferences],
        "dont_care_facets": sorted(intent.dont_care_facets),
        "version": intent.version,
    }


def _encode_preference(preference: Preference) -> dict[str, object]:
    return {
        "id": preference.id,
        "facet": preference.facet,
        "operator": None if preference.operator is None else preference.operator.value,
        "value": _encode_preference_value(preference.value),
        "semantic_text": preference.semantic_text,
        "semantic_polarity": (
            None if preference.semantic_polarity is None else preference.semantic_polarity.value
        ),
        "commitment": preference.commitment.value,
        "source": preference.source.value,
        "source_turn": preference.source_turn,
        "evidence_text": preference.evidence_text,
        "interpretation_confidence": preference.interpretation_confidence,
    }


def _encode_preference_value(value: PreferenceValue | None) -> object:
    if type(value) is tuple:
        return list(value)
    return value


def _encode_interaction(interaction: InteractionContext) -> dict[str, object]:
    return {"turns": [_encode_turn_record(record) for record in interaction.turns]}


def _encode_turn_record(record: TurnRecord) -> dict[str, object]:
    return {
        "turn": record.turn,
        "user_message": record.user_message,
        "intent_version_before": record.intent_version_before,
        "accepted_update": (
            None
            if record.accepted_update is None
            else _encode_state_update_batch(record.accepted_update)
        ),
        "intent_version_after": record.intent_version_after,
        "assistant_message": record.assistant_message,
        "question": record.question,
        "question_key": record.question_key,
        "ask_attribute": record.ask_attribute,
        "shown_product_ids": list(record.shown_product_ids),
        "feedback": [_encode_product_feedback(item) for item in record.feedback],
        "search_belief_probe_id": record.search_belief_probe_id,
    }


def _encode_product_feedback(feedback: ProductFeedback) -> dict[str, object]:
    return {
        "product_ids": list(feedback.product_ids),
        "signal": feedback.signal.value,
        "compared_to_ids": list(feedback.compared_to_ids),
        "evidence_text": feedback.evidence_text,
    }


def _encode_state_update_batch(batch: StateUpdateBatch) -> dict[str, object]:
    return {
        "turn": batch.turn,
        "base_intent_version": batch.base_intent_version,
        "operations": [_encode_operation(operation) for operation in batch.operations],
    }


def _encode_operation(operation: StateOperation) -> dict[str, object]:
    if type(operation) is AddPreference:
        return {
            "op": operation.op,
            "preference": _encode_preference(operation.preference),
        }
    if type(operation) is ReplaceFacet:
        return {
            "op": operation.op,
            "facet": operation.facet,
            "preferences": [_encode_preference(preference) for preference in operation.preferences],
        }
    if type(operation) is RemovePreference:
        return {
            "op": operation.op,
            "preference_ids": list(operation.preference_ids),
        }
    if type(operation) is ClearFacet:
        return {"op": operation.op, "facet": operation.facet}
    if type(operation) is SetDontCare:
        return {"op": operation.op, "facet": operation.facet}
    if type(operation) is SwitchGoal:
        return {
            "op": operation.op,
            "new_goal": operation.new_goal,
            "carry_preference_ids": list(operation.carry_preference_ids),
        }
    _invalid_snapshot()


def _encode_search_belief(belief: SearchBelief) -> dict[str, object]:
    return {
        "based_on_intent_version": belief.based_on_intent_version,
        "certainty": belief.certainty,
        "certainty_method": belief.certainty_method,
        "certainty_evidence": _encode_certainty_evidence(belief.certainty_evidence),
        "candidate_modes": [_encode_candidate_mode(mode) for mode in belief.candidate_modes],
        "facet_stats": [_encode_facet_stats(stats) for stats in belief.facet_stats],
    }


def _encode_certainty_evidence(evidence: CertaintyEvidence) -> dict[str, object]:
    return {
        "probe_id": evidence.probe_id,
        "probe_size": evidence.probe_size,
        "raw_concentration": evidence.raw_concentration,
        "quality_status": evidence.quality_status.value,
        "quality_reasons": list(evidence.quality_reasons),
    }


def _encode_candidate_mode(mode: CandidateMode) -> dict[str, object]:
    return {
        "id": mode.id,
        "label": mode.label,
        "mass": mode.mass,
        "representative_ids": list(mode.representative_ids),
    }


def _encode_facet_stats(stats: FacetStats) -> dict[str, object]:
    return {
        "facet": stats.facet,
        "entropy": stats.entropy,
        "coverage": stats.coverage,
        "top_values": [_encode_value_mass(value_mass) for value_mass in stats.top_values],
    }


def _encode_value_mass(value_mass: ValueMass) -> dict[str, object]:
    return {"value": value_mass.value, "mass": value_mass.mass}


def _load_session_context(value: object, *, path: tuple[ErrorPathSegment, ...]) -> SessionContext:
    fields = _load_object(value, ("session_id", "profile", "state"), path=path)
    profile_value = fields["profile"]
    return SessionContext(
        session_id=_load_string(fields["session_id"], path=path + ("session_id",)),
        profile=(
            None
            if profile_value is None
            else _load_profile(profile_value, path=path + ("profile",))
        ),
        state=_load_session_state(fields["state"], path=path + ("state",)),
    )


def _load_profile(value: object, *, path: tuple[ErrorPathSegment, ...]) -> ProfilePrior:
    names = (
        "purchase_frequency",
        "average_prior_rating",
        "rating_style",
        "preference_tags",
        "summary",
    )
    fields = _load_object(value, names, path=path)
    rating_value = fields["average_prior_rating"]
    return ProfilePrior(
        purchase_frequency=_load_string(
            fields["purchase_frequency"], path=path + ("purchase_frequency",)
        ),
        average_prior_rating=(
            None
            if rating_value is None
            else _load_number(rating_value, path=path + ("average_prior_rating",))
        ),
        rating_style=_load_string(fields["rating_style"], path=path + ("rating_style",)),
        preference_tags=_load_string_tuple(
            fields["preference_tags"], path=path + ("preference_tags",)
        ),
        summary=_load_string(fields["summary"], path=path + ("summary",)),
    )


def _load_session_state(value: object, *, path: tuple[ErrorPathSegment, ...]) -> SessionState:
    fields = _load_object(value, ("intent", "interaction", "search_belief"), path=path)
    belief_value = fields["search_belief"]
    return SessionState(
        intent=_load_intent(fields["intent"], path=path + ("intent",)),
        interaction=_load_interaction(fields["interaction"], path=path + ("interaction",)),
        search_belief=(
            None
            if belief_value is None
            else _load_search_belief(belief_value, path=path + ("search_belief",))
        ),
    )


def _load_intent(value: object, *, path: tuple[ErrorPathSegment, ...]) -> IntentState:
    fields = _load_object(
        value,
        ("goal", "preferences", "dont_care_facets", "version"),
        path=path,
    )
    return IntentState(
        goal=_load_optional_string(fields["goal"], path=path + ("goal",)),
        preferences=_load_array(
            fields["preferences"],
            path=path + ("preferences",),
            item_loader=_load_preference,
        ),
        dont_care_facets=_load_canonical_string_set(
            fields["dont_care_facets"], path=path + ("dont_care_facets",)
        ),
        version=_load_integer(fields["version"], path=path + ("version",)),
    )


def _load_preference(value: object, *, path: tuple[ErrorPathSegment, ...]) -> Preference:
    names = (
        "id",
        "facet",
        "operator",
        "value",
        "semantic_text",
        "semantic_polarity",
        "commitment",
        "source",
        "source_turn",
        "evidence_text",
        "interpretation_confidence",
    )
    fields = _load_object(value, names, path=path)
    operator_value = fields["operator"]
    polarity_value = fields["semantic_polarity"]
    return Preference(
        id=_load_string(fields["id"], path=path + ("id",)),
        facet=_load_optional_string(fields["facet"], path=path + ("facet",)),
        operator=(
            None
            if operator_value is None
            else _load_enum(operator_value, Operator, path=path + ("operator",))
        ),
        value=_load_preference_value(fields["value"], path=path + ("value",)),
        semantic_text=_load_optional_string(
            fields["semantic_text"], path=path + ("semantic_text",)
        ),
        semantic_polarity=(
            None
            if polarity_value is None
            else _load_enum(
                polarity_value,
                SemanticPolarity,
                path=path + ("semantic_polarity",),
            )
        ),
        commitment=_load_enum(fields["commitment"], Commitment, path=path + ("commitment",)),
        source=_load_enum(fields["source"], PreferenceSource, path=path + ("source",)),
        source_turn=_load_integer(fields["source_turn"], path=path + ("source_turn",)),
        evidence_text=_load_string(fields["evidence_text"], path=path + ("evidence_text",)),
        interpretation_confidence=_load_number(
            fields["interpretation_confidence"],
            path=path + ("interpretation_confidence",),
        ),
    )


def _load_preference_value(
    value: object,
    *,
    path: tuple[ErrorPathSegment, ...],
) -> PreferenceValue | None:
    if value is None:
        return None
    if type(value) is list:
        items = cast(list[object], value)
        return tuple(_load_scalar(item, path=path + (index,)) for index, item in enumerate(items))
    return _load_scalar(value, path=path)


def _load_interaction(value: object, *, path: tuple[ErrorPathSegment, ...]) -> InteractionContext:
    fields = _load_object(value, ("turns",), path=path)
    return InteractionContext(
        turns=_load_array(fields["turns"], path=path + ("turns",), item_loader=_load_turn_record)
    )


def _load_turn_record(value: object, *, path: tuple[ErrorPathSegment, ...]) -> TurnRecord:
    names = (
        "turn",
        "user_message",
        "intent_version_before",
        "accepted_update",
        "intent_version_after",
        "assistant_message",
        "question",
        "question_key",
        "ask_attribute",
        "shown_product_ids",
        "feedback",
        "search_belief_probe_id",
    )
    fields = _load_object(value, names, path=path)
    update_value = fields["accepted_update"]
    return TurnRecord(
        turn=_load_integer(fields["turn"], path=path + ("turn",)),
        user_message=_load_string(fields["user_message"], path=path + ("user_message",)),
        intent_version_before=_load_integer(
            fields["intent_version_before"], path=path + ("intent_version_before",)
        ),
        accepted_update=(
            None
            if update_value is None
            else _load_state_update_batch(update_value, path=path + ("accepted_update",))
        ),
        intent_version_after=_load_integer(
            fields["intent_version_after"], path=path + ("intent_version_after",)
        ),
        assistant_message=_load_string(
            fields["assistant_message"], path=path + ("assistant_message",)
        ),
        question=_load_optional_string(fields["question"], path=path + ("question",)),
        question_key=_load_optional_string(fields["question_key"], path=path + ("question_key",)),
        ask_attribute=_load_optional_string(
            fields["ask_attribute"], path=path + ("ask_attribute",)
        ),
        shown_product_ids=_load_string_tuple(
            fields["shown_product_ids"], path=path + ("shown_product_ids",)
        ),
        feedback=_load_array(
            fields["feedback"], path=path + ("feedback",), item_loader=_load_product_feedback
        ),
        search_belief_probe_id=_load_optional_string(
            fields["search_belief_probe_id"], path=path + ("search_belief_probe_id",)
        ),
    )


def _load_product_feedback(
    value: object,
    *,
    path: tuple[ErrorPathSegment, ...],
) -> ProductFeedback:
    names = ("product_ids", "signal", "compared_to_ids", "evidence_text")
    fields = _load_object(value, names, path=path)
    return ProductFeedback(
        product_ids=_load_string_tuple(fields["product_ids"], path=path + ("product_ids",)),
        signal=_load_enum(fields["signal"], FeedbackSignal, path=path + ("signal",)),
        compared_to_ids=_load_string_tuple(
            fields["compared_to_ids"], path=path + ("compared_to_ids",)
        ),
        evidence_text=_load_string(fields["evidence_text"], path=path + ("evidence_text",)),
    )


def _load_state_update_batch(
    value: object,
    *,
    path: tuple[ErrorPathSegment, ...],
) -> StateUpdateBatch:
    fields = _load_object(value, ("turn", "base_intent_version", "operations"), path=path)
    return StateUpdateBatch(
        turn=_load_integer(fields["turn"], path=path + ("turn",)),
        base_intent_version=_load_integer(
            fields["base_intent_version"], path=path + ("base_intent_version",)
        ),
        operations=_load_array(
            fields["operations"], path=path + ("operations",), item_loader=_load_operation
        ),
    )


def _load_operation(value: object, *, path: tuple[ErrorPathSegment, ...]) -> StateOperation:
    raw_fields = _load_unchecked_object(value, path=path)
    if "op" not in raw_fields:
        _invalid_snapshot(path + ("op",))
    discriminator = _load_string(raw_fields["op"], path=path + ("op",))

    if discriminator == "add_preference":
        fields = _load_object(value, ("op", "preference"), path=path)
        return AddPreference(
            preference=_load_preference(fields["preference"], path=path + ("preference",))
        )
    if discriminator == "replace_facet":
        fields = _load_object(value, ("op", "facet", "preferences"), path=path)
        return ReplaceFacet(
            facet=_load_string(fields["facet"], path=path + ("facet",)),
            preferences=_load_array(
                fields["preferences"],
                path=path + ("preferences",),
                item_loader=_load_preference,
            ),
        )
    if discriminator == "remove_preference":
        fields = _load_object(value, ("op", "preference_ids"), path=path)
        return RemovePreference(
            preference_ids=_load_string_tuple(
                fields["preference_ids"], path=path + ("preference_ids",)
            )
        )
    if discriminator == "clear_facet":
        fields = _load_object(value, ("op", "facet"), path=path)
        return ClearFacet(facet=_load_string(fields["facet"], path=path + ("facet",)))
    if discriminator == "set_dont_care":
        fields = _load_object(value, ("op", "facet"), path=path)
        return SetDontCare(facet=_load_string(fields["facet"], path=path + ("facet",)))
    if discriminator == "switch_goal":
        fields = _load_object(value, ("op", "new_goal", "carry_preference_ids"), path=path)
        return SwitchGoal(
            new_goal=_load_string(fields["new_goal"], path=path + ("new_goal",)),
            carry_preference_ids=_load_string_tuple(
                fields["carry_preference_ids"], path=path + ("carry_preference_ids",)
            ),
        )
    _invalid_snapshot(path + ("op",))


def _load_search_belief(value: object, *, path: tuple[ErrorPathSegment, ...]) -> SearchBelief:
    names = (
        "based_on_intent_version",
        "certainty",
        "certainty_method",
        "certainty_evidence",
        "candidate_modes",
        "facet_stats",
    )
    fields = _load_object(value, names, path=path)
    certainty_value = fields["certainty"]
    return SearchBelief(
        based_on_intent_version=_load_integer(
            fields["based_on_intent_version"], path=path + ("based_on_intent_version",)
        ),
        certainty=(
            None
            if certainty_value is None
            else _load_number(certainty_value, path=path + ("certainty",))
        ),
        certainty_method=_load_string(
            fields["certainty_method"], path=path + ("certainty_method",)
        ),
        certainty_evidence=_load_certainty_evidence(
            fields["certainty_evidence"], path=path + ("certainty_evidence",)
        ),
        candidate_modes=_load_array(
            fields["candidate_modes"],
            path=path + ("candidate_modes",),
            item_loader=_load_candidate_mode,
        ),
        facet_stats=_load_array(
            fields["facet_stats"],
            path=path + ("facet_stats",),
            item_loader=_load_facet_stats,
        ),
    )


def _load_certainty_evidence(
    value: object,
    *,
    path: tuple[ErrorPathSegment, ...],
) -> CertaintyEvidence:
    names = (
        "probe_id",
        "probe_size",
        "raw_concentration",
        "quality_status",
        "quality_reasons",
    )
    fields = _load_object(value, names, path=path)
    concentration_value = fields["raw_concentration"]
    return CertaintyEvidence(
        probe_id=_load_string(fields["probe_id"], path=path + ("probe_id",)),
        probe_size=_load_integer(fields["probe_size"], path=path + ("probe_size",)),
        raw_concentration=(
            None
            if concentration_value is None
            else _load_number(concentration_value, path=path + ("raw_concentration",))
        ),
        quality_status=_load_enum(
            fields["quality_status"], ProbeQuality, path=path + ("quality_status",)
        ),
        quality_reasons=_load_string_tuple(
            fields["quality_reasons"], path=path + ("quality_reasons",)
        ),
    )


def _load_candidate_mode(
    value: object,
    *,
    path: tuple[ErrorPathSegment, ...],
) -> CandidateMode:
    fields = _load_object(value, ("id", "label", "mass", "representative_ids"), path=path)
    return CandidateMode(
        id=_load_string(fields["id"], path=path + ("id",)),
        label=_load_string(fields["label"], path=path + ("label",)),
        mass=_load_number(fields["mass"], path=path + ("mass",)),
        representative_ids=_load_string_tuple(
            fields["representative_ids"], path=path + ("representative_ids",)
        ),
    )


def _load_facet_stats(value: object, *, path: tuple[ErrorPathSegment, ...]) -> FacetStats:
    fields = _load_object(value, ("facet", "entropy", "coverage", "top_values"), path=path)
    return FacetStats(
        facet=_load_string(fields["facet"], path=path + ("facet",)),
        entropy=_load_number(fields["entropy"], path=path + ("entropy",)),
        coverage=_load_number(fields["coverage"], path=path + ("coverage",)),
        top_values=_load_array(
            fields["top_values"], path=path + ("top_values",), item_loader=_load_value_mass
        ),
    )


def _load_value_mass(value: object, *, path: tuple[ErrorPathSegment, ...]) -> ValueMass:
    fields = _load_object(value, ("value", "mass"), path=path)
    return ValueMass(
        value=_load_scalar(fields["value"], path=path + ("value",)),
        mass=_load_number(fields["mass"], path=path + ("mass",)),
    )


def _collect_object_pairs(values: list[tuple[str, object]]) -> _ObjectPairs:
    return _ObjectPairs(tuple(values))


def _materialize_json(value: object, *, path: tuple[ErrorPathSegment, ...]) -> object:
    if type(value) is _ObjectPairs:
        pairs = value.values
        seen: set[str] = set()
        for key, _ in pairs:
            _validate_unicode(key, path=path + (key,))
            if key in seen:
                _invalid_snapshot(path + (key,))
            seen.add(key)
        return {key: _materialize_json(item, path=path + (key,)) for key, item in pairs}
    if type(value) is list:
        items = cast(list[object], value)
        return [_materialize_json(item, path=path + (index,)) for index, item in enumerate(items)]
    if type(value) is str:
        _validate_unicode(value, path=path)
        return value
    if type(value) is float and not math.isfinite(value):
        _invalid_snapshot(path)
    if value is None or type(value) in (bool, int, float):
        return value
    _invalid_snapshot(path)


def _validate_unicode(value: str, *, path: tuple[ErrorPathSegment, ...]) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        _invalid_snapshot(path)


def _load_object(
    value: object,
    names: tuple[str, ...],
    *,
    path: tuple[ErrorPathSegment, ...],
) -> dict[str, object]:
    fields = _load_unchecked_object(value, path=path)
    expected = frozenset(names)
    unknown = sorted(set(fields).difference(expected))
    if unknown:
        raise SessionContextError(
            code=ErrorCode.UNKNOWN_FIELD,
            path=path + (unknown[0],),
        )
    for name in names:
        if name not in fields:
            _invalid_snapshot(path + (name,))
    return fields


def _load_unchecked_object(
    value: object,
    *,
    path: tuple[ErrorPathSegment, ...],
) -> dict[str, object]:
    if type(value) is not dict:
        _invalid_snapshot(path)
    return cast(dict[str, object], value)


def _load_array(
    value: object,
    *,
    path: tuple[ErrorPathSegment, ...],
    item_loader: Callable[..., _LoadT],
) -> tuple[_LoadT, ...]:
    if type(value) is not list:
        _invalid_snapshot(path)
    items = cast(list[object], value)
    return tuple(item_loader(item, path=path + (index,)) for index, item in enumerate(items))


def _load_string_tuple(value: object, *, path: tuple[ErrorPathSegment, ...]) -> tuple[str, ...]:
    if type(value) is not list:
        _invalid_snapshot(path)
    items = cast(list[object], value)
    return tuple(_load_string(item, path=path + (index,)) for index, item in enumerate(items))


def _load_canonical_string_set(
    value: object,
    *,
    path: tuple[ErrorPathSegment, ...],
) -> frozenset[str]:
    items = _load_string_tuple(value, path=path)
    if items != tuple(sorted(set(items))):
        _invalid_snapshot(path)
    return frozenset(items)


def _load_optional_string(value: object, *, path: tuple[ErrorPathSegment, ...]) -> str | None:
    if value is None:
        return None
    return _load_string(value, path=path)


def _load_string(value: object, *, path: tuple[ErrorPathSegment, ...]) -> str:
    if type(value) is not str:
        _invalid_snapshot(path)
    return value


def _load_integer(value: object, *, path: tuple[ErrorPathSegment, ...]) -> int:
    if type(value) is not int:
        _invalid_snapshot(path)
    return value


def _load_number(value: object, *, path: tuple[ErrorPathSegment, ...]) -> int | float:
    if type(value) not in (int, float):
        _invalid_snapshot(path)
    if type(value) is float and not math.isfinite(value):
        _invalid_snapshot(path)
    return cast(int | float, value)


def _load_scalar(value: object, *, path: tuple[ErrorPathSegment, ...]) -> ScalarValue:
    if type(value) not in (str, int, float, bool):
        _invalid_snapshot(path)
    if type(value) is float and not math.isfinite(value):
        _invalid_snapshot(path)
    return cast(ScalarValue, value)


def _load_enum(
    value: object,
    enum_type: type[_EnumT],
    *,
    path: tuple[ErrorPathSegment, ...],
) -> _EnumT:
    wire_value = _load_string(value, path=path)
    try:
        return enum_type(wire_value)
    except ValueError:
        _invalid_snapshot(path)


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


def _invalid_snapshot(path: tuple[ErrorPathSegment, ...] = ()) -> NoReturn:
    raise SessionContextError(code=ErrorCode.INVALID_SNAPSHOT, path=path) from None
