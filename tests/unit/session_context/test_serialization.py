from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from shopping_copilot.session_context.aggregates import (
    InteractionContext,
    ProductFeedback,
    SessionContext,
    SessionState,
    TurnRecord,
)
from shopping_copilot.session_context.errors import (
    ErrorCode,
    ErrorPathSegment,
    SessionContextError,
)
from shopping_copilot.session_context.models import (
    CandidateMode,
    CertaintyEvidence,
    Commitment,
    FacetStats,
    FeedbackSignal,
    IntentState,
    Operator,
    Preference,
    PreferenceSource,
    ProbeQuality,
    ProfilePrior,
    SearchBelief,
    SemanticPolarity,
    ValueMass,
)
from shopping_copilot.session_context.operations import (
    AddPreference,
    ClearFacet,
    RemovePreference,
    ReplaceFacet,
    SetDontCare,
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
from shopping_copilot.session_context.serialization import (
    SCHEMA_ID,
    decode_snapshot,
    encode_snapshot,
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


def _initial_intent() -> IntentState:
    return IntentState(goal=None, preferences=(), dont_care_facets=frozenset(), version=0)


def _preference(
    *,
    preference_id: str,
    source_turn: int,
    facet: str,
    operator: Operator,
    value: str | int,
) -> Preference:
    positive = operator in (Operator.EQ, Operator.IN)
    return Preference(
        id=preference_id,
        facet=facet,
        operator=operator,
        value=value,
        semantic_text=f"canonical {facet} evidence",
        semantic_polarity=(SemanticPolarity.POSITIVE if positive else SemanticPolarity.NEGATIVE),
        commitment=Commitment.HARD,
        source=PreferenceSource.USER_EXPLICIT,
        source_turn=source_turn,
        evidence_text=f"turn {source_turn} evidence",
        interpretation_confidence=0.875,
    )


@pytest.fixture
def rich_context(registry: FacetRegistry) -> SessionContext:
    blue = _preference(
        preference_id="p_1_1_0",
        source_turn=1,
        facet="color",
        operator=Operator.EQ,
        value="blue",
    )
    red = _preference(
        preference_id="p_2_0_0",
        source_turn=2,
        facet="color",
        operator=Operator.EQ,
        value="red",
    )
    budget = _preference(
        preference_id="p_5_0_0",
        source_turn=5,
        facet="budget",
        operator=Operator.LE,
        value=100,
    )
    batches = (
        StateUpdateBatch(
            turn=1,
            base_intent_version=0,
            operations=(
                SwitchGoal(new_goal="旅行用ジャケット"),
                AddPreference(preference=blue),
            ),
        ),
        StateUpdateBatch(
            turn=2,
            base_intent_version=1,
            operations=(ReplaceFacet(facet="color", preferences=(red,)),),
        ),
        StateUpdateBatch(
            turn=3,
            base_intent_version=2,
            operations=(SetDontCare(facet="budget"),),
        ),
        StateUpdateBatch(
            turn=4,
            base_intent_version=3,
            operations=(ClearFacet(facet="budget"),),
        ),
        StateUpdateBatch(
            turn=5,
            base_intent_version=4,
            operations=(AddPreference(preference=budget),),
        ),
        StateUpdateBatch(
            turn=6,
            base_intent_version=5,
            operations=(RemovePreference(preference_ids=(budget.id,)),),
        ),
        StateUpdateBatch(
            turn=7,
            base_intent_version=6,
            operations=(SetDontCare(facet="budget"),),
        ),
        StateUpdateBatch(
            turn=8,
            base_intent_version=7,
            operations=(SetDontCare(facet="color"),),
        ),
    )

    current = _initial_intent()
    records: list[TurnRecord] = []
    for batch in batches:
        before = current.version
        current = reduce_intent(current, batch, registry)
        feedback = (
            (
                ProductFeedback(
                    product_ids=("sku-一",),
                    signal=FeedbackSignal.COMPARATIVE,
                    compared_to_ids=("sku-2",),
                    evidence_text="一番目の方が良い",
                ),
            )
            if batch.turn == 3
            else ()
        )
        records.append(
            TurnRecord(
                turn=batch.turn,
                user_message=f"ユーザー {batch.turn}",
                intent_version_before=before,
                accepted_update=batch,
                intent_version_after=current.version,
                assistant_message=f"回答 {batch.turn}",
                question="どの色ですか？" if batch.turn == 1 else None,
                question_key="preferred_color" if batch.turn == 1 else None,
                ask_attribute="color" if batch.turn == 1 else None,
                shown_product_ids=(
                    ("sku-一", "sku-2")
                    if batch.turn == 1
                    else (("sku-3",) if batch.turn == 2 else ())
                ),
                feedback=feedback,
                search_belief_probe_id="probe_8" if batch.turn == 8 else None,
            )
        )

    belief = SearchBelief(
        based_on_intent_version=current.version,
        certainty=0.625,
        certainty_method="bods_v1",
        certainty_evidence=CertaintyEvidence(
            probe_id="probe_8",
            probe_size=4,
            raw_concentration=0.625,
            quality_status=ProbeQuality.VALID,
            quality_reasons=(),
        ),
        candidate_modes=(
            CandidateMode(
                id="mode_a",
                label="Blue cluster",
                mass=0.65,
                representative_ids=("sku-一", "sku-2"),
            ),
            CandidateMode(
                id="mode_b",
                label="Red cluster",
                mass=0.25,
                representative_ids=("sku-3",),
            ),
        ),
        facet_stats=(
            FacetStats(
                facet="budget",
                entropy=0.5,
                coverage=0.75,
                top_values=(ValueMass(value=100, mass=0.7), ValueMass(value=200, mass=0.3)),
            ),
            FacetStats(
                facet="color",
                entropy=0.8,
                coverage=1.0,
                top_values=(
                    ValueMass(value="blue", mass=0.6),
                    ValueMass(value="red", mass=0.4),
                ),
            ),
        ),
    )
    return SessionContext(
        session_id="session-日本語",
        profile=ProfilePrior(
            purchase_frequency="monthly",
            average_prior_rating=4.75,
            rating_style="careful",
            preference_tags=("旅行", "青"),
            summary="軽いジャケットを好む",
        ),
        state=SessionState(
            intent=current,
            interaction=InteractionContext(turns=tuple(records)),
            search_belief=belief,
        ),
    )


def _wire(context: SessionContext, registry: FacetRegistry) -> dict[str, Any]:
    return json.loads(encode_snapshot(context, registry))


def _dump(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _set_path(document: Any, path: tuple[str | int, ...], value: object) -> None:
    target = document
    for segment in path[:-1]:
        target = target[segment]
    target[path[-1]] = value


def _delete_path(document: Any, path: tuple[str | int, ...]) -> None:
    target = document
    for segment in path[:-1]:
        target = target[segment]
    del target[path[-1]]


def _assert_decode_error(
    data: bytes,
    registry: FacetRegistry,
    code: ErrorCode,
    path: tuple[ErrorPathSegment, ...] = (),
) -> SessionContextError:
    with pytest.raises(SessionContextError) as caught:
        decode_snapshot(data, registry)
    assert caught.value.code is code
    assert caught.value.path == path
    return caught.value


def test_schema_id_is_frozen() -> None:
    assert SCHEMA_ID == "shopping-copilot/session-context/v1"


def test_all_snapshot_dtos_and_six_operations_round_trip(
    registry: FacetRegistry,
    rich_context: SessionContext,
) -> None:
    encoded = encode_snapshot(rich_context, registry)
    decoded = decode_snapshot(encoded, registry)

    assert decoded == rich_context
    assert type(decoded.state.intent.dont_care_facets) is frozenset
    assert type(decoded.state.interaction.turns) is tuple
    assert type(decoded.profile.preference_tags) is tuple if decoded.profile is not None else False
    assert decoded.state.search_belief is not None
    assert type(decoded.state.search_belief.candidate_modes) is tuple
    assert type(decoded.state.search_belief.facet_stats[0].top_values) is tuple

    operation_types = {
        type(operation)
        for record in decoded.state.interaction.turns
        for operation in record.accepted_update.operations
        if record.accepted_update is not None
    }
    assert operation_types == {
        AddPreference,
        ReplaceFacet,
        RemovePreference,
        ClearFacet,
        SetDontCare,
        SwitchGoal,
    }


def test_snapshot_bytes_are_canonical_repeatable_and_unicode(
    registry: FacetRegistry,
    rich_context: SessionContext,
) -> None:
    first = encode_snapshot(rich_context, registry)
    second = encode_snapshot(rich_context, registry)
    parsed = json.loads(first)
    canonical = json.dumps(
        parsed,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    assert first == second == canonical
    assert first.startswith(b'{"payload":')
    assert "旅行用ジャケット".encode() in first
    assert b"\\u65c5" not in first
    assert parsed["payload"]["state"]["intent"]["dont_care_facets"] == ["budget", "color"]
    assert encode_snapshot(decode_snapshot(first, registry), registry) == first


def test_decode_replays_every_accepted_batch(
    registry: FacetRegistry,
    rich_context: SessionContext,
) -> None:
    decoded = decode_snapshot(encode_snapshot(rich_context, registry), registry)
    replayed = _initial_intent()
    for record in decoded.state.interaction.turns:
        if record.accepted_update is not None:
            replayed = reduce_intent(replayed, record.accepted_update, registry)
    assert replayed == decoded.state.intent


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (("extra",), ("extra",)),
        (("payload", "extra"), ("payload", "extra")),
        (
            ("payload", "state", "intent", "extra"),
            ("payload", "state", "intent", "extra"),
        ),
        (
            (
                "payload",
                "state",
                "interaction",
                "turns",
                0,
                "accepted_update",
                "operations",
                0,
                "extra",
            ),
            (
                "payload",
                "state",
                "interaction",
                "turns",
                0,
                "accepted_update",
                "operations",
                0,
                "extra",
            ),
        ),
    ],
    ids=("envelope", "payload", "nested_dto", "operation"),
)
def test_unknown_fields_report_the_exact_wire_path(
    registry: FacetRegistry,
    rich_context: SessionContext,
    path: tuple[str | int, ...],
    expected: tuple[ErrorPathSegment, ...],
) -> None:
    document = _wire(rich_context, registry)
    _set_path(document, path, "unexpected")
    _assert_decode_error(_dump(document), registry, ErrorCode.UNKNOWN_FIELD, expected)


def test_multiple_unknown_fields_are_reported_lexicographically(
    registry: FacetRegistry,
    rich_context: SessionContext,
) -> None:
    document = _wire(rich_context, registry)
    payload = document["payload"]
    payload["z_field"] = 1
    payload["a_field"] = 2
    _assert_decode_error(
        _dump(document),
        registry,
        ErrorCode.UNKNOWN_FIELD,
        ("payload", "a_field"),
    )


def test_missing_field_is_invalid_at_the_missing_field_path(
    registry: FacetRegistry,
    rich_context: SessionContext,
) -> None:
    document = _wire(rich_context, registry)
    path = ("payload", "profile", "summary")
    _delete_path(document, path)
    _assert_decode_error(_dump(document), registry, ErrorCode.INVALID_SNAPSHOT, path)


@pytest.mark.parametrize(
    "data",
    [
        (
            b'{"schema":"shopping-copilot/session-context/v1",'
            b'"schema":"shopping-copilot/session-context/v1","payload":{}}'
        ),
        (
            b'{"schema":"shopping-copilot/session-context/v1","payload":'
            b'{"session_id":"one","session_id":"two","profile":null,"state":{}}}'
        ),
    ],
    ids=("root", "nested"),
)
def test_duplicate_json_keys_are_deterministic_boundary_errors(
    registry: FacetRegistry,
    data: bytes,
) -> None:
    _assert_decode_error(data, registry, ErrorCode.INVALID_SNAPSHOT)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"{",
        b"[]",
        b"null",
        b"{} trailing",
        b"\xff",
        b"\xef\xbb\xbf{}",
    ],
    ids=("empty", "truncated", "array", "null", "trailing", "invalid_utf8", "bom"),
)
def test_malformed_json_and_utf8_are_boundary_errors(
    registry: FacetRegistry,
    data: bytes,
) -> None:
    _assert_decode_error(data, registry, ErrorCode.INVALID_SNAPSHOT)


def test_deeply_nested_json_never_leaks_a_recursion_error(registry: FacetRegistry) -> None:
    depth = 2_000
    data = b"[" * depth + b"null" + b"]" * depth
    _assert_decode_error(data, registry, ErrorCode.INVALID_SNAPSHOT)


def test_non_bytes_input_is_a_boundary_error(registry: FacetRegistry) -> None:
    with pytest.raises(SessionContextError) as caught:
        decode_snapshot(bytearray(b"{}"), registry)  # type: ignore[arg-type]
    assert caught.value.code is ErrorCode.INVALID_SNAPSHOT


@pytest.mark.parametrize("wire_value", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_nonfinite_json_numbers_are_deterministic_boundary_errors(
    registry: FacetRegistry,
    rich_context: SessionContext,
    wire_value: str,
) -> None:
    data = encode_snapshot(rich_context, registry).replace(
        b'"average_prior_rating":4.75',
        f'"average_prior_rating":{wire_value}'.encode(),
    )
    _assert_decode_error(
        data,
        registry,
        ErrorCode.INVALID_SNAPSHOT,
    )


@pytest.mark.parametrize(
    ("path", "invalid_value"),
    [
        (
            (
                "payload",
                "state",
                "interaction",
                "turns",
                0,
                "accepted_update",
                "operations",
                1,
                "preference",
                "operator",
            ),
            "contains",
        ),
        (
            (
                "payload",
                "state",
                "interaction",
                "turns",
                0,
                "accepted_update",
                "operations",
                1,
                "preference",
                "semantic_polarity",
            ),
            "liked",
        ),
        (
            (
                "payload",
                "state",
                "interaction",
                "turns",
                0,
                "accepted_update",
                "operations",
                1,
                "preference",
                "commitment",
            ),
            "required",
        ),
        (
            (
                "payload",
                "state",
                "interaction",
                "turns",
                0,
                "accepted_update",
                "operations",
                1,
                "preference",
                "source",
            ),
            "assistant",
        ),
        (
            ("payload", "state", "interaction", "turns", 2, "feedback", 0, "signal"),
            "better",
        ),
        (
            ("payload", "state", "search_belief", "certainty_evidence", "quality_status"),
            "unknown",
        ),
    ],
    ids=("operator", "polarity", "commitment", "source", "feedback", "probe_quality"),
)
def test_invalid_enum_wire_values_are_invalid_snapshot_errors(
    registry: FacetRegistry,
    rich_context: SessionContext,
    path: tuple[str | int, ...],
    invalid_value: str,
) -> None:
    document = _wire(rich_context, registry)
    _set_path(document, path, invalid_value)
    _assert_decode_error(_dump(document), registry, ErrorCode.INVALID_SNAPSHOT, path)


@pytest.mark.parametrize("discriminator", ["merge_facet", 7, None])
def test_invalid_operation_discriminators_are_boundary_errors(
    registry: FacetRegistry,
    rich_context: SessionContext,
    discriminator: object,
) -> None:
    path = (
        "payload",
        "state",
        "interaction",
        "turns",
        0,
        "accepted_update",
        "operations",
        0,
        "op",
    )
    document = _wire(rich_context, registry)
    _set_path(document, path, discriminator)
    _assert_decode_error(_dump(document), registry, ErrorCode.INVALID_SNAPSHOT, path)


def test_missing_operation_discriminator_is_a_boundary_error(
    registry: FacetRegistry,
    rich_context: SessionContext,
) -> None:
    path = (
        "payload",
        "state",
        "interaction",
        "turns",
        0,
        "accepted_update",
        "operations",
        0,
        "op",
    )
    document = _wire(rich_context, registry)
    _delete_path(document, path)
    _assert_decode_error(_dump(document), registry, ErrorCode.INVALID_SNAPSHOT, path)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("payload",), []),
        (("payload", "profile", "preference_tags"), "not-an-array"),
        (("payload", "state", "intent", "version"), True),
        (("payload", "state", "interaction", "turns"), {}),
        (("payload", "state", "search_belief", "certainty"), "0.5"),
        (("payload", "state", "search_belief", "facet_stats", 0, "top_values"), {}),
    ],
    ids=("payload", "string_tuple", "bool_integer", "turn_array", "number", "nested_array"),
)
def test_per_type_loaders_reject_wrong_json_shapes_at_the_field_path(
    registry: FacetRegistry,
    rich_context: SessionContext,
    path: tuple[str | int, ...],
    value: object,
) -> None:
    document = _wire(rich_context, registry)
    _set_path(document, path, value)
    _assert_decode_error(_dump(document), registry, ErrorCode.INVALID_SNAPSHOT, path)


@pytest.mark.parametrize(
    "facets",
    [("budget", "budget"), ("color", "budget")],
    ids=("duplicate", "noncanonical_order"),
)
def test_set_like_wire_array_is_not_silently_repaired(
    registry: FacetRegistry,
    rich_context: SessionContext,
    facets: tuple[str, ...],
) -> None:
    path = ("payload", "state", "intent", "dont_care_facets")
    document = _wire(rich_context, registry)
    _set_path(document, path, list(facets))
    _assert_decode_error(_dump(document), registry, ErrorCode.INVALID_SNAPSHOT, path)


def test_unknown_schema_has_its_stable_error(
    registry: FacetRegistry,
    rich_context: SessionContext,
) -> None:
    document = _wire(rich_context, registry)
    document["schema"] = "shopping-copilot/session-context/v2"
    _assert_decode_error(
        _dump(document),
        registry,
        ErrorCode.UNKNOWN_SCHEMA_VERSION,
        ("schema",),
    )


def test_decode_reruns_replay_validation_and_prefixes_the_wire_path(
    registry: FacetRegistry,
    rich_context: SessionContext,
) -> None:
    document = _wire(rich_context, registry)
    document["payload"]["state"]["intent"]["goal"] = "tampered goal"
    _assert_decode_error(
        _dump(document),
        registry,
        ErrorCode.INVALID_SESSION_TRANSITION,
        ("payload", "state", "intent"),
    )


def test_encode_validates_before_emitting_bytes(
    registry: FacetRegistry,
    rich_context: SessionContext,
) -> None:
    invalid = replace(rich_context, session_id=" padded ")
    with pytest.raises(SessionContextError) as caught:
        encode_snapshot(invalid, registry)
    assert caught.value.code is ErrorCode.INVALID_SESSION_ID
    assert caught.value.path == ("session_id",)


def test_decode_maps_python_integer_digit_limit_to_invalid_snapshot(
    registry: FacetRegistry,
    rich_context: SessionContext,
) -> None:
    huge_digits = "1" + "0" * 5_000
    data = encode_snapshot(rich_context, registry).replace(
        b'"average_prior_rating":4.75',
        f'"average_prior_rating":{huge_digits}'.encode(),
    )
    error = _assert_decode_error(data, registry, ErrorCode.INVALID_SNAPSHOT)
    assert type(error) is SessionContextError


def test_encode_maps_python_integer_digit_limit_to_invalid_snapshot(
    registry: FacetRegistry,
    rich_context: SessionContext,
) -> None:
    assert rich_context.profile is not None
    huge_rating = 10**4_999
    context = replace(
        rich_context,
        profile=replace(rich_context.profile, average_prior_rating=huge_rating),
    )
    with pytest.raises(SessionContextError) as caught:
        encode_snapshot(context, registry)
    assert caught.value.code is ErrorCode.INVALID_SNAPSHOT


def test_encode_maps_huge_belief_scalar_to_invalid_snapshot_without_native_leak(
    registry: FacetRegistry,
    rich_context: SessionContext,
) -> None:
    belief = rich_context.state.search_belief
    assert belief is not None
    huge_budget = FacetStats(
        facet="budget",
        entropy=0.0,
        coverage=1.0,
        top_values=(ValueMass(value=10**4_999, mass=1.0),),
    )
    context = replace(
        rich_context,
        state=replace(
            rich_context.state,
            search_belief=replace(belief, facet_stats=(huge_budget, belief.facet_stats[1])),
        ),
    )

    with pytest.raises(SessionContextError) as caught:
        encode_snapshot(context, registry)
    assert caught.value.code is ErrorCode.INVALID_SNAPSHOT


def test_escaped_unpaired_surrogate_is_a_deterministic_boundary_error(
    registry: FacetRegistry,
    rich_context: SessionContext,
) -> None:
    data = encode_snapshot(rich_context, registry).replace(
        '"summary":"軽いジャケットを好む"'.encode(),
        b'"summary":"\\ud800"',
    )
    _assert_decode_error(
        data,
        registry,
        ErrorCode.INVALID_SNAPSHOT,
    )


def test_operation_sequence_order_is_preserved(
    registry: FacetRegistry,
    rich_context: SessionContext,
) -> None:
    decoded = decode_snapshot(encode_snapshot(rich_context, registry), registry)
    first_update = decoded.state.interaction.turns[0].accepted_update
    assert first_update is not None
    assert tuple(operation.op for operation in first_update.operations) == (
        "switch_goal",
        "add_preference",
    )
    assert isinstance(first_update.operations[0], SwitchGoal)
    assert isinstance(first_update.operations[1], AddPreference)


def test_empty_optional_snapshot_round_trips(registry: FacetRegistry) -> None:
    context = SessionContext(
        session_id="empty",
        profile=None,
        state=SessionState(
            intent=_initial_intent(),
            interaction=InteractionContext(turns=()),
            search_belief=None,
        ),
    )
    assert decode_snapshot(encode_snapshot(context, registry), registry) == context
