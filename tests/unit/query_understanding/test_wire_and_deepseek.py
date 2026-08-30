from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import pytest

from shopping_copilot.query_understanding.deepseek import (
    DeepSeekConfig,
    DeepSeekProvider,
    HttpResponse,
)
from shopping_copilot.query_understanding.errors import (
    QueryUnderstandingError,
    QueryUnderstandingErrorCode,
)
from shopping_copilot.query_understanding.models import (
    ActivePreferenceView,
    CategoryOption,
    GoalAction,
    PreferenceRelation,
    ReconcileRequest,
    ShownProductView,
    UnderstandingDisposition,
)
from shopping_copilot.query_understanding.prompt import PROMPT_VERSION, SYSTEM_PROMPT
from shopping_copilot.query_understanding.wire import (
    TOOL_NAME,
    decode_reconciled_intent,
    reconcile_session_intent_tool,
)
from shopping_copilot.session_context import FeedbackSignal, SemanticPolarity


@dataclass(frozen=True, slots=True, kw_only=True)
class _RecordedCall:
    url: str
    headers: dict[str, str]
    body: bytes
    timeout_seconds: float


class _RecordingTransport:
    def __init__(self, response: HttpResponse) -> None:
        self.response = response
        self.calls: list[_RecordedCall] = []

    def post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
    ) -> HttpResponse:
        self.calls.append(
            _RecordedCall(
                url=url,
                headers=dict(headers),
                body=body,
                timeout_seconds=timeout_seconds,
            )
        )
        return self.response


def _valid_frame_document() -> dict[str, object]:
    return {
        "base_intent_version": 3,
        "disposition": "ready",
        "goal": {"action": "keep", "value": None},
        "keep_active_refs": ["active_0"],
        "new_preferences": {
            "structured": [
                {
                    "facet": "color",
                    "relation": "not_in",
                    "values": ["black"],
                    "strength": "hard",
                    "basis": "explicit",
                    "meaning": "Avoid black products.",
                    "evidence": "not black",
                    "confidence": 0.97,
                }
            ],
            "price": [
                {
                    "relation": "le",
                    "value_usd": "120.00",
                    "strength": "hard",
                    "basis": "explicit",
                    "meaning": "The budget is at most 120 USD.",
                    "evidence": "under $120",
                    "confidence": 1,
                }
            ],
            "semantic": [
                {
                    "polarity": "positive",
                    "strength": "soft",
                    "basis": "explicit",
                    "meaning": "The product should feel special as a gift.",
                    "evidence": "something special as a gift",
                    "confidence": 0.8,
                }
            ],
        },
        "dont_care_facets": ["size"],
        "feedback": [
            {
                "target_refs": ["product_0"],
                "signal": "negative",
                "compared_to_refs": [],
                "evidence": "I do not like the first one.",
            }
        ],
        "directives": {
            "diversity": "increase",
            "comparison_requested": False,
            "explanation_requested": True,
        },
        "clarification": {"needed": False, "reason": None, "alternatives": []},
        "summary": "Keep the old material, avoid black, and stay under 120 USD.",
    }


def _arguments(document: dict[str, object] | None = None) -> str:
    return json.dumps(
        document or _valid_frame_document(),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _chat_response(
    *,
    arguments: str | None = None,
    function_name: str = TOOL_NAME,
) -> HttpResponse:
    document = {
        "id": "chatcmpl-contract-test",
        "model": "deepseek-v4-flash",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_contract_test",
                            "type": "function",
                            "function": {
                                "name": function_name,
                                "arguments": arguments or _arguments(),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 101,
            "completion_tokens": 37,
            "total_tokens": 138,
        },
    }
    return HttpResponse(
        status=200,
        body=json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
    )


def _request() -> ReconcileRequest:
    return ReconcileRequest(
        turn=4,
        base_intent_version=3,
        latest_utterance="Keep the material, but not black and under $120.",
        current_goal="find a wedding guest dress",
        active_preferences=(
            ActivePreferenceView(
                ref="active_0",
                facet="material",
                relation="eq",
                value="silk",
                meaning="The material must be silk.",
                strength="hard",
                source="user_explicit",
            ),
        ),
        dont_care_facets=(),
        last_assistant_message="Here are three dresses.",
        last_question="Do you have a color preference?",
        category_options=(
            CategoryOption(
                ref="category_0",
                scope_id="cs_private_scope_id",
                label="Dresses",
                is_root=False,
            ),
        ),
        shown_products=(
            ShownProductView(
                ref="product_0",
                product_ids=("PRIVATE-ASIN",),
                label="First shown dress",
            ),
        ),
        allowed_dont_care_facets=("color", "material", "price", "size"),
    )


def _assert_every_object_schema_is_strict(schema: object) -> None:
    if type(schema) is dict:
        if schema.get("type") == "object":
            assert schema.get("additionalProperties") is False
            properties = schema.get("properties")
            required = schema.get("required")
            assert type(properties) is dict
            assert type(required) is list
            assert set(required) == set(properties)
        for value in schema.values():
            _assert_every_object_schema_is_strict(value)
    elif type(schema) is list:
        for value in schema:
            _assert_every_object_schema_is_strict(value)


def test_tool_schema_is_closed_complete_and_strict_flag_is_explicit() -> None:
    ordinary = reconcile_session_intent_tool(strict=False)
    strict = reconcile_session_intent_tool(strict=True)

    assert set(ordinary) == {"type", "function"}
    assert ordinary["type"] == "function"
    ordinary_function = ordinary["function"]
    strict_function = strict["function"]
    assert type(ordinary_function) is dict
    assert type(strict_function) is dict
    assert ordinary_function["name"] == TOOL_NAME
    assert "strict" not in ordinary_function
    assert strict_function["strict"] is True
    assert strict_function["parameters"] == ordinary_function["parameters"]
    _assert_every_object_schema_is_strict(ordinary_function["parameters"])
    parameters = cast(dict[str, object], ordinary_function["parameters"])
    properties = cast(dict[str, object], parameters["properties"])
    preference_groups = cast(dict[str, object], properties["new_preferences"])
    group_properties = cast(dict[str, object], preference_groups["properties"])
    assert set(group_properties) == {"structured", "price", "semantic"}


def test_v1_4_prompt_and_native_schema_expose_fact_extraction_policy() -> None:
    assert PROMPT_VERSION == "query_understanding_v1_4"
    assert "nose won't get red and irritated" in SYSTEM_PROMPT
    assert "95% gossypium, 5% spandex" in SYSTEM_PROMPT
    assert "For that, what matters is: ..." in SYSTEM_PROMPT
    assert "Boots Rain" in SYSTEM_PROMPT

    tool = reconcile_session_intent_tool(strict=False)
    function = cast(dict[str, object], tool["function"])
    parameters = cast(dict[str, object], function["parameters"])
    root_properties = cast(dict[str, object], parameters["properties"])
    groups = cast(dict[str, object], root_properties["new_preferences"])
    group_properties = cast(dict[str, object], groups["properties"])
    structured = cast(dict[str, object], group_properties["structured"])
    item = cast(dict[str, object], structured["items"])
    item_properties = cast(dict[str, object], item["properties"])

    assert "lexical anchors" in cast(dict[str, object], item_properties["values"])["description"]
    assert "wearer reaction" in cast(dict[str, object], item_properties["facet"])["description"]
    assert "non-negotiable" in cast(dict[str, object], item_properties["strength"])["description"]


def test_decoder_builds_the_complete_typed_frame() -> None:
    frame = decode_reconciled_intent(_arguments())

    assert frame.base_intent_version == 3
    assert frame.disposition is UnderstandingDisposition.READY
    assert frame.goal.action is GoalAction.KEEP
    assert frame.goal.value is None
    assert frame.keep_active_refs == ("active_0",)
    assert frame.structured_preferences[0].relation is PreferenceRelation.NOT_IN
    assert frame.structured_preferences[0].values == ("black",)
    assert frame.price_preferences[0].value_usd == "120.00"
    assert frame.semantic_preferences[0].polarity is SemanticPolarity.POSITIVE
    assert frame.feedback[0].signal is FeedbackSignal.NEGATIVE
    assert frame.dont_care_facets == ("size",)
    assert frame.directives.explanation_requested is True
    assert frame.clarification.needed is False


def test_decoder_accepts_goal_revision_and_non_price_numeric_range() -> None:
    document = _valid_frame_document()
    document["goal"] = {"action": "revise", "value": "watch"}
    groups = cast(dict[str, object], document["new_preferences"])
    structured = cast(list[dict[str, object]], groups["structured"])
    structured[0].update(
        {
            "facet": "case_size",
            "relation": "le",
            "values": ["40 mm"],
            "meaning": "watch case size must be 40 mm or smaller",
        }
    )

    frame = decode_reconciled_intent(_arguments(document))

    assert frame.goal.action is GoalAction.REVISE
    assert frame.goal.value == "watch"
    assert frame.structured_preferences[0].relation is PreferenceRelation.LE
    assert frame.structured_preferences[0].values == ("40 mm",)


@pytest.mark.parametrize(
    ("case", "expected_path"),
    [
        ("unknown_root_field", ()),
        ("duplicate_active_ref", ("keep_active_refs",)),
        ("old_array_shape", ("new_preferences",)),
        ("missing_preference_group", ("new_preferences",)),
        ("unknown_relation", ("new_preferences", "structured", 0, "relation")),
        ("semantic_with_facet", ("new_preferences", "semantic", 0)),
        ("numeric_on_color", ("new_preferences", "structured", 0)),
        ("categorical_price", ("new_preferences", "price", 0, "relation")),
        ("goal_keep_with_value", ("goal",)),
        ("clarification_mismatch", ("clarification", "needed")),
    ],
)
def test_decoder_rejects_noncanonical_frames(
    case: str,
    expected_path: tuple[str | int, ...],
) -> None:
    document = _valid_frame_document()
    groups = cast(dict[str, object], document["new_preferences"])
    structured = cast(list[dict[str, object]], groups["structured"])
    price = cast(list[dict[str, object]], groups["price"])
    semantic = cast(list[dict[str, object]], groups["semantic"])
    if case == "unknown_root_field":
        document["unexpected"] = True
    elif case == "duplicate_active_ref":
        document["keep_active_refs"] = ["active_0", "active_0"]
    elif case == "old_array_shape":
        document["new_preferences"] = []
    elif case == "missing_preference_group":
        del groups["semantic"]
    elif case == "unknown_relation":
        structured[0]["relation"] = "contains"
    elif case == "semantic_with_facet":
        semantic[0]["facet"] = "feature"
    elif case == "numeric_on_color":
        structured[0]["value_usd"] = "10"
    elif case == "categorical_price":
        price[0]["relation"] = "eq"
    elif case == "goal_keep_with_value":
        document["goal"]["value"] = "new goal"  # type: ignore[index]
    else:
        document["disposition"] = "needs_clarification"

    with pytest.raises(QueryUnderstandingError) as caught:
        decode_reconciled_intent(_arguments(document))

    assert caught.value.code is QueryUnderstandingErrorCode.INVALID_FRAME
    assert caught.value.path == expected_path


@pytest.mark.parametrize(
    ("source_relation", "expected_relation"),
    [
        ("eq", PreferenceRelation.IN),
        ("neq", PreferenceRelation.NOT_IN),
    ],
)
def test_decoder_normalizes_multiple_structured_values(
    source_relation: str,
    expected_relation: PreferenceRelation,
) -> None:
    document = _valid_frame_document()
    groups = cast(dict[str, object], document["new_preferences"])
    structured = cast(list[dict[str, object]], groups["structured"])
    structured[0]["relation"] = source_relation
    structured[0]["values"] = ["black", "navy"]

    frame = decode_reconciled_intent(_arguments(document))

    assert frame.structured_preferences[0].relation is expected_relation
    assert frame.structured_preferences[0].values == ("black", "navy")


def test_decoder_accepts_explicit_empty_typed_groups_for_no_change() -> None:
    document = _valid_frame_document()
    groups = cast(dict[str, object], document["new_preferences"])
    groups.update({"structured": [], "price": [], "semantic": []})
    document["disposition"] = "no_change"

    frame = decode_reconciled_intent(_arguments(document))

    assert frame.structured_preferences == ()
    assert frame.price_preferences == ()
    assert frame.semantic_preferences == ()


def test_decoder_reports_safe_cross_group_shape_reason() -> None:
    document = _valid_frame_document()
    groups = cast(dict[str, object], document["new_preferences"])
    structured = cast(list[dict[str, object]], groups["structured"])
    structured[0]["value_usd"] = "private-value-is-not-reported"

    with pytest.raises(QueryUnderstandingError) as caught:
        decode_reconciled_intent(_arguments(document))

    assert caught.value.path == ("new_preferences", "structured", 0)
    assert dict(caught.value.details) == {
        "reason": "object_shape",
        "unexpected": "value_usd",
    }
    assert "private-value-is-not-reported" not in str(caught.value.details)


def test_decoder_rejects_duplicate_json_members() -> None:
    arguments = _arguments()
    duplicated = arguments.replace(
        '"base_intent_version":3',
        '"base_intent_version":3,"base_intent_version":3',
        1,
    )

    with pytest.raises(QueryUnderstandingError) as caught:
        decode_reconciled_intent(duplicated)

    assert caught.value.code is QueryUnderstandingErrorCode.INVALID_FRAME


def test_provider_sends_the_native_forced_tool_request_and_decodes_trace() -> None:
    transport = _RecordingTransport(_chat_response())
    provider = DeepSeekProvider(
        api_key="unit-test-secret",
        config=DeepSeekConfig(timeout_seconds=7.5, max_tokens=777),
        transport=transport,
    )

    result = provider.reconcile(_request())

    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call.url == "https://api.deepseek.com/chat/completions"
    assert call.timeout_seconds == 7.5
    assert call.headers == {
        "Accept": "application/json",
        "Authorization": "Bearer unit-test-secret",
        "Content-Type": "application/json",
    }
    payload = json.loads(call.body)
    assert payload["model"] == "deepseek-v4-flash"
    assert payload["stream"] is False
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 777
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["tool_choice"] == {
        "type": "function",
        "function": {"name": TOOL_NAME},
    }
    assert payload["tools"] == [reconcile_session_intent_tool(strict=False)]
    assert len(payload["messages"]) == 2
    user_content = payload["messages"][1]["content"]
    assert "active_0" in user_content
    assert "not black and under $120" in user_content
    assert "PRIVATE-ASIN" not in user_content
    assert "cs_private_scope_id" not in user_content
    assert "C_t" not in user_content
    assert result.frame.base_intent_version == 3
    assert result.trace.response_id == "chatcmpl-contract-test"
    assert result.trace.model == "deepseek-v4-flash"
    assert result.trace.prompt_tokens == 101
    assert result.trace.completion_tokens == 37
    assert result.trace.total_tokens == 138


def test_strict_tool_mode_uses_the_beta_endpoint_and_strict_schema() -> None:
    transport = _RecordingTransport(_chat_response())
    provider = DeepSeekProvider(
        api_key="unit-test-secret",
        config=DeepSeekConfig(strict_tools=True),
        transport=transport,
    )

    provider.reconcile(_request())

    assert transport.calls[0].url == "https://api.deepseek.com/beta/chat/completions"
    payload = json.loads(transport.calls[0].body)
    assert payload["tools"] == [reconcile_session_intent_tool(strict=True)]


@pytest.mark.parametrize("api_key", [None, "", "   "])
def test_missing_api_key_fails_before_transport(api_key: str | None) -> None:
    transport = _RecordingTransport(_chat_response())
    provider = DeepSeekProvider(api_key=api_key, transport=transport)

    with pytest.raises(QueryUnderstandingError) as caught:
        provider.reconcile(_request())

    assert caught.value.code is QueryUnderstandingErrorCode.MISSING_API_KEY
    assert transport.calls == []


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, QueryUnderstandingErrorCode.PROVIDER_AUTH),
        (403, QueryUnderstandingErrorCode.PROVIDER_AUTH),
        (429, QueryUnderstandingErrorCode.PROVIDER_RATE_LIMIT),
        (500, QueryUnderstandingErrorCode.PROVIDER_UNAVAILABLE),
        (503, QueryUnderstandingErrorCode.PROVIDER_UNAVAILABLE),
        (400, QueryUnderstandingErrorCode.INVALID_PROVIDER_RESPONSE),
    ],
)
def test_http_statuses_map_to_stable_provider_errors(
    status: int,
    expected: QueryUnderstandingErrorCode,
) -> None:
    transport = _RecordingTransport(HttpResponse(status=status, body=b"provider error"))
    provider = DeepSeekProvider(api_key="unit-test-secret", transport=transport)

    with pytest.raises(QueryUnderstandingError) as caught:
        provider.reconcile(_request())

    assert caught.value.code is expected
    assert caught.value.details == (("status", status),)


def test_plain_assistant_response_is_not_accepted_as_a_state_update() -> None:
    response = HttpResponse(
        status=200,
        body=json.dumps(
            {
                "id": "chatcmpl-plain",
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "Here are some dresses you might like.",
                        },
                    }
                ],
            },
            separators=(",", ":"),
        ).encode("utf-8"),
    )
    provider = DeepSeekProvider(
        api_key="unit-test-secret",
        transport=_RecordingTransport(response),
    )

    with pytest.raises(QueryUnderstandingError) as caught:
        provider.reconcile(_request())

    assert caught.value.code is QueryUnderstandingErrorCode.INVALID_TOOL_CALL


def test_wrong_function_name_is_not_accepted() -> None:
    provider = DeepSeekProvider(
        api_key="unit-test-secret",
        transport=_RecordingTransport(_chat_response(function_name="other_tool")),
    )

    with pytest.raises(QueryUnderstandingError) as caught:
        provider.reconcile(_request())

    assert caught.value.code is QueryUnderstandingErrorCode.INVALID_TOOL_CALL
