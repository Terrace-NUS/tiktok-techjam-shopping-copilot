"""Strict loaders for hand-authored and competition-simulator QU prompt suites."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

NATURAL_SUITE_SCHEMA = "shopping-copilot/query-understanding-natural-suite/v0"
SIMULATOR_SUITE_SCHEMA = "shopping-copilot/query-understanding-simulator-suite/v0"
SIMULATOR_SUITE_ID = "official-simulator-prompts-v0"
SIMULATOR_OTHER_SUITE_ID = "official-simulator-other-prompts-v1"
SIMULATOR_SUITE_IDS = frozenset({SIMULATOR_SUITE_ID, SIMULATOR_OTHER_SUITE_ID})

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:[_-][a-z0-9]+)*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TIERS = frozenset({"smoke", "full"})
_RELATION_FAMILIES = frozenset(
    {
        "include",
        "exclude",
        "lower",
        "upper",
        "semantic_positive",
        "semantic_negative",
    }
)
_ASSERTION_KINDS = frozenset(
    {
        "goal_contains",
        "goal_contains_any",
        "goal_not_contains",
        "preference",
        "preference_absent",
        "facet_absent",
        "dont_care",
        "state_unchanged",
        "clarification",
        "directive",
        "feedback",
    }
)
_SIMULATOR_HIDDEN_KEYS = frozenset(
    {
        "ground_truth",
        "parent_asin",
        "intent_card",
        "behavior",
        "user_profile",
        "target",
    }
)
_SIMULATOR_RESPONSE_SHAPES = frozenset(
    {
        "initial_requirement",
        "initial_exploration",
        "initial_preference",
        "attribute_disclosure",
        "explicit_no_preference",
        "no_additional_preference",
        "negative_feedback",
        "explicit_override",
    }
)
_SIMULATOR_ASK_ATTRIBUTES = frozenset(
    {
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
    }
)


@dataclass(frozen=True, slots=True)
class ShownProduct:
    label: str


@dataclass(frozen=True, slots=True)
class CriticalAssertion:
    """One semantic predicate; fields unused by its kind remain ``None``."""

    kind: str
    facet: str | None = None
    relation: str | None = None
    values: tuple[str, ...] = ()
    strength: str | None = None
    text_contains: str | None = None
    text: str | None = None
    texts: tuple[str, ...] = ()
    needed: bool | None = None
    present: bool | None = None
    name: str | None = None
    value: str | bool | None = None
    signal: str | None = None
    target_index: int | None = None


@dataclass(frozen=True, slots=True)
class PromptTurn:
    turn: int
    user_message: str
    last_assistant_message: str | None
    last_question: str | None
    shown_products: tuple[ShownProduct, ...] = ()
    critical_assertions: tuple[CriticalAssertion, ...] = ()
    response_shape: str | None = None
    ask_attribute: str | None = None


@dataclass(frozen=True, slots=True)
class PromptConversation:
    identifier: str
    tier: str
    turns: tuple[PromptTurn, ...]
    language: str | None = None
    domain: str | None = None
    tags: tuple[str, ...] = ()
    provenance: Mapping[str, str | int] | None = None
    initial_goal: str | None = None


@dataclass(frozen=True, slots=True)
class PromptSuite:
    schema: str
    suite_id: str
    cohort: Literal["natural", "simulator"]
    description: str
    conversations: tuple[PromptConversation, ...]
    language: str | None = None
    authorship: str | None = None
    oracle_policy: str | None = None
    source: str | None = None
    generator: Mapping[str, object] | None = None


def load_prompt_suite(path: str | Path) -> PromptSuite:
    """Strictly load either supported suite schema from UTF-8 JSON."""

    source = Path(path)
    try:
        parsed: object = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"prompt suite is not valid strict JSON: {error}") from error
    root = _object(parsed, path="$")
    schema = _text(root.get("schema"), path="$.schema")
    if schema == NATURAL_SUITE_SCHEMA:
        return _load_natural(root)
    if schema == SIMULATOR_SUITE_SCHEMA:
        _reject_hidden_simulator_keys(root, path="$")
        return _load_simulator(root)
    raise ValueError(f"$.schema is unsupported: {schema!r}")


def _load_natural(root: dict[str, object]) -> PromptSuite:
    _exact_keys(
        root,
        {
            "schema",
            "suite_id",
            "language",
            "authorship",
            "oracle_policy",
            "conversations",
        },
        path="$",
    )
    suite_id = _identifier(root["suite_id"], path="$.suite_id")
    language = _text(root["language"], path="$.language")
    authorship = _text(root["authorship"], path="$.authorship")
    oracle_policy = _text(root["oracle_policy"], path="$.oracle_policy")
    conversations = _array(root["conversations"], path="$.conversations")
    if not conversations:
        raise ValueError("$.conversations must not be empty")
    seen: set[str] = set()
    parsed = tuple(
        _natural_conversation(
            item,
            index=index,
            seen=seen,
        )
        for index, item in enumerate(conversations)
    )
    return PromptSuite(
        schema=NATURAL_SUITE_SCHEMA,
        suite_id=suite_id,
        cohort="natural",
        description=authorship,
        conversations=parsed,
        language=language,
        authorship=authorship,
        oracle_policy=oracle_policy,
    )


def _natural_conversation(
    value: object,
    *,
    index: int,
    seen: set[str],
) -> PromptConversation:
    path = f"$.conversations[{index}]"
    item = _object(value, path=path)
    _exact_keys(
        item,
        {"id", "tier", "language", "domain", "tags", "initial_goal", "turns"},
        path=path,
    )
    identifier = _unique_identifier(item["id"], path=f"{path}.id", seen=seen)
    tier = _enum_text(item["tier"], _TIERS, path=f"{path}.tier")
    language = _text(item["language"], path=f"{path}.language")
    domain = _identifier(item["domain"], path=f"{path}.domain")
    tags = _identifier_array(item["tags"], path=f"{path}.tags", allow_empty=False)
    raw_turns = _array(item["turns"], path=f"{path}.turns")
    if not raw_turns:
        raise ValueError(f"{path}.turns must not be empty")
    turns = tuple(
        _natural_turn(turn, path=f"{path}.turns[{turn_index}]", turn=turn_index + 1)
        for turn_index, turn in enumerate(raw_turns)
    )
    return PromptConversation(
        identifier=identifier,
        tier=tier,
        language=language,
        domain=domain,
        tags=tags,
        turns=turns,
        initial_goal=_nullable_text(item["initial_goal"], path=f"{path}.initial_goal"),
    )


def _natural_turn(value: object, *, path: str, turn: int) -> PromptTurn:
    item = _object(value, path=path)
    _exact_keys(
        item,
        {
            "user_message",
            "last_assistant_message",
            "last_question",
            "shown_products",
            "critical_assertions",
        },
        path=path,
    )
    products = tuple(
        _shown_product(product, path=f"{path}.shown_products[{index}]")
        for index, product in enumerate(
            _array(item["shown_products"], path=f"{path}.shown_products")
        )
    )
    assertions = tuple(
        _assertion(assertion, path=f"{path}.critical_assertions[{index}]")
        for index, assertion in enumerate(
            _array(item["critical_assertions"], path=f"{path}.critical_assertions")
        )
    )
    for index, assertion in enumerate(assertions):
        if assertion.kind == "feedback" and cast(int, assertion.target_index) >= len(products):
            raise ValueError(
                f"{path}.critical_assertions[{index}].target_index is outside shown_products"
            )
    return PromptTurn(
        turn=turn,
        user_message=_text(item["user_message"], path=f"{path}.user_message"),
        last_assistant_message=_optional_text(
            item["last_assistant_message"], path=f"{path}.last_assistant_message"
        ),
        last_question=_optional_text(item["last_question"], path=f"{path}.last_question"),
        shown_products=products,
        critical_assertions=assertions,
    )


def _shown_product(value: object, *, path: str) -> ShownProduct:
    item = _object(value, path=path)
    _exact_keys(item, {"label"}, path=path)
    return ShownProduct(label=_text(item["label"], path=f"{path}.label"))


def _assertion(value: object, *, path: str) -> CriticalAssertion:
    item = _object(value, path=path)
    kind = _enum_text(item.get("kind"), _ASSERTION_KINDS, path=f"{path}.kind")
    if kind in {"goal_contains", "goal_not_contains"}:
        _exact_keys(item, {"kind", "text"}, path=path)
        return CriticalAssertion(kind=kind, text=_text(item["text"], path=f"{path}.text"))
    if kind == "goal_contains_any":
        _exact_keys(item, {"kind", "texts"}, path=path)
        texts = tuple(
            _text(raw, path=f"{path}.texts[{index}]")
            for index, raw in enumerate(_array(item["texts"], path=f"{path}.texts"))
        )
        if not texts or len(set(texts)) != len(texts):
            raise ValueError(f"{path}.texts must contain unique alternatives")
        return CriticalAssertion(kind=kind, texts=texts)
    if kind in {"preference", "preference_absent"}:
        return _preference_assertion(item, kind=kind, path=path)
    if kind in {"facet_absent", "dont_care"}:
        expected = {"kind", "facet"} if kind == "facet_absent" else {"kind", "facet", "present"}
        _exact_keys(item, expected, path=path)
        return CriticalAssertion(
            kind=kind,
            facet=_text(item["facet"], path=f"{path}.facet"),
            present=(
                _bool(item["present"], path=f"{path}.present") if kind == "dont_care" else None
            ),
        )
    if kind == "state_unchanged":
        _exact_keys(item, {"kind"}, path=path)
        return CriticalAssertion(kind=kind)
    if kind == "clarification":
        _exact_keys(item, {"kind", "needed"}, path=path)
        return CriticalAssertion(kind=kind, needed=_bool(item["needed"], path=f"{path}.needed"))
    if kind == "directive":
        _exact_keys(item, {"kind", "name", "value"}, path=path)
        name = _enum_text(
            item["name"],
            {"diversity", "comparison_requested", "explanation_requested"},
            path=f"{path}.name",
        )
        raw = item["value"]
        if name == "diversity":
            parsed_value: str | bool = _enum_text(
                raw, {"auto", "increase", "decrease"}, path=f"{path}.value"
            )
        else:
            parsed_value = _bool(raw, path=f"{path}.value")
        return CriticalAssertion(kind=kind, name=name, value=parsed_value)
    if kind == "feedback":
        _exact_keys(item, {"kind", "target_index", "signal"}, path=path)
        return CriticalAssertion(
            kind=kind,
            signal=_enum_text(
                item["signal"],
                {"positive", "negative", "selected", "rejected", "comparative"},
                path=f"{path}.signal",
            ),
            target_index=_nonnegative_int(item["target_index"], path=f"{path}.target_index"),
        )
    raise AssertionError(f"unhandled assertion kind: {kind}")


def _preference_assertion(
    item: dict[str, object],
    *,
    kind: str,
    path: str,
) -> CriticalAssertion:
    expected = {"kind", "relation", "facet", "values", "strength", "text_contains"}
    _exact_keys(item, expected, path=path)
    relation = (
        None
        if item["relation"] is None
        else _enum_text(item["relation"], _RELATION_FAMILIES, path=f"{path}.relation")
    )
    facet = _nullable_text(item["facet"], path=f"{path}.facet")
    values = tuple(
        _text(raw, path=f"{path}.values[{index}]")
        for index, raw in enumerate(_array(item["values"], path=f"{path}.values"))
    )
    strength = (
        None
        if item["strength"] is None
        else _enum_text(item["strength"], {"hard", "soft"}, path=f"{path}.strength")
    )
    text_contains = _nullable_text(item["text_contains"], path=f"{path}.text_contains")
    if (
        facet is None
        and relation is None
        and not values
        and strength is None
        and text_contains is None
    ):
        raise ValueError(f"{path} preference matcher is too broad")
    return CriticalAssertion(
        kind=kind,
        facet=facet,
        relation=relation,
        values=values,
        strength=strength,
        text_contains=text_contains,
    )


def _load_simulator(root: dict[str, object]) -> PromptSuite:
    _exact_keys(
        root,
        {"schema", "suite_id", "source", "description", "generator", "conversations"},
        path="$",
    )
    suite_id = _text(root["suite_id"], path="$.suite_id")
    if suite_id not in SIMULATOR_SUITE_IDS:
        raise ValueError(f"$.suite_id must be one of {sorted(SIMULATOR_SUITE_IDS)!r}")
    source = _text(root["source"], path="$.source")
    if source != "official_conversation_simulator":
        raise ValueError("$.source must equal 'official_conversation_simulator'")
    description = _text(root["description"], path="$.description")
    generator = _simulator_generator(root["generator"])
    conversations = _array(root["conversations"], path="$.conversations")
    if not conversations:
        raise ValueError("$.conversations must not be empty")
    seen: set[str] = set()
    parsed = tuple(
        _simulator_conversation(item, index=index, seen=seen)
        for index, item in enumerate(conversations)
    )
    return PromptSuite(
        schema=SIMULATOR_SUITE_SCHEMA,
        suite_id=suite_id,
        cohort="simulator",
        description=description,
        source=source,
        generator=generator,
        conversations=parsed,
    )


def _simulator_generator(value: object) -> Mapping[str, object]:
    path = "$.generator"
    item = _object(value, path=path)
    _exact_keys(
        item,
        {
            "script",
            "suite_version",
            "dataset_sha256",
            "catalog_sha256",
            "evaluator_sha256",
            "selection_method",
            "selected_per_scenario",
            "visible_turns_per_conversation",
            "base_ask_schedule",
        },
        path=path,
    )
    for key in ("script", "suite_version", "selection_method"):
        _text(item[key], path=f"{path}.{key}")
    for key in ("dataset_sha256", "catalog_sha256", "evaluator_sha256"):
        digest = _text(item[key], path=f"{path}.{key}")
        if _SHA256.fullmatch(digest) is None:
            raise ValueError(f"{path}.{key} must be a lowercase 64-character SHA-256")
    _positive_int(item["selected_per_scenario"], path=f"{path}.selected_per_scenario")
    _positive_int(
        item["visible_turns_per_conversation"],
        path=f"{path}.visible_turns_per_conversation",
    )
    schedule = _array(item["base_ask_schedule"], path=f"{path}.base_ask_schedule")
    for index, raw in enumerate(schedule):
        if raw is not None:
            _enum_text(
                raw,
                _SIMULATOR_ASK_ATTRIBUTES,
                path=f"{path}.base_ask_schedule[{index}]",
            )
    return item


def _simulator_conversation(
    value: object,
    *,
    index: int,
    seen: set[str],
) -> PromptConversation:
    path = f"$.conversations[{index}]"
    item = _object(value, path=path)
    _exact_keys(item, {"id", "tier", "turns", "provenance"}, path=path)
    identifier = _unique_identifier(item["id"], path=f"{path}.id", seen=seen)
    tier = _enum_text(item["tier"], _TIERS, path=f"{path}.tier")
    provenance = _simulator_provenance(item["provenance"], path=f"{path}.provenance")
    raw_turns = _array(item["turns"], path=f"{path}.turns")
    if not raw_turns:
        raise ValueError(f"{path}.turns must not be empty")
    turns = tuple(
        _simulator_turn(turn, path=f"{path}.turns[{turn_index}]")
        for turn_index, turn in enumerate(raw_turns)
    )
    observed_turns = tuple(turn.turn for turn in turns)
    if any(right <= left for left, right in zip(observed_turns, observed_turns[1:], strict=False)):
        raise ValueError(f"{path}.turns must have strictly increasing turn numbers")
    return PromptConversation(
        identifier=identifier,
        tier=tier,
        turns=turns,
        provenance=provenance,
    )


def _simulator_provenance(value: object, *, path: str) -> Mapping[str, str | int]:
    item = _object(value, path=path)
    _exact_keys(
        item,
        {"sample_id", "scenario_type", "difficulty_bucket", "source_ordinal"},
        path=path,
    )
    sample_id = _text(item["sample_id"], path=f"{path}.sample_id")
    scenario = _enum_text(
        item["scenario_type"],
        {"buying", "browsing", "intent_override", "boundary"},
        path=f"{path}.scenario_type",
    )
    difficulty = _enum_text(
        item["difficulty_bucket"], {"easy", "medium", "hard"}, path=f"{path}.difficulty_bucket"
    )
    ordinal = _positive_int(item["source_ordinal"], path=f"{path}.source_ordinal")
    return {
        "sample_id": sample_id,
        "scenario_type": scenario,
        "difficulty_bucket": difficulty,
        "source_ordinal": ordinal,
    }


def _simulator_turn(value: object, *, path: str) -> PromptTurn:
    item = _object(value, path=path)
    _exact_keys(
        item,
        {
            "turn",
            "user_message",
            "last_assistant_message",
            "last_question",
            "response_shape",
            "ask_attribute",
        },
        path=path,
    )
    ask_attribute = (
        None
        if item["ask_attribute"] is None
        else _enum_text(
            item["ask_attribute"], _SIMULATOR_ASK_ATTRIBUTES, path=f"{path}.ask_attribute"
        )
    )
    return PromptTurn(
        turn=_positive_int(item["turn"], path=f"{path}.turn"),
        user_message=_text(item["user_message"], path=f"{path}.user_message"),
        last_assistant_message=_optional_text(
            item["last_assistant_message"], path=f"{path}.last_assistant_message"
        ),
        last_question=_optional_text(item["last_question"], path=f"{path}.last_question"),
        response_shape=_enum_text(
            item["response_shape"], _SIMULATOR_RESPONSE_SHAPES, path=f"{path}.response_shape"
        ),
        ask_attribute=ask_attribute,
    )


def _reject_hidden_simulator_keys(value: object, *, path: str) -> None:
    if type(value) is dict:
        for key, child in cast(dict[str, object], value).items():
            lowered = key.casefold()
            if (
                lowered in _SIMULATOR_HIDDEN_KEYS
                or "parent_asin" in lowered
                or lowered.startswith("target_")
            ):
                raise ValueError(f"{path} contains forbidden simulator key {key!r}")
            _reject_hidden_simulator_keys(child, path=f"{path}.{key}")
    elif type(value) is list:
        for index, child in enumerate(cast(list[object], value)):
            _reject_hidden_simulator_keys(child, path=f"{path}[{index}]")


def _object(value: object, *, path: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{path} must be an object")
    return cast(dict[str, object], value)


def _array(value: object, *, path: str) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"{path} must be an array")
    return cast(list[object], value)


def _exact_keys(item: Mapping[str, object], expected: set[str], *, path: str) -> None:
    _allowed_keys(item, allowed=expected, required=expected, path=path)


def _allowed_keys(
    item: Mapping[str, object],
    *,
    allowed: set[str],
    required: set[str],
    path: str,
) -> None:
    observed = set(item)
    missing = sorted(required - observed)
    extra = sorted(observed - allowed)
    if missing or extra:
        raise ValueError(f"{path} has invalid keys; missing={missing!r}, extra={extra!r}")


def _text(value: object, *, path: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{path} must be a non-empty trimmed string")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError(f"{path} contains a lone surrogate")
    return value


def _optional_text(value: object, *, path: str) -> str | None:
    return None if value is None else _text(value, path=path)


def _nullable_text(value: object, *, path: str) -> str | None:
    return None if value is None else _text(value, path=path)


def _identifier(value: object, *, path: str) -> str:
    result = _text(value, path=path)
    if _IDENTIFIER.fullmatch(result) is None:
        raise ValueError(f"{path} must be a canonical identifier")
    return result


def _unique_identifier(value: object, *, path: str, seen: set[str]) -> str:
    result = _identifier(value, path=path)
    if result in seen:
        raise ValueError(f"{path} duplicates conversation ID {result!r}")
    seen.add(result)
    return result


def _identifier_array(value: object, *, path: str, allow_empty: bool) -> tuple[str, ...]:
    raw = _array(value, path=path)
    if not allow_empty and not raw:
        raise ValueError(f"{path} must not be empty")
    result = tuple(_identifier(item, path=f"{path}[{index}]") for index, item in enumerate(raw))
    if len(result) != len(set(result)):
        raise ValueError(f"{path} must not contain duplicates")
    return result


def _enum_text(value: object, allowed: set[str] | frozenset[str], *, path: str) -> str:
    result = _text(value, path=path)
    if result not in allowed:
        raise ValueError(f"{path} must be one of {sorted(allowed)!r}")
    return result


def _bool(value: object, *, path: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{path} must be a boolean")
    return value


def _positive_int(value: object, *, path: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, path: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(raw: str) -> object:
    raise ValueError(f"non-finite JSON number: {raw}")


__all__ = (
    "NATURAL_SUITE_SCHEMA",
    "SIMULATOR_SUITE_ID",
    "SIMULATOR_OTHER_SUITE_ID",
    "SIMULATOR_SUITE_IDS",
    "SIMULATOR_SUITE_SCHEMA",
    "CriticalAssertion",
    "PromptConversation",
    "PromptSuite",
    "PromptTurn",
    "ShownProduct",
    "load_prompt_suite",
)
