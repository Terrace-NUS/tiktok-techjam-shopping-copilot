from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.query_understanding.evaluate_prompts import (
    evaluate_critical_assertions,
    main,
)
from scripts.query_understanding.suites import (
    NATURAL_SUITE_SCHEMA,
    SIMULATOR_SUITE_SCHEMA,
    CriticalAssertion,
    load_prompt_suite,
)
from shopping_copilot.query_understanding import (
    BehavioralDirectives,
    CategoryOption,
    ClarificationNeed,
    DiversityMode,
    ResolvedTurnIntent,
    ShownProductView,
    UnderstandingTrace,
    build_reconcile_request,
    request_payload,
)
from shopping_copilot.session_context import (
    Commitment,
    FeedbackSignal,
    IntentState,
    Operator,
    Preference,
    PreferenceSource,
    ProductFeedback,
    SemanticPolarity,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
NATURAL_SUITE = REPOSITORY_ROOT / "config/query_understanding/natural-prompts-v0.json"
INTENT_SPACE_NATURAL_SUITE = (
    REPOSITORY_ROOT / "config/query_understanding/intent-space-natural-prompts-v1.json"
)
EXPANDED_INTENT_SPACE_NATURAL_SUITE = (
    REPOSITORY_ROOT / "config/query_understanding/intent-space-natural-prompts-v2.json"
)
FACT_EXTRACTION_SUITE = (
    REPOSITORY_ROOT / "config/query_understanding/fact-extraction-prompts-v1.json"
)
SIMULATOR_SUITE = REPOSITORY_ROOT / "config/query_understanding/simulator-prompts-v0.json"
NATURAL_SHA256 = "b59580be67de6bc503092dfb58827121ddfacd43fe51ede89edee9a57b3ad902"
INTENT_SPACE_NATURAL_SHA256 = "650dc3b67942704c8ed634299a2ea389dc24331ce74641ba2a140a3b622d9391"
EXPANDED_INTENT_SPACE_NATURAL_SHA256 = (
    "fa092d4977fdd0aa6c158429323a8b06abf46e88f1378ad9fbcb42649d652c3f"
)
SIMULATOR_SHA256 = "3ef69b2c602251c3218313312e1defeab12fc2a9980eaa070e314b89c706609c"


def _natural_document() -> dict[str, object]:
    return {
        "schema": NATURAL_SUITE_SCHEMA,
        "suite_id": "test-natural-v0",
        "language": "English",
        "authorship": "Hand-authored unit fixture.",
        "oracle_policy": "Only critical retrieval-changing semantics are asserted.",
        "conversations": [
            {
                "id": "case_one",
                "tier": "smoke",
                "language": "en",
                "domain": "apparel",
                "tags": ["first_turn"],
                "initial_goal": None,
                "turns": [
                    {
                        "user_message": "Show me a navy cotton shirt under $120.",
                        "last_assistant_message": None,
                        "last_question": None,
                        "shown_products": [{"label": "Navy cotton shirt"}],
                        "critical_assertions": [
                            {
                                "kind": "preference",
                                "facet": "color",
                                "relation": "include",
                                "values": ["navy"],
                                "strength": None,
                                "text_contains": None,
                            }
                        ],
                    }
                ],
            }
        ],
    }


def _simulator_document() -> dict[str, object]:
    digest = "0" * 64
    return {
        "schema": SIMULATOR_SUITE_SCHEMA,
        "suite_id": "official-simulator-prompts-v0",
        "source": "official_toy_simulator",
        "description": "Visible toy-simulator messages only.",
        "generator": {
            "script": "scripts/query_understanding/generate_simulator_prompts.py",
            "suite_version": "v0",
            "dataset_sha256": digest,
            "catalog_sha256": digest,
            "evaluator_sha256": digest,
            "selection_method": "stable test selection",
            "selected_per_scenario": 1,
            "visible_turns_per_conversation": 1,
            "base_ask_schedule": [None],
        },
        "conversations": [
            {
                "id": "sim_case_one",
                "tier": "smoke",
                "turns": [
                    {
                        "turn": 1,
                        "user_message": "I'm looking for a shirt, but I'm still exploring.",
                        "last_assistant_message": None,
                        "last_question": None,
                        "response_shape": "initial_exploration",
                        "ask_attribute": None,
                    }
                ],
                "provenance": {
                    "sample_id": "public_0001",
                    "scenario_type": "browsing",
                    "difficulty_bucket": "medium",
                    "source_ordinal": 1,
                },
            }
        ],
    }


def _write_json(path: Path, document: object) -> None:
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")


def test_natural_loader_is_strict_and_keeps_nullable_initial_goal(tmp_path: Path) -> None:
    path = tmp_path / "natural.json"
    document = _natural_document()
    _write_json(path, document)

    suite = load_prompt_suite(path)

    assert suite.cohort == "natural"
    assert suite.conversations[0].initial_goal is None
    assert suite.conversations[0].turns[0].critical_assertions[0].relation == "include"

    document["unexpected"] = True  # type: ignore[index]
    _write_json(path, document)
    with pytest.raises(ValueError, match="invalid keys"):
        load_prompt_suite(path)


@pytest.mark.parametrize(
    "forbidden_key",
    ["ground_truth", "parent_asin", "intent_card", "behavior", "user_profile", "target"],
)
def test_simulator_loader_rejects_hidden_or_target_keys(
    tmp_path: Path,
    forbidden_key: str,
) -> None:
    path = tmp_path / "simulator.json"
    document = _simulator_document()
    provenance = document["conversations"][0]["provenance"]  # type: ignore[index]
    provenance[forbidden_key] = "secret"  # type: ignore[index]
    _write_json(path, document)

    with pytest.raises(ValueError, match="forbidden simulator key"):
        load_prompt_suite(path)


def _preference(
    *,
    identifier: str,
    facet: str | None,
    operator: Operator | None,
    value: str | int | tuple[str, ...] | None,
    semantic_text: str | None = None,
    polarity: SemanticPolarity | None = None,
    strength: Commitment = Commitment.HARD,
) -> Preference:
    return Preference(
        id=identifier,
        facet=facet,
        operator=operator,
        value=value,
        semantic_text=semantic_text,
        semantic_polarity=polarity,
        commitment=strength,
        source=PreferenceSource.USER_EXPLICIT,
        source_turn=1,
        evidence_text=semantic_text or "explicit fixture evidence",
        interpretation_confidence=1.0,
    )


def test_critical_assertions_use_relation_families_and_usd_price() -> None:
    before = IntentState(
        goal="find a dress", preferences=(), dont_care_facets=frozenset(), version=0
    )
    final = IntentState(
        goal="find a navy dress",
        preferences=(
            _preference(
                identifier="p_1_0_0",
                facet="color",
                operator=Operator.IN,
                value=("navy", "blue"),
            ),
            _preference(
                identifier="p_1_1_0",
                facet="price",
                operator=Operator.LE,
                value=12000,
            ),
            _preference(
                identifier="p_1_2_0",
                facet=None,
                operator=None,
                value=None,
                semantic_text="must not look touristy",
                polarity=SemanticPolarity.NEGATIVE,
                strength=Commitment.SOFT,
            ),
        ),
        dont_care_facets=frozenset({"brand"}),
        version=1,
    )
    shown = (
        ShownProductView(
            ref="product_0",
            product_ids=("fixture-product-0",),
            label="Navy dress",
        ),
    )
    resolved = ResolvedTurnIntent(
        update=None,
        final_intent=final,
        feedback=(
            ProductFeedback(
                product_ids=("fixture-product-0",),
                signal=FeedbackSignal.SELECTED,
                compared_to_ids=(),
                evidence_text="the first one",
            ),
        ),
        directives=BehavioralDirectives(
            diversity=DiversityMode.INCREASE,
            comparison_requested=True,
            explanation_requested=False,
        ),
        clarification=ClarificationNeed(needed=False, reason=None, alternatives=()),
        trace=UnderstandingTrace(
            attempts=(),
            interpretation_summary="fixture",
            semantic_fallback_facets=(),
        ),
    )
    assertions = (
        CriticalAssertion(kind="goal_contains", text="navy dress"),
        CriticalAssertion(kind="goal_contains_any", texts=("robe", "dress")),
        CriticalAssertion(kind="preference", facet="color", relation="include", values=("navy",)),
        CriticalAssertion(kind="preference", facet="price", relation="upper", values=("120",)),
        CriticalAssertion(
            kind="preference",
            relation="semantic_negative",
            strength="soft",
            text_contains="touristy",
        ),
        CriticalAssertion(kind="preference_absent", facet="material", relation="include"),
        CriticalAssertion(kind="dont_care", facet="brand", present=True),
        CriticalAssertion(kind="clarification", needed=False),
        CriticalAssertion(kind="directive", name="diversity", value="increase"),
        CriticalAssertion(kind="feedback", target_index=0, signal="selected"),
    )

    outcomes = evaluate_critical_assertions(
        assertions,
        before=before,
        resolved=resolved,
        shown_products=shown,
    )

    assert all(outcome.passed for outcome in outcomes)


def test_state_unchanged_requires_no_update_and_exact_prior_state() -> None:
    before = IntentState(
        goal="find a coat", preferences=(), dont_care_facets=frozenset(), version=0
    )
    resolved = ResolvedTurnIntent(
        update=None,
        final_intent=before,
        feedback=(),
        directives=BehavioralDirectives(
            diversity=DiversityMode.AUTO,
            comparison_requested=False,
            explanation_requested=False,
        ),
        clarification=ClarificationNeed(needed=False, reason=None, alternatives=()),
        trace=UnderstandingTrace(
            attempts=(), interpretation_summary="no change", semantic_fallback_facets=()
        ),
    )

    outcome = evaluate_critical_assertions(
        (CriticalAssertion(kind="state_unchanged"),),
        before=before,
        resolved=resolved,
        shown_products=(),
    )

    assert outcome[0].passed


def test_validate_only_never_requires_release_or_api_key(tmp_path: Path) -> None:
    suite_path = tmp_path / "natural.json"
    report_path = tmp_path / "report.json"
    _write_json(suite_path, _natural_document())

    exit_code = main(
        [
            "--cohort",
            "natural",
            "--natural-suite",
            str(suite_path),
            "--validate-only",
            "--tier",
            "smoke",
            "--conversation-id",
            "case_one",
            "--release",
            str(tmp_path / "missing-release"),
            "--output",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["mode"] == "validate_only"
    assert report["status"] == "valid"
    assert report["selection"]["conversation_ids"] == ["case_one"]
    assert report["suites"][0]["selected_turn_count"] == 1


def test_frozen_simulator_suite_identity_and_shape() -> None:
    payload = SIMULATOR_SUITE.read_bytes()
    suite = load_prompt_suite(SIMULATOR_SUITE)

    assert hashlib.sha256(payload).hexdigest() == SIMULATOR_SHA256
    assert len(suite.conversations) == 32
    assert sum(len(item.turns) for item in suite.conversations) == 128


def test_simulator_provenance_never_enters_model_request() -> None:
    suite = load_prompt_suite(SIMULATOR_SUITE)
    conversation = suite.conversations[0]
    turn = conversation.turns[0]
    request = build_reconcile_request(
        turn=turn.turn,
        latest_utterance=turn.user_message,
        current_intent=IntentState(
            goal=None,
            preferences=(),
            dont_care_facets=frozenset(),
            version=0,
        ),
        category_options=(
            CategoryOption(
                ref="category_0",
                scope_id="root_scope",
                label="All catalog products",
                is_root=True,
            ),
        ),
        last_assistant_message=turn.last_assistant_message,
        last_question=turn.last_question,
    )
    encoded = json.dumps(request_payload(request), ensure_ascii=False, sort_keys=True)

    assert conversation.provenance is not None
    for value in conversation.provenance.values():
        assert str(value) not in encoded
    assert "response_shape" not in encoded
    assert "ask_attribute" not in encoded
    assert "allowed_dont_care_facets" in encoded


def test_frozen_natural_suite_identity_and_shape() -> None:
    payload = NATURAL_SUITE.read_bytes()
    suite = load_prompt_suite(NATURAL_SUITE)

    assert hashlib.sha256(payload).hexdigest() == NATURAL_SHA256
    assert len(suite.conversations) == 40
    assert sum(len(item.turns) for item in suite.conversations) == 72


def test_intent_space_natural_suite_identity_shape_and_expectations() -> None:
    payload = INTENT_SPACE_NATURAL_SUITE.read_bytes()
    suite = load_prompt_suite(INTENT_SPACE_NATURAL_SUITE)
    expectations = [
        tag
        for conversation in suite.conversations
        for tag in conversation.tags
        if tag.startswith("expected_")
    ]

    assert hashlib.sha256(payload).hexdigest() == INTENT_SPACE_NATURAL_SHA256
    assert len(suite.conversations) == 24
    assert sum(len(item.turns) for item in suite.conversations) == 48
    assert expectations.count("expected_narrower") == 16
    assert expectations.count("expected_broader") == 3
    assert expectations.count("expected_stable") == 2
    assert expectations.count("expected_override") == 3
    assert len(expectations) == len(suite.conversations)


def test_expanded_intent_space_suite_identity_shape_and_expectations() -> None:
    payload = EXPANDED_INTENT_SPACE_NATURAL_SUITE.read_bytes()
    suite = load_prompt_suite(EXPANDED_INTENT_SPACE_NATURAL_SUITE)
    expectations = [
        tag
        for conversation in suite.conversations
        for tag in conversation.tags
        if tag.startswith("expected_")
    ]

    assert hashlib.sha256(payload).hexdigest() == EXPANDED_INTENT_SPACE_NATURAL_SHA256
    assert len(suite.conversations) == 60
    assert sum(len(item.turns) for item in suite.conversations) == 130
    assert expectations.count("expected_narrower") == 36
    assert expectations.count("expected_broader") == 10
    assert expectations.count("expected_stable") == 7
    assert expectations.count("expected_override") == 7
    assert len(expectations) == len(suite.conversations)
    assert (
        sum(
            len(turn.critical_assertions)
            for conversation in suite.conversations
            for turn in conversation.turns
        )
        == 49
    )


def test_fact_extraction_suite_covers_the_frozen_failure_shapes() -> None:
    suite = load_prompt_suite(FACT_EXTRACTION_SUITE)
    tags = {tag for conversation in suite.conversations for tag in conversation.tags}

    assert suite.suite_id == "fact-extraction-prompts-v1"
    assert len(suite.conversations) == 10
    assert sum(len(item.turns) for item in suite.conversations) == 11
    assert (
        sum(
            len(turn.critical_assertions)
            for conversation in suite.conversations
            for turn in conversation.turns
        )
        == 25
    )
    assert {
        "negation_scope",
        "subject_scope",
        "lexical_anchor",
        "no_duplicate_facet",
        "explicit_hard",
        "soft_fact",
    }.issubset(tags)
