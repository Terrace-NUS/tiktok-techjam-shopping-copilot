"""Validate or live-replay Query Understanding prompt suites.

Validation is entirely offline. Live replay requires an API key from an explicitly
named environment variable or file and never includes that key in the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.query_understanding.suites import (  # noqa: E402
    CriticalAssertion,
    PromptConversation,
    PromptSuite,
    PromptTurn,
    load_prompt_suite,
)
from shopping_copilot.catalog.semantic import (  # noqa: E402
    CatalogSemanticError,
    CatalogSemanticGateway,
)
from shopping_copilot.catalog.semantic.release import (  # noqa: E402
    load_catalog_semantic_release,
)
from shopping_copilot.catalog.semantic.runtime import (  # noqa: E402
    SYSTEM_PRODUCT_CATEGORY_FACET_ID,
)
from shopping_copilot.query_understanding import (  # noqa: E402
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    IntentMaterializer,
    QueryUnderstandingError,
    QueryUnderstandingService,
    ResolvedTurnIntent,
    ShownProductView,
    build_reconcile_request,
    category_options_from_registry,
    reconcile_session_intent_tool,
)
from shopping_copilot.query_understanding.deepseek import (  # noqa: E402
    DeepSeekConfig,
    DeepSeekProvider,
)
from shopping_copilot.session_context import IntentState, Preference  # noqa: E402

REPORT_SCHEMA = "shopping-copilot/query-understanding-prompt-evaluation/v0"
DEFAULT_NATURAL_SUITE = Path("config/query_understanding/natural-prompts-v0.json")
DEFAULT_SIMULATOR_SUITE = Path("config/query_understanding/simulator-prompts-v0.json")
DEFAULT_RELEASE = Path("artifacts/catalog-semantic/release-v0")

_RELATIONS: dict[str, frozenset[str]] = {
    "include": frozenset({"eq", "in"}),
    "exclude": frozenset({"neq", "not_in"}),
    "lower": frozenset({"gt", "ge"}),
    "upper": frozenset({"lt", "le"}),
    "semantic_positive": frozenset({"semantic_positive"}),
    "semantic_negative": frozenset({"semantic_negative"}),
}


@dataclass(frozen=True, slots=True)
class AssertionOutcome:
    kind: str
    passed: bool
    reason: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural-suite", type=Path, default=DEFAULT_NATURAL_SUITE)
    parser.add_argument("--simulator-suite", type=Path, default=DEFAULT_SIMULATOR_SUITE)
    parser.add_argument("--cohort", choices=("natural", "simulator", "all"), default="all")
    parser.add_argument(
        "--tier",
        choices=("smoke", "full"),
        default="smoke",
        help="smoke selects only smoke conversations; full selects every conversation",
    )
    parser.add_argument("--limit", type=int, default=None, help="maximum conversations per cohort")
    parser.add_argument(
        "--conversation-id",
        action="append",
        default=[],
        help="select one conversation by ID; repeat to select several",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    key_group = parser.add_mutually_exclusive_group()
    key_group.add_argument("--api-key-file", type=Path)
    key_group.add_argument(
        "--api-key-env",
        metavar="NAME",
        help="read the key from this explicitly named environment variable",
    )
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--strict-tools", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def evaluate_critical_assertions(
    assertions: tuple[CriticalAssertion, ...],
    *,
    before: IntentState,
    resolved: ResolvedTurnIntent,
    shown_products: tuple[ShownProductView, ...],
    category_labels: dict[str, str] | None = None,
) -> tuple[AssertionOutcome, ...]:
    """Evaluate only retrieval-changing semantic predicates, never exact JSON."""

    labels = category_labels or {}
    return tuple(
        _evaluate_assertion(
            assertion,
            before=before,
            resolved=resolved,
            shown_products=shown_products,
            category_labels=labels,
        )
        for assertion in assertions
    )


def _evaluate_assertion(
    assertion: CriticalAssertion,
    *,
    before: IntentState,
    resolved: ResolvedTurnIntent,
    shown_products: tuple[ShownProductView, ...],
    category_labels: dict[str, str],
) -> AssertionOutcome:
    kind = assertion.kind
    final = resolved.final_intent
    if kind in {"goal_contains", "goal_not_contains"}:
        assert assertion.text is not None
        contains = _contains(final.goal, assertion.text)
        passed = contains if kind == "goal_contains" else not contains
        return AssertionOutcome(
            kind, passed, "goal substring matched" if passed else "goal mismatch"
        )
    if kind == "goal_contains_any":
        matched = tuple(text for text in assertion.texts if _contains(final.goal, text))
        return AssertionOutcome(
            kind,
            bool(matched),
            "goal alternative matched" if matched else "goal alternatives mismatched",
        )
    if kind in {"preference", "preference_absent"}:
        matches = tuple(
            item
            for item in final.preferences
            if _preference_matches(item, assertion=assertion, category_labels=category_labels)
        )
        passed = bool(matches) if kind == "preference" else not matches
        reason = f"matched {len(matches)} preference(s)"
        return AssertionOutcome(kind, passed, reason)
    if kind == "facet_absent":
        assert assertion.facet is not None
        present = any(_facet_alias(item.facet) == assertion.facet for item in final.preferences)
        return AssertionOutcome(kind, not present, "facet present" if present else "facet absent")
    if kind == "dont_care":
        assert assertion.facet is not None and assertion.present is not None
        canonical = _canonical_facet(assertion.facet)
        present = canonical in final.dont_care_facets
        passed = present is assertion.present
        return AssertionOutcome(kind, passed, f"dont-care presence was {present}")
    if kind == "state_unchanged":
        passed = resolved.update is None and final == before
        return AssertionOutcome(kind, passed, "state unchanged" if passed else "state changed")
    if kind == "clarification":
        assert assertion.needed is not None
        actual = resolved.clarification.needed
        return AssertionOutcome(
            kind, actual is assertion.needed, f"clarification needed was {actual}"
        )
    if kind == "directive":
        assert assertion.name is not None
        actual: str | bool
        if assertion.name == "diversity":
            actual = resolved.directives.diversity.value
        else:
            actual = cast(bool, getattr(resolved.directives, assertion.name))
        return AssertionOutcome(kind, actual == assertion.value, f"directive value was {actual!r}")
    if kind == "feedback":
        assert assertion.target_index is not None and assertion.signal is not None
        if assertion.target_index >= len(shown_products):
            return AssertionOutcome(kind, False, "target_index is outside shown products")
        expected_ids = shown_products[assertion.target_index].product_ids
        matches = tuple(
            feedback
            for feedback in resolved.feedback
            if feedback.signal.value == assertion.signal
            and all(product_id in feedback.product_ids for product_id in expected_ids)
        )
        return AssertionOutcome(kind, bool(matches), f"matched {len(matches)} feedback item(s)")
    raise AssertionError(f"unhandled assertion kind: {kind}")


def _preference_matches(
    preference: Preference,
    *,
    assertion: CriticalAssertion,
    category_labels: dict[str, str],
) -> bool:
    actual_facet = _facet_alias(preference.facet)
    if assertion.facet is not None and actual_facet != assertion.facet:
        return False
    relation = (
        preference.operator.value
        if preference.operator is not None
        else f"semantic_{preference.semantic_polarity.value}"
    )
    if assertion.relation is not None and relation not in _RELATIONS[assertion.relation]:
        return False
    if assertion.strength is not None and preference.commitment.value != assertion.strength:
        return False
    if assertion.values:
        actual_values = _assertable_values(preference, category_labels=category_labels)
        if not all(_normalized(value) in actual_values for value in assertion.values):
            return False
    if assertion.text_contains is not None:
        corpus = " ".join(
            value
            for value in (
                preference.semantic_text,
                preference.evidence_text,
                *_display_values(preference, category_labels=category_labels),
            )
            if value
        )
        if not _contains(corpus, assertion.text_contains):
            return False
    return True


def _assertable_values(
    preference: Preference,
    *,
    category_labels: dict[str, str],
) -> frozenset[str]:
    if preference.facet == "price" and type(preference.value) in (int, float):
        dollars = Decimal(str(preference.value)) / 100
        text = format(dollars, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return frozenset({_normalized(text)})
    return frozenset(_normalized(value) for value in _display_values(preference, category_labels))


def _display_values(
    preference: Preference,
    category_labels: dict[str, str],
) -> tuple[str, ...]:
    value = preference.value
    raw = value if type(value) is tuple else (() if value is None else (value,))
    output: list[str] = []
    for item in raw:
        text = str(item)
        output.append(text)
        if preference.facet == SYSTEM_PRODUCT_CATEGORY_FACET_ID and text in category_labels:
            output.append(category_labels[text])
    return tuple(output)


def _contains(value: str | None, expected: str) -> bool:
    return _normalized(expected) in _normalized(value or "")


def _normalized(value: object) -> str:
    return " ".join(str(value).strip().casefold().split())


def _canonical_facet(facet: str) -> str:
    return SYSTEM_PRODUCT_CATEGORY_FACET_ID if facet == "category" else facet


def _facet_alias(facet: str | None) -> str | None:
    return "category" if facet == SYSTEM_PRODUCT_CATEGORY_FACET_ID else facet


def _selected_suites(args: argparse.Namespace) -> tuple[tuple[Path, PromptSuite], ...]:
    requested: list[tuple[Path, str]] = []
    if args.cohort in {"natural", "all"}:
        requested.append((args.natural_suite, "natural"))
    if args.cohort in {"simulator", "all"}:
        requested.append((args.simulator_suite, "simulator"))
    suites: list[tuple[Path, PromptSuite]] = []
    for path, expected_cohort in requested:
        suite = load_prompt_suite(path)
        if suite.cohort != expected_cohort:
            raise ValueError(f"{path} is a {suite.cohort} suite, expected {expected_cohort}")
        suites.append((path, suite))
    return tuple(suites)


def _select_conversations(
    suite: PromptSuite,
    *,
    tier: str,
    limit: int | None,
    conversation_ids: frozenset[str],
) -> tuple[PromptConversation, ...]:
    selected = tuple(
        conversation
        for conversation in suite.conversations
        if (tier == "full" or conversation.tier == "smoke")
        and (not conversation_ids or conversation.identifier in conversation_ids)
    )
    return selected if limit is None else selected[:limit]


def _suite_inventory(
    path: Path,
    suite: PromptSuite,
    selected: tuple[PromptConversation, ...],
) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "schema": suite.schema,
        "suite_id": suite.suite_id,
        "cohort": suite.cohort,
        "conversation_count": len(suite.conversations),
        "turn_count": sum(len(item.turns) for item in suite.conversations),
        "selected_conversation_count": len(selected),
        "selected_turn_count": sum(len(item.turns) for item in selected),
    }


def validate_suites(
    suites: tuple[tuple[Path, PromptSuite], ...],
    *,
    tier: str,
    limit: int | None,
    model_config: DeepSeekConfig,
    conversation_ids: frozenset[str] = frozenset(),
) -> dict[str, object]:
    """Return an offline validation report without loading a release or provider."""

    selected = tuple(
        (
            path,
            suite,
            _select_conversations(
                suite,
                tier=tier,
                limit=limit,
                conversation_ids=conversation_ids,
            ),
        )
        for path, suite in suites
    )
    return {
        "schema": REPORT_SCHEMA,
        "mode": "validate_only",
        "selection": {
            "tier": tier,
            "limit_per_cohort": limit,
            "conversation_ids": sorted(conversation_ids),
        },
        "suites": [_suite_inventory(path, suite, items) for path, suite, items in selected],
        "protocol": _protocol_identity(model_config),
        "status": "valid",
    }


def replay_suites(
    suites: tuple[tuple[Path, PromptSuite], ...],
    *,
    tier: str,
    limit: int | None,
    release_path: Path,
    api_key: str,
    model_config: DeepSeekConfig,
    conversation_ids: frozenset[str] = frozenset(),
) -> dict[str, object]:
    """Replay selected conversations through the real provider and trusted materializer."""

    release = load_catalog_semantic_release(release_path)
    gateway = CatalogSemanticGateway(release)
    materializer = IntentMaterializer(gateway=gateway, grounder=release.grounder)
    service = QueryUnderstandingService(
        provider=DeepSeekProvider(api_key=api_key, config=model_config),
        materializer=materializer,
    )
    category_options = category_options_from_registry(release.category_registry)
    allowed_dont_care_facets = tuple(
        spec.id for spec in gateway.registry if spec.id != SYSTEM_PRODUCT_CATEGORY_FACET_ID
    )
    category_labels = {scope.id: scope.label for scope in release.category_registry.scopes}

    selected = tuple(
        (
            path,
            suite,
            _select_conversations(
                suite,
                tier=tier,
                limit=limit,
                conversation_ids=conversation_ids,
            ),
        )
        for path, suite in suites
    )
    records: list[dict[str, object]] = []
    facet_counts: Counter[str] = Counter()
    strength_counts: Counter[str] = Counter()
    selected_turn_count = sum(
        len(items_turn.turns) for _, _, items in selected for items_turn in items
    )
    contract_successes = 0
    attempted_turn_count = 0
    repair_turn_count = 0
    repair_exhausted_count = 0
    critical_total = sum(
        len(turn.critical_assertions)
        for _, suite, conversations in selected
        if suite.cohort == "natural"
        for conversation in conversations
        for turn in conversation.turns
    )
    critical_passes = 0
    critical_evaluated_total = 0
    critical_turn_total = sum(
        bool(turn.critical_assertions)
        for _, suite, conversations in selected
        if suite.cohort == "natural"
        for conversation in conversations
        for turn in conversation.turns
    )
    critical_turn_passes = 0
    critical_evaluated_turn_total = 0
    token_totals: Counter[str] = Counter()

    for _, suite, conversations in selected:
        for conversation in conversations:
            current = IntentState(
                goal=conversation.initial_goal,
                preferences=(),
                dont_care_facets=frozenset(),
                version=0,
            )
            blocked = False
            for turn in conversation.turns:
                if blocked:
                    records.append(
                        _skipped_record(suite=suite, conversation=conversation, turn=turn)
                    )
                    continue
                shown = _shown_product_views(suite, conversation, turn)
                request = build_reconcile_request(
                    turn=turn.turn,
                    latest_utterance=turn.user_message,
                    current_intent=current,
                    category_options=category_options,
                    shown_products=shown,
                    last_assistant_message=turn.last_assistant_message,
                    last_question=turn.last_question,
                    allowed_dont_care_facets=allowed_dont_care_facets,
                )
                before = current
                attempted_turn_count += 1
                started = time.perf_counter()
                try:
                    resolved = service.resolve(current=current, request=request)
                except Exception as error:  # The report must preserve all remaining cases.
                    if (
                        isinstance(error, QueryUnderstandingError)
                        and error.code.value == "repair_exhausted"
                    ):
                        repair_turn_count += 1
                        repair_exhausted_count += 1
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    records.append(
                        _error_record(
                            suite=suite,
                            conversation=conversation,
                            turn=turn,
                            elapsed_ms=elapsed_ms,
                            error=error,
                        )
                    )
                    blocked = True
                    continue
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                contract_successes += 1
                attempts = len(resolved.trace.attempts)
                repair_turn_count += int(attempts > 1)
                for trace in resolved.trace.attempts:
                    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        value = getattr(trace, field)
                        if value is not None:
                            token_totals[field] += value
                outcomes = (
                    evaluate_critical_assertions(
                        turn.critical_assertions,
                        before=before,
                        resolved=resolved,
                        shown_products=shown,
                        category_labels=category_labels,
                    )
                    if suite.cohort == "natural"
                    else ()
                )
                if outcomes:
                    critical_evaluated_turn_total += 1
                    critical_evaluated_total += len(outcomes)
                    turn_passed = all(item.passed for item in outcomes)
                    critical_turn_passes += int(turn_passed)
                    critical_passes += sum(item.passed for item in outcomes)
                for preference in resolved.final_intent.preferences:
                    facet_counts[_facet_alias(preference.facet) or "semantic"] += 1
                    strength_counts[preference.commitment.value] += 1
                records.append(
                    _success_record(
                        suite=suite,
                        conversation=conversation,
                        turn=turn,
                        resolved=resolved,
                        outcomes=outcomes,
                        elapsed_ms=elapsed_ms,
                    )
                )
                current = resolved.final_intent

    return {
        "schema": REPORT_SCHEMA,
        "mode": "live_replay",
        "selection": {
            "tier": tier,
            "limit_per_cohort": limit,
            "conversation_ids": sorted(conversation_ids),
        },
        "suites": [
            _suite_inventory(path, suite, conversations) for path, suite, conversations in selected
        ],
        "release": {"path": str(release_path), "release_id": release.release_id},
        "protocol": _protocol_identity(model_config),
        "summary": {
            "selected_turn_count": selected_turn_count,
            "contract_success_count": contract_successes,
            "contract_success_rate": _ratio(contract_successes, selected_turn_count),
            "critical_assertion_count": critical_total,
            "critical_assertion_pass_count": critical_passes,
            "critical_semantic_pass_rate": _ratio(critical_passes, critical_total),
            "evaluated_critical_assertion_count": critical_evaluated_total,
            "conditional_critical_semantic_pass_rate": _ratio(
                critical_passes, critical_evaluated_total
            ),
            "critical_turn_count": critical_turn_total,
            "critical_turn_pass_count": critical_turn_passes,
            "critical_turn_pass_rate": _ratio(critical_turn_passes, critical_turn_total),
            "evaluated_critical_turn_count": critical_evaluated_turn_total,
            "conditional_critical_turn_pass_rate": _ratio(
                critical_turn_passes, critical_evaluated_turn_total
            ),
            "attempted_turn_count": attempted_turn_count,
            "repair_count": repair_turn_count,
            "repair_exhausted_count": repair_exhausted_count,
            "repair_rate": _ratio(repair_turn_count, attempted_turn_count),
            "successful_turn_token_usage": dict(sorted(token_totals.items())),
            "final_state_observation_facet_counts": dict(sorted(facet_counts.items())),
            "final_state_observation_strength_counts": dict(sorted(strength_counts.items())),
        },
        "turns": records,
    }


def _shown_product_views(
    suite: PromptSuite,
    conversation: PromptConversation,
    turn: PromptTurn,
) -> tuple[ShownProductView, ...]:
    return tuple(
        ShownProductView(
            ref=f"product_{index}",
            product_ids=(
                f"fixture/{suite.suite_id}/{conversation.identifier}/{turn.turn}/{index}",
            ),
            label=product.label,
        )
        for index, product in enumerate(turn.shown_products)
    )


def _base_record(
    *,
    suite: PromptSuite,
    conversation: PromptConversation,
    turn: PromptTurn,
) -> dict[str, object]:
    scenario = (
        None if conversation.provenance is None else conversation.provenance.get("scenario_type")
    )
    return {
        "suite_id": suite.suite_id,
        "cohort": suite.cohort,
        "conversation_id": conversation.identifier,
        "tier": conversation.tier,
        "turn": turn.turn,
        "scenario_type": scenario,
        "response_shape": turn.response_shape,
    }


def _success_record(
    *,
    suite: PromptSuite,
    conversation: PromptConversation,
    turn: PromptTurn,
    resolved: ResolvedTurnIntent,
    outcomes: tuple[AssertionOutcome, ...],
    elapsed_ms: float,
) -> dict[str, object]:
    traces = resolved.trace.attempts
    return {
        **_base_record(suite=suite, conversation=conversation, turn=turn),
        "status": "success",
        "contract_success": True,
        "latency_ms": round(elapsed_ms, 3),
        "attempt_count": len(traces),
        "attempts": [
            {
                "model": trace.model,
                "prompt_tokens": trace.prompt_tokens,
                "completion_tokens": trace.completion_tokens,
                "total_tokens": trace.total_tokens,
            }
            for trace in traces
        ],
        "critical_semantic_pass": None if not outcomes else all(item.passed for item in outcomes),
        "assertions": [
            {"kind": item.kind, "pass": item.passed, "reason": item.reason} for item in outcomes
        ],
        "final_intent": _intent_payload(resolved.final_intent),
        "directives": {
            "diversity": resolved.directives.diversity.value,
            "comparison_requested": resolved.directives.comparison_requested,
            "explanation_requested": resolved.directives.explanation_requested,
        },
        "clarification_needed": resolved.clarification.needed,
        "feedback_count": len(resolved.feedback),
        "semantic_fallback_facets": list(resolved.trace.semantic_fallback_facets),
        "ignored_dont_care_facets": list(resolved.trace.ignored_dont_care_facets),
        "error": None,
    }


def _error_record(
    *,
    suite: PromptSuite,
    conversation: PromptConversation,
    turn: PromptTurn,
    elapsed_ms: float,
    error: Exception,
) -> dict[str, object]:
    if isinstance(error, QueryUnderstandingError):
        code = error.code.value
        error_payload: dict[str, object] = {
            "code": code,
            "path": list(error.path),
            "details": {key: value for key, value in error.details},
        }
        detail_map = dict(error.details)
        attempt_count = detail_map.get("attempt_count")
    else:
        code = type(error).__name__
        error_payload = {"code": code}
        attempt_count = None
    return {
        **_base_record(suite=suite, conversation=conversation, turn=turn),
        "status": "error",
        "contract_success": False,
        "latency_ms": round(elapsed_ms, 3),
        "attempt_count": attempt_count,
        "critical_semantic_pass": False if suite.cohort == "natural" else None,
        "error": error_payload,
    }


def _skipped_record(
    *,
    suite: PromptSuite,
    conversation: PromptConversation,
    turn: PromptTurn,
) -> dict[str, object]:
    return {
        **_base_record(suite=suite, conversation=conversation, turn=turn),
        "status": "skipped_after_failure",
        "contract_success": False,
        "critical_semantic_pass": False if suite.cohort == "natural" else None,
        "error": None,
    }


def _intent_payload(intent: IntentState) -> dict[str, object]:
    return {
        "goal": intent.goal,
        "version": intent.version,
        "preferences": [
            {
                "facet": _facet_alias(item.facet),
                "relation": (
                    item.operator.value
                    if item.operator is not None
                    else f"semantic_{item.semantic_polarity.value}"
                ),
                "value": list(item.value) if type(item.value) is tuple else item.value,
                "semantic_text": item.semantic_text,
                "strength": item.commitment.value,
                "source": item.source.value,
            }
            for item in intent.preferences
        ],
        "dont_care_facets": sorted(_facet_alias(item) for item in intent.dont_care_facets),
    }


def _protocol_identity(config: DeepSeekConfig) -> dict[str, object]:
    tool = reconcile_session_intent_tool(strict=config.strict_tools)
    safe_model_config = {
        "model": config.model,
        "base_url": config.base_url,
        "timeout_seconds": config.timeout_seconds,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "strict_tools": config.strict_tools,
        "disable_thinking": config.disable_thinking,
    }
    return {
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": _sha256_bytes(SYSTEM_PROMPT.encode("utf-8")),
        "tool_schema_sha256": _sha256_bytes(_canonical_json(tool)),
        "model_config": safe_model_config,
        "model_config_sha256": _sha256_bytes(_canonical_json(safe_model_config)),
    }


def _api_key(args: argparse.Namespace) -> str:
    if args.api_key_file is not None:
        try:
            key = args.api_key_file.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise ValueError("API key file is unavailable") from error
    elif args.api_key_env is not None:
        key = os.environ.get(args.api_key_env, "").strip()
    else:
        raise ValueError("live replay requires --api-key-file or --api-key-env")
    if not key or any(character.isspace() for character in key):
        raise ValueError("explicit API key is empty or malformed")
    return key


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 6)


def _write_report(report: dict[str, object], output: Path | None) -> None:
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if output is None:
        sys.stdout.write(payload)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        config = DeepSeekConfig(
            model=args.model,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
            max_tokens=args.max_tokens,
            strict_tools=args.strict_tools,
            disable_thinking=not args.enable_thinking,
        )
        suites = _selected_suites(args)
        conversation_ids = frozenset(args.conversation_id)
        available_ids = {
            conversation.identifier for _, suite in suites for conversation in suite.conversations
        }
        missing_ids = sorted(conversation_ids.difference(available_ids))
        if missing_ids:
            raise ValueError(f"unknown conversation IDs: {', '.join(missing_ids)}")
        if args.validate_only:
            report = validate_suites(
                suites,
                tier=args.tier,
                limit=args.limit,
                model_config=config,
                conversation_ids=conversation_ids,
            )
        else:
            report = replay_suites(
                suites,
                tier=args.tier,
                limit=args.limit,
                release_path=args.release,
                api_key=_api_key(args),
                model_config=config,
                conversation_ids=conversation_ids,
            )
    except (CatalogSemanticError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    _write_report(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
