"""Tests for the stable session-context error boundary."""

from __future__ import annotations

import re
from dataclasses import FrozenInstanceError

import pytest

from shopping_copilot.session_context.errors import ErrorCode, SessionContextError

EXPECTED_ERROR_CODES = {
    "SESSION_NOT_FOUND": "session_not_found",
    "SESSION_ALREADY_EXISTS": "session_already_exists",
    "INVALID_SESSION_ID": "invalid_session_id",
    "INVALID_PROFILE": "invalid_profile",
    "TURN_OUT_OF_ORDER": "turn_out_of_order",
    "SESSION_COMMIT_CONFLICT": "session_commit_conflict",
    "INVALID_SESSION_TRANSITION": "invalid_session_transition",
    "EMPTY_BATCH": "empty_batch",
    "STALE_BASE_VERSION": "stale_base_version",
    "INVALID_OPERATION_ORDER": "invalid_operation_order",
    "MULTIPLE_GOAL_SWITCH": "multiple_goal_switch",
    "UNKNOWN_PREFERENCE_ID": "unknown_preference_id",
    "INVALID_CARRY_ID": "invalid_carry_id",
    "EMPTY_REPLACEMENT": "empty_replacement",
    "FACET_MISMATCH": "facet_mismatch",
    "NON_CANONICAL_VALUE": "non_canonical_value",
    "UNKNOWN_FACET": "unknown_facet",
    "INVALID_GOAL": "invalid_goal",
    "INVALID_REPRESENTATION": "invalid_representation",
    "INVALID_OPERATOR_VALUE": "invalid_operator_value",
    "INVALID_OPERATOR_FOR_FACET": "invalid_operator_for_facet",
    "INVALID_COMMITMENT_FOR_SOURCE": "invalid_commitment_for_source",
    "INVALID_SOURCE_TURN": "invalid_source_turn",
    "INVALID_CONFIDENCE": "invalid_confidence",
    "DUPLICATE_PREFERENCE_ID": "duplicate_preference_id",
    "PREFERENCE_ID_CONFLICT": "preference_id_conflict",
    "DUPLICATE_PREFERENCE_SEMANTICS": "duplicate_preference_semantics",
    "MULTIPLE_POSITIVE_SELECTOR": "multiple_positive_selector",
    "MULTIPLE_NEGATIVE_SELECTOR": "multiple_negative_selector",
    "EMPTY_NUMERIC_INTERSECTION": "empty_numeric_intersection",
    "EMPTY_CATEGORICAL_DOMAIN": "empty_categorical_domain",
    "DONT_CARE_CONFLICT": "dont_care_conflict",
    "INVALID_TURN_RECORD": "invalid_turn_record",
    "INVALID_TURN_SEQUENCE": "invalid_turn_sequence",
    "INVALID_QUESTION_FIELDS": "invalid_question_fields",
    "INVALID_FEEDBACK": "invalid_feedback",
    "INVALID_FEEDBACK_REFERENCE": "invalid_feedback_reference",
    "TURN_RECORD_VERSION_MISMATCH": "turn_record_version_mismatch",
    "STALE_SEARCH_BELIEF": "stale_search_belief",
    "CERTAINTY_QUALITY_MISMATCH": "certainty_quality_mismatch",
    "INVALID_PROBE_EVIDENCE": "invalid_probe_evidence",
    "INVALID_MASS_DISTRIBUTION": "invalid_mass_distribution",
    "DUPLICATE_MODE_ID": "duplicate_mode_id",
    "DUPLICATE_FACET_STATS": "duplicate_facet_stats",
    "DUPLICATE_FACET_VALUE": "duplicate_facet_value",
    "UNKNOWN_SCHEMA_VERSION": "unknown_schema_version",
    "INVALID_SNAPSHOT": "invalid_snapshot",
    "UNKNOWN_FIELD": "unknown_field",
}


def test_error_code_vocabulary_exactly_matches_the_frozen_contract() -> None:
    actual = {member.name: member.value for member in ErrorCode}

    assert actual == EXPECTED_ERROR_CODES
    assert len(actual.values()) == len(set(actual.values()))
    assert all(re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", value) for value in actual.values())


@pytest.mark.parametrize("code", tuple(ErrorCode), ids=lambda code: code.name.lower())
def test_error_codes_round_trip_through_their_wire_values(code: ErrorCode) -> None:
    assert ErrorCode(code.value) is code


def test_session_context_error_canonicalizes_immutable_diagnostics() -> None:
    error = SessionContextError(
        code=ErrorCode.FACET_MISMATCH,
        path=("operations", 2, "facet"),
        operation_index=2,
        details=(("z_value", 3), ("facet", "color"), ("enabled", True)),
    )

    assert isinstance(error, ValueError)
    assert error.code is ErrorCode.FACET_MISMATCH
    assert error.path == ("operations", 2, "facet")
    assert error.operation_index == 2
    assert error.details == (("enabled", True), ("facet", "color"), ("z_value", 3))
    assert str(error) == "facet_mismatch at operations.2.facet"

    with pytest.raises(FrozenInstanceError):
        error.path = ()


def test_session_context_error_accepts_every_safe_scalar_detail_type() -> None:
    error = SessionContextError(
        code=ErrorCode.INVALID_PROFILE,
        details=(
            ("text", "value"),
            ("integer", 2),
            ("number", 2.5),
            ("flag", False),
        ),
    )

    assert dict(error.details) == {
        "flag": False,
        "integer": 2,
        "number": 2.5,
        "text": "value",
    }


@pytest.mark.parametrize(
    ("kwargs", "exception_type"),
    [
        ({"code": "invalid_profile"}, TypeError),
        ({"code": ErrorCode.INVALID_PROFILE, "operation_index": True}, TypeError),
        ({"code": ErrorCode.INVALID_PROFILE, "operation_index": -1}, ValueError),
        ({"code": ErrorCode.INVALID_PROFILE, "path": ("profile", None)}, TypeError),
        ({"code": ErrorCode.INVALID_PROFILE, "path": ("profile", False)}, TypeError),
        ({"code": ErrorCode.INVALID_PROFILE, "details": (["key", "value"],)}, TypeError),
        ({"code": ErrorCode.INVALID_PROFILE, "details": (("key",),)}, TypeError),
        ({"code": ErrorCode.INVALID_PROFILE, "details": ((1, "value"),)}, TypeError),
        ({"code": ErrorCode.INVALID_PROFILE, "details": (("key", None),)}, TypeError),
        ({"code": ErrorCode.INVALID_PROFILE, "details": (("key", ("value",)),)}, TypeError),
        ({"code": ErrorCode.INVALID_PROFILE, "details": (("key", float("nan")),)}, ValueError),
        ({"code": ErrorCode.INVALID_PROFILE, "details": (("key", float("inf")),)}, ValueError),
        (
            {
                "code": ErrorCode.INVALID_PROFILE,
                "details": (("duplicate", 1), ("duplicate", 2)),
            },
            ValueError,
        ),
    ],
)
def test_session_context_error_rejects_unsafe_diagnostics(
    kwargs: dict[str, object],
    exception_type: type[Exception],
) -> None:
    with pytest.raises(exception_type):
        SessionContextError(**kwargs)  # type: ignore[arg-type]
