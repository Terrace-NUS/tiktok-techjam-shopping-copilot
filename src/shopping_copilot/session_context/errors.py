"""Stable machine-readable domain failures."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from .models import ScalarValue

ErrorPathSegment: TypeAlias = str | int
ErrorDetail: TypeAlias = tuple[str, ScalarValue]


class ErrorCode(str, Enum):
    """Frozen v1 error-code vocabulary."""

    SESSION_NOT_FOUND = "session_not_found"
    SESSION_ALREADY_EXISTS = "session_already_exists"
    INVALID_SESSION_ID = "invalid_session_id"
    INVALID_PROFILE = "invalid_profile"
    TURN_OUT_OF_ORDER = "turn_out_of_order"
    SESSION_COMMIT_CONFLICT = "session_commit_conflict"
    INVALID_SESSION_TRANSITION = "invalid_session_transition"

    EMPTY_BATCH = "empty_batch"
    STALE_BASE_VERSION = "stale_base_version"
    INVALID_OPERATION_ORDER = "invalid_operation_order"
    MULTIPLE_GOAL_SWITCH = "multiple_goal_switch"
    UNKNOWN_PREFERENCE_ID = "unknown_preference_id"
    INVALID_CARRY_ID = "invalid_carry_id"
    EMPTY_REPLACEMENT = "empty_replacement"
    FACET_MISMATCH = "facet_mismatch"
    NON_CANONICAL_VALUE = "non_canonical_value"

    UNKNOWN_FACET = "unknown_facet"
    INVALID_GOAL = "invalid_goal"
    INVALID_REPRESENTATION = "invalid_representation"
    INVALID_OPERATOR_VALUE = "invalid_operator_value"
    INVALID_OPERATOR_FOR_FACET = "invalid_operator_for_facet"
    INVALID_COMMITMENT_FOR_SOURCE = "invalid_commitment_for_source"
    INVALID_SOURCE_TURN = "invalid_source_turn"
    INVALID_CONFIDENCE = "invalid_confidence"
    DUPLICATE_PREFERENCE_ID = "duplicate_preference_id"
    PREFERENCE_ID_CONFLICT = "preference_id_conflict"
    DUPLICATE_PREFERENCE_SEMANTICS = "duplicate_preference_semantics"
    MULTIPLE_POSITIVE_SELECTOR = "multiple_positive_selector"
    MULTIPLE_NEGATIVE_SELECTOR = "multiple_negative_selector"
    EMPTY_NUMERIC_INTERSECTION = "empty_numeric_intersection"
    EMPTY_CATEGORICAL_DOMAIN = "empty_categorical_domain"
    DONT_CARE_CONFLICT = "dont_care_conflict"

    INVALID_TURN_RECORD = "invalid_turn_record"
    INVALID_TURN_SEQUENCE = "invalid_turn_sequence"
    INVALID_QUESTION_FIELDS = "invalid_question_fields"
    INVALID_FEEDBACK = "invalid_feedback"
    INVALID_FEEDBACK_REFERENCE = "invalid_feedback_reference"
    TURN_RECORD_VERSION_MISMATCH = "turn_record_version_mismatch"
    STALE_SEARCH_BELIEF = "stale_search_belief"
    CERTAINTY_QUALITY_MISMATCH = "certainty_quality_mismatch"
    INVALID_PROBE_EVIDENCE = "invalid_probe_evidence"
    INVALID_MASS_DISTRIBUTION = "invalid_mass_distribution"
    DUPLICATE_MODE_ID = "duplicate_mode_id"
    DUPLICATE_FACET_STATS = "duplicate_facet_stats"
    DUPLICATE_FACET_VALUE = "duplicate_facet_value"

    UNKNOWN_SCHEMA_VERSION = "unknown_schema_version"
    INVALID_SNAPSHOT = "invalid_snapshot"
    UNKNOWN_FIELD = "unknown_field"


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionContextError(ValueError):
    """Domain exception with a stable code and immutable diagnostic context."""

    code: ErrorCode
    path: tuple[ErrorPathSegment, ...] = ()
    operation_index: int | None = None
    details: tuple[ErrorDetail, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.code, ErrorCode):
            raise TypeError("code must be an ErrorCode")
        if self.operation_index is not None:
            if type(self.operation_index) is not int:
                raise TypeError("operation_index must be an int or None")
            if self.operation_index < 0:
                raise ValueError("operation_index must be non-negative")

        canonical_path = tuple(self.path)
        for segment in canonical_path:
            if type(segment) not in (str, int):
                raise TypeError("error path segments must be strings or integers")

        canonical_details: list[ErrorDetail] = []
        for detail in self.details:
            if type(detail) is not tuple or len(detail) != 2:
                raise TypeError("each error detail must be a key/value tuple")
            key, value = detail
            if type(key) is not str:
                raise TypeError("error detail keys must be strings")
            if type(value) not in (str, int, float, bool):
                raise TypeError("error detail values must be JSON scalars")
            if type(value) is float and not math.isfinite(value):
                raise ValueError("error detail numbers must be finite")
            canonical_details.append((key, value))

        sorted_details = tuple(sorted(canonical_details, key=lambda item: item[0]))
        if len({key for key, _ in sorted_details}) != len(sorted_details):
            raise ValueError("error detail keys must be unique")

        object.__setattr__(self, "path", canonical_path)
        object.__setattr__(self, "details", sorted_details)
        ValueError.__init__(self, self._render_message())

    def _render_message(self) -> str:
        location = ".".join(str(segment) for segment in self.path)
        if location:
            return f"{self.code.value} at {location}"
        return self.code.value
