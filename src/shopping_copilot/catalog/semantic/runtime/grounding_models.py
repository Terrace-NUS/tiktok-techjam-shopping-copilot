"""Immutable input and output DTOs for deterministic runtime grounding."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum

from shopping_copilot.session_context.models import (
    Operator,
    PreferenceValue,
    ScalarValue,
    SemanticPolarity,
)

from ..canonical import canonical_json_bytes, canonical_scalar_key, validate_semantic_string
from .models import SYSTEM_PRODUCT_CATEGORY_FACET_ID

_FACET_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_REASON_CODE_PATTERN = _FACET_ID_PATTERN


class GroundingDisposition(str, Enum):
    """Closed outcome family for one extracted facet candidate."""

    GROUNDED = "grounded"
    SEMANTIC_ONLY = "semantic_only"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True, kw_only=True)
class ExtractedRuntimeValueCandidate:
    """Small, untrusted Query Understanding handoff consumed by CS5B.

    ``value`` represents one proposed operator operand. ``alternative_values``
    represents competing scalar interpretations of that operand; it is not an
    ``IN`` value. Query Understanding keeps language and currency parsing on
    its side of this boundary.
    """

    facet_id: str | None
    operator: Operator | str | None
    value: PreferenceValue | None
    alternative_values: tuple[ScalarValue, ...]
    semantic_text: str
    semantic_polarity: SemanticPolarity

    def __post_init__(self) -> None:
        if self.facet_id is not None and type(self.facet_id) is not str:
            raise TypeError("ExtractedRuntimeValueCandidate.facet_id must be a string or None")
        if self.operator is not None and type(self.operator) not in (str, Operator):
            raise TypeError("ExtractedRuntimeValueCandidate.operator is invalid")
        if self.value is not None:
            _require_preference_value(
                self.value,
                name="ExtractedRuntimeValueCandidate.value",
                semantic=False,
                allow_empty_tuple=True,
            )
        if type(self.alternative_values) is not tuple or any(
            not _is_input_scalar(item) for item in self.alternative_values
        ):
            raise TypeError(
                "ExtractedRuntimeValueCandidate.alternative_values must be a scalar tuple"
            )
        if self.value is not None and self.alternative_values:
            raise ValueError("value and alternative_values are mutually exclusive")
        validate_semantic_string(
            self.semantic_text,
            name="ExtractedRuntimeValueCandidate.semantic_text",
        )
        if type(self.semantic_polarity) is not SemanticPolarity:
            raise TypeError("ExtractedRuntimeValueCandidate.semantic_polarity is invalid")


@dataclass(frozen=True, slots=True, kw_only=True)
class GroundedPredicate:
    """One trusted, normalized atomic structured predicate."""

    facet_id: str
    operator: Operator
    value: PreferenceValue

    def __post_init__(self) -> None:
        if type(self.facet_id) is not str or _FACET_ID_PATTERN.fullmatch(self.facet_id) is None:
            raise ValueError("GroundedPredicate.facet_id is invalid")
        if type(self.operator) is not Operator:
            raise TypeError("GroundedPredicate.operator is invalid")
        _require_preference_value(
            self.value,
            name="GroundedPredicate.value",
            semantic=True,
            allow_empty_tuple=False,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeValueGroundingResult:
    """Contract result for deterministic CS5B value grounding."""

    facet_id: str | None
    disposition: GroundingDisposition
    predicates: tuple[GroundedPredicate, ...]
    reason_code: str | None
    candidate_values: tuple[ScalarValue, ...]
    semantic_text: str | None
    semantic_polarity: SemanticPolarity | None

    def __post_init__(self) -> None:
        _validate_common_result_fields(self)
        if self.disposition is GroundingDisposition.GROUNDED:
            _validate_grounded_result(self)
        elif self.disposition is GroundingDisposition.SEMANTIC_ONLY:
            _validate_semantic_only_result(self)
        elif self.disposition is GroundingDisposition.AMBIGUOUS:
            _validate_ambiguous_result(self)
        else:  # pragma: no cover - the exact enum check above makes this unreachable
            raise TypeError("RuntimeValueGroundingResult.disposition is invalid")


def _validate_common_result_fields(result: RuntimeValueGroundingResult) -> None:
    if result.facet_id is not None and (
        type(result.facet_id) is not str or _FACET_ID_PATTERN.fullmatch(result.facet_id) is None
    ):
        raise ValueError("RuntimeValueGroundingResult.facet_id is invalid")
    if type(result.disposition) is not GroundingDisposition:
        raise TypeError("RuntimeValueGroundingResult.disposition is invalid")
    if type(result.predicates) is not tuple or any(
        type(item) is not GroundedPredicate for item in result.predicates
    ):
        raise TypeError("RuntimeValueGroundingResult.predicates is invalid")
    predicate_keys = tuple(_predicate_key(item) for item in result.predicates)
    if predicate_keys != tuple(sorted(set(predicate_keys))):
        raise ValueError("grounded predicates must be sorted and duplicate-free")
    if type(result.candidate_values) is not tuple or any(
        not _is_semantic_scalar(item) for item in result.candidate_values
    ):
        raise TypeError("RuntimeValueGroundingResult.candidate_values is invalid")
    candidate_keys = tuple(canonical_scalar_key(item) for item in result.candidate_values)
    if candidate_keys != tuple(sorted(set(candidate_keys))):
        raise ValueError("grounding candidates must be canonically sorted and duplicate-free")
    if (result.semantic_text is None) != (result.semantic_polarity is None):
        raise ValueError("semantic text and polarity must be both present or both absent")
    if result.semantic_text is not None:
        validate_semantic_string(
            result.semantic_text,
            name="RuntimeValueGroundingResult.semantic_text",
        )
    if (
        result.semantic_polarity is not None
        and type(result.semantic_polarity) is not SemanticPolarity
    ):
        raise TypeError("RuntimeValueGroundingResult.semantic_polarity is invalid")
    if result.reason_code is not None and (
        type(result.reason_code) is not str
        or _REASON_CODE_PATTERN.fullmatch(result.reason_code) is None
    ):
        raise ValueError("RuntimeValueGroundingResult.reason_code is invalid")


def _validate_grounded_result(result: RuntimeValueGroundingResult) -> None:
    if result.facet_id is None or not result.predicates:
        raise ValueError("GROUNDED requires a facet and at least one predicate")
    if result.reason_code is not None or result.candidate_values:
        raise ValueError("GROUNDED cannot contain a reason or candidate values")
    if any(item.facet_id != result.facet_id for item in result.predicates):
        raise ValueError("GROUNDED predicates must use the result facet")
    if result.facet_id == SYSTEM_PRODUCT_CATEGORY_FACET_ID and (
        len(result.predicates) != 1 or result.predicates[0].operator is not Operator.EQ
    ):
        raise ValueError("reserved category grounding requires exactly one EQ predicate")


def _validate_semantic_only_result(result: RuntimeValueGroundingResult) -> None:
    if result.predicates or result.candidate_values:
        raise ValueError("SEMANTIC_ONLY cannot contain predicates or candidate values")
    if result.reason_code is None or result.semantic_text is None:
        raise ValueError("SEMANTIC_ONLY requires a reason and semantic representation")
    if result.reason_code == "unknown_facet":
        if result.facet_id is not None:
            raise ValueError("unknown_facet must not expose a facet ID")
    elif result.facet_id is None:
        raise ValueError("recognized SEMANTIC_ONLY results must retain their facet ID")


def _validate_ambiguous_result(result: RuntimeValueGroundingResult) -> None:
    if result.facet_id is None or result.predicates:
        raise ValueError("AMBIGUOUS requires one facet and no predicates")
    if result.reason_code != "ambiguous_value" or len(result.candidate_values) < 2:
        raise ValueError("AMBIGUOUS requires at least two canonical candidates")
    if result.semantic_text is None:
        raise ValueError("AMBIGUOUS requires a semantic representation")


def _predicate_key(predicate: GroundedPredicate) -> tuple[str, str, bytes]:
    return predicate.facet_id, predicate.operator.value, canonical_json_bytes(predicate.value)


def _require_preference_value(
    value: PreferenceValue,
    *,
    name: str,
    semantic: bool,
    allow_empty_tuple: bool,
) -> None:
    if type(value) is tuple:
        if not value and not allow_empty_tuple:
            raise ValueError(f"{name} must not be empty")
        if any(
            not (_is_semantic_scalar(item) if semantic else _is_input_scalar(item))
            for item in value
        ):
            raise TypeError(f"{name} contains a non-scalar value")
    elif not (_is_semantic_scalar(value) if semantic else _is_input_scalar(value)):
        raise TypeError(f"{name} must be a scalar or scalar tuple")
    if semantic:
        canonical_json_bytes(value)


def _is_input_scalar(value: object) -> bool:
    return type(value) in (str, int, float, bool) and not (
        type(value) is float and not math.isfinite(value)
    )


def _is_semantic_scalar(value: object) -> bool:
    if not _is_input_scalar(value):
        return False
    if type(value) is str:
        try:
            validate_semantic_string(value, name="scalar")
        except (TypeError, ValueError):
            return False
    if type(value) is float and value == 0.0 and math.copysign(1.0, value) < 0.0:
        return False
    try:
        canonical_json_bytes(value)
    except (TypeError, ValueError):
        return False
    return True
