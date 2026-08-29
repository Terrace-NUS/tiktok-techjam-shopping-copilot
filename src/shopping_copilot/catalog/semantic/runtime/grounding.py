"""Release-bound deterministic grounding of structured Query Understanding candidates."""

from __future__ import annotations

from typing import cast

from shopping_copilot.session_context import FacetKind, FacetRegistry, Operator, SessionContextError
from shopping_copilot.session_context.models import PreferenceValue, ScalarValue
from shopping_copilot.session_context.registry import FacetSpec

from ..canonical import canonical_scalar_key
from ..category import CategoryRegistry
from ..facet.gate_b_models import EffectiveFacetCapabilitySet
from .build import project_session_facet_registry
from .capabilities import ExactCapabilityIndex
from .grounding_models import (
    ExtractedRuntimeValueCandidate,
    GroundedPredicate,
    GroundingDisposition,
    RuntimeValueGroundingResult,
)
from .models import (
    SYSTEM_PRODUCT_CATEGORY_FACET_ID,
    RuntimeFacetRegistryArtifact,
    RuntimeValueLexicon,
)


class RuntimeValueGrounder:
    """Pure CS5B service bound to one coherent set of runtime artifacts."""

    __slots__ = ("_capabilities", "_facet_registry", "_records", "_scope_ids")

    def __init__(
        self,
        *,
        runtime_registry: RuntimeFacetRegistryArtifact,
        runtime_lexicon: RuntimeValueLexicon,
        category_registry: CategoryRegistry,
        capabilities: EffectiveFacetCapabilitySet,
    ) -> None:
        self._facet_registry = project_session_facet_registry(
            runtime_registry=runtime_registry,
            runtime_lexicon=runtime_lexicon,
            category_registry=category_registry,
            capabilities=capabilities,
        )
        self._capabilities = ExactCapabilityIndex(capabilities)
        self._records = {item.facet_id: item for item in runtime_registry.entries}
        self._scope_ids = frozenset(item.id for item in category_registry.scopes)

    def ground(
        self,
        candidate: ExtractedRuntimeValueCandidate,
        *,
        final_category_scope_id: str | None,
    ) -> RuntimeValueGroundingResult:
        """Return a canonical trusted result without allocating IDs or changing session state."""

        if type(candidate) is not ExtractedRuntimeValueCandidate:
            raise TypeError("grounding requires an ExtractedRuntimeValueCandidate")
        if final_category_scope_id is not None and type(final_category_scope_id) is not str:
            raise TypeError("final_category_scope_id must be a string or None")

        facet_id = candidate.facet_id
        if facet_id is None or facet_id not in self._records:
            return _semantic_only(candidate, facet_id=None, reason_code="unknown_facet")
        if facet_id == SYSTEM_PRODUCT_CATEGORY_FACET_ID:
            return self._ground_reserved_category(candidate)
        if final_category_scope_id not in self._scope_ids:
            return _semantic_only(
                candidate,
                facet_id=facet_id,
                reason_code="unregistered_category_scope",
            )
        permissions = self._capabilities.lookup(facet_id, final_category_scope_id)
        if not permissions.intent_committable:
            return _semantic_only(
                candidate,
                facet_id=facet_id,
                reason_code="facet_not_committable",
            )
        operator = _parse_operator(candidate.operator)
        spec = self._facet_registry.require(facet_id)
        if operator is None or not _operator_supported(spec, operator):
            return _semantic_only(
                candidate,
                facet_id=facet_id,
                reason_code="unsupported_operator",
            )
        return self._ground_ordinary_value(candidate, spec=spec, operator=operator)

    def _ground_ordinary_value(
        self,
        candidate: ExtractedRuntimeValueCandidate,
        *,
        spec: FacetSpec,
        operator: Operator,
    ) -> RuntimeValueGroundingResult:
        if candidate.alternative_values:
            normalized = _normalize_alternatives(spec, candidate.alternative_values)
            if normalized is None:
                return _semantic_only(
                    candidate,
                    facet_id=spec.id,
                    reason_code="unknown_value",
                )
            if len(normalized) >= 2:
                return _ambiguous(candidate, facet_id=spec.id, values=normalized)
            if not normalized:
                return _semantic_only(
                    candidate,
                    facet_id=spec.id,
                    reason_code="unknown_value",
                )
            raw_value: PreferenceValue = _operand_from_single_alternative(
                normalized[0],
                operator=operator,
            )
        else:
            if candidate.value is None:
                return _semantic_only(
                    candidate,
                    facet_id=spec.id,
                    reason_code="unknown_value",
                )
            raw_value = candidate.value
        normalized_value = _normalize_operand(
            self._facet_registry,
            spec=spec,
            operator=operator,
            value=raw_value,
        )
        if normalized_value is None:
            return _semantic_only(
                candidate,
                facet_id=spec.id,
                reason_code="unknown_value",
            )
        predicates = _ordinary_predicates(
            facet_id=spec.id,
            operator=operator,
            value=normalized_value,
        )
        return RuntimeValueGroundingResult(
            facet_id=spec.id,
            disposition=GroundingDisposition.GROUNDED,
            predicates=predicates,
            reason_code=None,
            candidate_values=(),
            semantic_text=None,
            semantic_polarity=None,
        )

    def _ground_reserved_category(
        self,
        candidate: ExtractedRuntimeValueCandidate,
    ) -> RuntimeValueGroundingResult:
        facet_id = SYSTEM_PRODUCT_CATEGORY_FACET_ID
        operator = _parse_operator(candidate.operator)
        if operator is not Operator.EQ:
            return _semantic_only(
                candidate,
                facet_id=facet_id,
                reason_code="unsupported_operator",
            )
        spec = self._facet_registry.require(facet_id)
        if candidate.alternative_values:
            normalized = _normalize_alternatives(spec, candidate.alternative_values)
            if normalized is None:
                return _semantic_only(
                    candidate,
                    facet_id=facet_id,
                    reason_code="unregistered_category_scope",
                )
            if len(normalized) >= 2:
                return _ambiguous(candidate, facet_id=facet_id, values=normalized)
            if not normalized:
                return _semantic_only(
                    candidate,
                    facet_id=facet_id,
                    reason_code="unknown_value",
                )
            raw_value: PreferenceValue = normalized[0]
        else:
            if candidate.value is None:
                return _semantic_only(
                    candidate,
                    facet_id=facet_id,
                    reason_code="unknown_value",
                )
            raw_value = candidate.value
        try:
            normalized_value = self._facet_registry.normalize_value(
                facet_id,
                Operator.EQ,
                raw_value,
            )
        except SessionContextError:
            return _semantic_only(
                candidate,
                facet_id=facet_id,
                reason_code="unregistered_category_scope",
            )
        if type(normalized_value) is not str:
            return _semantic_only(
                candidate,
                facet_id=facet_id,
                reason_code="unregistered_category_scope",
            )
        return RuntimeValueGroundingResult(
            facet_id=facet_id,
            disposition=GroundingDisposition.GROUNDED,
            predicates=(
                GroundedPredicate(
                    facet_id=facet_id,
                    operator=Operator.EQ,
                    value=normalized_value,
                ),
            ),
            reason_code=None,
            candidate_values=(),
            semantic_text=None,
            semantic_polarity=None,
        )


def _parse_operator(value: Operator | str | None) -> Operator | None:
    if type(value) is Operator:
        return value
    if type(value) is not str:
        return None
    try:
        return Operator(value)
    except ValueError:
        return None


def _operator_supported(spec: FacetSpec, operator: Operator) -> bool:
    return operator in spec.operators or (
        spec.kind is FacetKind.NUMERIC and operator is Operator.EQ
    )


def _normalize_alternatives(
    spec: FacetSpec,
    values: tuple[ScalarValue, ...],
) -> tuple[ScalarValue, ...] | None:
    normalized: list[ScalarValue] = []
    for value in values:
        try:
            item = spec.normalizer(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if type(item) not in (str, int, float, bool):
            return None
        if spec.kind is FacetKind.NUMERIC and type(item) not in (int, float):
            return None
        normalized.append(item)
    by_key = {canonical_scalar_key(item): item for item in normalized}
    return tuple(by_key[key] for key in sorted(by_key))


def _normalize_operand(
    registry: FacetRegistry,
    *,
    spec: FacetSpec,
    operator: Operator,
    value: PreferenceValue,
) -> PreferenceValue | None:
    if spec.kind is FacetKind.NUMERIC and operator is Operator.EQ:
        if type(value) is tuple:
            return None
        try:
            normalized = spec.normalizer(cast(ScalarValue, value))
        except (TypeError, ValueError, OverflowError):
            return None
        return normalized if type(normalized) in (int, float) else None
    try:
        return registry.normalize_value(spec.id, operator, value)
    except SessionContextError:
        return None


def _operand_from_single_alternative(
    value: ScalarValue,
    *,
    operator: Operator,
) -> PreferenceValue:
    return (value,) if operator in (Operator.IN, Operator.NOT_IN) else value


def _ordinary_predicates(
    *,
    facet_id: str,
    operator: Operator,
    value: PreferenceValue,
) -> tuple[GroundedPredicate, ...]:
    operators = (Operator.GE, Operator.LE) if operator is Operator.EQ else (operator,)
    predicates = tuple(
        GroundedPredicate(facet_id=facet_id, operator=item, value=value) for item in operators
    )
    return tuple(
        sorted(
            predicates,
            key=lambda item: (item.facet_id, item.operator.value, _canonical_value(item.value)),
        )
    )


def _canonical_value(value: PreferenceValue) -> bytes:
    from ..canonical import canonical_json_bytes

    return canonical_json_bytes(value)


def _semantic_only(
    candidate: ExtractedRuntimeValueCandidate,
    *,
    facet_id: str | None,
    reason_code: str,
) -> RuntimeValueGroundingResult:
    return RuntimeValueGroundingResult(
        facet_id=facet_id,
        disposition=GroundingDisposition.SEMANTIC_ONLY,
        predicates=(),
        reason_code=reason_code,
        candidate_values=(),
        semantic_text=candidate.semantic_text,
        semantic_polarity=candidate.semantic_polarity,
    )


def _ambiguous(
    candidate: ExtractedRuntimeValueCandidate,
    *,
    facet_id: str,
    values: tuple[ScalarValue, ...],
) -> RuntimeValueGroundingResult:
    return RuntimeValueGroundingResult(
        facet_id=facet_id,
        disposition=GroundingDisposition.AMBIGUOUS,
        predicates=(),
        reason_code="ambiguous_value",
        candidate_values=values,
        semantic_text=candidate.semantic_text,
        semantic_polarity=candidate.semantic_polarity,
    )
