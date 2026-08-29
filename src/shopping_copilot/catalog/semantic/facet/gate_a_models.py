"""Normative Gate-A DTOs and closed typed-value foundations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeAlias

from ..canonical import (
    IJSON_SAFE_INTEGER_MAX,
    canonical_json_bytes,
    content_id_for_value,
    validate_semantic_string,
)
from .models import SourceLocator

CATALOG_FACET_SCHEMA: Literal["shopping-copilot/catalog-facet-schema/v0"] = (
    "shopping-copilot/catalog-facet-schema/v0"
)
FACET_APPLICABILITY_SCHEMA: Literal["shopping-copilot/facet-applicability/v0"] = (
    "shopping-copilot/facet-applicability/v0"
)
FACET_SOURCE_BINDINGS_SCHEMA: Literal["shopping-copilot/facet-source-bindings/v0"] = (
    "shopping-copilot/facet-source-bindings/v0"
)
GATE_A_SELECTION_SCHEMA: Literal["shopping-copilot/gate-a-selection/v0"] = (
    "shopping-copilot/gate-a-selection/v0"
)
GATE_A_CANDIDATE_SCHEMA: Literal["shopping-copilot/gate-a-candidate/v0"] = (
    "shopping-copilot/gate-a-candidate/v0"
)
GATE_A_BUILDER_VERSION = "catalog_semantic_gate_a_v0"

_CONTENT_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SCOPE_ID_PATTERN = re.compile(r"^cs_[0-9a-f]{64}$")
_SEMANTIC_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


class FacetDataType(str, Enum):
    """Closed catalog facet value families."""

    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    TEXT = "text"


class ItemCardinality(str, Enum):
    """Number of atomic values one item may carry for a facet."""

    SINGLE = "single"
    MULTI = "multi"


class ValueCompleteness(str, Enum):
    """Whether structured categorical evidence is exhaustive."""

    COMPLETE = "complete"
    PARTIAL = "partial"


class GateADecision(str, Enum):
    """Human extraction decision recorded in source control."""

    EXTRACTION_APPROVED = "extraction_approved"


class EvidenceStatus(str, Enum):
    """Outcome of deterministic catalog value normalization."""

    VALID = "valid"
    EMPTY = "empty"
    INVALID = "invalid"


class PriceNormalizationLane(str, Enum):
    """Auditable reason for one price normalization result."""

    EMPTY = "empty"
    EXACT = "exact"
    LOWER_BOUND = "lower_bound"
    INVALID = "invalid"


class ProductFacetStatus(str, Enum):
    """Resolved product-facet truth state."""

    KNOWN = "known"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True, kw_only=True)
class CatalogFacetDefinition:
    """One approved catalog facet identity and value shape."""

    id: str
    name: str
    data_type: FacetDataType
    item_cardinality: ItemCardinality

    def __post_init__(self) -> None:
        _require_semantic_id(self.id, name="CatalogFacetDefinition.id")
        if self.id == "other":
            raise ValueError("facet ID 'other' is reserved by the official adapter")
        validate_semantic_string(self.name, name="CatalogFacetDefinition.name")
        if type(self.data_type) is not FacetDataType:
            raise TypeError("CatalogFacetDefinition.data_type is invalid")
        if type(self.item_cardinality) is not ItemCardinality:
            raise TypeError("CatalogFacetDefinition.item_cardinality is invalid")
        if (
            self.data_type is not FacetDataType.CATEGORICAL
            and self.item_cardinality is not ItemCardinality.SINGLE
        ):
            raise ValueError("boolean, numeric, and text facets must be SINGLE")


@dataclass(frozen=True, slots=True, kw_only=True)
class CatalogFacetSchema:
    """Published, ordered set of Gate-A-approved facet definitions."""

    schema: Literal["shopping-copilot/catalog-facet-schema/v0"]
    facets: tuple[CatalogFacetDefinition, ...]

    def __post_init__(self) -> None:
        if self.schema != CATALOG_FACET_SCHEMA:
            raise ValueError("CatalogFacetSchema.schema is invalid")
        _require_exact_tuple(self.facets, CatalogFacetDefinition, name="facets")
        _require_sorted_unique(
            tuple(item.id for item in self.facets),
            name="CatalogFacetSchema facet IDs",
            nonempty=True,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class FacetApplicability:
    """Reviewed category scopes where one facet is semantically meaningful."""

    facet_id: str
    category_scope_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_semantic_id(self.facet_id, name="FacetApplicability.facet_id")
        _require_identifier_tuple(
            self.category_scope_ids,
            pattern=_SCOPE_ID_PATTERN,
            name="FacetApplicability.category_scope_ids",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class FacetApplicabilitySet:
    """Category-pinned applicability rows for every approved facet."""

    schema: Literal["shopping-copilot/facet-applicability/v0"]
    category_registry_id: str
    facet_schema_id: str
    entries: tuple[FacetApplicability, ...]

    def __post_init__(self) -> None:
        if self.schema != FACET_APPLICABILITY_SCHEMA:
            raise ValueError("FacetApplicabilitySet.schema is invalid")
        _require_content_id(
            self.category_registry_id,
            name="FacetApplicabilitySet.category_registry_id",
        )
        _require_content_id(self.facet_schema_id, name="FacetApplicabilitySet.facet_schema_id")
        _require_exact_tuple(self.entries, FacetApplicability, name="entries")
        _require_sorted_unique(
            tuple(item.facet_id for item in self.entries),
            name="FacetApplicabilitySet facet IDs",
            nonempty=True,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class FacetSourceBinding:
    """Reviewed exact raw source and closed extraction implementations."""

    id: str
    facet_id: str
    source: SourceLocator
    applicable_category_scope_ids: tuple[str, ...]
    extractor_id: str
    catalog_value_normalizer_id: str
    priority: int
    completeness: ValueCompleteness
    resolver_id: str

    def __post_init__(self) -> None:
        _require_semantic_id(self.id, name="FacetSourceBinding.id")
        _require_semantic_id(self.facet_id, name="FacetSourceBinding.facet_id")
        if type(self.source) is not SourceLocator:
            raise TypeError("FacetSourceBinding.source is invalid")
        _require_identifier_tuple(
            self.applicable_category_scope_ids,
            pattern=_SCOPE_ID_PATTERN,
            name="FacetSourceBinding.applicable_category_scope_ids",
        )
        _require_semantic_id(self.extractor_id, name="FacetSourceBinding.extractor_id")
        _require_semantic_id(
            self.catalog_value_normalizer_id,
            name="FacetSourceBinding.catalog_value_normalizer_id",
        )
        _require_nonnegative_int(self.priority, name="FacetSourceBinding.priority")
        if type(self.completeness) is not ValueCompleteness:
            raise TypeError("FacetSourceBinding.completeness is invalid")
        _require_semantic_id(self.resolver_id, name="FacetSourceBinding.resolver_id")


@dataclass(frozen=True, slots=True, kw_only=True)
class FacetSourceBindingSet:
    """Reviewed bindings pinned to category, facet, and applicability content."""

    schema: Literal["shopping-copilot/facet-source-bindings/v0"]
    category_registry_id: str
    facet_schema_id: str
    facet_applicability_id: str
    bindings: tuple[FacetSourceBinding, ...]

    def __post_init__(self) -> None:
        if self.schema != FACET_SOURCE_BINDINGS_SCHEMA:
            raise ValueError("FacetSourceBindingSet.schema is invalid")
        _require_content_id(
            self.category_registry_id,
            name="FacetSourceBindingSet.category_registry_id",
        )
        _require_content_id(self.facet_schema_id, name="FacetSourceBindingSet.facet_schema_id")
        _require_content_id(
            self.facet_applicability_id,
            name="FacetSourceBindingSet.facet_applicability_id",
        )
        _require_exact_tuple(self.bindings, FacetSourceBinding, name="bindings")
        _require_sorted_unique(
            tuple(item.id for item in self.bindings),
            name="FacetSourceBindingSet binding IDs",
            nonempty=True,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PriceExtractionExpectation:
    """Human-approved result counts for the frozen first price lane."""

    product_count: int
    source_present_count: int
    source_missing_count: int
    valid_count: int
    empty_count: int
    invalid_count: int
    exact_interval_count: int
    lower_bound_interval_count: int
    zero_exact_count: int

    def __post_init__(self) -> None:
        _validate_price_counts(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class GateAFacetApproval:
    """One explicit human Gate-A approval and its frozen-data expectation."""

    decision: GateADecision
    definition: CatalogFacetDefinition
    applicability: FacetApplicability
    bindings: tuple[FacetSourceBinding, ...]
    extraction_expectation: PriceExtractionExpectation
    rationale: str

    def __post_init__(self) -> None:
        if self.decision is not GateADecision.EXTRACTION_APPROVED:
            raise ValueError("GateAFacetApproval must be extraction-approved")
        if type(self.definition) is not CatalogFacetDefinition:
            raise TypeError("GateAFacetApproval.definition is invalid")
        if type(self.applicability) is not FacetApplicability:
            raise TypeError("GateAFacetApproval.applicability is invalid")
        _require_exact_tuple(self.bindings, FacetSourceBinding, name="bindings")
        if not self.bindings:
            raise ValueError("GateAFacetApproval.bindings must be non-empty")
        if type(self.extraction_expectation) is not PriceExtractionExpectation:
            raise TypeError("GateAFacetApproval.extraction_expectation is invalid")
        validate_semantic_string(self.rationale, name="GateAFacetApproval.rationale")
        if self.applicability.facet_id != self.definition.id:
            raise ValueError("Gate-A applicability does not name its definition")
        if any(binding.facet_id != self.definition.id for binding in self.bindings):
            raise ValueError("Gate-A binding does not name its definition")
        binding_ids = tuple(binding.id for binding in self.bindings)
        _require_sorted_unique(
            binding_ids,
            name="GateAFacetApproval binding IDs",
            nonempty=True,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class GateASelection:
    """Source-controlled human decisions used to build normative Gate-A artifacts."""

    schema: Literal["shopping-copilot/gate-a-selection/v0"]
    catalog_id: str
    category_registry_id: str
    product_category_assignment_id: str
    source_profile_manifest_sha256: str
    builder_version: str
    approvals: tuple[GateAFacetApproval, ...]

    def __post_init__(self) -> None:
        if self.schema != GATE_A_SELECTION_SCHEMA:
            raise ValueError("GateASelection.schema is invalid")
        _require_content_id(self.catalog_id, name="GateASelection.catalog_id")
        _require_content_id(
            self.category_registry_id,
            name="GateASelection.category_registry_id",
        )
        _require_content_id(
            self.product_category_assignment_id,
            name="GateASelection.product_category_assignment_id",
        )
        _require_identifier(
            self.source_profile_manifest_sha256,
            pattern=_SHA256_PATTERN,
            name="GateASelection.source_profile_manifest_sha256",
        )
        if self.builder_version != GATE_A_BUILDER_VERSION:
            raise ValueError("GateASelection.builder_version is unsupported")
        _require_exact_tuple(self.approvals, GateAFacetApproval, name="approvals")
        _require_sorted_unique(
            tuple(item.definition.id for item in self.approvals),
            name="GateASelection facet IDs",
            nonempty=True,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CategoricalValue:
    """Canonical categorical facet value."""

    kind: Literal["categorical"]
    values: tuple[str, ...]
    completeness: ValueCompleteness

    def __post_init__(self) -> None:
        if self.kind != "categorical":
            raise ValueError("CategoricalValue.kind is invalid")
        if type(self.values) is not tuple or not self.values:
            raise ValueError("CategoricalValue.values must be a non-empty tuple")
        for value in self.values:
            validate_semantic_string(value, name="CategoricalValue.values")
        if self.values != tuple(sorted(set(self.values), key=canonical_json_bytes)):
            raise ValueError("CategoricalValue.values must be sorted and unique")
        if type(self.completeness) is not ValueCompleteness:
            raise TypeError("CategoricalValue.completeness is invalid")


@dataclass(frozen=True, slots=True, kw_only=True)
class BooleanValue:
    """Canonical boolean facet value."""

    kind: Literal["boolean"]
    value: bool

    def __post_init__(self) -> None:
        if self.kind != "boolean" or type(self.value) is not bool:
            raise ValueError("BooleanValue is invalid")


@dataclass(frozen=True, slots=True, kw_only=True)
class TextValue:
    """Canonical search-only text facet value."""

    kind: Literal["text"]
    value: str

    def __post_init__(self) -> None:
        if self.kind != "text":
            raise ValueError("TextValue.kind is invalid")
        validate_semantic_string(self.value, name="TextValue.value")


@dataclass(frozen=True, slots=True, kw_only=True)
class NumericValue:
    """Canonical numeric interval value."""

    kind: Literal["numeric"]
    lower: int | float | None
    lower_inclusive: bool
    upper: int | float | None
    upper_inclusive: bool
    unit: str

    def __post_init__(self) -> None:
        if self.kind != "numeric":
            raise ValueError("NumericValue.kind is invalid")
        _validate_numeric_endpoint(self.lower, name="NumericValue.lower")
        _validate_numeric_endpoint(self.upper, name="NumericValue.upper")
        if self.lower is None and self.lower_inclusive:
            raise ValueError("an absent lower endpoint cannot be inclusive")
        if self.upper is None and self.upper_inclusive:
            raise ValueError("an absent upper endpoint cannot be inclusive")
        if self.lower is None and self.upper is None:
            raise ValueError("NumericValue requires at least one endpoint")
        if self.lower is not None and self.upper is not None:
            if self.lower > self.upper:
                raise ValueError("NumericValue endpoints are reversed")
            if self.lower == self.upper and not (self.lower_inclusive and self.upper_inclusive):
                raise ValueError("equal NumericValue endpoints must both be inclusive")
        validate_semantic_string(self.unit, name="NumericValue.unit")


ResolvedFacetValue: TypeAlias = CategoricalValue | BooleanValue | TextValue | NumericValue


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceExtraction:
    """Internal exact-source extraction result before catalog normalization."""

    present: bool
    raw_value: object

    def __post_init__(self) -> None:
        if type(self.present) is not bool:
            raise TypeError("SourceExtraction.present must be boolean")


@dataclass(frozen=True, slots=True, kw_only=True)
class PriceNormalizationResult:
    """Typed price outcome before evidence identity is assigned in CS3."""

    status: EvidenceStatus
    lane: PriceNormalizationLane
    value: NumericValue | None

    def __post_init__(self) -> None:
        if type(self.status) is not EvidenceStatus:
            raise TypeError("PriceNormalizationResult.status is invalid")
        if type(self.lane) is not PriceNormalizationLane:
            raise TypeError("PriceNormalizationResult.lane is invalid")
        if self.status is EvidenceStatus.VALID:
            if self.lane not in (
                PriceNormalizationLane.EXACT,
                PriceNormalizationLane.LOWER_BOUND,
            ):
                raise ValueError("valid price result has an invalid lane")
            if type(self.value) is not NumericValue:
                raise TypeError("valid price result requires NumericValue")
        elif self.value is not None:
            raise ValueError("non-valid price result cannot carry a value")
        elif self.status is EvidenceStatus.EMPTY and self.lane is not PriceNormalizationLane.EMPTY:
            raise ValueError("empty price result has an invalid lane")
        elif (
            self.status is EvidenceStatus.INVALID
            and self.lane is not PriceNormalizationLane.INVALID
        ):
            raise ValueError("invalid price result has an invalid lane")


@dataclass(frozen=True, slots=True, kw_only=True)
class PriorityExactResolution:
    """Result of the closed same-priority exact resolver."""

    status: ProductFacetStatus
    value: ResolvedFacetValue | None

    def __post_init__(self) -> None:
        if self.status is ProductFacetStatus.KNOWN:
            if self.value is None:
                raise ValueError("KNOWN resolution requires a value")
        elif self.status is ProductFacetStatus.CONFLICT:
            if self.value is not None:
                raise ValueError("CONFLICT resolution cannot carry a value")
        else:
            raise ValueError("priority exact resolver returns only KNOWN or CONFLICT")


@dataclass(frozen=True, slots=True, kw_only=True)
class PriceExtractionAudit:
    """Rebuilt counts proving the approved price parser against frozen data."""

    facet_id: str
    binding_id: str
    product_count: int
    source_present_count: int
    source_missing_count: int
    valid_count: int
    empty_count: int
    invalid_count: int
    exact_interval_count: int
    lower_bound_interval_count: int
    zero_exact_count: int

    def __post_init__(self) -> None:
        _require_semantic_id(self.facet_id, name="PriceExtractionAudit.facet_id")
        _require_semantic_id(self.binding_id, name="PriceExtractionAudit.binding_id")
        _validate_price_counts(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class GateACandidateBuild:
    """Complete normative CS2 candidate derived from reviewed Gate-A input."""

    schema: Literal["shopping-copilot/gate-a-candidate/v0"]
    catalog_id: str
    category_registry_id: str
    product_category_assignment_id: str
    source_profile_manifest_sha256: str
    builder_version: str
    selection: GateASelection
    facet_schema: CatalogFacetSchema
    applicability: FacetApplicabilitySet
    bindings: FacetSourceBindingSet
    price_audits: tuple[PriceExtractionAudit, ...]

    def __post_init__(self) -> None:
        if self.schema != GATE_A_CANDIDATE_SCHEMA:
            raise ValueError("GateACandidateBuild.schema is invalid")
        _require_content_id(self.catalog_id, name="GateACandidateBuild.catalog_id")
        _require_content_id(
            self.category_registry_id,
            name="GateACandidateBuild.category_registry_id",
        )
        _require_content_id(
            self.product_category_assignment_id,
            name="GateACandidateBuild.product_category_assignment_id",
        )
        _require_identifier(
            self.source_profile_manifest_sha256,
            pattern=_SHA256_PATTERN,
            name="GateACandidateBuild.source_profile_manifest_sha256",
        )
        if self.builder_version != GATE_A_BUILDER_VERSION:
            raise ValueError("GateACandidateBuild.builder_version is unsupported")
        if type(self.selection) is not GateASelection:
            raise TypeError("GateACandidateBuild.selection is invalid")
        if type(self.facet_schema) is not CatalogFacetSchema:
            raise TypeError("GateACandidateBuild.facet_schema is invalid")
        if type(self.applicability) is not FacetApplicabilitySet:
            raise TypeError("GateACandidateBuild.applicability is invalid")
        if type(self.bindings) is not FacetSourceBindingSet:
            raise TypeError("GateACandidateBuild.bindings is invalid")
        _require_exact_tuple(self.price_audits, PriceExtractionAudit, name="price_audits")
        if (
            self.selection.catalog_id != self.catalog_id
            or self.selection.category_registry_id != self.category_registry_id
            or self.selection.product_category_assignment_id != self.product_category_assignment_id
            or self.selection.source_profile_manifest_sha256 != self.source_profile_manifest_sha256
            or self.selection.builder_version != self.builder_version
        ):
            raise ValueError("GateACandidateBuild differs from its reviewed selection")
        expected_definitions = tuple(item.definition for item in self.selection.approvals)
        expected_applicability = tuple(item.applicability for item in self.selection.approvals)
        expected_bindings = tuple(
            binding for approval in self.selection.approvals for binding in approval.bindings
        )
        if self.facet_schema.facets != expected_definitions:
            raise ValueError("Gate-A facet schema differs from reviewed approvals")
        if self.applicability.entries != expected_applicability:
            raise ValueError("Gate-A applicability differs from reviewed approvals")
        if self.bindings.bindings != expected_bindings:
            raise ValueError("Gate-A bindings differ from reviewed approvals")
        if self.applicability.category_registry_id != self.category_registry_id:
            raise ValueError("Gate-A applicability CategoryRegistry pin is stale")
        expected_facet_schema_id = content_id_for_value(self.facet_schema)
        if self.applicability.facet_schema_id != expected_facet_schema_id:
            raise ValueError("Gate-A applicability facet schema pin is invalid")
        if self.bindings.category_registry_id != self.category_registry_id:
            raise ValueError("Gate-A bindings CategoryRegistry pin is stale")
        if self.bindings.facet_schema_id != expected_facet_schema_id:
            raise ValueError("Gate-A bindings facet schema pin is invalid")
        if self.bindings.facet_applicability_id != content_id_for_value(self.applicability):
            raise ValueError("Gate-A bindings applicability pin is invalid")
        audit_keys = tuple((item.facet_id, item.binding_id) for item in self.price_audits)
        binding_keys = tuple((item.facet_id, item.id) for item in self.bindings.bindings)
        if audit_keys != binding_keys:
            raise ValueError("Gate-A extraction audits do not cover the exact binding set")


def _validate_price_counts(value: object) -> None:
    field_names = (
        "product_count",
        "source_present_count",
        "source_missing_count",
        "valid_count",
        "empty_count",
        "invalid_count",
        "exact_interval_count",
        "lower_bound_interval_count",
        "zero_exact_count",
    )
    counts = {name: getattr(value, name) for name in field_names}
    for name, count in counts.items():
        _require_nonnegative_int(count, name=f"{type(value).__name__}.{name}")
    if counts["source_present_count"] + counts["source_missing_count"] != counts["product_count"]:
        raise ValueError("price source presence counts are inconsistent")
    if (
        counts["valid_count"] + counts["empty_count"] + counts["invalid_count"]
        != counts["source_present_count"]
    ):
        raise ValueError("price normalization counts are inconsistent")
    if (
        counts["exact_interval_count"] + counts["lower_bound_interval_count"]
        != counts["valid_count"]
    ):
        raise ValueError("price interval counts are inconsistent")
    if counts["zero_exact_count"] > counts["exact_interval_count"]:
        raise ValueError("zero exact count exceeds exact intervals")


def _validate_numeric_endpoint(value: object, *, name: str) -> None:
    if value is None:
        return
    if type(value) is int:
        if not -IJSON_SAFE_INTEGER_MAX <= value <= IJSON_SAFE_INTEGER_MAX:
            raise ValueError(f"{name} is outside the I-JSON safe range")
        return
    if type(value) is float:
        canonical_json_bytes(value)
        return
    raise TypeError(f"{name} must be int, float, or None")


def _require_content_id(value: object, *, name: str) -> None:
    _require_identifier(value, pattern=_CONTENT_ID_PATTERN, name=name)


def _require_semantic_id(value: object, *, name: str) -> None:
    _require_identifier(value, pattern=_SEMANTIC_ID_PATTERN, name=name)


def _require_identifier(value: object, *, pattern: re.Pattern[str], name: str) -> None:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise ValueError(f"{name} is invalid")


def _require_identifier_tuple(
    values: tuple[str, ...],
    *,
    pattern: re.Pattern[str],
    name: str,
) -> None:
    if type(values) is not tuple or not values:
        raise ValueError(f"{name} must be a non-empty tuple")
    for value in values:
        _require_identifier(value, pattern=pattern, name=name)
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be sorted and unique")


def _require_sorted_unique(
    values: tuple[str, ...],
    *,
    name: str,
    nonempty: bool,
) -> None:
    if type(values) is not tuple or (nonempty and not values):
        raise ValueError(f"{name} must be a {'non-empty ' if nonempty else ''}tuple")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be sorted and unique")


def _require_exact_tuple(values: object, expected_type: type[object], *, name: str) -> None:
    if type(values) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    if any(type(item) is not expected_type for item in values):
        raise TypeError(f"{name} contains an invalid item")


def _require_nonnegative_int(value: object, *, name: str) -> None:
    if type(value) is not int or not 0 <= value <= IJSON_SAFE_INTEGER_MAX:
        raise ValueError(f"{name} must be a non-negative I-JSON integer")
