"""Immutable CS3 evidence, resolved-index, and catalog-statistics DTOs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ..canonical import IJSON_SAFE_INTEGER_MAX, content_id_for_value, validate_semantic_string
from .gate_a_models import EvidenceStatus, ProductFacetStatus, ResolvedFacetValue

FACET_EVIDENCE_STORE_SCHEMA: Literal["shopping-copilot/facet-evidence-store/v0"] = (
    "shopping-copilot/facet-evidence-store/v0"
)
PRODUCT_FACET_INDEX_SCHEMA: Literal["shopping-copilot/product-facet-index/v0"] = (
    "shopping-copilot/product-facet-index/v0"
)
CATALOG_FACET_STATS_SCHEMA: Literal["shopping-copilot/catalog-facet-stats/v0"] = (
    "shopping-copilot/catalog-facet-stats/v0"
)
RESOLUTION_CANDIDATE_SCHEMA: Literal["shopping-copilot/resolution-candidate/v0"] = (
    "shopping-copilot/resolution-candidate/v0"
)
CATALOG_READ_ONLY_AUDIT_SCHEMA: Literal["shopping-copilot/catalog-read-only-audit/v0"] = (
    "shopping-copilot/catalog-read-only-audit/v0"
)
RESOLUTION_POLICY_ID: Literal["structured_resolution_v1"] = "structured_resolution_v1"

_CONTENT_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_EVIDENCE_ID_PATTERN = re.compile(r"^ev_[0-9a-f]{64}$")
_SCOPE_ID_PATTERN = re.compile(r"^cs_[0-9a-f]{64}$")
_SEMANTIC_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


@dataclass(frozen=True, slots=True, kw_only=True)
class FacetValueEvidence:
    """One auditable structured extraction result for a product and binding."""

    id: str
    parent_asin: str
    facet_id: str
    binding_id: str
    status: EvidenceStatus
    raw_value_json: str
    canonical_value: ResolvedFacetValue | None

    def __post_init__(self) -> None:
        _require_identifier(self.id, pattern=_EVIDENCE_ID_PATTERN, name="FacetValueEvidence.id")
        validate_semantic_string(self.parent_asin, name="FacetValueEvidence.parent_asin")
        if self.parent_asin != self.parent_asin.strip():
            raise ValueError("FacetValueEvidence.parent_asin must be trimmed")
        _require_identifier(
            self.facet_id,
            pattern=_SEMANTIC_ID_PATTERN,
            name="FacetValueEvidence.facet_id",
        )
        _require_identifier(
            self.binding_id,
            pattern=_SEMANTIC_ID_PATTERN,
            name="FacetValueEvidence.binding_id",
        )
        if type(self.status) is not EvidenceStatus:
            raise TypeError("FacetValueEvidence.status is invalid")
        if type(self.raw_value_json) is not str or not self.raw_value_json:
            raise ValueError("FacetValueEvidence.raw_value_json must be non-empty")
        if self.status is EvidenceStatus.VALID:
            if self.canonical_value is None:
                raise ValueError("VALID evidence requires a canonical value")
        elif self.canonical_value is not None:
            raise ValueError("EMPTY and INVALID evidence cannot carry a canonical value")


@dataclass(frozen=True, slots=True, kw_only=True)
class FacetEvidenceStore:
    """Complete audit evidence pinned to one catalog-semantic input set."""

    schema: Literal["shopping-copilot/facet-evidence-store/v0"]
    catalog_id: str
    product_category_assignment_id: str
    facet_applicability_id: str
    facet_source_bindings_id: str
    resolution_policy_id: Literal["structured_resolution_v1"]
    evidence: tuple[FacetValueEvidence, ...]

    def __post_init__(self) -> None:
        if self.schema != FACET_EVIDENCE_STORE_SCHEMA:
            raise ValueError("FacetEvidenceStore.schema is invalid")
        _require_content_id(self.catalog_id, name="FacetEvidenceStore.catalog_id")
        _require_content_id(
            self.product_category_assignment_id,
            name="FacetEvidenceStore.product_category_assignment_id",
        )
        _require_content_id(
            self.facet_applicability_id,
            name="FacetEvidenceStore.facet_applicability_id",
        )
        _require_content_id(
            self.facet_source_bindings_id,
            name="FacetEvidenceStore.facet_source_bindings_id",
        )
        if self.resolution_policy_id != RESOLUTION_POLICY_ID:
            raise ValueError("FacetEvidenceStore.resolution_policy_id is invalid")
        _require_exact_tuple(self.evidence, FacetValueEvidence, name="FacetEvidenceStore.evidence")
        keys = tuple((item.parent_asin, item.binding_id) for item in self.evidence)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("FacetEvidenceStore.evidence must be sorted and unique by key")
        ids = tuple(item.id for item in self.evidence)
        if len(ids) != len(set(ids)):
            raise ValueError("FacetEvidenceStore evidence IDs must be unique")


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedProductFacetValue:
    """One resolved product-facet result under the pinned structured policy."""

    parent_asin: str
    facet_id: str
    status: ProductFacetStatus
    value: ResolvedFacetValue | None
    evidence_ids: tuple[str, ...]
    resolution_policy_id: Literal["structured_resolution_v1"]

    def __post_init__(self) -> None:
        validate_semantic_string(self.parent_asin, name="ResolvedProductFacetValue.parent_asin")
        if self.parent_asin != self.parent_asin.strip():
            raise ValueError("ResolvedProductFacetValue.parent_asin must be trimmed")
        _require_identifier(
            self.facet_id,
            pattern=_SEMANTIC_ID_PATTERN,
            name="ResolvedProductFacetValue.facet_id",
        )
        if type(self.status) is not ProductFacetStatus:
            raise TypeError("ResolvedProductFacetValue.status is invalid")
        if self.resolution_policy_id != RESOLUTION_POLICY_ID:
            raise ValueError("ResolvedProductFacetValue.resolution_policy_id is invalid")
        if type(self.evidence_ids) is not tuple:
            raise TypeError("ResolvedProductFacetValue.evidence_ids must be a tuple")
        for evidence_id in self.evidence_ids:
            _require_identifier(
                evidence_id,
                pattern=_EVIDENCE_ID_PATTERN,
                name="ResolvedProductFacetValue.evidence_ids",
            )
        if self.evidence_ids != tuple(sorted(set(self.evidence_ids))):
            raise ValueError("ResolvedProductFacetValue.evidence_ids must be sorted and unique")
        if self.status is ProductFacetStatus.KNOWN:
            if self.value is None or not self.evidence_ids:
                raise ValueError("KNOWN product-facet values require value and evidence")
        elif self.status is ProductFacetStatus.CONFLICT:
            if self.value is not None or not self.evidence_ids:
                raise ValueError("CONFLICT product-facet values require evidence and no value")
        elif self.value is not None or self.evidence_ids:
            raise ValueError("UNKNOWN and NOT_APPLICABLE cannot carry value or evidence")


@dataclass(frozen=True, slots=True, kw_only=True)
class ProductFacetIndex:
    """Sparse physical product-facet index containing KNOWN and CONFLICT rows."""

    schema: Literal["shopping-copilot/product-facet-index/v0"]
    catalog_id: str
    product_category_assignment_id: str
    facet_applicability_id: str
    facet_source_bindings_id: str
    facet_evidence_store_id: str
    resolution_policy_id: Literal["structured_resolution_v1"]
    entries: tuple[ResolvedProductFacetValue, ...]

    def __post_init__(self) -> None:
        if self.schema != PRODUCT_FACET_INDEX_SCHEMA:
            raise ValueError("ProductFacetIndex.schema is invalid")
        for name in (
            "catalog_id",
            "product_category_assignment_id",
            "facet_applicability_id",
            "facet_source_bindings_id",
            "facet_evidence_store_id",
        ):
            _require_content_id(getattr(self, name), name=f"ProductFacetIndex.{name}")
        if self.resolution_policy_id != RESOLUTION_POLICY_ID:
            raise ValueError("ProductFacetIndex.resolution_policy_id is invalid")
        _require_exact_tuple(
            self.entries, ResolvedProductFacetValue, name="ProductFacetIndex.entries"
        )
        keys = tuple((item.parent_asin, item.facet_id) for item in self.entries)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("ProductFacetIndex.entries must be sorted and unique by key")
        if any(
            item.status not in (ProductFacetStatus.KNOWN, ProductFacetStatus.CONFLICT)
            for item in self.entries
        ):
            raise ValueError("sparse ProductFacetIndex may store only KNOWN and CONFLICT")


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedValueCount:
    """Count of one complete canonical KNOWN value payload."""

    canonical_value_json: str
    product_count: int

    def __post_init__(self) -> None:
        if type(self.canonical_value_json) is not str or not self.canonical_value_json:
            raise ValueError("ResolvedValueCount.canonical_value_json must be non-empty")
        _require_positive_int(self.product_count, name="ResolvedValueCount.product_count")


@dataclass(frozen=True, slots=True, kw_only=True)
class FacetScopeCatalogStats:
    """Resolved status and value distribution for one facet and category scope."""

    facet_id: str
    category_scope_id: str
    scope_product_count: int
    known_count: int
    unknown_count: int
    conflict_count: int
    not_applicable_count: int
    known_value_counts: tuple[ResolvedValueCount, ...]

    def __post_init__(self) -> None:
        _require_identifier(
            self.facet_id,
            pattern=_SEMANTIC_ID_PATTERN,
            name="FacetScopeCatalogStats.facet_id",
        )
        _require_identifier(
            self.category_scope_id,
            pattern=_SCOPE_ID_PATTERN,
            name="FacetScopeCatalogStats.category_scope_id",
        )
        for name in (
            "scope_product_count",
            "known_count",
            "unknown_count",
            "conflict_count",
            "not_applicable_count",
        ):
            _require_nonnegative_int(getattr(self, name), name=f"FacetScopeCatalogStats.{name}")
        if (
            self.known_count + self.unknown_count + self.conflict_count + self.not_applicable_count
            != self.scope_product_count
        ):
            raise ValueError("FacetScopeCatalogStats status counts do not conserve products")
        _require_exact_tuple(
            self.known_value_counts,
            ResolvedValueCount,
            name="FacetScopeCatalogStats.known_value_counts",
        )
        if sum(item.product_count for item in self.known_value_counts) != self.known_count:
            raise ValueError("FacetScopeCatalogStats known value counts do not sum to KNOWN")


@dataclass(frozen=True, slots=True, kw_only=True)
class CatalogFacetStatsArtifact:
    """Complete deterministic Gate-B statistics derived from one resolved index."""

    schema: Literal["shopping-copilot/catalog-facet-stats/v0"]
    catalog_id: str
    category_registry_id: str
    product_category_assignment_id: str
    facet_schema_id: str
    facet_applicability_id: str
    product_facet_index_id: str
    resolution_policy_id: Literal["structured_resolution_v1"]
    rows: tuple[FacetScopeCatalogStats, ...]

    def __post_init__(self) -> None:
        if self.schema != CATALOG_FACET_STATS_SCHEMA:
            raise ValueError("CatalogFacetStatsArtifact.schema is invalid")
        for name in (
            "catalog_id",
            "category_registry_id",
            "product_category_assignment_id",
            "facet_schema_id",
            "facet_applicability_id",
            "product_facet_index_id",
        ):
            _require_content_id(getattr(self, name), name=f"CatalogFacetStatsArtifact.{name}")
        if self.resolution_policy_id != RESOLUTION_POLICY_ID:
            raise ValueError("CatalogFacetStatsArtifact.resolution_policy_id is invalid")
        _require_exact_tuple(
            self.rows, FacetScopeCatalogStats, name="CatalogFacetStatsArtifact.rows"
        )
        keys = tuple((item.facet_id, item.category_scope_id) for item in self.rows)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("CatalogFacetStatsArtifact.rows must be sorted and unique")


@dataclass(frozen=True, slots=True, kw_only=True)
class CatalogReadOnlyAudit:
    """Deterministic proof that CS3 observed identical catalog bytes before and after staging."""

    schema: Literal["shopping-copilot/catalog-read-only-audit/v0"]
    catalog_id_before: str
    catalog_id_after_staging: str
    byte_size_before: int
    byte_size_after_staging: int
    unchanged: bool
    output_is_separate: bool

    def __post_init__(self) -> None:
        if self.schema != CATALOG_READ_ONLY_AUDIT_SCHEMA:
            raise ValueError("CatalogReadOnlyAudit.schema is invalid")
        _require_content_id(
            self.catalog_id_before,
            name="CatalogReadOnlyAudit.catalog_id_before",
        )
        _require_content_id(
            self.catalog_id_after_staging,
            name="CatalogReadOnlyAudit.catalog_id_after_staging",
        )
        _require_positive_int(self.byte_size_before, name="CatalogReadOnlyAudit.byte_size_before")
        _require_positive_int(
            self.byte_size_after_staging,
            name="CatalogReadOnlyAudit.byte_size_after_staging",
        )
        if type(self.unchanged) is not bool or type(self.output_is_separate) is not bool:
            raise TypeError("CatalogReadOnlyAudit flags must be boolean")
        observed_unchanged = (
            self.catalog_id_before == self.catalog_id_after_staging
            and self.byte_size_before == self.byte_size_after_staging
        )
        if not self.unchanged or not observed_unchanged:
            raise ValueError("CatalogReadOnlyAudit requires unchanged catalog bytes")
        if not self.output_is_separate:
            raise ValueError("CatalogReadOnlyAudit requires a separate output target")


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolutionCandidateBuild:
    """Complete CS3 candidate before Gate-B review or runtime promotion."""

    schema: Literal["shopping-copilot/resolution-candidate/v0"]
    builder_version: str
    category_registry_id: str
    facet_schema_id: str
    gate_a_selection_id: str
    evidence_store: FacetEvidenceStore
    product_facet_index: ProductFacetIndex
    stats: CatalogFacetStatsArtifact

    def __post_init__(self) -> None:
        if self.schema != RESOLUTION_CANDIDATE_SCHEMA:
            raise ValueError("ResolutionCandidateBuild.schema is invalid")
        _require_identifier(
            self.builder_version,
            pattern=_SEMANTIC_ID_PATTERN,
            name="ResolutionCandidateBuild.builder_version",
        )
        for name in ("category_registry_id", "facet_schema_id", "gate_a_selection_id"):
            _require_content_id(getattr(self, name), name=f"ResolutionCandidateBuild.{name}")
        if type(self.evidence_store) is not FacetEvidenceStore:
            raise TypeError("ResolutionCandidateBuild.evidence_store is invalid")
        if type(self.product_facet_index) is not ProductFacetIndex:
            raise TypeError("ResolutionCandidateBuild.product_facet_index is invalid")
        if type(self.stats) is not CatalogFacetStatsArtifact:
            raise TypeError("ResolutionCandidateBuild.stats is invalid")
        if self.product_facet_index.catalog_id != self.evidence_store.catalog_id:
            raise ValueError("CS3 index catalog pin differs from evidence store")
        if (
            self.product_facet_index.product_category_assignment_id
            != self.evidence_store.product_category_assignment_id
            or self.product_facet_index.facet_applicability_id
            != self.evidence_store.facet_applicability_id
            or self.product_facet_index.facet_source_bindings_id
            != self.evidence_store.facet_source_bindings_id
            or self.product_facet_index.facet_evidence_store_id
            != content_id_for_value(self.evidence_store)
        ):
            raise ValueError("CS3 ProductFacetIndex pins differ from evidence store")
        if (
            self.stats.catalog_id != self.evidence_store.catalog_id
            or self.stats.category_registry_id != self.category_registry_id
            or self.stats.product_category_assignment_id
            != self.evidence_store.product_category_assignment_id
            or self.stats.facet_schema_id != self.facet_schema_id
            or self.stats.facet_applicability_id != self.evidence_store.facet_applicability_id
            or self.stats.product_facet_index_id != content_id_for_value(self.product_facet_index)
        ):
            raise ValueError("CS3 stats pins differ from the resolved index")


def _require_content_id(value: object, *, name: str) -> None:
    _require_identifier(value, pattern=_CONTENT_ID_PATTERN, name=name)


def _require_identifier(value: object, *, pattern: re.Pattern[str], name: str) -> None:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise ValueError(f"{name} is invalid")


def _require_exact_tuple(values: object, expected_type: type[object], *, name: str) -> None:
    if type(values) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    if any(type(item) is not expected_type for item in values):
        raise TypeError(f"{name} contains an invalid item")


def _require_nonnegative_int(value: object, *, name: str) -> None:
    if type(value) is not int or not 0 <= value <= IJSON_SAFE_INTEGER_MAX:
        raise ValueError(f"{name} must be a non-negative I-JSON integer")


def _require_positive_int(value: object, *, name: str) -> None:
    _require_nonnegative_int(value, name=name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
