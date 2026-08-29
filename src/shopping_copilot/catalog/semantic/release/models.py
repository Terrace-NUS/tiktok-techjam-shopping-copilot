"""Immutable CS6 release manifest and reviewed-config contract DTOs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, TypeAlias, cast

from ..canonical import IJSON_SAFE_INTEGER_MAX, validate_semantic_string
from ..category import CategoryRegistry, CategoryScope, ProductCategoryAssignmentSet
from ..facet import (
    CatalogFacetDefinition,
    CatalogFacetSchema,
    CatalogFacetStatsArtifact,
    EffectiveFacetCapability,
    EffectiveFacetCapabilitySet,
    FacetApplicability,
    FacetApplicabilitySet,
    FacetEvidenceStore,
    FacetSourceBinding,
    FacetSourceBindingSet,
    ProductFacetIndex,
)
from ..facet.resolution_models import RESOLUTION_POLICY_ID
from ..runtime import (
    RuntimeFacetRegistryArtifact,
    RuntimeValueGrounder,
    RuntimeValueLexicon,
)

REVIEWED_SEMANTIC_CONFIG_SCHEMA: Literal["shopping-copilot/reviewed-semantic-config/v0"] = (
    "shopping-copilot/reviewed-semantic-config/v0"
)
CATALOG_SEMANTIC_RELEASE_SCHEMA: Literal["shopping-copilot/catalog-semantic-release/v0"] = (
    "shopping-copilot/catalog-semantic-release/v0"
)
CATALOG_SEMANTIC_RELEASE_BUILDER_VERSION = "catalog_semantic_release_v0"

ArtifactKind: TypeAlias = Literal[
    "catalog",
    "category_registry",
    "product_category_assignment",
    "facet_schema",
    "facet_applicability",
    "facet_source_bindings",
    "facet_evidence_store",
    "product_facet_index",
    "facet_stats",
    "effective_capabilities",
    "runtime_value_lexicon",
    "runtime_registry",
    "reviewed_config",
]

ARTIFACT_KINDS: tuple[ArtifactKind, ...] = (
    "catalog",
    "category_registry",
    "effective_capabilities",
    "facet_applicability",
    "facet_evidence_store",
    "facet_schema",
    "facet_source_bindings",
    "facet_stats",
    "product_category_assignment",
    "product_facet_index",
    "reviewed_config",
    "runtime_registry",
    "runtime_value_lexicon",
)

ARTIFACT_SPEC: Mapping[ArtifactKind, tuple[str, str, str]] = MappingProxyType(
    {
        "catalog": ("catalog_id", "shopping-copilot/raw-catalog-jsonl/v1", "catalog.jsonl"),
        "category_registry": (
            "category_registry_id",
            "shopping-copilot/category-registry/v0",
            "category-registry.json",
        ),
        "product_category_assignment": (
            "product_category_assignment_id",
            "shopping-copilot/product-category-assignment/v0",
            "product-category-assignment.json",
        ),
        "facet_schema": (
            "facet_schema_id",
            "shopping-copilot/catalog-facet-schema/v0",
            "catalog-facet-schema.json",
        ),
        "facet_applicability": (
            "facet_applicability_id",
            "shopping-copilot/facet-applicability/v0",
            "facet-applicability.json",
        ),
        "facet_source_bindings": (
            "facet_source_bindings_id",
            "shopping-copilot/facet-source-bindings/v0",
            "facet-source-bindings.json",
        ),
        "facet_evidence_store": (
            "facet_evidence_store_id",
            "shopping-copilot/facet-evidence-store/v0",
            "facet-evidence-store.json",
        ),
        "product_facet_index": (
            "product_facet_index_id",
            "shopping-copilot/product-facet-index/v0",
            "product-facet-index.json",
        ),
        "facet_stats": (
            "facet_stats_id",
            "shopping-copilot/catalog-facet-stats/v0",
            "catalog-facet-stats.json",
        ),
        "effective_capabilities": (
            "effective_capabilities_id",
            "shopping-copilot/effective-facet-capabilities/v0",
            "effective-facet-capabilities.json",
        ),
        "runtime_value_lexicon": (
            "runtime_value_lexicon_id",
            "shopping-copilot/runtime-value-lexicon/v0",
            "runtime-value-lexicon.json",
        ),
        "runtime_registry": (
            "runtime_registry_id",
            "shopping-copilot/runtime-facet-registry/v0",
            "runtime-facet-registry.json",
        ),
        "reviewed_config": (
            "reviewed_config_id",
            "shopping-copilot/reviewed-semantic-config/v0",
            "reviewed-semantic-config.json",
        ),
    }
)

RELEASE_MANIFEST_FILENAME = "catalog-semantic-release.json"

_CONTENT_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_SEMANTIC_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactRef:
    """Exact byte identity for one release member."""

    kind: ArtifactKind
    schema: str
    content_id: str
    byte_size: int

    def __post_init__(self) -> None:
        if self.kind not in ARTIFACT_SPEC:
            raise ValueError("ArtifactRef.kind is invalid")
        expected_schema = ARTIFACT_SPEC[self.kind][1]
        if self.schema != expected_schema:
            raise ValueError("ArtifactRef.schema is invalid for its kind")
        _require_content_id(self.content_id, name="ArtifactRef.content_id")
        if type(self.byte_size) is not int or not 0 < self.byte_size <= IJSON_SAFE_INTEGER_MAX:
            raise ValueError("ArtifactRef.byte_size must be a positive safe integer")


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeValueAlias:
    """Reviewed exact surface-to-canonical value mapping."""

    surface_form: str
    canonical_value: str | bool

    def __post_init__(self) -> None:
        validate_semantic_string(self.surface_form, name="RuntimeValueAlias.surface_form")
        if type(self.canonical_value) not in (str, bool):
            raise TypeError("RuntimeValueAlias.canonical_value is invalid")
        if type(self.canonical_value) is str:
            validate_semantic_string(
                self.canonical_value,
                name="RuntimeValueAlias.canonical_value",
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReviewedRuntimeFacetConfig:
    """Human-reviewed runtime normalizer and alias projection for one ordinary facet."""

    facet_id: str
    intent_value_normalizer_id: str
    aliases: tuple[RuntimeValueAlias, ...]

    def __post_init__(self) -> None:
        _require_semantic_id(self.facet_id, name="ReviewedRuntimeFacetConfig.facet_id")
        _require_semantic_id(
            self.intent_value_normalizer_id,
            name="ReviewedRuntimeFacetConfig.intent_value_normalizer_id",
        )
        _require_exact_tuple(
            self.aliases,
            RuntimeValueAlias,
            name="ReviewedRuntimeFacetConfig.aliases",
        )
        alias_keys = tuple(item.surface_form for item in self.aliases)
        if alias_keys != tuple(sorted(set(alias_keys), key=lambda item: item.encode("utf-8"))):
            raise ValueError("runtime aliases must be sorted and unique")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReviewedSemanticConfig:
    """Exact content-addressed projection of all human-reviewed semantic inputs."""

    schema: Literal["shopping-copilot/reviewed-semantic-config/v0"]
    catalog_id: str
    category_graph_id: str
    resolution_policy_id: Literal["structured_resolution_v1"]
    builder_version: str
    category_scopes: tuple[CategoryScope, ...]
    facets: tuple[CatalogFacetDefinition, ...]
    facet_applicability: tuple[FacetApplicability, ...]
    source_bindings: tuple[FacetSourceBinding, ...]
    capabilities: tuple[EffectiveFacetCapability, ...]
    runtime_facets: tuple[ReviewedRuntimeFacetConfig, ...]

    def __post_init__(self) -> None:
        if self.schema != REVIEWED_SEMANTIC_CONFIG_SCHEMA:
            raise ValueError("ReviewedSemanticConfig.schema is invalid")
        _require_content_id(self.catalog_id, name="ReviewedSemanticConfig.catalog_id")
        _require_content_id(
            self.category_graph_id,
            name="ReviewedSemanticConfig.category_graph_id",
        )
        if self.resolution_policy_id != RESOLUTION_POLICY_ID:
            raise ValueError("ReviewedSemanticConfig resolution policy is invalid")
        _require_builder_version(self.builder_version)
        _require_sorted_object_tuple(self.category_scopes, CategoryScope, "id", "category_scopes")
        _require_sorted_object_tuple(self.facets, CatalogFacetDefinition, "id", "facets")
        _require_sorted_object_tuple(
            self.facet_applicability,
            FacetApplicability,
            "facet_id",
            "facet_applicability",
        )
        _require_sorted_object_tuple(
            self.source_bindings,
            FacetSourceBinding,
            "id",
            "source_bindings",
        )
        _require_exact_tuple(
            self.capabilities,
            EffectiveFacetCapability,
            name="ReviewedSemanticConfig.capabilities",
        )
        capability_keys = tuple(
            (item.facet_id, item.category_scope_id) for item in self.capabilities
        )
        if capability_keys != tuple(sorted(set(capability_keys))):
            raise ValueError("reviewed capabilities must be sorted and unique")
        _require_sorted_object_tuple(
            self.runtime_facets,
            ReviewedRuntimeFacetConfig,
            "facet_id",
            "runtime_facets",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CatalogSemanticReleaseManifest:
    """Canonical manifest binding the exact 13 P0 release artifacts."""

    schema: Literal["shopping-copilot/catalog-semantic-release/v0"]
    catalog_id: str
    category_registry_id: str
    product_category_assignment_id: str
    facet_schema_id: str
    facet_applicability_id: str
    facet_source_bindings_id: str
    facet_evidence_store_id: str
    product_facet_index_id: str
    facet_stats_id: str
    effective_capabilities_id: str
    runtime_value_lexicon_id: str
    runtime_registry_id: str
    reviewed_config_id: str
    resolution_policy_id: Literal["structured_resolution_v1"]
    builder_version: str
    artifacts: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        if self.schema != CATALOG_SEMANTIC_RELEASE_SCHEMA:
            raise ValueError("CatalogSemanticReleaseManifest.schema is invalid")
        for field_name, _, _ in ARTIFACT_SPEC.values():
            _require_content_id(
                getattr(self, field_name),
                name=f"CatalogSemanticReleaseManifest.{field_name}",
            )
        if self.resolution_policy_id != RESOLUTION_POLICY_ID:
            raise ValueError("release resolution policy is invalid")
        _require_builder_version(self.builder_version)
        _require_exact_tuple(self.artifacts, ArtifactRef, name="release artifacts")
        kinds = tuple(item.kind for item in self.artifacts)
        if kinds != ARTIFACT_KINDS:
            raise ValueError("release must contain exactly 13 sorted artifact kinds")
        refs = {item.kind: item for item in self.artifacts}
        for kind, (field_name, schema, _) in ARTIFACT_SPEC.items():
            ref = refs[kind]
            if ref.schema != schema or ref.content_id != getattr(self, field_name):
                raise ValueError("release manifest field-to-artifact mapping is invalid")


@dataclass(frozen=True, slots=True, kw_only=True)
class VerifiedCatalogSemanticRelease:
    """Runtime holder returned only after the self-contained release verifies."""

    release_id: str
    manifest: CatalogSemanticReleaseManifest
    category_registry: CategoryRegistry
    product_category_assignments: ProductCategoryAssignmentSet
    facet_schema: CatalogFacetSchema
    facet_applicability: FacetApplicabilitySet
    facet_source_bindings: FacetSourceBindingSet
    facet_evidence_store: FacetEvidenceStore
    product_facet_index: ProductFacetIndex
    facet_stats: CatalogFacetStatsArtifact
    effective_capabilities: EffectiveFacetCapabilitySet
    runtime_value_lexicon: RuntimeValueLexicon
    runtime_registry: RuntimeFacetRegistryArtifact
    reviewed_config: ReviewedSemanticConfig
    grounder: RuntimeValueGrounder


def _require_content_id(value: object, *, name: str) -> None:
    if type(value) is not str or _CONTENT_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} is invalid")


def _require_semantic_id(value: object, *, name: str) -> None:
    if type(value) is not str or _SEMANTIC_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} is invalid")


def _require_builder_version(value: object) -> None:
    _require_semantic_id(value, name="builder_version")


def _require_exact_tuple(values: object, expected_type: type[object], *, name: str) -> None:
    if type(values) is not tuple or any(type(item) is not expected_type for item in values):
        raise TypeError(f"{name} is invalid")


def _require_sorted_object_tuple(
    values: object,
    expected_type: type[object],
    key_name: str,
    name: str,
) -> None:
    _require_exact_tuple(values, expected_type, name=f"ReviewedSemanticConfig.{name}")
    typed_values = cast(tuple[object, ...], values)
    keys = tuple(getattr(item, key_name) for item in typed_values)
    if keys != tuple(sorted(set(keys))):
        raise ValueError(f"{name} must be sorted and unique")
