"""Immutable CS5A runtime registry, numeric lexicon, and candidate DTOs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from shopping_copilot.session_context import CATEGORICAL_OPERATORS, NUMERIC_OPERATORS

from ..canonical import content_id_for_value
from ..category.models import require_builder_version, require_content_id
from ..facet.resolution_models import RESOLUTION_POLICY_ID

RUNTIME_FACET_REGISTRY_SCHEMA: Literal["shopping-copilot/runtime-facet-registry/v0"] = (
    "shopping-copilot/runtime-facet-registry/v0"
)
RUNTIME_VALUE_LEXICON_SCHEMA: Literal["shopping-copilot/runtime-value-lexicon/v0"] = (
    "shopping-copilot/runtime-value-lexicon/v0"
)
RUNTIME_PROJECTION_CANDIDATE_SCHEMA: Literal["shopping-copilot/runtime-projection-candidate/v0"] = (
    "shopping-copilot/runtime-projection-candidate/v0"
)
RUNTIME_PROJECTION_BUILDER_VERSION = "catalog_semantic_runtime_projection_v0"
CATEGORY_SCOPE_ID_NORMALIZER_ID = "category_scope_id_v1"
SYSTEM_PRODUCT_CATEGORY_FACET_ID = "system_product_category"

_SEMANTIC_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeFacetSpecRecord:
    """Declarative session-context FacetSpec projection."""

    facet_id: str
    kind: Literal["categorical", "numeric"]
    operator_values: tuple[str, ...]
    intent_value_normalizer_id: str

    def __post_init__(self) -> None:
        _require_semantic_id(self.facet_id, name="RuntimeFacetSpecRecord.facet_id")
        if self.kind not in ("categorical", "numeric"):
            raise ValueError("RuntimeFacetSpecRecord.kind is invalid")
        if type(self.operator_values) is not tuple or any(
            type(item) is not str for item in self.operator_values
        ):
            raise TypeError("RuntimeFacetSpecRecord.operator_values must be a string tuple")
        expected_operators = (
            tuple(sorted(operator.value for operator in CATEGORICAL_OPERATORS))
            if self.kind == "categorical"
            else tuple(sorted(operator.value for operator in NUMERIC_OPERATORS))
        )
        if self.operator_values != expected_operators:
            raise ValueError("RuntimeFacetSpecRecord operator family is invalid")
        _require_semantic_id(
            self.intent_value_normalizer_id,
            name="RuntimeFacetSpecRecord.intent_value_normalizer_id",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeFacetRegistryArtifact:
    """Content-addressed declarative registry for session-context projection."""

    schema: Literal["shopping-copilot/runtime-facet-registry/v0"]
    category_registry_id: str
    facet_schema_id: str
    effective_capabilities_id: str
    resolution_policy_id: Literal["structured_resolution_v1"]
    entries: tuple[RuntimeFacetSpecRecord, ...]

    def __post_init__(self) -> None:
        if self.schema != RUNTIME_FACET_REGISTRY_SCHEMA:
            raise ValueError("RuntimeFacetRegistryArtifact.schema is invalid")
        for name in (
            "category_registry_id",
            "facet_schema_id",
            "effective_capabilities_id",
        ):
            require_content_id(
                getattr(self, name),
                name=f"RuntimeFacetRegistryArtifact.{name}",
            )
        if self.resolution_policy_id != RESOLUTION_POLICY_ID:
            raise ValueError("RuntimeFacetRegistryArtifact resolution policy is invalid")
        _require_exact_tuple(
            self.entries,
            RuntimeFacetSpecRecord,
            name="RuntimeFacetRegistryArtifact.entries",
        )
        facet_ids = tuple(item.facet_id for item in self.entries)
        if not facet_ids or facet_ids != tuple(sorted(set(facet_ids))):
            raise ValueError("runtime facet records must be non-empty, sorted, and unique")
        reserved = next(
            (item for item in self.entries if item.facet_id == SYSTEM_PRODUCT_CATEGORY_FACET_ID),
            None,
        )
        if (
            reserved is None
            or reserved.kind != "categorical"
            or reserved.intent_value_normalizer_id != CATEGORY_SCOPE_ID_NORMALIZER_ID
        ):
            raise ValueError("runtime registry requires the reserved category FacetSpec")


@dataclass(frozen=True, slots=True, kw_only=True)
class NumericRuntimeDomain:
    """Closed runtime facts for the sole P0 numeric facet."""

    kind: Literal["numeric"]
    facet_id: Literal["price"]
    intent_value_normalizer_id: str
    canonical_unit: Literal["USD_CENT"]
    integer_only: Literal[True]

    def __post_init__(self) -> None:
        if self.kind != "numeric" or self.facet_id != "price":
            raise ValueError("NumericRuntimeDomain supports only numeric price")
        _require_semantic_id(
            self.intent_value_normalizer_id,
            name="NumericRuntimeDomain.intent_value_normalizer_id",
        )
        if self.canonical_unit != "USD_CENT" or self.integer_only is not True:
            raise ValueError("NumericRuntimeDomain must use integer USD_CENT")


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeValueLexicon:
    """Runtime value domains pinned to registry, applicability, index, and policy."""

    schema: Literal["shopping-copilot/runtime-value-lexicon/v0"]
    runtime_registry_id: str
    category_registry_id: str
    facet_applicability_id: str
    product_facet_index_id: str
    resolution_policy_id: Literal["structured_resolution_v1"]
    domains: tuple[NumericRuntimeDomain, ...]

    def __post_init__(self) -> None:
        if self.schema != RUNTIME_VALUE_LEXICON_SCHEMA:
            raise ValueError("RuntimeValueLexicon.schema is invalid")
        for name in (
            "runtime_registry_id",
            "category_registry_id",
            "facet_applicability_id",
            "product_facet_index_id",
        ):
            require_content_id(getattr(self, name), name=f"RuntimeValueLexicon.{name}")
        if self.resolution_policy_id != RESOLUTION_POLICY_ID:
            raise ValueError("RuntimeValueLexicon resolution policy is invalid")
        _require_exact_tuple(
            self.domains,
            NumericRuntimeDomain,
            name="RuntimeValueLexicon.domains",
        )
        facet_ids = tuple(item.facet_id for item in self.domains)
        if facet_ids != tuple(sorted(set(facet_ids))):
            raise ValueError("runtime domains must be sorted and unique")


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeProjectionCandidateBuild:
    """Complete CS5A projection before grounding or release assembly."""

    schema: Literal["shopping-copilot/runtime-projection-candidate/v0"]
    builder_version: str
    catalog_id: str
    gate_b_selection_id: str
    effective_capabilities_id: str
    runtime_registry: RuntimeFacetRegistryArtifact
    runtime_lexicon: RuntimeValueLexicon

    def __post_init__(self) -> None:
        if self.schema != RUNTIME_PROJECTION_CANDIDATE_SCHEMA:
            raise ValueError("RuntimeProjectionCandidateBuild.schema is invalid")
        if self.builder_version != RUNTIME_PROJECTION_BUILDER_VERSION:
            raise ValueError("RuntimeProjectionCandidateBuild.builder version is unsupported")
        require_builder_version(self.builder_version)
        for name in (
            "catalog_id",
            "gate_b_selection_id",
            "effective_capabilities_id",
        ):
            require_content_id(
                getattr(self, name),
                name=f"RuntimeProjectionCandidateBuild.{name}",
            )
        if type(self.runtime_registry) is not RuntimeFacetRegistryArtifact:
            raise TypeError("RuntimeProjectionCandidateBuild.runtime_registry is invalid")
        if type(self.runtime_lexicon) is not RuntimeValueLexicon:
            raise TypeError("RuntimeProjectionCandidateBuild.runtime_lexicon is invalid")
        if (
            self.runtime_registry.effective_capabilities_id != self.effective_capabilities_id
            or self.runtime_lexicon.runtime_registry_id
            != content_id_for_value(self.runtime_registry)
            or self.runtime_lexicon.category_registry_id
            != self.runtime_registry.category_registry_id
        ):
            raise ValueError("runtime projection artifact pins differ")


def _require_semantic_id(value: object, *, name: str) -> None:
    if type(value) is not str or _SEMANTIC_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} is invalid")


def _require_exact_tuple(values: object, expected_type: type[object], *, name: str) -> None:
    if type(values) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    if any(type(item) is not expected_type for item in values):
        raise TypeError(f"{name} contains an invalid item")
