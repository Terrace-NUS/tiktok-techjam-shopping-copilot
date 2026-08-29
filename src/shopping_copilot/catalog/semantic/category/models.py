"""Immutable contract DTOs and local review DTOs for category semantics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal, cast

from ..canonical import (
    IJSON_SAFE_INTEGER_MAX,
    canonical_json_bytes,
    validate_semantic_string,
)

CATEGORY_GRAPH_CORE_SCHEMA: Literal["shopping-copilot/category-graph-core/v0"] = (
    "shopping-copilot/category-graph-core/v0"
)
CATEGORY_REGISTRY_SCHEMA: Literal["shopping-copilot/category-registry/v0"] = (
    "shopping-copilot/category-registry/v0"
)
PRODUCT_CATEGORY_ASSIGNMENT_SCHEMA: Literal["shopping-copilot/product-category-assignment/v0"] = (
    "shopping-copilot/product-category-assignment/v0"
)
CATEGORY_GRAPH_PROPOSAL_SCHEMA: Literal["shopping-copilot/category-graph-proposal/v0"] = (
    "shopping-copilot/category-graph-proposal/v0"
)
CATEGORY_SCOPE_SELECTION_SCHEMA: Literal["shopping-copilot/category-scope-selection/v0"] = (
    "shopping-copilot/category-scope-selection/v0"
)
CATEGORY_SCOPE_SELECTION_TEMPLATE_SCHEMA = "shopping-copilot/category-scope-selection-template/v0"
CATEGORY_SCOPES_CANDIDATE_SCHEMA = "shopping-copilot/reviewed-category-scopes-candidate/v0"

_CONTENT_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_NODE_ID_PATTERN = re.compile(r"^cn_[0-9a-f]{64}$")
_SCOPE_ID_PATTERN = re.compile(r"^cs_[0-9a-f]{64}$")
_BUILDER_VERSION_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


class ProductCategoryAssignmentStatus(str, Enum):
    """Confidence state for one product's category membership."""

    KNOWN = "known"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


class CategoryMatchResult(str, Enum):
    """Three-valued category membership match result."""

    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True, kw_only=True)
class CategoryNode:
    """One canonical category path prefix from the contract graph."""

    id: str
    parent_id: str | None
    canonical_path: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.id, pattern=_NODE_ID_PATTERN, name="CategoryNode.id")
        if self.parent_id is not None:
            _require_identifier(
                self.parent_id,
                pattern=_NODE_ID_PATTERN,
                name="CategoryNode.parent_id",
            )
        _require_semantic_string_tuple(
            self.canonical_path,
            name="CategoryNode.canonical_path",
            nonempty=True,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CategoryScope:
    """Reviewed union of complete canonical category subtrees."""

    id: str
    label: str
    root_node_ids: tuple[str, ...]
    member_node_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.id, pattern=_SCOPE_ID_PATTERN, name="CategoryScope.id")
        validate_semantic_string(self.label, name="CategoryScope.label")
        _require_identifier_tuple(
            self.root_node_ids,
            pattern=_NODE_ID_PATTERN,
            name="CategoryScope.root_node_ids",
            nonempty=True,
        )
        _require_identifier_tuple(
            self.member_node_ids,
            pattern=_NODE_ID_PATTERN,
            name="CategoryScope.member_node_ids",
            nonempty=True,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CategoryRegistry:
    """Published category graph and reviewed user-facing scopes."""

    schema: Literal["shopping-copilot/category-registry/v0"]
    catalog_id: str
    category_graph_id: str
    root_scope_id: str
    nodes: tuple[CategoryNode, ...]
    scopes: tuple[CategoryScope, ...]

    def __post_init__(self) -> None:
        if self.schema != CATEGORY_REGISTRY_SCHEMA:
            raise ValueError("CategoryRegistry.schema is invalid")
        _require_content_id(self.catalog_id, name="CategoryRegistry.catalog_id")
        _require_content_id(
            self.category_graph_id,
            name="CategoryRegistry.category_graph_id",
        )
        _require_identifier(
            self.root_scope_id,
            pattern=_SCOPE_ID_PATTERN,
            name="CategoryRegistry.root_scope_id",
        )
        _require_exact_tuple(self.nodes, name="CategoryRegistry.nodes")
        _require_exact_tuple(self.scopes, name="CategoryRegistry.scopes")


@dataclass(frozen=True, slots=True, kw_only=True)
class ProductCategoryAssignment:
    """One product's resolved or uncertain terminal category nodes."""

    parent_asin: str
    status: ProductCategoryAssignmentStatus
    leaf_node_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_semantic_string(
            self.parent_asin,
            name="ProductCategoryAssignment.parent_asin",
        )
        if self.parent_asin != self.parent_asin.strip():
            raise ValueError("ProductCategoryAssignment.parent_asin must be trimmed")
        if not isinstance(self.status, ProductCategoryAssignmentStatus):
            raise TypeError("ProductCategoryAssignment.status is invalid")
        _require_identifier_tuple(
            self.leaf_node_ids,
            pattern=_NODE_ID_PATTERN,
            name="ProductCategoryAssignment.leaf_node_ids",
            nonempty=self.status is not ProductCategoryAssignmentStatus.UNKNOWN,
        )
        if self.status is ProductCategoryAssignmentStatus.UNKNOWN and self.leaf_node_ids:
            raise ValueError("UNKNOWN category assignment must not contain leaf IDs")
        if self.status is ProductCategoryAssignmentStatus.CONFLICT and len(self.leaf_node_ids) < 2:
            raise ValueError("CONFLICT category assignment requires at least two leaf IDs")


@dataclass(frozen=True, slots=True, kw_only=True)
class ProductCategoryAssignmentSet:
    """Published product-to-category assignment artifact."""

    schema: Literal["shopping-copilot/product-category-assignment/v0"]
    catalog_id: str
    category_graph_id: str
    assignments: tuple[ProductCategoryAssignment, ...]

    def __post_init__(self) -> None:
        if self.schema != PRODUCT_CATEGORY_ASSIGNMENT_SCHEMA:
            raise ValueError("ProductCategoryAssignmentSet.schema is invalid")
        _require_content_id(
            self.catalog_id,
            name="ProductCategoryAssignmentSet.catalog_id",
        )
        _require_content_id(
            self.category_graph_id,
            name="ProductCategoryAssignmentSet.category_graph_id",
        )
        _require_exact_tuple(
            self.assignments,
            name="ProductCategoryAssignmentSet.assignments",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RawPathMapping:
    """Audit-only mapping from one exact raw prefix to canonical identity."""

    raw_path: tuple[str, ...]
    canonical_path: tuple[str, ...]
    node_id: str
    direct_product_count: int
    subtree_product_count: int

    def __post_init__(self) -> None:
        _require_semantic_string_tuple(
            self.raw_path,
            name="RawPathMapping.raw_path",
            nonempty=True,
        )
        _require_semantic_string_tuple(
            self.canonical_path,
            name="RawPathMapping.canonical_path",
            nonempty=True,
        )
        _require_identifier(
            self.node_id,
            pattern=_NODE_ID_PATTERN,
            name="RawPathMapping.node_id",
        )
        _require_nonnegative_safe_integer(
            self.direct_product_count,
            name="RawPathMapping.direct_product_count",
        )
        _require_positive_safe_integer(
            self.subtree_product_count,
            name="RawPathMapping.subtree_product_count",
        )
        if self.direct_product_count > self.subtree_product_count:
            raise ValueError("RawPathMapping.direct_product_count must not exceed subtree support")


@dataclass(frozen=True, slots=True, kw_only=True)
class CategoryNormalizationCollision:
    """Audit-only group of distinct raw paths sharing canonical identity."""

    canonical_path: tuple[str, ...]
    raw_paths: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        _require_semantic_string_tuple(
            self.canonical_path,
            name="CategoryNormalizationCollision.canonical_path",
            nonempty=True,
        )
        _require_exact_tuple(
            self.raw_paths,
            name="CategoryNormalizationCollision.raw_paths",
        )
        if len(self.raw_paths) < 2:
            raise ValueError("CategoryNormalizationCollision.raw_paths requires at least two paths")
        for index, path in enumerate(self.raw_paths):
            _require_semantic_string_tuple(
                path,
                name=f"CategoryNormalizationCollision.raw_paths[{index}]",
                nonempty=True,
            )
        if self.raw_paths != tuple(sorted(self.raw_paths, key=canonical_json_bytes)):
            raise ValueError("CategoryNormalizationCollision.raw_paths must be sorted")
        if len(set(self.raw_paths)) != len(self.raw_paths):
            raise ValueError("CategoryNormalizationCollision.raw_paths must be unique")


@dataclass(frozen=True, slots=True, kw_only=True)
class CategoryGraphProposal:
    """Pass-A graph plus provenance; never a release artifact."""

    schema: Literal["shopping-copilot/category-graph-proposal/v0"]
    catalog_id: str
    category_graph_id: str
    builder_version: str
    unicode_data_version: str
    catalog_byte_size: int
    product_count: int
    raw_prefix_count: int
    nodes: tuple[CategoryNode, ...]
    raw_path_mappings: tuple[RawPathMapping, ...]
    collisions: tuple[CategoryNormalizationCollision, ...]

    def __post_init__(self) -> None:
        if self.schema != CATEGORY_GRAPH_PROPOSAL_SCHEMA:
            raise ValueError("CategoryGraphProposal.schema is invalid")
        _require_content_id(self.catalog_id, name="CategoryGraphProposal.catalog_id")
        _require_content_id(
            self.category_graph_id,
            name="CategoryGraphProposal.category_graph_id",
        )
        _require_builder_version(self.builder_version)
        validate_semantic_string(
            self.unicode_data_version,
            name="CategoryGraphProposal.unicode_data_version",
        )
        _require_positive_safe_integer(
            self.catalog_byte_size,
            name="CategoryGraphProposal.catalog_byte_size",
        )
        _require_positive_safe_integer(
            self.product_count,
            name="CategoryGraphProposal.product_count",
        )
        _require_positive_safe_integer(
            self.raw_prefix_count,
            name="CategoryGraphProposal.raw_prefix_count",
        )
        _require_exact_tuple(self.nodes, name="CategoryGraphProposal.nodes")
        _require_exact_tuple(
            self.raw_path_mappings,
            name="CategoryGraphProposal.raw_path_mappings",
        )
        _require_exact_tuple(
            self.collisions,
            name="CategoryGraphProposal.collisions",
        )
        if self.raw_prefix_count != len(self.raw_path_mappings):
            raise ValueError("CategoryGraphProposal.raw_prefix_count differs from mappings")
        raw_paths = tuple(mapping.raw_path for mapping in self.raw_path_mappings)
        if raw_paths != tuple(sorted(raw_paths, key=canonical_json_bytes)):
            raise ValueError("CategoryGraphProposal.raw_path_mappings must be sorted")
        if len(set(raw_paths)) != len(raw_paths):
            raise ValueError("CategoryGraphProposal raw paths must be unique")
        collision_paths = tuple(collision.canonical_path for collision in self.collisions)
        if collision_paths != tuple(sorted(collision_paths, key=canonical_json_bytes)):
            raise ValueError("CategoryGraphProposal.collisions must be sorted")
        if len(set(collision_paths)) != len(collision_paths):
            raise ValueError("CategoryGraphProposal collision paths must be unique")


@dataclass(frozen=True, slots=True, kw_only=True)
class CategoryScopeSelection:
    """Human selection before deterministic scope closure materialization."""

    label: str
    root_node_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_semantic_string(self.label, name="CategoryScopeSelection.label")
        _require_identifier_tuple(
            self.root_node_ids,
            pattern=_NODE_ID_PATTERN,
            name="CategoryScopeSelection.root_node_ids",
            nonempty=True,
        )
        _require_sorted_unique(self.root_node_ids, name="CategoryScopeSelection.root_node_ids")


@dataclass(frozen=True, slots=True, kw_only=True)
class CategoryScopeSelectionDocument:
    """Versioned source-controlled input for CS1 Pass B."""

    schema: Literal["shopping-copilot/category-scope-selection/v0"]
    catalog_id: str
    category_graph_id: str
    builder_version: str
    scopes: tuple[CategoryScopeSelection, ...]

    def __post_init__(self) -> None:
        if self.schema != CATEGORY_SCOPE_SELECTION_SCHEMA:
            raise ValueError("CategoryScopeSelectionDocument.schema is invalid")
        _require_content_id(
            self.catalog_id,
            name="CategoryScopeSelectionDocument.catalog_id",
        )
        _require_content_id(
            self.category_graph_id,
            name="CategoryScopeSelectionDocument.category_graph_id",
        )
        _require_builder_version(self.builder_version)
        _require_exact_tuple(
            self.scopes,
            name="CategoryScopeSelectionDocument.scopes",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CategoryCandidateBuild:
    """Complete CS1 candidate result before release assembly."""

    builder_version: str
    registry: CategoryRegistry
    assignments: ProductCategoryAssignmentSet

    def __post_init__(self) -> None:
        _require_builder_version(self.builder_version)
        if self.registry.catalog_id != self.assignments.catalog_id:
            raise ValueError("candidate registry and assignments catalog IDs differ")
        if self.registry.category_graph_id != self.assignments.category_graph_id:
            raise ValueError("candidate registry and assignments graph IDs differ")


def require_builder_version(value: object) -> str:
    """Validate the shared lower-snake-case builder version grammar."""

    return _require_builder_version(value)


def require_content_id(value: object, *, name: str) -> str:
    """Validate the shared full SHA-256 content-ID grammar."""

    return _require_content_id(value, name=name)


def _require_builder_version(value: object) -> str:
    if type(value) is not str or _BUILDER_VERSION_PATTERN.fullmatch(value) is None:
        raise ValueError("builder_version is invalid")
    return value


def _require_content_id(value: object, *, name: str) -> str:
    if type(value) is not str or _CONTENT_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a sha256 content ID")
    return value


def _require_identifier(value: object, *, pattern: re.Pattern[str], name: str) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise ValueError(f"{name} has an invalid identifier")
    return value


def _require_exact_tuple(value: object, *, name: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    return value


def _require_semantic_string_tuple(
    value: object,
    *,
    name: str,
    nonempty: bool,
) -> tuple[str, ...]:
    _require_exact_tuple(value, name=name)
    items = cast(tuple[object, ...], value)
    if nonempty and not items:
        raise ValueError(f"{name} must be non-empty")
    for index, item in enumerate(items):
        validate_semantic_string(item, name=f"{name}[{index}]")
    return cast(tuple[str, ...], items)


def _require_identifier_tuple(
    value: object,
    *,
    pattern: re.Pattern[str],
    name: str,
    nonempty: bool,
) -> tuple[str, ...]:
    _require_exact_tuple(value, name=name)
    items = cast(tuple[object, ...], value)
    if nonempty and not items:
        raise ValueError(f"{name} must be non-empty")
    for index, item in enumerate(items):
        _require_identifier(item, pattern=pattern, name=f"{name}[{index}]")
    return cast(tuple[str, ...], items)


def _require_sorted_unique(values: tuple[str, ...], *, name: str) -> None:
    if tuple(sorted(values)) != values:
        raise ValueError(f"{name} must already be sorted")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")


def _require_nonnegative_safe_integer(value: object, *, name: str) -> int:
    if type(value) is not int or not 0 <= value <= IJSON_SAFE_INTEGER_MAX:
        raise ValueError(f"{name} must be a non-negative I-JSON safe integer")
    return value


def _require_positive_safe_integer(value: object, *, name: str) -> int:
    result = _require_nonnegative_safe_integer(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result
