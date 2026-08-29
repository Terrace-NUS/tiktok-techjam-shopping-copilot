"""Pure validators, ID functions, closure logic, and category matcher."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set

from ..canonical import canonical_json_bytes, content_id_for_value, sha256_hex
from ..errors import CategoryBuildError
from ..raw_catalog import OFFICIAL_PRODUCT_COUNT
from .models import (
    CATEGORY_GRAPH_CORE_SCHEMA,
    CategoryMatchResult,
    CategoryNode,
    CategoryRegistry,
    CategoryScope,
    ProductCategoryAssignment,
    ProductCategoryAssignmentSet,
    ProductCategoryAssignmentStatus,
    require_content_id,
)
from .normalization import category_node_id, normalize_category_path


def category_graph_core_document(
    catalog_id: str,
    nodes: tuple[CategoryNode, ...],
) -> dict[str, object]:
    """Return the exact contract graph-core hash preimage."""

    require_content_id(catalog_id, name="category graph catalog_id")
    validate_category_nodes(nodes)
    return {
        "schema": CATEGORY_GRAPH_CORE_SCHEMA,
        "catalog_id": catalog_id,
        "nodes": [
            {
                "id": node.id,
                "parent_id": node.parent_id,
                "canonical_path": list(node.canonical_path),
            }
            for node in nodes
        ],
    }


def category_graph_id(
    catalog_id: str,
    nodes: tuple[CategoryNode, ...],
) -> str:
    """Return the graph identity, distinct from CategoryRegistry content ID."""

    return content_id_for_value(category_graph_core_document(catalog_id, nodes))


def category_scope_id(
    graph_id: str,
    root_node_ids: tuple[str, ...],
) -> str:
    """Return the exact contract ID for one already-canonical root tuple."""

    require_content_id(graph_id, name="scope category_graph_id")
    _require_sorted_unique(root_node_ids, name="scope root_node_ids", nonempty=True)
    payload = canonical_json_bytes(
        {
            "category_graph_id": graph_id,
            "root_node_ids": list(root_node_ids),
        }
    )
    return f"cs_{sha256_hex(payload)}"


def validate_category_nodes(nodes: tuple[CategoryNode, ...]) -> None:
    """Validate canonical graph identity, ordering, and immediate parents."""

    if type(nodes) is not tuple or not nodes:
        raise CategoryBuildError("category nodes must be a non-empty tuple")
    if any(type(node) is not CategoryNode for node in nodes):
        raise CategoryBuildError("category nodes must contain only CategoryNode objects")
    ids = tuple(node.id for node in nodes)
    _require_sorted_unique(ids, name="category node IDs", nonempty=True)
    paths = tuple(node.canonical_path for node in nodes)
    if len(set(paths)) != len(paths):
        raise CategoryBuildError("canonical category paths must be unique")
    by_id = {node.id: node for node in nodes}

    for node in nodes:
        if normalize_category_path(node.canonical_path) != node.canonical_path:
            raise CategoryBuildError("CategoryNode.canonical_path is not canonical")
        expected_id = category_node_id(node.canonical_path)
        if node.id != expected_id:
            raise CategoryBuildError("CategoryNode.id does not match canonical_path")
        if len(node.canonical_path) == 1:
            if node.parent_id is not None:
                raise CategoryBuildError("root CategoryNode must have parent_id=None")
            continue
        expected_parent_id = category_node_id(node.canonical_path[:-1])
        if node.parent_id != expected_parent_id:
            raise CategoryBuildError("CategoryNode parent is not the immediate prefix")
        parent = by_id.get(expected_parent_id)
        if parent is None or parent.canonical_path != node.canonical_path[:-1]:
            raise CategoryBuildError("CategoryNode parent is missing from graph")


def materialize_category_scope(
    *,
    graph_id: str,
    nodes: tuple[CategoryNode, ...],
    label: str,
    root_node_ids: tuple[str, ...],
) -> CategoryScope:
    """Build exact subtree-union membership for one reviewed root selection."""

    validate_category_nodes(nodes)
    _require_sorted_unique(root_node_ids, name="scope root_node_ids", nonempty=True)
    by_id = {node.id: node for node in nodes}
    unknown = tuple(root for root in root_node_ids if root not in by_id)
    if unknown:
        raise CategoryBuildError(f"scope contains unknown root node ID: {unknown[0]}")

    root_paths = tuple(by_id[root].canonical_path for root in root_node_ids)
    for index, path in enumerate(root_paths):
        for other_index, other_path in enumerate(root_paths):
            if index == other_index:
                continue
            if _is_prefix(other_path, path):
                raise CategoryBuildError("scope contains redundant ancestor/descendant roots")

    member_ids = tuple(
        sorted(
            node.id
            for node in nodes
            if any(_is_prefix(root_path, node.canonical_path) for root_path in root_paths)
        )
    )
    return CategoryScope(
        id=category_scope_id(graph_id, root_node_ids),
        label=label,
        root_node_ids=root_node_ids,
        member_node_ids=member_ids,
    )


def validate_category_scope(
    scope: CategoryScope,
    *,
    graph_id: str,
    nodes: tuple[CategoryNode, ...],
) -> None:
    """Validate one published scope without silently rematerializing it."""

    expected = materialize_category_scope(
        graph_id=graph_id,
        nodes=nodes,
        label=scope.label,
        root_node_ids=scope.root_node_ids,
    )
    if scope != expected:
        raise CategoryBuildError("CategoryScope does not equal its exact materialization")


def validate_category_registry(registry: CategoryRegistry) -> None:
    """Validate graph, scopes, root scope, and all cross-references."""

    validate_category_nodes(registry.nodes)
    expected_graph_id = category_graph_id(registry.catalog_id, registry.nodes)
    if registry.category_graph_id != expected_graph_id:
        raise CategoryBuildError("CategoryRegistry.category_graph_id is invalid")
    if not registry.scopes:
        raise CategoryBuildError("CategoryRegistry.scopes must be non-empty")
    if any(type(scope) is not CategoryScope for scope in registry.scopes):
        raise CategoryBuildError("CategoryRegistry.scopes contains an invalid object")
    scope_ids = tuple(scope.id for scope in registry.scopes)
    _require_sorted_unique(scope_ids, name="category scope IDs", nonempty=True)
    membership_sets: set[tuple[str, ...]] = set()
    for scope in registry.scopes:
        validate_category_scope(
            scope,
            graph_id=registry.category_graph_id,
            nodes=registry.nodes,
        )
        if scope.member_node_ids in membership_sets:
            raise CategoryBuildError("equal-membership duplicate CategoryScopes are forbidden")
        membership_sets.add(scope.member_node_ids)

    scopes_by_id = {scope.id: scope for scope in registry.scopes}
    root_scope = scopes_by_id.get(registry.root_scope_id)
    if root_scope is None:
        raise CategoryBuildError("root_scope_id does not reference a published scope")
    expected_members = tuple(node.id for node in registry.nodes)
    expected_roots = tuple(sorted(node.id for node in registry.nodes if node.parent_id is None))
    if root_scope.member_node_ids != expected_members:
        raise CategoryBuildError("root scope must contain every CategoryNode")
    if root_scope.root_node_ids != expected_roots:
        raise CategoryBuildError("root scope roots must equal all graph roots")


def validate_product_category_assignment_set(
    assignment_set: ProductCategoryAssignmentSet,
    *,
    registry: CategoryRegistry,
    expected_product_ids: Set[str] | None = None,
    terminal_node_ids_by_product: Mapping[str, Set[str]] | None = None,
) -> None:
    """Validate assignment status shapes, graph references, and optional raw truth."""

    validate_category_registry(registry)
    if assignment_set.catalog_id != registry.catalog_id:
        raise CategoryBuildError("assignment catalog_id does not match CategoryRegistry")
    if assignment_set.category_graph_id != registry.category_graph_id:
        raise CategoryBuildError("assignment category_graph_id does not match CategoryRegistry")

    if any(type(item) is not ProductCategoryAssignment for item in assignment_set.assignments):
        raise CategoryBuildError("assignment set contains an invalid object")
    parent_asins = tuple(item.parent_asin for item in assignment_set.assignments)
    _require_sorted_unique(parent_asins, name="assignment parent_asins", nonempty=False)
    valid_node_ids = {node.id for node in registry.nodes}
    for assignment in assignment_set.assignments:
        _require_sorted_unique(
            assignment.leaf_node_ids,
            name=f"leaf IDs for {assignment.parent_asin}",
            nonempty=assignment.status is not ProductCategoryAssignmentStatus.UNKNOWN,
        )
        if any(node_id not in valid_node_ids for node_id in assignment.leaf_node_ids):
            raise CategoryBuildError("assignment references an unknown CategoryNode")
        if terminal_node_ids_by_product is not None:
            allowed = terminal_node_ids_by_product.get(assignment.parent_asin)
            if allowed is None:
                raise CategoryBuildError("assignment has no matching raw catalog product")
            if any(node_id not in allowed for node_id in assignment.leaf_node_ids):
                raise CategoryBuildError("assignment references a non-terminal product node")

    if expected_product_ids is not None and set(parent_asins) != set(expected_product_ids):
        raise CategoryBuildError("assignment product set does not match raw catalog")


def validate_official_p0_assignments(
    assignment_set: ProductCategoryAssignmentSet,
) -> None:
    """Apply the stricter official publication gate without weakening generic DTOs."""

    if len(assignment_set.assignments) != OFFICIAL_PRODUCT_COUNT:
        raise CategoryBuildError(
            f"official P0 requires exactly {OFFICIAL_PRODUCT_COUNT} assignments"
        )
    if any(
        item.status is not ProductCategoryAssignmentStatus.KNOWN
        for item in assignment_set.assignments
    ):
        raise CategoryBuildError("official P0 requires every category assignment to be KNOWN")


def match_category(
    assignment: ProductCategoryAssignment,
    scope: CategoryScope,
) -> CategoryMatchResult:
    """Apply the contract's exact three-valued assignment/scope matcher."""

    if assignment.status is not ProductCategoryAssignmentStatus.KNOWN:
        return CategoryMatchResult.UNKNOWN
    if set(assignment.leaf_node_ids).intersection(scope.member_node_ids):
        return CategoryMatchResult.SATISFIED
    return CategoryMatchResult.VIOLATED


def _require_sorted_unique(
    values: Sequence[str],
    *,
    name: str,
    nonempty: bool,
) -> None:
    if type(values) is not tuple:
        raise CategoryBuildError(f"{name} must be a tuple")
    if nonempty and not values:
        raise CategoryBuildError(f"{name} must be non-empty")
    if tuple(sorted(values)) != tuple(values):
        raise CategoryBuildError(f"{name} must already be sorted")
    if len(set(values)) != len(values):
        raise CategoryBuildError(f"{name} must be unique")


def _is_prefix(prefix: tuple[str, ...], path: tuple[str, ...]) -> bool:
    return len(prefix) <= len(path) and path[: len(prefix)] == prefix
