"""Deterministic two-pass builders for category graph and reviewed scopes."""

from __future__ import annotations

from collections import Counter, defaultdict

from ..canonical import canonical_json_bytes
from ..errors import CategoryBuildError, CategorySelectionError
from ..raw_catalog import RawCatalogScan
from .models import (
    CATEGORY_GRAPH_PROPOSAL_SCHEMA,
    CATEGORY_REGISTRY_SCHEMA,
    CATEGORY_SCOPE_SELECTION_TEMPLATE_SCHEMA,
    CATEGORY_SCOPES_CANDIDATE_SCHEMA,
    PRODUCT_CATEGORY_ASSIGNMENT_SCHEMA,
    CategoryCandidateBuild,
    CategoryGraphProposal,
    CategoryNode,
    CategoryNormalizationCollision,
    CategoryRegistry,
    CategoryScopeSelectionDocument,
    ProductCategoryAssignment,
    ProductCategoryAssignmentSet,
    ProductCategoryAssignmentStatus,
    RawPathMapping,
)
from .normalization import (
    CATEGORY_BUILDER_VERSION,
    CATEGORY_UNICODE_DATA_VERSION,
    category_node_id,
    ensure_category_builder_runtime,
    normalize_category_path,
)
from .validation import (
    category_graph_id,
    materialize_category_scope,
    validate_category_nodes,
    validate_category_registry,
    validate_official_p0_assignments,
    validate_product_category_assignment_set,
)


def build_category_graph_proposal(scan: RawCatalogScan) -> CategoryGraphProposal:
    """Pass A: build canonical graph and exact raw-to-canonical provenance."""

    ensure_category_builder_runtime()
    direct_support: Counter[tuple[str, ...]] = Counter()
    subtree_support: Counter[tuple[str, ...]] = Counter()
    raw_to_canonical: dict[tuple[str, ...], tuple[str, ...]] = {}
    canonical_paths: set[tuple[str, ...]] = set()

    for record in scan.records:
        canonical_path = normalize_category_path(record.raw_path)
        direct_support[record.raw_path] += 1
        for depth in range(1, len(record.raw_path) + 1):
            raw_prefix = record.raw_path[:depth]
            canonical_prefix = canonical_path[:depth]
            subtree_support[raw_prefix] += 1
            previous = raw_to_canonical.setdefault(raw_prefix, canonical_prefix)
            if previous != canonical_prefix:
                raise CategoryBuildError(
                    "the closed normalizer produced inconsistent prefix output"
                )
            canonical_paths.add(canonical_prefix)

    nodes = tuple(
        sorted(
            (
                CategoryNode(
                    id=category_node_id(path),
                    parent_id=None if len(path) == 1 else category_node_id(path[:-1]),
                    canonical_path=path,
                )
                for path in canonical_paths
            ),
            key=lambda node: node.id,
        )
    )
    validate_category_nodes(nodes)
    graph_id = category_graph_id(scan.catalog_id, nodes)

    mappings = tuple(
        RawPathMapping(
            raw_path=raw_path,
            canonical_path=raw_to_canonical[raw_path],
            node_id=category_node_id(raw_to_canonical[raw_path]),
            direct_product_count=direct_support[raw_path],
            subtree_product_count=subtree_support[raw_path],
        )
        for raw_path in sorted(raw_to_canonical, key=canonical_json_bytes)
    )

    raw_paths_by_canonical: dict[tuple[str, ...], list[tuple[str, ...]]] = defaultdict(list)
    for raw_path, canonical_path in raw_to_canonical.items():
        raw_paths_by_canonical[canonical_path].append(raw_path)
    collisions = tuple(
        CategoryNormalizationCollision(
            canonical_path=canonical_path,
            raw_paths=tuple(sorted(raw_paths, key=canonical_json_bytes)),
        )
        for canonical_path, raw_paths in sorted(
            raw_paths_by_canonical.items(),
            key=lambda item: canonical_json_bytes(item[0]),
        )
        if len(raw_paths) > 1
    )

    return CategoryGraphProposal(
        schema=CATEGORY_GRAPH_PROPOSAL_SCHEMA,
        catalog_id=scan.catalog_id,
        category_graph_id=graph_id,
        builder_version=CATEGORY_BUILDER_VERSION,
        unicode_data_version=CATEGORY_UNICODE_DATA_VERSION,
        catalog_byte_size=scan.byte_size,
        product_count=scan.product_count,
        raw_prefix_count=len(raw_to_canonical),
        nodes=nodes,
        raw_path_mappings=mappings,
        collisions=collisions,
    )


def build_category_candidate(
    scan: RawCatalogScan,
    selection: CategoryScopeSelectionDocument,
    *,
    enforce_official_gate: bool = True,
) -> CategoryCandidateBuild:
    """Pass B: rebuild exact graph, materialize reviewed scopes and assignments."""

    proposal = build_category_graph_proposal(scan)
    _validate_selection_pins(selection, proposal)
    if not selection.scopes:
        raise CategorySelectionError("scope selection must contain at least the root scope")

    scopes = tuple(
        sorted(
            (
                materialize_category_scope(
                    graph_id=proposal.category_graph_id,
                    nodes=proposal.nodes,
                    label=item.label,
                    root_node_ids=item.root_node_ids,
                )
                for item in selection.scopes
            ),
            key=lambda scope: scope.id,
        )
    )
    all_node_ids = tuple(node.id for node in proposal.nodes)
    root_scopes = tuple(scope for scope in scopes if scope.member_node_ids == all_node_ids)
    if len(root_scopes) != 1:
        raise CategorySelectionError(
            "selection must materialize exactly one scope covering the complete graph"
        )

    registry = CategoryRegistry(
        schema=CATEGORY_REGISTRY_SCHEMA,
        catalog_id=proposal.catalog_id,
        category_graph_id=proposal.category_graph_id,
        root_scope_id=root_scopes[0].id,
        nodes=proposal.nodes,
        scopes=scopes,
    )
    validate_category_registry(registry)

    assignments = tuple(
        sorted(
            (
                ProductCategoryAssignment(
                    parent_asin=record.parent_asin,
                    status=ProductCategoryAssignmentStatus.KNOWN,
                    leaf_node_ids=(category_node_id(normalize_category_path(record.raw_path)),),
                )
                for record in scan.records
            ),
            key=lambda assignment: assignment.parent_asin,
        )
    )
    assignment_set = ProductCategoryAssignmentSet(
        schema=PRODUCT_CATEGORY_ASSIGNMENT_SCHEMA,
        catalog_id=proposal.catalog_id,
        category_graph_id=proposal.category_graph_id,
        assignments=assignments,
    )
    terminal_node_ids_by_product = {
        record.parent_asin: {category_node_id(normalize_category_path(record.raw_path))}
        for record in scan.records
    }
    validate_product_category_assignment_set(
        assignment_set,
        registry=registry,
        expected_product_ids=set(terminal_node_ids_by_product),
        terminal_node_ids_by_product=terminal_node_ids_by_product,
    )
    if enforce_official_gate:
        validate_official_p0_assignments(assignment_set)
    return CategoryCandidateBuild(
        builder_version=proposal.builder_version,
        registry=registry,
        assignments=assignment_set,
    )


def category_scope_selection_template(
    proposal: CategoryGraphProposal,
) -> dict[str, object]:
    """Return a deliberately non-selection template for human review."""

    root_node_ids = tuple(sorted(node.id for node in proposal.nodes if node.parent_id is None))
    return {
        "schema": CATEGORY_SCOPE_SELECTION_TEMPLATE_SCHEMA,
        "catalog_id": proposal.catalog_id,
        "category_graph_id": proposal.category_graph_id,
        "builder_version": proposal.builder_version,
        "required_root_node_ids": list(root_node_ids),
        "instructions": (
            "Create a shopping-copilot/category-scope-selection/v0 document; "
            "include exactly one reviewed root scope using required_root_node_ids, "
            "then add only human-approved user-facing scopes."
        ),
        "scopes": [],
    }


def reviewed_category_scopes_candidate_document(
    candidate: CategoryCandidateBuild,
) -> dict[str, object]:
    """Return the reviewed-config fragment that CS6 must copy exactly."""

    return {
        "schema": CATEGORY_SCOPES_CANDIDATE_SCHEMA,
        "catalog_id": candidate.registry.catalog_id,
        "category_graph_id": candidate.registry.category_graph_id,
        "builder_version": candidate.builder_version,
        "category_scopes": list(candidate.registry.scopes),
    }


def _validate_selection_pins(
    selection: CategoryScopeSelectionDocument,
    proposal: CategoryGraphProposal,
) -> None:
    if selection.catalog_id != proposal.catalog_id:
        raise CategorySelectionError("scope selection catalog_id is stale")
    if selection.category_graph_id != proposal.category_graph_id:
        raise CategorySelectionError("scope selection category_graph_id is stale")
    if selection.builder_version != proposal.builder_version:
        raise CategorySelectionError("scope selection builder_version is unsupported")
