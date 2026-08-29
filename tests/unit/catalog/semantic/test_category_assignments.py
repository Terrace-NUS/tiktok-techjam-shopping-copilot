from __future__ import annotations

from dataclasses import replace

import pytest

import shopping_copilot.catalog.semantic.category.validation as category_validation
from shopping_copilot.catalog.semantic.category import (
    CATEGORY_REGISTRY_SCHEMA,
    PRODUCT_CATEGORY_ASSIGNMENT_SCHEMA,
    CategoryMatchResult,
    CategoryNode,
    CategoryRegistry,
    ProductCategoryAssignment,
    ProductCategoryAssignmentSet,
    ProductCategoryAssignmentStatus,
    category_graph_id,
    category_node_id,
    match_category,
    materialize_category_scope,
    validate_official_p0_assignments,
    validate_product_category_assignment_set,
)
from shopping_copilot.catalog.semantic.errors import CategoryBuildError

CATALOG_ID = f"sha256:{'c' * 64}"


def _node(path: tuple[str, ...]) -> CategoryNode:
    return CategoryNode(
        id=category_node_id(path),
        parent_id=None if len(path) == 1 else category_node_id(path[:-1]),
        canonical_path=path,
    )


def _registry_and_leaf_scopes():
    paths = (("catalog",), ("catalog", "left"), ("catalog", "right"))
    nodes = tuple(sorted((_node(path) for path in paths), key=lambda node: node.id))
    by_path = {node.canonical_path: node for node in nodes}
    graph_id = category_graph_id(CATALOG_ID, nodes)
    root_scope = materialize_category_scope(
        graph_id=graph_id,
        nodes=nodes,
        label="All products",
        root_node_ids=(by_path[("catalog",)].id,),
    )
    left_scope = materialize_category_scope(
        graph_id=graph_id,
        nodes=nodes,
        label="Left",
        root_node_ids=(by_path[("catalog", "left")].id,),
    )
    registry = CategoryRegistry(
        schema=CATEGORY_REGISTRY_SCHEMA,
        catalog_id=CATALOG_ID,
        category_graph_id=graph_id,
        root_scope_id=root_scope.id,
        nodes=nodes,
        scopes=tuple(sorted((root_scope, left_scope), key=lambda scope: scope.id)),
    )
    return registry, left_scope, by_path[("catalog", "left")].id, by_path[("catalog", "right")].id


def _assignment_set(
    registry: CategoryRegistry,
    assignments: tuple[ProductCategoryAssignment, ...],
) -> ProductCategoryAssignmentSet:
    return ProductCategoryAssignmentSet(
        schema=PRODUCT_CATEGORY_ASSIGNMENT_SCHEMA,
        catalog_id=registry.catalog_id,
        category_graph_id=registry.category_graph_id,
        assignments=assignments,
    )


def test_generic_assignment_validation_accepts_known_unknown_and_conflict() -> None:
    registry, _, left_id, right_id = _registry_and_leaf_scopes()
    conflict_ids = tuple(sorted((left_id, right_id)))
    assignments = (
        ProductCategoryAssignment(
            parent_asin="A",
            status=ProductCategoryAssignmentStatus.KNOWN,
            leaf_node_ids=(left_id,),
        ),
        ProductCategoryAssignment(
            parent_asin="B",
            status=ProductCategoryAssignmentStatus.UNKNOWN,
            leaf_node_ids=(),
        ),
        ProductCategoryAssignment(
            parent_asin="C",
            status=ProductCategoryAssignmentStatus.CONFLICT,
            leaf_node_ids=conflict_ids,
        ),
    )
    assignment_set = _assignment_set(registry, assignments)

    validate_product_category_assignment_set(
        assignment_set,
        registry=registry,
        expected_product_ids={"A", "B", "C"},
        terminal_node_ids_by_product={
            "A": {left_id},
            "B": set(),
            "C": {left_id, right_id},
        },
    )


@pytest.mark.parametrize(
    ("status", "leaf_ids", "message"),
    (
        (ProductCategoryAssignmentStatus.KNOWN, (), "non-empty"),
        (ProductCategoryAssignmentStatus.UNKNOWN, (f"cn_{'1' * 64}",), "must not contain"),
        (ProductCategoryAssignmentStatus.CONFLICT, (f"cn_{'1' * 64}",), "at least two"),
    ),
)
def test_assignment_status_shape_is_enforced(
    status: ProductCategoryAssignmentStatus,
    leaf_ids: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ProductCategoryAssignment(
            parent_asin="A",
            status=status,
            leaf_node_ids=leaf_ids,
        )


def test_category_matcher_preserves_all_three_assignment_states() -> None:
    _, left_scope, left_id, right_id = _registry_and_leaf_scopes()
    known_inside = ProductCategoryAssignment(
        parent_asin="A",
        status=ProductCategoryAssignmentStatus.KNOWN,
        leaf_node_ids=(left_id,),
    )
    known_outside = ProductCategoryAssignment(
        parent_asin="B",
        status=ProductCategoryAssignmentStatus.KNOWN,
        leaf_node_ids=(right_id,),
    )
    unknown = ProductCategoryAssignment(
        parent_asin="C",
        status=ProductCategoryAssignmentStatus.UNKNOWN,
        leaf_node_ids=(),
    )
    conflict = ProductCategoryAssignment(
        parent_asin="D",
        status=ProductCategoryAssignmentStatus.CONFLICT,
        leaf_node_ids=tuple(sorted((left_id, right_id))),
    )

    assert match_category(known_inside, left_scope) is CategoryMatchResult.SATISFIED
    assert match_category(known_outside, left_scope) is CategoryMatchResult.VIOLATED
    assert match_category(unknown, left_scope) is CategoryMatchResult.UNKNOWN
    assert match_category(conflict, left_scope) is CategoryMatchResult.UNKNOWN


def test_official_gate_is_stricter_than_generic_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, _, left_id, right_id = _registry_and_leaf_scopes()
    generic_assignments = (
        ProductCategoryAssignment(
            parent_asin="A",
            status=ProductCategoryAssignmentStatus.KNOWN,
            leaf_node_ids=(left_id,),
        ),
        ProductCategoryAssignment(
            parent_asin="B",
            status=ProductCategoryAssignmentStatus.UNKNOWN,
            leaf_node_ids=(),
        ),
        ProductCategoryAssignment(
            parent_asin="C",
            status=ProductCategoryAssignmentStatus.CONFLICT,
            leaf_node_ids=tuple(sorted((left_id, right_id))),
        ),
    )
    generic_set = _assignment_set(registry, generic_assignments)
    validate_product_category_assignment_set(generic_set, registry=registry)

    with pytest.raises(CategoryBuildError, match="exactly 50000"):
        validate_official_p0_assignments(generic_set)

    monkeypatch.setattr(category_validation, "OFFICIAL_PRODUCT_COUNT", len(generic_assignments))
    with pytest.raises(CategoryBuildError, match="every category assignment to be KNOWN"):
        validate_official_p0_assignments(generic_set)

    all_known_set = replace(
        generic_set,
        assignments=tuple(
            replace(
                assignment,
                status=ProductCategoryAssignmentStatus.KNOWN,
                leaf_node_ids=(left_id,),
            )
            for assignment in generic_set.assignments
        ),
    )
    validate_official_p0_assignments(all_known_set)
