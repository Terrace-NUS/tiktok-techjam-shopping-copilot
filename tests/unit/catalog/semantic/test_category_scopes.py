from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from shopping_copilot.catalog.semantic.category import (
    CATEGORY_REGISTRY_SCHEMA,
    CategoryNode,
    CategoryRegistry,
    category_graph_id,
    category_node_id,
    category_scope_id,
    materialize_category_scope,
    validate_category_registry,
    validate_category_scope,
)
from shopping_copilot.catalog.semantic.errors import CategoryBuildError

CATALOG_ID = f"sha256:{'b' * 64}"


def _node(path: tuple[str, ...]) -> CategoryNode:
    return CategoryNode(
        id=category_node_id(path),
        parent_id=None if len(path) == 1 else category_node_id(path[:-1]),
        canonical_path=path,
    )


def _forest() -> tuple[CategoryNode, ...]:
    paths = (
        ("catalog-a",),
        ("catalog-a", "clothing"),
        ("catalog-a", "clothing", "shirts"),
        ("catalog-a", "clothing", "shoes"),
        ("catalog-a", "clothing", "shoes", "loafers"),
        ("catalog-b",),
        ("catalog-b", "shoes"),
        ("catalog-b", "shoes", "sneakers"),
    )
    return tuple(sorted((_node(path) for path in paths), key=lambda node: node.id))


def _by_path(nodes: tuple[CategoryNode, ...]) -> dict[tuple[str, ...], CategoryNode]:
    return {node.canonical_path: node for node in nodes}


def _root_scope(nodes: tuple[CategoryNode, ...], graph_id: str):
    roots = tuple(sorted(node.id for node in nodes if node.parent_id is None))
    return materialize_category_scope(
        graph_id=graph_id,
        nodes=nodes,
        label="All products",
        root_node_ids=roots,
    )


def test_scope_id_uses_the_exact_contract_preimage() -> None:
    nodes = _forest()
    graph_id = category_graph_id(CATALOG_ID, nodes)
    root_id = _by_path(nodes)[("catalog-a", "clothing")].id
    preimage = f'{{"category_graph_id":"{graph_id}","root_node_ids":["{root_id}"]}}'.encode()

    assert category_scope_id(graph_id, (root_id,)) == (f"cs_{hashlib.sha256(preimage).hexdigest()}")


def test_scope_materialization_is_the_exact_union_of_complete_subtrees() -> None:
    nodes = _forest()
    by_path = _by_path(nodes)
    graph_id = category_graph_id(CATALOG_ID, nodes)
    roots = tuple(
        sorted(
            (
                by_path[("catalog-a", "clothing", "shoes")].id,
                by_path[("catalog-b", "shoes")].id,
            )
        )
    )

    scope = materialize_category_scope(
        graph_id=graph_id,
        nodes=nodes,
        label="Shoes",
        root_node_ids=roots,
    )
    expected_paths = {
        ("catalog-a", "clothing", "shoes"),
        ("catalog-a", "clothing", "shoes", "loafers"),
        ("catalog-b", "shoes"),
        ("catalog-b", "shoes", "sneakers"),
    }

    assert scope.member_node_ids == tuple(sorted(by_path[path].id for path in expected_paths))
    validate_category_scope(scope, graph_id=graph_id, nodes=nodes)

    incomplete = replace(scope, member_node_ids=scope.member_node_ids[:-1])
    with pytest.raises(CategoryBuildError, match="exact materialization"):
        validate_category_scope(incomplete, graph_id=graph_id, nodes=nodes)


def test_scope_materialization_rejects_unknown_and_redundant_roots() -> None:
    nodes = _forest()
    by_path = _by_path(nodes)
    graph_id = category_graph_id(CATALOG_ID, nodes)

    with pytest.raises(CategoryBuildError, match="unknown root"):
        materialize_category_scope(
            graph_id=graph_id,
            nodes=nodes,
            label="Unknown",
            root_node_ids=(f"cn_{'f' * 64}",),
        )

    redundant = tuple(
        sorted(
            (
                by_path[("catalog-a", "clothing")].id,
                by_path[("catalog-a", "clothing", "shoes")].id,
            )
        )
    )
    with pytest.raises(CategoryBuildError, match="redundant"):
        materialize_category_scope(
            graph_id=graph_id,
            nodes=nodes,
            label="Redundant",
            root_node_ids=redundant,
        )


def test_overlapping_and_refinement_scopes_coexist_but_equal_membership_is_rejected() -> None:
    nodes = _forest()
    by_path = _by_path(nodes)
    graph_id = category_graph_id(CATALOG_ID, nodes)
    root_scope = _root_scope(nodes, graph_id)
    clothing = materialize_category_scope(
        graph_id=graph_id,
        nodes=nodes,
        label="Clothing",
        root_node_ids=(by_path[("catalog-a", "clothing")].id,),
    )
    shoes = materialize_category_scope(
        graph_id=graph_id,
        nodes=nodes,
        label="Shoes",
        root_node_ids=(by_path[("catalog-a", "clothing", "shoes")].id,),
    )
    loafers = materialize_category_scope(
        graph_id=graph_id,
        nodes=nodes,
        label="Loafers",
        root_node_ids=(by_path[("catalog-a", "clothing", "shoes", "loafers")].id,),
    )

    assert set(loafers.member_node_ids) < set(shoes.member_node_ids)
    assert set(shoes.member_node_ids) < set(clothing.member_node_ids)

    registry = CategoryRegistry(
        schema=CATEGORY_REGISTRY_SCHEMA,
        catalog_id=CATALOG_ID,
        category_graph_id=graph_id,
        root_scope_id=root_scope.id,
        nodes=nodes,
        scopes=tuple(sorted((root_scope, clothing, shoes, loafers), key=lambda item: item.id)),
    )
    validate_category_registry(registry)

    duplicate_membership = replace(shoes, label="Footwear")
    invalid_registry = replace(
        registry,
        scopes=tuple(
            sorted(
                (*registry.scopes, duplicate_membership),
                key=lambda item: item.id,
            )
        ),
    )
    with pytest.raises(CategoryBuildError, match="unique"):
        validate_category_registry(invalid_registry)
