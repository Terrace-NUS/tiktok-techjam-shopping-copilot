from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

import shopping_copilot.catalog.semantic.category.normalization as category_normalization
from shopping_copilot.catalog.semantic import canonical_json_bytes
from shopping_copilot.catalog.semantic.category import (
    CategoryNode,
    build_category_graph_proposal,
    category_graph_core_document,
    category_graph_id,
    category_node_id,
    normalize_category_segment,
    validate_category_nodes,
)
from shopping_copilot.catalog.semantic.errors import CategoryBuildError
from shopping_copilot.catalog.semantic.raw_catalog import (
    RAW_CATALOG_SCHEMA,
    RawCatalogCategoryRecord,
    RawCatalogScan,
)

CATALOG_ID = f"sha256:{'a' * 64}"


def _node(path: tuple[str, ...]) -> CategoryNode:
    return CategoryNode(
        id=category_node_id(path),
        parent_id=None if len(path) == 1 else category_node_id(path[:-1]),
        canonical_path=path,
    )


def _nodes(*paths: tuple[str, ...]) -> tuple[CategoryNode, ...]:
    return tuple(sorted((_node(path) for path in paths), key=lambda node: node.id))


def test_category_node_id_uses_the_exact_contract_preimage() -> None:
    path = ("clothing", "men")
    preimage = b'{"canonical_path":["clothing","men"]}'

    assert canonical_json_bytes({"canonical_path": list(path)}) == preimage
    assert category_node_id(path) == f"cn_{hashlib.sha256(preimage).hexdigest()}"


def test_category_graph_id_uses_the_exact_sorted_node_document() -> None:
    nodes = _nodes(("catalog",), ("catalog", "shoes"))
    expected_document = {
        "schema": "shopping-copilot/category-graph-core/v0",
        "catalog_id": CATALOG_ID,
        "nodes": [
            {
                "id": node.id,
                "parent_id": node.parent_id,
                "canonical_path": list(node.canonical_path),
            }
            for node in nodes
        ],
    }
    expected_id = f"sha256:{hashlib.sha256(canonical_json_bytes(expected_document)).hexdigest()}"

    assert category_graph_core_document(CATALOG_ID, nodes) == expected_document
    assert category_graph_id(CATALOG_ID, nodes) == expected_id


def test_normalization_is_nfkc_trim_whitespace_collapse_and_casefold() -> None:
    assert normalize_category_segment("  ＳＨＯＥＳ\t Accessories  ") == ("shoes accessories")
    assert normalize_category_segment("Straße") == "strasse"
    assert normalize_category_segment("\ua7f1") == "s"
    assert normalize_category_segment("\u1c89") == "\u1c8a"
    assert normalize_category_segment("left\x1cright") == "left right"

    with pytest.raises(CategoryBuildError, match="control character"):
        normalize_category_segment("left\x00right")


def test_normalizer_fails_closed_when_pinned_unicode_components_disagree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(category_normalization, "UCD_VERSION", "unexpected")

    with pytest.raises(CategoryBuildError, match="backend version differs"):
        category_normalization.ensure_category_builder_runtime()


def test_normalizer_pins_casefold_content_and_exact_whitespace_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert category_normalization._OBSERVED_CASEFOLD_TABLE_SHA256 == (
        category_normalization._PINNED_CASEFOLD_TABLE_SHA256
    )
    assert category_normalization._WHITESPACE_CODEPOINTS == frozenset(
        {
            *range(0x0009, 0x000E),
            *range(0x001C, 0x0021),
            0x0085,
            0x00A0,
            0x1680,
            *range(0x2000, 0x200B),
            0x2028,
            0x2029,
            0x202F,
            0x205F,
            0x3000,
        }
    )
    assert category_normalization._OBSERVED_WHITESPACE_CODEPOINTS_SHA256 == (
        category_normalization._PINNED_WHITESPACE_CODEPOINTS_SHA256
    )

    monkeypatch.setattr(category_normalization, "_OBSERVED_CASEFOLD_TABLE_SHA256", "0" * 64)
    with pytest.raises(CategoryBuildError, match="full-casefold table differs"):
        category_normalization.ensure_category_builder_runtime()


def test_normalizer_fails_closed_when_whitespace_domain_digest_disagrees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        category_normalization,
        "_OBSERVED_WHITESPACE_CODEPOINTS_SHA256",
        "0" * 64,
    )

    with pytest.raises(CategoryBuildError, match="whitespace domain differs"):
        category_normalization.ensure_category_builder_runtime()


def test_graph_proposal_normalizes_prefixes_sets_immediate_parents_and_sorts_ids() -> None:
    scan = RawCatalogScan(
        schema=RAW_CATALOG_SCHEMA,
        catalog_id=CATALOG_ID,
        byte_size=123,
        product_count=3,
        records=(
            RawCatalogCategoryRecord(
                parent_asin="A",
                raw_path=("  Ｓhoes  ", " MEN "),
            ),
            RawCatalogCategoryRecord(
                parent_asin="B",
                raw_path=("shoes", "men", "Loafers"),
            ),
            RawCatalogCategoryRecord(
                parent_asin="C",
                raw_path=("Accessories",),
            ),
        ),
    )

    proposal = build_category_graph_proposal(scan)
    assert tuple(node.id for node in proposal.nodes) == tuple(
        sorted(node.id for node in proposal.nodes)
    )

    by_path = {node.canonical_path: node for node in proposal.nodes}
    assert set(by_path) == {
        ("accessories",),
        ("shoes",),
        ("shoes", "men"),
        ("shoes", "men", "loafers"),
    }
    assert by_path[("shoes",)].parent_id is None
    assert by_path[("shoes", "men")].parent_id == by_path[("shoes",)].id
    assert by_path[("shoes", "men", "loafers")].parent_id == by_path[("shoes", "men")].id
    assert proposal.raw_prefix_count == 6
    assert {collision.canonical_path for collision in proposal.collisions} == {
        ("shoes",),
        ("shoes", "men"),
    }
    assert all(len(collision.raw_paths) == 2 for collision in proposal.collisions)
    validate_category_nodes(proposal.nodes)


def test_graph_validation_rejects_noncanonical_order_and_wrong_parent() -> None:
    nodes = _nodes(
        ("catalog",),
        ("catalog", "clothing"),
        ("catalog", "clothing", "shoes"),
    )

    with pytest.raises(CategoryBuildError, match="already be sorted"):
        validate_category_nodes(tuple(reversed(nodes)))

    child = next(node for node in nodes if len(node.canonical_path) == 3)
    wrong_child = replace(child, parent_id=None)
    wrong_nodes = tuple(
        sorted(
            (wrong_child if node.id == child.id else node for node in nodes),
            key=lambda node: node.id,
        )
    )
    with pytest.raises(CategoryBuildError, match="immediate prefix"):
        validate_category_nodes(wrong_nodes)
