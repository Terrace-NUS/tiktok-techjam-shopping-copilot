from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from shopping_copilot.retrieval.evidence import (
    RETRIEVAL_EVIDENCE_POLICY_ID,
    RetrievalEvidenceError,
    RetrievalEvidenceIndex,
    build_retrieval_evidence_index,
)

RELEASE_ID = "sha256:" + "b" * 64


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows)
        + "\n",
        encoding="utf-8",
        newline="",
    )


def _rich_rows() -> list[dict[str, object]]:
    return [
        {
            "parent_asin": "A",
            "title": "Phantombrand Unisex Scarlet Linen Boho Sneaker Model 314 Size 7",
            "store": "North Star",
            "categories": ["Women", "Athletic Shoes", "Trail Running"],
            "features": [
                "Organic cotton construction",
                "Waterproof for beach weddings",
                "Colour block finish",
            ],
            "description": [
                "Useful for festival dancing",
                "A bamboo-inspired story in violet packaging",
            ],
            "details": {
                "Brand Name": "Atelier One",
                "Color": ["Slate Grey", "Ocean Blue"],
                "Department": "Womens Shoes",
                "Target Gender": "Female",
                "Material": "Leather",
                "Fabric": "Silk",
                "Outer Material": "Suede",
                "Sole Material": "Rubber",
                "Size": ["8 Wide", "Medium"],
                "Style": "Bohemian",
                "Pattern": "Striped",
                "Theme": "Western",
                "Closure Type": "Lace Up",
                "Special Feature": ["Arch Support", "Water Resistant"],
                "Occasion": "Ceremony",
                "Recommended Uses": "Trail Running",
                "Package Dimensions": "Small package, 99 inches",
                "Product Dimensions": "42 inches",
            },
        },
        {
            "parent_asin": "B",
            "title": "Redwood Display Stand 42",
            "store": None,
            "categories": ["Accessories"],
            "features": ["Not waterproof", "Non stain proof"],
            "description": ["No beach use", "Without trail running support"],
            "details": {
                "Special Features": "Not wind resistant",
                "Occasion": "Not hiking",
                "Package Dimensions": "8 x 9 x 99 inches",
                "Product Dimensions": "42 inches",
            },
        },
    ]


def _catalog_id(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _build(path: Path) -> RetrievalEvidenceIndex:
    return build_retrieval_evidence_index(
        path,
        catalog_id=_catalog_id(path),
        catalog_semantic_release_id=RELEASE_ID,
        expected_parent_asins={"A", "B"},
    )


def test_each_facet_uses_only_its_frozen_raw_source_boundaries(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    _write_rows(catalog, _rich_rows())
    index = _build(catalog)

    assert index.match("brand", "north star") == frozenset({"A"})
    assert index.match("brand", "atelier one") == frozenset({"A"})
    assert index.match("brand", "phantombrand") == frozenset()

    assert index.match("color", "slate gray") == frozenset({"A"})
    assert index.match("color", "scarlet") == frozenset({"A"})
    assert index.match("color", "violet") == frozenset()

    assert index.match("department", "women") == frozenset({"A"})
    assert index.match("department", "athletic shoes") == frozenset({"A"})
    assert index.match("department", "phantombrand") == frozenset()

    assert index.match("gender", "womens") == frozenset({"A"})
    assert index.match("gender", "female") == frozenset({"A"})
    assert index.match("gender", "unisex") == frozenset()

    for material in ("leather", "silk", "suede", "rubber", "linen", "cotton"):
        assert index.match("material", material) == frozenset({"A"})
    assert index.match("material", "bamboo") == frozenset()

    assert index.match("size", "7") == frozenset({"A"})
    assert index.match("size", "8 wide") == frozenset({"A"})
    assert index.match("size", "medium") == frozenset({"A"})

    assert index.match("style", "boho") == frozenset({"A"})
    assert index.match("style", "striped") == frozenset({"A"})
    assert index.match("style", "minimalist") == frozenset()

    assert index.match("feature", "arch support") == frozenset({"A"})
    assert index.match("feature", "festival dancing") == frozenset({"A"})
    assert index.match("feature", "phantombrand") == frozenset()

    assert index.match("use_case", "trail running") == frozenset({"A"})
    assert index.match("use_case", "ceremony") == frozenset({"A"})
    assert index.match("use_case", "festival dancing") == frozenset({"A"})
    assert index.match("use_case", "phantombrand") == frozenset()


def test_aliases_boundaries_phrases_and_multivalues_are_deterministic(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    _write_rows(catalog, _rich_rows())
    index = _build(catalog)

    assert index.match("color", "slate grey") == frozenset({"A"})
    assert index.match("color", "ocean blue") == frozenset({"A"})
    assert index.match("color", "gray ocean") == frozenset()
    assert index.match("color", "red") == frozenset()

    assert index.match("feature", "colour block") == frozenset({"A"})
    assert index.match("material", "organic cotton") == frozenset({"A"})
    assert index.match("material", "organic construction") == frozenset()
    assert index.match("brand", "star north") == frozenset()


def test_local_negation_is_not_positive_feature_or_use_case_evidence(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    _write_rows(catalog, _rich_rows())
    index = _build(catalog)

    assert index.match("feature", "waterproof") == frozenset({"A"})
    assert index.match("feature", "stain proof") == frozenset()
    assert index.match("feature", "wind resistant") == frozenset()
    assert index.match("use_case", "beach") == frozenset({"A"})
    assert index.match("use_case", "trail running") == frozenset({"A"})
    assert index.match("use_case", "hiking") == frozenset()


def test_size_rejects_dimensions_and_title_numbers_without_size_context(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    _write_rows(catalog, _rich_rows())
    index = _build(catalog)

    assert index.match("size", "99") == frozenset()
    assert index.match("size", "42") == frozenset()
    assert index.match("size", "314") == frozenset()
    assert index.match("size", "small package") == frozenset()


def test_duplicates_unknown_facets_and_expected_set_mismatches_fail(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.jsonl"
    _write_rows(duplicate, [{"parent_asin": "A"}, {"parent_asin": "A"}])
    with pytest.raises(RetrievalEvidenceError, match="duplicate parent_asin"):
        build_retrieval_evidence_index(
            duplicate,
            catalog_id=_catalog_id(duplicate),
            catalog_semantic_release_id=RELEASE_ID,
        )

    catalog = tmp_path / "catalog.jsonl"
    _write_rows(catalog, [{"parent_asin": "A"}, {"parent_asin": "B"}])
    with pytest.raises(RetrievalEvidenceError, match="set mismatch"):
        build_retrieval_evidence_index(
            catalog,
            catalog_id=_catalog_id(catalog),
            catalog_semantic_release_id=RELEASE_ID,
            expected_parent_asins={"A", "UNKNOWN"},
        )

    index = build_retrieval_evidence_index(
        catalog,
        catalog_id=_catalog_id(catalog),
        catalog_semantic_release_id=RELEASE_ID,
    )
    with pytest.raises(ValueError, match="unknown facet"):
        index.match("price", "cheap")
    with pytest.raises(ValueError, match="searchable token"):
        index.match("brand", " -- ")

    with pytest.raises(RetrievalEvidenceError, match="cannot read catalog"):
        build_retrieval_evidence_index(
            tmp_path / "missing.jsonl",
            catalog_id="sha256:" + "0" * 64,
            catalog_semantic_release_id=RELEASE_ID,
        )

    with pytest.raises(RetrievalEvidenceError, match="do not match catalog_id"):
        build_retrieval_evidence_index(
            catalog,
            catalog_id="sha256:" + "0" * 64,
            catalog_semantic_release_id=RELEASE_ID,
        )


def test_build_is_read_only_and_index_identity_is_stable_and_content_bound(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "catalog.jsonl"
    rows = list(reversed(_rich_rows()))
    _write_rows(catalog, rows)
    before = catalog.read_bytes()

    first = _build(catalog)
    second = _build(catalog)

    assert catalog.read_bytes() == before
    assert first.parent_asins == ("A", "B")
    assert first.catalog_id == _catalog_id(catalog)
    assert first.catalog_semantic_release_id == RELEASE_ID
    assert first.policy_id == RETRIEVAL_EVIDENCE_POLICY_ID
    assert first.index_id == second.index_id
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", first.index_id)

    reordered = tmp_path / "reordered.jsonl"
    _write_rows(reordered, list(reversed(rows)))
    reordered_index = _build(reordered)
    assert reordered_index.match("brand", "north star") == first.match("brand", "north star")
    assert reordered_index.index_id != first.index_id

    other_release_id = build_retrieval_evidence_index(
        catalog,
        catalog_id=_catalog_id(catalog),
        catalog_semantic_release_id="sha256:" + "c" * 64,
        expected_parent_asins={"A", "B"},
    )
    assert other_release_id.index_id != first.index_id

    changed = tmp_path / "changed.jsonl"
    changed_rows = _rich_rows()
    changed_rows[0]["store"] = "Different Brand"
    _write_rows(changed, changed_rows)
    assert _build(changed).index_id != first.index_id
