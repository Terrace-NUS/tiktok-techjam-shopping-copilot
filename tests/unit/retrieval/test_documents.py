from __future__ import annotations

import json
from pathlib import Path

import pytest

from shopping_copilot.retrieval.documents import (
    DOCUMENT_FIELD_ORDER,
    FIELD_CHARACTER_LIMITS,
    ProductDocumentError,
    load_product_documents,
)


def _valid_product(parent_asin: str = "A") -> dict[str, object]:
    return {
        "parent_asin": parent_asin,
        "title": "Product",
        "categories": ["Root", "Category"],
        "store": "Store",
        "features": ["Feature"],
        "details": {"Color": "Blue"},
        "description": ["Description"],
    }


def _write_rows(path: Path, rows: list[object]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows)
        + "\n",
        encoding="utf-8",
        newline="",
    )


def test_documents_are_read_only_deterministic_and_explicitly_labeled(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    first = {
        "description": ["  Good\tfor walking ", "", " Daily   wear"],
        "details": {
            "Sizes": [" 8 ", "9"],
            "Nested": {"z": " last ", "a": " first\nvalue "},
            "Empty": None,
            "Color": " Red ",
        },
        "features": [" Breathable ", "Non\nslip"],
        "store": " Shop\tName ",
        "categories": [" Root ", " Shoes "],
        "title": "  Fancy\t Shoe  ",
        "parent_asin": "B",
    }
    second = _valid_product("A")
    second["store"] = None
    _write_rows(catalog, [first, second])
    before = catalog.read_bytes()

    documents_a = load_product_documents(catalog)
    documents_b = load_product_documents(catalog)

    assert documents_a == documents_b
    assert catalog.read_bytes() == before
    assert [document.parent_asin for document in documents_a] == ["B", "A"]
    assert documents_a[0].text == (
        "title: Fancy Shoe\n"
        "categories: Root > Shoes\n"
        "store: Shop Name\n"
        "features: Breathable | Non slip\n"
        "details: Color: Red | Nested: {a: first value; z: last} | Sizes: 8, 9\n"
        "description: Good for walking | Daily wear"
    )
    assert documents_a[1].text.splitlines()[2] == "store: "
    assert tuple(line.split(":", 1)[0] for line in documents_a[0].text.splitlines()) == (
        DOCUMENT_FIELD_ORDER
    )


def test_every_field_is_normalized_then_bounded_by_its_fixed_character_limit(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "catalog.jsonl"
    row = _valid_product()
    row.update(
        {
            "title": " word\t" * 1_000,
            "categories": [" category " * 1_000],
            "store": " store\n" * 1_000,
            "features": [" feature\t" * 1_000],
            "details": {"Long key": " detail\n" * 2_000},
            "description": [" description\t" * 2_000],
        }
    )
    _write_rows(catalog, [row])

    document = load_product_documents(catalog)[0]

    for line, field in zip(document.text.splitlines(), DOCUMENT_FIELD_ORDER, strict=True):
        label, value = line.split(": ", 1)
        assert label == field
        assert len(value) <= FIELD_CHARACTER_LIMITS[field]
        assert "\t" not in value
        assert "\n" not in value
        assert "  " not in value


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ([], "row must be an object"),
        ({key: value for key, value in _valid_product().items() if key != "title"}, "title"),
        ({**_valid_product(), "parent_asin": None}, "parent_asin must be a string"),
        ({**_valid_product(), "parent_asin": " \t "}, "parent_asin must not be empty"),
        ({**_valid_product(), "title": []}, "title must be a string"),
        ({**_valid_product(), "categories": "Root"}, "categories must be an array"),
        ({**_valid_product(), "categories": ["Root", 7]}, r"categories\[1\]"),
        ({**_valid_product(), "store": 7}, "store must be a string or null"),
        ({**_valid_product(), "features": [False]}, r"features\[0\]"),
        ({**_valid_product(), "details": []}, "details must be an object"),
        ({**_valid_product(), "description": None}, "description must be an array"),
    ],
)
def test_non_objects_missing_fields_and_wrong_field_types_are_rejected(
    tmp_path: Path, row: object, message: str
) -> None:
    catalog = tmp_path / "catalog.jsonl"
    _write_rows(catalog, [row])

    with pytest.raises(ProductDocumentError, match=message) as raised:
        load_product_documents(catalog)

    assert raised.value.line_number == 1


def test_duplicate_parent_asin_is_rejected_with_both_line_numbers(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    _write_rows(catalog, [_valid_product("DUPLICATE"), _valid_product("DUPLICATE")])

    with pytest.raises(ProductDocumentError, match="first seen on line 1") as raised:
        load_product_documents(catalog)

    assert raised.value.line_number == 2


def test_invalid_json_blank_rows_and_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    invalid_rows = {
        "invalid.jsonl": b"not json\n",
        "blank.jsonl": b" \t\r\n",
        "duplicate-key.jsonl": (
            b'{"parent_asin":"A","parent_asin":"B","title":"x",'
            b'"categories":[],"store":null,"features":[],"details":{},'
            b'"description":[]}\n'
        ),
    }

    for filename, payload in invalid_rows.items():
        catalog = tmp_path / filename
        catalog.write_bytes(payload)
        with pytest.raises(ProductDocumentError) as raised:
            load_product_documents(catalog)
        assert raised.value.line_number == 1


def test_expected_parent_asin_set_must_match_exactly(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    _write_rows(catalog, [_valid_product("A"), _valid_product("B")])

    assert len(load_product_documents(catalog, expected_parent_asins={"A", "B"})) == 2

    with pytest.raises(ProductDocumentError) as raised:
        load_product_documents(catalog, expected_parent_asins={"A", "C"})

    assert raised.value.line_number is None
    assert "missing=1 ['C']" in str(raised.value)
    assert "unexpected=1 ['B']" in str(raised.value)


def test_expected_parent_asins_itself_is_validated(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    _write_rows(catalog, [_valid_product()])

    with pytest.raises(TypeError, match="must be a set"):
        load_product_documents(catalog, expected_parent_asins=["A"])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="only strings"):
        load_product_documents(catalog, expected_parent_asins={1})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        load_product_documents(catalog, expected_parent_asins={" "})
