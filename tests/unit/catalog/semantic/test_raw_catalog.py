from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from shopping_copilot.catalog.semantic.canonical import (
    IJSON_SAFE_INTEGER_MAX,
    IJSON_SAFE_INTEGER_MIN,
)
from shopping_copilot.catalog.semantic.errors import RawCatalogValidationError
from shopping_copilot.catalog.semantic.raw_catalog import (
    RAW_CATALOG_SCHEMA,
    scan_raw_catalog,
)


def _row_bytes(
    parent_asin: str = "p1",
    *,
    categories: object = ("Root", "Shoes"),
    details: object = None,
    **extra: object,
) -> bytes:
    row: dict[str, object] = {
        "parent_asin": parent_asin,
        "categories": list(categories) if type(categories) is tuple else categories,
        "details": {} if details is None else details,
        **extra,
    }
    return json.dumps(
        row,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_catalog(tmp_path: Path, payload: bytes) -> Path:
    path = tmp_path / "catalog.jsonl"
    path.write_bytes(payload)
    return path


def _assert_invalid(
    tmp_path: Path,
    payload: bytes,
    *,
    reason: str,
    line_number: int | None = 1,
    expected_product_count: int = 1,
) -> None:
    path = _write_catalog(tmp_path, payload)

    with pytest.raises(RawCatalogValidationError) as captured:
        scan_raw_catalog(path, expected_product_count=expected_product_count)

    assert reason in captured.value.reason
    assert captured.value.line_number == line_number


def test_scan_raw_catalog_preserves_exact_identity_and_raw_category_text(tmp_path: Path) -> None:
    first = _row_bytes(
        "p1",
        categories=(" Root ", "\u978b"),
        details={"Nested": {"value": IJSON_SAFE_INTEGER_MAX}},
        arbitrary=[True, None, "\u96ea"],
    )
    second = _row_bytes(
        "p2",
        categories=("Root", "Clothing"),
        details={},
        minimum=IJSON_SAFE_INTEGER_MIN,
    )
    payload = first + b"\n" + second
    path = _write_catalog(tmp_path, payload)

    scan = scan_raw_catalog(path, expected_product_count=2)

    assert scan.schema == RAW_CATALOG_SCHEMA
    assert scan.catalog_id == f"sha256:{hashlib.sha256(payload).hexdigest()}"
    assert scan.byte_size == len(payload)
    assert scan.product_count == 2
    assert tuple(record.parent_asin for record in scan.records) == ("p1", "p2")
    assert scan.records[0].raw_path == (" Root ", "\u978b")
    assert path.read_bytes() == payload


def test_scan_raw_catalog_accepts_one_terminal_newline_without_counting_blank_row(
    tmp_path: Path,
) -> None:
    payload = _row_bytes() + b"\n"
    path = _write_catalog(tmp_path, payload)

    scan = scan_raw_catalog(path, expected_product_count=1)

    assert scan.product_count == 1
    assert scan.byte_size == len(payload)
    assert scan.catalog_id == f"sha256:{hashlib.sha256(payload).hexdigest()}"


def test_scan_raw_catalog_enforces_exact_expected_product_count(tmp_path: Path) -> None:
    path = _write_catalog(tmp_path, _row_bytes())

    with pytest.raises(RawCatalogValidationError) as captured:
        scan_raw_catalog(path, expected_product_count=2)

    assert captured.value.reason == "catalog must contain exactly 2 product records"
    assert captured.value.line_number is None


@pytest.mark.parametrize("expected_product_count", [0, -1, True, 1.5])
def test_scan_raw_catalog_requires_positive_integer_expected_count(
    tmp_path: Path,
    expected_product_count: object,
) -> None:
    path = _write_catalog(tmp_path, _row_bytes())

    with pytest.raises(ValueError, match="positive integer"):
        scan_raw_catalog(path, expected_product_count=expected_product_count)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "payload",
    [
        b'{"parent_asin":"p1","parent_asin":"p2","categories":["Root"],"details":{}}',
        b'{"parent_asin":"p1","categories":["Root"],"details":{"x":1,"x":2}}',
    ],
)
def test_scan_raw_catalog_rejects_duplicate_json_members(
    tmp_path: Path,
    payload: bytes,
) -> None:
    _assert_invalid(tmp_path, payload, reason="duplicate JSON object member")


def test_scan_raw_catalog_rejects_utf8_bom(tmp_path: Path) -> None:
    _assert_invalid(
        tmp_path,
        b"\xef\xbb\xbf" + _row_bytes(),
        reason="UTF-8 BOM is forbidden",
    )


@pytest.mark.parametrize("blank", [b"\n", b"  \t\r\n"])
def test_scan_raw_catalog_rejects_blank_physical_line(
    tmp_path: Path,
    blank: bytes,
) -> None:
    payload = _row_bytes() + b"\n" + blank
    _assert_invalid(
        tmp_path,
        payload,
        reason="blank physical line is forbidden",
        line_number=2,
    )


def test_scan_raw_catalog_rejects_invalid_utf8(tmp_path: Path) -> None:
    _assert_invalid(tmp_path, b"\xff", reason="catalog line is not valid UTF-8")


@pytest.mark.parametrize(
    "payload,reason",
    [
        (b"[]", "catalog record must be a JSON object"),
        (_row_bytes() + b" trailing", "invalid JSON object record"),
        (_row_bytes() + b" " + _row_bytes("p2"), "invalid JSON object record"),
    ],
)
def test_scan_raw_catalog_rejects_non_object_and_trailing_json(
    tmp_path: Path,
    payload: bytes,
    reason: str,
) -> None:
    _assert_invalid(tmp_path, payload, reason=reason)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"categories":["Root"],"details":{}}',
        b'{"parent_asin":null,"categories":["Root"],"details":{}}',
        b'{"parent_asin":"","categories":["Root"],"details":{}}',
        b'{"parent_asin":" p1","categories":["Root"],"details":{}}',
        b'{"parent_asin":"p1 ","categories":["Root"],"details":{}}',
    ],
)
def test_scan_raw_catalog_rejects_invalid_parent_asin(
    tmp_path: Path,
    payload: bytes,
) -> None:
    _assert_invalid(
        tmp_path,
        payload,
        reason="parent_asin must be a non-empty trimmed string",
    )


def test_scan_raw_catalog_rejects_duplicate_parent_asin(tmp_path: Path) -> None:
    payload = _row_bytes("p1") + b"\n" + _row_bytes("p1")
    _assert_invalid(
        tmp_path,
        payload,
        reason="parent_asin must be unique",
        line_number=2,
        expected_product_count=2,
    )


@pytest.mark.parametrize(
    "categories",
    [None, "Root", [], ["Root", ""], ["Root", 1]],
)
def test_scan_raw_catalog_rejects_invalid_categories(
    tmp_path: Path,
    categories: object,
) -> None:
    payload = _row_bytes(categories=categories)
    expected_reason = (
        "categories must be a non-empty array"
        if type(categories) is not list or not categories
        else "category path segments must be non-empty strings"
    )
    _assert_invalid(tmp_path, payload, reason=expected_reason)


@pytest.mark.parametrize("details", [None, [], "details", 1])
def test_scan_raw_catalog_rejects_missing_or_non_object_details(
    tmp_path: Path,
    details: object,
) -> None:
    if details is None:
        payload = b'{"parent_asin":"p1","categories":["Root"]}'
    else:
        payload = _row_bytes(details=details)
    _assert_invalid(tmp_path, payload, reason="details must be a JSON object")


@pytest.mark.parametrize(
    "number_token",
    [
        str(IJSON_SAFE_INTEGER_MAX + 1).encode("ascii"),
        str(IJSON_SAFE_INTEGER_MIN - 1).encode("ascii"),
    ],
)
def test_scan_raw_catalog_rejects_out_of_range_integer_anywhere_in_row(
    tmp_path: Path,
    number_token: bytes,
) -> None:
    payload = (
        b'{"parent_asin":"p1","categories":["Root"],"details":{"number":' + number_token + b"}}"
    )
    _assert_invalid(
        tmp_path,
        payload,
        reason="integer token is outside the I-JSON safe range",
    )


@pytest.mark.parametrize(
    "number_token,reason",
    [
        (b"NaN", "invalid JSON object record"),
        (b"Infinity", "invalid JSON object record"),
        (b"-Infinity", "invalid JSON object record"),
        (b"1e400", "non-finite number is forbidden"),
    ],
)
def test_scan_raw_catalog_rejects_non_finite_number_tokens(
    tmp_path: Path,
    number_token: bytes,
    reason: str,
) -> None:
    payload = (
        b'{"parent_asin":"p1","categories":["Root"],"details":{"number":' + number_token + b"}}"
    )
    _assert_invalid(tmp_path, payload, reason=reason)


def test_scan_raw_catalog_rejects_escaped_lone_unicode_surrogate(tmp_path: Path) -> None:
    payload = b'{"parent_asin":"p1","categories":["Root"],"details":{"value":"\\ud800"}}'
    _assert_invalid(tmp_path, payload, reason="lone Unicode surrogate is forbidden")
