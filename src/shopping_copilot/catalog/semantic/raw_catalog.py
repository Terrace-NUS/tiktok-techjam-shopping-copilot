"""Strict, read-only scanner for the frozen raw catalog release input."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from .canonical import IJSON_SAFE_INTEGER_MAX, IJSON_SAFE_INTEGER_MIN
from .errors import CatalogChangedError, RawCatalogValidationError

RAW_CATALOG_SCHEMA: Literal["shopping-copilot/raw-catalog-jsonl/v1"] = (
    "shopping-copilot/raw-catalog-jsonl/v1"
)
OFFICIAL_PRODUCT_COUNT = 50_000
_UTF8_BOM = b"\xef\xbb\xbf"
_CONTENT_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True, kw_only=True)
class RawCatalogCategoryRecord:
    """The category fields retained from one strictly valid raw product row."""

    parent_asin: str
    raw_path: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.parent_asin) is not str
            or not self.parent_asin
            or self.parent_asin != self.parent_asin.strip()
        ):
            raise ValueError("raw parent_asin must be a non-empty trimmed string")
        if type(self.raw_path) is not tuple or not self.raw_path:
            raise ValueError("raw category path must be a non-empty tuple")
        for segment in self.raw_path:
            if type(segment) is not str or not segment:
                raise ValueError("raw category path segments must be non-empty strings")
            if any(0xD800 <= ord(character) <= 0xDFFF for character in segment):
                raise ValueError("raw category path segment contains a lone surrogate")


@dataclass(frozen=True, slots=True, kw_only=True)
class RawCatalogScan:
    """Exact source identity and category observations from a strict scan."""

    schema: Literal["shopping-copilot/raw-catalog-jsonl/v1"]
    catalog_id: str
    byte_size: int
    product_count: int
    records: tuple[RawCatalogCategoryRecord, ...]

    def __post_init__(self) -> None:
        if self.schema != RAW_CATALOG_SCHEMA:
            raise ValueError("RawCatalogScan.schema is invalid")
        if (
            type(self.catalog_id) is not str
            or _CONTENT_ID_PATTERN.fullmatch(self.catalog_id) is None
        ):
            raise ValueError("RawCatalogScan.catalog_id is invalid")
        if type(self.byte_size) is not int or self.byte_size <= 0:
            raise ValueError("RawCatalogScan.byte_size must be positive")
        if type(self.product_count) is not int or self.product_count <= 0:
            raise ValueError("RawCatalogScan.product_count must be positive")
        if type(self.records) is not tuple:
            raise TypeError("RawCatalogScan.records must be a tuple")
        if self.product_count != len(self.records):
            raise ValueError("RawCatalogScan.product_count differs from records")
        parent_asins = tuple(record.parent_asin for record in self.records)
        if len(set(parent_asins)) != len(parent_asins):
            raise ValueError("RawCatalogScan parent_asins must be unique")


class _DuplicateJsonKeyError(ValueError):
    pass


def scan_raw_catalog(
    path: str | Path,
    *,
    expected_product_count: int = OFFICIAL_PRODUCT_COUNT,
) -> RawCatalogScan:
    """Hash and strictly validate exact raw JSONL bytes in two read passes."""

    if type(expected_product_count) is not int or expected_product_count <= 0:
        raise ValueError("expected_product_count must be a positive integer")

    source = Path(path)
    initial_digest, initial_size = _hash_file(source)
    records: list[RawCatalogCategoryRecord] = []
    seen_parent_asins: set[str] = set()
    verification_digest = hashlib.sha256()
    verification_size = 0

    with source.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            verification_digest.update(raw_line)
            verification_size += len(raw_line)
            if line_number == 1 and raw_line.startswith(_UTF8_BOM):
                raise RawCatalogValidationError("UTF-8 BOM is forbidden", line_number=1)
            record = _parse_line(raw_line, line_number=line_number)
            if record.parent_asin in seen_parent_asins:
                raise RawCatalogValidationError(
                    "parent_asin must be unique", line_number=line_number
                )
            seen_parent_asins.add(record.parent_asin)
            records.append(record)

    if verification_digest.hexdigest() != initial_digest or verification_size != initial_size:
        raise CatalogChangedError("catalog changed between identity and parse passes")
    if len(records) != expected_product_count:
        raise RawCatalogValidationError(
            f"catalog must contain exactly {expected_product_count} product records"
        )

    return RawCatalogScan(
        schema=RAW_CATALOG_SCHEMA,
        catalog_id=f"sha256:{initial_digest}",
        byte_size=initial_size,
        product_count=len(records),
        records=tuple(records),
    )


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            byte_size += len(chunk)
    return digest.hexdigest(), byte_size


def _parse_line(raw_line: bytes, *, line_number: int) -> RawCatalogCategoryRecord:
    if not raw_line.strip():
        raise RawCatalogValidationError("blank physical line is forbidden", line_number=line_number)
    try:
        text = raw_line.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RawCatalogValidationError(
            "catalog line is not valid UTF-8", line_number=line_number
        ) from error
    try:
        parsed: object = json.loads(
            text,
            parse_constant=_reject_nonfinite_token,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except _DuplicateJsonKeyError as error:
        raise RawCatalogValidationError(
            "duplicate JSON object member", line_number=line_number
        ) from error
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise RawCatalogValidationError(
            "invalid JSON object record", line_number=line_number
        ) from error

    if type(parsed) is not dict:
        raise RawCatalogValidationError(
            "catalog record must be a JSON object", line_number=line_number
        )
    _validate_raw_json_value(parsed, line_number=line_number)
    row = cast(dict[str, object], parsed)

    parent_asin = row.get("parent_asin")
    if type(parent_asin) is not str or not parent_asin or parent_asin != parent_asin.strip():
        raise RawCatalogValidationError(
            "parent_asin must be a non-empty trimmed string", line_number=line_number
        )

    categories = row.get("categories")
    if type(categories) is not list or not categories:
        raise RawCatalogValidationError(
            "categories must be a non-empty array", line_number=line_number
        )
    raw_path: list[str] = []
    for segment in categories:
        if type(segment) is not str or not segment:
            raise RawCatalogValidationError(
                "category path segments must be non-empty strings",
                line_number=line_number,
            )
        raw_path.append(segment)

    if type(row.get("details")) is not dict:
        raise RawCatalogValidationError("details must be a JSON object", line_number=line_number)

    return RawCatalogCategoryRecord(
        parent_asin=parent_asin,
        raw_path=tuple(raw_path),
    )


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_nonfinite_token(raw: str) -> object:
    raise ValueError(f"non-finite number token: {raw}")


def _validate_raw_json_value(value: object, *, line_number: int) -> None:
    if value is None or type(value) in (bool, str):
        if type(value) is str:
            for character in value:
                if 0xD800 <= ord(character) <= 0xDFFF:
                    raise RawCatalogValidationError(
                        "lone Unicode surrogate is forbidden", line_number=line_number
                    )
        return
    if type(value) is int:
        if not IJSON_SAFE_INTEGER_MIN <= value <= IJSON_SAFE_INTEGER_MAX:
            raise RawCatalogValidationError(
                "integer token is outside the I-JSON safe range",
                line_number=line_number,
            )
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise RawCatalogValidationError(
                "non-finite number is forbidden", line_number=line_number
            )
        return
    if type(value) is list:
        for item in cast(list[object], value):
            _validate_raw_json_value(item, line_number=line_number)
        return
    if type(value) is dict:
        for key, item in cast(dict[str, object], value).items():
            _validate_raw_json_value(key, line_number=line_number)
            _validate_raw_json_value(item, line_number=line_number)
        return
    raise RawCatalogValidationError("unsupported parsed JSON value", line_number=line_number)
