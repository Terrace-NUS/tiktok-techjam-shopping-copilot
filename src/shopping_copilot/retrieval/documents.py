"""Deterministic, read-only construction of dense-retrieval product documents."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Set
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

DOCUMENT_SCHEMA_VERSION = "product_document_v1"
DOCUMENT_FIELD_ORDER = (
    "title",
    "categories",
    "store",
    "features",
    "details",
    "description",
)

# These fixed limits cover the observed catalog maxima for the compact fields and
# approximately the 99th percentile for the two long free-text fields. They bound
# pathological rows without making document bytes depend on an embedding model.
FIELD_CHARACTER_LIMITS: Mapping[str, int] = MappingProxyType(
    {
        "title": 384,
        "categories": 256,
        "store": 128,
        "features": 2_048,
        "details": 1_536,
        "description": 2_048,
    }
)


@dataclass(frozen=True, slots=True)
class ProductDocument:
    """One catalog product rendered for deterministic text embedding."""

    parent_asin: str
    text: str


class ProductDocumentError(ValueError):
    """Raised when catalog bytes cannot produce trustworthy product documents."""

    def __init__(self, message: str, *, line_number: int | None = None) -> None:
        self.line_number = line_number
        if line_number is not None:
            message = f"line {line_number}: {message}"
        super().__init__(message)


class _DuplicateJsonKeyError(ValueError):
    """Internal signal for JSON objects whose meaning is ambiguous."""


def load_product_documents(
    catalog_path: str | Path,
    *,
    expected_parent_asins: Set[str] | None = None,
) -> tuple[ProductDocument, ...]:
    """Read a JSONL catalog and return documents in physical source-row order.

    The source path is opened only in binary read mode. If ``expected_parent_asins``
    is supplied, its exact equality with the loaded catalog is checked before the
    result is returned.
    """

    expected = _validate_expected_parent_asins(expected_parent_asins)
    documents: list[ProductDocument] = []
    first_line_by_asin: dict[str, int] = {}

    with Path(catalog_path).open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            row = _parse_jsonl_row(raw_line, line_number=line_number)
            document = _build_product_document(row, line_number=line_number)

            previous_line = first_line_by_asin.get(document.parent_asin)
            if previous_line is not None:
                raise ProductDocumentError(
                    (
                        f"duplicate parent_asin {document.parent_asin!r}; "
                        f"first seen on line {previous_line}"
                    ),
                    line_number=line_number,
                )
            first_line_by_asin[document.parent_asin] = line_number
            documents.append(document)

    if expected is not None:
        actual = frozenset(first_line_by_asin)
        missing = expected - actual
        unexpected = actual - expected
        if missing or unexpected:
            raise ProductDocumentError(
                "catalog parent_asin set mismatch: "
                f"missing={len(missing)} {_stable_asin_sample(missing)}, "
                f"unexpected={len(unexpected)} {_stable_asin_sample(unexpected)}"
            )

    return tuple(documents)


def _parse_jsonl_row(raw_line: bytes, *, line_number: int) -> dict[str, object]:
    if not raw_line.strip():
        raise ProductDocumentError("blank JSONL row", line_number=line_number)

    try:
        text = raw_line.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ProductDocumentError("row is not valid UTF-8", line_number=line_number) from error

    try:
        parsed: object = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonstandard_number,
            parse_float=_parse_finite_float,
        )
    except _DuplicateJsonKeyError as error:
        raise ProductDocumentError(
            f"duplicate JSON key {error.args[0]!r}", line_number=line_number
        ) from error
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ProductDocumentError("invalid JSON", line_number=line_number) from error

    if type(parsed) is not dict:
        raise ProductDocumentError("JSONL row must be an object", line_number=line_number)
    return cast(dict[str, object], parsed)


def _build_product_document(row: dict[str, object], *, line_number: int) -> ProductDocument:
    parent_asin = _required_string(row, "parent_asin", line_number=line_number)
    if not parent_asin.strip():
        raise ProductDocumentError("parent_asin must not be empty", line_number=line_number)

    title = _required_string(row, "title", line_number=line_number)
    categories = _required_string_list(row, "categories", line_number=line_number)
    store = _required_nullable_string(row, "store", line_number=line_number)
    features = _required_string_list(row, "features", line_number=line_number)
    details = _required_object(row, "details", line_number=line_number)
    description = _required_string_list(row, "description", line_number=line_number)

    try:
        rendered_fields = {
            "title": _normalize_whitespace(title),
            "categories": _join_normalized(categories, separator=" > "),
            "store": "" if store is None else _normalize_whitespace(store),
            "features": _join_normalized(features, separator=" | "),
            "details": _render_details(details),
            "description": _join_normalized(description, separator=" | "),
        }
        lines = [
            f"{field}: {_truncate(rendered_fields[field], FIELD_CHARACTER_LIMITS[field])}"
            for field in DOCUMENT_FIELD_ORDER
        ]
        text = "\n".join(lines)
        text.encode("utf-8")
        parent_asin.encode("utf-8")
    except (RecursionError, UnicodeEncodeError) as error:
        raise ProductDocumentError(
            "document fields contain unsupported Unicode or nesting",
            line_number=line_number,
        ) from error

    return ProductDocument(parent_asin=parent_asin, text=text)


def _required_string(row: dict[str, object], field: str, *, line_number: int) -> str:
    value = _required_field(row, field, line_number=line_number)
    if type(value) is not str:
        raise ProductDocumentError(f"{field} must be a string", line_number=line_number)
    return value


def _required_nullable_string(
    row: dict[str, object], field: str, *, line_number: int
) -> str | None:
    value = _required_field(row, field, line_number=line_number)
    if value is not None and type(value) is not str:
        raise ProductDocumentError(f"{field} must be a string or null", line_number=line_number)
    return value


def _required_string_list(row: dict[str, object], field: str, *, line_number: int) -> list[str]:
    value = _required_field(row, field, line_number=line_number)
    if type(value) is not list:
        raise ProductDocumentError(f"{field} must be an array", line_number=line_number)
    values = cast(list[object], value)
    for index, item in enumerate(values):
        if type(item) is not str:
            raise ProductDocumentError(
                f"{field}[{index}] must be a string", line_number=line_number
            )
    return cast(list[str], values)


def _required_object(row: dict[str, object], field: str, *, line_number: int) -> dict[str, object]:
    value = _required_field(row, field, line_number=line_number)
    if type(value) is not dict:
        raise ProductDocumentError(f"{field} must be an object", line_number=line_number)
    return cast(dict[str, object], value)


def _required_field(row: dict[str, object], field: str, *, line_number: int) -> object:
    if field not in row:
        raise ProductDocumentError(f"missing required field {field!r}", line_number=line_number)
    return row[field]


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def _join_normalized(values: list[str], *, separator: str) -> str:
    normalized = (_normalize_whitespace(value) for value in values)
    return separator.join(value for value in normalized if value)


def _render_details(details: dict[str, object]) -> str:
    rendered: list[str] = []
    for key in sorted(details):
        normalized_key = _normalize_whitespace(key)
        value = _render_detail_value(details[key])
        if normalized_key and value:
            rendered.append(f"{normalized_key}: {value}")
    return " | ".join(rendered)


def _render_detail_value(value: object) -> str:
    if value is None:
        return ""
    if type(value) is str:
        return _normalize_whitespace(value)
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        return str(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite detail number")
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    if type(value) is list:
        rendered_items = (_render_detail_value(item) for item in cast(list[object], value))
        return ", ".join(item for item in rendered_items if item)
    if type(value) is dict:
        mapping_items: list[str] = []
        for key in sorted(cast(dict[str, object], value)):
            normalized_key = _normalize_whitespace(key)
            rendered_value = _render_detail_value(cast(dict[str, object], value)[key])
            if normalized_key and rendered_value:
                mapping_items.append(f"{normalized_key}: {rendered_value}")
        return "{" + "; ".join(mapping_items) + "}" if mapping_items else ""
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit].rstrip()


def _validate_expected_parent_asins(
    expected_parent_asins: Set[str] | None,
) -> frozenset[str] | None:
    if expected_parent_asins is None:
        return None
    if not isinstance(expected_parent_asins, Set):
        raise TypeError("expected_parent_asins must be a set")
    for parent_asin in expected_parent_asins:
        if type(parent_asin) is not str:
            raise TypeError("expected_parent_asins must contain only strings")
        if not parent_asin.strip():
            raise ValueError("expected_parent_asins must not contain empty values")
    return frozenset(expected_parent_asins)


def _stable_asin_sample(values: Set[str]) -> str:
    sample = sorted(values)[:5]
    suffix = ", ..." if len(values) > len(sample) else ""
    return "[" + ", ".join(repr(value) for value in sample) + suffix + "]"


def _reject_nonstandard_number(value: str) -> object:
    raise ValueError(f"non-standard JSON number: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number: {value}")
    return parsed


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result
