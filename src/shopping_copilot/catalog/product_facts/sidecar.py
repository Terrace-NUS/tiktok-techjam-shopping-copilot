"""Strict loading for source-bound product-fact sidecars."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator, Mapping, Set
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

from .models import ProductFactCard
from .source import product_fact_request_from_raw_line
from .wire import decode_product_fact_card

PRODUCT_FACT_SIDECAR_SCHEMA = "shopping-copilot/product-fact-sidecar/v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class VerifiedProductFactCard:
    """One locally grounded card tied to the exact raw catalog row."""

    source_id: str
    extractor_model: str
    card: ProductFactCard


def load_product_fact_sidecar(
    sidecar_path: str | Path,
    *,
    catalog_path: str | Path,
    expected_parent_asins: Set[str] | None = None,
) -> Mapping[str, VerifiedProductFactCard]:
    """Load JSONL cards and revalidate every fact against immutable catalog bytes."""

    records = _load_records(Path(sidecar_path))
    expected = _validate_expected(expected_parent_asins)
    observed_ids = frozenset(records)
    if expected is not None and observed_ids != expected:
        missing = sorted(expected - observed_ids)
        unexpected = sorted(observed_ids - expected)
        raise ValueError(
            "product-fact sidecar IDs differ from expected set: "
            f"missing={missing[:5]!r}, unexpected={unexpected[:5]!r}"
        )

    requests = {}
    with Path(catalog_path).open("rb") as stream:
        for raw_line in stream:
            request = product_fact_request_from_raw_line(raw_line)
            if request.parent_asin in records:
                requests[request.parent_asin] = request
                if len(requests) == len(records):
                    break
    missing_catalog = sorted(observed_ids - requests.keys())
    if missing_catalog:
        raise ValueError(f"catalog is missing sidecar products: {missing_catalog[:5]!r}")

    verified: dict[str, VerifiedProductFactCard] = {}
    for parent_asin in sorted(records):
        record = records[parent_asin]
        request = requests[parent_asin]
        if record.get("source_id") != request.source_id:
            raise ValueError(f"product-fact source_id is stale: {parent_asin}")
        extractor = _object(record.get("extractor"), name="extractor")
        model = _text(extractor.get("model"), name="extractor.model")
        arguments = json.dumps(
            {
                "parent_asin": parent_asin,
                "facts": record.get("facts"),
                "summary": record.get("summary"),
            },
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        card = decode_product_fact_card(arguments, request)
        if card.warnings:
            raise ValueError(
                f"product-fact sidecar contains ungrounded facts: {parent_asin}: {card.warnings[0]}"
            )
        verified[parent_asin] = VerifiedProductFactCard(
            source_id=request.source_id,
            extractor_model=model,
            card=card,
        )
    return MappingProxyType(verified)


def _load_records(path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    try:
        for line_number, line in enumerate(_text_lines(path), start=1):
            if not line.strip():
                raise ValueError(f"blank product-fact sidecar row: {line_number}")
            decoded: object = json.loads(line)
            if type(decoded) is not dict:
                raise ValueError(f"product-fact row is not an object: {line_number}")
            record = cast(dict[str, object], decoded)
            if record.get("schema") != PRODUCT_FACT_SIDECAR_SCHEMA:
                raise ValueError(f"unknown product-fact schema: {line_number}")
            parent_asin = _text(record.get("parent_asin"), name="parent_asin")
            if parent_asin in records:
                raise ValueError(f"duplicate product-fact card: {parent_asin}")
            records[parent_asin] = record
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load product-fact sidecar: {path}") from error
    if not records:
        raise ValueError("product-fact sidecar must not be empty")
    return records


def _text_lines(path: Path) -> Iterator[str]:
    if path.suffix.casefold() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            yield from stream
        return
    with path.open("r", encoding="utf-8") as stream:
        yield from stream


def _validate_expected(values: Set[str] | None) -> frozenset[str] | None:
    if values is None:
        return None
    if not isinstance(values, Set):
        raise TypeError("expected_parent_asins must be a set")
    if any(type(value) is not str or not value.strip() for value in values):
        raise ValueError("expected_parent_asins contains an invalid product ID")
    return frozenset(values)


def _object(value: object, *, name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"product-fact {name} must be an object")
    return cast(dict[str, object], value)


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"product-fact {name} must be non-empty")
    return value
