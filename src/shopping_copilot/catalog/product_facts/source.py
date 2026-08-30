"""Lossless projection of raw catalog rows into addressable model sources."""

from __future__ import annotations

import hashlib
import json
from typing import cast

from .models import ProductFactRequest, ProductSourceItem


def product_fact_request_from_raw_line(raw_line: bytes) -> ProductFactRequest:
    """Parse one raw row without truncating any field sent to the model."""

    if type(raw_line) is not bytes or not raw_line.strip():
        raise ValueError("raw catalog line must be non-empty bytes")
    try:
        decoded: object = json.loads(raw_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("raw catalog line must be valid UTF-8 JSON") from error
    if type(decoded) is not dict:
        raise ValueError("raw catalog line must contain an object")
    row = cast(dict[str, object], decoded)
    parent_asin = row.get("parent_asin")
    if type(parent_asin) is not str or not parent_asin.strip():
        raise ValueError("raw catalog row requires parent_asin")

    sources: list[ProductSourceItem] = []
    for field, value in row.items():
        if field == "parent_asin" or value is None:
            continue
        _append_value(sources, field=field, value=value)
    return ProductFactRequest(
        parent_asin=parent_asin,
        source_id=f"sha256:{hashlib.sha256(raw_line).hexdigest()}",
        sources=tuple(sources),
    )


def _append_value(items: list[ProductSourceItem], *, field: str, value: object) -> None:
    if type(value) is list:
        for index, entry in enumerate(cast(list[object], value)):
            _append_scalar(items, ref=f"{field}_{index}", field=field, value=entry)
        return
    if type(value) is dict:
        for index, (key, entry) in enumerate(cast(dict[str, object], value).items()):
            rendered = f"{key}: {_render(entry)}"
            items.append(ProductSourceItem(ref=f"{field}_{index}", field=field, text=rendered))
        return
    _append_scalar(items, ref=field, field=field, value=value)


def _append_scalar(
    items: list[ProductSourceItem],
    *,
    ref: str,
    field: str,
    value: object,
) -> None:
    rendered = _render(value)
    if rendered:
        items.append(ProductSourceItem(ref=ref, field=field, text=rendered))


def _render(value: object) -> str:
    if type(value) is str:
        return value
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
