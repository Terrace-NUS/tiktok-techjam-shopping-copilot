"""Closed, host-independent category normalizer bound to Unicode 17.0.0."""

from __future__ import annotations

from collections.abc import Sequence

from pyunormalize import NFKC, UCD_VERSION  # type: ignore[import-untyped]

from ..canonical import canonical_json_bytes, sha256_hex
from ..errors import CategoryBuildError
from ._casefold_v17 import (
    CASEFOLD_SOURCE_SHA256,
    CASEFOLD_UNICODE_VERSION,
    FULL_CASEFOLD,
)

CATEGORY_UNICODE_DATA_VERSION = "17.0.0"
CATEGORY_NORMALIZER_ID = "category_nfkc_trim_ws_full_casefold_ucd17_0"
CATEGORY_BUILDER_VERSION = "catalog_semantic_v0_ucd17_0"
_PINNED_CASEFOLD_SOURCE_SHA256 = "ff8d8fefbf123574205085d6714c36149eb946d717a0c585c27f0f4ef58c4183"
_PINNED_CASEFOLD_ENTRY_COUNT = 1_585
_PINNED_CASEFOLD_TABLE_SHA256 = "d9e787eda0915bb1a1d18cd13963a609be15f6a29ac2575ad2ce827bb031ce41"
_PINNED_WHITESPACE_CODEPOINTS_SHA256 = (
    "19a137ee374009d90b9b6bc812129aa3685b89e6825f53157e3183035140c1c9"
)

# This freezes Python's historical ``str.isspace`` domain rather than relying
# on the host interpreter. It is Unicode White_Space plus U+001C..U+001F.
_WHITESPACE_CODEPOINTS = frozenset(
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
_OBSERVED_CASEFOLD_TABLE_SHA256 = sha256_hex(
    canonical_json_bytes(
        [[codepoint, FULL_CASEFOLD[codepoint]] for codepoint in sorted(FULL_CASEFOLD)]
    )
)
_OBSERVED_WHITESPACE_CODEPOINTS_SHA256 = sha256_hex(
    canonical_json_bytes(sorted(_WHITESPACE_CODEPOINTS))
)


def ensure_category_builder_runtime() -> None:
    """Fail closed if either vendored Unicode component is incoherent."""

    if UCD_VERSION != CATEGORY_UNICODE_DATA_VERSION:
        raise CategoryBuildError(
            "category NFKC backend version differs from the pinned builder: "
            f"expected {CATEGORY_UNICODE_DATA_VERSION}, observed {UCD_VERSION}"
        )
    if (
        CASEFOLD_UNICODE_VERSION != CATEGORY_UNICODE_DATA_VERSION
        or CASEFOLD_SOURCE_SHA256 != _PINNED_CASEFOLD_SOURCE_SHA256
        or len(FULL_CASEFOLD) != _PINNED_CASEFOLD_ENTRY_COUNT
        or _OBSERVED_CASEFOLD_TABLE_SHA256 != _PINNED_CASEFOLD_TABLE_SHA256
    ):
        raise CategoryBuildError("category full-casefold table differs from the pinned builder")
    if (
        len(_WHITESPACE_CODEPOINTS) != 29
        or _OBSERVED_WHITESPACE_CODEPOINTS_SHA256 != _PINNED_WHITESPACE_CODEPOINTS_SHA256
    ):
        raise CategoryBuildError("category whitespace domain differs from the pinned builder")


def normalize_category_segment(raw_segment: str) -> str:
    """Apply NFKC, trim, whitespace collapse, then locale-free case-folding."""

    ensure_category_builder_runtime()
    if type(raw_segment) is not str:
        raise TypeError("raw category segment must be a string")
    normalized = NFKC(raw_segment)
    collapsed = _trim_and_collapse_whitespace(normalized)
    canonical = _full_casefold(collapsed)
    if not canonical:
        raise CategoryBuildError("category segment is empty after normalization")
    for character in canonical:
        codepoint = ord(character)
        if codepoint <= 0x1F or 0x7F <= codepoint <= 0x9F:
            raise CategoryBuildError(
                "category segment contains a control character after normalization"
            )
        if 0xD800 <= codepoint <= 0xDFFF:
            raise CategoryBuildError("category segment contains a lone surrogate")
    return canonical


def normalize_category_path(raw_path: Sequence[str]) -> tuple[str, ...]:
    """Normalize one non-empty raw category path without semantic merging."""

    copied = tuple(raw_path)
    if not copied:
        raise CategoryBuildError("category path must be non-empty")
    if any(type(segment) is not str for segment in copied):
        raise TypeError("category path segments must be strings")
    return tuple(normalize_category_segment(segment) for segment in copied)


def category_node_id(canonical_path: Sequence[str]) -> str:
    """Return the exact contract ID for one canonical category prefix."""

    copied = tuple(canonical_path)
    if not copied:
        raise CategoryBuildError("canonical category path must be non-empty")
    if normalize_category_path(copied) != copied:
        raise CategoryBuildError("category node ID requires an already-canonical path")
    payload = canonical_json_bytes({"canonical_path": list(copied)})
    return f"cn_{sha256_hex(payload)}"


def _trim_and_collapse_whitespace(value: str) -> str:
    output: list[str] = []
    pending_space = False
    for character in value:
        if ord(character) in _WHITESPACE_CODEPOINTS:
            if output:
                pending_space = True
            continue
        if pending_space:
            output.append(" ")
            pending_space = False
        output.append(character)
    return "".join(output)


def _full_casefold(value: str) -> str:
    return "".join(FULL_CASEFOLD.get(ord(character), character) for character in value)
