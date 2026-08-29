"""Verify the closed CS1 backend against pinned, explicitly listed Unicode 17 rows."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from pyunormalize import NFKC, UCD_VERSION  # type: ignore[import-untyped]

from shopping_copilot.catalog.semantic.category._casefold_v17 import (
    CASEFOLD_SOURCE_SHA256,
    CASEFOLD_UNICODE_VERSION,
    FULL_CASEFOLD,
)

UNICODE_VERSION = "17.0.0"
CASEFOLD_SHA256 = "ff8d8fefbf123574205085d6714c36149eb946d717a0c585c27f0f4ef58c4183"
NORMALIZATION_TEST_SHA256 = "5019ffd530751a741900c849c0e010332f142a3612234639bd200b82138a87db"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("casefold", type=Path, help="official CaseFolding-17.0.0.txt")
    parser.add_argument(
        "normalization_test",
        type=Path,
        help="official NormalizationTest-17.0.0.txt",
    )
    args = parser.parse_args()

    if UCD_VERSION != UNICODE_VERSION or CASEFOLD_UNICODE_VERSION != UNICODE_VERSION:
        parser.error("installed Unicode backend versions are not 17.0.0")
    if CASEFOLD_SOURCE_SHA256 != CASEFOLD_SHA256:
        parser.error("generated casefold module declares the wrong source hash")

    casefold_bytes = _read_pinned(args.casefold, expected_sha256=CASEFOLD_SHA256)
    expected_casefold = _parse_full_casefold(casefold_bytes.decode("utf-8"))
    if dict(FULL_CASEFOLD) != expected_casefold:
        parser.error("generated full casefold table differs from official C/F rows")

    normalization_bytes = _read_pinned(
        args.normalization_test,
        expected_sha256=NORMALIZATION_TEST_SHA256,
    )
    normalization_rows = _verify_nfkc(normalization_bytes.decode("utf-8"))
    print(
        "unicode-backend: verified "
        f"{len(expected_casefold)} full-casefold mappings and "
        f"{normalization_rows} listed NFKC test rows"
    )
    return 0


def _read_pinned(path: Path, *, expected_sha256: str) -> bytes:
    payload = path.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise ValueError(
            f"Unicode source hash mismatch for {path}: "
            f"expected {expected_sha256}, observed {observed}"
        )
    return payload


def _parse_full_casefold(source: str) -> dict[int, str]:
    mappings: dict[int, str] = {}
    for raw_line in source.splitlines():
        data = raw_line.split("#", 1)[0].strip()
        if not data:
            continue
        source_hex, status, mapping_hex, trailing = (field.strip() for field in data.split(";"))
        if trailing or status not in {"C", "F", "S", "T"}:
            raise ValueError("malformed official CaseFolding row")
        if status in {"C", "F"}:
            mappings[int(source_hex, 16)] = "".join(
                chr(int(item, 16)) for item in mapping_hex.split()
            )
    return mappings


def _verify_nfkc(source: str) -> int:
    row_count = 0
    for line_number, raw_line in enumerate(source.splitlines(), start=1):
        data = raw_line.split("#", 1)[0].strip()
        if not data or data.startswith("@"):
            continue
        fields = tuple(field.strip() for field in data.split(";"))
        if len(fields) != 6 or fields[5]:
            raise ValueError(f"malformed NormalizationTest row at line {line_number}")
        columns = tuple(_decode_codepoints(field) for field in fields[:5])
        expected_nfkc = columns[3]
        if any(NFKC(value) != expected_nfkc for value in columns):
            raise ValueError(f"NFKC listed-row failure at line {line_number}")
        row_count += 1
    return row_count


def _decode_codepoints(value: str) -> str:
    return "".join(chr(int(item, 16)) for item in value.split())


if __name__ == "__main__":
    raise SystemExit(main())
