"""Generate the pinned full Unicode case-fold table used by CS1.

The input is the exact official Unicode 17.0.0 ``CaseFolding.txt`` file.  This
script deliberately performs no network access: download/review the source
separately, then let the fixed SHA-256 gate prove which bytes were consumed.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

UNICODE_VERSION = "17.0.0"
SOURCE_URL = "https://www.unicode.org/Public/17.0.0/ucd/CaseFolding.txt"
SOURCE_SHA256 = "ff8d8fefbf123574205085d6714c36149eb946d717a0c585c27f0f4ef58c4183"
EXPECTED_COMMON_ROWS = 1_481
EXPECTED_FULL_ROWS = 104


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="official CaseFolding-17.0.0.txt")
    parser.add_argument("output", type=Path, help="generated Python module")
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    observed_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if observed_sha256 != SOURCE_SHA256:
        parser.error(
            "CaseFolding source SHA-256 mismatch: "
            f"expected {SOURCE_SHA256}, observed {observed_sha256}"
        )

    mappings, common_rows, full_rows = _parse_casefold(source_bytes.decode("utf-8"))
    if common_rows != EXPECTED_COMMON_ROWS or full_rows != EXPECTED_FULL_ROWS:
        parser.error("CaseFolding C/F row counts differ from the pinned release")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_render_module(mappings), encoding="utf-8", newline="\n")
    return 0


def _parse_casefold(source: str) -> tuple[dict[int, str], int, int]:
    mappings: dict[int, str] = {}
    common_rows = 0
    full_rows = 0
    for line_number, raw_line in enumerate(source.splitlines(), start=1):
        data = raw_line.split("#", 1)[0].strip()
        if not data:
            continue
        fields = tuple(field.strip() for field in data.split(";"))
        if len(fields) != 4 or fields[3]:
            raise ValueError(f"malformed CaseFolding row at line {line_number}")
        source_hex, status, mapping_hex, _ = fields
        if status not in {"C", "F", "S", "T"}:
            raise ValueError(f"unknown CaseFolding status at line {line_number}")
        if status not in {"C", "F"}:
            continue
        common_rows += status == "C"
        full_rows += status == "F"
        source_codepoint = int(source_hex, 16)
        mapped = "".join(chr(int(item, 16)) for item in mapping_hex.split())
        if not mapped or mapped == chr(source_codepoint):
            raise ValueError(f"invalid CaseFolding mapping at line {line_number}")
        if source_codepoint in mappings:
            raise ValueError(f"duplicate full CaseFolding source at line {line_number}")
        mappings[source_codepoint] = mapped
    return mappings, common_rows, full_rows


def _render_module(mappings: dict[int, str]) -> str:
    lines = [
        '"""Generated Unicode 17.0.0 default full case-fold mapping.',
        "",
        "DO NOT EDIT. Regenerate with scripts/catalog_semantic/generate_casefold.py.",
        f"Source: {SOURCE_URL}",
        f"Source SHA-256: {SOURCE_SHA256}",
        "Unicode terms: https://www.unicode.org/terms_of_use.html",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from collections.abc import Mapping",
        "from types import MappingProxyType",
        "",
        f'CASEFOLD_UNICODE_VERSION = "{UNICODE_VERSION}"',
        f'CASEFOLD_SOURCE_SHA256 = "{SOURCE_SHA256}"',
        "",
        "FULL_CASEFOLD: Mapping[int, str] = MappingProxyType(",
        "    {",
    ]
    for codepoint, mapped in sorted(mappings.items()):
        lines.append(f"        0x{codepoint:04X}: {_python_string_literal(mapped)},")
    lines.extend(("    }", ")", ""))
    return "\n".join(lines)


def _python_string_literal(value: str) -> str:
    escaped: list[str] = []
    for character in value:
        codepoint = ord(character)
        if character == "\\":
            escaped.append("\\\\")
        elif character == '"':
            escaped.append('\\"')
        elif 0x20 <= codepoint <= 0x7E:
            escaped.append(character)
        elif codepoint <= 0xFFFF:
            escaped.append(f"\\u{codepoint:04x}")
        else:
            escaped.append(f"\\U{codepoint:08x}")
    return f'"{"".join(escaped)}"'


if __name__ == "__main__":
    raise SystemExit(main())
