"""Canonical JSON, JSONL, and Markdown renderers for raw profile DTOs."""

from __future__ import annotations

import unicodedata
from dataclasses import asdict
from io import TextIOBase
from typing import TextIO

from .models import CatalogProfile, ProductCategoryAssignment, TypeCount
from .profiler import canonical_json_dumps


def catalog_profile_to_json(profile: CatalogProfile) -> str:
    """Return one canonical JSON document without a trailing newline."""

    _require_profile(profile)
    return canonical_json_dumps(asdict(profile))


def product_category_assignment_to_json(assignment: ProductCategoryAssignment) -> str:
    """Return one canonical assignment record suitable for JSONL."""

    if type(assignment) is not ProductCategoryAssignment:
        raise TypeError("assignment must be a ProductCategoryAssignment")
    return canonical_json_dumps(asdict(assignment))


def write_catalog_profile_json(profile: CatalogProfile, stream: TextIO) -> None:
    """Write canonical profile JSON and exactly one trailing newline."""

    _require_text_stream(stream)
    stream.write(catalog_profile_to_json(profile))
    stream.write("\n")


def write_category_nodes_jsonl(profile: CatalogProfile, stream: TextIO) -> None:
    """Write category nodes in the deterministic DTO order."""

    _require_profile(profile)
    _require_text_stream(stream)
    for node in profile.category_nodes:
        stream.write(canonical_json_dumps(asdict(node)))
        stream.write("\n")


def write_detail_keys_jsonl(profile: CatalogProfile, stream: TextIO) -> None:
    """Write raw details-key profiles in exact key order."""

    _require_profile(profile)
    _require_text_stream(stream)
    for detail_key in profile.detail_keys:
        stream.write(canonical_json_dumps(asdict(detail_key)))
        stream.write("\n")


def write_category_detail_coverage_jsonl(profile: CatalogProfile, stream: TextIO) -> None:
    """Write observed category-subtree/key pairs in deterministic order."""

    _require_profile(profile)
    _require_text_stream(stream)
    for coverage in profile.category_detail_coverage:
        stream.write(canonical_json_dumps(asdict(coverage)))
        stream.write("\n")


class CanonicalAssignmentJsonlSink:
    """Callable sink for streaming assignments directly to canonical JSONL."""

    def __init__(self, stream: TextIO) -> None:
        _require_text_stream(stream)
        self._stream = stream

    def __call__(self, assignment: ProductCategoryAssignment) -> None:
        self._stream.write(product_category_assignment_to_json(assignment))
        self._stream.write("\n")


def catalog_profile_to_markdown(profile: CatalogProfile) -> str:
    """Render a deterministic human review report from one profile."""

    _require_profile(profile)
    lines = [
        "# Raw Catalog Profile",
        "",
        f"- Schema version: `{profile.schema_version}`",
        f"- Raw catalog SHA-256: `{profile.catalog_sha256}`",
        f"- File size: {profile.file_size_bytes} bytes",
        f"- Physical lines: {profile.physical_line_count}",
        f"- JSON object product rows: {profile.product_row_count}",
        f"- Non-object or invalid physical records: {profile.invalid_record_count}",
        (f"- Object rows with field diagnostics: {profile.product_row_with_diagnostics_count}"),
        f"- Unique valid parent ASINs: {profile.unique_parent_asin_count}",
        (
            "- Valid raw category assignments: "
            f"{profile.valid_category_assignment_count}/{profile.category_assignment_count}"
        ),
        f"- Category prefix nodes: {len(profile.category_nodes)}",
        f"- Raw details keys: {len(profile.detail_keys)}",
        f"- Stable sample seed: `{_escape_markdown(profile.seed)}`",
        "",
        "## Top-level schema",
        "",
        "| Field | Present | Missing | Null | Empty | JSON types |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for field in profile.top_level_fields:
        lines.append(
            "| "
            + " | ".join(
                (
                    _escape_markdown(field.field),
                    str(field.present_count),
                    str(field.missing_count),
                    str(field.null_count),
                    str(field.empty_count),
                    _escape_markdown(_render_types(field.type_counts)),
                )
            )
            + " |"
        )

    lines.extend(
        (
            "",
            "## Diagnostics",
            "",
            "| Code | Count | Sample lines |",
            "|---|---:|---|",
        )
    )
    if profile.diagnostics:
        for diagnostic in profile.diagnostics:
            sample_lines = ", ".join(str(sample.line_number) for sample in diagnostic.samples)
            lines.append(
                f"| {_escape_markdown(diagnostic.code)} | {diagnostic.count} | "
                f"{_escape_markdown(sample_lines)} |"
            )
    else:
        lines.append("| _none_ | 0 | |")

    lines.extend(
        (
            "",
            "## Category prefix tree",
            "",
            "| Raw path | Direct support | Subtree support | Category ID |",
            "|---|---:|---:|---|",
        )
    )
    ranked_nodes = sorted(
        profile.category_nodes,
        key=lambda node: (-node.subtree_support, node.path, node.category_id),
    )
    for node in ranked_nodes:
        path = " > ".join(node.path)
        lines.append(
            f"| {_escape_markdown(path)} | {node.direct_support} | {node.subtree_support} | "
            f"`{node.category_id}` |"
        )

    lines.extend(
        (
            "",
            "## Raw details keys",
            "",
            "| Key | Support | Nonempty | Null | Empty | Distinct | JSON types |",
            "|---|---:|---:|---:|---:|---:|---|",
        )
    )
    for detail_key in profile.detail_keys:
        lines.append(
            "| "
            + " | ".join(
                (
                    _escape_markdown(detail_key.raw_key),
                    str(detail_key.support_count),
                    str(detail_key.nonempty_count),
                    str(detail_key.null_count),
                    str(detail_key.empty_count),
                    str(detail_key.distinct_value_count),
                    _escape_markdown(_render_types(detail_key.type_counts)),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _render_types(type_counts: tuple[TypeCount, ...]) -> str:
    return ", ".join(f"{item.value_type}:{item.count}" for item in type_counts)


def _escape_markdown(value: str) -> str:
    escaped: list[str] = []
    for character in value:
        if character == "\\":
            escaped.append("\\\\")
        elif character == "|":
            escaped.append("\\|")
        elif character == "\n":
            escaped.append(" ")
        elif unicodedata.category(character) in ("Cc", "Zl", "Zp"):
            escaped.append(f"\\u{ord(character):04x}")
        else:
            escaped.append(character)
    return "".join(escaped)


def _require_profile(profile: CatalogProfile) -> None:
    if type(profile) is not CatalogProfile:
        raise TypeError("profile must be a CatalogProfile")


def _require_text_stream(stream: TextIO) -> None:
    if not isinstance(stream, TextIOBase) and not hasattr(stream, "write"):
        raise TypeError("stream must be a writable text stream")
