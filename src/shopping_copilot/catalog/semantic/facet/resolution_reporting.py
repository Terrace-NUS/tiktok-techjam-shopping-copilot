"""Human-readable CS3 evidence and resolved-statistics review report."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from ..canonical import content_id_for_value
from ..category import CategoryRegistry
from .gate_a_models import EvidenceStatus, NumericValue
from .resolution_models import FacetValueEvidence, ResolutionCandidateBuild


def resolution_candidate_markdown(
    build: ResolutionCandidateBuild,
    *,
    registry: CategoryRegistry,
) -> str:
    """Render CS3 results without implying a Gate-B runtime decision."""

    evidence_counts = Counter(item.status for item in build.evidence_store.evidence)
    root_row = next(
        row
        for row in build.stats.rows
        if row.facet_id == "price" and row.category_scope_id == registry.root_scope_id
    )
    scope_labels = {item.id: item.label for item in registry.scopes}
    examples = _stable_examples(build)
    lines = [
        "# CS3 Resolution Candidate: `price` v0",
        "",
        "- Gate A extraction: **APPROVED**",
        "- Gate B runtime use: **NOT APPROVED**",
        "- Raw catalog handling: **READ ONLY**",
        f"- Catalog: `{build.evidence_store.catalog_id}`",
        f"- FacetEvidenceStore: `{content_id_for_value(build.evidence_store)}`",
        f"- ProductFacetIndex: `{content_id_for_value(build.product_facet_index)}`",
        f"- CatalogFacetStatsArtifact: `{content_id_for_value(build.stats)}`",
        "",
        "## What this build did",
        "",
        "The builder read the frozen catalog and produced separate derived files. It did not",
        "rewrite catalog rows, fill missing prices, delete fields, or add products. Each present",
        "top-level `price` value now has an auditable status and stable evidence ID.",
        "",
        "## Price evidence",
        "",
        "| Status | Products | Meaning |",
        "| --- | ---: | --- |",
        f"| VALID | {evidence_counts[EvidenceStatus.VALID]:,} | usable structured price interval |",
        f"| EMPTY | {evidence_counts[EvidenceStatus.EMPTY]:,} | source explicitly contains `null` |",
        f"| INVALID | {evidence_counts[EvidenceStatus.INVALID]:,} | source present but rule rejects it |",
        f"| total evidence rows | {len(build.evidence_store.evidence):,} | one per present approved source |",
        "",
        "## Resolved catalog result",
        "",
        "For the full catalog, the sparse query index stores only prices that are actually known.",
        "Empty and invalid prices remain in the audit file and resolve to UNKNOWN.",
        "",
        "| Product-facet state | Products |",
        "| --- | ---: |",
        f"| KNOWN | {root_row.known_count:,} |",
        f"| UNKNOWN | {root_row.unknown_count:,} |",
        f"| CONFLICT | {root_row.conflict_count:,} |",
        f"| NOT_APPLICABLE | {root_row.not_applicable_count:,} |",
        f"| total | {root_row.scope_product_count:,} |",
        "",
        "## Category-conditioned coverage",
        "",
        "| Scope | Products | Known | Unknown | Conflict | Not applicable |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in build.stats.rows:
        label = _markdown_cell(scope_labels[row.category_scope_id])
        lines.append(
            f"| {label} | {row.scope_product_count:,} | {row.known_count:,} | "
            f"{row.unknown_count:,} | {row.conflict_count:,} | "
            f"{row.not_applicable_count:,} |"
        )
    lines.extend(
        [
            "",
            "## Stable evidence examples",
            "",
            "| Example | Product | Raw copied value | Result | Evidence ID |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for label, item in examples:
        result = item.status.value.upper()
        if item.canonical_value is not None:
            result += f" `{_value_summary(item.canonical_value)}`"
        lines.append(
            f"| {label} | `{item.parent_asin}` | `{_markdown_code(item.raw_value_json)}` | "
            f"{result} | `{item.id}` |"
        )
    lines.extend(
        [
            "",
            "## Retrieval safety boundary",
            "",
            f"A known-only budget filter would discard {root_row.unknown_count:,} products whose",
            "price is unknown. UNKNOWN does not mean over budget, so the safe matching primitive",
            "drops only a proven VIOLATED result and retains UNKNOWN for recall. This CS3 build",
            "does not enable hard filtering, soft ranking, session price commits, or clarification.",
            "Those choices require the next human Gate-B review.",
            "",
        ]
    )
    return "\n".join(lines)


def _stable_examples(
    build: ResolutionCandidateBuild,
) -> tuple[tuple[str, FacetValueEvidence], ...]:
    evidence = build.evidence_store.evidence
    predicates: tuple[tuple[str, Callable[[FacetValueEvidence], bool]], ...] = (
        (
            "exact",
            lambda item: (
                type(item.canonical_value) is NumericValue
                and item.canonical_value.lower == item.canonical_value.upper
                and item.canonical_value.lower != 0
            ),
        ),
        (
            "lower bound",
            lambda item: (
                type(item.canonical_value) is NumericValue and item.canonical_value.upper is None
            ),
        ),
        (
            "zero anomaly",
            lambda item: (
                type(item.canonical_value) is NumericValue
                and item.canonical_value.lower == 0
                and item.canonical_value.upper == 0
            ),
        ),
        ("empty", lambda item: item.status is EvidenceStatus.EMPTY),
        ("invalid", lambda item: item.status is EvidenceStatus.INVALID),
    )
    examples: list[tuple[str, FacetValueEvidence]] = []
    for label, predicate in predicates:
        item = next((candidate for candidate in evidence if predicate(candidate)), None)
        if item is not None:
            examples.append((label, item))
    return tuple(examples)


def _value_summary(value: object) -> str:
    if type(value) is NumericValue:
        upper = "+inf" if value.upper is None else str(value.upper)
        return f"[{value.lower},{upper}] {value.unit}"
    return type(value).__name__


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _markdown_code(value: str) -> str:
    return value.replace("`", "\\`").replace("|", "\\|")
