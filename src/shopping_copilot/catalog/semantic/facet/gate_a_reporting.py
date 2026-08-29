"""Human-readable report for normative Gate-A candidates."""

from __future__ import annotations

from ..canonical import content_id_for_value
from .gate_a_models import GateACandidateBuild


def gate_a_candidate_markdown(build: GateACandidateBuild) -> str:
    """Render the approved extraction boundary without implying Gate-B promotion."""

    audit = build.price_audits[0]
    binding = build.bindings.bindings[0]
    lines = [
        "# Gate A Candidate: `price` v0",
        "",
        "- Owner decision: **EXTRACTION_APPROVED**",
        "- Runtime / Gate B decision: **NOT APPROVED**",
        f"- Catalog: `{build.catalog_id}`",
        f"- CategoryRegistry: `{build.category_registry_id}`",
        f"- CatalogFacetSchema: `{content_id_for_value(build.facet_schema)}`",
        f"- FacetApplicabilitySet: `{content_id_for_value(build.applicability)}`",
        f"- FacetSourceBindingSet: `{content_id_for_value(build.bindings)}`",
        "",
        "## Approved rule",
        "",
        "`price` is a single numeric catalog facet. It is meaningful for every catalog",
        "category, and its only approved source is the exact top-level `price` field.",
        "",
        f"- Extractor: `{binding.extractor_id}`",
        f"- Catalog normalizer: `{binding.catalog_value_normalizer_id}`",
        f"- Resolver: `{binding.resolver_id}`",
        f"- Priority: `{binding.priority}`",
        "- Unit: integer `USD_CENT` intervals",
        "",
        "| Raw value | Result |",
        "| --- | --- |",
        "| `null` | EMPTY; no product price fact |",
        "| non-negative exact-cent JSON number | inclusive exact interval |",
        "| exact `from <decimal>` string | inclusive lower-bound interval |",
        "| em dash or any other string/type | INVALID audit evidence |",
        "| negative, signed zero, over-precision, non-finite, or unsafe cents | INVALID |",
        "",
        "Binary-float round trips and rounding are forbidden. Numeric JSON tokens are",
        "parsed through `Decimal`; the normalizer rejects Python `float` input.",
        "",
        "## Frozen-catalog verification",
        "",
        "| Outcome | Products |",
        "| --- | ---: |",
        f"| source present | {audit.source_present_count:,} |",
        f"| source missing | {audit.source_missing_count:,} |",
        f"| VALID | {audit.valid_count:,} |",
        f"| EMPTY | {audit.empty_count:,} |",
        f"| INVALID | {audit.invalid_count:,} |",
        f"| exact intervals | {audit.exact_interval_count:,} |",
        f"| lower-bound intervals | {audit.lower_bound_interval_count:,} |",
        f"| exact zero intervals | {audit.zero_exact_count:,} |",
        "",
        "The exact zero value remains valid structured evidence and is retained as an",
        "anomaly for later runtime review. Missing or invalid prices never become negative",
        "facts.",
        "",
        "## Boundary",
        "",
        "This candidate approves deterministic extraction only. It does not approve hard",
        "budget filtering, session commits, clarification questions, Probe behavior, or",
        "official `ask_attribute` mapping. Those decisions require resolved CS3 statistics",
        "and a separate Gate-B review.",
        "",
        "All other profiled source locators remain `NEEDS_REVIEW`.",
        "",
    ]
    return "\n".join(lines)
