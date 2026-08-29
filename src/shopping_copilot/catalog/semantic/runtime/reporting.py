"""Human-readable CS5 runtime projection and grounding report."""

from __future__ import annotations

from ..canonical import content_id_for_value
from .models import RuntimeProjectionCandidateBuild


def runtime_projection_candidate_markdown(build: RuntimeProjectionCandidateBuild) -> str:
    """Render the implemented runtime foundation and its deliberate non-goals."""

    lines = [
        "# CS5 Runtime Projection and Grounding Candidate v0",
        "",
        "- Gate B owner approval: **CONSUMED**",
        "- Session FacetRegistry projection: **IMPLEMENTED**",
        "- Exact-scope capability lookup: **IMPLEMENTED**",
        "- Deterministic grounding: **IMPLEMENTED**",
        "- Retrieval integration: **NOT IMPLEMENTED**",
        "- Session gateway integration: **NOT IMPLEMENTED**",
        f"- RuntimeFacetRegistryArtifact: `{content_id_for_value(build.runtime_registry)}`",
        f"- RuntimeValueLexicon: `{content_id_for_value(build.runtime_lexicon)}`",
        "",
        "## Projected session facets",
        "",
        "| Facet | Kind | Operators | Intent normalizer |",
        "| --- | --- | --- | --- |",
    ]
    for record in build.runtime_registry.entries:
        lines.append(
            f"| `{record.facet_id}` | {record.kind} | "
            f"{', '.join(record.operator_values)} | `{record.intent_value_normalizer_id}` |"
        )
    lines.extend(
        [
            "",
            "## Value boundary",
            "",
            "`usd_cent_int_v1` accepts only non-boolean integers in the signed I-JSON safe",
            "range and returns them unchanged. Strings, floats, booleans, and out-of-range",
            "integers fail. Currency and language parsing happen before this boundary.",
            "",
            "`category_scope_id_v1` is bound to this CategoryRegistry and accepts only one of",
            "its published exact CategoryScope IDs.",
            "",
            "## Capability boundary",
            "",
            "Capability lookup uses only `(facet_id, active_category_scope_id)` equality.",
            "A missing row grants no permission. It never walks to a parent, child, or union",
            "scope. The approved price rows allow intent, retrieval, and Probe while proactive",
            "clarification remains disabled.",
            "",
            "## Grounding boundary",
            "",
            "CS5B accepts an already structured Query Understanding candidate, checks the",
            "exact facet, operator, value, category scope, and Gate-B permission, then returns",
            "GROUNDED, SEMANTIC_ONLY, or AMBIGUOUS. Numeric equality becomes inclusive GE then",
            "LE predicates. The service does not parse a user message, allocate a Preference",
            "ID, call the reducer, rank products, create SearchBelief, or ask a question.",
            "",
        ]
    )
    return "\n".join(lines)
