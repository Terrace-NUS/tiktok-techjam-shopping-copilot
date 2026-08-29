"""Human-readable report for an owner-approved Gate-B capability candidate."""

from __future__ import annotations

from ..canonical import content_id_for_value
from ..category import CategoryRegistry
from .gate_b_models import GateBCandidateBuild


def gate_b_candidate_markdown(
    build: GateBCandidateBuild,
    *,
    registry: CategoryRegistry,
) -> str:
    """Render approval state and the remaining runtime-integration boundary."""

    labels = {item.id: item.label for item in registry.scopes}
    lines = [
        "# Gate B `price` Approved Candidate v0",
        "",
        "- Repository-owner approval: **RECORDED**",
        "- Effective capability artifact: **PUBLISHED IN THIS CANDIDATE**",
        "- Runtime integration: **NOT YET COMPLETE**",
        "- Proactive price clarification: **DISABLED**",
        f"- Gate-B selection: `{content_id_for_value(build.selection)}`",
        f"- EffectiveFacetCapabilitySet: `{content_id_for_value(build.capabilities)}`",
        f"- Reviewed proposal: `{build.gate_b_review_proposal_id}`",
        "",
        "## Approved exact-scope rows",
        "",
        "Every row is independent. No permission is inherited between parent, child, or union",
        "scopes. A runtime consumer must look up the exact active scope.",
        "",
        "| Scope | Decision | Commit explicit budget | Retrieval | Probe | Ask budget |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in build.capabilities.entries:
        lines.append(
            f"| {_cell(labels[entry.category_scope_id])} | {entry.decision.value} | "
            f"{_yes_no(entry.intent_committable)} | {_yes_no(entry.retrieval_eligible)} | "
            f"{_yes_no(entry.probe_eligible)} | {_yes_no(entry.clarification_eligible)} |"
        )
    lines.extend(
        [
            "",
            "## Approved value boundary",
            "",
            f"The future intent normalizer is `{build.selection.intent_value_normalizer_id}`.",
            "It accepts an already interpreted integer number of USD cents. Natural-language",
            "amount parsing remains Query Understanding's responsibility. Numeric price has no",
            "reviewed value aliases.",
            "",
            "## What this does not do",
            "",
            "This candidate does not install a session-context FacetRegistry, implement query",
            "grounding, change retrieval, produce Probe beliefs, ask a question, or map to the",
            "official adapter. Those CS5 components may now consume this approved contract, but",
            "they must be implemented and tested separately.",
            "",
        ]
    )
    return "\n".join(lines)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
