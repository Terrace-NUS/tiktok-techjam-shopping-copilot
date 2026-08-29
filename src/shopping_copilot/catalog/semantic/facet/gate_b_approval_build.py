"""Deterministic projection of an owner-approved Gate-B selection."""

from __future__ import annotations

from ..canonical import content_id_for_value
from ..category import CategoryRegistry
from ..errors import GateBBuildError, GateBSelectionError
from .gate_a_models import GateACandidateBuild
from .gate_b_build import GateBPriceReviewBuild
from .gate_b_models import (
    EFFECTIVE_FACET_CAPABILITIES_SCHEMA,
    GATE_B_BUILDER_VERSION,
    GATE_B_CANDIDATE_SCHEMA,
    EffectiveFacetCapability,
    EffectiveFacetCapabilitySet,
    GateBCandidateBuild,
    GateBSelection,
    RuntimePromotionDecision,
)
from .resolution_models import RESOLUTION_POLICY_ID, ResolutionCandidateBuild


def build_gate_b_candidate(
    selection: GateBSelection,
    *,
    registry: CategoryRegistry,
    gate_a: GateACandidateBuild,
    resolution: ResolutionCandidateBuild,
    review: GateBPriceReviewBuild,
) -> GateBCandidateBuild:
    """Validate the human decision and publish only its exact reviewed projection."""

    _validate_selection_pins(
        selection,
        registry=registry,
        gate_a=gate_a,
        resolution=resolution,
        review=review,
    )
    _validate_exact_capability_coverage(
        selection.approvals,
        registry=registry,
        gate_a=gate_a,
    )
    capabilities = EffectiveFacetCapabilitySet(
        schema=EFFECTIVE_FACET_CAPABILITIES_SCHEMA,
        category_registry_id=selection.category_registry_id,
        facet_schema_id=selection.facet_schema_id,
        facet_applicability_id=selection.facet_applicability_id,
        product_facet_index_id=selection.product_facet_index_id,
        resolution_policy_id=RESOLUTION_POLICY_ID,
        entries=selection.approvals,
    )
    return GateBCandidateBuild(
        schema=GATE_B_CANDIDATE_SCHEMA,
        builder_version=GATE_B_BUILDER_VERSION,
        catalog_id=selection.catalog_id,
        catalog_facet_stats_id=selection.catalog_facet_stats_id,
        gate_b_review_proposal_id=selection.gate_b_review_proposal_id,
        public_target_audit_id=selection.public_target_audit_id,
        selection=selection,
        capabilities=capabilities,
    )


def _validate_selection_pins(
    selection: GateBSelection,
    *,
    registry: CategoryRegistry,
    gate_a: GateACandidateBuild,
    resolution: ResolutionCandidateBuild,
    review: GateBPriceReviewBuild,
) -> None:
    proposal = review.proposal
    audit = review.public_target_audit
    expected = {
        "catalog_id": resolution.evidence_store.catalog_id,
        "category_registry_id": resolution.category_registry_id,
        "facet_schema_id": resolution.facet_schema_id,
        "facet_applicability_id": resolution.evidence_store.facet_applicability_id,
        "product_facet_index_id": content_id_for_value(resolution.product_facet_index),
        "catalog_facet_stats_id": content_id_for_value(resolution.stats),
        "gate_b_review_proposal_id": content_id_for_value(proposal),
        "public_target_audit_id": content_id_for_value(audit),
    }
    for name, value in expected.items():
        if getattr(selection, name) != value:
            raise GateBSelectionError(f"Gate-B selection has a stale pin: {name}")
    if registry.catalog_id != selection.catalog_id:
        raise GateBSelectionError("Gate-B selection catalog differs from CategoryRegistry")
    if gate_a.category_registry_id != selection.category_registry_id:
        raise GateBSelectionError("Gate-B selection CategoryRegistry differs from Gate A")
    if selection.approvals != proposal.proposed_capabilities:
        raise GateBSelectionError("Gate-B approvals differ from the owner-reviewed proposal")
    if selection.intent_value_normalizer_id != proposal.proposed_intent_normalizer_id:
        raise GateBSelectionError("Gate-B approved intent normalizer differs from proposal")
    if selection.reviewed_value_aliases != proposal.reviewed_value_aliases:
        raise GateBSelectionError("Gate-B approved aliases differ from proposal")


def _validate_exact_capability_coverage(
    entries: tuple[EffectiveFacetCapability, ...],
    *,
    registry: CategoryRegistry,
    gate_a: GateACandidateBuild,
) -> None:
    facet_ids = tuple(item.id for item in gate_a.facet_schema.facets)
    scope_ids = tuple(item.id for item in registry.scopes)
    expected_keys = tuple(
        sorted((facet_id, scope_id) for facet_id in facet_ids for scope_id in scope_ids)
    )
    observed_keys = tuple((item.facet_id, item.category_scope_id) for item in entries)
    if observed_keys != expected_keys:
        raise GateBBuildError("Gate-B capabilities do not cover every exact facet/scope pair")

    scopes = {item.id: item for item in registry.scopes}
    applicability = {item.facet_id: item for item in gate_a.applicability.entries}
    for entry in entries:
        applicable = applicability.get(entry.facet_id)
        if applicable is None:
            raise GateBBuildError("Gate-B capability references a facet without applicability")
        applicable_nodes: set[str] = set()
        for scope_id in applicable.category_scope_ids:
            try:
                applicable_nodes.update(scopes[scope_id].member_node_ids)
            except KeyError as error:
                raise GateBBuildError("Gate-A applicability references an unknown scope") from error
        scope_nodes = set(scopes[entry.category_scope_id].member_node_ids)
        if scope_nodes.isdisjoint(applicable_nodes) and (
            entry.decision
            not in (RuntimePromotionDecision.SEMANTIC_ONLY, RuntimePromotionDecision.REJECT)
            or entry.intent_committable
            or entry.retrieval_eligible
            or entry.probe_eligible
            or entry.clarification_eligible
        ):
            raise GateBBuildError("inapplicable exact scope cannot receive runtime permissions")
