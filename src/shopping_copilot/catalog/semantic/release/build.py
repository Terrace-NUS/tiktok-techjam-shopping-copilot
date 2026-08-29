"""Deterministic CS6 assembly and deep cross-artifact validation."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from ..canonical import canonical_json_text, content_id_for_value
from ..category import (
    CategoryRegistry,
    CategoryScope,
    ProductCategoryAssignment,
    ProductCategoryAssignmentSet,
    ProductCategoryAssignmentStatus,
    build_category_graph_proposal,
    category_node_id,
    normalize_category_path,
    validate_product_category_assignment_set,
)
from ..category.normalization import ensure_category_builder_runtime
from ..errors import ReleaseBuildError
from ..facet import (
    BooleanValue,
    CatalogFacetSchema,
    CatalogFacetStatsArtifact,
    CategoricalValue,
    EffectiveFacetCapabilitySet,
    EvidenceStatus,
    FacetApplicability,
    FacetApplicabilitySet,
    FacetDataType,
    FacetEvidenceStore,
    FacetSourceBinding,
    FacetSourceBindingSet,
    FacetValueEvidence,
    NumericValue,
    ProductFacetIndex,
    ProductFacetStatus,
    ResolvedProductFacetValue,
    ResolvedValueCount,
    TextValue,
    canonical_raw_value_json,
    evidence_id_for,
    require_catalog_value_normalizer,
    require_extractor,
    require_resolver,
)
from ..facet.gate_a_models import ResolvedFacetValue
from ..facet.resolution_models import (
    CATALOG_FACET_STATS_SCHEMA,
    RESOLUTION_POLICY_ID,
    FacetScopeCatalogStats,
)
from ..raw_catalog import RawCatalogScan
from ..runtime import (
    SYSTEM_PRODUCT_CATEGORY_FACET_ID,
    NumericRuntimeDomain,
    RuntimeFacetRegistryArtifact,
    RuntimeValueGrounder,
    RuntimeValueLexicon,
    require_intent_value_normalizer,
)
from .models import (
    ARTIFACT_KINDS,
    CATALOG_SEMANTIC_RELEASE_BUILDER_VERSION,
    CATALOG_SEMANTIC_RELEASE_SCHEMA,
    REVIEWED_SEMANTIC_CONFIG_SCHEMA,
    ArtifactKind,
    ArtifactRef,
    CatalogSemanticReleaseManifest,
    ReviewedRuntimeFacetConfig,
    ReviewedSemanticConfig,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class DecodedReleaseArtifacts:
    """The 11 decoded generated projections preceding reviewed config."""

    category_registry: CategoryRegistry
    product_category_assignments: ProductCategoryAssignmentSet
    facet_schema: CatalogFacetSchema
    facet_applicability: FacetApplicabilitySet
    facet_source_bindings: FacetSourceBindingSet
    facet_evidence_store: FacetEvidenceStore
    product_facet_index: ProductFacetIndex
    facet_stats: CatalogFacetStatsArtifact
    effective_capabilities: EffectiveFacetCapabilitySet
    runtime_value_lexicon: RuntimeValueLexicon
    runtime_registry: RuntimeFacetRegistryArtifact


def build_reviewed_semantic_config(
    artifacts: DecodedReleaseArtifacts,
) -> ReviewedSemanticConfig:
    """Project reviewed artifact content into the exact contract config artifact."""

    if type(artifacts) is not DecodedReleaseArtifacts:
        raise TypeError("reviewed config build requires DecodedReleaseArtifacts")
    ordinary_records = {
        item.facet_id: item
        for item in artifacts.runtime_registry.entries
        if item.facet_id != SYSTEM_PRODUCT_CATEGORY_FACET_ID
    }
    domains: dict[str, NumericRuntimeDomain] = {
        item.facet_id: item for item in artifacts.runtime_value_lexicon.domains
    }
    if set(ordinary_records) != set(domains):
        raise ReleaseBuildError("runtime registry and lexicon ordinary facets differ")
    runtime_facets: list[ReviewedRuntimeFacetConfig] = []
    for facet_id in sorted(ordinary_records):
        record = ordinary_records[facet_id]
        domain = domains[facet_id]
        if domain.intent_value_normalizer_id != record.intent_value_normalizer_id:
            raise ReleaseBuildError("runtime normalizer IDs differ")
        if type(domain) is not NumericRuntimeDomain:
            raise ReleaseBuildError("release v0 supports only the projected numeric price domain")
        runtime_facets.append(
            ReviewedRuntimeFacetConfig(
                facet_id=facet_id,
                intent_value_normalizer_id=record.intent_value_normalizer_id,
                aliases=(),
            )
        )
    registry = artifacts.category_registry
    return ReviewedSemanticConfig(
        schema=REVIEWED_SEMANTIC_CONFIG_SCHEMA,
        catalog_id=registry.catalog_id,
        category_graph_id=registry.category_graph_id,
        resolution_policy_id=RESOLUTION_POLICY_ID,
        builder_version=CATALOG_SEMANTIC_RELEASE_BUILDER_VERSION,
        category_scopes=registry.scopes,
        facets=artifacts.facet_schema.facets,
        facet_applicability=artifacts.facet_applicability.entries,
        source_bindings=artifacts.facet_source_bindings.bindings,
        capabilities=artifacts.effective_capabilities.entries,
        runtime_facets=tuple(runtime_facets),
    )


def build_release_manifest(refs: tuple[ArtifactRef, ...]) -> CatalogSemanticReleaseManifest:
    """Build the canonical 13-member release manifest from exact byte refs."""

    if type(refs) is not tuple or tuple(item.kind for item in refs) != ARTIFACT_KINDS:
        raise ReleaseBuildError("release refs must contain exactly 13 sorted artifact kinds")
    by_kind = {item.kind: item for item in refs}
    return CatalogSemanticReleaseManifest(
        schema=CATALOG_SEMANTIC_RELEASE_SCHEMA,
        catalog_id=by_kind["catalog"].content_id,
        category_registry_id=by_kind["category_registry"].content_id,
        product_category_assignment_id=by_kind["product_category_assignment"].content_id,
        facet_schema_id=by_kind["facet_schema"].content_id,
        facet_applicability_id=by_kind["facet_applicability"].content_id,
        facet_source_bindings_id=by_kind["facet_source_bindings"].content_id,
        facet_evidence_store_id=by_kind["facet_evidence_store"].content_id,
        product_facet_index_id=by_kind["product_facet_index"].content_id,
        facet_stats_id=by_kind["facet_stats"].content_id,
        effective_capabilities_id=by_kind["effective_capabilities"].content_id,
        runtime_value_lexicon_id=by_kind["runtime_value_lexicon"].content_id,
        runtime_registry_id=by_kind["runtime_registry"].content_id,
        reviewed_config_id=by_kind["reviewed_config"].content_id,
        resolution_policy_id=RESOLUTION_POLICY_ID,
        builder_version=CATALOG_SEMANTIC_RELEASE_BUILDER_VERSION,
        artifacts=refs,
    )


def validate_decoded_release(
    *,
    scan: RawCatalogScan,
    refs: dict[ArtifactKind, ArtifactRef],
    artifacts: DecodedReleaseArtifacts,
    reviewed_config: ReviewedSemanticConfig,
) -> RuntimeValueGrounder:
    """Verify byte pins, semantic cross-references, closed code, and runtime projection."""

    _require_release_builder(reviewed_config.builder_version)
    if scan.catalog_id != refs["catalog"].content_id or scan.byte_size != refs["catalog"].byte_size:
        raise ReleaseBuildError("raw catalog identity differs from its release ref")
    _validate_exact_content_pins(refs=refs, artifacts=artifacts)
    expected_config = build_reviewed_semantic_config(artifacts)
    if reviewed_config != expected_config:
        raise ReleaseBuildError("reviewed config differs from exact published projections")
    if refs["reviewed_config"].content_id != content_id_for_value(reviewed_config):
        raise ReleaseBuildError("reviewed config content ID is stale")
    _validate_category_truth(scan=scan, artifacts=artifacts)
    _validate_facet_structure(artifacts)
    _validate_evidence_and_resolution(artifacts)
    _validate_stats_structure(artifacts)
    _validate_closed_implementations(artifacts)
    return RuntimeValueGrounder(
        runtime_registry=artifacts.runtime_registry,
        runtime_lexicon=artifacts.runtime_value_lexicon,
        category_registry=artifacts.category_registry,
        capabilities=artifacts.effective_capabilities,
    )


def _validate_exact_content_pins(
    *,
    refs: dict[ArtifactKind, ArtifactRef],
    artifacts: DecodedReleaseArtifacts,
) -> None:
    registry_id = refs["category_registry"].content_id
    assignment_id = refs["product_category_assignment"].content_id
    schema_id = refs["facet_schema"].content_id
    applicability_id = refs["facet_applicability"].content_id
    bindings_id = refs["facet_source_bindings"].content_id
    evidence_id = refs["facet_evidence_store"].content_id
    index_id = refs["product_facet_index"].content_id
    capabilities_id = refs["effective_capabilities"].content_id
    runtime_registry_id = refs["runtime_registry"].content_id
    catalog_id = refs["catalog"].content_id
    registry = artifacts.category_registry
    assignments = artifacts.product_category_assignments
    applicability = artifacts.facet_applicability
    bindings = artifacts.facet_source_bindings
    evidence = artifacts.facet_evidence_store
    index = artifacts.product_facet_index
    stats = artifacts.facet_stats
    capabilities = artifacts.effective_capabilities
    runtime_registry = artifacts.runtime_registry
    lexicon = artifacts.runtime_value_lexicon
    checks = (
        (registry.catalog_id, catalog_id),
        (assignments.catalog_id, catalog_id),
        (assignments.category_graph_id, registry.category_graph_id),
        (applicability.category_registry_id, registry_id),
        (applicability.facet_schema_id, schema_id),
        (bindings.category_registry_id, registry_id),
        (bindings.facet_schema_id, schema_id),
        (bindings.facet_applicability_id, applicability_id),
        (evidence.catalog_id, catalog_id),
        (evidence.product_category_assignment_id, assignment_id),
        (evidence.facet_applicability_id, applicability_id),
        (evidence.facet_source_bindings_id, bindings_id),
        (index.catalog_id, catalog_id),
        (index.product_category_assignment_id, assignment_id),
        (index.facet_applicability_id, applicability_id),
        (index.facet_source_bindings_id, bindings_id),
        (index.facet_evidence_store_id, evidence_id),
        (stats.catalog_id, catalog_id),
        (stats.category_registry_id, registry_id),
        (stats.product_category_assignment_id, assignment_id),
        (stats.facet_schema_id, schema_id),
        (stats.facet_applicability_id, applicability_id),
        (stats.product_facet_index_id, index_id),
        (capabilities.category_registry_id, registry_id),
        (capabilities.facet_schema_id, schema_id),
        (capabilities.facet_applicability_id, applicability_id),
        (capabilities.product_facet_index_id, index_id),
        (runtime_registry.category_registry_id, registry_id),
        (runtime_registry.facet_schema_id, schema_id),
        (runtime_registry.effective_capabilities_id, capabilities_id),
        (lexicon.runtime_registry_id, runtime_registry_id),
        (lexicon.category_registry_id, registry_id),
        (lexicon.facet_applicability_id, applicability_id),
        (lexicon.product_facet_index_id, index_id),
    )
    if any(observed != expected for observed, expected in checks):
        raise ReleaseBuildError("one or more release artifact cross-references are stale")
    policies = (
        evidence.resolution_policy_id,
        index.resolution_policy_id,
        stats.resolution_policy_id,
        capabilities.resolution_policy_id,
        runtime_registry.resolution_policy_id,
        lexicon.resolution_policy_id,
    )
    if any(item != RESOLUTION_POLICY_ID for item in policies):
        raise ReleaseBuildError("release artifacts do not share the pinned resolution policy")


def _validate_category_truth(*, scan: RawCatalogScan, artifacts: DecodedReleaseArtifacts) -> None:
    registry = artifacts.category_registry
    proposal = build_category_graph_proposal(scan)
    if (
        registry.catalog_id != proposal.catalog_id
        or registry.category_graph_id != proposal.category_graph_id
        or registry.nodes != proposal.nodes
    ):
        raise ReleaseBuildError("CategoryRegistry graph differs from the exact raw catalog")
    terminal_ids = {
        item.parent_asin: {category_node_id(normalize_category_path(item.raw_path))}
        for item in scan.records
    }
    validate_product_category_assignment_set(
        artifacts.product_category_assignments,
        registry=registry,
        expected_product_ids=set(terminal_ids),
        terminal_node_ids_by_product=terminal_ids,
    )


def _validate_facet_structure(artifacts: DecodedReleaseArtifacts) -> None:
    facet_ids = {item.id for item in artifacts.facet_schema.facets}
    scope_by_id = {item.id: item for item in artifacts.category_registry.scopes}
    applicability = {item.facet_id: item for item in artifacts.facet_applicability.entries}
    if set(applicability) != facet_ids:
        raise ReleaseBuildError("facet applicability must contain exactly every facet")
    for row in applicability.values():
        if any(scope_id not in scope_by_id for scope_id in row.category_scope_ids):
            raise ReleaseBuildError("facet applicability references an unknown scope")
    for binding in artifacts.facet_source_bindings.bindings:
        if binding.facet_id not in facet_ids:
            raise ReleaseBuildError("source binding references an unknown facet")
        if any(scope_id not in scope_by_id for scope_id in binding.applicable_category_scope_ids):
            raise ReleaseBuildError("source binding references an unknown scope")
    for capability in artifacts.effective_capabilities.entries:
        capability_row = applicability.get(capability.facet_id)
        scope = scope_by_id.get(capability.category_scope_id)
        if capability_row is None or scope is None:
            raise ReleaseBuildError("capability references an unknown facet or scope")
        applicable_members = _scope_members(capability_row, scope_by_id=scope_by_id)
        if not applicable_members.intersection(scope.member_node_ids):
            raise ReleaseBuildError("capability scope is disjoint from facet applicability")


def _validate_evidence_and_resolution(artifacts: DecodedReleaseArtifacts) -> None:
    assignments = {
        item.parent_asin: item for item in artifacts.product_category_assignments.assignments
    }
    bindings = {item.id: item for item in artifacts.facet_source_bindings.bindings}
    definitions = {item.id: item for item in artifacts.facet_schema.facets}
    scope_by_id = {item.id: item for item in artifacts.category_registry.scopes}
    evidence_by_key: dict[tuple[str, str], FacetValueEvidence] = {}
    evidence_by_id: dict[str, FacetValueEvidence] = {}
    for item in artifacts.facet_evidence_store.evidence:
        assignment = assignments.get(item.parent_asin)
        binding = bindings.get(item.binding_id)
        definition = definitions.get(item.facet_id)
        if assignment is None or binding is None or definition is None:
            raise ReleaseBuildError("evidence references an unknown product, binding, or facet")
        if binding.facet_id != item.facet_id or not _binding_applies(
            assignment,
            binding,
            scope_by_id=scope_by_id,
        ):
            raise ReleaseBuildError("evidence is inconsistent with its binding")
        expected_id = evidence_id_for(
            parent_asin=item.parent_asin,
            facet_id=item.facet_id,
            binding_id=item.binding_id,
            status=item.status,
            raw_value_json=item.raw_value_json,
            canonical_value=item.canonical_value,
        )
        if item.id != expected_id:
            raise ReleaseBuildError("evidence ID differs from its canonical payload")
        try:
            raw_value = json.loads(item.raw_value_json)
        except (json.JSONDecodeError, RecursionError, ValueError) as error:
            raise ReleaseBuildError("evidence raw value is invalid JSON") from error
        if canonical_raw_value_json(raw_value) != item.raw_value_json:
            raise ReleaseBuildError("evidence raw value is not canonical JSON")
        if item.canonical_value is not None and not _value_matches_definition(
            item.canonical_value,
            data_type=definition.data_type,
            facet_id=definition.id,
        ):
            raise ReleaseBuildError("evidence value type differs from its facet definition")
        evidence_by_key[(item.parent_asin, item.binding_id)] = item
        evidence_by_id[item.id] = item
    expected_evidence_keys = {
        (assignment.parent_asin, binding.id)
        for assignment in assignments.values()
        for binding in bindings.values()
        if _binding_applies(assignment, binding, scope_by_id=scope_by_id)
    }
    if set(evidence_by_key) != expected_evidence_keys:
        raise ReleaseBuildError("evidence store is incomplete for applicable bindings")
    expected_index = _resolve_expected_index(
        artifacts,
        evidence_by_key=evidence_by_key,
        scope_by_id=scope_by_id,
    )
    if artifacts.product_facet_index.entries != expected_index:
        raise ReleaseBuildError("ProductFacetIndex differs from the pinned resolver output")
    for entry in artifacts.product_facet_index.entries:
        for evidence_id in entry.evidence_ids:
            referenced_evidence = evidence_by_id.get(evidence_id)
            if referenced_evidence is None or (
                referenced_evidence.parent_asin,
                referenced_evidence.facet_id,
            ) != (
                entry.parent_asin,
                entry.facet_id,
            ):
                raise ReleaseBuildError("index evidence reference is invalid")


def _resolve_expected_index(
    artifacts: DecodedReleaseArtifacts,
    *,
    evidence_by_key: dict[tuple[str, str], FacetValueEvidence],
    scope_by_id: Mapping[str, CategoryScope],
) -> tuple[ResolvedProductFacetValue, ...]:
    applicability = {item.facet_id: item for item in artifacts.facet_applicability.entries}
    bindings_by_facet: dict[str, list[FacetSourceBinding]] = {}
    for binding in artifacts.facet_source_bindings.bindings:
        bindings_by_facet.setdefault(binding.facet_id, []).append(binding)
    entries: list[ResolvedProductFacetValue] = []
    for assignment in artifacts.product_category_assignments.assignments:
        for definition in artifacts.facet_schema.facets:
            row = applicability[definition.id]
            if not _applicability_matches(assignment, row, scope_by_id=scope_by_id):
                continue
            accepted: list[tuple[FacetSourceBinding, FacetValueEvidence]] = []
            for binding in bindings_by_facet.get(definition.id, []):
                item = evidence_by_key.get((assignment.parent_asin, binding.id))
                if item is not None and item.status is EvidenceStatus.VALID:
                    accepted.append((binding, item))
            if not accepted:
                continue
            selected_priority = min(binding.priority for binding, _ in accepted)
            selected = tuple(
                (binding, item)
                for binding, item in accepted
                if binding.priority == selected_priority
            )
            resolver_ids = {binding.resolver_id for binding, _ in selected}
            if len(resolver_ids) != 1:
                raise ReleaseBuildError("one selected facet layer has multiple resolvers")
            resolver = require_resolver(next(iter(resolver_ids)))
            values = tuple(cast(ResolvedFacetValue, item.canonical_value) for _, item in selected)
            resolution = resolver(values)
            entries.append(
                ResolvedProductFacetValue(
                    parent_asin=assignment.parent_asin,
                    facet_id=definition.id,
                    status=resolution.status,
                    value=resolution.value,
                    evidence_ids=tuple(sorted(item.id for _, item in selected)),
                    resolution_policy_id=RESOLUTION_POLICY_ID,
                )
            )
    return tuple(sorted(entries, key=lambda item: (item.parent_asin, item.facet_id)))


def _validate_stats_structure(artifacts: DecodedReleaseArtifacts) -> None:
    expected = _expected_stats(artifacts)
    if artifacts.facet_stats != expected:
        raise ReleaseBuildError("CatalogFacetStats differs from the release index")


def _expected_stats(artifacts: DecodedReleaseArtifacts) -> CatalogFacetStatsArtifact:
    registry = artifacts.category_registry
    assignments = artifacts.product_category_assignments
    applicability = {item.facet_id: item for item in artifacts.facet_applicability.entries}
    index = {
        (item.parent_asin, item.facet_id): item for item in artifacts.product_facet_index.entries
    }
    scope_by_id = {item.id: item for item in registry.scopes}
    rows: list[FacetScopeCatalogStats] = []
    for definition in artifacts.facet_schema.facets:
        for scope in registry.scopes:
            counts: Counter[ProductFacetStatus] = Counter()
            values: Counter[str] = Counter()
            scope_count = 0
            for assignment in assignments.assignments:
                if assignment.status is not ProductCategoryAssignmentStatus.KNOWN:
                    continue
                if not set(assignment.leaf_node_ids).intersection(scope.member_node_ids):
                    continue
                scope_count += 1
                if not _applicability_matches(
                    assignment,
                    applicability[definition.id],
                    scope_by_id=scope_by_id,
                ):
                    counts[ProductFacetStatus.NOT_APPLICABLE] += 1
                    continue
                item = index.get((assignment.parent_asin, definition.id))
                if item is None:
                    counts[ProductFacetStatus.UNKNOWN] += 1
                    continue
                counts[item.status] += 1
                if item.status is ProductFacetStatus.KNOWN:
                    if item.value is None:
                        raise ReleaseBuildError("KNOWN index entry has no value")
                    values[canonical_json_text(item.value)] += 1
            value_counts = tuple(
                ResolvedValueCount(canonical_value_json=value, product_count=count)
                for value, count in sorted(
                    values.items(),
                    key=lambda pair: (-pair[1], pair[0].encode("utf-8")),
                )
            )
            rows.append(
                FacetScopeCatalogStats(
                    facet_id=definition.id,
                    category_scope_id=scope.id,
                    scope_product_count=scope_count,
                    known_count=counts[ProductFacetStatus.KNOWN],
                    unknown_count=counts[ProductFacetStatus.UNKNOWN],
                    conflict_count=counts[ProductFacetStatus.CONFLICT],
                    not_applicable_count=counts[ProductFacetStatus.NOT_APPLICABLE],
                    known_value_counts=value_counts,
                )
            )
    return CatalogFacetStatsArtifact(
        schema=CATALOG_FACET_STATS_SCHEMA,
        catalog_id=artifacts.product_facet_index.catalog_id,
        category_registry_id=content_id_for_value(registry),
        product_category_assignment_id=content_id_for_value(assignments),
        facet_schema_id=content_id_for_value(artifacts.facet_schema),
        facet_applicability_id=content_id_for_value(artifacts.facet_applicability),
        product_facet_index_id=content_id_for_value(artifacts.product_facet_index),
        resolution_policy_id=RESOLUTION_POLICY_ID,
        rows=tuple(sorted(rows, key=lambda item: (item.facet_id, item.category_scope_id))),
    )


def _validate_closed_implementations(artifacts: DecodedReleaseArtifacts) -> None:
    ensure_category_builder_runtime()
    for binding in artifacts.facet_source_bindings.bindings:
        require_extractor(binding.extractor_id)
        require_catalog_value_normalizer(binding.catalog_value_normalizer_id)
        require_resolver(binding.resolver_id)
    for record in artifacts.runtime_registry.entries:
        require_intent_value_normalizer(
            record.intent_value_normalizer_id,
            registry=artifacts.category_registry,
        )


def _value_matches_definition(
    value: ResolvedFacetValue,
    *,
    data_type: FacetDataType,
    facet_id: str,
) -> bool:
    expected_type: type[object]
    if data_type is FacetDataType.BOOLEAN:
        expected_type = BooleanValue
    elif data_type is FacetDataType.CATEGORICAL:
        expected_type = CategoricalValue
    elif data_type is FacetDataType.NUMERIC:
        expected_type = NumericValue
    else:
        expected_type = TextValue
    if type(value) is not expected_type:
        return False
    if facet_id != "price":
        return True
    if type(value) is not NumericValue or value.unit != "USD_CENT":
        return False
    return all(endpoint is None or type(endpoint) is int for endpoint in (value.lower, value.upper))


def _require_release_builder(value: str) -> None:
    if value != CATALOG_SEMANTIC_RELEASE_BUILDER_VERSION:
        raise ReleaseBuildError(f"unsupported catalog semantic builder version: {value}")


def _binding_applies(
    assignment: ProductCategoryAssignment,
    binding: FacetSourceBinding,
    *,
    scope_by_id: Mapping[str, CategoryScope],
) -> bool:
    if assignment.status is not ProductCategoryAssignmentStatus.KNOWN:
        return False
    members = _scope_members_from_ids(
        binding.applicable_category_scope_ids,
        scope_by_id=scope_by_id,
    )
    return bool(set(assignment.leaf_node_ids).intersection(members))


def _applicability_matches(
    assignment: ProductCategoryAssignment,
    row: FacetApplicability,
    *,
    scope_by_id: Mapping[str, CategoryScope],
) -> bool:
    if assignment.status is not ProductCategoryAssignmentStatus.KNOWN:
        return False
    members = _scope_members(row, scope_by_id=scope_by_id)
    return bool(set(assignment.leaf_node_ids).intersection(members))


def _scope_members(
    row: FacetApplicability,
    *,
    scope_by_id: Mapping[str, CategoryScope],
) -> set[str]:
    return _scope_members_from_ids(row.category_scope_ids, scope_by_id=scope_by_id)


def _scope_members_from_ids(
    scope_ids: tuple[str, ...],
    *,
    scope_by_id: Mapping[str, CategoryScope],
) -> set[str]:
    members: set[str] = set()
    for scope_id in scope_ids:
        scope = scope_by_id.get(scope_id)
        if scope is None:
            raise ReleaseBuildError("semantic projection references an unknown scope")
        members.update(scope.member_node_ids)
    return members
