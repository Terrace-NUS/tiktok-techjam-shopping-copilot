"""Read-only CS3 evidence extraction, structured resolution, and statistics."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import cast

import rfc8785

from ..canonical import (
    JsonValue,
    canonical_json_bytes,
    canonical_json_text,
    content_id_for_value,
    sha256_hex,
)
from ..category import (
    CategoryRegistry,
    ProductCategoryAssignment,
    ProductCategoryAssignmentSet,
    ProductCategoryAssignmentStatus,
)
from ..errors import ResolutionBuildError
from ..raw_catalog import OFFICIAL_PRODUCT_COUNT, scan_raw_catalog
from .gate_a_implementations import (
    USD_CENT_UNIT,
    require_catalog_value_normalizer,
    require_extractor,
    require_resolver,
)
from .gate_a_models import (
    BooleanValue,
    CatalogFacetDefinition,
    CategoricalValue,
    EvidenceStatus,
    FacetApplicability,
    FacetDataType,
    FacetSourceBinding,
    GateACandidateBuild,
    ItemCardinality,
    NumericValue,
    ProductFacetStatus,
    ResolvedFacetValue,
    TextValue,
)
from .resolution_models import (
    CATALOG_FACET_STATS_SCHEMA,
    FACET_EVIDENCE_STORE_SCHEMA,
    PRODUCT_FACET_INDEX_SCHEMA,
    RESOLUTION_CANDIDATE_SCHEMA,
    RESOLUTION_POLICY_ID,
    CatalogFacetStatsArtifact,
    FacetEvidenceStore,
    FacetScopeCatalogStats,
    FacetValueEvidence,
    ProductFacetIndex,
    ResolutionCandidateBuild,
    ResolvedProductFacetValue,
    ResolvedValueCount,
)


class _DuplicateJsonKeyError(ValueError):
    pass


def build_resolution_candidate(
    catalog_path: str | Path,
    *,
    registry: CategoryRegistry,
    assignments: ProductCategoryAssignmentSet,
    category_registry_id: str,
    product_category_assignment_id: str,
    gate_a: GateACandidateBuild,
    expected_product_count: int = OFFICIAL_PRODUCT_COUNT,
    enforce_official_gate: bool = True,
) -> ResolutionCandidateBuild:
    """Build CS3 artifacts without mutating or reserializing the raw catalog."""

    _validate_upstream_pins(
        registry=registry,
        assignments=assignments,
        category_registry_id=category_registry_id,
        product_category_assignment_id=product_category_assignment_id,
        gate_a=gate_a,
        expected_product_count=expected_product_count,
        enforce_official_gate=enforce_official_gate,
    )
    scan = scan_raw_catalog(catalog_path, expected_product_count=expected_product_count)
    if scan.catalog_id != gate_a.catalog_id:
        raise ResolutionBuildError("CS3 raw catalog ID differs from reviewed Gate-A input")

    facet_schema_id = content_id_for_value(gate_a.facet_schema)
    facet_applicability_id = content_id_for_value(gate_a.applicability)
    facet_source_bindings_id = content_id_for_value(gate_a.bindings)
    evidence_store = _build_evidence_store(
        Path(catalog_path),
        assignments=assignments,
        registry=registry,
        gate_a=gate_a,
        product_category_assignment_id=product_category_assignment_id,
        facet_applicability_id=facet_applicability_id,
        facet_source_bindings_id=facet_source_bindings_id,
    )
    validate_evidence_store(
        evidence_store,
        registry=registry,
        assignments=assignments,
        gate_a=gate_a,
        product_category_assignment_id=product_category_assignment_id,
    )
    product_facet_index = _build_product_facet_index(
        registry=registry,
        assignments=assignments,
        gate_a=gate_a,
        evidence_store=evidence_store,
    )
    validate_product_facet_index(
        product_facet_index,
        registry=registry,
        assignments=assignments,
        gate_a=gate_a,
        evidence_store=evidence_store,
    )
    stats = _build_stats(
        registry=registry,
        assignments=assignments,
        gate_a=gate_a,
        category_registry_id=category_registry_id,
        facet_schema_id=facet_schema_id,
        product_facet_index=product_facet_index,
    )
    validate_stats_artifact(
        stats,
        registry=registry,
        assignments=assignments,
        gate_a=gate_a,
        category_registry_id=category_registry_id,
        facet_schema_id=facet_schema_id,
        product_facet_index=product_facet_index,
    )
    return ResolutionCandidateBuild(
        schema=RESOLUTION_CANDIDATE_SCHEMA,
        builder_version=gate_a.builder_version,
        category_registry_id=category_registry_id,
        facet_schema_id=facet_schema_id,
        gate_a_selection_id=content_id_for_value(gate_a.selection),
        evidence_store=evidence_store,
        product_facet_index=product_facet_index,
        stats=stats,
    )


def evidence_id_for(
    *,
    parent_asin: str,
    facet_id: str,
    binding_id: str,
    status: EvidenceStatus,
    raw_value_json: str,
    canonical_value: ResolvedFacetValue | None,
) -> str:
    """Return the contract's full-payload evidence identity."""

    payload = {
        "parent_asin": parent_asin,
        "facet_id": facet_id,
        "binding_id": binding_id,
        "status": status,
        "raw_value_json": raw_value_json,
        "canonical_value": canonical_value,
    }
    return f"ev_{sha256_hex(canonical_json_bytes(payload))}"


def canonical_raw_value_json(value: object) -> str:
    """Return RFC 8785 text for one opaque copied source value.

    Raw values are not semantic scalar fields. In particular, RFC 8785 is allowed
    to serialize a raw negative-zero JSON number as ``0`` even though negative zero
    is rejected as a canonical semantic numeric value.
    """

    _validate_opaque_json(value)
    try:
        return rfc8785.dumps(cast(JsonValue, value)).decode("utf-8")
    except rfc8785.CanonicalizationError as error:
        raise ResolutionBuildError("raw source value cannot be represented by JCS") from error


def validate_evidence_store(
    store: FacetEvidenceStore,
    *,
    registry: CategoryRegistry,
    assignments: ProductCategoryAssignmentSet,
    gate_a: GateACandidateBuild,
    product_category_assignment_id: str,
) -> None:
    """Validate all CS3 evidence identities, types, applicability, and pins."""

    if type(store) is not FacetEvidenceStore:
        raise TypeError("validate_evidence_store requires FacetEvidenceStore")
    expected_pins = (
        gate_a.catalog_id,
        product_category_assignment_id,
        content_id_for_value(gate_a.applicability),
        content_id_for_value(gate_a.bindings),
        RESOLUTION_POLICY_ID,
    )
    observed_pins = (
        store.catalog_id,
        store.product_category_assignment_id,
        store.facet_applicability_id,
        store.facet_source_bindings_id,
        store.resolution_policy_id,
    )
    if observed_pins != expected_pins:
        raise ResolutionBuildError("FacetEvidenceStore has stale upstream pins")

    assignment_by_id = {item.parent_asin: item for item in assignments.assignments}
    binding_by_id = {item.id: item for item in gate_a.bindings.bindings}
    definition_by_id = {item.id: item for item in gate_a.facet_schema.facets}
    for item in store.evidence:
        assignment = assignment_by_id.get(item.parent_asin)
        binding = binding_by_id.get(item.binding_id)
        if assignment is None or binding is None:
            raise ResolutionBuildError("evidence references an unknown product or binding")
        if binding.facet_id != item.facet_id:
            raise ResolutionBuildError("evidence facet differs from its binding")
        if not _binding_applies(assignment, binding, registry=registry):
            raise ResolutionBuildError("evidence binding is not applicable to its product")
        if (
            canonical_raw_value_json(_parse_raw_value_json(item.raw_value_json))
            != item.raw_value_json
        ):
            raise ResolutionBuildError("evidence raw_value_json is not canonical")
        definition = definition_by_id.get(item.facet_id)
        if definition is None:
            raise ResolutionBuildError("evidence references an unknown facet")
        if item.canonical_value is not None:
            _validate_value_variant(item.canonical_value, definition=definition)
        expected_id = evidence_id_for(
            parent_asin=item.parent_asin,
            facet_id=item.facet_id,
            binding_id=item.binding_id,
            status=item.status,
            raw_value_json=item.raw_value_json,
            canonical_value=item.canonical_value,
        )
        if item.id != expected_id:
            raise ResolutionBuildError("evidence ID differs from its canonical payload")


def validate_product_facet_index(
    index: ProductFacetIndex,
    *,
    registry: CategoryRegistry,
    assignments: ProductCategoryAssignmentSet,
    gate_a: GateACandidateBuild,
    evidence_store: FacetEvidenceStore,
) -> None:
    """Re-resolve the complete sparse index and require exact equality."""

    if type(index) is not ProductFacetIndex:
        raise TypeError("validate_product_facet_index requires ProductFacetIndex")
    expected = _build_product_facet_index(
        registry=registry,
        assignments=assignments,
        gate_a=gate_a,
        evidence_store=evidence_store,
    )
    if index != expected:
        raise ResolutionBuildError("ProductFacetIndex differs from structured resolution")


def validate_stats_artifact(
    stats: CatalogFacetStatsArtifact,
    *,
    registry: CategoryRegistry,
    assignments: ProductCategoryAssignmentSet,
    gate_a: GateACandidateBuild,
    category_registry_id: str,
    facet_schema_id: str,
    product_facet_index: ProductFacetIndex,
) -> None:
    """Recompute every facet/scope row from the same index and require equality."""

    if type(stats) is not CatalogFacetStatsArtifact:
        raise TypeError("validate_stats_artifact requires CatalogFacetStatsArtifact")
    expected = _build_stats(
        registry=registry,
        assignments=assignments,
        gate_a=gate_a,
        category_registry_id=category_registry_id,
        facet_schema_id=facet_schema_id,
        product_facet_index=product_facet_index,
    )
    if stats != expected:
        raise ResolutionBuildError("CatalogFacetStatsArtifact differs from resolved index")


def lookup_product_facet(
    parent_asin: str,
    facet_id: str,
    *,
    registry: CategoryRegistry,
    assignments: ProductCategoryAssignmentSet,
    gate_a: GateACandidateBuild,
    index: ProductFacetIndex,
) -> ResolvedProductFacetValue:
    """Perform the contract's exact sparse-index lookup for one product and facet."""

    assignment_by_id = {item.parent_asin: item for item in assignments.assignments}
    assignment = assignment_by_id.get(parent_asin)
    if assignment is None:
        raise KeyError(f"unknown catalog product: {parent_asin}")
    applicability_by_facet = {item.facet_id: item for item in gate_a.applicability.entries}
    applicability = applicability_by_facet.get(facet_id)
    if applicability is None:
        raise KeyError(f"unknown approved facet: {facet_id}")
    if assignment.status is not ProductCategoryAssignmentStatus.KNOWN:
        return _implicit_result(parent_asin, facet_id, ProductFacetStatus.UNKNOWN)
    if not _applicability_matches(assignment, applicability, registry=registry):
        return _implicit_result(parent_asin, facet_id, ProductFacetStatus.NOT_APPLICABLE)
    entry_by_key = {(item.parent_asin, item.facet_id): item for item in index.entries}
    return entry_by_key.get(
        (parent_asin, facet_id),
        _implicit_result(parent_asin, facet_id, ProductFacetStatus.UNKNOWN),
    )


def _build_evidence_store(
    catalog_path: Path,
    *,
    assignments: ProductCategoryAssignmentSet,
    registry: CategoryRegistry,
    gate_a: GateACandidateBuild,
    product_category_assignment_id: str,
    facet_applicability_id: str,
    facet_source_bindings_id: str,
) -> FacetEvidenceStore:
    assignment_by_id = {item.parent_asin: item for item in assignments.assignments}
    evidence: list[FacetValueEvidence] = []
    seen_products: set[str] = set()
    digest = hashlib.sha256()

    try:
        stream = catalog_path.open("rb")
    except OSError as error:
        raise ResolutionBuildError("CS3 catalog is unavailable") from error
    with stream:
        for line_number, raw_line in enumerate(stream, start=1):
            digest.update(raw_line)
            standard_row, decimal_row = _parse_catalog_row(raw_line, line_number=line_number)
            parent_asin = standard_row.get("parent_asin")
            if type(parent_asin) is not str or parent_asin != decimal_row.get("parent_asin"):
                raise ResolutionBuildError(f"invalid parent_asin at physical line {line_number}")
            if parent_asin in seen_products:
                raise ResolutionBuildError("duplicate parent_asin during CS3 evidence build")
            seen_products.add(parent_asin)
            assignment = assignment_by_id.get(parent_asin)
            if assignment is None:
                raise ResolutionBuildError("catalog product is absent from category assignments")

            for binding in gate_a.bindings.bindings:
                if not _binding_applies(assignment, binding, registry=registry):
                    continue
                extractor = require_extractor(binding.extractor_id)
                standard_extraction = extractor(standard_row)
                decimal_extraction = extractor(decimal_row)
                if standard_extraction.present != decimal_extraction.present:
                    raise ResolutionBuildError("dual JSON parse changed source presence")
                if not decimal_extraction.present:
                    continue
                normalizer = require_catalog_value_normalizer(binding.catalog_value_normalizer_id)
                normalized = normalizer(decimal_extraction.raw_value)
                raw_value_json = canonical_raw_value_json(standard_extraction.raw_value)
                evidence.append(
                    FacetValueEvidence(
                        id=evidence_id_for(
                            parent_asin=parent_asin,
                            facet_id=binding.facet_id,
                            binding_id=binding.id,
                            status=normalized.status,
                            raw_value_json=raw_value_json,
                            canonical_value=normalized.value,
                        ),
                        parent_asin=parent_asin,
                        facet_id=binding.facet_id,
                        binding_id=binding.id,
                        status=normalized.status,
                        raw_value_json=raw_value_json,
                        canonical_value=normalized.value,
                    )
                )

    if f"sha256:{digest.hexdigest()}" != gate_a.catalog_id:
        raise ResolutionBuildError("catalog changed before or during CS3 evidence build")
    if seen_products != set(assignment_by_id):
        raise ResolutionBuildError("CS3 catalog product set differs from category assignments")
    return FacetEvidenceStore(
        schema=FACET_EVIDENCE_STORE_SCHEMA,
        catalog_id=gate_a.catalog_id,
        product_category_assignment_id=product_category_assignment_id,
        facet_applicability_id=facet_applicability_id,
        facet_source_bindings_id=facet_source_bindings_id,
        resolution_policy_id=RESOLUTION_POLICY_ID,
        evidence=tuple(sorted(evidence, key=lambda item: (item.parent_asin, item.binding_id))),
    )


def _build_product_facet_index(
    *,
    registry: CategoryRegistry,
    assignments: ProductCategoryAssignmentSet,
    gate_a: GateACandidateBuild,
    evidence_store: FacetEvidenceStore,
) -> ProductFacetIndex:
    evidence_by_key = {
        (item.parent_asin, item.binding_id): item for item in evidence_store.evidence
    }
    bindings_by_facet: dict[str, list[FacetSourceBinding]] = defaultdict(list)
    for binding in gate_a.bindings.bindings:
        bindings_by_facet[binding.facet_id].append(binding)
    applicability_by_facet = {item.facet_id: item for item in gate_a.applicability.entries}
    entries: list[ResolvedProductFacetValue] = []

    for assignment in assignments.assignments:
        if assignment.status is not ProductCategoryAssignmentStatus.KNOWN:
            continue
        for definition in gate_a.facet_schema.facets:
            applicability = applicability_by_facet[definition.id]
            if not _applicability_matches(assignment, applicability, registry=registry):
                continue
            applicable_bindings = tuple(
                binding
                for binding in bindings_by_facet[definition.id]
                if _binding_applies(assignment, binding, registry=registry)
            )
            valid_pairs = tuple(
                (binding, evidence_by_key.get((assignment.parent_asin, binding.id)))
                for binding in applicable_bindings
            )
            accepted = tuple(
                (binding, item)
                for binding, item in valid_pairs
                if item is not None and item.status is EvidenceStatus.VALID
            )
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
                raise ResolutionBuildError("one selected facet layer has multiple resolver IDs")
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

    return ProductFacetIndex(
        schema=PRODUCT_FACET_INDEX_SCHEMA,
        catalog_id=evidence_store.catalog_id,
        product_category_assignment_id=evidence_store.product_category_assignment_id,
        facet_applicability_id=evidence_store.facet_applicability_id,
        facet_source_bindings_id=evidence_store.facet_source_bindings_id,
        facet_evidence_store_id=content_id_for_value(evidence_store),
        resolution_policy_id=RESOLUTION_POLICY_ID,
        entries=tuple(sorted(entries, key=lambda item: (item.parent_asin, item.facet_id))),
    )


def _build_stats(
    *,
    registry: CategoryRegistry,
    assignments: ProductCategoryAssignmentSet,
    gate_a: GateACandidateBuild,
    category_registry_id: str,
    facet_schema_id: str,
    product_facet_index: ProductFacetIndex,
) -> CatalogFacetStatsArtifact:
    applicability_by_facet = {item.facet_id: item for item in gate_a.applicability.entries}
    entry_by_key = {(item.parent_asin, item.facet_id): item for item in product_facet_index.entries}
    rows: list[FacetScopeCatalogStats] = []
    for definition in gate_a.facet_schema.facets:
        applicability = applicability_by_facet[definition.id]
        for scope in registry.scopes:
            counts: Counter[ProductFacetStatus] = Counter()
            value_counts: Counter[str] = Counter()
            scope_product_count = 0
            for assignment in assignments.assignments:
                if assignment.status is not ProductCategoryAssignmentStatus.KNOWN:
                    continue
                if not set(assignment.leaf_node_ids).intersection(scope.member_node_ids):
                    continue
                scope_product_count += 1
                if not _applicability_matches(assignment, applicability, registry=registry):
                    counts[ProductFacetStatus.NOT_APPLICABLE] += 1
                    continue
                entry = entry_by_key.get((assignment.parent_asin, definition.id))
                if entry is None:
                    counts[ProductFacetStatus.UNKNOWN] += 1
                    continue
                counts[entry.status] += 1
                if entry.status is ProductFacetStatus.KNOWN:
                    if entry.value is None:
                        raise ResolutionBuildError("KNOWN index entry lost its value")
                    value_counts[canonical_json_text(entry.value)] += 1
            resolved_counts = tuple(
                ResolvedValueCount(canonical_value_json=value, product_count=count)
                for value, count in sorted(
                    value_counts.items(),
                    key=lambda item: (-item[1], item[0].encode("utf-8")),
                )
            )
            rows.append(
                FacetScopeCatalogStats(
                    facet_id=definition.id,
                    category_scope_id=scope.id,
                    scope_product_count=scope_product_count,
                    known_count=counts[ProductFacetStatus.KNOWN],
                    unknown_count=counts[ProductFacetStatus.UNKNOWN],
                    conflict_count=counts[ProductFacetStatus.CONFLICT],
                    not_applicable_count=counts[ProductFacetStatus.NOT_APPLICABLE],
                    known_value_counts=resolved_counts,
                )
            )
    return CatalogFacetStatsArtifact(
        schema=CATALOG_FACET_STATS_SCHEMA,
        catalog_id=product_facet_index.catalog_id,
        category_registry_id=category_registry_id,
        product_category_assignment_id=product_facet_index.product_category_assignment_id,
        facet_schema_id=facet_schema_id,
        facet_applicability_id=product_facet_index.facet_applicability_id,
        product_facet_index_id=content_id_for_value(product_facet_index),
        resolution_policy_id=RESOLUTION_POLICY_ID,
        rows=tuple(sorted(rows, key=lambda item: (item.facet_id, item.category_scope_id))),
    )


def _validate_upstream_pins(
    *,
    registry: CategoryRegistry,
    assignments: ProductCategoryAssignmentSet,
    category_registry_id: str,
    product_category_assignment_id: str,
    gate_a: GateACandidateBuild,
    expected_product_count: int,
    enforce_official_gate: bool,
) -> None:
    if type(expected_product_count) is not int or expected_product_count <= 0:
        raise ValueError("expected_product_count must be positive")
    if enforce_official_gate and expected_product_count != OFFICIAL_PRODUCT_COUNT:
        raise ResolutionBuildError("official CS3 gate requires exactly 50,000 products")
    if len(assignments.assignments) != expected_product_count:
        raise ResolutionBuildError("CS3 assignment count differs from expected product count")
    if enforce_official_gate and any(
        item.status is not ProductCategoryAssignmentStatus.KNOWN for item in assignments.assignments
    ):
        raise ResolutionBuildError("official CS3 gate requires all category assignments KNOWN")
    if registry.catalog_id != assignments.catalog_id or registry.catalog_id != gate_a.catalog_id:
        raise ResolutionBuildError("CS3 upstream catalog pins differ")
    if gate_a.category_registry_id != category_registry_id:
        raise ResolutionBuildError("CS3 Gate-A CategoryRegistry pin is stale")
    if gate_a.product_category_assignment_id != product_category_assignment_id:
        raise ResolutionBuildError("CS3 Gate-A assignment pin is stale")
    if registry.category_graph_id != assignments.category_graph_id:
        raise ResolutionBuildError("CS3 category graph pins differ")


def _binding_applies(
    assignment: ProductCategoryAssignment,
    binding: FacetSourceBinding,
    *,
    registry: CategoryRegistry,
) -> bool:
    if assignment.status is not ProductCategoryAssignmentStatus.KNOWN:
        return False
    scope_by_id = {item.id: item for item in registry.scopes}
    member_ids: set[str] = set()
    for scope_id in binding.applicable_category_scope_ids:
        try:
            member_ids.update(scope_by_id[scope_id].member_node_ids)
        except KeyError as error:
            raise ResolutionBuildError(f"binding references unknown scope: {scope_id}") from error
    return bool(set(assignment.leaf_node_ids).intersection(member_ids))


def _applicability_matches(
    assignment: ProductCategoryAssignment,
    applicability: FacetApplicability,
    *,
    registry: CategoryRegistry,
) -> bool:
    if assignment.status is not ProductCategoryAssignmentStatus.KNOWN:
        return False
    scope_by_id = {item.id: item for item in registry.scopes}
    member_ids: set[str] = set()
    for scope_id in applicability.category_scope_ids:
        try:
            member_ids.update(scope_by_id[scope_id].member_node_ids)
        except KeyError as error:
            raise ResolutionBuildError(f"facet references unknown scope: {scope_id}") from error
    return bool(set(assignment.leaf_node_ids).intersection(member_ids))


def _validate_value_variant(
    value: ResolvedFacetValue,
    *,
    definition: CatalogFacetDefinition,
) -> None:
    expected_type: type[object]
    if definition.data_type is FacetDataType.BOOLEAN:
        expected_type = BooleanValue
    elif definition.data_type is FacetDataType.CATEGORICAL:
        expected_type = CategoricalValue
    elif definition.data_type is FacetDataType.NUMERIC:
        expected_type = NumericValue
    else:
        expected_type = TextValue
    if type(value) is not expected_type:
        raise ResolutionBuildError("facet value variant differs from CatalogFacetDefinition")
    if type(value) is CategoricalValue:
        if definition.item_cardinality is ItemCardinality.SINGLE and len(value.values) != 1:
            raise ResolutionBuildError("SINGLE categorical evidence has multiple values")
    if definition.id == "price":
        if type(value) is not NumericValue:
            raise ResolutionBuildError("price must resolve to NumericValue")
        if value.unit != USD_CENT_UNIT:
            raise ResolutionBuildError("price must use USD_CENT")
        for endpoint in (value.lower, value.upper):
            if endpoint is not None and type(endpoint) is not int:
                raise ResolutionBuildError("price endpoints must be integer cents")


def _implicit_result(
    parent_asin: str,
    facet_id: str,
    status: ProductFacetStatus,
) -> ResolvedProductFacetValue:
    return ResolvedProductFacetValue(
        parent_asin=parent_asin,
        facet_id=facet_id,
        status=status,
        value=None,
        evidence_ids=(),
        resolution_policy_id=RESOLUTION_POLICY_ID,
    )


def _parse_catalog_row(
    raw_line: bytes,
    *,
    line_number: int,
) -> tuple[dict[str, object], dict[str, object]]:
    if not raw_line.strip():
        raise ResolutionBuildError(f"blank catalog line at physical line {line_number}")
    try:
        text = raw_line.decode("utf-8")
        standard: object = json.loads(
            text,
            parse_constant=_reject_nonfinite_token,
            object_pairs_hook=_object_without_duplicate_keys,
        )
        decimal: object = json.loads(
            text,
            parse_float=Decimal,
            parse_constant=_reject_nonfinite_token,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (
        UnicodeDecodeError,
        _DuplicateJsonKeyError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as error:
        raise ResolutionBuildError(f"invalid strict JSON at physical line {line_number}") from error
    if type(standard) is not dict or type(decimal) is not dict:
        raise ResolutionBuildError(f"catalog row is not an object at physical line {line_number}")
    return cast(dict[str, object], standard), cast(dict[str, object], decimal)


def _parse_raw_value_json(raw_value_json: str) -> object:
    try:
        return json.loads(
            raw_value_json,
            parse_constant=_reject_nonfinite_token,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (
        _DuplicateJsonKeyError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as error:
        raise ResolutionBuildError("evidence raw_value_json is invalid") from error


def _validate_opaque_json(value: object) -> None:
    if value is None or type(value) in (bool, int, float, str):
        return
    if type(value) is list:
        for item in cast(list[object], value):
            _validate_opaque_json(item)
        return
    if type(value) is dict:
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                raise ResolutionBuildError("raw JSON object key is not a string")
            _validate_opaque_json(item)
        return
    raise ResolutionBuildError(f"unsupported opaque raw JSON value: {type(value).__name__}")


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_nonfinite_token(raw: str) -> object:
    raise ValueError(f"non-finite JSON token: {raw}")
