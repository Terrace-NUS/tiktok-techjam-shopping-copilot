"""Deterministic materialization of reviewed normative Gate-A artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import cast

from ..canonical import content_id_for_value
from ..category import CategoryRegistry, ProductCategoryAssignmentSet
from ..errors import GateABuildError, GateASelectionError
from .gate_a_implementations import (
    PRICE_EXTRACTOR_ID,
    PRICE_NORMALIZER_ID,
    PRIORITY_EXACT_RESOLVER_ID,
    require_catalog_value_normalizer,
    require_extractor,
    require_resolver,
)
from .gate_a_models import (
    CATALOG_FACET_SCHEMA,
    FACET_APPLICABILITY_SCHEMA,
    FACET_SOURCE_BINDINGS_SCHEMA,
    GATE_A_CANDIDATE_SCHEMA,
    CatalogFacetSchema,
    EvidenceStatus,
    FacetApplicabilitySet,
    FacetDataType,
    FacetSourceBinding,
    FacetSourceBindingSet,
    GateACandidateBuild,
    GateASelection,
    ItemCardinality,
    PriceExtractionAudit,
    PriceExtractionExpectation,
    PriceNormalizationLane,
    ValueCompleteness,
)
from .models import GateASourceProfileBuild, SourceKind, SourceLocator


class _DuplicateJsonKeyError(ValueError):
    pass


def build_gate_a_candidate(
    catalog_path: str | Path,
    *,
    registry: CategoryRegistry,
    assignments: ProductCategoryAssignmentSet,
    category_registry_id: str,
    product_category_assignment_id: str,
    source_profile: GateASourceProfileBuild,
    source_profile_manifest_sha256: str,
    selection: GateASelection,
) -> GateACandidateBuild:
    """Build approved facet artifacts and re-audit the frozen price source."""

    _validate_selection_pins(
        selection,
        registry=registry,
        assignments=assignments,
        category_registry_id=category_registry_id,
        product_category_assignment_id=product_category_assignment_id,
        source_profile=source_profile,
        source_profile_manifest_sha256=source_profile_manifest_sha256,
    )
    _validate_approved_contract(selection, registry=registry, source_profile=source_profile)

    facet_schema = CatalogFacetSchema(
        schema=CATALOG_FACET_SCHEMA,
        facets=tuple(approval.definition for approval in selection.approvals),
    )
    facet_schema_id = content_id_for_value(facet_schema)
    applicability = FacetApplicabilitySet(
        schema=FACET_APPLICABILITY_SCHEMA,
        category_registry_id=category_registry_id,
        facet_schema_id=facet_schema_id,
        entries=tuple(approval.applicability for approval in selection.approvals),
    )
    facet_applicability_id = content_id_for_value(applicability)
    bindings = FacetSourceBindingSet(
        schema=FACET_SOURCE_BINDINGS_SCHEMA,
        category_registry_id=category_registry_id,
        facet_schema_id=facet_schema_id,
        facet_applicability_id=facet_applicability_id,
        bindings=tuple(
            binding for approval in selection.approvals for binding in approval.bindings
        ),
    )

    approval = selection.approvals[0]
    binding = approval.bindings[0]
    price_audit = _audit_price_binding(
        Path(catalog_path),
        binding=binding,
        expected_product_ids={item.parent_asin for item in assignments.assignments},
        expected_catalog_id=selection.catalog_id,
    )
    if price_audit_counts(price_audit) != approval.extraction_expectation:
        raise GateASelectionError(
            "reviewed price extraction expectation differs from rebuilt frozen data"
        )

    return GateACandidateBuild(
        schema=GATE_A_CANDIDATE_SCHEMA,
        catalog_id=selection.catalog_id,
        category_registry_id=category_registry_id,
        product_category_assignment_id=product_category_assignment_id,
        source_profile_manifest_sha256=source_profile_manifest_sha256,
        builder_version=selection.builder_version,
        selection=selection,
        facet_schema=facet_schema,
        applicability=applicability,
        bindings=bindings,
        price_audits=(price_audit,),
    )


def price_audit_counts(audit: PriceExtractionAudit) -> PriceExtractionExpectation:
    """Project a rebuilt audit into the exact reviewed expectation shape."""

    return PriceExtractionExpectation(
        product_count=audit.product_count,
        source_present_count=audit.source_present_count,
        source_missing_count=audit.source_missing_count,
        valid_count=audit.valid_count,
        empty_count=audit.empty_count,
        invalid_count=audit.invalid_count,
        exact_interval_count=audit.exact_interval_count,
        lower_bound_interval_count=audit.lower_bound_interval_count,
        zero_exact_count=audit.zero_exact_count,
    )


def _validate_selection_pins(
    selection: GateASelection,
    *,
    registry: CategoryRegistry,
    assignments: ProductCategoryAssignmentSet,
    category_registry_id: str,
    product_category_assignment_id: str,
    source_profile: GateASourceProfileBuild,
    source_profile_manifest_sha256: str,
) -> None:
    if (
        selection.catalog_id != registry.catalog_id
        or selection.catalog_id != assignments.catalog_id
    ):
        raise GateASelectionError("Gate-A catalog pin is stale")
    if selection.category_registry_id != category_registry_id:
        raise GateASelectionError("Gate-A CategoryRegistry pin is stale")
    if selection.product_category_assignment_id != product_category_assignment_id:
        raise GateASelectionError("Gate-A assignment pin is stale")
    if selection.source_profile_manifest_sha256 != source_profile_manifest_sha256:
        raise GateASelectionError("Gate-A source-profile manifest pin is stale")
    if (
        source_profile.catalog_id != selection.catalog_id
        or source_profile.category_registry_id != category_registry_id
        or source_profile.product_category_assignment_id != product_category_assignment_id
    ):
        raise GateASelectionError("Gate-A source profile has stale upstream pins")


def _validate_approved_contract(
    selection: GateASelection,
    *,
    registry: CategoryRegistry,
    source_profile: GateASourceProfileBuild,
) -> None:
    # This builder version intentionally implements the first reviewed vertical
    # slice only. A later multi-facet builder receives a new immutable version.
    if len(selection.approvals) != 1 or selection.approvals[0].definition.id != "price":
        raise GateASelectionError("Gate-A v0 supports exactly the approved price facet")
    approval = selection.approvals[0]
    if (
        approval.definition.data_type is not FacetDataType.NUMERIC
        or approval.definition.item_cardinality is not ItemCardinality.SINGLE
    ):
        raise GateASelectionError("price must remain NUMERIC and SINGLE")
    if len(approval.bindings) != 1:
        raise GateASelectionError("Gate-A v0 price must have exactly one binding")
    binding = approval.bindings[0]
    if binding.source != SourceLocator(kind=SourceKind.TOP_LEVEL, key="price"):
        raise GateASelectionError("price binding must use exact top-level price")
    if (
        binding.extractor_id != PRICE_EXTRACTOR_ID
        or binding.catalog_value_normalizer_id != PRICE_NORMALIZER_ID
        or binding.resolver_id != PRIORITY_EXACT_RESOLVER_ID
        or binding.priority != 0
        or binding.completeness is not ValueCompleteness.COMPLETE
    ):
        raise GateASelectionError("price binding implementation contract changed")
    if approval.applicability.category_scope_ids != (registry.root_scope_id,):
        raise GateASelectionError("price applicability must equal the catalog root scope")
    if binding.applicable_category_scope_ids != approval.applicability.category_scope_ids:
        raise GateASelectionError("price binding scopes must equal price applicability")

    scope_by_id = {scope.id: scope for scope in registry.scopes}
    facet_nodes = _scope_nodes(approval.applicability.category_scope_ids, scope_by_id)
    for candidate in approval.bindings:
        binding_nodes = _scope_nodes(candidate.applicable_category_scope_ids, scope_by_id)
        if not binding_nodes <= facet_nodes:
            raise GateASelectionError("binding scope exceeds facet applicability")
        if candidate.source not in source_profile.sources:
            raise GateASelectionError("binding source was not observed in the source profile")
        require_extractor(candidate.extractor_id)
        require_catalog_value_normalizer(candidate.catalog_value_normalizer_id)
        require_resolver(candidate.resolver_id)

    resolver_ids = {candidate.resolver_id for candidate in approval.bindings}
    if len(resolver_ids) != 1:
        raise GateASelectionError("one facet cannot declare multiple resolver IDs")
    _validate_equal_priority_overlaps(approval.bindings, scope_by_id)


def _scope_nodes(
    scope_ids: tuple[str, ...],
    scope_by_id: Mapping[str, object],
) -> set[str]:
    result: set[str] = set()
    for scope_id in scope_ids:
        scope = scope_by_id.get(scope_id)
        if scope is None or not hasattr(scope, "member_node_ids"):
            raise GateASelectionError(f"Gate-A references unknown CategoryScope: {scope_id}")
        result.update(cast(tuple[str, ...], scope.member_node_ids))
    return result


def _validate_equal_priority_overlaps(
    bindings: tuple[FacetSourceBinding, ...],
    scope_by_id: Mapping[str, object],
) -> None:
    for index, left in enumerate(bindings):
        for right in bindings[index + 1 :]:
            if (
                left.facet_id != right.facet_id
                or left.source != right.source
                or left.priority != right.priority
            ):
                continue
            left_nodes = _scope_nodes(left.applicable_category_scope_ids, scope_by_id)
            right_nodes = _scope_nodes(right.applicable_category_scope_ids, scope_by_id)
            if not left_nodes.intersection(right_nodes):
                continue
            left_behavior = (
                left.extractor_id,
                left.catalog_value_normalizer_id,
                left.resolver_id,
                left.completeness,
            )
            right_behavior = (
                right.extractor_id,
                right.catalog_value_normalizer_id,
                right.resolver_id,
                right.completeness,
            )
            if left_behavior != right_behavior:
                raise GateASelectionError("equal-priority overlapping bindings are ambiguous")


def _audit_price_binding(
    path: Path,
    *,
    binding: FacetSourceBinding,
    expected_product_ids: set[str],
    expected_catalog_id: str,
) -> PriceExtractionAudit:
    extractor = require_extractor(binding.extractor_id)
    normalizer = require_catalog_value_normalizer(binding.catalog_value_normalizer_id)
    digest = hashlib.sha256()
    seen_ids: set[str] = set()
    source_present_count = 0
    source_missing_count = 0
    valid_count = 0
    empty_count = 0
    invalid_count = 0
    exact_interval_count = 0
    lower_bound_interval_count = 0
    zero_exact_count = 0

    try:
        stream = path.open("rb")
    except OSError as error:
        raise GateABuildError("Gate-A catalog is unavailable") from error
    with stream:
        for line_number, raw_line in enumerate(stream, start=1):
            digest.update(raw_line)
            row = _parse_decimal_row(raw_line, line_number=line_number)
            parent_asin = row.get("parent_asin")
            if type(parent_asin) is not str or not parent_asin:
                raise GateABuildError(f"invalid parent_asin at physical line {line_number}")
            if parent_asin in seen_ids:
                raise GateABuildError("duplicate parent_asin in Gate-A price audit")
            seen_ids.add(parent_asin)
            extracted = extractor(row)
            if not extracted.present:
                source_missing_count += 1
                continue
            source_present_count += 1
            normalized = normalizer(extracted.raw_value)
            if normalized.status is EvidenceStatus.VALID:
                valid_count += 1
                if normalized.lane is PriceNormalizationLane.EXACT:
                    exact_interval_count += 1
                    value = normalized.value
                    if value is not None and value.lower == 0 and value.upper == 0:
                        zero_exact_count += 1
                else:
                    lower_bound_interval_count += 1
            elif normalized.status is EvidenceStatus.EMPTY:
                empty_count += 1
            else:
                invalid_count += 1

    if f"sha256:{digest.hexdigest()}" != expected_catalog_id:
        raise GateABuildError("catalog changed before or during Gate-A price audit")
    if seen_ids != expected_product_ids:
        raise GateABuildError("Gate-A price audit product set differs from assignments")
    return PriceExtractionAudit(
        facet_id=binding.facet_id,
        binding_id=binding.id,
        product_count=len(seen_ids),
        source_present_count=source_present_count,
        source_missing_count=source_missing_count,
        valid_count=valid_count,
        empty_count=empty_count,
        invalid_count=invalid_count,
        exact_interval_count=exact_interval_count,
        lower_bound_interval_count=lower_bound_interval_count,
        zero_exact_count=zero_exact_count,
    )


def _parse_decimal_row(raw_line: bytes, *, line_number: int) -> dict[str, object]:
    if not raw_line.strip():
        raise GateABuildError(f"blank catalog line at physical line {line_number}")
    try:
        parsed: object = json.loads(
            raw_line.decode("utf-8"),
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
        raise GateABuildError(f"invalid strict JSON at physical line {line_number}") from error
    if type(parsed) is not dict:
        raise GateABuildError(f"catalog row is not an object at physical line {line_number}")
    return cast(dict[str, object], parsed)


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_nonfinite_token(raw: str) -> object:
    raise ValueError(f"non-finite JSON token: {raw}")
