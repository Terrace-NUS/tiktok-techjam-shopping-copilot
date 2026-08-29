"""Deterministic CS5A projection into session-context runtime foundations."""

from __future__ import annotations

from shopping_copilot.session_context import (
    CATEGORICAL_OPERATORS,
    NUMERIC_OPERATORS,
    FacetKind,
    FacetRegistry,
    FacetSpec,
    Operator,
)

from ..canonical import content_id_for_value
from ..category import CategoryRegistry
from ..errors import RuntimeProjectionBuildError
from ..facet.gate_a_models import FacetDataType, GateACandidateBuild
from ..facet.gate_b_models import (
    PRICE_INTENT_NORMALIZER_ID,
    EffectiveFacetCapabilitySet,
    GateBCandidateBuild,
)
from ..facet.resolution_models import RESOLUTION_POLICY_ID
from .models import (
    CATEGORY_SCOPE_ID_NORMALIZER_ID,
    RUNTIME_FACET_REGISTRY_SCHEMA,
    RUNTIME_PROJECTION_BUILDER_VERSION,
    RUNTIME_PROJECTION_CANDIDATE_SCHEMA,
    RUNTIME_VALUE_LEXICON_SCHEMA,
    SYSTEM_PRODUCT_CATEGORY_FACET_ID,
    NumericRuntimeDomain,
    RuntimeFacetRegistryArtifact,
    RuntimeFacetSpecRecord,
    RuntimeProjectionCandidateBuild,
    RuntimeValueLexicon,
)
from .normalizers import require_intent_value_normalizer


def build_runtime_projection_candidate(
    *,
    registry: CategoryRegistry,
    gate_a: GateACandidateBuild,
    gate_b: GateBCandidateBuild,
) -> RuntimeProjectionCandidateBuild:
    """Project approved semantics into a declarative price/category registry and lexicon."""

    _validate_upstream(registry=registry, gate_a=gate_a, gate_b=gate_b)
    projected_facet_ids = _projected_ordinary_facet_ids(gate_b.capabilities)
    definitions = {item.id: item for item in gate_a.facet_schema.facets}
    records: list[RuntimeFacetSpecRecord] = []
    domains: list[NumericRuntimeDomain] = []
    for facet_id in projected_facet_ids:
        definition = definitions[facet_id]
        if definition.data_type is not FacetDataType.NUMERIC or facet_id != "price":
            raise RuntimeProjectionBuildError(
                "CS5A runtime v0 supports only projected numeric price"
            )
        records.append(
            RuntimeFacetSpecRecord(
                facet_id="price",
                kind="numeric",
                operator_values=_numeric_operator_values(),
                intent_value_normalizer_id=gate_b.selection.intent_value_normalizer_id,
            )
        )
        domains.append(
            NumericRuntimeDomain(
                kind="numeric",
                facet_id="price",
                intent_value_normalizer_id=gate_b.selection.intent_value_normalizer_id,
                canonical_unit="USD_CENT",
                integer_only=True,
            )
        )
    records.append(
        RuntimeFacetSpecRecord(
            facet_id=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
            kind="categorical",
            operator_values=_categorical_operator_values(),
            intent_value_normalizer_id=CATEGORY_SCOPE_ID_NORMALIZER_ID,
        )
    )
    runtime_registry = RuntimeFacetRegistryArtifact(
        schema=RUNTIME_FACET_REGISTRY_SCHEMA,
        category_registry_id=content_id_for_value(registry),
        facet_schema_id=content_id_for_value(gate_a.facet_schema),
        effective_capabilities_id=content_id_for_value(gate_b.capabilities),
        resolution_policy_id=RESOLUTION_POLICY_ID,
        entries=tuple(sorted(records, key=lambda item: item.facet_id)),
    )
    runtime_lexicon = RuntimeValueLexicon(
        schema=RUNTIME_VALUE_LEXICON_SCHEMA,
        runtime_registry_id=content_id_for_value(runtime_registry),
        category_registry_id=content_id_for_value(registry),
        facet_applicability_id=content_id_for_value(gate_a.applicability),
        product_facet_index_id=gate_b.capabilities.product_facet_index_id,
        resolution_policy_id=RESOLUTION_POLICY_ID,
        domains=tuple(sorted(domains, key=lambda item: item.facet_id)),
    )
    candidate = RuntimeProjectionCandidateBuild(
        schema=RUNTIME_PROJECTION_CANDIDATE_SCHEMA,
        builder_version=RUNTIME_PROJECTION_BUILDER_VERSION,
        catalog_id=gate_b.catalog_id,
        gate_b_selection_id=content_id_for_value(gate_b.selection),
        effective_capabilities_id=content_id_for_value(gate_b.capabilities),
        runtime_registry=runtime_registry,
        runtime_lexicon=runtime_lexicon,
    )
    project_session_facet_registry(
        runtime_registry=runtime_registry,
        runtime_lexicon=runtime_lexicon,
        category_registry=registry,
        capabilities=gate_b.capabilities,
    )
    return candidate


def project_session_facet_registry(
    *,
    runtime_registry: RuntimeFacetRegistryArtifact,
    runtime_lexicon: RuntimeValueLexicon,
    category_registry: CategoryRegistry,
    capabilities: EffectiveFacetCapabilitySet,
) -> FacetRegistry:
    """Resolve declarative records into trusted release-bound FacetSpec callables."""

    if type(runtime_registry) is not RuntimeFacetRegistryArtifact:
        raise TypeError("runtime registry projection requires exact artifact type")
    if type(runtime_lexicon) is not RuntimeValueLexicon:
        raise TypeError("runtime registry projection requires exact lexicon type")
    if type(category_registry) is not CategoryRegistry:
        raise TypeError("runtime registry projection requires CategoryRegistry")
    if type(capabilities) is not EffectiveFacetCapabilitySet:
        raise TypeError("runtime registry projection requires effective capabilities")
    if runtime_registry.category_registry_id != content_id_for_value(category_registry):
        raise RuntimeProjectionBuildError("runtime registry CategoryRegistry pin is stale")
    if runtime_registry.effective_capabilities_id != content_id_for_value(capabilities):
        raise RuntimeProjectionBuildError("runtime registry capability pin is stale")
    if (
        runtime_lexicon.runtime_registry_id != content_id_for_value(runtime_registry)
        or runtime_lexicon.category_registry_id != runtime_registry.category_registry_id
        or runtime_lexicon.product_facet_index_id != capabilities.product_facet_index_id
        or runtime_lexicon.resolution_policy_id != runtime_registry.resolution_policy_id
    ):
        raise RuntimeProjectionBuildError("runtime registry and lexicon pins differ")

    projected = _projected_ordinary_facet_ids(capabilities)
    ordinary_records = {
        item.facet_id: item
        for item in runtime_registry.entries
        if item.facet_id != SYSTEM_PRODUCT_CATEGORY_FACET_ID
    }
    domains: dict[str, NumericRuntimeDomain] = {
        item.facet_id: item for item in runtime_lexicon.domains
    }
    if tuple(sorted(ordinary_records)) != projected or tuple(sorted(domains)) != projected:
        raise RuntimeProjectionBuildError(
            "runtime ordinary records/domains differ from approved projected facets"
        )
    specs: list[FacetSpec] = []
    for record in runtime_registry.entries:
        if record.facet_id == SYSTEM_PRODUCT_CATEGORY_FACET_ID:
            if record.intent_value_normalizer_id != CATEGORY_SCOPE_ID_NORMALIZER_ID:
                raise RuntimeProjectionBuildError("reserved category normalizer ID is invalid")
        else:
            domain = domains[record.facet_id]
            if domain.intent_value_normalizer_id != record.intent_value_normalizer_id:
                raise RuntimeProjectionBuildError("runtime domain normalizer differs from registry")
        normalizer = require_intent_value_normalizer(
            record.intent_value_normalizer_id,
            registry=category_registry,
        )
        specs.append(
            FacetSpec(
                id=record.facet_id,
                kind=FacetKind(record.kind),
                operators=frozenset(Operator(value) for value in record.operator_values),
                normalizer=normalizer,
            )
        )
    result = FacetRegistry(specs=specs)
    _validate_normalizer_fixed_points(result, category_registry=category_registry)
    return result


def _validate_upstream(
    *,
    registry: CategoryRegistry,
    gate_a: GateACandidateBuild,
    gate_b: GateBCandidateBuild,
) -> None:
    if gate_b.catalog_id != registry.catalog_id:
        raise RuntimeProjectionBuildError("runtime projection catalog pin is stale")
    if gate_b.capabilities.category_registry_id != content_id_for_value(registry):
        raise RuntimeProjectionBuildError("runtime projection CategoryRegistry pin is stale")
    if gate_b.capabilities.facet_schema_id != content_id_for_value(gate_a.facet_schema):
        raise RuntimeProjectionBuildError("runtime projection facet schema pin is stale")
    if gate_b.capabilities.facet_applicability_id != content_id_for_value(gate_a.applicability):
        raise RuntimeProjectionBuildError("runtime projection applicability pin is stale")
    if gate_b.selection.intent_value_normalizer_id != PRICE_INTENT_NORMALIZER_ID:
        raise RuntimeProjectionBuildError("runtime projection price normalizer is unsupported")
    if gate_b.selection.reviewed_value_aliases:
        raise RuntimeProjectionBuildError("numeric price cannot publish value aliases")


def _projected_ordinary_facet_ids(
    capabilities: EffectiveFacetCapabilitySet,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                item.facet_id
                for item in capabilities.entries
                if item.intent_committable or item.probe_eligible
            }
        )
    )


def _validate_normalizer_fixed_points(
    registry: FacetRegistry,
    *,
    category_registry: CategoryRegistry,
) -> None:
    price = registry.get("price")
    if price is not None:
        for value in (0, -1, 1):
            if price.normalizer(value) != value:
                raise RuntimeProjectionBuildError("price normalizer is not a fixed point")
    category = registry.require(SYSTEM_PRODUCT_CATEGORY_FACET_ID)
    for scope in category_registry.scopes:
        if category.normalizer(scope.id) != scope.id:
            raise RuntimeProjectionBuildError("category normalizer is not a fixed point")


def _numeric_operator_values() -> tuple[str, ...]:
    return tuple(sorted(item.value for item in NUMERIC_OPERATORS))


def _categorical_operator_values() -> tuple[str, ...]:
    return tuple(sorted(item.value for item in CATEGORICAL_OPERATORS))
