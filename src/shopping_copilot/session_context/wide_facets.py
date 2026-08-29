"""Static structured vocabulary backed by the retrieval evidence layer."""

from __future__ import annotations

from .registry import (
    CATEGORICAL_OPERATORS,
    FacetAuthority,
    FacetKind,
    FacetRegistry,
    FacetSpec,
    canonical_text,
)

RETRIEVAL_DERIVED_FACET_IDS = (
    "brand",
    "color",
    "department",
    "feature",
    "gender",
    "material",
    "size",
    "style",
    "use_case",
)


def retrieval_derived_facet_specs() -> tuple[FacetSpec, ...]:
    """Return the reviewed competition vocabulary in deterministic ID order."""

    return tuple(
        FacetSpec(
            id=facet_id,
            kind=FacetKind.CATEGORICAL,
            operators=CATEGORICAL_OPERATORS,
            normalizer=canonical_text,
            authority=FacetAuthority.RETRIEVAL_DERIVED,
        )
        for facet_id in RETRIEVAL_DERIVED_FACET_IDS
    )


def with_retrieval_derived_facets(registry: FacetRegistry) -> FacetRegistry:
    """Layer retrieval-derived facets over a catalog-verified registry.

    A future catalog release may verify one of these facet IDs itself. In that
    case its release-bound definition wins instead of being shadowed here.
    """

    if type(registry) is not FacetRegistry:
        raise TypeError("wide facet composition requires an exact FacetRegistry")
    existing_ids = {spec.id for spec in registry}
    additions = tuple(
        spec for spec in retrieval_derived_facet_specs() if spec.id not in existing_ids
    )
    return FacetRegistry(specs=registry.specs + additions)
