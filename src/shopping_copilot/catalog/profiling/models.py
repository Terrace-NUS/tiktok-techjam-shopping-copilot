"""Immutable data-transfer objects for raw catalog profiling.

The DTOs in this module deliberately describe observed source data.  They do
not publish category scopes, facet schemas, or runtime matching behavior.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class ProfileConfig:
    """Deterministic limits and seed for one profiling run."""

    seed: str = "raw-profile-v1"
    sample_limit: int = 20
    top_value_limit: int = 50

    def __post_init__(self) -> None:
        if type(self.seed) is not str or not self.seed:
            raise ValueError("seed must be a non-empty string")
        if type(self.sample_limit) is not int or self.sample_limit < 0:
            raise ValueError("sample_limit must be a non-negative integer")
        if type(self.top_value_limit) is not int or self.top_value_limit < 0:
            raise ValueError("top_value_limit must be a non-negative integer")


@dataclass(frozen=True, slots=True, kw_only=True)
class TypeCount:
    """Count for one lossless JSON value-kind label."""

    value_type: str
    count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class TopLevelFieldProfile:
    """Presence and raw JSON-shape statistics for a top-level field."""

    field: str
    present_count: int
    missing_count: int
    null_count: int
    empty_count: int
    type_counts: tuple[TypeCount, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class CategoryNode:
    """One raw full-path prefix observed in the catalog."""

    category_id: str
    path: tuple[str, ...]
    parent_id: str | None
    direct_support: int
    subtree_support: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ProductCategoryAssignment:
    """Raw, per-line category assignment emitted while the catalog streams."""

    line_number: int
    parent_asin: str | None
    raw_categories_json: str | None
    raw_path: tuple[str, ...]
    category_node_ids: tuple[str, ...]
    leaf_category_id: str | None
    category_valid: bool
    diagnostics: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class ValueMass:
    """Frequency of one exact canonical-JSON representation."""

    canonical_value_json: str
    value_type: str
    count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ValueSample:
    """Stable, bounded example for one raw details key."""

    sample_hash: str
    parent_asin: str | None
    line_number: int
    canonical_value_json: str
    value_type: str


@dataclass(frozen=True, slots=True, kw_only=True)
class DetailKeyProfile:
    """Raw row support and value-shape statistics for one exact details key.

    ``null_count``, ``empty_count``, and ``nonempty_count`` are disjoint and
    sum to ``support_count``. Empty means a blank string or an empty array or
    object; null is reported separately.
    """

    raw_key: str
    support_count: int
    nonempty_count: int
    null_count: int
    empty_count: int
    type_counts: tuple[TypeCount, ...]
    distinct_value_count: int
    distinct_nonempty_value_count: int
    top_values: tuple[ValueMass, ...]
    samples: tuple[ValueSample, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class CategoryDetailCoverage:
    """Raw row coverage within one category prefix subtree.

    ``product_count`` is retained as an output field for readability, but its
    exact denominator is valid category-assignment rows. Duplicate product IDs
    remain counted and are separately diagnosed; semantic publication must
    fail or deduplicate under a later reviewed policy.
    """

    category_id: str
    raw_key: str
    product_count: int
    present_count: int
    nonempty_count: int
    presence_coverage: float
    nonempty_coverage: float


@dataclass(frozen=True, slots=True, kw_only=True)
class DiagnosticSample:
    """Stable example attached to one diagnostic code."""

    sample_hash: str
    parent_asin: str | None
    line_number: int
    raw_value_json: str


@dataclass(frozen=True, slots=True, kw_only=True)
class DiagnosticProfile:
    """Row-level count and examples for one profiler diagnostic."""

    code: str
    count: int
    samples: tuple[DiagnosticSample, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class CatalogProfile:
    """Complete immutable result of a raw catalog profiling pass."""

    schema_version: str
    catalog_sha256: str
    file_size_bytes: int
    physical_line_count: int
    product_row_count: int
    invalid_record_count: int
    category_assignment_count: int
    valid_category_assignment_count: int
    product_row_with_diagnostics_count: int
    unique_parent_asin_count: int
    seed: str
    sample_limit: int
    top_value_limit: int
    top_level_fields: tuple[TopLevelFieldProfile, ...]
    category_nodes: tuple[CategoryNode, ...]
    detail_keys: tuple[DetailKeyProfile, ...]
    category_detail_coverage: tuple[CategoryDetailCoverage, ...]
    diagnostics: tuple[DiagnosticProfile, ...]
