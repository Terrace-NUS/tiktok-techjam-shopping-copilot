"""Immutable, non-normative DTOs for the CS2 Gate-A source-profile proposal.

These objects describe observed source evidence.  They deliberately do not
publish CatalogFacetDefinition, FacetApplicability, or FacetSourceBinding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from ..canonical import IJSON_SAFE_INTEGER_MAX, canonical_json_bytes, validate_semantic_string

GATE_A_PROFILE_SELECTION_SCHEMA: Literal["shopping-copilot/gate-a-profile-selection/v0"] = (
    "shopping-copilot/gate-a-profile-selection/v0"
)
GATE_A_SOURCE_PROFILE_SCHEMA: Literal["shopping-copilot/gate-a-source-profile/v0"] = (
    "shopping-copilot/gate-a-source-profile/v0"
)
FACET_PROFILE_BUILDER_VERSION = "catalog_semantic_gate_a_profile_v0"

_CONTENT_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_NODE_ID_PATTERN = re.compile(r"^cn_[0-9a-f]{64}$")
_SCOPE_ID_PATTERN = re.compile(r"^cs_[0-9a-f]{64}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_JSON_VALUE_TYPES = frozenset({"array", "boolean", "integer", "null", "number", "object", "string"})


class SourceKind(str, Enum):
    """Structured raw catalog source family from the Gate-A contract."""

    TOP_LEVEL = "top_level"
    DETAILS = "details"


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceLocator:
    """Exact, case-sensitive raw source locator."""

    kind: SourceKind
    key: str

    def __post_init__(self) -> None:
        if type(self.kind) is not SourceKind:
            raise TypeError("SourceLocator.kind must be SourceKind")
        _require_raw_string(self.key, name="SourceLocator.key")


@dataclass(frozen=True, slots=True, kw_only=True)
class GateAProfileSelection:
    """Source-controlled engineering input for exhaustive Gate-A profiling."""

    schema: Literal["shopping-copilot/gate-a-profile-selection/v0"]
    catalog_id: str
    category_registry_id: str
    product_category_assignment_id: str
    builder_version: str
    top_level_keys: tuple[str, ...]
    include_all_details: bool
    sample_seed: str
    sample_limit: int
    top_value_limit: int

    def __post_init__(self) -> None:
        if self.schema != GATE_A_PROFILE_SELECTION_SCHEMA:
            raise ValueError("GateAProfileSelection.schema is invalid")
        _require_content_id(self.catalog_id, name="GateAProfileSelection.catalog_id")
        _require_content_id(
            self.category_registry_id,
            name="GateAProfileSelection.category_registry_id",
        )
        _require_content_id(
            self.product_category_assignment_id,
            name="GateAProfileSelection.product_category_assignment_id",
        )
        if self.builder_version != FACET_PROFILE_BUILDER_VERSION:
            raise ValueError("GateAProfileSelection.builder_version is unsupported")
        _require_sorted_semantic_strings(
            self.top_level_keys,
            name="GateAProfileSelection.top_level_keys",
            nonempty=True,
        )
        if type(self.include_all_details) is not bool:
            raise TypeError("GateAProfileSelection.include_all_details must be bool")
        validate_semantic_string(self.sample_seed, name="GateAProfileSelection.sample_seed")
        _require_bounded_count(
            self.sample_limit, name="GateAProfileSelection.sample_limit", maximum=100
        )
        _require_bounded_count(
            self.top_value_limit,
            name="GateAProfileSelection.top_value_limit",
            maximum=200,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TypeCount:
    """Count of one lossless JSON source-value type."""

    value_type: str
    count: int

    def __post_init__(self) -> None:
        if self.value_type not in _JSON_VALUE_TYPES:
            raise ValueError("TypeCount.value_type is invalid")
        _require_nonnegative(self.count, name="TypeCount.count")


@dataclass(frozen=True, slots=True, kw_only=True)
class ValueMass:
    """Frequency of one exact JCS source value."""

    canonical_value_json: str
    value_type: str
    count: int

    def __post_init__(self) -> None:
        validate_semantic_string(
            self.canonical_value_json,
            name="ValueMass.canonical_value_json",
        )
        if self.value_type not in _JSON_VALUE_TYPES:
            raise ValueError("ValueMass.value_type is invalid")
        _require_positive(self.count, name="ValueMass.count")


@dataclass(frozen=True, slots=True, kw_only=True)
class TopLevelFieldObservation:
    """Observed shape of one top-level field, including non-candidate fields."""

    key: str
    present_count: int
    missing_count: int
    null_count: int
    empty_count: int
    type_counts: tuple[TypeCount, ...]

    def __post_init__(self) -> None:
        _require_raw_string(self.key, name="TopLevelFieldObservation.key")
        for name, value in (
            ("present_count", self.present_count),
            ("missing_count", self.missing_count),
            ("null_count", self.null_count),
            ("empty_count", self.empty_count),
        ):
            _require_nonnegative(value, name=f"TopLevelFieldObservation.{name}")
        _require_sorted_type_counts(self.type_counts, name="TopLevelFieldObservation.type_counts")
        if sum(item.count for item in self.type_counts) != self.present_count:
            raise ValueError("TopLevelFieldObservation type counts differ from present_count")
        if self.null_count > self.present_count or self.empty_count > self.present_count:
            raise ValueError("TopLevelFieldObservation sub-count exceeds present_count")


@dataclass(frozen=True, slots=True, kw_only=True)
class ProfiledScope:
    """Reviewed category scope and its exact assignment-derived denominator."""

    category_scope_id: str
    label: str
    product_count: int
    is_root: bool

    def __post_init__(self) -> None:
        _require_identifier(
            self.category_scope_id,
            pattern=_SCOPE_ID_PATTERN,
            name="ProfiledScope.category_scope_id",
        )
        validate_semantic_string(self.label, name="ProfiledScope.label")
        _require_positive(self.product_count, name="ProfiledScope.product_count")
        if type(self.is_root) is not bool:
            raise TypeError("ProfiledScope.is_root must be bool")


@dataclass(frozen=True, slots=True, kw_only=True)
class ScopeSourceProfile:
    """Observed source value statistics within one exact reviewed scope."""

    source: SourceLocator
    category_scope_id: str
    product_count: int
    present_count: int
    nonempty_count: int
    null_count: int
    empty_count: int
    type_counts: tuple[TypeCount, ...]
    distinct_value_count: int
    distinct_nonempty_value_count: int
    dominant_nonempty_value_count: int
    top_values: tuple[ValueMass, ...]

    def __post_init__(self) -> None:
        if type(self.source) is not SourceLocator:
            raise TypeError("ScopeSourceProfile.source must be SourceLocator")
        _require_identifier(
            self.category_scope_id,
            pattern=_SCOPE_ID_PATTERN,
            name="ScopeSourceProfile.category_scope_id",
        )
        for name, value in (
            ("product_count", self.product_count),
            ("present_count", self.present_count),
            ("nonempty_count", self.nonempty_count),
            ("null_count", self.null_count),
            ("empty_count", self.empty_count),
            ("distinct_value_count", self.distinct_value_count),
            ("distinct_nonempty_value_count", self.distinct_nonempty_value_count),
            ("dominant_nonempty_value_count", self.dominant_nonempty_value_count),
        ):
            _require_nonnegative(value, name=f"ScopeSourceProfile.{name}")
        if self.present_count > self.product_count:
            raise ValueError("ScopeSourceProfile.present_count exceeds product_count")
        if self.nonempty_count + self.null_count + self.empty_count != self.present_count:
            raise ValueError("ScopeSourceProfile value-state counts do not partition present_count")
        _require_sorted_type_counts(self.type_counts, name="ScopeSourceProfile.type_counts")
        if sum(item.count for item in self.type_counts) != self.present_count:
            raise ValueError("ScopeSourceProfile type counts differ from present_count")
        if self.distinct_nonempty_value_count > self.distinct_value_count:
            raise ValueError("ScopeSourceProfile nonempty distinct count is invalid")
        if self.dominant_nonempty_value_count > self.nonempty_count:
            raise ValueError("ScopeSourceProfile dominant count exceeds nonempty_count")
        _require_sorted_value_masses(self.top_values)
        if any(item.count > self.nonempty_count for item in self.top_values):
            raise ValueError("ScopeSourceProfile top-value count exceeds nonempty_count")


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceSample:
    """Stable bounded nonempty example for one exact source locator."""

    sample_hash: str
    source: SourceLocator
    parent_asin: str
    line_number: int
    leaf_node_ids: tuple[str, ...]
    category_scope_ids: tuple[str, ...]
    canonical_value_json: str
    value_type: str

    def __post_init__(self) -> None:
        _require_identifier(
            self.sample_hash, pattern=_SHA256_PATTERN, name="SourceSample.sample_hash"
        )
        if type(self.source) is not SourceLocator:
            raise TypeError("SourceSample.source must be SourceLocator")
        validate_semantic_string(self.parent_asin, name="SourceSample.parent_asin")
        _require_positive(self.line_number, name="SourceSample.line_number")
        _require_sorted_identifiers(
            self.leaf_node_ids,
            pattern=_NODE_ID_PATTERN,
            name="SourceSample.leaf_node_ids",
        )
        _require_sorted_identifiers(
            self.category_scope_ids,
            pattern=_SCOPE_ID_PATTERN,
            name="SourceSample.category_scope_ids",
        )
        validate_semantic_string(
            self.canonical_value_json,
            name="SourceSample.canonical_value_json",
        )
        if self.value_type not in _JSON_VALUE_TYPES:
            raise ValueError("SourceSample.value_type is invalid")


@dataclass(frozen=True, slots=True, kw_only=True)
class PriceStringValue:
    """Exact observed string lane retained for price review."""

    canonical_value_json: str
    count: int

    def __post_init__(self) -> None:
        validate_semantic_string(
            self.canonical_value_json,
            name="PriceStringValue.canonical_value_json",
        )
        _require_positive(self.count, name="PriceStringValue.count")


@dataclass(frozen=True, slots=True, kw_only=True)
class PriceAudit:
    """Lossless first-lane observations; it does not approve a price extractor."""

    present_count: int
    null_count: int
    numeric_count: int
    numeric_exact_cent_count: int
    numeric_non_cent_count: int
    string_count: int
    other_count: int
    minimum_exact_cents: int | None
    maximum_exact_cents: int | None
    string_values: tuple[PriceStringValue, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("present_count", self.present_count),
            ("null_count", self.null_count),
            ("numeric_count", self.numeric_count),
            ("numeric_exact_cent_count", self.numeric_exact_cent_count),
            ("numeric_non_cent_count", self.numeric_non_cent_count),
            ("string_count", self.string_count),
            ("other_count", self.other_count),
        ):
            _require_nonnegative(value, name=f"PriceAudit.{name}")
        if self.numeric_exact_cent_count + self.numeric_non_cent_count != self.numeric_count:
            raise ValueError("PriceAudit numeric counts are inconsistent")
        if (
            self.null_count + self.numeric_count + self.string_count + self.other_count
            != self.present_count
        ):
            raise ValueError("PriceAudit source-type counts are inconsistent")
        if (self.minimum_exact_cents is None) != (self.maximum_exact_cents is None):
            raise ValueError("PriceAudit exact-cent bounds must be both present or absent")
        if self.minimum_exact_cents is not None:
            maximum_exact_cents = self.maximum_exact_cents
            if maximum_exact_cents is None:
                raise ValueError("PriceAudit maximum exact cents is missing")
            _require_nonnegative(self.minimum_exact_cents, name="PriceAudit.minimum_exact_cents")
            _require_nonnegative(maximum_exact_cents, name="PriceAudit.maximum_exact_cents")
            if self.minimum_exact_cents > maximum_exact_cents:
                raise ValueError("PriceAudit exact-cent bounds are reversed")
        if tuple(item.canonical_value_json for item in self.string_values) != tuple(
            sorted(item.canonical_value_json for item in self.string_values)
        ):
            raise ValueError("PriceAudit.string_values must be sorted")
        if sum(item.count for item in self.string_values) != self.string_count:
            raise ValueError("PriceAudit string values differ from string_count")


@dataclass(frozen=True, slots=True, kw_only=True)
class GateASourceProfileBuild:
    """Complete CS2 source-profile proposal before any Gate-A approval."""

    schema: Literal["shopping-copilot/gate-a-source-profile/v0"]
    catalog_id: str
    category_registry_id: str
    product_category_assignment_id: str
    builder_version: str
    selection: GateAProfileSelection
    top_level_fields: tuple[TopLevelFieldObservation, ...]
    scopes: tuple[ProfiledScope, ...]
    sources: tuple[SourceLocator, ...]
    scope_source_profiles: tuple[ScopeSourceProfile, ...]
    samples: tuple[SourceSample, ...]
    price_audit: PriceAudit

    def __post_init__(self) -> None:
        if self.schema != GATE_A_SOURCE_PROFILE_SCHEMA:
            raise ValueError("GateASourceProfileBuild.schema is invalid")
        _require_content_id(self.catalog_id, name="GateASourceProfileBuild.catalog_id")
        _require_content_id(
            self.category_registry_id,
            name="GateASourceProfileBuild.category_registry_id",
        )
        _require_content_id(
            self.product_category_assignment_id,
            name="GateASourceProfileBuild.product_category_assignment_id",
        )
        if self.builder_version != FACET_PROFILE_BUILDER_VERSION:
            raise ValueError("GateASourceProfileBuild.builder_version is unsupported")
        if type(self.selection) is not GateAProfileSelection:
            raise TypeError("GateASourceProfileBuild.selection is invalid")
        if type(self.price_audit) is not PriceAudit:
            raise TypeError("GateASourceProfileBuild.price_audit is invalid")
        _require_exact_tuple(
            self.top_level_fields, TopLevelFieldObservation, name="top_level_fields"
        )
        _require_exact_tuple(self.scopes, ProfiledScope, name="scopes")
        _require_exact_tuple(self.sources, SourceLocator, name="sources")
        _require_exact_tuple(
            self.scope_source_profiles,
            ScopeSourceProfile,
            name="scope_source_profiles",
        )
        _require_exact_tuple(self.samples, SourceSample, name="samples")
        if (
            self.selection.catalog_id != self.catalog_id
            or self.selection.category_registry_id != self.category_registry_id
            or self.selection.product_category_assignment_id != self.product_category_assignment_id
            or self.selection.builder_version != self.builder_version
        ):
            raise ValueError("GateASourceProfileBuild differs from its pinned selection")
        top_field_keys = tuple(item.key for item in self.top_level_fields)
        if top_field_keys != tuple(sorted(set(top_field_keys), key=canonical_json_bytes)):
            raise ValueError("GateASourceProfileBuild.top_level_fields must be sorted and unique")
        scope_ids = tuple(item.category_scope_id for item in self.scopes)
        if scope_ids != tuple(sorted(set(scope_ids))):
            raise ValueError("GateASourceProfileBuild.scopes must be sorted and unique")
        if sum(item.is_root for item in self.scopes) != 1:
            raise ValueError("GateASourceProfileBuild must contain exactly one root scope")
        source_keys = tuple(source_sort_key(item) for item in self.sources)
        if source_keys != tuple(sorted(set(source_keys))):
            raise ValueError("GateASourceProfileBuild.sources must be sorted and unique")
        expected_profile_keys = tuple(
            (source_sort_key(source), scope_id) for source in self.sources for scope_id in scope_ids
        )
        profile_keys = tuple(
            (source_sort_key(item.source), item.category_scope_id)
            for item in self.scope_source_profiles
        )
        if profile_keys != expected_profile_keys:
            raise ValueError(
                "GateASourceProfileBuild.scope_source_profiles must be the exact source/scope matrix"
            )
        product_count_by_scope = {
            item.category_scope_id: item.product_count for item in self.scopes
        }
        if any(
            item.product_count != product_count_by_scope[item.category_scope_id]
            for item in self.scope_source_profiles
        ):
            raise ValueError("GateASourceProfileBuild scope denominators are inconsistent")
        known_sources = set(self.sources)
        known_scope_ids = set(scope_ids)
        sample_keys = tuple(
            (
                source_sort_key(item.source),
                item.sample_hash,
                item.parent_asin,
                item.line_number,
                item.canonical_value_json,
            )
            for item in self.samples
        )
        if sample_keys != tuple(sorted(sample_keys)):
            raise ValueError("GateASourceProfileBuild.samples must be deterministically sorted")
        if len({item.sample_hash for item in self.samples}) != len(self.samples):
            raise ValueError("GateASourceProfileBuild sample hashes must be unique")
        if any(item.source not in known_sources for item in self.samples):
            raise ValueError("GateASourceProfileBuild sample references an unknown source")
        if any(not set(item.category_scope_ids) <= known_scope_ids for item in self.samples):
            raise ValueError("GateASourceProfileBuild sample references an unknown scope")
        if SourceLocator(kind=SourceKind.TOP_LEVEL, key="price") not in known_sources:
            raise ValueError("GateASourceProfileBuild must contain the first price lane")


def source_sort_key(source: SourceLocator) -> tuple[str, bytes]:
    """Return the deterministic proposal order for exact source locators."""

    return source.kind.value, canonical_json_bytes(source.key)


def _require_content_id(value: object, *, name: str) -> None:
    _require_identifier(value, pattern=_CONTENT_ID_PATTERN, name=name)


def _require_raw_string(value: object, *, name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError(f"{name} must not contain a lone surrogate")


def _require_identifier(value: object, *, pattern: re.Pattern[str], name: str) -> None:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise ValueError(f"{name} is invalid")


def _require_sorted_identifiers(
    values: tuple[str, ...],
    *,
    pattern: re.Pattern[str],
    name: str,
) -> None:
    if type(values) is not tuple or not values:
        raise ValueError(f"{name} must be a non-empty tuple")
    for value in values:
        _require_identifier(value, pattern=pattern, name=name)
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be sorted and unique")


def _require_sorted_semantic_strings(
    values: tuple[str, ...],
    *,
    name: str,
    nonempty: bool,
) -> None:
    if type(values) is not tuple or (nonempty and not values):
        raise ValueError(f"{name} must be a {'non-empty ' if nonempty else ''}tuple")
    for value in values:
        validate_semantic_string(value, name=name)
    expected = tuple(sorted(set(values), key=canonical_json_bytes))
    if values != expected:
        raise ValueError(f"{name} must be canonical, sorted, and unique")


def _require_sorted_type_counts(values: tuple[TypeCount, ...], *, name: str) -> None:
    _require_exact_tuple(values, TypeCount, name=name)
    types = tuple(item.value_type for item in values)
    if types != tuple(sorted(set(types))):
        raise ValueError(f"{name} must be sorted and unique")


def _require_sorted_value_masses(values: tuple[ValueMass, ...]) -> None:
    _require_exact_tuple(values, ValueMass, name="ScopeSourceProfile.top_values")
    keys = tuple((-item.count, item.value_type, item.canonical_value_json) for item in values)
    if keys != tuple(sorted(keys)):
        raise ValueError("ScopeSourceProfile.top_values must be deterministically sorted")
    identities = tuple((item.value_type, item.canonical_value_json) for item in values)
    if len(set(identities)) != len(identities):
        raise ValueError("ScopeSourceProfile.top_values must be unique")


def _require_exact_tuple(values: object, expected_type: type[object], *, name: str) -> None:
    if type(values) is not tuple:
        raise TypeError(f"GateASourceProfileBuild.{name} must be a tuple")
    if any(type(item) is not expected_type for item in values):
        raise TypeError(f"GateASourceProfileBuild.{name} contains an invalid item")


def _require_nonnegative(value: object, *, name: str) -> None:
    if type(value) is not int or not 0 <= value <= IJSON_SAFE_INTEGER_MAX:
        raise ValueError(f"{name} must be a non-negative I-JSON integer")


def _require_positive(value: object, *, name: str) -> None:
    if type(value) is not int or not 0 < value <= IJSON_SAFE_INTEGER_MAX:
        raise ValueError(f"{name} must be a positive I-JSON integer")


def _require_bounded_count(value: object, *, name: str, maximum: int) -> None:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be an integer between 0 and {maximum}")
