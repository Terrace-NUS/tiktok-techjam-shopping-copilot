"""Deterministic, category-conditioned source profiling for CS2 Gate A."""

from __future__ import annotations

import bisect
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import cast

from ..canonical import canonical_json_bytes, canonical_json_text, sha256_hex
from ..category import (
    CategoryRegistry,
    ProductCategoryAssignmentSet,
    ProductCategoryAssignmentStatus,
    category_node_id,
    normalize_category_path,
)
from ..errors import CatalogChangedError, FacetProfileBuildError, FacetProfileSelectionError
from .models import (
    FACET_PROFILE_BUILDER_VERSION,
    GATE_A_SOURCE_PROFILE_SCHEMA,
    GateAProfileSelection,
    GateASourceProfileBuild,
    PriceAudit,
    PriceStringValue,
    ProfiledScope,
    ScopeSourceProfile,
    SourceKind,
    SourceLocator,
    SourceSample,
    TopLevelFieldObservation,
    TypeCount,
    ValueMass,
    source_sort_key,
)

_OFFICIAL_PRODUCT_COUNT = 50_000


class _DuplicateJsonKeyError(ValueError):
    pass


@dataclass(slots=True)
class _ObservationAccumulator:
    present_count: int = 0
    nonempty_count: int = 0
    null_count: int = 0
    empty_count: int = 0
    type_counts: Counter[str] = field(default_factory=Counter)
    value_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    nonempty_value_counts: Counter[tuple[str, str]] = field(default_factory=Counter)

    def observe(self, value: object) -> tuple[str, str, bool]:
        value_type = _json_value_type(value)
        canonical_value_json = canonical_json_text(value)
        self.present_count += 1
        self.type_counts[value_type] += 1
        self.value_counts[(value_type, canonical_value_json)] += 1
        if value is None:
            self.null_count += 1
            return value_type, canonical_value_json, False
        if _is_empty(value):
            self.empty_count += 1
            return value_type, canonical_value_json, False
        self.nonempty_count += 1
        self.nonempty_value_counts[(value_type, canonical_value_json)] += 1
        return value_type, canonical_value_json, True


@dataclass(slots=True)
class _TopFieldAccumulator:
    present_count: int = 0
    null_count: int = 0
    empty_count: int = 0
    type_counts: Counter[str] = field(default_factory=Counter)

    def observe(self, value: object) -> None:
        self.present_count += 1
        self.type_counts[_json_value_type(value)] += 1
        if value is None:
            self.null_count += 1
        elif _is_empty(value):
            self.empty_count += 1


class _BoundedSourceSampler:
    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._entries: list[tuple[tuple[str, str, int, str], SourceSample]] = []

    def consider(self, sample: SourceSample) -> None:
        if self._limit == 0:
            return
        rank = (
            sample.sample_hash,
            sample.parent_asin,
            sample.line_number,
            sample.canonical_value_json,
        )
        entry = (rank, sample)
        if len(self._entries) < self._limit:
            bisect.insort(self._entries, entry, key=lambda item: item[0])
            return
        if rank >= self._entries[-1][0]:
            return
        bisect.insort(self._entries, entry, key=lambda item: item[0])
        self._entries.pop()

    def values(self) -> tuple[SourceSample, ...]:
        return tuple(sample for _, sample in self._entries)


@dataclass(slots=True)
class _PriceAccumulator:
    present_count: int = 0
    null_count: int = 0
    numeric_count: int = 0
    numeric_exact_cent_count: int = 0
    numeric_non_cent_count: int = 0
    string_count: int = 0
    other_count: int = 0
    minimum_exact_cents: int | None = None
    maximum_exact_cents: int | None = None
    string_values: Counter[str] = field(default_factory=Counter)

    def observe(self, value: object) -> None:
        self.present_count += 1
        if value is None:
            self.null_count += 1
            return
        if type(value) is str:
            self.string_count += 1
            self.string_values[canonical_json_text(value)] += 1
            return
        if type(value) is int:
            number = Decimal(value)
        elif type(value) is Decimal:
            number = value
        else:
            self.other_count += 1
            return
        self.numeric_count += 1
        cents = number * 100
        if number < 0 or cents != cents.to_integral_value():
            self.numeric_non_cent_count += 1
            return
        exact_cents = int(cents)
        self.numeric_exact_cent_count += 1
        self.minimum_exact_cents = (
            exact_cents
            if self.minimum_exact_cents is None
            else min(self.minimum_exact_cents, exact_cents)
        )
        self.maximum_exact_cents = (
            exact_cents
            if self.maximum_exact_cents is None
            else max(self.maximum_exact_cents, exact_cents)
        )


def build_gate_a_source_profile(
    catalog_path: str | Path,
    *,
    registry: CategoryRegistry,
    assignments: ProductCategoryAssignmentSet,
    category_registry_id: str,
    product_category_assignment_id: str,
    selection: GateAProfileSelection,
    expected_product_count: int = _OFFICIAL_PRODUCT_COUNT,
    enforce_official_gate: bool = True,
) -> GateASourceProfileBuild:
    """Profile exact raw sources inside every reviewed CategoryScope.

    This is an observation build.  It does not create or approve facet IDs,
    applicability, bindings, extractors, normalizers, or runtime capability.
    """

    _validate_build_inputs(
        registry=registry,
        assignments=assignments,
        category_registry_id=category_registry_id,
        product_category_assignment_id=product_category_assignment_id,
        selection=selection,
        expected_product_count=expected_product_count,
        enforce_official_gate=enforce_official_gate,
    )
    source = Path(catalog_path)
    assignment_by_asin = {item.parent_asin: item for item in assignments.assignments}
    active_scope_ids_by_leaf = _active_scope_ids_by_leaf(registry)
    active_scope_ids_by_asin: dict[str, tuple[str, ...]] = {}
    scope_product_counts: Counter[str] = Counter()
    for assignment in assignments.assignments:
        active_scope_ids = tuple(
            sorted(
                {
                    scope_id
                    for leaf_id in assignment.leaf_node_ids
                    for scope_id in active_scope_ids_by_leaf.get(leaf_id, ())
                }
            )
        )
        if registry.root_scope_id not in active_scope_ids:
            raise FacetProfileBuildError("category root scope does not cover an assignment")
        active_scope_ids_by_asin[assignment.parent_asin] = active_scope_ids
        scope_product_counts.update(active_scope_ids)

    top_field_accumulators: dict[str, _TopFieldAccumulator] = {}
    scope_accumulators: dict[tuple[SourceLocator, str], _ObservationAccumulator] = {}
    samplers: dict[SourceLocator, _BoundedSourceSampler] = {}
    observed_sources: set[SourceLocator] = set()
    configured_top_level_keys = set(selection.top_level_keys)
    price_accumulator = _PriceAccumulator()
    observed_asins: set[str] = set()
    digest = hashlib.sha256()
    product_count = 0

    with source.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            digest.update(raw_line)
            row, decimal_row = _parse_profile_line(raw_line, line_number=line_number)
            product_count += 1
            parent_asin = _require_parent_asin(row, line_number=line_number)
            if parent_asin in observed_asins:
                raise FacetProfileBuildError(f"duplicate parent_asin at line {line_number}")
            observed_asins.add(parent_asin)
            matched_assignment = assignment_by_asin.get(parent_asin)
            if matched_assignment is None:
                raise FacetProfileBuildError("raw product is absent from category assignments")
            raw_path = _require_raw_path(row, line_number=line_number)
            expected_leaf = category_node_id(normalize_category_path(raw_path))
            if matched_assignment.leaf_node_ids != (expected_leaf,):
                raise FacetProfileBuildError("category assignment differs from exact raw path")
            active_scope_ids = active_scope_ids_by_asin[parent_asin]

            for key, value in row.items():
                top_field_accumulators.setdefault(key, _TopFieldAccumulator()).observe(value)

            details = row.get("details")
            if type(details) is not dict:
                raise FacetProfileBuildError(f"details must be an object at line {line_number}")
            detail_values = cast(dict[str, object], details)
            source_values: list[tuple[SourceLocator, object]] = []
            for key in selection.top_level_keys:
                if key in row:
                    source_values.append(
                        (SourceLocator(kind=SourceKind.TOP_LEVEL, key=key), row[key])
                    )
            if selection.include_all_details:
                source_values.extend(
                    (SourceLocator(kind=SourceKind.DETAILS, key=key), value)
                    for key, value in detail_values.items()
                )

            for locator, value in source_values:
                observed_sources.add(locator)
                value_type = ""
                canonical_value_json = ""
                is_nonempty = False
                for scope_id in active_scope_ids:
                    accumulator = scope_accumulators.setdefault(
                        (locator, scope_id),
                        _ObservationAccumulator(),
                    )
                    observed_type, observed_json, observed_nonempty = accumulator.observe(value)
                    if not value_type:
                        value_type = observed_type
                        canonical_value_json = observed_json
                        is_nonempty = observed_nonempty
                if is_nonempty:
                    sample = _source_sample(
                        selection=selection,
                        locator=locator,
                        parent_asin=parent_asin,
                        line_number=line_number,
                        leaf_node_ids=matched_assignment.leaf_node_ids,
                        category_scope_ids=active_scope_ids,
                        canonical_value_json=canonical_value_json,
                        value_type=value_type,
                    )
                    samplers.setdefault(
                        locator,
                        _BoundedSourceSampler(selection.sample_limit),
                    ).consider(sample)

            decimal_price = decimal_row.get("price")
            if "price" in decimal_row:
                price_accumulator.observe(decimal_price)

    actual_catalog_id = f"sha256:{digest.hexdigest()}"
    if actual_catalog_id != selection.catalog_id:
        raise CatalogChangedError("catalog bytes differ from the Gate-A profile selection")
    if product_count != expected_product_count:
        raise FacetProfileBuildError(
            f"Gate-A source profile expected {expected_product_count} products, got {product_count}"
        )
    if observed_asins != set(assignment_by_asin):
        raise FacetProfileBuildError("raw catalog and category assignment product sets differ")
    if len(observed_asins) != len(assignments.assignments):
        raise FacetProfileBuildError("raw product count differs from category assignments")
    missing_top_level_keys = configured_top_level_keys - set(top_field_accumulators)
    if missing_top_level_keys:
        raise FacetProfileSelectionError(
            f"configured top-level source is not observed: {min(missing_top_level_keys)}"
        )

    scopes = tuple(
        ProfiledScope(
            category_scope_id=scope.id,
            label=scope.label,
            product_count=scope_product_counts[scope.id],
            is_root=scope.id == registry.root_scope_id,
        )
        for scope in registry.scopes
    )
    if sum(item.is_root for item in scopes) != 1:
        raise FacetProfileBuildError("profile must contain exactly one root scope")
    if next(item for item in scopes if item.is_root).product_count != product_count:
        raise FacetProfileBuildError("root scope denominator differs from raw product count")

    sources = tuple(sorted(observed_sources, key=source_sort_key))
    scope_source_profiles = tuple(
        _scope_source_profile(
            source=locator,
            category_scope_id=scope.category_scope_id,
            product_count=scope.product_count,
            accumulator=scope_accumulators.get(
                (locator, scope.category_scope_id),
                _ObservationAccumulator(),
            ),
            top_value_limit=selection.top_value_limit,
        )
        for locator in sources
        for scope in scopes
    )
    samples = tuple(
        sample
        for locator in sources
        for sample in samplers.get(locator, _BoundedSourceSampler(0)).values()
    )
    top_level_fields = tuple(
        TopLevelFieldObservation(
            key=key,
            present_count=accumulator.present_count,
            missing_count=product_count - accumulator.present_count,
            null_count=accumulator.null_count,
            empty_count=accumulator.empty_count,
            type_counts=_type_counts(accumulator.type_counts),
        )
        for key, accumulator in sorted(
            top_field_accumulators.items(),
            key=lambda item: canonical_json_bytes(item[0]),
        )
    )
    price_audit = PriceAudit(
        present_count=price_accumulator.present_count,
        null_count=price_accumulator.null_count,
        numeric_count=price_accumulator.numeric_count,
        numeric_exact_cent_count=price_accumulator.numeric_exact_cent_count,
        numeric_non_cent_count=price_accumulator.numeric_non_cent_count,
        string_count=price_accumulator.string_count,
        other_count=price_accumulator.other_count,
        minimum_exact_cents=price_accumulator.minimum_exact_cents,
        maximum_exact_cents=price_accumulator.maximum_exact_cents,
        string_values=tuple(
            PriceStringValue(canonical_value_json=value, count=count)
            for value, count in sorted(price_accumulator.string_values.items())
        ),
    )
    return GateASourceProfileBuild(
        schema=GATE_A_SOURCE_PROFILE_SCHEMA,
        catalog_id=selection.catalog_id,
        category_registry_id=category_registry_id,
        product_category_assignment_id=product_category_assignment_id,
        builder_version=FACET_PROFILE_BUILDER_VERSION,
        selection=selection,
        top_level_fields=top_level_fields,
        scopes=scopes,
        sources=sources,
        scope_source_profiles=scope_source_profiles,
        samples=samples,
        price_audit=price_audit,
    )


def _validate_build_inputs(
    *,
    registry: CategoryRegistry,
    assignments: ProductCategoryAssignmentSet,
    category_registry_id: str,
    product_category_assignment_id: str,
    selection: GateAProfileSelection,
    expected_product_count: int,
    enforce_official_gate: bool,
) -> None:
    if type(registry) is not CategoryRegistry:
        raise TypeError("registry must be CategoryRegistry")
    if type(assignments) is not ProductCategoryAssignmentSet:
        raise TypeError("assignments must be ProductCategoryAssignmentSet")
    if type(selection) is not GateAProfileSelection:
        raise TypeError("selection must be GateAProfileSelection")
    if type(expected_product_count) is not int or expected_product_count <= 0:
        raise ValueError("expected_product_count must be positive")
    if selection.builder_version != FACET_PROFILE_BUILDER_VERSION:
        raise FacetProfileSelectionError("profiling selection builder version is unsupported")
    if (
        selection.catalog_id != registry.catalog_id
        or selection.catalog_id != assignments.catalog_id
    ):
        raise FacetProfileSelectionError("profiling selection catalog pin is stale")
    if selection.category_registry_id != category_registry_id:
        raise FacetProfileSelectionError("profiling selection CategoryRegistry pin is stale")
    if selection.product_category_assignment_id != product_category_assignment_id:
        raise FacetProfileSelectionError("profiling selection assignment pin is stale")
    if registry.category_graph_id != assignments.category_graph_id:
        raise FacetProfileBuildError("category registry and assignments graph IDs differ")
    if len(assignments.assignments) != expected_product_count:
        raise FacetProfileBuildError("category assignment count differs from profiling gate")
    if any(
        item.status is not ProductCategoryAssignmentStatus.KNOWN for item in assignments.assignments
    ):
        raise FacetProfileBuildError("Gate-A profiling requires KNOWN category assignments")
    if "price" not in selection.top_level_keys:
        raise FacetProfileSelectionError(
            "Gate-A first-lane profile must include exact top-level price"
        )
    if enforce_official_gate:
        if expected_product_count != _OFFICIAL_PRODUCT_COUNT:
            raise ValueError("official Gate-A profile requires exactly 50,000 products")
        if not selection.include_all_details:
            raise FacetProfileSelectionError(
                "official Gate-A profile must include every observed details key"
            )
        if not {"price", "store"}.issubset(selection.top_level_keys):
            raise FacetProfileSelectionError(
                "official Gate-A first review lanes require top-level price and store"
            )


def _active_scope_ids_by_leaf(registry: CategoryRegistry) -> dict[str, tuple[str, ...]]:
    result: dict[str, list[str]] = defaultdict(list)
    for scope in registry.scopes:
        for node_id in scope.member_node_ids:
            result[node_id].append(scope.id)
    return {node_id: tuple(sorted(scope_ids)) for node_id, scope_ids in result.items()}


def _scope_source_profile(
    *,
    source: SourceLocator,
    category_scope_id: str,
    product_count: int,
    accumulator: _ObservationAccumulator,
    top_value_limit: int,
) -> ScopeSourceProfile:
    ranked_values = sorted(
        accumulator.nonempty_value_counts.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )
    return ScopeSourceProfile(
        source=source,
        category_scope_id=category_scope_id,
        product_count=product_count,
        present_count=accumulator.present_count,
        nonempty_count=accumulator.nonempty_count,
        null_count=accumulator.null_count,
        empty_count=accumulator.empty_count,
        type_counts=_type_counts(accumulator.type_counts),
        distinct_value_count=len(accumulator.value_counts),
        distinct_nonempty_value_count=len(accumulator.nonempty_value_counts),
        dominant_nonempty_value_count=ranked_values[0][1] if ranked_values else 0,
        top_values=tuple(
            ValueMass(
                value_type=value_type,
                canonical_value_json=canonical_value_json,
                count=count,
            )
            for (value_type, canonical_value_json), count in ranked_values[:top_value_limit]
        ),
    )


def _source_sample(
    *,
    selection: GateAProfileSelection,
    locator: SourceLocator,
    parent_asin: str,
    line_number: int,
    leaf_node_ids: tuple[str, ...],
    category_scope_ids: tuple[str, ...],
    canonical_value_json: str,
    value_type: str,
) -> SourceSample:
    sample_hash = sha256_hex(
        canonical_json_bytes(
            {
                "seed": selection.sample_seed,
                "catalog_id": selection.catalog_id,
                "source": locator,
                "parent_asin": parent_asin,
                "line_number": line_number,
                "canonical_value_json": canonical_value_json,
            }
        )
    )
    return SourceSample(
        sample_hash=sample_hash,
        source=locator,
        parent_asin=parent_asin,
        line_number=line_number,
        leaf_node_ids=leaf_node_ids,
        category_scope_ids=category_scope_ids,
        canonical_value_json=canonical_value_json,
        value_type=value_type,
    )


def _parse_profile_line(
    raw_line: bytes,
    *,
    line_number: int,
) -> tuple[dict[str, object], dict[str, object]]:
    if not raw_line.strip():
        raise FacetProfileBuildError(f"blank catalog line at physical line {line_number}")
    try:
        text = raw_line.decode("utf-8")
    except UnicodeDecodeError as error:
        raise FacetProfileBuildError(f"invalid UTF-8 at physical line {line_number}") from error
    try:
        parsed: object = json.loads(
            text,
            parse_constant=_reject_nonfinite_token,
            object_pairs_hook=_object_without_duplicate_keys,
        )
        decimal_parsed: object = json.loads(
            text,
            parse_float=Decimal,
            parse_constant=_reject_nonfinite_token,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (_DuplicateJsonKeyError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise FacetProfileBuildError(
            f"invalid strict JSON at physical line {line_number}"
        ) from error
    if type(parsed) is not dict or type(decimal_parsed) is not dict:
        raise FacetProfileBuildError(f"catalog row is not an object at line {line_number}")
    return cast(dict[str, object], parsed), cast(dict[str, object], decimal_parsed)


def _require_parent_asin(row: dict[str, object], *, line_number: int) -> str:
    value = row.get("parent_asin")
    if type(value) is not str or not value or value != value.strip():
        raise FacetProfileBuildError(f"invalid parent_asin at physical line {line_number}")
    return value


def _require_raw_path(row: dict[str, object], *, line_number: int) -> tuple[str, ...]:
    value = row.get("categories")
    if type(value) is not list or not value:
        raise FacetProfileBuildError(f"invalid categories at physical line {line_number}")
    result: list[str] = []
    for item in cast(list[object], value):
        if type(item) is not str or not item:
            raise FacetProfileBuildError(f"invalid category segment at physical line {line_number}")
        result.append(item)
    return tuple(result)


def _type_counts(values: Counter[str]) -> tuple[TypeCount, ...]:
    return tuple(
        TypeCount(value_type=value_type, count=count)
        for value_type, count in sorted(values.items())
    )


def _json_value_type(value: object) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        return "number"
    if type(value) is str:
        return "string"
    if type(value) is list:
        return "array"
    if type(value) is dict:
        return "object"
    raise FacetProfileBuildError(f"unsupported raw JSON value type: {type(value).__name__}")


def _is_empty(value: object) -> bool:
    if type(value) is str:
        return not value.strip()
    if type(value) in (list, dict):
        return not value
    return False


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_nonfinite_token(raw: str) -> object:
    raise ValueError(f"non-finite number token: {raw}")
