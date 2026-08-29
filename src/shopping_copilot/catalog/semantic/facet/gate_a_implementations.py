"""Closed Gate-A extractor, normalizer, and resolver implementations."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from decimal import Decimal, InvalidOperation
from types import MappingProxyType

from ..canonical import IJSON_SAFE_INTEGER_MAX, canonical_json_bytes
from .gate_a_models import (
    CategoricalValue,
    EvidenceStatus,
    NumericValue,
    PriceNormalizationLane,
    PriceNormalizationResult,
    PriorityExactResolution,
    ProductFacetStatus,
    ResolvedFacetValue,
    SourceExtraction,
    ValueCompleteness,
)

PRICE_EXTRACTOR_ID = "top_level_price_usd_v1"
PRICE_NORMALIZER_ID = "usd_cent_interval_v1"
PRIORITY_EXACT_RESOLVER_ID = "priority_exact_v1"
USD_CENT_UNIT = "USD_CENT"

_FROM_DECIMAL_PATTERN = re.compile(r"^from (0|[1-9][0-9]*)(?:\.([0-9]{1,2}))?$")

SourceExtractor = Callable[[Mapping[str, object]], SourceExtraction]
CatalogValueNormalizer = Callable[[object], PriceNormalizationResult]
FacetResolver = Callable[[tuple[ResolvedFacetValue, ...]], PriorityExactResolution]


def extract_top_level_price_usd_v1(row: Mapping[str, object]) -> SourceExtraction:
    """Copy the exact top-level price value without interpreting another field."""

    if not isinstance(row, Mapping):
        raise TypeError("price extractor requires a catalog object")
    if "price" not in row:
        return SourceExtraction(present=False, raw_value=None)
    return SourceExtraction(present=True, raw_value=row["price"])


def normalize_usd_cent_interval_v1(raw_value: object) -> PriceNormalizationResult:
    """Normalize the approved numeric and exact `from <decimal>` price lanes."""

    if raw_value is None:
        return PriceNormalizationResult(
            status=EvidenceStatus.EMPTY,
            lane=PriceNormalizationLane.EMPTY,
            value=None,
        )

    if type(raw_value) is int:
        return _exact_decimal_price(Decimal(raw_value))

    if type(raw_value) is Decimal:
        return _exact_decimal_price(raw_value)

    if type(raw_value) is str:
        match = _FROM_DECIMAL_PATTERN.fullmatch(raw_value)
        if match is None:
            return _invalid_price()
        try:
            dollars = Decimal(raw_value.removeprefix("from "))
        except InvalidOperation:
            return _invalid_price()
        cents = _exact_cents(dollars)
        if cents is None:
            return _invalid_price()
        return PriceNormalizationResult(
            status=EvidenceStatus.VALID,
            lane=PriceNormalizationLane.LOWER_BOUND,
            value=NumericValue(
                kind="numeric",
                lower=cents,
                lower_inclusive=True,
                upper=None,
                upper_inclusive=False,
                unit=USD_CENT_UNIT,
            ),
        )

    # Floats are deliberately invalid: the approved path must receive the
    # original JSON number token through Decimal, never a binary round trip.
    return _invalid_price()


def resolve_priority_exact_v1(
    values: tuple[ResolvedFacetValue, ...],
) -> PriorityExactResolution:
    """Resolve one already-selected priority layer by exact canonical agreement."""

    if type(values) is not tuple or not values:
        raise ValueError("priority_exact_v1 requires a non-empty value tuple")
    allowed_types = (CategoricalValue, NumericValue)
    first_type = type(values[0])
    if first_type not in allowed_types and values[0].kind not in ("boolean", "text"):
        raise TypeError("priority_exact_v1 received an unsupported value variant")
    if any(type(value) is not first_type for value in values):
        raise TypeError("priority_exact_v1 cannot resolve mixed value variants")

    if first_type is CategoricalValue:
        categorical = tuple(value for value in values if type(value) is CategoricalValue)
        atomic_payloads = {canonical_json_bytes(value.values) for value in categorical}
        if len(atomic_payloads) != 1:
            return PriorityExactResolution(
                status=ProductFacetStatus.CONFLICT,
                value=None,
            )
        completeness = (
            ValueCompleteness.COMPLETE
            if any(value.completeness is ValueCompleteness.COMPLETE for value in categorical)
            else ValueCompleteness.PARTIAL
        )
        return PriorityExactResolution(
            status=ProductFacetStatus.KNOWN,
            value=CategoricalValue(
                kind="categorical",
                values=categorical[0].values,
                completeness=completeness,
            ),
        )

    if len({canonical_json_bytes(value) for value in values}) == 1:
        return PriorityExactResolution(
            status=ProductFacetStatus.KNOWN,
            value=values[0],
        )
    return PriorityExactResolution(
        status=ProductFacetStatus.CONFLICT,
        value=None,
    )


EXTRACTOR_REGISTRY: Mapping[str, SourceExtractor] = MappingProxyType(
    {PRICE_EXTRACTOR_ID: extract_top_level_price_usd_v1}
)
CATALOG_VALUE_NORMALIZER_REGISTRY: Mapping[str, CatalogValueNormalizer] = MappingProxyType(
    {PRICE_NORMALIZER_ID: normalize_usd_cent_interval_v1}
)
RESOLVER_REGISTRY: Mapping[str, FacetResolver] = MappingProxyType(
    {PRIORITY_EXACT_RESOLVER_ID: resolve_priority_exact_v1}
)


def require_extractor(extractor_id: str) -> SourceExtractor:
    """Return one pinned extractor or fail without fallback."""

    try:
        return EXTRACTOR_REGISTRY[extractor_id]
    except KeyError as error:
        raise ValueError(f"unknown closed extractor ID: {extractor_id}") from error


def require_catalog_value_normalizer(normalizer_id: str) -> CatalogValueNormalizer:
    """Return one pinned catalog normalizer or fail without fallback."""

    try:
        return CATALOG_VALUE_NORMALIZER_REGISTRY[normalizer_id]
    except KeyError as error:
        raise ValueError(f"unknown closed catalog normalizer ID: {normalizer_id}") from error


def require_resolver(resolver_id: str) -> FacetResolver:
    """Return one pinned resolver or fail without fallback."""

    try:
        return RESOLVER_REGISTRY[resolver_id]
    except KeyError as error:
        raise ValueError(f"unknown closed resolver ID: {resolver_id}") from error


def _exact_decimal_price(dollars: Decimal) -> PriceNormalizationResult:
    cents = _exact_cents(dollars)
    if cents is None:
        return _invalid_price()
    return PriceNormalizationResult(
        status=EvidenceStatus.VALID,
        lane=PriceNormalizationLane.EXACT,
        value=NumericValue(
            kind="numeric",
            lower=cents,
            lower_inclusive=True,
            upper=cents,
            upper_inclusive=True,
            unit=USD_CENT_UNIT,
        ),
    )


def _exact_cents(dollars: Decimal) -> int | None:
    if not dollars.is_finite() or dollars.is_signed():
        return None
    cents = dollars * 100
    if cents != cents.to_integral_value():
        return None
    result = int(cents)
    if not 0 <= result <= IJSON_SAFE_INTEGER_MAX:
        return None
    return result


def _invalid_price() -> PriceNormalizationResult:
    return PriceNormalizationResult(
        status=EvidenceStatus.INVALID,
        lane=PriceNormalizationLane.INVALID,
        value=None,
    )
