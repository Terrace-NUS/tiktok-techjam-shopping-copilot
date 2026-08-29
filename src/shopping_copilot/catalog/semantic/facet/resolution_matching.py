"""Pure four-state matching used to audit CS3 numeric filtering safety."""

from __future__ import annotations

from enum import Enum

from ..errors import ResolutionBuildError
from .gate_a_models import NumericValue, ProductFacetStatus
from .resolution_models import ResolvedProductFacetValue


class FacetMatchResult(str, Enum):
    """Exact result of matching a resolved product fact against one constraint."""

    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


def match_numeric_interval(
    product: ResolvedProductFacetValue,
    allowed: NumericValue,
) -> FacetMatchResult:
    """Match product interval ``D`` against allowed interval ``C``.

    ``D subset C`` is satisfied, disjoint intervals are violated, and partial
    overlap is unknown. Unknown/conflicting product facts never become violations.
    """

    if type(product) is not ResolvedProductFacetValue:
        raise TypeError("numeric matcher requires ResolvedProductFacetValue")
    if type(allowed) is not NumericValue:
        raise TypeError("numeric matcher requires NumericValue constraint")
    if product.status is ProductFacetStatus.NOT_APPLICABLE:
        return FacetMatchResult.NOT_APPLICABLE
    if product.status in (ProductFacetStatus.UNKNOWN, ProductFacetStatus.CONFLICT):
        return FacetMatchResult.UNKNOWN
    if type(product.value) is not NumericValue:
        raise ResolutionBuildError("numeric matcher received a non-numeric KNOWN value")
    if product.value.unit != allowed.unit:
        raise ResolutionBuildError("numeric matcher cannot compare different units")
    if _interval_subset(product.value, allowed):
        return FacetMatchResult.SATISFIED
    if _interval_disjoint(product.value, allowed):
        return FacetMatchResult.VIOLATED
    return FacetMatchResult.UNKNOWN


def safe_filter_keeps(result: FacetMatchResult) -> bool:
    """Conservative retrieval rule: drop only products proven to violate."""

    if type(result) is not FacetMatchResult:
        raise TypeError("safe_filter_keeps requires FacetMatchResult")
    return result is not FacetMatchResult.VIOLATED


def _interval_subset(inner: NumericValue, outer: NumericValue) -> bool:
    return _lower_is_inside(inner, outer) and _upper_is_inside(inner, outer)


def _lower_is_inside(inner: NumericValue, outer: NumericValue) -> bool:
    if outer.lower is None:
        return True
    if inner.lower is None:
        return False
    if inner.lower > outer.lower:
        return True
    if inner.lower < outer.lower:
        return False
    return not inner.lower_inclusive or outer.lower_inclusive


def _upper_is_inside(inner: NumericValue, outer: NumericValue) -> bool:
    if outer.upper is None:
        return True
    if inner.upper is None:
        return False
    if inner.upper < outer.upper:
        return True
    if inner.upper > outer.upper:
        return False
    return not inner.upper_inclusive or outer.upper_inclusive


def _interval_disjoint(left: NumericValue, right: NumericValue) -> bool:
    return _ends_before(left, right) or _ends_before(right, left)


def _ends_before(left: NumericValue, right: NumericValue) -> bool:
    if left.upper is None or right.lower is None:
        return False
    if left.upper < right.lower:
        return True
    if left.upper > right.lower:
        return False
    return not (left.upper_inclusive and right.lower_inclusive)
