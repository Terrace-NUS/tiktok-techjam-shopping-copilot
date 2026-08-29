"""Closed CS5A intent normalizers resolved from declarative runtime IDs."""

from __future__ import annotations

from collections.abc import Callable

from shopping_copilot.session_context.models import ScalarValue

from ..canonical import IJSON_SAFE_INTEGER_MAX, IJSON_SAFE_INTEGER_MIN
from ..category import CategoryRegistry
from ..facet.gate_b_models import PRICE_INTENT_NORMALIZER_ID
from .models import CATEGORY_SCOPE_ID_NORMALIZER_ID

IntentValueNormalizer = Callable[[ScalarValue], ScalarValue]


def normalize_usd_cent_int_v1(value: ScalarValue) -> ScalarValue:
    """Accept only non-boolean I-JSON-safe integer cents and return a fixed point."""

    if type(value) is not int:
        raise TypeError("USD_CENT intent values must be non-boolean integers")
    if not IJSON_SAFE_INTEGER_MIN <= value <= IJSON_SAFE_INTEGER_MAX:
        raise ValueError("USD_CENT intent value is outside the I-JSON safe range")
    return value


def require_intent_value_normalizer(
    normalizer_id: str,
    *,
    registry: CategoryRegistry,
) -> IntentValueNormalizer:
    """Resolve a closed normalizer ID without identity or fuzzy fallback."""

    if normalizer_id == PRICE_INTENT_NORMALIZER_ID:
        return normalize_usd_cent_int_v1
    if normalizer_id == CATEGORY_SCOPE_ID_NORMALIZER_ID:
        return _category_scope_id_normalizer(registry)
    raise ValueError(f"unknown closed intent normalizer ID: {normalizer_id}")


def _category_scope_id_normalizer(registry: CategoryRegistry) -> IntentValueNormalizer:
    if type(registry) is not CategoryRegistry:
        raise TypeError("category scope normalizer requires CategoryRegistry")
    valid_scope_ids = frozenset(item.id for item in registry.scopes)

    def normalize(value: ScalarValue) -> ScalarValue:
        if type(value) is not str:
            raise TypeError("category scope values must be strings")
        if value not in valid_scope_ids:
            raise ValueError("category scope value is not published in this release")
        return value

    return normalize
