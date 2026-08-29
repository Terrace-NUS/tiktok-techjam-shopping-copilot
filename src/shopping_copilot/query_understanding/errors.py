"""Stable failures for the Query Understanding application boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from shopping_copilot.session_context.errors import ErrorPathSegment
from shopping_copilot.session_context.models import ScalarValue

QueryUnderstandingErrorDetail: TypeAlias = tuple[str, ScalarValue]


class QueryUnderstandingErrorCode(str, Enum):
    """Small error vocabulary shared by provider, wire, and planner layers."""

    MISSING_API_KEY = "missing_api_key"
    PROVIDER_AUTH = "provider_auth"
    PROVIDER_RATE_LIMIT = "provider_rate_limit"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_TIMEOUT = "provider_timeout"
    INVALID_PROVIDER_RESPONSE = "invalid_provider_response"
    INVALID_TOOL_CALL = "invalid_tool_call"
    INVALID_FRAME = "invalid_frame"
    STALE_INTENT_VERSION = "stale_intent_version"
    UNKNOWN_ACTIVE_REF = "unknown_active_ref"
    UNKNOWN_CATEGORY_REF = "unknown_category_ref"
    UNKNOWN_PRODUCT_REF = "unknown_product_ref"
    INVALID_PREFERENCE = "invalid_preference"
    INVALID_FINAL_STATE = "invalid_final_state"
    PREVIEW_REJECTED = "preview_rejected"
    REPAIR_EXHAUSTED = "repair_exhausted"


@dataclass(frozen=True, slots=True, kw_only=True)
class QueryUnderstandingError(ValueError):
    """Machine-readable failure without provider secrets or raw responses."""

    code: QueryUnderstandingErrorCode
    path: tuple[ErrorPathSegment, ...] = ()
    details: tuple[QueryUnderstandingErrorDetail, ...] = ()

    def __post_init__(self) -> None:
        if type(self.code) is not QueryUnderstandingErrorCode:
            raise TypeError("code must be a QueryUnderstandingErrorCode")
        path = tuple(self.path)
        if any(type(item) not in (str, int) for item in path):
            raise TypeError("error path segments must be strings or integers")
        details: list[QueryUnderstandingErrorDetail] = []
        for detail in self.details:
            if type(detail) is not tuple or len(detail) != 2:
                raise TypeError("each error detail must be a key/value tuple")
            key, value = detail
            if type(key) is not str or type(value) not in (str, int, float, bool):
                raise TypeError("error details must contain a string key and JSON scalar")
            if type(value) is float and not math.isfinite(value):
                raise ValueError("error detail numbers must be finite")
            details.append((key, value))
        canonical_details = tuple(sorted(details, key=lambda item: item[0]))
        if len({key for key, _ in canonical_details}) != len(canonical_details):
            raise ValueError("error detail keys must be unique")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "details", canonical_details)
        ValueError.__init__(self, self._render_message())

    def _render_message(self) -> str:
        location = ".".join(str(item) for item in self.path)
        return self.code.value if not location else f"{self.code.value} at {location}"
