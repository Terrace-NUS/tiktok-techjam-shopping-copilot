"""Stable catalog-bound runtime failures kept separate from session-context v1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from shopping_copilot.session_context.errors import ErrorPathSegment
from shopping_copilot.session_context.models import ScalarValue

CatalogGatewayErrorDetail: TypeAlias = tuple[str, ScalarValue]


class CatalogGatewayErrorCode(str, Enum):
    """Frozen P0 catalog-semantic gateway and envelope error vocabulary."""

    RELEASE_MISMATCH = "release_mismatch"
    INVALID_RESERVED_CATEGORY_OPERATION = "invalid_reserved_category_operation"
    UNKNOWN_CATEGORY_SCOPE = "unknown_category_scope"
    FACET_NOT_COMMITTABLE = "facet_not_committable"
    VALUE_NOT_GROUNDED = "value_not_grounded"
    INAPPLICABLE_PREFERENCE_AFTER_CATEGORY_CHANGE = "inapplicable_preference_after_category_change"
    PROBE_FACET_NOT_ELIGIBLE = "probe_facet_not_eligible"
    UNTRUSTED_SEARCH_BELIEF = "untrusted_search_belief"
    CATALOG_COMMIT_MISMATCH = "catalog_commit_mismatch"
    INVALID_SESSION_ENVELOPE = "invalid_session_envelope"
    SESSION_SNAPSHOT_HASH_MISMATCH = "session_snapshot_hash_mismatch"
    SESSION_REPLAY_MISMATCH = "session_replay_mismatch"


@dataclass(frozen=True, slots=True, kw_only=True)
class CatalogGatewayError(ValueError):
    """Machine-readable failure raised only by the catalog-bound runtime layer."""

    code: CatalogGatewayErrorCode
    path: tuple[ErrorPathSegment, ...] = ()
    operation_index: int | None = None
    details: tuple[CatalogGatewayErrorDetail, ...] = ()

    def __post_init__(self) -> None:
        if type(self.code) is not CatalogGatewayErrorCode:
            raise TypeError("code must be a CatalogGatewayErrorCode")
        if self.operation_index is not None and (
            type(self.operation_index) is not int or self.operation_index < 0
        ):
            raise ValueError("operation_index must be a non-negative int or None")
        path = tuple(self.path)
        if any(type(item) not in (str, int) for item in path):
            raise TypeError("error path segments must be strings or integers")
        details: list[CatalogGatewayErrorDetail] = []
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
