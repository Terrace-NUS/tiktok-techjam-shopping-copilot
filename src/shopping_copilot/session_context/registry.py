"""Immutable facet schema and deterministic scalar normalization."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import cast

from .errors import ErrorCode, ErrorPathSegment, SessionContextError
from .models import Operator, PreferenceValue, ScalarValue

ScalarNormalizer = Callable[[ScalarValue], ScalarValue]


class FacetKind(str, Enum):
    """Closed family of structured facet kinds."""

    CATEGORICAL = "categorical"
    NUMERIC = "numeric"


class FacetAuthority(str, Enum):
    """Evidence boundary that authorizes a facet to enter committed intent."""

    CATALOG_VERIFIED = "catalog_verified"
    RETRIEVAL_DERIVED = "retrieval_derived"


CATEGORICAL_OPERATORS = frozenset({Operator.EQ, Operator.NEQ, Operator.IN, Operator.NOT_IN})
NUMERIC_OPERATORS = frozenset({Operator.LT, Operator.LE, Operator.GT, Operator.GE})

_FACET_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


def canonical_text(value: ScalarValue) -> ScalarValue:
    """Return the deliberately small v1 canonical form for a text scalar."""

    if type(value) is not str:
        raise TypeError("text values must be strings")
    normalized = unicodedata.normalize("NFKC", value)
    if any(unicodedata.category(character) == "Cc" for character in normalized):
        raise ValueError("text values must not contain control characters")
    normalized = " ".join(normalized.split()).casefold()
    if not normalized:
        raise ValueError("text values must not be empty")
    return normalized


def canonical_number(value: ScalarValue) -> ScalarValue:
    """Return the canonical finite JSON number form, excluding booleans."""

    if type(value) not in (int, float):
        raise TypeError("numeric values must be integers or floats")
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("numeric values must be finite")
        if value.is_integer():
            return int(value)
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class FacetSpec:
    """Trusted immutable configuration for one canonical structured facet."""

    id: str
    kind: FacetKind
    operators: frozenset[Operator]
    normalizer: ScalarNormalizer
    authority: FacetAuthority = FacetAuthority.CATALOG_VERIFIED

    def __post_init__(self) -> None:
        if type(self.id) is not str or _FACET_ID_PATTERN.fullmatch(self.id) is None:
            raise ValueError("facet id must be canonical lower snake case")
        if self.id == "other":
            raise ValueError("'other' is an adapter key, not a structured facet")
        if not isinstance(self.kind, FacetKind):
            raise TypeError("facet kind must be a FacetKind")
        if not isinstance(self.authority, FacetAuthority):
            raise TypeError("facet authority must be a FacetAuthority")
        operators = frozenset(self.operators)
        if not all(isinstance(operator, Operator) for operator in operators):
            raise TypeError("facet operators must be Operator values")
        expected = (
            CATEGORICAL_OPERATORS if self.kind is FacetKind.CATEGORICAL else NUMERIC_OPERATORS
        )
        if operators != expected:
            raise ValueError("facet operators must equal the closed family for its kind")
        if not callable(self.normalizer):
            raise TypeError("facet normalizer must be callable")
        object.__setattr__(self, "operators", operators)


@dataclass(frozen=True, slots=True, init=False)
class FacetRegistry:
    """Immutable, catalog-independent collection of reviewed facet specs."""

    _specs: tuple[FacetSpec, ...]

    def __init__(self, *, specs: Iterable[FacetSpec]) -> None:
        copied = tuple(specs)
        if not all(type(spec) is FacetSpec for spec in copied):
            raise TypeError("registry entries must be FacetSpec values")
        ids = tuple(spec.id for spec in copied)
        if len(set(ids)) != len(ids):
            raise ValueError("facet ids must be unique")
        object.__setattr__(self, "_specs", tuple(sorted(copied, key=lambda spec: spec.id)))

    @property
    def specs(self) -> tuple[FacetSpec, ...]:
        """Return specs in deterministic facet-ID order."""

        return self._specs

    def __iter__(self) -> Iterator[FacetSpec]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def get(self, facet: str) -> FacetSpec | None:
        """Look up one canonical facet without raising a domain error."""

        return next((spec for spec in self._specs if spec.id == facet), None)

    def require(
        self,
        facet: str,
        *,
        path: tuple[ErrorPathSegment, ...] = (),
    ) -> FacetSpec:
        """Look up a facet or raise the stable untrusted-input error."""

        spec = self.get(facet)
        if spec is None:
            raise SessionContextError(
                code=ErrorCode.UNKNOWN_FACET,
                path=path,
                details=(("facet", facet),) if type(facet) is str else (),
            )
        return spec

    def normalize_value(
        self,
        facet: str,
        operator: Operator,
        value: PreferenceValue,
        *,
        path: tuple[ErrorPathSegment, ...] = (),
    ) -> PreferenceValue:
        """Normalize one operator value without repairing its caller-owned object."""

        spec = self.require(facet, path=path)
        if not isinstance(operator, Operator) or operator not in spec.operators:
            raise SessionContextError(
                code=ErrorCode.INVALID_OPERATOR_FOR_FACET,
                path=path,
                details=(("facet", facet),),
            )

        if operator in (Operator.IN, Operator.NOT_IN):
            if type(value) is not tuple or not value:
                raise _invalid_operator_value(path)
            if not all(_is_scalar(item) for item in value):
                raise _invalid_operator_value(path)
            if len({type(item) for item in value}) != 1:
                raise _invalid_operator_value(path)
            normalized_items = tuple(
                _normalize_scalar(spec.normalizer, item, path=path) for item in value
            )
            if len({type(item) for item in normalized_items}) != 1:
                raise _invalid_operator_value(path)
            if len({_typed_scalar(item) for item in normalized_items}) != len(normalized_items):
                normalized_items = tuple(
                    {_typed_scalar(item): item for item in normalized_items}.values()
                )
            return _sort_homogeneous(normalized_items)

        if type(value) is tuple or not _is_scalar(value):
            raise _invalid_operator_value(path)
        normalized = _normalize_scalar(
            spec.normalizer,
            cast(ScalarValue, value),
            path=path,
        )
        if spec.kind is FacetKind.NUMERIC and type(normalized) not in (int, float):
            raise _invalid_operator_value(path)
        return normalized


def _normalize_scalar(
    normalizer: ScalarNormalizer,
    value: ScalarValue,
    *,
    path: tuple[ErrorPathSegment, ...],
) -> ScalarValue:
    try:
        normalized = normalizer(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise _invalid_operator_value(path) from error
    if not _is_scalar(normalized):
        raise _invalid_operator_value(path)
    return normalized


def _is_scalar(value: object) -> bool:
    return type(value) in (str, int, float, bool) and not (
        type(value) is float and not math.isfinite(value)
    )


def _typed_scalar(value: ScalarValue) -> tuple[type[object], ScalarValue]:
    return type(value), value


def _sort_homogeneous(values: tuple[ScalarValue, ...]) -> tuple[ScalarValue, ...]:
    value_type = type(values[0])
    if value_type is str:
        return cast(tuple[ScalarValue, ...], tuple(sorted(cast(tuple[str, ...], values))))
    if value_type is bool:
        return cast(tuple[ScalarValue, ...], tuple(sorted(cast(tuple[bool, ...], values))))
    if value_type is int:
        return cast(tuple[ScalarValue, ...], tuple(sorted(cast(tuple[int, ...], values))))
    return cast(tuple[ScalarValue, ...], tuple(sorted(cast(tuple[float, ...], values))))


def _invalid_operator_value(
    path: tuple[ErrorPathSegment, ...],
) -> SessionContextError:
    return SessionContextError(code=ErrorCode.INVALID_OPERATOR_VALUE, path=path)
