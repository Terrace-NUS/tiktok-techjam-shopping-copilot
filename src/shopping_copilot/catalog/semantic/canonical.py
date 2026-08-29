"""RFC 8785 canonical JSON primitives shared by catalog-semantic stages."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import TypeAlias, cast

import rfc8785

from .errors import CanonicalJsonError

IJSON_SAFE_INTEGER_MAX = 9_007_199_254_740_991
IJSON_SAFE_INTEGER_MIN = -IJSON_SAFE_INTEGER_MAX

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
CanonicalScalar: TypeAlias = str | int | float | bool


def canonical_json_bytes(value: object) -> bytes:
    """Return RFC 8785 JCS bytes after contract-defined object conversion."""

    converted = _to_json_value(value, path=())
    try:
        return rfc8785.dumps(converted)
    except rfc8785.CanonicalizationError as error:
        raise CanonicalJsonError(str(error)) from error


def canonical_json_text(value: object) -> str:
    """Return canonical JSON as UTF-8 text without BOM or trailing newline."""

    return canonical_json_bytes(value).decode("utf-8")


def sha256_hex(payload: bytes) -> str:
    """Return a full lowercase SHA-256 hexadecimal digest."""

    return hashlib.sha256(payload).hexdigest()


def content_id_for_bytes(payload: bytes) -> str:
    """Return the contract content identifier for exact bytes."""

    return f"sha256:{sha256_hex(payload)}"


def content_id_for_value(value: object) -> str:
    """Return the contract content identifier for canonical JSON bytes."""

    return content_id_for_bytes(canonical_json_bytes(value))


def canonical_scalar_key(value: CanonicalScalar) -> tuple[int, bytes]:
    """Return the contract's deterministic scalar ordering key."""

    if type(value) is bool:
        rank = 0
    elif type(value) is int:
        rank = 1
    elif type(value) is float:
        rank = 2
    elif type(value) is str:
        rank = 3
    else:
        raise CanonicalJsonError("canonical scalar must be bool, int, float, or str")
    return rank, canonical_json_bytes(value)


def validate_semantic_string(value: object, *, name: str) -> str:
    """Validate the stricter string domain used by semantic artifact fields."""

    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must be non-empty")
    for character in value:
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            raise ValueError(f"{name} must not contain a lone surrogate")
        if codepoint <= 0x1F or 0x7F <= codepoint <= 0x9F:
            raise ValueError(f"{name} must not contain control characters")
    return value


def _to_json_value(value: object, *, path: tuple[str | int, ...]) -> JsonValue:
    if isinstance(value, Enum):
        return _to_json_value(value.value, path=path)

    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _to_json_value(getattr(value, item.name), path=(*path, item.name))
            for item in fields(value)
        }

    if value is None or type(value) is bool:
        return cast(JsonScalar, value)

    if type(value) is int:
        if not IJSON_SAFE_INTEGER_MIN <= value <= IJSON_SAFE_INTEGER_MAX:
            raise CanonicalJsonError(f"integer outside I-JSON safe range at {_render_path(path)}")
        return value

    if type(value) is float:
        number = value
        if not math.isfinite(number):
            raise CanonicalJsonError(f"non-finite float at {_render_path(path)}")
        if number == 0.0 and math.copysign(1.0, number) < 0.0:
            raise CanonicalJsonError(f"negative zero at {_render_path(path)}")
        return number

    if type(value) is str:
        text = value
        for character in text:
            if 0xD800 <= ord(character) <= 0xDFFF:
                raise CanonicalJsonError(f"lone surrogate at {_render_path(path)}")
        return text

    if type(value) in (tuple, list):
        sequence = cast(Sequence[object], value)
        return [_to_json_value(item, path=(*path, index)) for index, item in enumerate(sequence)]

    if type(value) is dict:
        mapping = cast(Mapping[object, object], value)
        converted: dict[str, JsonValue] = {}
        for key, item in mapping.items():
            if type(key) is not str:
                raise CanonicalJsonError(f"non-string object key at {_render_path(path)}")
            converted[key] = _to_json_value(item, path=(*path, key))
        return converted

    raise CanonicalJsonError(
        f"unsupported canonical JSON value {type(value).__name__} at {_render_path(path)}"
    )


def _render_path(path: tuple[str | int, ...]) -> str:
    return "$" if not path else "$" + "".join(f"[{item!r}]" for item in path)
