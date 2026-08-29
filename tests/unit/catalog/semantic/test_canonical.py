from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum

import pytest

from shopping_copilot.catalog.semantic.canonical import (
    IJSON_SAFE_INTEGER_MAX,
    IJSON_SAFE_INTEGER_MIN,
    canonical_json_bytes,
    canonical_json_text,
    canonical_scalar_key,
    content_id_for_bytes,
    content_id_for_value,
    validate_semantic_string,
)
from shopping_copilot.catalog.semantic.errors import CanonicalJsonError


class _WireValue(str, Enum):
    ALPHA = "alpha"


@dataclass(frozen=True)
class _CanonicalPayload:
    z_values: tuple[int, ...]
    enum_value: _WireValue
    optional: None


def test_canonical_json_converts_contract_types_and_uses_jcs_key_order() -> None:
    payload = _CanonicalPayload(
        z_values=(2, 1),
        enum_value=_WireValue.ALPHA,
        optional=None,
    )

    encoded = canonical_json_bytes(payload)

    assert encoded == b'{"enum_value":"alpha","optional":null,"z_values":[2,1]}'
    assert canonical_json_text(payload) == encoded.decode("utf-8")
    assert not encoded.startswith(b"\xef\xbb\xbf")
    assert not encoded.endswith(b"\n")


def test_canonical_json_preserves_array_order_and_emits_utf8_without_ascii_escaping() -> None:
    value = {"z": ("\u96ea", "\u00e9"), "a": "\u732b"}

    assert canonical_json_bytes(value) == '{"a":"\u732b","z":["\u96ea","\u00e9"]}'.encode()


def test_content_identifiers_hash_exact_bytes_and_canonical_values() -> None:
    payload = b'{"a":1}'
    expected = f"sha256:{hashlib.sha256(payload).hexdigest()}"

    assert content_id_for_bytes(payload) == expected
    assert content_id_for_value({"a": 1}) == expected


@pytest.mark.parametrize("value", [IJSON_SAFE_INTEGER_MIN, IJSON_SAFE_INTEGER_MAX])
def test_canonical_json_accepts_i_json_safe_integer_boundaries(value: int) -> None:
    assert canonical_json_bytes(value) == str(value).encode("ascii")


@pytest.mark.parametrize(
    "value",
    [IJSON_SAFE_INTEGER_MIN - 1, IJSON_SAFE_INTEGER_MAX + 1],
)
def test_canonical_json_rejects_integers_outside_i_json_safe_range(value: int) -> None:
    with pytest.raises(CanonicalJsonError, match="outside I-JSON safe range"):
        canonical_json_bytes({"nested": [value]})


def test_canonical_scalar_order_keeps_bool_distinct_from_integer() -> None:
    values = ("a", 1.5, 1, True, False)

    assert tuple(sorted(values, key=canonical_scalar_key)) == (False, True, 1, 1.5, "a")
    assert canonical_json_bytes(True) == b"true"
    assert canonical_json_bytes(1) == b"1"


def test_canonical_json_accepts_finite_float_and_positive_zero() -> None:
    assert canonical_json_bytes(1.5) == b"1.5"
    assert canonical_json_bytes(0.0) == b"0"


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_canonical_json_rejects_non_finite_float(value: float) -> None:
    with pytest.raises(CanonicalJsonError, match="non-finite float"):
        canonical_json_bytes(value)


def test_canonical_json_rejects_negative_zero_instead_of_normalizing_it() -> None:
    with pytest.raises(CanonicalJsonError, match="negative zero"):
        canonical_json_bytes(-0.0)


def test_canonical_json_rejects_non_string_object_keys() -> None:
    with pytest.raises(CanonicalJsonError, match="non-string object key"):
        canonical_json_bytes({1: "value"})


def test_canonical_json_rejects_lone_unicode_surrogate() -> None:
    with pytest.raises(CanonicalJsonError, match="lone surrogate"):
        canonical_json_bytes({"value": "\ud800"})


def test_canonical_json_rejects_unsupported_values_without_string_coercion() -> None:
    with pytest.raises(CanonicalJsonError, match="unsupported canonical JSON value set"):
        canonical_json_bytes({"values": {"a", "b"}})


@pytest.mark.parametrize(
    "value",
    ["", "contains\x00nul", "contains\x1funit-separator", "contains\x7fdelete", "\udfff"],
)
def test_semantic_string_validator_rejects_empty_control_and_surrogate_values(
    value: str,
) -> None:
    with pytest.raises(ValueError):
        validate_semantic_string(value, name="field")


def test_semantic_string_validator_accepts_nonempty_unicode() -> None:
    assert validate_semantic_string("\u978b \u00e9", name="field") == "\u978b \u00e9"


@pytest.mark.parametrize("value", [None, 1, True, b"text"])
def test_semantic_string_validator_requires_exact_string_type(value: object) -> None:
    with pytest.raises(TypeError, match="must be a string"):
        validate_semantic_string(value, name="field")


def test_canonical_scalar_key_rejects_non_scalar() -> None:
    with pytest.raises(CanonicalJsonError, match="canonical scalar must be"):
        canonical_scalar_key(None)  # type: ignore[arg-type]
