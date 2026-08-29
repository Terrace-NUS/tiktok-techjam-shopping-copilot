"""Exact equality for immutable domain objects, including scalar runtime types."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from dataclasses import fields, is_dataclass
from typing import cast


def exact_domain_equal(left: object, right: object) -> bool:
    """Compare values without treating bool/int or int/float as interchangeable."""

    if type(left) is not type(right):
        return False
    if is_dataclass(left) and not isinstance(left, type):
        return all(
            exact_domain_equal(getattr(left, item.name), getattr(right, item.name))
            for item in fields(left)
        )
    if type(left) in (tuple, list):
        left_sequence = cast(Sequence[object], left)
        right_sequence = cast(Sequence[object], right)
        return len(left_sequence) == len(right_sequence) and all(
            exact_domain_equal(left_item, right_item)
            for left_item, right_item in zip(left_sequence, right_sequence, strict=True)
        )
    if type(left) in (set, frozenset):
        return cast(Set[object], left) == cast(Set[object], right)
    if type(left) is dict:
        left_mapping = cast(Mapping[object, object], left)
        right_mapping = cast(Mapping[object, object], right)
        if left_mapping.keys() != right_mapping.keys():
            return False
        return all(
            exact_domain_equal(left_mapping[key], right_mapping[key]) for key in left_mapping
        )
    return bool(left == right)
