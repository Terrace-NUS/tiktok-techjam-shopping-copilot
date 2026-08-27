"""Closed committed-operation vocabulary for intent updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from .models import Preference


@dataclass(frozen=True, slots=True, kw_only=True)
class AddPreference:
    """Add one non-conflicting committed preference."""

    op: Literal["add_preference"] = field(default="add_preference", init=False)
    preference: Preference


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplaceFacet:
    """Replace the complete active preference state for one facet."""

    op: Literal["replace_facet"] = field(default="replace_facet", init=False)
    facet: str
    preferences: tuple[Preference, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class RemovePreference:
    """Remove semantic or structured preferences by stable ID."""

    op: Literal["remove_preference"] = field(default="remove_preference", init=False)
    preference_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class ClearFacet:
    """Return one facet to the unset state."""

    op: Literal["clear_facet"] = field(default="clear_facet", init=False)
    facet: str


@dataclass(frozen=True, slots=True, kw_only=True)
class SetDontCare:
    """Mark one facet as explicitly irrelevant."""

    op: Literal["set_dont_care"] = field(default="set_dont_care", init=False)
    facet: str


@dataclass(frozen=True, slots=True, kw_only=True)
class SwitchGoal:
    """Replace the active goal and optionally carry existing preferences."""

    op: Literal["switch_goal"] = field(default="switch_goal", init=False)
    new_goal: str
    carry_preference_ids: tuple[str, ...] = ()


StateOperation: TypeAlias = (
    AddPreference | ReplaceFacet | RemovePreference | ClearFacet | SetDontCare | SwitchGoal
)


@dataclass(frozen=True, slots=True, kw_only=True)
class StateUpdateBatch:
    """Ordered, version-checked committed intent update."""

    turn: int
    base_intent_version: int
    operations: tuple[StateOperation, ...]
