"""Exact-scope capability lookup with fail-closed missing-row behavior."""

from __future__ import annotations

from dataclasses import dataclass

from ..facet.gate_b_models import (
    EffectiveFacetCapability,
    EffectiveFacetCapabilitySet,
    RuntimePromotionDecision,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ExactFacetPermissions:
    """Consumer view of one exact row; missing rows expose no permission."""

    facet_id: str
    category_scope_id: str
    matched: bool
    decision: RuntimePromotionDecision | None
    intent_committable: bool
    retrieval_eligible: bool
    probe_eligible: bool
    clarification_eligible: bool

    def __post_init__(self) -> None:
        if type(self.facet_id) is not str or type(self.category_scope_id) is not str:
            raise TypeError("exact capability keys must be strings")
        flags = (
            self.matched,
            self.intent_committable,
            self.retrieval_eligible,
            self.probe_eligible,
            self.clarification_eligible,
        )
        if any(type(item) is not bool for item in flags):
            raise TypeError("exact capability flags must be booleans")
        if self.decision is not None and type(self.decision) is not RuntimePromotionDecision:
            raise TypeError("exact capability decision is invalid")
        if self.matched != (self.decision is not None):
            raise ValueError("exact capability matched state is inconsistent")
        if not self.matched and any(flags[1:]):
            raise ValueError("a missing exact capability row must deny every permission")


class ExactCapabilityIndex:
    """Immutable exact-key index; it performs no parent, child, or union lookup."""

    __slots__ = ("_by_key",)

    def __init__(self, capabilities: EffectiveFacetCapabilitySet) -> None:
        if type(capabilities) is not EffectiveFacetCapabilitySet:
            raise TypeError("ExactCapabilityIndex requires EffectiveFacetCapabilitySet")
        self._by_key: dict[tuple[str, str], EffectiveFacetCapability] = {
            (item.facet_id, item.category_scope_id): item for item in capabilities.entries
        }

    def lookup(self, facet_id: str, category_scope_id: str) -> ExactFacetPermissions:
        """Return one exact permission view, denying all permissions if absent."""

        if type(facet_id) is not str or type(category_scope_id) is not str:
            raise TypeError("exact capability lookup keys must be strings")
        entry = self._by_key.get((facet_id, category_scope_id))
        if entry is None:
            return ExactFacetPermissions(
                facet_id=facet_id,
                category_scope_id=category_scope_id,
                matched=False,
                decision=None,
                intent_committable=False,
                retrieval_eligible=False,
                probe_eligible=False,
                clarification_eligible=False,
            )
        return _permission_view(entry)


def _permission_view(entry: EffectiveFacetCapability) -> ExactFacetPermissions:
    return ExactFacetPermissions(
        facet_id=entry.facet_id,
        category_scope_id=entry.category_scope_id,
        matched=True,
        decision=entry.decision,
        intent_committable=entry.intent_committable,
        retrieval_eligible=entry.retrieval_eligible,
        probe_eligible=entry.probe_eligible,
        clarification_eligible=entry.clarification_eligible,
    )
