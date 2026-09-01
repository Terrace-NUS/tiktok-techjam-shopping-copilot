"""Project a grounded product card into reviewable simulator disclosures."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, replace
from typing import Literal

from shopping_copilot.catalog.product_facts import (
    ProductFact,
    ProductFactCard,
    ProductFactPolarity,
)

AskAttribute = Literal[
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
]
ScenarioType = Literal["buying", "browsing", "intent_override", "boundary"]
Commitment = Literal["hard", "soft"]

_ATTRIBUTE_ORDER: tuple[AskAttribute, ...] = (
    "material",
    "feature",
    "color",
    "size",
    "style",
    "use_case",
    "brand",
    "budget",
    "category",
)
_ATTRIBUTE_CAPS: dict[AskAttribute, int] = {
    "category": 1,
    "material": 2,
    "color": 2,
    "size": 2,
    "style": 2,
    "brand": 1,
    "budget": 1,
    "feature": 4,
    "use_case": 2,
}
_EXCLUDED_FACETS = frozenset(
    {
        "average_rating",
        "date_first_available",
        "dimensions",
        "discontinued",
        "manufacturer",
        "model",
        "model_name",
        "model_number",
        "origin",
        "product_dimensions",
        "product_name",
        "product_type",
        "package_dimensions",
        "rating_number",
        "series",
        "weight",
    }
)
_LOW_VALUE_TEXT = re.compile(
    r"\b(?:an amazon brand|imported|is discontinued by manufacturer|package dimensions|"
    r"see more|click here|since its foundation|the very essence of style)\b",
    re.IGNORECASE,
)
_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True, kw_only=True)
class DisclosureFact:
    """One card fact approved for possible simulator disclosure."""

    id: str
    facet: str
    ask_attribute: AskAttribute
    value: str
    aliases: tuple[str, ...]
    component: str | None
    polarity: Literal["present", "absent"]
    meaning: str
    evidence: str
    source_ref: str
    confidence: float
    score: float
    commitment: Commitment
    utterance: str


@dataclass(frozen=True, slots=True, kw_only=True)
class FactDecision:
    """Audit decision for a grounded fact before disclosure-plan selection."""

    facet: str
    value: str
    component: str | None
    selected: bool
    reason: str
    disclosure_id: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class DisclosurePlan:
    """A bounded, scenario-aware view over a complete grounded product card."""

    parent_asin: str
    scenario_type: ScenarioType
    product_type: str
    summary: str
    disclosures: tuple[DisclosureFact, ...]
    decisions: tuple[FactDecision, ...]


def project_product_card_disclosures(
    card: ProductFactCard,
    *,
    scenario_type: ScenarioType,
    minimum_facts: int = 0,
    maximum_facts: int = 10,
) -> DisclosurePlan:
    """Select shopping-meaningful facts without the legacy four-string cap."""

    if type(card) is not ProductFactCard:
        raise TypeError("card must be an exact ProductFactCard")
    if scenario_type not in {"buying", "browsing", "intent_override", "boundary"}:
        raise ValueError("scenario_type is invalid")
    if type(minimum_facts) is not int or not 0 <= minimum_facts <= 20:
        raise ValueError("minimum_facts must be an integer between zero and 20")
    if type(maximum_facts) is not int or not 1 <= maximum_facts <= 20:
        raise ValueError("maximum_facts must be an integer between 1 and 20")
    if minimum_facts > maximum_facts:
        raise ValueError("minimum_facts cannot exceed maximum_facts")

    grouped_candidates: dict[tuple[str, str, str], list[tuple[ProductFact, DisclosureFact]]] = {}
    decisions: list[FactDecision] = []
    for fact in card.facts:
        reason = _exclusion_reason(fact)
        if reason is not None:
            decisions.append(_decision(fact, selected=False, reason=reason))
            continue
        ask_attribute = _ask_attribute(fact.facet)
        preferred_value = _preferred_value(fact)
        key = (
            ask_attribute,
            _normalize_key(fact.component or "item"),
            _normalize_key(preferred_value),
        )
        disclosure_id = _disclosure_id(card.parent_asin, fact)
        disclosure = DisclosureFact(
            id=disclosure_id,
            facet=fact.facet,
            ask_attribute=ask_attribute,
            value=preferred_value,
            aliases=tuple(_clean(alias) for alias in fact.aliases),
            component=(None if fact.component is None else _clean(fact.component)),
            polarity=fact.polarity.value,
            meaning=_clean(fact.meaning),
            evidence=_clean(fact.evidence),
            source_ref=fact.source_ref,
            confidence=float(fact.confidence),
            score=_fact_score(fact, ask_attribute=ask_attribute),
            commitment="soft",
            utterance=_utterance(
                fact,
                ask_attribute=ask_attribute,
                preferred_value=preferred_value,
            ),
        )
        grouped_candidates.setdefault(key, []).append((fact, disclosure))

    candidates: list[DisclosureFact] = []
    for values in grouped_candidates.values():
        _, winner = min(values, key=lambda item: (-item[1].score, item[1].id))
        candidates.append(winner)
        for fact, disclosure in values:
            if disclosure.id == winner.id:
                continue
            decisions.append(_decision(fact, selected=False, reason="semantic_duplicate"))

    selected = _diverse_selection(
        candidates,
        minimum_facts=minimum_facts,
        maximum_facts=maximum_facts,
    )
    if len(selected) < minimum_facts:
        raise ValueError(
            f"product card has only {len(selected)} eligible grounded facts; "
            f"minimum_facts={minimum_facts}"
        )
    hard_count = {
        "buying": min(2, len(selected)),
        "browsing": 0,
        "intent_override": min(2, len(selected)),
        "boundary": min(1, len(selected)),
    }[scenario_type]
    selected = tuple(
        replace(item, commitment="hard" if index < hard_count else "soft")
        for index, item in enumerate(selected)
    )
    selected_ids = {item.id for item in selected}
    candidate_by_id = {item.id: item for item in candidates}
    for candidate in candidates:
        if candidate.id in selected_ids:
            decisions.append(
                FactDecision(
                    facet=candidate.facet,
                    value=candidate.value,
                    component=candidate.component,
                    selected=True,
                    reason="selected_for_disclosure",
                    disclosure_id=candidate.id,
                )
            )
        else:
            decisions.append(
                FactDecision(
                    facet=candidate.facet,
                    value=candidate.value,
                    component=candidate.component,
                    selected=False,
                    reason="lower_priority_or_attribute_cap",
                    disclosure_id=candidate.id,
                )
            )
    if set(candidate_by_id) != {item.disclosure_id for item in decisions if item.disclosure_id}:
        raise AssertionError("fact decision audit is incomplete")
    return DisclosurePlan(
        parent_asin=card.parent_asin,
        scenario_type=scenario_type,
        product_type=_product_type(card),
        summary=_clean(card.summary),
        disclosures=selected,
        decisions=tuple(decisions),
    )


def _exclusion_reason(fact: ProductFact) -> str | None:
    if fact.polarity is ProductFactPolarity.ABSENT and not (
        fact.facet == "material" or fact.facet.endswith("_material")
    ):
        return "negative_or_absent_catalog_fact"
    if fact.confidence < 0.85:
        return "confidence_below_0_85"
    if (
        fact.facet in _EXCLUDED_FACETS
        or "model" in fact.facet
        or "origin" in fact.facet
        or "rating" in fact.facet
        or fact.facet == "brand_type"
    ):
        return "identity_or_non_shopping_metadata"
    value = _clean(fact.value)
    if "�" in value:
        return "corrupted_source_text"
    if value.casefold().startswith("not real "):
        return "ambiguous_negated_material_phrase"
    if len(value) > 180:
        return "value_too_long_for_customer_disclosure"
    if _LOW_VALUE_TEXT.search(value):
        return "marketing_or_navigation_text"
    if fact.facet == "feature" and len(value) > 120 and " no " not in f" {value.casefold()} ":
        return "raw_feature_too_long"
    return None


def _ask_attribute(facet: str) -> AskAttribute:
    if facet == "brand":
        return "brand"
    if facet == "price":
        return "budget"
    if facet in {"category", "product_type", "gender", "department"}:
        return "category"
    if facet == "material" or facet.endswith("_material"):
        return "material"
    if facet == "color" or facet.endswith("_color"):
        return "color"
    if facet == "size" or facet.endswith("_size") or facet.startswith("size_"):
        return "size"
    if (
        facet == "style"
        or facet.endswith("_style")
        or facet
        in {
            "fit",
            "neckline",
            "pattern",
            "sleeve_length",
        }
    ):
        return "style"
    if facet in {"activity", "occasion", "use_case"} or facet.endswith("_use_case"):
        return "use_case"
    return "feature"


def _fact_score(fact: ProductFact, *, ask_attribute: AskAttribute) -> float:
    base = {
        "material": 100.0,
        "feature": 92.0 if fact.facet != "feature" else 72.0,
        "use_case": 90.0,
        "color": 86.0,
        "size": 84.0,
        "style": 82.0,
        "brand": 74.0,
        "category": 68.0,
        "budget": 62.0,
    }[ask_attribute]
    if fact.facet in {"closure", "closure_type"}:
        base = 112.0
    elif fact.facet == "care_instruction":
        base = 105.0 + min(5.0, len(fact.value) / 25.0)
    elif fact.facet == "recycled_content":
        base = 104.0
    if fact.component is not None and ask_attribute != "material":
        base += 4.0
    if fact.polarity is ProductFactPolarity.ABSENT:
        base += 5.0
    if ask_attribute == "material" and "%" in fact.value:
        base += 4.0
    if fact.aliases:
        base += 2.0
    if len(fact.value) <= 60:
        base += 2.0
    return base + float(fact.confidence)


def _diverse_selection(
    candidates: list[DisclosureFact],
    *,
    minimum_facts: int,
    maximum_facts: int,
) -> tuple[DisclosureFact, ...]:
    ordered = sorted(candidates, key=lambda item: (-item.score, item.id))
    selected: list[DisclosureFact] = []
    counts: dict[AskAttribute, int] = {attribute: 0 for attribute in _ATTRIBUTE_ORDER}
    selected_ids: set[str] = set()
    facet_group_counts: Counter[str] = Counter()
    by_attribute = {
        attribute: [item for item in ordered if item.ask_attribute == attribute]
        for attribute in _ATTRIBUTE_ORDER
    }
    for attribute in _ATTRIBUTE_ORDER:
        if len(selected) >= maximum_facts or not by_attribute[attribute]:
            continue
        item = by_attribute[attribute][0]
        selected.append(item)
        selected_ids.add(item.id)
        counts[attribute] += 1
        facet_group_counts[_facet_group(item.facet)] += 1
    for item in ordered:
        if len(selected) >= maximum_facts:
            break
        if item.id in selected_ids:
            continue
        if counts[item.ask_attribute] >= _ATTRIBUTE_CAPS[item.ask_attribute]:
            continue
        facet_group = _facet_group(item.facet)
        if facet_group_counts[facet_group] >= _facet_group_cap(facet_group):
            continue
        selected.append(item)
        selected_ids.add(item.id)
        counts[item.ask_attribute] += 1
        facet_group_counts[facet_group] += 1
    # A released journey may require a fixed minimum amount of auditable evidence.
    # Backfill with the best remaining grounded facts only after the diversity-first
    # pass. The maximum still bounds disclosure, and duplicate facts remain excluded.
    for item in ordered:
        if len(selected) >= minimum_facts:
            break
        if item.id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item.id)
    return tuple(selected)


def _product_type(card: ProductFactCard) -> str:
    product_types = [
        _clean(fact.value)
        for fact in card.facts
        if fact.facet == "product_type" and fact.polarity is ProductFactPolarity.PRESENT
    ]
    if product_types:
        return product_types[0]
    category_values = [
        _clean(fact.value)
        for fact in card.facts
        if fact.facet == "category" and fact.polarity is ProductFactPolarity.PRESENT
    ]
    return category_values[-1] if category_values else "product"


def _utterance(
    fact: ProductFact,
    *,
    ask_attribute: AskAttribute,
    preferred_value: str,
) -> str:
    value = preferred_value
    component = _clean(fact.component) if fact.component else "item"
    if ask_attribute == "brand":
        return f"I prefer the {value} brand."
    if ask_attribute == "budget":
        displayed = value if "$" in value else f"${value}"
        return f"My budget is around {displayed}."
    if ask_attribute == "material":
        if fact.polarity is ProductFactPolarity.ABSENT:
            return f"The {component} should not use {value}."
        return f"The {component} should use {value}."
    if ask_attribute == "color":
        return f"I prefer {value} for the {component}."
    if ask_attribute == "size":
        return f"The {component} should be {value}."
    if ask_attribute == "style":
        return f"I prefer {value}."
    if ask_attribute == "use_case":
        return f"It should work well for {value}."
    if ask_attribute == "category":
        return f"It should be suitable for {value}."
    if fact.facet in {"closure", "closure_type"}:
        return f"It should use {value}."
    if fact.facet == "care_instruction":
        return f"The care instructions should be {value}."
    if fact.facet == "recycled_content":
        return f"I prefer {value}."
    return f"It should have {value}."


def _facet_group(facet: str) -> str:
    if facet in {"closure", "closure_type"}:
        return "closure"
    if facet == "care_instruction":
        return "care_instruction"
    return facet


def _facet_group_cap(group: str) -> int:
    if group in {"closure", "care_instruction", "brand", "price", "department", "gender"}:
        return 1
    if group == "feature":
        return 2
    return 20


def _decision(fact: ProductFact, *, selected: bool, reason: str) -> FactDecision:
    return FactDecision(
        facet=fact.facet,
        value=_clean(fact.value),
        component=(None if fact.component is None else _clean(fact.component)),
        selected=selected,
        reason=reason,
    )


def _disclosure_id(parent_asin: str, fact: ProductFact) -> str:
    payload = "\0".join(
        (
            parent_asin,
            fact.facet,
            fact.component or "",
            _clean(fact.value).casefold(),
            fact.source_ref,
        )
    ).encode("utf-8")
    return f"df_{hashlib.sha256(payload).hexdigest()[:20]}"


def _normalize_key(value: str) -> str:
    return _TOKEN_RE.sub(" ", value.casefold()).strip()


def _clean(value: str) -> str:
    return _SPACE_RE.sub(" ", value).strip(" \t\r\n.,;")


def _preferred_value(fact: ProductFact) -> str:
    value = _clean(fact.value)
    if value.isascii():
        return value
    for alias in fact.aliases:
        cleaned = _clean(alias)
        if cleaned.isascii():
            return cleaned
    return value
