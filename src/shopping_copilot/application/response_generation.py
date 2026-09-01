"""Deterministic, evidence-bound natural-language response generation."""

from __future__ import annotations

import math
import textwrap
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from shopping_copilot.retrieval.deepseek_ranking import QualityRankingHit
from shopping_copilot.session_context import Commitment, IntentState, Operator, Preference

from .quality_ranking import ApertureRankingResult

RESPONSE_SCHEMA = "shopping-copilot/deterministic-response-narrative/v1"
TRANSPARENCY_MOVEMENT_THRESHOLD = 0.12


@dataclass(frozen=True, slots=True, kw_only=True)
class ProductNarrative:
    """One product explanation copied only from ranking and catalog evidence."""

    parent_asin: str
    title: str
    reason: str
    caveat: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ResponseNarrative:
    """Auditable plan and final text for one deterministic assistant response."""

    schema: str
    transparency: float
    previous_transparency: float | None
    presentation_band: str
    movement: str
    category_labels: tuple[str, ...]
    products: tuple[ProductNarrative, ...]
    follow_up: str
    message: str


class DeterministicResponseComposer:
    """Make T and ranking evidence visible without another model call."""

    def __init__(self, *, maximum_product_notes: int = 3) -> None:
        if type(maximum_product_notes) is not int or maximum_product_notes <= 0:
            raise ValueError("maximum_product_notes must be positive")
        self._maximum_product_notes = maximum_product_notes

    def compose(
        self,
        *,
        recommendations: tuple[str, ...],
        transparency: float,
        previous_transparency: float | None,
        ranking: ApertureRankingResult | None,
        intent: IntentState,
        product_metadata: Mapping[str, Mapping[str, object]],
    ) -> ResponseNarrative:
        _validate_transparency(transparency, name="transparency")
        if previous_transparency is not None:
            _validate_transparency(previous_transparency, name="previous_transparency")
        if type(recommendations) is not tuple or any(
            type(parent_asin) is not str or not parent_asin.strip()
            for parent_asin in recommendations
        ):
            raise TypeError("recommendations must contain non-empty product IDs")
        if type(intent) is not IntentState:
            raise TypeError("intent must be an exact IntentState")
        if not isinstance(product_metadata, Mapping):
            raise TypeError("product_metadata must be a mapping")

        band = _presentation_band(transparency)
        movement = _movement(transparency, previous_transparency)
        categories = _category_labels(recommendations, product_metadata)
        quality_hits = _quality_hits(ranking)
        products = tuple(
            _product_narrative(
                parent_asin,
                metadata=product_metadata.get(parent_asin, {}),
                quality_hit=quality_hits.get(parent_asin),
            )
            for parent_asin in recommendations[: self._maximum_product_notes]
        )
        follow_up = (
            "Which requirement can be relaxed so I can reopen the search?"
            if not recommendations
            else _follow_up(
                band=band,
                displayed_ids=recommendations[: self._maximum_product_notes],
                intent=intent,
                quality_hits=quality_hits,
            )
        )
        message = _render_message(
            recommendations=recommendations,
            band=band,
            movement=movement,
            categories=categories,
            products=products,
            follow_up=follow_up,
        )
        return ResponseNarrative(
            schema=RESPONSE_SCHEMA,
            transparency=transparency,
            previous_transparency=previous_transparency,
            presentation_band=band,
            movement=movement,
            category_labels=categories,
            products=products,
            follow_up=follow_up,
            message=message,
        )


def _presentation_band(transparency: float) -> str:
    if transparency < 0.35:
        return "broad"
    if transparency < 0.70:
        return "narrowing"
    return "focused"


def _movement(current: float, previous: float | None) -> str:
    if previous is None:
        return "initial"
    delta = current - previous
    if delta >= TRANSPARENCY_MOVEMENT_THRESHOLD:
        return "narrowed"
    if delta <= -TRANSPARENCY_MOVEMENT_THRESHOLD:
        return "broadened"
    return "stable"


def _quality_hits(
    ranking: ApertureRankingResult | None,
) -> dict[str, QualityRankingHit]:
    if ranking is None or ranking.quality_pipeline is None:
        return {}
    return {hit.parent_asin: hit for hit in ranking.quality_pipeline.quality_ranking.hits}


def _product_narrative(
    parent_asin: str,
    *,
    metadata: Mapping[str, object],
    quality_hit: QualityRankingHit | None,
) -> ProductNarrative:
    raw_title = metadata.get("title")
    title = (
        parent_asin
        if type(raw_title) is not str or not raw_title.strip()
        else _shorten(raw_title, width=100)
    )
    reason = (
        "This is one of the strongest remaining matches from the current search."
        if quality_hit is None or not quality_hit.reason
        else _sentence(_shorten(quality_hit.reason, width=220))
    )
    caveat = None
    if quality_hit is not None and quality_hit.concerns:
        candidate = _sentence(_shorten(quality_hit.concerns[0], width=160))
        if candidate.casefold().rstrip(".") not in reason.casefold().rstrip("."):
            caveat = candidate
    return ProductNarrative(
        parent_asin=parent_asin,
        title=title,
        reason=reason,
        caveat=caveat,
    )


def _render_message(
    *,
    recommendations: tuple[str, ...],
    band: str,
    movement: str,
    categories: tuple[str, ...],
    products: tuple[ProductNarrative, ...],
    follow_up: str,
) -> str:
    if not recommendations:
        return (
            f"I couldn't find a reliable match after applying the current requirements. {follow_up}"
        )

    category_phrase = _natural_list(categories)
    if movement == "narrowed":
        lead = (
            "Your latest detail narrowed the search, so I shifted from exploration "
            "toward the closest matches."
        )
    elif movement == "broadened":
        lead = "Your latest change reopened the search, so I broadened the selection"
        lead += (
            f" across {category_phrase}."
            if category_phrase
            else " across several genuinely different directions."
        )
    elif band == "broad":
        lead = "Your request is still fairly open, so I kept the selection broad"
        lead += (
            f" across {category_phrase} instead of filling it with near-duplicates."
            if category_phrase
            else " across several product directions instead of filling it with near-duplicates."
        )
    elif band == "narrowing":
        lead = (
            "I have started to narrow the search while keeping a few genuinely "
            "different options in the shortlist."
        )
    else:
        lead = (
            "Your requirements are now specific, so I prioritized the closest matches "
            "and kept the final set focused."
        )

    lines = [lead, "", "The strongest options right now are:"]
    for product in products:
        note = f"- {product.title}: {product.reason}"
        if product.caveat is not None:
            note += f" Note: {product.caveat}"
        lines.append(note)
    lines.extend(("", follow_up))
    return "\n".join(lines)


def _follow_up(
    *,
    band: str,
    displayed_ids: tuple[str, ...],
    intent: IntentState,
    quality_hits: Mapping[str, QualityRankingHit],
) -> str:
    unsupported = Counter(
        preference_id
        for parent_asin in displayed_ids
        for preference_id in (
            quality_hits[parent_asin].unsupported_preference_ids
            if parent_asin in quality_hits
            else ()
        )
    )
    preferences = {item.id: item for item in intent.preferences}
    for preference_id, count in unsupported.most_common():
        preference = preferences.get(preference_id)
        if preference is None or count < min(2, len(displayed_ids)):
            continue
        label = _preference_label(preference)
        if preference.commitment is Commitment.HARD:
            return (
                f"I could not verify {label} consistently in the product data. "
                "Should I exclude every option where that detail is not explicitly documented?"
            )
        return (
            f"I could not verify {label} consistently. "
            "Should I treat it as non-negotiable in the next pass?"
        )
    if band == "broad":
        return "Which direction should I narrow first?"
    if band == "narrowing":
        return "Which style or trade-off should I prioritize next?"
    return "Is there one remaining detail you want me to treat as non-negotiable?"


def _preference_label(preference: Preference) -> str:
    facet = None if preference.facet is None else preference.facet.replace("_", " ")
    value = _value_text(preference.value)
    if facet and value:
        if preference.operator in (Operator.NEQ, Operator.NOT_IN):
            return f"avoiding {value} for {facet}"
        return f"the requested {facet} ({value})"
    if preference.semantic_text:
        return _shorten(preference.semantic_text, width=100)
    return _shorten(preference.evidence_text, width=100)


def _value_text(value: object) -> str:
    if value is None:
        return ""
    if type(value) is tuple:
        return _natural_list(tuple(str(item) for item in value))
    return str(value)


def _category_labels(
    recommendations: tuple[str, ...],
    metadata: Mapping[str, Mapping[str, object]],
    *,
    limit: int = 4,
) -> tuple[str, ...]:
    labels: list[str] = []
    for parent_asin in recommendations:
        product = metadata.get(parent_asin, {})
        values = _flatten_strings(product.get("categories"))
        if not values:
            continue
        label = _shorten(values[-1], width=60)
        if label.casefold() in {item.casefold() for item in labels}:
            continue
        labels.append(label)
        if len(labels) == limit:
            break
    return tuple(labels)


def _flatten_strings(value: object) -> list[str]:
    if type(value) is str:
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            result.extend(_flatten_strings(item))
        return result
    return []


def _natural_list(values: tuple[str, ...]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def _shorten(value: str, *, width: int) -> str:
    return textwrap.shorten(" ".join(value.split()), width=width, placeholder="…")


def _sentence(value: str) -> str:
    return value if value.endswith((".", "!", "?")) else f"{value}."


def _validate_transparency(value: float, *, name: str) -> None:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite float in [0, 1]")
