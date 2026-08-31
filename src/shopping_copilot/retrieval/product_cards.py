"""Search views derived from verified product-fact cards."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum

from shopping_copilot.catalog.product_facts import (
    ProductFact,
    ProductFactPolarity,
    VerifiedProductFactCard,
)

from .documents import DOCUMENT_FIELD_ORDER, FIELD_CHARACTER_LIMITS, ProductDocument
from .evidence import SUPPORTED_FACETS

_FACT_TEXT_LIMIT = 4_000
_DIRECT_FACETS = frozenset(SUPPORTED_FACETS)
_FACET_ALIASES = {
    "band_color": "color",
    "case_color": "color",
    "care_instruction": "feature",
    "closure": "feature",
    "closure_type": "feature",
    "country_of_origin": "feature",
    "fabric_type": "material",
    "frame_material": "material",
    "imported": "feature",
    "lens_material": "material",
    "lining_material": "material",
    "origin": "feature",
    "outsole_material": "material",
    "pattern": "style",
    "product_type": "style",
    "season": "use_case",
    "sole_material": "material",
    "upper_material": "material",
    "water_resistance": "feature",
}
_SEARCH_FACT_PRIORITY = {
    "category": 0,
    "brand": 1,
    "gender": 2,
    "department": 3,
    "material": 4,
    "color": 5,
    "size": 6,
    "style": 7,
    "feature": 8,
    "care_instruction": 9,
    "closure": 10,
    "closure_type": 10,
    "use_case": 11,
    "price": 12,
}


class ProductCardMode(str, Enum):
    """How covered products are projected into retrieval documents and evidence."""

    AUGMENT = "augment"
    REPLACE = "replace"


def project_product_documents(
    documents: Iterable[ProductDocument],
    cards: Mapping[str, VerifiedProductFactCard],
    *,
    mode: ProductCardMode,
) -> tuple[ProductDocument, ...]:
    """Project covered products while leaving every uncovered document byte-identical."""

    if type(mode) is not ProductCardMode:
        raise TypeError("mode must be a ProductCardMode")
    if not isinstance(cards, Mapping):
        raise TypeError("cards must be a mapping")
    observed: set[str] = set()
    result: list[ProductDocument] = []
    for document in documents:
        verified = cards.get(document.parent_asin)
        if verified is None:
            result.append(document)
            continue
        if verified.card.parent_asin != document.parent_asin:
            raise ValueError(f"product card ID differs from document: {document.parent_asin}")
        observed.add(document.parent_asin)
        if mode is ProductCardMode.AUGMENT:
            fields = _parse_document(document)
            prefix = _render_card(verified)
            raw_features = fields["features"]
            fields["features"] = prefix if not raw_features else f"{prefix} | {raw_features}"
            replacement = _document_from_fields(document.parent_asin, fields)
        else:
            replacement = _replacement_document(verified)
        result.append(replacement)
    unknown = sorted(set(cards) - observed)
    if unknown:
        raise KeyError(f"product cards are outside the document corpus: {unknown[0]}")
    return tuple(result)


def enrich_product_documents(
    documents: Iterable[ProductDocument],
    cards: Mapping[str, VerifiedProductFactCard],
) -> tuple[ProductDocument, ...]:
    """Prepend compact semantic facts to card-covered document feature text."""

    return project_product_documents(documents, cards, mode=ProductCardMode.AUGMENT)


def replace_product_documents(
    documents: Iterable[ProductDocument],
    cards: Mapping[str, VerifiedProductFactCard],
) -> tuple[ProductDocument, ...]:
    """Replace covered old documents with views derived only from verified new cards."""

    return project_product_documents(documents, cards, mode=ProductCardMode.REPLACE)


def product_fact_facet_overrides(
    cards: Mapping[str, VerifiedProductFactCard],
    *,
    complete: bool = False,
) -> Mapping[str, Mapping[str, tuple[str, ...]]]:
    """Project grounded present facts into deterministic hard/facet evidence."""

    if type(complete) is not bool:
        raise TypeError("complete must be a boolean")
    result: dict[str, Mapping[str, tuple[str, ...]]] = {}
    for parent_asin, verified in cards.items():
        by_facet: dict[str, list[str]] = {}
        touched: set[str] = set()
        for fact in verified.card.facts:
            facet = _retrieval_facet(fact.facet)
            if facet is None:
                continue
            touched.add(facet)
            if fact.polarity is ProductFactPolarity.ABSENT:
                continue
            values = by_facet.setdefault(facet, [])
            values.extend((fact.value, *fact.aliases, fact.meaning, fact.evidence))
        selected = set(SUPPORTED_FACETS) if complete else touched
        result[parent_asin] = {
            facet: _unique_text(by_facet.get(facet, [])) for facet in sorted(selected)
        }
    return result


def _retrieval_facet(facet: str) -> str | None:
    if facet in _DIRECT_FACETS:
        return facet
    return _FACET_ALIASES.get(facet)


def _render_card(verified: VerifiedProductFactCard) -> str:
    facts = sorted(
        verified.card.facts,
        key=lambda fact: (_SEARCH_FACT_PRIORITY.get(fact.facet, 100), fact.facet, fact.value),
    )
    parts = [f"semantic summary: {verified.card.summary}", "structured product facts:"]
    for fact in facts:
        if len(" | ".join(parts)) >= _FACT_TEXT_LIMIT:
            break
        parts.append(_render_fact(fact))
    return " | ".join(parts)[:_FACT_TEXT_LIMIT].rstrip()


def _render_fact(fact: ProductFact) -> str:
    polarity = "does not have" if fact.polarity is ProductFactPolarity.ABSENT else "has"
    component = "" if fact.component is None else f" ({fact.component})"
    aliases = "" if not fact.aliases else f" [also: {', '.join(fact.aliases)}]"
    return f"{polarity} {fact.facet}{component}: {fact.value}{aliases}"


def _replacement_document(verified: VerifiedProductFactCard) -> ProductDocument:
    facts = tuple(
        sorted(
            verified.card.facts,
            key=lambda fact: (
                _SEARCH_FACT_PRIORITY.get(fact.facet, 100),
                fact.facet,
                fact.value,
            ),
        )
    )
    title = _first_present_value(facts, {"product_name"}) or verified.card.summary
    categories = _present_values(facts, {"category", "product_type"})
    brand = _first_present_value(facts, {"brand", "manufacturer"}) or ""
    detail_parts = tuple(_render_fact(fact) for fact in facts)
    fields = {
        "title": _bounded(title, "title"),
        "categories": _bounded(" > ".join(categories), "categories"),
        "store": _bounded(brand, "store"),
        "features": _bounded(_render_card(verified), "features"),
        "details": _bounded(" | ".join(detail_parts), "details"),
        "description": _bounded(verified.card.summary, "description"),
    }
    return _document_from_fields(verified.card.parent_asin, fields)


def _document_from_fields(parent_asin: str, fields: Mapping[str, str]) -> ProductDocument:
    text = "\n".join(f"{field}: {fields[field]}" for field in DOCUMENT_FIELD_ORDER)
    return ProductDocument(parent_asin=parent_asin, text=text)


def _first_present_value(facts: tuple[ProductFact, ...], facets: set[str]) -> str | None:
    return next(
        (
            fact.value
            for fact in facts
            if fact.facet in facets and fact.polarity is ProductFactPolarity.PRESENT
        ),
        None,
    )


def _present_values(facts: tuple[ProductFact, ...], facets: set[str]) -> tuple[str, ...]:
    return _unique_text(
        fact.value
        for fact in facts
        if fact.facet in facets and fact.polarity is ProductFactPolarity.PRESENT
    )


def _bounded(value: str, field: str) -> str:
    normalized = " ".join(value.split())
    return normalized[: FIELD_CHARACTER_LIMITS[field]].rstrip()


def _parse_document(document: ProductDocument) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in document.text.splitlines():
        field, separator, value = line.partition(":")
        if not separator or field not in DOCUMENT_FIELD_ORDER or field in fields:
            raise ValueError(f"malformed ProductDocument: {document.parent_asin}")
        fields[field] = value.strip()
    if tuple(fields) != DOCUMENT_FIELD_ORDER:
        raise ValueError(f"malformed ProductDocument field order: {document.parent_asin}")
    return fields


def _unique_text(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(value.split())
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return tuple(result)
