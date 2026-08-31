from __future__ import annotations

from shopping_copilot.catalog.product_facts import (
    ProductFact,
    ProductFactCard,
    ProductFactPolarity,
    VerifiedProductFactCard,
)
from shopping_copilot.retrieval.documents import ProductDocument
from shopping_copilot.retrieval.product_cards import (
    enrich_product_documents,
    product_fact_facet_overrides,
    replace_product_documents,
)


def _fact(
    *,
    facet: str,
    value: str,
    polarity: ProductFactPolarity = ProductFactPolarity.PRESENT,
    aliases: tuple[str, ...] = (),
) -> ProductFact:
    return ProductFact(
        facet=facet,
        value=value,
        aliases=aliases,
        polarity=polarity,
        component=None,
        meaning=f"Grounded {facet}: {value}",
        evidence=value,
        source_ref="features_0",
        confidence=1.0,
    )


def _verified(*facts: ProductFact) -> VerifiedProductFactCard:
    return VerifiedProductFactCard(
        source_id="sha256:" + "a" * 64,
        extractor_model="test",
        card=ProductFactCard(
            parent_asin="A",
            facts=facts,
            summary="A semantic shopping summary.",
        ),
    )


def _document(parent_asin: str) -> ProductDocument:
    return ProductDocument(
        parent_asin=parent_asin,
        text=(
            "title: Raw title\n"
            "categories: Shoes\n"
            "store: Store\n"
            "features: Raw feature\n"
            "details: Material: Polyester\n"
            "description: Raw description"
        ),
    )


def test_card_enrichment_changes_only_covered_documents() -> None:
    documents = (_document("A"), _document("B"))
    card = _verified(_fact(facet="material", value="cotton", aliases=("gossypium",)))

    enriched = enrich_product_documents(documents, {"A": card})

    assert "semantic summary: A semantic shopping summary." in enriched[0].text
    assert "has material: cotton [also: gossypium]" in enriched[0].text
    assert enriched[0].text.endswith("description: Raw description")
    assert enriched[1] == documents[1]


def test_card_replacement_uses_no_old_document_text_and_keeps_uncovered_bytes() -> None:
    documents = (_document("A"), _document("B"))
    card = _verified(
        _fact(facet="product_name", value="New grounded title"),
        _fact(facet="category", value="Walking shoes"),
        _fact(facet="brand", value="New Brand"),
        _fact(facet="material", value="cotton"),
    )

    replaced = replace_product_documents(documents, {"A": card})

    assert replaced[0].text.startswith("title: New grounded title\n")
    assert "semantic summary: A semantic shopping summary." in replaced[0].text
    assert "Raw title" not in replaced[0].text
    assert "Raw feature" not in replaced[0].text
    assert "Polyester" not in replaced[0].text
    assert "Raw description" not in replaced[0].text
    assert replaced[1] == documents[1]


def test_fact_projection_replaces_touched_facets_and_does_not_assert_absence() -> None:
    cards = {
        "A": _verified(
            _fact(facet="material", value="cotton", aliases=("gossypium",)),
            _fact(
                facet="material",
                value="polyester",
                polarity=ProductFactPolarity.ABSENT,
            ),
            _fact(facet="closure_type", value="button closure", aliases=("button fastening",)),
        )
    }

    overrides = product_fact_facet_overrides(cards)

    assert "cotton" in overrides["A"]["material"]
    assert "gossypium" in overrides["A"]["material"]
    assert "polyester" not in overrides["A"]["material"]
    assert "button fastening" in overrides["A"]["feature"]

    complete = product_fact_facet_overrides(cards, complete=True)
    assert set(complete["A"]) == {
        "brand",
        "color",
        "department",
        "feature",
        "gender",
        "material",
        "size",
        "style",
        "use_case",
    }
    assert complete["A"]["brand"] == ()
