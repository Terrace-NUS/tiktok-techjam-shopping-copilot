from __future__ import annotations

from shopping_copilot.benchmark import project_product_card_disclosures
from shopping_copilot.catalog.product_facts import (
    ProductFact,
    ProductFactCard,
    ProductFactPolarity,
)


def _fact(
    facet: str,
    value: str,
    *,
    polarity: ProductFactPolarity = ProductFactPolarity.PRESENT,
    component: str | None = None,
    source_ref: str | None = None,
) -> ProductFact:
    return ProductFact(
        facet=facet,
        value=value,
        aliases=(),
        polarity=polarity,
        component=component,
        meaning=f"Meaning of {value}",
        evidence=value,
        source_ref=source_ref or f"source_{facet}_{value}",
        confidence=1.0,
    )


def _card() -> ProductFactCard:
    return ProductFactCard(
        parent_asin="A",
        summary="A grounded running shoe.",
        facts=(
            _fact("product_type", "Running Shoes"),
            _fact("product_name", "Unique exact title"),
            _fact("average_rating", "4.8"),
            _fact("material", "100% Rubber"),
            _fact(
                "material",
                "polyester",
                polarity=ProductFactPolarity.ABSENT,
                component="upper",
            ),
            _fact("material", "Mesh", component="upper"),
            _fact("closure", "Lace-Up"),
            _fact("closure_type", "Lace-Up"),
            _fact("care_instruction", "hand wash"),
            _fact(
                "care_instruction",
                "HAND WASH IN COLD WATER / LAY FLAT TO DRY / DRY CLEAN IF NEEDED",
            ),
            _fact("color", "black"),
            _fact("style", "athletic"),
            _fact("use_case", "road running"),
            _fact("brand", "Example"),
            _fact("price", "49.99"),
        ),
    )


def test_projection_uses_grounded_facts_without_the_legacy_four_value_cap() -> None:
    plan = project_product_card_disclosures(
        _card(),
        scenario_type="buying",
        maximum_facts=10,
    )

    assert plan.product_type == "Running Shoes"
    assert 6 <= len(plan.disclosures) <= 10
    assert sum(item.commitment == "hard" for item in plan.disclosures) == 2
    assert all(item.facet not in {"product_name", "average_rating"} for item in plan.disclosures)
    assert any(item.facet == "closure" and item.value == "Lace-Up" for item in plan.disclosures)
    assert sum(item.facet == "care_instruction" for item in plan.disclosures) == 1


def test_absent_component_material_becomes_a_negative_material_preference() -> None:
    plan = project_product_card_disclosures(
        _card(),
        scenario_type="browsing",
        maximum_facts=10,
    )

    polyester = next(item for item in plan.disclosures if item.value == "polyester")
    assert polyester.ask_attribute == "material"
    assert polyester.polarity == "absent"
    assert polyester.component == "upper"
    assert polyester.utterance == "The upper should not use polyester."
    assert all(item.commitment == "soft" for item in plan.disclosures)


def test_projection_is_deterministic() -> None:
    first = project_product_card_disclosures(
        _card(),
        scenario_type="intent_override",
        maximum_facts=8,
    )
    second = project_product_card_disclosures(
        _card(),
        scenario_type="intent_override",
        maximum_facts=8,
    )

    assert first == second


def test_projection_can_backfill_a_release_minimum_without_duplicates() -> None:
    card = ProductFactCard(
        parent_asin="B",
        summary="A feature-rich watch.",
        facts=(
            _fact("product_type", "Wrist Watch"),
            _fact("material", "stainless steel"),
            _fact("feature", "alarm"),
            _fact("feature", "chronograph"),
            _fact("feature", "day indicator"),
            _fact("feature", "date indicator"),
            _fact("feature", "water resistant"),
            _fact("brand", "Example"),
            _fact("price", "47.95"),
        ),
    )

    plan = project_product_card_disclosures(
        card,
        scenario_type="buying",
        minimum_facts=6,
        maximum_facts=8,
    )

    assert 6 <= len(plan.disclosures) <= 8
    assert len({item.id for item in plan.disclosures}) == len(plan.disclosures)


def test_projection_rejects_impossible_release_minimum() -> None:
    card = ProductFactCard(
        parent_asin="C",
        summary="A sparse card.",
        facts=(
            _fact("product_type", "Scarf"),
            _fact("material", "wool"),
        ),
    )

    try:
        project_product_card_disclosures(
            card,
            scenario_type="browsing",
            minimum_facts=6,
            maximum_facts=8,
        )
    except ValueError as error:
        assert "eligible grounded facts" in str(error)
    else:
        raise AssertionError("an impossible minimum must fail")
