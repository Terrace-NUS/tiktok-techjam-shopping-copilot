from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from shopping_copilot.application.quality_ranking import RealWorldRankingResult
from shopping_copilot.application.response_generation import DeterministicResponseComposer
from shopping_copilot.retrieval.deepseek_ranking import (
    CandidateVerdict,
    QualityRankingHit,
)
from shopping_copilot.session_context import (
    Commitment,
    IntentState,
    Operator,
    Preference,
    PreferenceSource,
)


def test_broad_response_exposes_category_breadth_and_ranking_reasons() -> None:
    composer = DeterministicResponseComposer()
    ranking = _ranking(
        _hit("boot", reason="Warm waterproof footwear for snow."),
        _hit("glove", reason="Insulated hand protection."),
        _hit("coat", reason="A packable layer for cold weather."),
    )

    narrative = composer.compose(
        recommendations=("boot", "glove", "coat"),
        transparency=0.10,
        previous_transparency=None,
        ranking=ranking,
        intent=_intent(),
        product_metadata=_metadata(),
    )

    assert narrative.presentation_band == "broad"
    assert narrative.movement == "initial"
    assert narrative.category_labels == ("Snow Boots", "Gloves", "Down Jackets")
    assert "kept the selection broad" in narrative.message
    assert "instead of filling it with near-duplicates" in narrative.message
    assert "Warm waterproof footwear for snow." in narrative.message
    assert narrative.follow_up == "Which direction should I narrow first?"
    assert narrative.message.endswith(narrative.follow_up)


def test_transparency_change_is_visible_as_narrowing_or_broadening() -> None:
    composer = DeterministicResponseComposer()
    common = {
        "recommendations": ("boot", "glove", "coat"),
        "ranking": _ranking(_hit("boot"), _hit("glove"), _hit("coat")),
        "intent": _intent(),
        "product_metadata": _metadata(),
    }

    narrowed = composer.compose(
        transparency=0.80,
        previous_transparency=0.40,
        **common,
    )
    broadened = composer.compose(
        transparency=0.20,
        previous_transparency=0.75,
        **common,
    )

    assert narrowed.movement == "narrowed"
    assert "latest detail narrowed the search" in narrowed.message
    assert broadened.movement == "broadened"
    assert "latest change reopened the search" in broadened.message
    assert "Snow Boots, Gloves, and Down Jackets" in broadened.message


def test_repeated_unsupported_preference_becomes_evidence_bound_follow_up() -> None:
    composer = DeterministicResponseComposer()
    intent = _intent(include_size=True)
    ranking = _ranking(
        _hit("boot", unsupported=("p_size",)),
        _hit("glove", unsupported=("p_size",)),
        _hit("coat"),
    )

    narrative = composer.compose(
        recommendations=("boot", "glove", "coat"),
        transparency=0.82,
        previous_transparency=0.80,
        ranking=ranking,
        intent=intent,
        product_metadata=_metadata(),
    )

    assert narrative.presentation_band == "focused"
    assert narrative.follow_up == (
        "I could not verify the requested size (10) consistently in the product data. "
        "Should I exclude every option where that detail is not explicitly documented?"
    )
    assert narrative.message.endswith(narrative.follow_up)


def test_no_results_has_matching_message_and_recorded_question() -> None:
    narrative = DeterministicResponseComposer().compose(
        recommendations=(),
        transparency=0.90,
        previous_transparency=0.80,
        ranking=None,
        intent=_intent(),
        product_metadata={},
    )

    assert narrative.products == ()
    assert narrative.follow_up == "Which requirement can be relaxed so I can reopen the search?"
    assert narrative.message.endswith(narrative.follow_up)


def _ranking(*hits: QualityRankingHit) -> RealWorldRankingResult:
    quality_pipeline = SimpleNamespace(
        quality_ranking=SimpleNamespace(
            hits=hits,
            traces=(),
        )
    )
    return RealWorldRankingResult(
        mode="deepseek_quality_dpp",
        recommendations=tuple(item.parent_asin for item in hits),
        quality_pipeline=cast(Any, quality_pipeline),
        quality_slate=None,
        fallback_cross_encoder=None,
        fallback_slate=None,
        formal_mmr_fallback_hits=(),
        quality_failure=None,
        fallback_failure=None,
    )


def _hit(
    parent_asin: str,
    *,
    reason: str = "A strong current match.",
    unsupported: tuple[str, ...] = (),
) -> QualityRankingHit:
    return QualityRankingHit(
        parent_asin=parent_asin,
        rank=1,
        shortlist_rank=1,
        bge_relevance=0.8,
        deepseek_fit=0.8,
        quality=0.8,
        verdict=CandidateVerdict.STRONG_MATCH,
        matched_preference_ids=(),
        unsupported_preference_ids=unsupported,
        conflict_preference_ids=(),
        concerns=(),
        reason=reason,
    )


def _intent(*, include_size: bool = False) -> IntentState:
    preferences = ()
    if include_size:
        preferences = (
            Preference(
                id="p_size",
                facet="size",
                operator=Operator.EQ,
                value=10,
                semantic_text=None,
                semantic_polarity=None,
                commitment=Commitment.HARD,
                source=PreferenceSource.USER_EXPLICIT,
                source_turn=1,
                evidence_text="size 10",
                interpretation_confidence=1.0,
            ),
        )
    return IntentState(
        goal="winter travel",
        preferences=preferences,
        dont_care_facets=frozenset(),
        version=1,
    )


def _metadata() -> dict[str, dict[str, object]]:
    return {
        "boot": {"title": "Black snow boot", "categories": ["Shoes", "Snow Boots"]},
        "glove": {"title": "Winter glove", "categories": ["Accessories", "Gloves"]},
        "coat": {"title": "Packable down coat", "categories": ["Coats", "Down Jackets"]},
    }
