from __future__ import annotations

import pytest

from shopping_copilot.catalog.semantic.category import (
    CATEGORY_REGISTRY_SCHEMA,
    CategoryNode,
    CategoryRegistry,
    CategoryScope,
)
from shopping_copilot.catalog.semantic.runtime import SYSTEM_PRODUCT_CATEGORY_FACET_ID
from shopping_copilot.query_compiler import (
    CompilationTarget,
    ConstraintPolicy,
    DiversityDirective,
    QueryCompiler,
    QueryCompilerError,
    RankingReason,
)
from shopping_copilot.query_understanding import (
    BehavioralDirectives,
    ClarificationNeed,
    DiversityMode,
    ResolvedTurnIntent,
    UnderstandingTrace,
)
from shopping_copilot.session_context import (
    Commitment,
    IntentState,
    Operator,
    Preference,
    PreferenceSource,
    SemanticPolarity,
)

CATALOG_ID = "sha256:" + "1" * 64
CATEGORY_GRAPH_ID = "sha256:" + "2" * 64
RELEASE_ID = "sha256:" + "3" * 64
ROOT_SCOPE_ID = "cs_" + "1" * 64
FOOTWEAR_SCOPE_ID = "cs_" + "2" * 64
ROOT_NODE_ID = "cn_" + "1" * 64
FOOTWEAR_NODE_ID = "cn_" + "2" * 64


def _category_registry() -> CategoryRegistry:
    return CategoryRegistry(
        schema=CATEGORY_REGISTRY_SCHEMA,
        catalog_id=CATALOG_ID,
        category_graph_id=CATEGORY_GRAPH_ID,
        root_scope_id=ROOT_SCOPE_ID,
        nodes=(
            CategoryNode(
                id=ROOT_NODE_ID,
                parent_id=None,
                canonical_path=("All",),
            ),
            CategoryNode(
                id=FOOTWEAR_NODE_ID,
                parent_id=ROOT_NODE_ID,
                canonical_path=("All", "Footwear"),
            ),
        ),
        scopes=(
            CategoryScope(
                id=ROOT_SCOPE_ID,
                label="All products",
                root_node_ids=(ROOT_NODE_ID,),
                member_node_ids=(FOOTWEAR_NODE_ID, ROOT_NODE_ID),
            ),
            CategoryScope(
                id=FOOTWEAR_SCOPE_ID,
                label="Footwear",
                root_node_ids=(FOOTWEAR_NODE_ID,),
                member_node_ids=(FOOTWEAR_NODE_ID,),
            ),
        ),
    )


def _preference(
    preference_id: str,
    *,
    facet: str | None,
    operator: Operator | None = None,
    value: str | int | tuple[str, ...] | None = None,
    semantic_text: str | None = None,
    semantic_polarity: SemanticPolarity | None = None,
    commitment: Commitment = Commitment.HARD,
    source: PreferenceSource = PreferenceSource.USER_EXPLICIT,
) -> Preference:
    return Preference(
        id=preference_id,
        facet=facet,
        operator=operator,
        value=value,
        semantic_text=semantic_text,
        semantic_polarity=semantic_polarity,
        commitment=commitment,
        source=source,
        source_turn=1,
        evidence_text="user evidence",
        interpretation_confidence=1.0,
    )


def _resolved(
    *preferences: Preference,
    goal: str | None = "commuting shoes",
    clarification: ClarificationNeed | None = None,
) -> ResolvedTurnIntent:
    return ResolvedTurnIntent(
        update=None,
        final_intent=IntentState(
            goal=goal,
            preferences=preferences,
            dont_care_facets=frozenset({"brand"}),
            version=4,
        ),
        feedback=(),
        directives=BehavioralDirectives(
            diversity=DiversityMode.INCREASE,
            comparison_requested=True,
            explanation_requested=False,
        ),
        clarification=clarification
        or ClarificationNeed(needed=False, reason=None, alternatives=()),
        trace=UnderstandingTrace(
            attempts=(),
            interpretation_summary="compiled test intent",
            semantic_fallback_facets=(),
        ),
    )


def test_compiler_produces_lexical_semantic_constraint_and_ranking_views() -> None:
    resolved = _resolved(
        _preference(
            "p_1_0_0",
            facet=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
            operator=Operator.EQ,
            value=FOOTWEAR_SCOPE_ID,
        ),
        _preference(
            "p_1_1_0",
            facet="color",
            operator=Operator.IN,
            value=("red", "blue"),
        ),
        _preference(
            "p_1_2_0",
            facet="material",
            operator=Operator.NOT_IN,
            value=("plastic",),
        ),
        _preference(
            "p_1_3_0",
            facet="price",
            operator=Operator.LE,
            value=12500,
        ),
        _preference(
            "p_1_4_0",
            facet="style",
            operator=Operator.EQ,
            value="minimal",
            commitment=Commitment.SOFT,
            source=PreferenceSource.SYSTEM_INFERRED,
        ),
        _preference(
            "p_1_5_0",
            facet=None,
            semantic_text="does not look cheap",
            semantic_polarity=SemanticPolarity.NEGATIVE,
        ),
    )

    compiled = QueryCompiler(
        catalog_semantic_release_id=RELEASE_ID,
        category_registry=_category_registry(),
    ).compile(resolved)

    assert compiled.search_ready is True
    assert compiled.intent_version == 4
    assert compiled.q_lex == "commuting shoes Footwear red blue minimal"
    assert "Exclude material: plastic." in compiled.q_sem
    assert "Required price: at most USD 125.00." in compiled.q_sem
    assert "Avoid: does not look cheap." in compiled.q_sem
    assert [item.facet for item in compiled.hard_constraints] == [
        SYSTEM_PRODUCT_CATEGORY_FACET_ID,
        "color",
        "material",
        "price",
    ]
    assert [item.policy for item in compiled.hard_constraints] == [
        ConstraintPolicy.VERIFIED_CATEGORY,
        ConstraintPolicy.CLOSED_WORLD_RETRIEVAL_EVIDENCE,
        ConstraintPolicy.CLOSED_WORLD_RETRIEVAL_EVIDENCE,
        ConstraintPolicy.CONSERVATIVE_PRICE,
    ]
    assert [item.preference_id for item in compiled.ranking_preferences] == [
        "p_1_4_0",
        "p_1_5_0",
    ]
    assert compiled.ranking_preferences[0].reason is RankingReason.SOFT_COMMITMENT
    assert compiled.ranking_preferences[1].reason is RankingReason.SEMANTIC_ONLY
    assert compiled.ranking_preferences[1].commitment is Commitment.HARD
    assert compiled.dont_care_facets == ("brand",)
    assert compiled.directives.diversity is DiversityDirective.INCREASE
    assert compiled.directives.comparison_requested is True
    assert len(compiled.trace) == len(resolved.final_intent.preferences)
    assert compiled.trace[1].targets == (
        CompilationTarget.Q_SEM,
        CompilationTarget.Q_LEX,
        CompilationTarget.HARD_CONSTRAINT,
    )
    assert (
        QueryCompiler(
            catalog_semantic_release_id=RELEASE_ID,
            category_registry=_category_registry(),
        ).compile(resolved)
        == compiled
    )


def test_root_category_is_an_explainable_noop_and_empty_intent_is_not_searchable() -> None:
    root = _preference(
        "p_1_0_0",
        facet=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
        operator=Operator.EQ,
        value=ROOT_SCOPE_ID,
    )
    clarification = ClarificationNeed(
        needed=True,
        reason="No product need yet",
        alternatives=("shoes", "bags"),
    )

    compiled = QueryCompiler(
        catalog_semantic_release_id=RELEASE_ID,
        category_registry=_category_registry(),
    ).compile(_resolved(root, goal=None, clarification=clarification))

    assert compiled.q_lex == ""
    assert compiled.q_sem == ""
    assert compiled.search_ready is False
    assert compiled.hard_constraints == ()
    assert compiled.ranking_preferences == ()
    assert compiled.trace[0].targets == (CompilationTarget.NOOP,)
    assert compiled.requires_clarification is True
    assert compiled.clarification_reason == "No product need yet"


def test_category_scope_must_belong_to_the_active_registry() -> None:
    unknown_scope = "cs_" + "f" * 64
    preference = _preference(
        "p_1_0_0",
        facet=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
        operator=Operator.EQ,
        value=unknown_scope,
    )

    with pytest.raises(QueryCompilerError, match="active category registry"):
        QueryCompiler(
            catalog_semantic_release_id=RELEASE_ID,
            category_registry=_category_registry(),
        ).compile(_resolved(preference))


def test_category_multi_value_constraints_compile_all_scope_labels_and_polarity() -> None:
    include = _preference(
        "p_1_0_0",
        facet=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
        operator=Operator.IN,
        value=(FOOTWEAR_SCOPE_ID, ROOT_SCOPE_ID),
    )
    exclude = _preference(
        "p_1_1_0",
        facet=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
        operator=Operator.NOT_IN,
        value=(FOOTWEAR_SCOPE_ID,),
    )

    compiled = QueryCompiler(
        catalog_semantic_release_id=RELEASE_ID,
        category_registry=_category_registry(),
    ).compile(_resolved(include, exclude))

    assert compiled.q_lex == "commuting shoes Footwear All products"
    assert "Required category: Footwear, All products." in compiled.q_sem
    assert "Exclude category: Footwear." in compiled.q_sem
    assert [item.value for item in compiled.hard_constraints] == [
        (FOOTWEAR_SCOPE_ID, ROOT_SCOPE_ID),
        (FOOTWEAR_SCOPE_ID,),
    ]
