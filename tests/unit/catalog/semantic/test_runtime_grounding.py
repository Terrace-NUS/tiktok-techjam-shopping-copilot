from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from test_runtime_projection import _build_runtime, _build_runtime_bundle

from shopping_copilot.catalog.semantic import IJSON_SAFE_INTEGER_MAX, content_id_for_value
from shopping_copilot.catalog.semantic.category import decode_category_registry
from shopping_copilot.catalog.semantic.facet import RuntimePromotionDecision
from shopping_copilot.catalog.semantic.runtime import (
    SYSTEM_PRODUCT_CATEGORY_FACET_ID,
    ExtractedRuntimeValueCandidate,
    GroundedPredicate,
    GroundingDisposition,
    RuntimeValueGrounder,
    load_runtime_value_grounder,
)
from shopping_copilot.session_context import Operator, SemanticPolarity


def _candidate(
    *,
    facet_id: str | None = "price",
    operator: Operator | str | None = Operator.LE,
    value: object = 2500,
    alternative_values: tuple[object, ...] = (),
    semantic_text: str = "不超过 25 美元",
) -> ExtractedRuntimeValueCandidate:
    return ExtractedRuntimeValueCandidate(
        facet_id=facet_id,
        operator=operator,
        value=value,  # type: ignore[arg-type]
        alternative_values=alternative_values,  # type: ignore[arg-type]
        semantic_text=semantic_text,
        semantic_polarity=SemanticPolarity.POSITIVE,
    )


def _grounder(tmp_path: Path):
    runtime, _, registry, _, gate_b = _build_runtime(tmp_path)
    return (
        RuntimeValueGrounder(
            runtime_registry=runtime.runtime_registry,
            runtime_lexicon=runtime.runtime_lexicon,
            category_registry=registry,
            capabilities=gate_b.capabilities,
        ),
        registry,
    )


def test_price_candidate_grounds_without_allocating_session_fields(tmp_path: Path) -> None:
    grounder, registry = _grounder(tmp_path)
    result = grounder.ground(
        _candidate(),
        final_category_scope_id=registry.scopes[0].id,
    )
    assert result.disposition is GroundingDisposition.GROUNDED
    assert result.predicates == (
        GroundedPredicate(facet_id="price", operator=Operator.LE, value=2500),
    )
    assert result.reason_code is None
    assert result.candidate_values == ()
    assert result.semantic_text is None
    assert not hasattr(result, "preference_id")
    assert not hasattr(result, "id")


def test_numeric_equality_expands_to_canonical_inclusive_bounds(tmp_path: Path) -> None:
    grounder, registry = _grounder(tmp_path)
    result = grounder.ground(
        _candidate(operator="eq", value=2500, semantic_text="正好 25 美元"),
        final_category_scope_id=registry.scopes[0].id,
    )
    assert result.predicates == (
        GroundedPredicate(facet_id="price", operator=Operator.GE, value=2500),
        GroundedPredicate(facet_id="price", operator=Operator.LE, value=2500),
    )


def test_signed_safe_integer_price_follows_the_frozen_normalizer_contract(tmp_path: Path) -> None:
    grounder, registry = _grounder(tmp_path)
    result = grounder.ground(
        _candidate(value=-1, semantic_text="价格至少负一美分"),
        final_category_scope_id=registry.scopes[0].id,
    )
    assert result.predicates == (
        GroundedPredicate(facet_id="price", operator=Operator.LE, value=-1),
    )


@pytest.mark.parametrize(
    "value",
    ["2500", 25.0, -0.0, True, (), (2500,), IJSON_SAFE_INTEGER_MAX + 1],
)
def test_invalid_price_value_stays_semantic_only(tmp_path: Path, value: object) -> None:
    grounder, registry = _grounder(tmp_path)
    result = grounder.ground(
        _candidate(value=value),
        final_category_scope_id=registry.scopes[0].id,
    )
    assert result.disposition is GroundingDisposition.SEMANTIC_ONLY
    assert result.facet_id == "price"
    assert result.reason_code == "unknown_value"
    assert result.predicates == ()
    assert result.semantic_text == "不超过 25 美元"


def test_unknown_facet_does_not_invent_a_structured_name(tmp_path: Path) -> None:
    grounder, registry = _grounder(tmp_path)
    result = grounder.ground(
        _candidate(facet_id="budget", semantic_text="便宜一点"),
        final_category_scope_id=registry.scopes[0].id,
    )
    assert result.disposition is GroundingDisposition.SEMANTIC_ONLY
    assert result.facet_id is None
    assert result.reason_code == "unknown_facet"


@pytest.mark.parametrize("operator", [None, "contains", "LE", Operator.NEQ])
def test_unsupported_price_operator_is_explicit(
    tmp_path: Path,
    operator: Operator | str | None,
) -> None:
    grounder, registry = _grounder(tmp_path)
    result = grounder.ground(
        _candidate(operator=operator),
        final_category_scope_id=registry.scopes[0].id,
    )
    assert result.disposition is GroundingDisposition.SEMANTIC_ONLY
    assert result.reason_code == "unsupported_operator"


def test_unregistered_final_category_scope_fails_before_capability_lookup(tmp_path: Path) -> None:
    grounder, _ = _grounder(tmp_path)
    result = grounder.ground(
        _candidate(),
        final_category_scope_id="cs_" + "0" * 64,
    )
    assert result.disposition is GroundingDisposition.SEMANTIC_ONLY
    assert result.reason_code == "unregistered_category_scope"


def test_noncommittable_exact_scope_never_produces_a_predicate(tmp_path: Path) -> None:
    runtime, _, registry, _, gate_b = _build_runtime(tmp_path)
    entries = tuple(
        replace(
            item,
            decision=RuntimePromotionDecision.SEARCH_ONLY,
            intent_committable=False,
            retrieval_eligible=True,
            probe_eligible=True,
            clarification_eligible=False,
        )
        for item in gate_b.capabilities.entries
    )
    capabilities = replace(gate_b.capabilities, entries=entries)
    runtime_registry = replace(
        runtime.runtime_registry,
        effective_capabilities_id=content_id_for_value(capabilities),
    )
    runtime_lexicon = replace(
        runtime.runtime_lexicon,
        runtime_registry_id=content_id_for_value(runtime_registry),
    )
    grounder = RuntimeValueGrounder(
        runtime_registry=runtime_registry,
        runtime_lexicon=runtime_lexicon,
        category_registry=registry,
        capabilities=capabilities,
    )
    result = grounder.ground(
        _candidate(),
        final_category_scope_id=registry.scopes[0].id,
    )
    assert result.disposition is GroundingDisposition.SEMANTIC_ONLY
    assert result.reason_code == "facet_not_committable"
    assert result.predicates == ()


def test_ambiguity_is_deduplicated_sorted_and_discovery_order_independent(tmp_path: Path) -> None:
    grounder, registry = _grounder(tmp_path)
    first = grounder.ground(
        _candidate(
            value=None,
            alternative_values=(3000, 1000, 3000),
            semantic_text="大概 10 或 30 美元",
        ),
        final_category_scope_id=registry.scopes[0].id,
    )
    second = grounder.ground(
        _candidate(
            value=None,
            alternative_values=(1000, 3000),
            semantic_text="大概 10 或 30 美元",
        ),
        final_category_scope_id=registry.scopes[0].id,
    )
    assert first == second
    assert first.disposition is GroundingDisposition.AMBIGUOUS
    assert first.reason_code == "ambiguous_value"
    assert first.candidate_values == (1000, 3000)
    assert first.predicates == ()


def test_one_invalid_alternative_fails_closed_instead_of_being_dropped(tmp_path: Path) -> None:
    grounder, registry = _grounder(tmp_path)
    result = grounder.ground(
        _candidate(value=None, alternative_values=(1000, "thirty dollars")),
        final_category_scope_id=registry.scopes[0].id,
    )
    assert result.disposition is GroundingDisposition.SEMANTIC_ONLY
    assert result.reason_code == "unknown_value"


def test_reserved_category_grounds_only_published_scope_with_eq(tmp_path: Path) -> None:
    grounder, registry = _grounder(tmp_path)
    scope_id = registry.scopes[0].id
    grounded = grounder.ground(
        _candidate(
            facet_id=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
            operator=Operator.EQ,
            value=scope_id,
            semantic_text="找服饰鞋包",
        ),
        final_category_scope_id=None,
    )
    assert grounded.predicates == (
        GroundedPredicate(
            facet_id=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
            operator=Operator.EQ,
            value=scope_id,
        ),
    )

    unsupported = grounder.ground(
        _candidate(
            facet_id=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
            operator=Operator.IN,
            value=(scope_id,),
            semantic_text="找这些类别",
        ),
        final_category_scope_id=None,
    )
    assert unsupported.reason_code == "unsupported_operator"

    unregistered = grounder.ground(
        _candidate(
            facet_id=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
            operator=Operator.EQ,
            value="cs_" + "0" * 64,
            semantic_text="找未知类别",
        ),
        final_category_scope_id=None,
    )
    assert unregistered.reason_code == "unregistered_category_scope"


def test_verified_bundle_loader_constructs_the_same_grounding_boundary(tmp_path: Path) -> None:
    _, approved, output = _build_runtime_bundle(tmp_path)
    grounder = load_runtime_value_grounder(
        output,
        category_candidate_dir=approved[3],
        gate_b_candidate_dir=approved[9],
    )
    registry = decode_category_registry((approved[3] / "category-registry.json").read_bytes())
    result = grounder.ground(
        _candidate(),
        final_category_scope_id=registry.scopes[0].id,
    )
    assert result.disposition is GroundingDisposition.GROUNDED
