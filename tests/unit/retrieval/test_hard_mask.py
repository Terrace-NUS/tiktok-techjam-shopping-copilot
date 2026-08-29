from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from shopping_copilot.catalog.semantic.category import (
    CategoryScope,
    ProductCategoryAssignment,
    ProductCategoryAssignmentStatus,
)
from shopping_copilot.catalog.semantic.facet import (
    USD_CENT_UNIT,
    NumericValue,
    ProductFacetStatus,
    ResolvedProductFacetValue,
)
from shopping_copilot.catalog.semantic.facet.resolution_models import RESOLUTION_POLICY_ID
from shopping_copilot.catalog.semantic.release import VerifiedCatalogSemanticRelease
from shopping_copilot.catalog.semantic.runtime import SYSTEM_PRODUCT_CATEGORY_FACET_ID
from shopping_copilot.query_compiler import (
    COMPILED_QUERY_SCHEMA,
    QUERY_COMPILER_VERSION,
    CompiledDirectives,
    CompiledHardConstraint,
    CompiledQuery,
    ConstraintPolicy,
    DiversityDirective,
)
from shopping_copilot.retrieval.dense import DenseIndex
from shopping_copilot.retrieval.errors import CompiledQueryBindingError
from shopping_copilot.retrieval.evidence import (
    RetrievalEvidenceIndex,
    build_retrieval_evidence_index,
)
from shopping_copilot.retrieval.hard_mask import (
    ConstraintDisposition,
    HardMaskResolver,
)
from shopping_copilot.retrieval.models import (
    DENSE_INDEX_SCHEMA,
    DenseArtifactRef,
    DenseIndexManifest,
    EmbeddingSpec,
)
from shopping_copilot.session_context import Operator
from shopping_copilot.session_context.models import PreferenceValue

RELEASE_ID = "sha256:" + "2" * 64
GRAPH_ID = "sha256:" + "3" * 64
SHOE_NODE_ID = "cn_" + "1" * 64
BAG_NODE_ID = "cn_" + "2" * 64
SHOE_SCOPE_ID = "cs_" + "1" * 64
BAG_SCOPE_ID = "cs_" + "2" * 64
PARENT_ASINS = ("A", "B", "C", "D", "E", "F")


def _row(
    parent_asin: str,
    *,
    color: str,
    material: str,
    category: str,
) -> dict[str, object]:
    return {
        "parent_asin": parent_asin,
        "title": f"{color} {material} product",
        "categories": [category],
        "details": {"Color": color, "Material": material},
    }


_CATALOG_ROWS = (
    _row("A", color="Red", material="Cotton", category="Shoes"),
    _row("B", color="Blue", material="Cotton", category="Shoes"),
    _row("C", color="Red", material="Wool", category="Shoes"),
    _row("D", color="Green", material="Leather", category="Bags"),
    _row("E", color="Black", material="Plastic", category="Bags"),
    _row("F", color="Blue", material="Metal", category="Bags"),
)
_CATALOG_BYTES = "".join(json.dumps(row, sort_keys=True) + "\n" for row in _CATALOG_ROWS).encode(
    "utf-8"
)
CATALOG_ID = "sha256:" + hashlib.sha256(_CATALOG_BYTES).hexdigest()


@pytest.fixture
def bound_system(
    tmp_path: Path,
) -> tuple[HardMaskResolver, DenseIndex, RetrievalEvidenceIndex]:
    catalog_path = tmp_path / "catalog.jsonl"
    catalog_path.write_bytes(_CATALOG_BYTES)
    evidence = build_retrieval_evidence_index(
        catalog_path,
        catalog_id=CATALOG_ID,
        catalog_semantic_release_id=RELEASE_ID,
        expected_parent_asins=frozenset(PARENT_ASINS),
    )
    dense = _dense_index()
    resolver = HardMaskResolver(
        release=_release(),
        evidence_index=evidence,
        dense_index=dense,
    )
    return resolver, dense, evidence


def _dense_index(
    *,
    catalog_id: str = CATALOG_ID,
    release_id: str = RELEASE_ID,
) -> DenseIndex:
    spec = EmbeddingSpec(
        backend="fake",
        backend_version="1.0",
        model_id="example/model",
        model_revision="revision",
        dimension=2,
        max_sequence_length=32,
        query_instruction="",
        document_instruction="",
        pooling="cls",
    )
    manifest = DenseIndexManifest(
        schema=DENSE_INDEX_SCHEMA,
        builder_version="dense_index_v0",
        catalog_id=catalog_id,
        catalog_semantic_release_id=release_id,
        document_template_id="product_document_v1",
        document_corpus_id="sha256:" + "4" * 64,
        product_count=len(PARENT_ASINS),
        embedding=spec,
        vector_dtype="float32",
        artifacts=(
            DenseArtifactRef(
                kind="parent_asins",
                filename="parent-asins.json",
                content_id="sha256:" + "5" * 64,
                byte_size=1,
            ),
            DenseArtifactRef(
                kind="vectors",
                filename="vectors.npy",
                content_id="sha256:" + "6" * 64,
                byte_size=1,
            ),
        ),
    )
    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.6], [0.6, 0.8], [-1.0, 0.0], [0.0, -1.0]],
        dtype=np.float32,
    )
    return DenseIndex(
        index_id="sha256:" + "7" * 64,
        manifest=manifest,
        parent_asins=PARENT_ASINS,
        vectors=vectors,
    )


def _release(
    *,
    catalog_id: str = CATALOG_ID,
    release_id: str = RELEASE_ID,
) -> VerifiedCatalogSemanticRelease:
    shoes = CategoryScope(
        id=SHOE_SCOPE_ID,
        label="Shoes",
        root_node_ids=(SHOE_NODE_ID,),
        member_node_ids=(SHOE_NODE_ID,),
    )
    bags = CategoryScope(
        id=BAG_SCOPE_ID,
        label="Bags",
        root_node_ids=(BAG_NODE_ID,),
        member_node_ids=(BAG_NODE_ID,),
    )
    assignments = tuple(
        ProductCategoryAssignment(
            parent_asin=parent_asin,
            status=ProductCategoryAssignmentStatus.KNOWN,
            leaf_node_ids=(SHOE_NODE_ID if parent_asin in {"A", "B", "C"} else BAG_NODE_ID,),
        )
        for parent_asin in PARENT_ASINS
    )
    prices = (5000, 10000, None, None, 15000, 20000)
    known_price_entries = tuple(
        _price_entry(parent_asin, price, index=index)
        for index, (parent_asin, price) in enumerate(zip(PARENT_ASINS, prices, strict=True))
        if price is not None
    )
    price_entries = tuple(
        sorted(
            (*known_price_entries, _price_conflict_entry("D")), key=lambda item: item.parent_asin
        )
    )

    release = object.__new__(VerifiedCatalogSemanticRelease)
    object.__setattr__(release, "release_id", release_id)
    object.__setattr__(release, "manifest", SimpleNamespace(catalog_id=catalog_id))
    object.__setattr__(
        release,
        "category_registry",
        SimpleNamespace(
            catalog_id=catalog_id,
            category_graph_id=GRAPH_ID,
            scopes=(bags, shoes),
        ),
    )
    object.__setattr__(
        release,
        "product_category_assignments",
        SimpleNamespace(catalog_id=catalog_id, assignments=assignments),
    )
    object.__setattr__(
        release,
        "product_facet_index",
        SimpleNamespace(catalog_id=catalog_id, entries=price_entries),
    )
    return release


def _price_entry(
    parent_asin: str,
    price: int | None,
    *,
    index: int,
) -> ResolvedProductFacetValue:
    assert price is not None
    return ResolvedProductFacetValue(
        parent_asin=parent_asin,
        facet_id="price",
        status=ProductFacetStatus.KNOWN,
        value=NumericValue(
            kind="numeric",
            lower=price,
            lower_inclusive=True,
            upper=price,
            upper_inclusive=True,
            unit=USD_CENT_UNIT,
        ),
        evidence_ids=("ev_" + format(index + 1, "x") * 64,),
        resolution_policy_id=RESOLUTION_POLICY_ID,
    )


def _price_conflict_entry(parent_asin: str) -> ResolvedProductFacetValue:
    return ResolvedProductFacetValue(
        parent_asin=parent_asin,
        facet_id="price",
        status=ProductFacetStatus.CONFLICT,
        value=None,
        evidence_ids=("ev_" + "d" * 64,),
        resolution_policy_id=RESOLUTION_POLICY_ID,
    )


def _constraint(
    preference_id: str,
    *,
    facet: str,
    operator: Operator,
    value: PreferenceValue,
) -> CompiledHardConstraint:
    if facet == SYSTEM_PRODUCT_CATEGORY_FACET_ID:
        policy = ConstraintPolicy.VERIFIED_CATEGORY
    elif facet == "price":
        policy = ConstraintPolicy.CONSERVATIVE_PRICE
    else:
        policy = ConstraintPolicy.CLOSED_WORLD_RETRIEVAL_EVIDENCE
    return CompiledHardConstraint(
        preference_id=preference_id,
        facet=facet,
        operator=operator,
        value=value,
        policy=policy,
    )


def _query(*constraints: CompiledHardConstraint) -> CompiledQuery:
    return CompiledQuery(
        schema=COMPILED_QUERY_SCHEMA,
        compiler_version=QUERY_COMPILER_VERSION,
        catalog_id=CATALOG_ID,
        catalog_semantic_release_id=RELEASE_ID,
        category_graph_id=GRAPH_ID,
        intent_version=3,
        q_lex="product",
        q_sem="Looking for a product.",
        search_ready=True,
        hard_constraints=tuple(constraints),
        ranking_preferences=(),
        dont_care_facets=(),
        directives=CompiledDirectives(
            diversity=DiversityDirective.AUTO,
            comparison_requested=False,
            explanation_requested=False,
        ),
        requires_clarification=False,
        clarification_reason=None,
        trace=(),
    )


@pytest.mark.parametrize(
    ("constraint", "expected"),
    [
        (
            _constraint(
                "category_include",
                facet=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
                operator=Operator.EQ,
                value=SHOE_SCOPE_ID,
            ),
            ("A", "B", "C"),
        ),
        (
            _constraint(
                "category_exclude",
                facet=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
                operator=Operator.NEQ,
                value=SHOE_SCOPE_ID,
            ),
            ("D", "E", "F"),
        ),
        (
            _constraint("color_include", facet="color", operator=Operator.EQ, value="red"),
            ("A", "C"),
        ),
        (
            _constraint("color_exclude", facet="color", operator=Operator.NEQ, value="red"),
            ("B", "D", "E", "F"),
        ),
    ],
)
def test_category_and_text_include_exclude(
    bound_system: tuple[HardMaskResolver, DenseIndex, RetrievalEvidenceIndex],
    constraint: CompiledHardConstraint,
    expected: tuple[str, ...],
) -> None:
    resolver, _, _ = bound_system

    result = resolver.resolve(_query(constraint))

    assert result.eligible_parent_asins == expected
    assert result.eligible_mask.eligible_count == len(expected)
    assert result.eligible_mask.values.flags.writeable is False
    assert result.trace[0].disposition is ConstraintDisposition.APPLIED


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        (Operator.LT, ("A", "C", "D")),
        (Operator.LE, ("A", "B", "C", "D")),
        (Operator.GT, ("C", "D", "F")),
        (Operator.GE, ("C", "D", "E", "F")),
    ],
)
def test_price_uses_four_interval_operators_and_keeps_unknown(
    bound_system: tuple[HardMaskResolver, DenseIndex, RetrievalEvidenceIndex],
    operator: Operator,
    expected: tuple[str, ...],
) -> None:
    resolver, _, _ = bound_system
    threshold = 10000 if operator in {Operator.LT, Operator.LE} else 15000

    result = resolver.resolve(
        _query(_constraint("price", facet="price", operator=operator, value=threshold))
    )

    assert result.eligible_parent_asins == expected
    assert {"C", "D"}.issubset(result.eligible_parent_asins)


def test_no_constraints_returns_the_full_bound_catalog(
    bound_system: tuple[HardMaskResolver, DenseIndex, RetrievalEvidenceIndex],
) -> None:
    resolver, _, _ = bound_system

    result = resolver.resolve(_query())

    assert result.eligible_parent_asins == PARENT_ASINS
    assert result.eligible_mask.eligible_count == len(PARENT_ASINS)
    assert result.hard_filter_relaxed is False
    assert result.relaxed_constraints == ()
    assert result.trace == ()


def test_values_inside_one_constraint_are_unioned(
    bound_system: tuple[HardMaskResolver, DenseIndex, RetrievalEvidenceIndex],
) -> None:
    resolver, _, _ = bound_system

    result = resolver.resolve(
        _query(
            _constraint(
                "colors",
                facet="color",
                operator=Operator.IN,
                value=("red", "green"),
            )
        )
    )

    assert result.eligible_parent_asins == ("A", "C", "D")


def test_category_values_inside_one_constraint_are_unioned(
    bound_system: tuple[HardMaskResolver, DenseIndex, RetrievalEvidenceIndex],
) -> None:
    resolver, _, _ = bound_system

    result = resolver.resolve(
        _query(
            _constraint(
                "categories",
                facet=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
                operator=Operator.IN,
                value=(SHOE_SCOPE_ID, BAG_SCOPE_ID),
            )
        )
    )

    assert result.eligible_parent_asins == PARENT_ASINS


def test_exclusions_run_first_even_if_compiled_after_an_include(
    bound_system: tuple[HardMaskResolver, DenseIndex, RetrievalEvidenceIndex],
) -> None:
    resolver, _, _ = bound_system
    category = _constraint(
        "category",
        facet=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
        operator=Operator.EQ,
        value=SHOE_SCOPE_ID,
    )
    exclude_red = _constraint(
        "exclude_red",
        facet="color",
        operator=Operator.NOT_IN,
        value=("red",),
    )

    result = resolver.resolve(_query(category, exclude_red))

    assert result.eligible_parent_asins == ("B",)
    assert tuple(item.preference_id for item in result.trace) == ("exclude_red", "category")
    assert result.trace[0].before_count == 6
    assert result.trace[0].after_count == 4
    assert result.trace[1].before_count == 4


def test_multiple_includes_relax_stably_and_continue_from_the_unchanged_set(
    bound_system: tuple[HardMaskResolver, DenseIndex, RetrievalEvidenceIndex],
) -> None:
    resolver, _, _ = bound_system
    red = _constraint("red", facet="color", operator=Operator.EQ, value="red")
    plastic = _constraint(
        "plastic",
        facet="material",
        operator=Operator.EQ,
        value="plastic",
    )
    wool = _constraint("wool", facet="material", operator=Operator.EQ, value="wool")

    result = resolver.resolve(_query(red, plastic, wool))

    assert result.eligible_parent_asins == ("C",)
    assert result.hard_filter_relaxed is True
    assert result.relaxed_constraints == (plastic,)
    assert tuple(item.disposition for item in result.trace) == (
        ConstraintDisposition.APPLIED,
        ConstraintDisposition.RELAXED_TO_RANKING,
        ConstraintDisposition.APPLIED,
    )
    assert result.trace[1].before_count == result.trace[1].after_count == 2
    assert result.trace[2].before_count == 2


def test_exclusion_that_empties_catalog_is_never_relaxed_and_skips_includes(
    bound_system: tuple[HardMaskResolver, DenseIndex, RetrievalEvidenceIndex],
) -> None:
    resolver, _, _ = bound_system
    category = _constraint(
        "category",
        facet=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
        operator=Operator.EQ,
        value=SHOE_SCOPE_ID,
    )
    exclude_every_color = _constraint(
        "exclude_all",
        facet="color",
        operator=Operator.NOT_IN,
        value=("red", "blue", "green", "black"),
    )

    result = resolver.resolve(_query(category, exclude_every_color))

    assert result.eligible_parent_asins == ()
    assert result.eligible_mask.eligible_count == 0
    assert result.hard_filter_relaxed is False
    assert result.relaxed_constraints == ()
    assert result.trace[0].disposition is ConstraintDisposition.APPLIED
    assert result.trace[1].disposition is ConstraintDisposition.SKIPPED_EMPTY_UPSTREAM
    assert result.trace[1].matched_count == 0


def test_constructor_and_query_fail_closed_on_cross_binding(
    bound_system: tuple[HardMaskResolver, DenseIndex, RetrievalEvidenceIndex],
) -> None:
    resolver, dense, evidence = bound_system
    different_release = "sha256:" + "a" * 64
    different_catalog = "sha256:" + "b" * 64

    with pytest.raises(CompiledQueryBindingError, match="different catalogs"):
        HardMaskResolver(
            release=_release(catalog_id=different_catalog),
            evidence_index=evidence,
            dense_index=dense,
        )
    with pytest.raises(CompiledQueryBindingError, match="different catalogs"):
        HardMaskResolver(
            release=_release(),
            evidence_index=replace(evidence, catalog_id=different_catalog),
            dense_index=dense,
        )
    with pytest.raises(CompiledQueryBindingError, match="different catalogs"):
        HardMaskResolver(
            release=_release(),
            evidence_index=evidence,
            dense_index=_dense_index(catalog_id=different_catalog),
        )

    with pytest.raises(CompiledQueryBindingError, match="semantic releases"):
        HardMaskResolver(
            release=_release(release_id=different_release),
            evidence_index=evidence,
            dense_index=dense,
        )
    with pytest.raises(CompiledQueryBindingError, match="semantic releases"):
        HardMaskResolver(
            release=_release(),
            evidence_index=replace(
                evidence,
                catalog_semantic_release_id=different_release,
            ),
            dense_index=dense,
        )
    with pytest.raises(CompiledQueryBindingError, match="semantic releases"):
        HardMaskResolver(
            release=_release(),
            evidence_index=evidence,
            dense_index=_dense_index(release_id=different_release),
        )

    with pytest.raises(CompiledQueryBindingError, match="different product sets"):
        HardMaskResolver(
            release=_release(),
            evidence_index=replace(evidence, parent_asins=PARENT_ASINS[:-1]),
            dense_index=dense,
        )
    incomplete_release = _release()
    incomplete_assignments = incomplete_release.product_category_assignments.assignments[:-1]
    object.__setattr__(
        incomplete_release,
        "product_category_assignments",
        SimpleNamespace(catalog_id=CATALOG_ID, assignments=incomplete_assignments),
    )
    with pytest.raises(CompiledQueryBindingError, match="different product sets"):
        HardMaskResolver(
            release=incomplete_release,
            evidence_index=evidence,
            dense_index=dense,
        )

    for query in (
        replace(_query(), catalog_id=different_catalog),
        replace(_query(), catalog_semantic_release_id=different_release),
        replace(_query(), category_graph_id="sha256:" + "c" * 64),
    ):
        with pytest.raises(CompiledQueryBindingError, match="compiled query"):
            resolver.resolve(query)


def test_evidence_failure_is_explicit_not_a_silent_empty_match(
    bound_system: tuple[HardMaskResolver, DenseIndex, RetrievalEvidenceIndex],
) -> None:
    resolver, _, _ = bound_system
    unsupported = CompiledHardConstraint(
        preference_id="unsupported",
        facet="unknown_facet",
        operator=Operator.EQ,
        value="value",
        policy=ConstraintPolicy.CLOSED_WORLD_RETRIEVAL_EVIDENCE,
    )

    with pytest.raises(ValueError, match="retrieval evidence cannot resolve"):
        resolver.resolve(_query(unsupported))


@pytest.mark.parametrize(
    "constraint",
    [
        _constraint("eq_tuple", facet="color", operator=Operator.EQ, value=("red",)),
        _constraint("in_scalar", facet="color", operator=Operator.IN, value="red"),
        _constraint("price_tuple", facet="price", operator=Operator.LE, value=(10000,)),
        _constraint("price_float", facet="price", operator=Operator.LE, value=10000.5),
    ],
)
def test_operator_value_shapes_and_integer_cent_prices_fail_closed(
    bound_system: tuple[HardMaskResolver, DenseIndex, RetrievalEvidenceIndex],
    constraint: CompiledHardConstraint,
) -> None:
    resolver, _, _ = bound_system

    with pytest.raises(ValueError, match="value tuple|scalar value|integer cent"):
        resolver.resolve(_query(constraint))


def test_constructor_rejects_an_unrecognized_evidence_policy(
    bound_system: tuple[HardMaskResolver, DenseIndex, RetrievalEvidenceIndex],
) -> None:
    _, dense, evidence = bound_system

    with pytest.raises(CompiledQueryBindingError, match="unsupported policy"):
        HardMaskResolver(
            release=_release(),
            evidence_index=replace(evidence, policy_id="different_policy"),
            dense_index=dense,
        )
