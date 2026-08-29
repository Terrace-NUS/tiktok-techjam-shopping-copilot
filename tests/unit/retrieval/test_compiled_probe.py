from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from shopping_copilot.catalog.semantic.category import (
    CATEGORY_REGISTRY_SCHEMA,
    CategoryNode,
    CategoryRegistry,
    CategoryScope,
)
from shopping_copilot.query_compiler import QueryCompiler
from shopping_copilot.query_understanding import (
    BehavioralDirectives,
    ClarificationNeed,
    DiversityMode,
    ResolvedTurnIntent,
    UnderstandingTrace,
)
from shopping_copilot.retrieval import (
    CompiledDenseProbeRunner,
    CompiledQueryBindingError,
    CompiledQueryNotSearchableError,
    DenseIndex,
    DenseIndexManifest,
    DenseRetriever,
    EmbeddingSpec,
)
from shopping_copilot.retrieval.models import DENSE_INDEX_SCHEMA, DenseArtifactRef
from shopping_copilot.session_context import IntentState

CATALOG_ID = "sha256:" + "1" * 64
CATEGORY_GRAPH_ID = "sha256:" + "2" * 64
RELEASE_ID = "sha256:" + "3" * 64
ROOT_SCOPE_ID = "cs_" + "1" * 64
ROOT_NODE_ID = "cn_" + "1" * 64


def _registry() -> CategoryRegistry:
    return CategoryRegistry(
        schema=CATEGORY_REGISTRY_SCHEMA,
        catalog_id=CATALOG_ID,
        category_graph_id=CATEGORY_GRAPH_ID,
        root_scope_id=ROOT_SCOPE_ID,
        nodes=(CategoryNode(id=ROOT_NODE_ID, parent_id=None, canonical_path=("All",)),),
        scopes=(
            CategoryScope(
                id=ROOT_SCOPE_ID,
                label="All products",
                root_node_ids=(ROOT_NODE_ID,),
                member_node_ids=(ROOT_NODE_ID,),
            ),
        ),
    )


def _resolved(goal: str | None) -> ResolvedTurnIntent:
    return ResolvedTurnIntent(
        update=None,
        final_intent=IntentState(
            goal=goal,
            preferences=(),
            dont_care_facets=frozenset(),
            version=1,
        ),
        feedback=(),
        directives=BehavioralDirectives(
            diversity=DiversityMode.AUTO,
            comparison_requested=False,
            explanation_requested=False,
        ),
        clarification=ClarificationNeed(needed=False, reason=None, alternatives=()),
        trace=UnderstandingTrace(
            attempts=(),
            interpretation_summary="probe test",
            semantic_fallback_facets=(),
        ),
    )


def _spec() -> EmbeddingSpec:
    return EmbeddingSpec(
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


def _index() -> DenseIndex:
    manifest = DenseIndexManifest(
        schema=DENSE_INDEX_SCHEMA,
        builder_version="dense_index_v0",
        catalog_id=CATALOG_ID,
        catalog_semantic_release_id=RELEASE_ID,
        document_template_id="product_document_v1",
        document_corpus_id="sha256:" + "4" * 64,
        product_count=3,
        embedding=_spec(),
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
    return DenseIndex(
        index_id="sha256:" + "7" * 64,
        manifest=manifest,
        parent_asins=("A", "B", "C"),
        vectors=np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32),
    )


class _FakeEmbedder:
    def __init__(self) -> None:
        self.seen_queries: list[str] = []

    @property
    def spec(self) -> EmbeddingSpec:
        return _spec()

    def encode_documents(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        raise AssertionError("probe must not encode documents")

    def encode_query(self, text: str) -> np.ndarray:
        self.seen_queries.append(text)
        return np.array([1.0, 0.0], dtype=np.float32)


def _compiler() -> QueryCompiler:
    return QueryCompiler(
        catalog_semantic_release_id=RELEASE_ID,
        category_registry=_registry(),
    )


def test_runner_consumes_q_sem_once_and_applies_bound_mask_before_fixed_top_k() -> None:
    index = _index()
    embedder = _FakeEmbedder()
    runner = CompiledDenseProbeRunner(
        retriever=DenseRetriever(index=index, embedder=embedder),
        probe_k=2,
    )
    query = _compiler().compile(_resolved("walking shoes"))

    result = runner.run(query, eligible_mask=index.make_eligibility_mask({"B"}))

    assert runner.probe_k == 2
    assert embedder.seen_queries == ["Looking for walking shoes."]
    assert [hit.parent_asin for hit in result.observation.hits] == ["B"]
    assert result.ranking.eligible_mask is not None
    assert result.observation.coherence.reason == "insufficient_candidates"


def test_runner_rejects_empty_or_differently_bound_compiled_queries() -> None:
    index = _index()
    embedder = _FakeEmbedder()
    runner = CompiledDenseProbeRunner(
        retriever=DenseRetriever(index=index, embedder=embedder),
        probe_k=2,
    )

    with pytest.raises(CompiledQueryNotSearchableError, match="no searchable intent"):
        runner.run(_compiler().compile(_resolved(None)))
    assert embedder.seen_queries == []

    query = _compiler().compile(_resolved("walking shoes"))
    with pytest.raises(CompiledQueryBindingError, match="different catalog bindings"):
        runner.run(replace(query, catalog_semantic_release_id="sha256:" + "f" * 64))
    assert embedder.seen_queries == []
