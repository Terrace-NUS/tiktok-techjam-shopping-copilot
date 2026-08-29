from __future__ import annotations

import numpy as np
import pytest

from shopping_copilot.retrieval import (
    DenseIndex,
    DenseIndexManifest,
    EmbeddingSpec,
    VectorCandidate,
    VectorDiversityPolicy,
    VectorMMRReranker,
)
from shopping_copilot.retrieval.models import DENSE_INDEX_SCHEMA, DenseArtifactRef


def _index() -> DenseIndex:
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
        catalog_id="sha256:" + "1" * 64,
        catalog_semantic_release_id="sha256:" + "2" * 64,
        document_template_id="product_document_v1",
        document_corpus_id="sha256:" + "3" * 64,
        product_count=4,
        embedding=spec,
        vector_dtype="float32",
        artifacts=(
            DenseArtifactRef(
                kind="parent_asins",
                filename="parent-asins.json",
                content_id="sha256:" + "4" * 64,
                byte_size=1,
            ),
            DenseArtifactRef(
                kind="vectors",
                filename="vectors.npy",
                content_id="sha256:" + "5" * 64,
                byte_size=1,
            ),
        ),
    )
    vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.99995, 0.01],
            [0.8, 0.6],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return DenseIndex(
        index_id="sha256:" + "a" * 64,
        manifest=manifest,
        parent_asins=("A", "B", "C", "D"),
        vectors=vectors,
    )


def test_low_relevance_weight_prefers_a_distinct_vector() -> None:
    index = _index()
    scores = index.score_vector(np.asarray([1.0, 0.0], dtype=np.float32))
    reranker = VectorMMRReranker(index=index)

    diverse = reranker.rerank(
        scores,
        candidate_k=4,
        top_k=2,
        relevance_weight=0.30,
    )
    focused = reranker.rerank(
        scores,
        candidate_k=4,
        top_k=2,
        relevance_weight=0.90,
    )

    assert [hit.parent_asin for hit in diverse.hits] == ["A", "D"]
    assert [hit.parent_asin for hit in focused.hits] == ["A", "B"]
    assert diverse.hits[0].maximum_similarity_to_selected == 0.0
    assert diverse.hits[1].maximum_similarity_to_selected == pytest.approx(0.0)


def test_candidate_window_is_a_relevance_boundary() -> None:
    index = _index()
    scores = index.score_vector(np.asarray([1.0, 0.0], dtype=np.float32))

    result = VectorMMRReranker(index=index).rerank(
        scores,
        candidate_k=3,
        top_k=2,
        relevance_weight=0.10,
    )

    assert "D" not in {hit.parent_asin for hit in result.hits}
    assert [hit.parent_asin for hit in result.hits] == ["A", "C"]


def test_mask_is_applied_before_candidate_truncation() -> None:
    index = _index()
    scores = index.score_vector(np.asarray([1.0, 0.0], dtype=np.float32))
    mask = index.make_eligibility_mask({"C", "D"})

    result = VectorMMRReranker(index=index).rerank(
        scores,
        candidate_k=4,
        top_k=2,
        relevance_weight=0.50,
        eligible_mask=mask,
    )

    assert [hit.parent_asin for hit in result.hits] == ["C", "D"]


def test_transparency_continuously_increases_relevance_weight() -> None:
    policy = VectorDiversityPolicy(
        minimum_relevance_weight=0.30,
        maximum_relevance_weight=0.90,
    )

    assert policy.relevance_weight(0.0) == pytest.approx(0.30)
    assert policy.relevance_weight(0.5) == pytest.approx(0.60)
    assert policy.relevance_weight(1.0) == pytest.approx(0.90)


def test_generic_candidates_use_fused_relevance_and_vector_novelty() -> None:
    result = VectorMMRReranker(index=_index()).rerank_candidates(
        (
            VectorCandidate(parent_asin="A", candidate_rank=1, relevance=1.0),
            VectorCandidate(parent_asin="B", candidate_rank=2, relevance=0.99),
            VectorCandidate(parent_asin="D", candidate_rank=3, relevance=0.80),
        ),
        top_k=2,
        relevance_weight=0.30,
    )

    assert [hit.parent_asin for hit in result.hits] == ["A", "D"]
    assert result.hits[1].candidate_rank == 3


@pytest.mark.parametrize(
    ("candidate_k", "top_k", "weight"),
    [(0, 1, 0.5), (2, 0, 0.5), (1, 2, 0.5), (2, 1, -0.1), (2, 1, 1.1)],
)
def test_invalid_runtime_parameters_are_rejected(
    candidate_k: int,
    top_k: int,
    weight: float,
) -> None:
    index = _index()
    scores = index.score_vector(np.asarray([1.0, 0.0], dtype=np.float32))

    with pytest.raises(ValueError):
        VectorMMRReranker(index=index).rerank(
            scores,
            candidate_k=candidate_k,
            top_k=top_k,
            relevance_weight=weight,
        )
