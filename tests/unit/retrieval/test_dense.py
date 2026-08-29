from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from shopping_copilot.retrieval import (
    DenseIndex,
    DenseIndexIntegrityError,
    DenseIndexManifest,
    DenseRetriever,
    EmbeddingSpec,
    create_dense_retriever,
)
from shopping_copilot.retrieval.models import DENSE_INDEX_SCHEMA, DenseArtifactRef


def _spec() -> EmbeddingSpec:
    return EmbeddingSpec(
        backend="fake",
        backend_version="1.0",
        model_id="example/model",
        model_revision="revision",
        dimension=2,
        max_sequence_length=32,
        query_instruction="query: ",
        document_instruction="passage: ",
        pooling="cls",
    )


def _manifest(spec: EmbeddingSpec | None = None) -> DenseIndexManifest:
    return DenseIndexManifest(
        schema=DENSE_INDEX_SCHEMA,
        builder_version="dense_index_v0",
        catalog_id="sha256:" + "1" * 64,
        catalog_semantic_release_id="sha256:" + "2" * 64,
        document_template_id="product_document_v1",
        document_corpus_id="sha256:" + "3" * 64,
        product_count=3,
        embedding=spec or _spec(),
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


def _index(*, index_hex: str = "a") -> DenseIndex:
    return DenseIndex(
        index_id="sha256:" + index_hex * 64,
        manifest=_manifest(),
        parent_asins=("A", "B", "C"),
        vectors=np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )


class _FakeEmbedder:
    def __init__(self, spec: EmbeddingSpec, query: np.ndarray) -> None:
        self._spec = spec
        self.query = query
        self.query_calls = 0

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    def encode_documents(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        raise AssertionError("online retrieval must not encode documents")

    def encode_query(self, text: str) -> np.ndarray:
        self.query_calls += 1
        return self.query


def test_exact_scores_and_parent_asin_tie_break_are_stable() -> None:
    index = _index()

    scores = index.score_vector(np.array([2.0, 0.0], dtype=np.float32))
    first = index.select_top_k(scores, top_k=3)
    second = index.select_top_k(scores, top_k=3)

    assert scores.values.tolist() == pytest.approx([1.0, 1.0, 0.0])
    assert [hit.parent_asin for hit in first] == ["A", "B", "C"]
    assert first == second
    assert [hit.rank for hit in first] == [1, 2, 3]


def test_eligibility_is_applied_before_top_k_and_never_backfilled_with_excluded_items() -> None:
    index = _index()
    scores = index.score_vector(np.array([1.0, 0.0], dtype=np.float32))

    hits = index.select_top_k(
        scores,
        top_k=3,
        eligible_mask=index.make_eligibility_mask({"A"}),
    )

    assert [hit.parent_asin for hit in hits] == ["A"]
    assert (
        index.select_top_k(
            scores,
            top_k=3,
            eligible_mask=index.make_eligibility_mask(set()),
        )
        == ()
    )


def test_raw_eligibility_masks_fail_closed() -> None:
    index = _index()
    scores = index.score_vector(np.array([1.0, 0.0], dtype=np.float32))

    with pytest.raises(TypeError, match="DenseEligibilityMask"):
        index.select_top_k(  # type: ignore[arg-type]
            scores,
            top_k=1,
            eligible_mask=np.ones(3, dtype=np.bool_),
        )


def test_unbound_or_float64_score_vectors_fail_closed() -> None:
    index = _index()

    with pytest.raises(TypeError, match="DenseScoreSnapshot"):
        index.select_top_k(  # type: ignore[arg-type]
            np.array([1.0, 1.0 - 1e-12, 0.0], dtype=np.float64),
            top_k=2,
        )


def test_scores_and_masks_cannot_cross_loaded_index_instances() -> None:
    first = _index(index_hex="a")
    second = _index(index_hex="b")
    first_scores = first.score_vector(np.array([1.0, 0.0], dtype=np.float32))
    second_scores = second.score_vector(np.array([1.0, 0.0], dtype=np.float32))
    first_mask = first.make_eligibility_mask({"A"})

    with pytest.raises(DenseIndexIntegrityError, match="different dense index"):
        second.select_top_k(first_scores, top_k=1)
    with pytest.raises(DenseIndexIntegrityError, match="different dense index"):
        second.select_top_k(second_scores, top_k=1, eligible_mask=first_mask)


def test_eligibility_mask_is_built_from_known_product_ids() -> None:
    index = _index()

    with pytest.raises(KeyError, match="unknown eligible"):
        index.make_eligibility_mask({"UNKNOWN"})
    with pytest.raises(TypeError, match="iterable"):
        index.make_eligibility_mask("A")


def test_retriever_encodes_query_once_and_reuses_complete_score_vector() -> None:
    index = _index()
    embedder = _FakeEmbedder(_spec(), np.array([1.0, 0.0], dtype=np.float32))
    retriever = DenseRetriever(index=index, embedder=embedder)

    result = retriever.search_with_scores("red walking shoes", top_k=2)

    assert embedder.query_calls == 1
    assert [hit.parent_asin for hit in result.hits] == ["A", "B"]
    assert result.scores.values.flags.writeable is False


def test_dense_index_constructor_enforces_mapping_and_vector_invariants() -> None:
    with pytest.raises(DenseIndexIntegrityError, match="sorted and unique"):
        DenseIndex(
            index_id="sha256:" + "a" * 64,
            manifest=_manifest(),
            parent_asins=("B", "A", "C"),
            vectors=np.array(
                [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                dtype=np.float32,
            ),
        )
    with pytest.raises(DenseIndexIntegrityError, match="L2-normalized"):
        DenseIndex(
            index_id="sha256:" + "a" * 64,
            manifest=_manifest(),
            parent_asins=("A", "B", "C"),
            vectors=np.array(
                [[2.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                dtype=np.float32,
            ),
        )


def test_index_vectors_are_exposed_as_a_non_writeable_view() -> None:
    vectors = _index().vectors

    assert vectors.flags.writeable is False
    with pytest.raises(ValueError):
        vectors.setflags(write=True)


def test_retriever_refuses_an_encoder_that_differs_from_the_index() -> None:
    index = _index()
    different = replace(_spec(), model_revision="other")

    with pytest.raises(DenseIndexIntegrityError, match="specification"):
        DenseRetriever(
            index=index,
            embedder=_FakeEmbedder(different, np.array([1.0, 0.0], dtype=np.float32)),
        )


def test_factory_requires_an_active_semantic_release() -> None:
    with pytest.raises(ValueError, match="release_dir"):
        create_dense_retriever(index_path="index", release_dir=None)
