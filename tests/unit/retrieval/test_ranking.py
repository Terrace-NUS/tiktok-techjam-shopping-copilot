from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from shopping_copilot.retrieval import (
    CrossEncoderRelevanceReranker,
    DenseIndex,
    DenseIndexManifest,
    EmbeddingSpec,
    GreedyDPPSelector,
    LatentAspectXQuADSelector,
    VectorCandidate,
)
from shopping_copilot.retrieval.models import DENSE_INDEX_SCHEMA, DenseArtifactRef


class _FakeCrossEncoder:
    @property
    def model_id(self) -> str:
        return "fake/reranker"

    def score(
        self,
        query: str,
        documents: Sequence[str],
        *,
        batch_size: int,
    ) -> tuple[float, ...]:
        assert query == "winter trip"
        assert batch_size == 2
        return tuple(float(document.count("winter")) for document in documents)


def _index() -> DenseIndex:
    spec = EmbeddingSpec(
        backend="fake",
        backend_version="1.0",
        model_id="example/model",
        model_revision="revision",
        dimension=3,
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
            [1.0, 0.0, 0.0],
            [0.999, 0.045, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
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


def _candidates() -> tuple[VectorCandidate, ...]:
    return (
        VectorCandidate(parent_asin="A", candidate_rank=1, relevance=1.0),
        VectorCandidate(parent_asin="B", candidate_rank=2, relevance=0.9),
        VectorCandidate(parent_asin="C", candidate_rank=3, relevance=0.7),
        VectorCandidate(parent_asin="D", candidate_rank=4, relevance=0.6),
    )


def test_cross_encoder_blends_model_score_with_route_prior() -> None:
    result = CrossEncoderRelevanceReranker(scorer=_FakeCrossEncoder()).rerank(
        "winter trip",
        _candidates(),
        documents={
            "A": "plain boot",
            "B": "winter winter boot",
            "C": "winter coat",
            "D": "plain hat",
        },
        prior_weight=0.25,
        batch_size=2,
    )

    assert result.model_id == "fake/reranker"
    assert [hit.parent_asin for hit in result.hits] == ["B", "C", "A", "D"]
    assert [item.candidate_rank for item in result.candidates] == [1, 2, 3, 4]
    assert result.hits[0].candidate_rank == 2


def test_dpp_uses_transparency_to_move_between_diversity_and_relevance() -> None:
    selector = GreedyDPPSelector(index=_index())

    diverse = selector.select(_candidates(), top_k=2, relevance_weight=0.1)
    focused = selector.select(_candidates(), top_k=2, relevance_weight=1.0)

    assert diverse.hits[0].parent_asin == "A"
    assert diverse.hits[1].parent_asin in {"C", "D"}
    assert [hit.parent_asin for hit in focused.hits] == ["A", "B"]


def test_latent_xquad_covers_vector_aspects_without_category_labels() -> None:
    result = LatentAspectXQuADSelector(index=_index(), maximum_aspects=3).select(
        _candidates(),
        top_k=3,
        relevance_weight=0.2,
    )

    assert result.latent_aspect_count == 3
    assert len({hit.latent_aspect for hit in result.hits}) >= 2
    assert result.hits[0].parent_asin == "A"


@pytest.mark.parametrize("selector_name", ["dpp", "xquad"])
def test_slate_selectors_reject_non_contiguous_candidates(selector_name: str) -> None:
    invalid = (VectorCandidate(parent_asin="A", candidate_rank=2, relevance=1.0),)
    selector = (
        GreedyDPPSelector(index=_index())
        if selector_name == "dpp"
        else LatentAspectXQuADSelector(index=_index())
    )

    with pytest.raises(ValueError, match="contiguous"):
        selector.select(invalid, top_k=1, relevance_weight=0.5)
