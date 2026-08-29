from __future__ import annotations

import numpy as np
import pytest

from shopping_copilot.retrieval import (
    DenseIndex,
    DenseIndexIntegrityError,
    DenseIndexManifest,
    EmbeddingSpec,
)
from shopping_copilot.retrieval.models import DENSE_INDEX_SCHEMA, DenseArtifactRef
from shopping_copilot.retrieval.probe import FixedDenseProbe


def _index() -> DenseIndex:
    artifacts = (
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
    )
    manifest = DenseIndexManifest(
        schema=DENSE_INDEX_SCHEMA,
        builder_version="dense_index_v0",
        catalog_id="sha256:" + "1" * 64,
        catalog_semantic_release_id="sha256:" + "2" * 64,
        document_template_id="product_document_v1",
        document_corpus_id="sha256:" + "3" * 64,
        product_count=4,
        embedding=EmbeddingSpec(
            backend="fake",
            backend_version="1.0",
            model_id="example/model",
            model_revision="revision",
            dimension=2,
            max_sequence_length=32,
            query_instruction="",
            document_instruction="",
            pooling="cls",
        ),
        vector_dtype="float32",
        artifacts=artifacts,
    )
    return DenseIndex(
        index_id="sha256:" + "a" * 64,
        manifest=manifest,
        parent_asins=("A", "B", "C", "D"),
        vectors=np.array(
            [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.0]],
            dtype=np.float32,
        ),
    )


def test_fixed_probe_reuses_scores_and_reports_raw_coherence() -> None:
    index = _index()
    scores = index.score_vector(np.array([1.0, 0.0], dtype=np.float32))
    probe = FixedDenseProbe(index)
    result = index.rank_scores(scores, top_k=3)

    observation = probe.observe(result, probe_k=3)

    assert [hit.parent_asin for hit in observation.hits] == ["A", "B", "C"]
    assert observation.coherence.available is True
    assert observation.coherence.n == 3
    assert observation.coherence.debiased_pairwise_cosine is not None


def test_fixed_probe_applies_the_same_eligibility_mask_before_top_k() -> None:
    index = _index()
    scores = index.score_vector(np.array([1.0, 0.0], dtype=np.float32))
    mask = index.make_eligibility_mask({"B"})
    result = index.rank_scores(scores, top_k=3, eligible_mask=mask)

    observation = FixedDenseProbe(index).observe(
        result,
        probe_k=3,
    )

    assert [hit.parent_asin for hit in observation.hits] == ["B"]
    assert observation.coherence.available is False
    assert observation.coherence.reason == "insufficient_candidates"


def test_probe_rejects_too_shallow_or_foreign_rankings() -> None:
    index = _index()
    scores = index.score_vector(np.array([1.0, 0.0], dtype=np.float32))
    result = index.rank_scores(scores, top_k=2)

    with pytest.raises(ValueError, match="ranking depth"):
        FixedDenseProbe(index).observe(result, probe_k=3)
    with pytest.raises(DenseIndexIntegrityError, match="different dense index"):
        FixedDenseProbe(_index()).observe(result, probe_k=2)
