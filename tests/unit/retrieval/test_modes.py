from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from shopping_copilot.retrieval import DenseIndex, DenseIndexManifest, EmbeddingSpec
from shopping_copilot.retrieval.models import DENSE_INDEX_SCHEMA, DenseArtifactRef
from shopping_copilot.retrieval.modes import SemanticModeProbe


def _unit(degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    return np.array([math.cos(radians), math.sin(radians)], dtype=np.float32)


def _index(vectors: list[np.ndarray], *, parent_asins: tuple[str, ...] | None = None) -> DenseIndex:
    if parent_asins is None:
        parent_asins = tuple(chr(ord("A") + index) for index in range(len(vectors)))
    matrix = np.asarray(vectors, dtype=np.float32)
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
        product_count=len(vectors),
        embedding=EmbeddingSpec(
            backend="fake",
            backend_version="1.0",
            model_id="example/model",
            model_revision="revision",
            dimension=int(matrix.shape[1]),
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
        parent_asins=parent_asins,
        vectors=matrix,
    )


def test_fixed_leader_prevents_similarity_bridge() -> None:
    index = _index([_unit(0.0), _unit(18.0), _unit(36.0)])
    scores = index.score_vector(_unit(0.0))
    result = index.rank_scores(scores, top_k=3)

    observation = SemanticModeProbe(index).observe(result, probe_k=3)

    assert [membership.mode_id for membership in observation.memberships] == [
        "mode_0001",
        "mode_0001",
        "mode_0002",
    ]
    assert [(mode.leader_id, mode.size) for mode in observation.modes] == [
        ("A", 2),
        ("C", 1),
    ]
    np.testing.assert_allclose(observation.modes[0].centroid, _unit(9.0), atol=1e-7)
    assert observation.modes[0].centroid.dtype == np.float64
    assert observation.modes[0].centroid.flags.writeable is False


def test_equal_leader_similarity_uses_the_earliest_mode() -> None:
    index = _index([_unit(-18.0), _unit(18.0), _unit(0.0)], parent_asins=("A", "B", "X"))
    scores = index.score_vector(_unit(180.0))
    result = index.rank_scores(scores, top_k=3)
    assert [hit.parent_asin for hit in result.hits] == ["A", "B", "X"]

    observation = SemanticModeProbe(index).observe(result, probe_k=3)

    assert [membership.mode_id for membership in observation.memberships] == [
        "mode_0001",
        "mode_0002",
        "mode_0001",
    ]
    assert observation.memberships[-1].similarity_to_leader == 0.951057


@pytest.mark.parametrize(
    ("cosine", "expected_mode_count"),
    [
        (0.9399998, 1),
        (0.9399991, 2),
    ],
)
def test_threshold_is_applied_after_rounding_to_six_places(
    cosine: float,
    expected_mode_count: int,
) -> None:
    second = np.array([cosine, math.sqrt(1.0 - cosine * cosine)], dtype=np.float32)
    index = _index([_unit(0.0), second])
    result = index.rank_scores(index.score_vector(_unit(0.0)), top_k=2)

    observation = SemanticModeProbe(index).observe(result, probe_k=2, threshold=0.94)

    assert len(observation.modes) == expected_mode_count
    if expected_mode_count == 1:
        assert observation.memberships[1].similarity_to_leader == 0.94


def test_duplicate_mode_reports_equal_weight_geometry_and_concentration() -> None:
    index = _index([_unit(0.0), _unit(1.0), _unit(-1.0), _unit(90.0)])
    result = index.rank_scores(index.score_vector(_unit(0.0)), top_k=4)

    observation = SemanticModeProbe(index).observe(result, probe_k=4)

    assert [mode.size for mode in observation.modes] == [3, 1]
    assert observation.modes[0].representative_ids == ("A", "B", "C")
    assert observation.largest_mode_share == 0.75
    expected_effective_count = math.exp(-(0.75 * math.log(0.75) + 0.25 * math.log(0.25)))
    assert observation.effective_mode_count == pytest.approx(expected_effective_count)
    assert observation.raw_listing_coherence.n == 4
    assert observation.equal_mode_coherence.n == 2
    assert observation.equal_mode_coherence.available is True
    assert observation.duplicate_concentration_warning is True


def test_empty_mask_stays_empty_and_mode_coherence_is_insufficient() -> None:
    index = _index([_unit(0.0), _unit(90.0)])
    scores = index.score_vector(_unit(0.0))
    mask = index.make_eligibility_mask(set())
    result = index.rank_scores(scores, top_k=2, eligible_mask=mask)

    observation = SemanticModeProbe(index).observe(result, probe_k=2)

    assert observation.hits == ()
    assert observation.memberships == ()
    assert observation.modes == ()
    assert observation.largest_mode_share == 0.0
    assert observation.effective_mode_count == 0.0
    assert observation.raw_listing_coherence.reason == "empty_candidates"
    assert observation.equal_mode_coherence.reason == "insufficient_candidates"
    assert observation.equal_mode_coherence.n == 0
    assert observation.duplicate_concentration_warning is False


def test_observation_inherits_masked_hits_without_resurrecting_products() -> None:
    index = _index([_unit(0.0), _unit(45.0), _unit(90.0)])
    scores = index.score_vector(_unit(0.0))
    mask = index.make_eligibility_mask({"B"})
    result = index.rank_scores(scores, top_k=3, eligible_mask=mask)

    observation = SemanticModeProbe(index).observe(result, probe_k=3)

    assert [hit.parent_asin for hit in observation.hits] == ["B"]
    assert [membership.parent_asin for membership in observation.memberships] == ["B"]
    assert observation.equal_mode_coherence.available is False
    assert observation.equal_mode_coherence.reason == "insufficient_candidates"


def test_observation_does_not_encode_score_or_rank_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _index([_unit(0.0), _unit(45.0), _unit(90.0)])
    scores = index.score_vector(_unit(0.0))
    result = index.rank_scores(scores, top_k=3)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("semantic mode Probe must reuse the existing ranking")

    forbidden_call: Callable[..., None] = forbidden
    monkeypatch.setattr(index, "score_vector", forbidden_call)
    monkeypatch.setattr(index, "rank_scores", forbidden_call)

    observation = SemanticModeProbe(index).observe(result, probe_k=3)

    assert observation.hits == result.hits
    assert all(
        observed is existing
        for observed, existing in zip(observation.hits, result.hits, strict=True)
    )


def test_probe_rejects_a_depth_larger_than_the_source_ranking() -> None:
    index = _index([_unit(0.0), _unit(90.0)])
    result = index.rank_scores(index.score_vector(_unit(0.0)), top_k=1)

    with pytest.raises(ValueError, match="ranking depth"):
        SemanticModeProbe(index).observe(result, probe_k=2)
