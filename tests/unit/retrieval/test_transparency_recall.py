from __future__ import annotations

import math

import numpy as np

from shopping_copilot.retrieval import DenseIndex, DenseIndexManifest, EmbeddingSpec
from shopping_copilot.retrieval.models import DENSE_INDEX_SCHEMA, DenseArtifactRef
from shopping_copilot.retrieval.transparency_recall import (
    TransparencyAwareDenseRecall,
    TransparencyRecallPolicy,
)


def _unit(angle_degrees: float) -> tuple[float, float]:
    radians = math.radians(angle_degrees)
    return (math.cos(radians), math.sin(radians))


def _index() -> DenseIndex:
    vectors = np.asarray(
        [
            _unit(-2.0),
            _unit(-1.0),
            _unit(0.0),
            _unit(1.0),
            _unit(33.0),
            _unit(34.0),
            _unit(35.0),
            _unit(36.0),
            _unit(-36.0),
            _unit(-35.0),
            _unit(-34.0),
            _unit(-33.0),
        ],
        dtype=np.float32,
    )
    spec = EmbeddingSpec(
        backend="fake",
        backend_version="1",
        model_id="fake/model",
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
        product_count=len(vectors),
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
    return DenseIndex(
        index_id="sha256:" + "a" * 64,
        manifest=manifest,
        parent_asins=tuple(f"P{index:02d}" for index in range(len(vectors))),
        vectors=vectors,
    )


def _policy() -> TransparencyRecallPolicy:
    return TransparencyRecallPolicy(
        candidate_pool_k=9,
        frontier_k=12,
        maximum_directions=3,
        minimum_normalized_center_relevance=0.0,
        maximum_center_similarity=0.95,
        dense_budget_at_high_transparency=3,
        dense_budget_range=3,
        lexical_budget_at_low_transparency=1,
        lexical_budget_range=1,
    )


def test_low_transparency_recalls_round_robin_from_multiple_vector_directions() -> None:
    index = _index()
    policy = _policy()
    recall = TransparencyAwareDenseRecall(index=index, policy=policy)
    scores = index.score_vector(np.asarray([1.0, 0.0], dtype=np.float32))

    result = recall.recall(
        scores,
        eligible_mask=index.make_eligibility_mask(index.parent_asins),
        transparency=0.0,
    )

    assert result.requested_direction_count == 3
    assert len(result.directions) == 3
    assert len(result.candidates) == 9
    counts = {
        direction.direction_id: sum(
            item.direction_id == direction.direction_id for item in result.candidates
        )
        for direction in result.directions
    }
    assert counts == {"direction_1": 3, "direction_2": 3, "direction_3": 3}
    assert result.timings.total_ms >= 0.0


def test_high_transparency_uses_one_direction_without_changing_pool_size() -> None:
    index = _index()
    recall = TransparencyAwareDenseRecall(index=index, policy=_policy())
    scores = index.score_vector(np.asarray([1.0, 0.0], dtype=np.float32))

    result = recall.recall(
        scores,
        eligible_mask=index.make_eligibility_mask(index.parent_asins),
        transparency=1.0,
    )

    assert result.requested_direction_count == 1
    assert len(result.directions) == 1
    assert len(result.candidates) == 9
    assert {item.direction_id for item in result.candidates} == {"direction_1"}


def test_high_transparency_deepens_around_one_center_instead_of_query_broadening() -> None:
    index = _index()
    recall = TransparencyAwareDenseRecall(index=index, policy=_policy())
    scores = index.score_vector(np.asarray([1.0, 0.0], dtype=np.float32))
    mask = index.make_eligibility_mask(index.parent_asins)

    broad = recall.recall(scores, eligible_mask=mask, transparency=0.0)
    focused = recall.recall(scores, eligible_mask=mask, transparency=1.0)

    broad_vectors = index.vectors[[index.row_index(item.parent_asin) for item in broad.candidates]]
    focused_vectors = index.vectors[
        [index.row_index(item.parent_asin) for item in focused.candidates]
    ]
    assert _mean_pairwise(focused_vectors) > _mean_pairwise(broad_vectors)


def test_hard_mask_is_applied_before_frontier_centers_and_direction_expansion() -> None:
    index = _index()
    recall = TransparencyAwareDenseRecall(index=index, policy=_policy())
    scores = index.score_vector(np.asarray([1.0, 0.0], dtype=np.float32))
    eligible = index.parent_asins[:8]

    result = recall.recall(
        scores,
        eligible_mask=index.make_eligibility_mask(eligible),
        transparency=0.0,
    )

    assert result.frontier_count == len(eligible)
    assert {item.parent_asin for item in result.candidates} <= set(eligible)
    assert {item.center_parent_asin for item in result.directions} <= set(eligible)


def test_route_budgets_keep_total_fixed_while_shifting_toward_exact_routes() -> None:
    policy = TransparencyRecallPolicy()

    broad = policy.budgets(0.0)
    focused = policy.budgets(1.0)

    assert broad.total == focused.total == policy.candidate_pool_k
    assert broad.dense > focused.dense
    assert broad.lexical < focused.lexical
    assert broad.facet < focused.facet


def _mean_pairwise(vectors: np.ndarray) -> float:
    similarities = vectors @ vectors.T
    count = len(vectors)
    return float((similarities.sum() - np.trace(similarities)) / (count * (count - 1)))
