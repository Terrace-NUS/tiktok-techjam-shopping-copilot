from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from shopping_copilot.query_compiler import (
    COMPILED_QUERY_SCHEMA,
    QUERY_COMPILER_VERSION,
    CompiledDirectives,
    CompiledQuery,
    DiversityDirective,
)
from shopping_copilot.retrieval import (
    CompiledQueryBindingError,
    DenseIndex,
    DenseIndexManifest,
    DenseRetriever,
    EmbeddingSpec,
)
from shopping_copilot.retrieval.documents import ProductDocument
from shopping_copilot.retrieval.lexical import LexicalProbe
from shopping_copilot.retrieval.models import DENSE_INDEX_SCHEMA, DenseArtifactRef
from shopping_copilot.retrieval.multi_probe import CompiledProbeRunner
from shopping_copilot.retrieval.transparency import (
    DiagnosticStatus,
    TransparencyCalibration,
)
from shopping_copilot.session_context import ProbeQuality

CATALOG_ID = "sha256:" + "1" * 64
RELEASE_ID = "sha256:" + "2" * 64


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
        document_corpus_id="sha256:" + "3" * 64,
        product_count=3,
        embedding=_spec(),
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
        index_id="sha256:" + "6" * 64,
        manifest=manifest,
        parent_asins=("A", "B", "C"),
        vectors=np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32),
    )


class _FakeEmbedder:
    def __init__(self) -> None:
        self.query_calls = 0

    @property
    def spec(self) -> EmbeddingSpec:
        return _spec()

    def encode_documents(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        raise AssertionError("the online Probe must not encode documents")

    def encode_query(self, text: str) -> np.ndarray:
        self.query_calls += 1
        return np.array([1.0, 0.0], dtype=np.float32)


def _document(parent_asin: str, title: str) -> ProductDocument:
    return ProductDocument(
        parent_asin=parent_asin,
        text=(
            f"title: {title}\n"
            "categories: shoes\n"
            "store: example\n"
            "features: walking\n"
            "details: color: black\n"
            "description: comfortable shoe"
        ),
    )


def _query(*, q_lex: str = "walking shoe", search_ready: bool = True) -> CompiledQuery:
    return CompiledQuery(
        schema=COMPILED_QUERY_SCHEMA,
        compiler_version=QUERY_COMPILER_VERSION,
        catalog_id=CATALOG_ID,
        catalog_semantic_release_id=RELEASE_ID,
        category_graph_id="sha256:" + "7" * 64,
        intent_version=4,
        q_lex=q_lex,
        q_sem="Looking for comfortable walking shoes.",
        search_ready=search_ready,
        hard_constraints=(),
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


def _runner() -> tuple[CompiledProbeRunner, _FakeEmbedder, DenseIndex]:
    index = _index()
    embedder = _FakeEmbedder()
    lexical = LexicalProbe(
        (
            _document("A", "walking shoe alpha"),
            _document("B", "walking shoe beta"),
            _document("C", "walking shoe gamma"),
        ),
        probe_k=3,
    )
    runner = CompiledProbeRunner(
        retriever=DenseRetriever(index=index, embedder=embedder),
        lexical_probe=lexical,
        calibration=TransparencyCalibration(
            policy_id="semantic_mode_linear_test",
            low_anchor=-1.0,
            high_anchor=1.0,
        ),
        probe_k=3,
        mode_threshold=0.94,
    )
    return runner, embedder, index


def test_runner_scores_once_and_produces_deterministic_snapshot_and_belief() -> None:
    runner, embedder, _ = _runner()

    first = runner.run(_query())
    second = runner.run(_query())

    assert embedder.query_calls == 2
    assert first.snapshot.probe_id == second.snapshot.probe_id
    assert first.snapshot.eligible_count == 3
    assert first.snapshot.lexical.available is True
    assert len(first.snapshot.semantic.modes) == 3
    assert first.estimate.certainty is not None
    assert first.search_belief.certainty == first.estimate.certainty
    assert first.search_belief.certainty_evidence.quality_status is ProbeQuality.VALID
    assert sum(mode.mass for mode in first.search_belief.candidate_modes) == pytest.approx(1.0)


def test_one_bound_mask_is_applied_before_both_probe_top_k_views() -> None:
    runner, _, index = _runner()

    result = runner.run(
        _query(),
        eligible_mask=index.make_eligibility_mask({"B", "C"}),
        hard_filter_relaxed=True,
    )

    assert [hit.parent_asin for hit in result.ranking.hits] == ["B", "C"]
    assert {hit.parent_asin for hit in result.snapshot.lexical.hits} == {"B", "C"}
    assert result.estimate.certainty is not None
    assert result.estimate.diagnostics.status is DiagnosticStatus.DEGRADED
    assert "dense_probe_underfilled" in result.estimate.diagnostics.reason_codes
    assert "hard_filter_relaxed" in result.estimate.diagnostics.reason_codes


def test_lexical_failure_degrades_diagnostics_without_overwriting_valid_semantic_ct() -> None:
    runner, _, _ = _runner()

    result = runner.run(_query(q_lex=""))

    assert result.snapshot.lexical.reason == "empty_query"
    assert result.estimate.certainty is not None
    assert result.estimate.diagnostics.status is DiagnosticStatus.DEGRADED
    assert result.search_belief.certainty_evidence.quality_status is ProbeQuality.VALID


def test_runner_fails_closed_on_catalog_or_query_binding_mismatch() -> None:
    runner, _, _ = _runner()
    with pytest.raises(CompiledQueryBindingError, match="different catalog bindings"):
        runner.run(replace(_query(), catalog_id="sha256:" + "f" * 64))

    wrong_lexical = LexicalProbe(
        (_document("A", "shoe"), _document("B", "shoe")),
        probe_k=3,
    )
    index = _index()
    with pytest.raises(CompiledQueryBindingError, match="different catalog products"):
        CompiledProbeRunner(
            retriever=DenseRetriever(index=index, embedder=_FakeEmbedder()),
            lexical_probe=wrong_lexical,
            calibration=TransparencyCalibration(
                policy_id="semantic_mode_linear_test",
                low_anchor=-1.0,
                high_anchor=1.0,
            ),
            probe_k=3,
        )
