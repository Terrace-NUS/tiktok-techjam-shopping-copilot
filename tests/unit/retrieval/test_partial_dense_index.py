from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from shopping_copilot.catalog.semantic import canonical_json_bytes
from shopping_copilot.retrieval import (
    DenseIndexManifest,
    EmbeddingSpec,
    encode_dense_index_manifest,
    load_dense_index,
    write_partially_reembedded_dense_index,
)
from shopping_copilot.retrieval.bundle import (
    PARTIAL_DENSE_INDEX_BUILDER_VERSION,
    PARTIAL_PRODUCT_DOCUMENT_TEMPLATE_ID,
    document_corpus_id,
)
from shopping_copilot.retrieval.documents import ProductDocument
from shopping_copilot.retrieval.models import DENSE_INDEX_SCHEMA, DenseArtifactRef


class _ReplacementEmbedder:
    def __init__(self, spec: EmbeddingSpec) -> None:
        self.spec = spec
        self.seen: list[str] = []

    def encode_documents(self, texts: Sequence[str], *, batch_size: int) -> NDArray[np.float32]:
        del batch_size
        self.seen.extend(texts)
        return np.asarray([[3.0, 4.0] for _ in texts], dtype=np.float32)

    def encode_query(self, text: str) -> NDArray[np.float32]:
        del text
        return np.asarray([1.0, 0.0], dtype=np.float32)


def _hash(path: Path) -> tuple[str, int]:
    payload = path.read_bytes()
    return "sha256:" + hashlib.sha256(payload).hexdigest(), len(payload)


def _base_bundle(path: Path) -> tuple[Path, tuple[ProductDocument, ...], EmbeddingSpec]:
    path.mkdir()
    documents = (
        ProductDocument(parent_asin="A", text="old A"),
        ProductDocument(parent_asin="B", text="old B"),
    )
    spec = EmbeddingSpec(
        backend="fake",
        backend_version="1",
        model_id="fake/model",
        model_revision="revision",
        dimension=2,
        max_sequence_length=32,
        query_instruction="query: ",
        document_instruction="",
        pooling="cls",
    )
    (path / "parent-asins.json").write_bytes(canonical_json_bytes(("A", "B")))
    with (path / "vectors.npy").open("wb") as stream:
        np.save(
            stream,
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            allow_pickle=False,
        )
    refs = tuple(
        DenseArtifactRef(
            kind=kind,  # type: ignore[arg-type]
            filename=filename,
            content_id=content_id,
            byte_size=byte_size,
        )
        for kind, filename in (
            ("parent_asins", "parent-asins.json"),
            ("vectors", "vectors.npy"),
        )
        for content_id, byte_size in (_hash(path / filename),)
    )
    manifest = DenseIndexManifest(
        schema=DENSE_INDEX_SCHEMA,
        builder_version="dense_index_v0",
        catalog_id="sha256:" + "1" * 64,
        catalog_semantic_release_id="sha256:" + "2" * 64,
        document_template_id="product_document_v1",
        document_corpus_id=document_corpus_id(documents),
        product_count=2,
        embedding=spec,
        vector_dtype="float32",
        artifacts=refs,
    )
    (path / "bundle-manifest.json").write_bytes(encode_dense_index_manifest(manifest))
    return path, documents, spec


def test_partial_index_reembeds_only_replaced_rows_and_is_reproducible(tmp_path: Path) -> None:
    base, documents, spec = _base_bundle(tmp_path / "base")
    replacement = ProductDocument(parent_asin="A", text="new fact card A")
    embedder = _ReplacementEmbedder(spec)

    index = write_partially_reembedded_dense_index(
        base,
        tmp_path / "partial",
        base_documents=documents,
        replacement_documents={"A": replacement},
        embedder=embedder,
        batch_size=8,
    )

    assert embedder.seen == ["new fact card A"]
    assert np.allclose(index.vectors[0], [0.6, 0.8])
    assert np.array_equal(index.vectors[1], load_dense_index(base).vectors[1])
    assert index.manifest.builder_version == PARTIAL_DENSE_INDEX_BUILDER_VERSION
    assert index.manifest.document_template_id == PARTIAL_PRODUCT_DOCUMENT_TEMPLATE_ID
    assert index.manifest.document_corpus_id == document_corpus_id((replacement, documents[1]))

    second = write_partially_reembedded_dense_index(
        base,
        tmp_path / "partial",
        base_documents=documents,
        replacement_documents={"A": replacement},
        embedder=_ReplacementEmbedder(spec),
    )
    assert second.index_id == index.index_id
