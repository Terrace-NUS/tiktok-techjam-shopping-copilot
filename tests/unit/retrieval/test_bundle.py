from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest

from shopping_copilot.catalog.semantic import canonical_json_bytes
from shopping_copilot.retrieval import (
    DenseIndexIntegrityError,
    DenseIndexManifest,
    EmbeddingSpec,
    decode_dense_index_manifest,
    encode_dense_index_manifest,
    load_dense_index,
    validate_dense_index,
)
from shopping_copilot.retrieval.models import DENSE_INDEX_SCHEMA, DenseArtifactRef


def _hash(path: Path) -> tuple[str, int]:
    payload = path.read_bytes()
    return "sha256:" + hashlib.sha256(payload).hexdigest(), len(payload)


def _write_bundle(path: Path) -> tuple[DenseIndexManifest, Path]:
    path.mkdir()
    parent_asins = ("A", "B", "C")
    (path / "parent-asins.json").write_bytes(canonical_json_bytes(parent_asins))
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    with (path / "vectors.npy").open("wb") as stream:
        np.save(stream, vectors, allow_pickle=False)
    refs = []
    for kind, filename in (("parent_asins", "parent-asins.json"), ("vectors", "vectors.npy")):
        content_id, byte_size = _hash(path / filename)
        refs.append(
            DenseArtifactRef(
                kind=kind,  # type: ignore[arg-type]
                filename=filename,
                content_id=content_id,
                byte_size=byte_size,
            )
        )
    manifest = DenseIndexManifest(
        schema=DENSE_INDEX_SCHEMA,
        builder_version="dense_index_v0",
        catalog_id="sha256:" + "1" * 64,
        catalog_semantic_release_id="sha256:" + "2" * 64,
        document_template_id="product_document_v1",
        document_corpus_id="sha256:" + "3" * 64,
        product_count=3,
        embedding=EmbeddingSpec(
            backend="fake",
            backend_version="1.0",
            model_id="example/model",
            model_revision="revision",
            dimension=2,
            max_sequence_length=32,
            query_instruction="query: ",
            document_instruction="",
            pooling="cls",
        ),
        vector_dtype="float32",
        artifacts=tuple(refs),
    )
    (path / "bundle-manifest.json").write_bytes(encode_dense_index_manifest(manifest))
    return manifest, path


def test_manifest_round_trip_and_bundle_load_are_strict(tmp_path: Path) -> None:
    manifest, path = _write_bundle(tmp_path / "index")

    payload = encode_dense_index_manifest(manifest)
    assert decode_dense_index_manifest(payload) == manifest
    index = load_dense_index(
        path,
        expected_catalog_id=manifest.catalog_id,
        expected_release_id=manifest.catalog_semantic_release_id,
        mmap=False,
    )

    assert index.manifest == manifest
    assert index.parent_asins == ("A", "B", "C")
    assert index.vectors.shape == (3, 2)
    assert validate_dense_index(path) == "sha256:" + hashlib.sha256(payload).hexdigest()


def test_noncanonical_manifest_and_unexpected_member_are_rejected(tmp_path: Path) -> None:
    _, source = _write_bundle(tmp_path / "source")
    noncanonical = shutil.copytree(source, tmp_path / "noncanonical")
    manifest = noncanonical / "bundle-manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(DenseIndexIntegrityError, match="canonical"):
        load_dense_index(noncanonical)

    extra = shutil.copytree(source, tmp_path / "extra")
    (extra / "unexpected.txt").write_text("x", encoding="utf-8")
    with pytest.raises(DenseIndexIntegrityError, match="members"):
        load_dense_index(extra)


@pytest.mark.parametrize("filename", ["parent-asins.json", "vectors.npy"])
def test_any_material_artifact_tampering_is_rejected(tmp_path: Path, filename: str) -> None:
    _, path = _write_bundle(tmp_path / "index")
    artifact = path / filename
    artifact.write_bytes(artifact.read_bytes() + b"tampered")

    with pytest.raises(DenseIndexIntegrityError, match="artifact bytes"):
        load_dense_index(path)


def test_wrong_expected_release_or_catalog_fails_closed(tmp_path: Path) -> None:
    _, path = _write_bundle(tmp_path / "index")

    with pytest.raises(DenseIndexIntegrityError, match="catalog ID"):
        load_dense_index(path, expected_catalog_id="sha256:" + "9" * 64)
    with pytest.raises(DenseIndexIntegrityError, match="release ID"):
        load_dense_index(path, expected_release_id="sha256:" + "9" * 64)
