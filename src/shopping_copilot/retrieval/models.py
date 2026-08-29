"""Immutable contracts for the first dense retrieval route."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

DENSE_INDEX_SCHEMA: Literal["shopping-copilot/dense-index-bundle/v0"] = (
    "shopping-copilot/dense-index-bundle/v0"
)
DENSE_INDEX_BUILDER_VERSION = "dense_index_v0"
DENSE_INDEX_MANIFEST_FILENAME = "bundle-manifest.json"
PARENT_ASINS_FILENAME = "parent-asins.json"
VECTORS_FILENAME = "vectors.npy"
DENSE_INDEX_FILENAMES = frozenset(
    {DENSE_INDEX_MANIFEST_FILENAME, PARENT_ASINS_FILENAME, VECTORS_FILENAME}
)
PRODUCT_DOCUMENT_TEMPLATE_ID = "product_document_v1"

DEFAULT_MODEL_ID = "BAAI/bge-small-en-v1.5"
DEFAULT_MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
DEFAULT_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

ArtifactKind = Literal["parent_asins", "vectors"]
ARTIFACT_FILENAMES: Mapping[ArtifactKind, str] = MappingProxyType(
    {"parent_asins": PARENT_ASINS_FILENAME, "vectors": VECTORS_FILENAME}
)

_CONTENT_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


@dataclass(frozen=True, slots=True, kw_only=True)
class EmbeddingSpec:
    """Everything that can materially change query or document vectors."""

    backend: str
    backend_version: str
    model_id: str
    model_revision: str
    dimension: int
    max_sequence_length: int
    query_instruction: str
    document_instruction: str
    pooling: str
    normalization: Literal["l2"] = "l2"

    def __post_init__(self) -> None:
        _require_text(self.backend, name="EmbeddingSpec.backend")
        _require_text(self.backend_version, name="EmbeddingSpec.backend_version")
        _require_text(self.model_id, name="EmbeddingSpec.model_id")
        _require_text(self.model_revision, name="EmbeddingSpec.model_revision")
        if type(self.dimension) is not int or self.dimension <= 0:
            raise ValueError("EmbeddingSpec.dimension must be positive")
        if type(self.max_sequence_length) is not int or self.max_sequence_length <= 0:
            raise ValueError("EmbeddingSpec.max_sequence_length must be positive")
        if type(self.query_instruction) is not str:
            raise TypeError("EmbeddingSpec.query_instruction must be a string")
        if type(self.document_instruction) is not str:
            raise TypeError("EmbeddingSpec.document_instruction must be a string")
        _require_text(self.pooling, name="EmbeddingSpec.pooling")
        if self.normalization != "l2":
            raise ValueError("EmbeddingSpec.normalization must be l2")


@dataclass(frozen=True, slots=True, kw_only=True)
class DenseArtifactRef:
    """Hash and byte size for one material dense-index artifact."""

    kind: ArtifactKind
    filename: str
    content_id: str
    byte_size: int

    def __post_init__(self) -> None:
        if self.kind not in ARTIFACT_FILENAMES:
            raise ValueError("DenseArtifactRef.kind is invalid")
        if self.filename != ARTIFACT_FILENAMES[self.kind]:
            raise ValueError("DenseArtifactRef.filename is invalid for its kind")
        _require_content_id(self.content_id, name="DenseArtifactRef.content_id")
        if type(self.byte_size) is not int or self.byte_size <= 0:
            raise ValueError("DenseArtifactRef.byte_size must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class DenseIndexManifest:
    """Canonical manifest for an immutable, release-bound dense index."""

    schema: Literal["shopping-copilot/dense-index-bundle/v0"]
    builder_version: str
    catalog_id: str
    catalog_semantic_release_id: str
    document_template_id: str
    document_corpus_id: str
    product_count: int
    embedding: EmbeddingSpec
    vector_dtype: Literal["float32"]
    artifacts: tuple[DenseArtifactRef, ...]

    def __post_init__(self) -> None:
        if self.schema != DENSE_INDEX_SCHEMA:
            raise ValueError("DenseIndexManifest.schema is invalid")
        if _IDENTIFIER_PATTERN.fullmatch(self.builder_version) is None:
            raise ValueError("DenseIndexManifest.builder_version is invalid")
        _require_content_id(self.catalog_id, name="DenseIndexManifest.catalog_id")
        _require_content_id(
            self.catalog_semantic_release_id,
            name="DenseIndexManifest.catalog_semantic_release_id",
        )
        if _IDENTIFIER_PATTERN.fullmatch(self.document_template_id) is None:
            raise ValueError("DenseIndexManifest.document_template_id is invalid")
        _require_content_id(
            self.document_corpus_id,
            name="DenseIndexManifest.document_corpus_id",
        )
        if type(self.product_count) is not int or self.product_count <= 0:
            raise ValueError("DenseIndexManifest.product_count must be positive")
        if not isinstance(self.embedding, EmbeddingSpec):
            raise TypeError("DenseIndexManifest.embedding is invalid")
        if self.vector_dtype != "float32":
            raise ValueError("DenseIndexManifest.vector_dtype must be float32")
        if type(self.artifacts) is not tuple:
            raise TypeError("DenseIndexManifest.artifacts must be a tuple")
        expected_kinds = tuple(sorted(ARTIFACT_FILENAMES))
        observed_kinds = tuple(item.kind for item in self.artifacts)
        if observed_kinds != expected_kinds:
            raise ValueError("DenseIndexManifest.artifacts are incomplete or unordered")


@dataclass(frozen=True, slots=True, kw_only=True)
class DenseHit:
    """One stable dense-route result."""

    parent_asin: str
    score: float
    rank: int

    def __post_init__(self) -> None:
        _require_text(self.parent_asin, name="DenseHit.parent_asin")
        if self.parent_asin != self.parent_asin.strip():
            raise ValueError("DenseHit.parent_asin must be trimmed")
        if type(self.score) is not float or not math.isfinite(self.score):
            raise ValueError("DenseHit.score must be a finite float")
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("DenseHit.rank must be positive")


def default_embedding_spec() -> EmbeddingSpec:
    """Return the pinned encoder selected for the R0 experiment."""

    return EmbeddingSpec(
        backend="sentence_transformers",
        backend_version="5.7.0",
        model_id=DEFAULT_MODEL_ID,
        model_revision=DEFAULT_MODEL_REVISION,
        dimension=384,
        max_sequence_length=512,
        query_instruction=DEFAULT_QUERY_INSTRUCTION,
        document_instruction="",
        pooling="cls",
    )


def _require_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError(f"{name} contains a lone surrogate")
    return value


def _require_content_id(value: object, *, name: str) -> str:
    if type(value) is not str or _CONTENT_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a full sha256 content ID")
    return value
