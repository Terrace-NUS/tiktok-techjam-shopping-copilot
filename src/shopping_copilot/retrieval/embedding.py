"""Pluggable text embedding boundary and the pinned Sentence Transformers backend."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from .errors import EmbeddingBackendUnavailableError, QueryEmbeddingError
from .models import EmbeddingSpec

FloatMatrix = NDArray[np.float32]
FloatVector = NDArray[np.float32]


class _SentenceTransformerModel(Protocol):
    max_seq_length: int

    def encode(self, sentences: Sequence[str], **kwargs: object) -> object:
        """Encode text using backend-specific keyword arguments."""


class TextEmbedder(Protocol):
    """Backend-neutral contract used by index build and online retrieval."""

    @property
    def spec(self) -> EmbeddingSpec:
        """Return the exact model and text-encoding contract."""

    def encode_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> FloatMatrix:
        """Encode a non-empty ordered document sequence."""

    def encode_query(self, text: str) -> FloatVector:
        """Encode one semantic query."""


class SentenceTransformerTextEmbedder:
    """Lazy adapter around one immutable Sentence Transformers revision."""

    def __init__(
        self,
        spec: EmbeddingSpec,
        *,
        device: str | None = None,
        local_files_only: bool = False,
        show_progress_bar: bool = False,
    ) -> None:
        self._spec = spec
        self._show_progress_bar = show_progress_bar
        try:
            module = import_module("sentence_transformers")
            installed_version = str(module.__version__)
            if spec.backend != "sentence_transformers" or installed_version != spec.backend_version:
                raise ValueError("installed embedding backend differs from the index specification")
            self._model = cast(
                _SentenceTransformerModel,
                module.SentenceTransformer(
                    spec.model_id,
                    revision=spec.model_revision,
                    device=device,
                    trust_remote_code=False,
                    local_files_only=local_files_only,
                ),
            )
            self._model.max_seq_length = spec.max_sequence_length
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise EmbeddingBackendUnavailableError(
                "cannot load the configured Sentence Transformers model; "
                "install the retrieval extra and make the pinned model revision available"
            ) from error

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    def encode_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> FloatMatrix:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        prepared = _prepare_texts(
            texts,
            instruction=self._spec.document_instruction,
            name="document",
        )
        try:
            encoded = self._model.encode(
                prepared,
                batch_size=batch_size,
                show_progress_bar=self._show_progress_bar,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise EmbeddingBackendUnavailableError("document embedding failed") from error
        matrix = np.asarray(encoded, dtype=np.float32)
        return _validate_matrix(
            matrix,
            expected_rows=len(prepared),
            expected_dimension=self._spec.dimension,
            name="document embeddings",
        )

    def encode_query(self, text: str) -> FloatVector:
        if type(text) is not str or not text.strip():
            raise QueryEmbeddingError("q_sem must be a non-empty string")
        prepared = self._spec.query_instruction + text.strip()
        try:
            encoded = self._model.encode(
                [prepared],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise QueryEmbeddingError("query embedding failed") from error
        matrix = _validate_matrix(
            np.asarray(encoded, dtype=np.float32),
            expected_rows=1,
            expected_dimension=self._spec.dimension,
            name="query embedding",
        )
        return cast(FloatVector, matrix[0].copy())


def normalize_rows(matrix: FloatMatrix, *, name: str) -> FloatMatrix:
    """Validate and L2-normalize a float32 matrix without accepting zero rows."""

    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty matrix")
    if matrix.dtype != np.float32:
        matrix = matrix.astype(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains a non-finite value")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise ValueError(f"{name} contains a zero or invalid vector")
    normalized = matrix / norms[:, np.newaxis]
    return cast(FloatMatrix, np.asarray(normalized, dtype=np.float32))


def normalize_vector(vector: FloatVector, *, expected_dimension: int, name: str) -> FloatVector:
    """Validate and L2-normalize one float32 vector."""

    if vector.ndim != 1 or vector.shape != (expected_dimension,):
        raise ValueError(f"{name} has the wrong shape")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains a non-finite value")
    norm = float(np.linalg.norm(vector))
    if not math_is_finite_positive(norm):
        raise ValueError(f"{name} is zero or invalid")
    return cast(FloatVector, np.asarray(vector / norm, dtype=np.float32))


def math_is_finite_positive(value: float) -> bool:
    """Small helper kept separate for straightforward unit testing."""

    return bool(np.isfinite(value) and value > 0.0)


def _prepare_texts(texts: Sequence[str], *, instruction: str, name: str) -> list[str]:
    if not texts:
        raise ValueError(f"{name} texts must not be empty")
    prepared: list[str] = []
    for text in texts:
        if type(text) is not str or not text.strip():
            raise ValueError(f"{name} text must be a non-empty string")
        prepared.append(instruction + text.strip())
    return prepared


def _validate_matrix(
    matrix: NDArray[np.generic],
    *,
    expected_rows: int,
    expected_dimension: int,
    name: str,
) -> FloatMatrix:
    if matrix.shape != (expected_rows, expected_dimension):
        raise ValueError(
            f"{name} has shape {matrix.shape}, expected ({expected_rows}, {expected_dimension})"
        )
    return normalize_rows(np.asarray(matrix, dtype=np.float32), name=name)
