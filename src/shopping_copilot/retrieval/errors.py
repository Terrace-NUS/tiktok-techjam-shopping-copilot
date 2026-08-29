"""Retrieval-specific failures exposed by the dense vertical slice."""

from __future__ import annotations


class RetrievalError(Exception):
    """Base class for expected retrieval failures."""


class DenseIndexIntegrityError(RetrievalError):
    """Raised when a dense-index bundle fails closed validation."""


class DenseIndexBusyError(RetrievalError):
    """Raised when another process is publishing the same index."""


class EmbeddingBackendUnavailableError(RetrievalError):
    """Raised when the configured embedding backend cannot be loaded."""


class RankingBackendUnavailableError(RetrievalError):
    """Raised when an optional semantic ranking backend cannot run."""


class QueryEmbeddingError(RetrievalError):
    """Raised when a query cannot be converted into a valid vector."""


class CompiledQueryNotSearchableError(RetrievalError):
    """Raised when an empty compiled intent is sent to a retrieval route."""


class CompiledQueryBindingError(RetrievalError):
    """Raised when a compiled query belongs to a different catalog release."""
