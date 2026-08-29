"""Expected failures at the Query Compiler boundary."""

from __future__ import annotations


class QueryCompilerError(ValueError):
    """Raised when catalog-bound intent cannot be compiled safely."""
