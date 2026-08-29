"""Deterministic lexical evidence over compiled ``q_lex`` text.

This module intentionally does not depend on the editable starter agent.  It
owns a small FTS5 index and reports lexical evidence; it does not decide how
retrieval or intent clarity should behave.
"""

from __future__ import annotations

import math
import re
import sqlite3
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from .documents import DOCUMENT_FIELD_ORDER, ProductDocument

LEXICAL_PROBE_K = 80
LEXICAL_QUERY_TOKEN_LIMIT = 40

# FTS columns are deliberately ordered to preserve this fixed weight tuple.
_INDEX_FIELDS = ("title", "categories", "features", "details", "store", "description")
_FIELD_WEIGHTS = (6.0, 4.0, 2.5, 2.5, 1.5, 1.0)
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

LexicalUnavailableReason = Literal[
    "empty_query",
    "no_eligible_documents",
    "no_matches",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class LexicalHit:
    """One FTS result.  ``raw_bm25`` is better when numerically smaller."""

    parent_asin: str
    raw_bm25: float
    rank: int


@dataclass(frozen=True, slots=True, kw_only=True)
class LexicalProbeObservation:
    """Target-free lexical evidence for one compiled query."""

    probe_k: int
    tokens: tuple[str, ...]
    eligible_count: int
    matched_count: int
    matched_token_count: int
    mean_normalized_idf: float | None
    hits: tuple[LexicalHit, ...]
    available: bool
    reason: LexicalUnavailableReason | None


class LexicalProbe:
    """An immutable-corpus, deterministic SQLite FTS5 lexical probe."""

    def __init__(
        self,
        documents: Iterable[ProductDocument],
        *,
        probe_k: int = LEXICAL_PROBE_K,
    ) -> None:
        if type(probe_k) is not int or probe_k <= 0:
            raise ValueError("probe_k must be a positive integer")
        self.probe_k = probe_k
        self._connection = sqlite3.connect(":memory:")
        self._connection.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )

        rows: list[tuple[str, ...]] = []
        parent_asins: set[str] = set()
        for document in documents:
            if type(document) is not ProductDocument:
                raise TypeError("documents must contain exact ProductDocument values")
            if document.parent_asin in parent_asins:
                raise ValueError(f"duplicate parent_asin: {document.parent_asin!r}")
            parent_asins.add(document.parent_asin)
            fields = _parse_product_document(document)
            rows.append((document.parent_asin, *(fields[field] for field in _INDEX_FIELDS)))
        if not rows:
            raise ValueError("documents must not be empty")

        self._connection.executemany(
            "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._connection.execute(
            "CREATE VIRTUAL TABLE products_vocab USING fts5vocab(products, 'row')"
        )
        self._connection.commit()
        self._parent_asins = frozenset(parent_asins)

    @property
    def parent_asins(self) -> frozenset[str]:
        """Return the immutable catalog identity set bound to this probe."""

        return self._parent_asins

    def observe(
        self,
        q_lex: str,
        *,
        eligible_parent_asins: Iterable[str] | None = None,
    ) -> LexicalProbeObservation:
        """Observe ``q_lex``, filtering eligibility before the fixed Top-K cut."""

        if type(q_lex) is not str:
            raise TypeError("q_lex must be a string")
        tokens = _query_tokens(q_lex)
        eligible = self._resolve_eligibility(eligible_parent_asins)
        if not tokens:
            return self._unavailable(tokens, len(eligible), "empty_query")
        if not eligible:
            return self._unavailable(tokens, 0, "no_eligible_documents")

        expression = " OR ".join(f'"{token}"' for token in tokens)
        weights = ", ".join(str(weight) for weight in (0.0, *_FIELD_WEIGHTS))
        rows = self._connection.execute(
            "SELECT parent_asin, bm25(products, "
            + weights
            + ") AS lexical_score FROM products WHERE products MATCH ? "
            "ORDER BY lexical_score ASC, parent_asin ASC",
            (expression,),
        ).fetchall()
        # Filtering the complete ordered hit stream is intentionally before the
        # Top-K slice.  At the fixed 50k catalog scale this is simple and safe.
        matched = [
            (str(parent_asin), float(score))
            for parent_asin, score in rows
            if parent_asin in eligible
        ]
        if not matched:
            return self._unavailable(tokens, len(eligible), "no_matches")

        dfs = self._document_frequencies(tokens)
        matched_dfs = [dfs[token] for token in tokens if dfs[token] > 0]
        denominator = math.log(len(self._parent_asins) + 1)
        mean_normalized_idf = sum(
            math.log((len(self._parent_asins) + 1) / (df + 1)) / denominator for df in matched_dfs
        ) / len(matched_dfs)
        hits = tuple(
            LexicalHit(parent_asin=parent_asin, raw_bm25=score, rank=rank)
            for rank, (parent_asin, score) in enumerate(matched[: self.probe_k], start=1)
        )
        return LexicalProbeObservation(
            probe_k=self.probe_k,
            tokens=tokens,
            eligible_count=len(eligible),
            matched_count=len(matched),
            matched_token_count=len(matched_dfs),
            mean_normalized_idf=mean_normalized_idf,
            hits=hits,
            available=True,
            reason=None,
        )

    def _resolve_eligibility(self, eligible_parent_asins: Iterable[str] | None) -> frozenset[str]:
        if eligible_parent_asins is None:
            return self._parent_asins
        if isinstance(eligible_parent_asins, (str, bytes)):
            raise TypeError("eligible_parent_asins must be an iterable of product IDs")
        eligible = frozenset(eligible_parent_asins)
        if any(type(parent_asin) is not str or not parent_asin for parent_asin in eligible):
            raise ValueError("eligible_parent_asins contains an invalid ID")
        unknown = eligible - self._parent_asins
        if unknown:
            raise KeyError(f"unknown eligible parent_asin: {sorted(unknown)[0]}")
        return eligible

    def _document_frequencies(self, tokens: tuple[str, ...]) -> dict[str, int]:
        result: dict[str, int] = {}
        for token in tokens:
            row = self._connection.execute(
                "SELECT doc FROM products_vocab WHERE term = ?", (token,)
            ).fetchone()
            result[token] = 0 if row is None else int(row[0])
        return result

    def _unavailable(
        self,
        tokens: tuple[str, ...],
        eligible_count: int,
        reason: LexicalUnavailableReason,
    ) -> LexicalProbeObservation:
        dfs = self._document_frequencies(tokens) if tokens else {}
        return LexicalProbeObservation(
            probe_k=self.probe_k,
            tokens=tokens,
            eligible_count=eligible_count,
            matched_count=0,
            matched_token_count=sum(df > 0 for df in dfs.values()),
            mean_normalized_idf=None,
            hits=(),
            available=False,
            reason=reason,
        )


def _query_tokens(q_lex: str) -> tuple[str, ...]:
    tokens: list[str] = []
    seen: set[str] = set()
    for match in _TOKEN_RE.finditer(q_lex):
        token = _remove_diacritics(match.group(0)).casefold()
        if token and token not in seen:
            seen.add(token)
            tokens.append(token)
            if len(tokens) == LEXICAL_QUERY_TOKEN_LIMIT:
                break
    return tuple(tokens)


def _remove_diacritics(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFD", value)
        if unicodedata.category(character) != "Mn"
    )


def _parse_product_document(document: ProductDocument) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in document.text.splitlines():
        label, separator, value = line.partition(": ")
        if not separator or label not in DOCUMENT_FIELD_ORDER or label in fields:
            raise ValueError(f"malformed ProductDocument text for {document.parent_asin!r}")
        fields[label] = value
    if tuple(fields) != DOCUMENT_FIELD_ORDER:
        raise ValueError(f"malformed ProductDocument field order for {document.parent_asin!r}")
    return fields
