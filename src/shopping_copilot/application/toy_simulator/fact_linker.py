from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*", re.IGNORECASE)
_SINGLE_TOKEN_WRAPPER_WORDS = frozenset(
    {
        "additional",
        "also",
        "and",
        "are",
        "at",
        "change",
        "details",
        "different",
        "direction",
        "earlier",
        "for",
        "here",
        "ignore",
        "in",
        "instead",
        "is",
        "matter",
        "matters",
        "my",
        "need",
        "now",
        "option",
        "preference",
        "requirement",
        "right",
        "something",
        "the",
        "want",
    }
)


def fact_tokens(value: object) -> tuple[str, ...]:
    """Return the punctuation-insensitive token key used for catalog fact linking."""

    return tuple(match.group(0).casefold() for match in _TOKEN_RE.finditer(str(value or "")))


@dataclass(frozen=True, slots=True)
class CatalogFactEntry:
    """One participant-visible catalog fact addressable by its original wording."""

    text: str
    posting_size: int
    structured: bool = False


@dataclass(frozen=True, slots=True)
class CatalogFactMatch:
    """A non-overlapping catalog fact grounded in a user-message token span."""

    text: str
    token_start: int
    token_end: int
    posting_size: int


class CatalogFactLinker:
    """Find verbatim catalog facts without depending on simulator prompt wrappers."""

    def __init__(
        self,
        entries: Mapping[tuple[str, ...], CatalogFactEntry],
        *,
        max_single_token_postings: int = 250,
    ) -> None:
        self._entries = dict(entries)
        self._max_fact_tokens = max((len(key) for key in self._entries), default=0)
        self._max_single_token_postings = max_single_token_postings

    def link(self, message: str, *, limit: int = 8) -> list[CatalogFactMatch]:
        if limit < 1:
            return []
        token_matches = list(_TOKEN_RE.finditer(message))
        tokens = tuple(match.group(0).casefold() for match in token_matches)
        if not tokens or not self._entries:
            return []

        candidates: list[CatalogFactMatch] = []
        for start in range(len(tokens)):
            max_end = min(len(tokens), start + self._max_fact_tokens)
            for end in range(start + 1, max_end + 1):
                key = tokens[start:end]
                entry = self._entries.get(key)
                if entry is None:
                    continue
                if len(key) == 1 and not self._accept_single_token(key[0], entry):
                    continue
                candidates.append(
                    CatalogFactMatch(
                        text=message[token_matches[start].start() : token_matches[end - 1].end()],
                        token_start=start,
                        token_end=end,
                        posting_size=entry.posting_size,
                    )
                )

        # Prefer the most specific catalog phrase, then the rarer fact. Selecting
        # non-overlapping spans prevents a long original attribute from also
        # producing several weaker nested attributes.
        candidates.sort(
            key=lambda match: (
                -(match.token_end - match.token_start),
                match.posting_size,
                match.token_start,
                match.text.casefold(),
            )
        )
        selected: list[CatalogFactMatch] = []
        occupied: set[int] = set()
        for candidate in candidates:
            span = set(range(candidate.token_start, candidate.token_end))
            if occupied & span:
                continue
            selected.append(candidate)
            occupied.update(span)
            if len(selected) >= limit:
                break
        selected.sort(key=lambda match: (match.token_start, match.token_end))
        return selected

    def _accept_single_token(self, token: str, entry: CatalogFactEntry) -> bool:
        if token in _SINGLE_TOKEN_WRAPPER_WORDS:
            return False
        return entry.structured or entry.posting_size <= self._max_single_token_postings


__all__ = [
    "CatalogFactEntry",
    "CatalogFactLinker",
    "CatalogFactMatch",
    "fact_tokens",
]
