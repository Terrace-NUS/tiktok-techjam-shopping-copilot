"""Typed contracts for LLM-generated, source-grounded product fact cards."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum

_FACET_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_CONTENT_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class ProductFactPolarity(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"


@dataclass(frozen=True, slots=True, kw_only=True)
class ProductSourceItem:
    """One complete, addressable piece of a raw catalog row."""

    ref: str
    field: str
    text: str

    def __post_init__(self) -> None:
        if type(self.ref) is not str or not self.ref:
            raise ValueError("product source ref must be non-empty")
        if type(self.field) is not str or not self.field:
            raise ValueError("product source field must be non-empty")
        if type(self.text) is not str or not self.text:
            raise ValueError("product source text must be non-empty")


@dataclass(frozen=True, slots=True, kw_only=True)
class ProductFactRequest:
    """Complete model input for one immutable catalog product."""

    parent_asin: str
    source_id: str
    sources: tuple[ProductSourceItem, ...]

    def __post_init__(self) -> None:
        if type(self.parent_asin) is not str or not self.parent_asin.strip():
            raise ValueError("parent_asin must be non-empty")
        if type(self.source_id) is not str or _CONTENT_ID_PATTERN.fullmatch(self.source_id) is None:
            raise ValueError("source_id must be a sha256 content ID")
        if type(self.sources) is not tuple or not self.sources:
            raise ValueError("product fact request must contain sources")
        refs = tuple(item.ref for item in self.sources)
        if len(set(refs)) != len(refs):
            raise ValueError("product source refs must be unique")


@dataclass(frozen=True, slots=True, kw_only=True)
class ProductFact:
    """One atomic product assertion with an exact source citation."""

    facet: str
    value: str
    aliases: tuple[str, ...]
    polarity: ProductFactPolarity
    component: str | None
    meaning: str
    evidence: str
    source_ref: str
    confidence: float

    def __post_init__(self) -> None:
        if type(self.facet) is not str or _FACET_PATTERN.fullmatch(self.facet) is None:
            raise ValueError("product fact facet must be lower_snake_case")
        if type(self.value) is not str or not self.value.strip():
            raise ValueError("product fact value must be non-empty")
        if type(self.aliases) is not tuple or any(
            type(alias) is not str or not alias.strip() for alias in self.aliases
        ):
            raise TypeError("product fact aliases must be non-empty strings")
        normalized_aliases = tuple(alias.casefold().strip() for alias in self.aliases)
        if len(set(normalized_aliases)) != len(normalized_aliases):
            raise ValueError("product fact aliases must be unique")
        if type(self.polarity) is not ProductFactPolarity:
            raise TypeError("product fact polarity is invalid")
        if self.component is not None and (
            type(self.component) is not str or not self.component.strip()
        ):
            raise ValueError("product fact component must be non-empty or null")
        for name, text in (
            ("meaning", self.meaning),
            ("evidence", self.evidence),
            ("source_ref", self.source_ref),
        ):
            if type(text) is not str or not text.strip():
                raise ValueError(f"product fact {name} must be non-empty")
        if type(self.confidence) not in (int, float) or not math.isfinite(self.confidence):
            raise TypeError("product fact confidence must be finite")
        if not 0 <= self.confidence <= 1:
            raise ValueError("product fact confidence must be between zero and one")


@dataclass(frozen=True, slots=True, kw_only=True)
class ProductFactCard:
    """All extracted facts for one product before local release metadata is added."""

    parent_asin: str
    facts: tuple[ProductFact, ...]
    summary: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.parent_asin) is not str or not self.parent_asin.strip():
            raise ValueError("product fact card parent_asin must be non-empty")
        if type(self.facts) is not tuple:
            raise TypeError("product fact card facts must be a tuple")
        if type(self.summary) is not str or not self.summary.strip():
            raise ValueError("product fact card summary must be non-empty")
        if type(self.warnings) is not tuple or any(
            type(warning) is not str or not warning.strip() for warning in self.warnings
        ):
            raise TypeError("product fact card warnings must be non-empty strings")


@dataclass(frozen=True, slots=True, kw_only=True)
class ProductFactTrace:
    response_id: str | None
    model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ProductFactResult:
    card: ProductFactCard
    trace: ProductFactTrace
