"""Read-only raw-catalog evidence used by deterministic hard-mask matching."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterator, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, TypeAlias, cast

from pyunormalize import NFKC  # type: ignore[import-untyped]

from shopping_copilot.catalog.semantic.canonical import content_id_for_value

RETRIEVAL_EVIDENCE_SCHEMA = "shopping-copilot/retrieval-evidence-index/v1"
RETRIEVAL_EVIDENCE_POLICY_ID = "raw_catalog_evidence_v1"
RETRIEVAL_EVIDENCE_PRODUCT_FACT_POLICY_ID = "raw_catalog_with_product_fact_evidence_v1"
RETRIEVAL_EVIDENCE_PRODUCT_FACT_REPLACEMENT_POLICY_ID = "product_fact_replacement_evidence_v1"

SUPPORTED_FACETS = (
    "brand",
    "color",
    "department",
    "feature",
    "gender",
    "material",
    "size",
    "style",
    "use_case",
)
_FACET_INDEX = MappingProxyType({facet: index for index, facet in enumerate(SUPPORTED_FACETS)})

_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_ALIASES = MappingProxyType(
    {
        "grey": "gray",
        "colour": "color",
        "womens": "women",
        "mens": "men",
    }
)

_BRAND_DETAIL_KEYS = frozenset({"brand", "brand name"})
_COLOR_DETAIL_KEYS = frozenset({"color"})
_DEPARTMENT_DETAIL_KEYS = frozenset({"department"})
_GENDER_DETAIL_KEYS = frozenset({"department", "target gender"})
_MATERIAL_DETAIL_KEYS = frozenset(
    {
        "material",
        "material type",
        "fabric",
        "fabric type",
        "outer material",
        "sole material",
    }
)
_SIZE_DETAIL_KEYS = frozenset(
    {
        "size",
        "size name",
        "size type",
        "apparel size",
        "band size",
        "chest size",
        "clothing size",
        "inseam",
        "item size",
        "neck size",
        "product size",
        "ring size",
        "shoe size",
        "sleeve size",
        "waist size",
    }
)
_STYLE_DETAIL_KEYS = frozenset({"style", "pattern", "theme", "closure type"})
_FEATURE_DETAIL_KEYS = frozenset({"special feature", "special features", "feature", "features"})
_USE_CASE_DETAIL_KEYS = frozenset(
    {
        "occasion",
        "recommended use",
        "recommended uses",
        "recommended uses for product",
        "specific uses for product",
        "usage",
    }
)

_CONTROLLED_GENDER_TOKENS = frozenset(
    {
        "baby",
        "babies",
        "boy",
        "boys",
        "child",
        "children",
        "female",
        "gentlemen",
        "girl",
        "girls",
        "infant",
        "infants",
        "kid",
        "kids",
        "ladies",
        "lady",
        "male",
        "man",
        "men",
        "toddler",
        "toddlers",
        "unisex",
        "woman",
        "women",
        "youth",
    }
)
_SIZE_MARKERS = frozenset(
    {"size", "sized", "sizes", "sizing", "waist", "inseam", "neck", "sleeve", "us", "uk", "eu"}
)
_SIZE_WORDS = frozenset(
    {
        "extra",
        "petite",
        "plus",
        "regular",
        "short",
        "small",
        "medium",
        "large",
        "tall",
        "wide",
        "narrow",
        "x",
        "xx",
        "xxx",
    }
)
_AUDIENCE_SIZE_CONTEXT = frozenset(
    {"baby", "boy", "boys", "girl", "girls", "kid", "kids", "men", "toddler", "women"}
)
_COMPACT_SIZE_PATTERN = re.compile(r"(?:xxxs|xxs|xs|xl|xxl|xxxl|xxxxl|\d+t)")
_NUMERIC_SIZE_PATTERN = re.compile(r"\d+")
_NEGATION_TOKENS = frozenset({"never", "no", "non", "not", "without"})
_NEGATION_SENSITIVE_FACETS = frozenset({"feature", "use_case"})

TokenSequence: TypeAlias = tuple[int, ...]
FacetSegments: TypeAlias = tuple[TokenSequence, ...]
ProductEvidence: TypeAlias = tuple[FacetSegments, ...]
FrozenPostings: TypeAlias = tuple[Mapping[int, tuple[int, ...]], ...]
_CONTENT_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


class RetrievalEvidenceError(ValueError):
    """Raised when raw catalog evidence cannot be indexed safely."""

    def __init__(self, message: str, *, line_number: int | None = None) -> None:
        self.line_number = line_number
        if line_number is not None:
            message = f"line {line_number}: {message}"
        super().__init__(message)


class _DuplicateJsonKeyError(ValueError):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievalEvidenceIndex:
    """Immutable token postings and source-bounded phrase evidence."""

    index_id: str
    catalog_id: str
    catalog_semantic_release_id: str
    policy_id: str
    parent_asins: tuple[str, ...]
    _vocabulary: Mapping[str, int] = field(repr=False, compare=False)
    _products: tuple[ProductEvidence, ...] = field(repr=False, compare=False)
    _postings: FrozenPostings = field(repr=False, compare=False)
    _negator_ids: frozenset[int] = field(repr=False, compare=False)

    def match(self, facet: str, value: str) -> frozenset[str]:
        """Return products with token-boundary evidence for one facet value."""

        if type(facet) is not str:
            raise TypeError("facet must be a string")
        facet_index = _FACET_INDEX.get(facet)
        if facet_index is None:
            raise ValueError(f"unknown facet: {facet!r}")
        if type(value) is not str:
            raise TypeError("value must be a string")
        query_tokens = _normalize_tokens(value)
        if not query_tokens:
            raise ValueError("value must contain at least one searchable token")

        query_ids: list[int] = []
        for token in query_tokens:
            token_id = self._vocabulary.get(token)
            if token_id is None:
                return frozenset()
            query_ids.append(token_id)
        phrase = tuple(query_ids)

        posting = self._postings[facet_index]
        candidate_sets = [posting.get(token_id, ()) for token_id in set(phrase)]
        if any(not candidates for candidates in candidate_sets):
            return frozenset()
        candidate_sets.sort(key=len)
        candidate_rows = set(candidate_sets[0])
        for candidates in candidate_sets[1:]:
            candidate_rows.intersection_update(candidates)
            if not candidate_rows:
                return frozenset()

        must_verify = len(phrase) > 1 or facet in _NEGATION_SENSITIVE_FACETS
        if must_verify:
            reject_negated = facet in _NEGATION_SENSITIVE_FACETS
            candidate_rows = {
                row
                for row in candidate_rows
                if any(
                    _contains_phrase(
                        segment,
                        phrase,
                        negator_ids=self._negator_ids if reject_negated else frozenset(),
                    )
                    for segment in self._products[row][facet_index]
                )
            }

        return frozenset(self.parent_asins[row] for row in candidate_rows)


def build_retrieval_evidence_index(
    catalog_path: str | Path,
    *,
    catalog_id: str,
    catalog_semantic_release_id: str,
    expected_parent_asins: AbstractSet[str] | None = None,
    facet_text_overrides: Mapping[str, Mapping[str, tuple[str, ...]]] | None = None,
    facet_text_override_mode: Literal["augment", "replace"] = "augment",
) -> RetrievalEvidenceIndex:
    """Build raw evidence with optional source-grounded product-fact additions."""

    _require_content_id(catalog_id, name="catalog_id")
    _require_content_id(
        catalog_semantic_release_id,
        name="catalog_semantic_release_id",
    )
    expected = _validate_expected_parent_asins(expected_parent_asins)
    overrides = _validate_facet_text_overrides(facet_text_overrides)
    if facet_text_override_mode not in {"augment", "replace"}:
        raise ValueError("facet_text_override_mode must be 'augment' or 'replace'")

    vocabulary: dict[str, int] = {}
    evidence_by_asin: dict[str, ProductEvidence] = {}
    evidence_id_by_asin: dict[str, str] = {}
    first_line_by_asin: dict[str, int] = {}

    source = Path(catalog_path)
    catalog_digest = hashlib.sha256()
    try:
        with source.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                catalog_digest.update(raw_line)
                row = _parse_row(raw_line, line_number=line_number)
                parent_asin = _parent_asin(row, line_number=line_number)
                previous_line = first_line_by_asin.get(parent_asin)
                if previous_line is not None:
                    raise RetrievalEvidenceError(
                        (
                            f"duplicate parent_asin {parent_asin!r}; "
                            f"first seen on line {previous_line}"
                        ),
                        line_number=line_number,
                    )

                facet_texts = _extract_facet_texts(row, line_number=line_number)
                product_overrides = overrides.get(parent_asin)
                if product_overrides is not None:
                    facet_texts = _apply_facet_text_overrides(
                        facet_texts,
                        product_overrides,
                        mode=facet_text_override_mode,
                    )
                evidence_id_by_asin[parent_asin] = content_id_for_value(
                    {
                        "parent_asin": parent_asin,
                        "facets": {
                            facet: list(facet_texts[index])
                            for index, facet in enumerate(SUPPORTED_FACETS)
                        },
                    }
                )
                row_cache: dict[str, TokenSequence] = {}
                evidence_by_asin[parent_asin] = tuple(
                    tuple(
                        _encode_cached_text(text, vocabulary=vocabulary, cache=row_cache)
                        for text in facet_texts[facet_index]
                    )
                    for facet_index in range(len(SUPPORTED_FACETS))
                )
                first_line_by_asin[parent_asin] = line_number
    except OSError as error:
        raise RetrievalEvidenceError(f"cannot read catalog: {error}") from error

    actual_catalog_id = f"sha256:{catalog_digest.hexdigest()}"
    if actual_catalog_id != catalog_id:
        raise RetrievalEvidenceError(
            f"catalog bytes do not match catalog_id: expected {catalog_id}, got {actual_catalog_id}"
        )

    if not evidence_by_asin:
        raise RetrievalEvidenceError("catalog must contain at least one product")

    unknown_overrides = sorted(set(overrides) - evidence_by_asin.keys())
    if unknown_overrides:
        raise RetrievalEvidenceError(
            f"facet evidence override names an unknown product: {unknown_overrides[0]}"
        )

    actual = frozenset(evidence_by_asin)
    if expected is not None and actual != expected:
        missing = expected - actual
        unexpected = actual - expected
        raise RetrievalEvidenceError(
            "catalog parent_asin set mismatch: "
            f"missing={len(missing)} {_stable_sample(missing)}, "
            f"unexpected={len(unexpected)} {_stable_sample(unexpected)}"
        )

    parent_asins = tuple(sorted(evidence_by_asin))
    products = tuple(evidence_by_asin[parent_asin] for parent_asin in parent_asins)
    del evidence_by_asin
    postings = _build_postings(products)
    if not overrides:
        policy_id = RETRIEVAL_EVIDENCE_POLICY_ID
    elif facet_text_override_mode == "replace":
        policy_id = RETRIEVAL_EVIDENCE_PRODUCT_FACT_REPLACEMENT_POLICY_ID
    else:
        policy_id = RETRIEVAL_EVIDENCE_PRODUCT_FACT_POLICY_ID
    index_id = content_id_for_value(
        {
            "schema": RETRIEVAL_EVIDENCE_SCHEMA,
            "catalog_id": catalog_id,
            "catalog_semantic_release_id": catalog_semantic_release_id,
            "policy_id": policy_id,
            "products": [
                {
                    "parent_asin": parent_asin,
                    "evidence_id": evidence_id_by_asin[parent_asin],
                }
                for parent_asin in parent_asins
            ],
        }
    )
    negator_ids = frozenset(vocabulary[token] for token in _NEGATION_TOKENS if token in vocabulary)
    return RetrievalEvidenceIndex(
        index_id=index_id,
        catalog_id=catalog_id,
        catalog_semantic_release_id=catalog_semantic_release_id,
        policy_id=policy_id,
        parent_asins=parent_asins,
        _vocabulary=MappingProxyType(dict(vocabulary)),
        _products=products,
        _postings=postings,
        _negator_ids=negator_ids,
    )


def _validate_facet_text_overrides(
    value: Mapping[str, Mapping[str, tuple[str, ...]]] | None,
) -> Mapping[str, Mapping[str, tuple[str, ...]]]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise TypeError("facet_text_overrides must be a mapping or None")
    result: dict[str, Mapping[str, tuple[str, ...]]] = {}
    for parent_asin, by_facet in value.items():
        if type(parent_asin) is not str or not parent_asin.strip():
            raise ValueError("facet evidence override has an invalid product ID")
        if not isinstance(by_facet, Mapping):
            raise TypeError("facet evidence product override must be a mapping")
        product: dict[str, tuple[str, ...]] = {}
        for facet, texts in by_facet.items():
            if facet not in _FACET_INDEX:
                raise ValueError(f"facet evidence override names an unknown facet: {facet!r}")
            if type(texts) is not tuple:
                raise TypeError("facet evidence override values must be tuples")
            if any(type(text) is not str or not text.strip() for text in texts):
                raise ValueError("facet evidence override contains invalid text")
            product[facet] = tuple(text.strip() for text in texts)
        result[parent_asin] = MappingProxyType(product)
    return MappingProxyType(result)


def _apply_facet_text_overrides(
    original: tuple[tuple[str, ...], ...],
    overrides: Mapping[str, tuple[str, ...]],
    *,
    mode: Literal["augment", "replace"],
) -> tuple[tuple[str, ...], ...]:
    values = list(original)
    for facet, texts in overrides.items():
        facet_index = _FACET_INDEX[facet]
        source = texts if mode == "replace" else (*values[facet_index], *texts)
        values[facet_index] = _canonical_segments(source)
    return tuple(values)


def _extract_facet_texts(
    row: dict[str, object],
    *,
    line_number: int,
) -> tuple[tuple[str, ...], ...]:
    title = _optional_string(row, "title", line_number=line_number)
    store = _optional_string(row, "store", line_number=line_number, nullable=True)
    categories = _optional_string_list(row, "categories", line_number=line_number)
    features = _optional_string_list(row, "features", line_number=line_number)
    description = _optional_string_list(row, "description", line_number=line_number)
    details = _optional_details(row, line_number=line_number)

    title_values = () if title is None else (title,)
    store_values = () if store is None else (store,)
    gender_sources = (
        *_detail_values(details, _GENDER_DETAIL_KEYS),
        *categories,
    )
    gender_values = tuple(_controlled_gender_text(value) for value in gender_sources)

    by_facet: dict[str, tuple[str, ...]] = {
        "brand": _canonical_segments((*store_values, *_detail_values(details, _BRAND_DETAIL_KEYS))),
        "color": _canonical_segments(
            (*_detail_values(details, _COLOR_DETAIL_KEYS), *title_values, *features)
        ),
        "department": _canonical_segments(
            (*_detail_values(details, _DEPARTMENT_DETAIL_KEYS), *categories)
        ),
        "feature": _canonical_segments(
            (*features, *_detail_values(details, _FEATURE_DETAIL_KEYS), *description)
        ),
        "gender": _canonical_segments(gender_values),
        "material": _canonical_segments(
            (*_detail_values(details, _MATERIAL_DETAIL_KEYS), *title_values, *features)
        ),
        "size": _canonical_segments(
            (*_detail_values(details, _SIZE_DETAIL_KEYS), *_size_title_segments(title))
        ),
        "style": _canonical_segments(
            (
                *_detail_values(details, _STYLE_DETAIL_KEYS),
                *title_values,
                *features,
                *categories,
            )
        ),
        "use_case": _canonical_segments(
            (
                *categories,
                *features,
                *description,
                *_detail_values(details, _USE_CASE_DETAIL_KEYS),
            )
        ),
    }
    return tuple(by_facet[facet] for facet in SUPPORTED_FACETS)


def _detail_values(details: dict[str, object], allowed_keys: frozenset[str]) -> tuple[str, ...]:
    values: list[str] = []
    for raw_key, raw_value in details.items():
        if _normalize_text(raw_key) in allowed_keys:
            values.extend(_flatten_detail_value(raw_value))
    return tuple(values)


def _flatten_detail_value(value: object) -> Iterator[str]:
    if value is None:
        return
    if type(value) is str:
        yield value
        return
    if type(value) is bool:
        yield "true" if value else "false"
        return
    if type(value) is int:
        yield str(value)
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise RetrievalEvidenceError("detail value must be finite")
        yield json.dumps(value, ensure_ascii=False, allow_nan=False)
        return
    if type(value) is list:
        for item in cast(list[object], value):
            yield from _flatten_detail_value(item)
        return
    if type(value) is dict:
        mapping = cast(dict[str, object], value)
        for key in sorted(mapping):
            yield from _flatten_detail_value(mapping[key])
        return
    raise RetrievalEvidenceError(f"unsupported detail value type: {type(value).__name__}")


def _size_title_segments(title: str | None) -> tuple[str, ...]:
    if title is None:
        return ()
    tokens = _normalize_tokens(title)
    segments: set[str] = set()
    for index, token in enumerate(tokens):
        if _COMPACT_SIZE_PATTERN.fullmatch(token) is not None:
            segments.add(token)

        if token in _SIZE_MARKERS:
            captured = [token]
            for candidate in tokens[index + 1 : index + 4]:
                if (
                    _NUMERIC_SIZE_PATTERN.fullmatch(candidate) is not None
                    or candidate in _SIZE_WORDS
                ):
                    captured.append(candidate)
                else:
                    break
            if len(captured) > 1:
                segments.add(" ".join(captured))
            if token == "size" and index > 0 and tokens[index - 1] == "one":
                segments.add("one size")

        if token in {"small", "medium", "large"}:
            context = tokens[max(0, index - 2) : index]
            if any(item in _AUDIENCE_SIZE_CONTEXT for item in context):
                segments.add(token)
    return tuple(sorted(segments))


def _controlled_gender_text(value: str) -> str:
    return " ".join(
        token for token in _normalize_tokens(value) if token in _CONTROLLED_GENDER_TOKENS
    )


def _canonical_segments(values: tuple[str, ...]) -> tuple[str, ...]:
    segments: set[str] = set()
    for value in values:
        normalized = _normalize_text(value)
        if normalized:
            segments.add(normalized)
    return tuple(sorted(segments))


def _normalize_text(value: str) -> str:
    return " ".join(_normalize_tokens(value))


def _normalize_tokens(value: str) -> tuple[str, ...]:
    normalized = NFKC(value).casefold()
    return tuple(_ALIASES.get(token, token) for token in _TOKEN_PATTERN.findall(normalized))


def _encode_text(text: str, vocabulary: dict[str, int]) -> TokenSequence:
    encoded: list[int] = []
    for token in text.split():
        token_id = vocabulary.get(token)
        if token_id is None:
            token_id = len(vocabulary)
            vocabulary[token] = token_id
        encoded.append(token_id)
    return tuple(encoded)


def _encode_cached_text(
    text: str,
    *,
    vocabulary: dict[str, int],
    cache: dict[str, TokenSequence],
) -> TokenSequence:
    encoded = cache.get(text)
    if encoded is None:
        encoded = _encode_text(text, vocabulary)
        cache[text] = encoded
    return encoded


def _build_postings(products: tuple[ProductEvidence, ...]) -> FrozenPostings:
    frozen: list[Mapping[int, tuple[int, ...]]] = []
    for facet_index in range(len(SUPPORTED_FACETS)):
        mutable: defaultdict[int, set[int]] = defaultdict(set)
        for row, product in enumerate(products):
            segments = product[facet_index]
            observed_tokens = {token_id for segment in segments for token_id in segment}
            for token_id in observed_tokens:
                mutable[token_id].add(row)
        compact: dict[int, tuple[int, ...]] = {}
        while mutable:
            token_id, rows = mutable.popitem()
            compact[token_id] = tuple(sorted(rows))
        frozen.append(MappingProxyType(compact))
    return tuple(frozen)


def _contains_phrase(
    segment: TokenSequence,
    phrase: TokenSequence,
    *,
    negator_ids: frozenset[int],
) -> bool:
    width = len(phrase)
    for start in range(len(segment) - width + 1):
        if segment[start : start + width] != phrase:
            continue
        preceding = segment[max(0, start - 3) : start]
        if not negator_ids.intersection(preceding):
            return True
    return False


def _parse_row(raw_line: bytes, *, line_number: int) -> dict[str, object]:
    if not raw_line.strip():
        raise RetrievalEvidenceError("blank JSONL row", line_number=line_number)
    try:
        text = raw_line.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RetrievalEvidenceError("row is not valid UTF-8", line_number=line_number) from error
    try:
        parsed: object = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonstandard_number,
            parse_float=_parse_finite_float,
        )
    except _DuplicateJsonKeyError as error:
        raise RetrievalEvidenceError(
            f"duplicate JSON key {error.args[0]!r}", line_number=line_number
        ) from error
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise RetrievalEvidenceError("invalid JSON", line_number=line_number) from error
    if type(parsed) is not dict:
        raise RetrievalEvidenceError("JSONL row must be an object", line_number=line_number)
    return cast(dict[str, object], parsed)


def _parent_asin(row: dict[str, object], *, line_number: int) -> str:
    value = row.get("parent_asin")
    if type(value) is not str or not value or value != value.strip():
        raise RetrievalEvidenceError(
            "parent_asin must be a non-empty trimmed string", line_number=line_number
        )
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise RetrievalEvidenceError(
            "parent_asin contains a lone surrogate", line_number=line_number
        )
    return value


def _optional_string(
    row: dict[str, object],
    field_name: str,
    *,
    line_number: int,
    nullable: bool = False,
) -> str | None:
    value = row.get(field_name)
    if value is None and (nullable or field_name not in row):
        return None
    if type(value) is not str:
        raise RetrievalEvidenceError(f"{field_name} must be a string", line_number=line_number)
    return value


def _optional_string_list(
    row: dict[str, object],
    field_name: str,
    *,
    line_number: int,
) -> tuple[str, ...]:
    value = row.get(field_name)
    if value is None and field_name not in row:
        return ()
    if type(value) is not list:
        raise RetrievalEvidenceError(f"{field_name} must be an array", line_number=line_number)
    values = cast(list[object], value)
    for index, item in enumerate(values):
        if type(item) is not str:
            raise RetrievalEvidenceError(
                f"{field_name}[{index}] must be a string", line_number=line_number
            )
    return tuple(cast(list[str], values))


def _optional_details(row: dict[str, object], *, line_number: int) -> dict[str, object]:
    value = row.get("details")
    if value is None and "details" not in row:
        return {}
    if type(value) is not dict:
        raise RetrievalEvidenceError("details must be an object", line_number=line_number)
    return cast(dict[str, object], value)


def _validate_expected_parent_asins(
    values: AbstractSet[str] | None,
) -> frozenset[str] | None:
    if values is None:
        return None
    if not isinstance(values, AbstractSet):
        raise TypeError("expected_parent_asins must be a set")
    for value in values:
        if type(value) is not str:
            raise TypeError("expected_parent_asins must contain only strings")
        if not value or value != value.strip():
            raise ValueError("expected_parent_asins must contain trimmed non-empty strings")
    return frozenset(values)


def _require_content_id(value: object, *, name: str) -> str:
    if type(value) is not str or _CONTENT_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a full sha256 content ID")
    return value


def _stable_sample(values: AbstractSet[str]) -> str:
    sample = sorted(values)[:5]
    suffix = ", ..." if len(values) > len(sample) else ""
    return "[" + ", ".join(repr(value) for value in sample) + suffix + "]"


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_nonstandard_number(value: str) -> object:
    raise ValueError(f"non-standard JSON number: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number: {value}")
    return parsed
