"""Streaming, deterministic profiler for the frozen raw JSONL catalog."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeVar, cast

from .models import (
    CatalogProfile,
    CategoryDetailCoverage,
    CategoryNode,
    DetailKeyProfile,
    DiagnosticProfile,
    DiagnosticSample,
    ProductCategoryAssignment,
    ProfileConfig,
    TopLevelFieldProfile,
    TypeCount,
    ValueMass,
    ValueSample,
)

AssignmentSink = Callable[[ProductCategoryAssignment], None]

_SCHEMA_VERSION = "raw-catalog-profile-v1"
_MISSING = object()
_SAMPLE_EXCERPT_CHARS = 512

_SampleT = TypeVar("_SampleT")
_SampleRank = tuple[str, str, int, str]


class CatalogChangedError(RuntimeError):
    """Raised if the catalog bytes change between hashing and profiling."""


class _DuplicateJsonKeyError(ValueError):
    """Internal signal for an object that cannot be audited losslessly."""


class _BoundedSampler(Generic[_SampleT]):
    """Retain the lexicographically smallest stable sample ranks."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._entries: list[tuple[_SampleRank, _SampleT]] = []

    def consider(self, rank: _SampleRank, value: _SampleT) -> None:
        if self._limit == 0:
            return
        entry = (rank, value)
        if len(self._entries) < self._limit:
            bisect.insort(self._entries, entry, key=lambda item: item[0])
            return
        if rank >= self._entries[-1][0]:
            return
        bisect.insort(self._entries, entry, key=lambda item: item[0])
        self._entries.pop()

    def values(self) -> tuple[_SampleT, ...]:
        return tuple(value for _, value in self._entries)


@dataclass(slots=True)
class _FieldAccumulator:
    present_count: int = 0
    null_count: int = 0
    empty_count: int = 0
    type_counts: Counter[str] = field(default_factory=Counter)


@dataclass(slots=True)
class _DetailAccumulator:
    sampler: _BoundedSampler[ValueSample]
    support_count: int = 0
    nonempty_count: int = 0
    null_count: int = 0
    empty_count: int = 0
    type_counts: Counter[str] = field(default_factory=Counter)
    value_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    distinct_nonempty_values: set[tuple[str, str]] = field(default_factory=set)


@dataclass(slots=True)
class _DiagnosticAccumulator:
    sampler: _BoundedSampler[DiagnosticSample]
    count: int = 0


@dataclass(slots=True)
class _State:
    catalog_hash: str
    config: ProfileConfig
    physical_line_count: int = 0
    product_row_count: int = 0
    invalid_record_count: int = 0
    category_assignment_count: int = 0
    valid_category_assignment_count: int = 0
    product_row_with_diagnostics_count: int = 0
    top_fields: dict[str, _FieldAccumulator] = field(default_factory=dict)
    details: dict[str, _DetailAccumulator] = field(default_factory=dict)
    direct_category_support: Counter[tuple[str, ...]] = field(default_factory=Counter)
    subtree_category_support: Counter[tuple[str, ...]] = field(default_factory=Counter)
    category_key_present: Counter[tuple[tuple[str, ...], str]] = field(default_factory=Counter)
    category_key_nonempty: Counter[tuple[tuple[str, ...], str]] = field(default_factory=Counter)
    diagnostics: dict[str, _DiagnosticAccumulator] = field(default_factory=dict)
    seen_parent_asins: set[str] = field(default_factory=set)


def canonical_json_dumps(value: object) -> str:
    """Serialize a JSON-compatible value into the profiler's canonical form."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def category_id_for_path(path: Sequence[str]) -> str:
    """Return SHA-256 of the canonical JSON array for one exact raw path."""

    copied = tuple(path)
    if not copied or any(type(component) is not str for component in copied):
        raise ValueError("category path must be a non-empty sequence of strings")
    payload = canonical_json_dumps(list(copied)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def catalog_file_sha256(path: str | Path) -> str:
    """Hash the exact raw catalog bytes without opening the file for writing."""

    digest, _ = _hash_file(Path(path))
    return digest


def profile_catalog(
    path: str | Path,
    *,
    config: ProfileConfig | None = None,
    assignment_sink: AssignmentSink | None = None,
) -> CatalogProfile:
    """Profile a JSONL catalog as a deterministic, read-only line stream.

    The file is read twice: once to bind all stable samples to the exact raw
    SHA-256, and once as a binary line stream.  A second digest verifies that
    the source did not change between those passes.  Stable samples are
    bounded; exact distinct-value and top-value counts use memory proportional
    to the number of distinct raw detail values observed.
    """

    source = Path(path)
    effective_config = ProfileConfig() if config is None else config
    if type(effective_config) is not ProfileConfig:
        raise TypeError("config must be a ProfileConfig")
    catalog_hash, file_size = _hash_file(source)
    state = _State(catalog_hash=catalog_hash, config=effective_config)
    verification_hash = hashlib.sha256()
    verification_size = 0

    with source.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            state.physical_line_count += 1
            verification_hash.update(raw_line)
            verification_size += len(raw_line)
            _consume_line(
                state,
                raw_line=raw_line,
                line_number=line_number,
                assignment_sink=assignment_sink,
            )

    if verification_hash.hexdigest() != catalog_hash or verification_size != file_size:
        raise CatalogChangedError("catalog changed between hashing and profiling")
    return _finalize(state, file_size=file_size)


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
    return digest.hexdigest(), byte_count


def _consume_line(
    state: _State,
    *,
    raw_line: bytes,
    line_number: int,
    assignment_sink: AssignmentSink | None,
) -> None:
    if not raw_line.strip():
        state.invalid_record_count += 1
        _record_diagnostic(
            state,
            code="blank_line",
            parent_asin=None,
            line_number=line_number,
            raw_value_json=canonical_json_dumps(""),
        )
        return

    try:
        text = raw_line.decode("utf-8")
    except UnicodeDecodeError:
        state.invalid_record_count += 1
        _record_diagnostic(
            state,
            code="invalid_utf8",
            parent_asin=None,
            line_number=line_number,
            raw_value_json=canonical_json_dumps(
                {"hex_excerpt": raw_line[:_SAMPLE_EXCERPT_CHARS].hex()}
            ),
        )
        return

    try:
        parsed: object = json.loads(
            text,
            parse_constant=_reject_nonstandard_number,
            parse_float=_parse_finite_float,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except _DuplicateJsonKeyError:
        state.invalid_record_count += 1
        _record_diagnostic(
            state,
            code="duplicate_json_key",
            parent_asin=None,
            line_number=line_number,
            raw_value_json=canonical_json_dumps(text[:_SAMPLE_EXCERPT_CHARS]),
        )
        return
    except (json.JSONDecodeError, RecursionError, ValueError):
        state.invalid_record_count += 1
        _record_diagnostic(
            state,
            code="invalid_json",
            parent_asin=None,
            line_number=line_number,
            raw_value_json=canonical_json_dumps(text[:_SAMPLE_EXCERPT_CHARS]),
        )
        return

    try:
        canonical_json_dumps(parsed).encode("utf-8")
    except (RecursionError, UnicodeEncodeError, ValueError):
        state.invalid_record_count += 1
        _record_diagnostic(
            state,
            code="non_canonical_json_value",
            parent_asin=None,
            line_number=line_number,
            raw_value_json=canonical_json_dumps(text[:_SAMPLE_EXCERPT_CHARS]),
        )
        return

    if type(parsed) is not dict:
        state.invalid_record_count += 1
        _record_diagnostic(
            state,
            code="row_not_object",
            parent_asin=None,
            line_number=line_number,
            raw_value_json=canonical_json_dumps(parsed),
        )
        return

    row = cast(dict[str, object], parsed)
    state.product_row_count += 1
    state.category_assignment_count += 1
    _consume_product(
        state,
        row=row,
        line_number=line_number,
        assignment_sink=assignment_sink,
    )


def _consume_product(
    state: _State,
    *,
    row: dict[str, object],
    line_number: int,
    assignment_sink: AssignmentSink | None,
) -> None:
    for field_name, value in row.items():
        accumulator = state.top_fields.setdefault(field_name, _FieldAccumulator())
        accumulator.present_count += 1
        accumulator.type_counts[_json_value_type(value)] += 1
        if value is None:
            accumulator.null_count += 1
        if _is_empty(value):
            accumulator.empty_count += 1

    row_diagnostics: list[str] = []
    parent_asin = _validated_parent_asin(
        state,
        row=row,
        line_number=line_number,
        row_diagnostics=row_diagnostics,
    )
    present_detail_keys, nonempty_detail_keys = _consume_details(
        state,
        row=row,
        parent_asin=parent_asin,
        line_number=line_number,
        row_diagnostics=row_diagnostics,
    )
    raw_categories = row.get("categories", _MISSING)
    raw_categories_json = (
        None if raw_categories is _MISSING else canonical_json_dumps(raw_categories)
    )
    raw_path = _validated_category_path(
        state,
        raw_categories=raw_categories,
        parent_asin=parent_asin,
        line_number=line_number,
        row_diagnostics=row_diagnostics,
    )

    category_node_ids: tuple[str, ...] = ()
    leaf_category_id: str | None = None
    if raw_path:
        prefixes = tuple(raw_path[:index] for index in range(1, len(raw_path) + 1))
        category_node_ids = tuple(category_id_for_path(prefix) for prefix in prefixes)
        leaf_category_id = category_node_ids[-1]
        state.direct_category_support[raw_path] += 1
        for prefix in prefixes:
            state.subtree_category_support[prefix] += 1
            for raw_key in present_detail_keys:
                state.category_key_present[(prefix, raw_key)] += 1
            for raw_key in nonempty_detail_keys:
                state.category_key_nonempty[(prefix, raw_key)] += 1
        state.valid_category_assignment_count += 1

    if row_diagnostics:
        state.product_row_with_diagnostics_count += 1

    if assignment_sink is not None:
        assignment_sink(
            ProductCategoryAssignment(
                line_number=line_number,
                parent_asin=parent_asin,
                raw_categories_json=raw_categories_json,
                raw_path=raw_path,
                category_node_ids=category_node_ids,
                leaf_category_id=leaf_category_id,
                category_valid=bool(raw_path),
                diagnostics=tuple(sorted(set(row_diagnostics))),
            )
        )


def _validated_parent_asin(
    state: _State,
    *,
    row: dict[str, object],
    line_number: int,
    row_diagnostics: list[str],
) -> str | None:
    raw_parent_asin = row.get("parent_asin", _MISSING)
    if raw_parent_asin is _MISSING:
        _add_row_diagnostic(
            state,
            row_diagnostics=row_diagnostics,
            code="parent_asin_missing",
            parent_asin=None,
            line_number=line_number,
            raw_value=_MISSING,
        )
        return None
    if type(raw_parent_asin) is not str or not raw_parent_asin.strip():
        _add_row_diagnostic(
            state,
            row_diagnostics=row_diagnostics,
            code="parent_asin_invalid",
            parent_asin=None,
            line_number=line_number,
            raw_value=raw_parent_asin,
        )
        return None

    parent_asin = raw_parent_asin
    if parent_asin in state.seen_parent_asins:
        _add_row_diagnostic(
            state,
            row_diagnostics=row_diagnostics,
            code="parent_asin_duplicate",
            parent_asin=parent_asin,
            line_number=line_number,
            raw_value=parent_asin,
        )
    else:
        state.seen_parent_asins.add(parent_asin)
    return parent_asin


def _consume_details(
    state: _State,
    *,
    row: dict[str, object],
    parent_asin: str | None,
    line_number: int,
    row_diagnostics: list[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    raw_details = row.get("details", _MISSING)
    if raw_details is _MISSING:
        _add_row_diagnostic(
            state,
            row_diagnostics=row_diagnostics,
            code="details_missing",
            parent_asin=parent_asin,
            line_number=line_number,
            raw_value=_MISSING,
        )
        return (), ()
    if type(raw_details) is not dict:
        _add_row_diagnostic(
            state,
            row_diagnostics=row_diagnostics,
            code="details_not_object",
            parent_asin=parent_asin,
            line_number=line_number,
            raw_value=raw_details,
        )
        return (), ()

    details = cast(dict[str, object], raw_details)
    present_keys: list[str] = []
    nonempty_keys: list[str] = []
    for raw_key in sorted(details):
        value = details[raw_key]
        present_keys.append(raw_key)
        accumulator = state.details.get(raw_key)
        if accumulator is None:
            accumulator = _DetailAccumulator(
                sampler=_BoundedSampler[ValueSample](state.config.sample_limit)
            )
            state.details[raw_key] = accumulator
        accumulator.support_count += 1
        value_type = _json_value_type(value)
        accumulator.type_counts[value_type] += 1
        canonical_value = canonical_json_dumps(value)
        value_identity = (canonical_value, value_type)
        accumulator.value_counts[value_identity] += 1
        if value is None:
            accumulator.null_count += 1
        elif _is_empty(value):
            accumulator.empty_count += 1
        else:
            accumulator.nonempty_count += 1
            accumulator.distinct_nonempty_values.add(value_identity)
            nonempty_keys.append(raw_key)

        sample_hash, parent_token = _stable_sample_hash(
            catalog_hash=state.catalog_hash,
            seed=state.config.seed,
            parent_asin=parent_asin,
            line_number=line_number,
            key=raw_key,
        )
        sample = ValueSample(
            sample_hash=sample_hash,
            parent_asin=parent_asin,
            line_number=line_number,
            canonical_value_json=canonical_value,
            value_type=value_type,
        )
        rank: _SampleRank = (sample_hash, parent_token, line_number, canonical_value)
        accumulator.sampler.consider(rank, sample)
    return tuple(present_keys), tuple(nonempty_keys)


def _validated_category_path(
    state: _State,
    *,
    raw_categories: object,
    parent_asin: str | None,
    line_number: int,
    row_diagnostics: list[str],
) -> tuple[str, ...]:
    if raw_categories is _MISSING:
        _add_row_diagnostic(
            state,
            row_diagnostics=row_diagnostics,
            code="categories_missing",
            parent_asin=parent_asin,
            line_number=line_number,
            raw_value=_MISSING,
        )
        return ()
    if type(raw_categories) is not list:
        _add_row_diagnostic(
            state,
            row_diagnostics=row_diagnostics,
            code="categories_not_array",
            parent_asin=parent_asin,
            line_number=line_number,
            raw_value=raw_categories,
        )
        return ()

    components = cast(list[object], raw_categories)
    if not components:
        _add_row_diagnostic(
            state,
            row_diagnostics=row_diagnostics,
            code="categories_empty",
            parent_asin=parent_asin,
            line_number=line_number,
            raw_value=raw_categories,
        )
        return ()
    has_non_string = any(type(component) is not str for component in components)
    has_empty_string = any(
        type(component) is str and not component.strip() for component in components
    )
    if has_non_string:
        _add_row_diagnostic(
            state,
            row_diagnostics=row_diagnostics,
            code="category_component_not_string",
            parent_asin=parent_asin,
            line_number=line_number,
            raw_value=raw_categories,
        )
    if has_empty_string:
        _add_row_diagnostic(
            state,
            row_diagnostics=row_diagnostics,
            code="category_component_empty",
            parent_asin=parent_asin,
            line_number=line_number,
            raw_value=raw_categories,
        )
    if has_non_string or has_empty_string:
        return ()
    return tuple(cast(list[str], components))


def _add_row_diagnostic(
    state: _State,
    *,
    row_diagnostics: list[str],
    code: str,
    parent_asin: str | None,
    line_number: int,
    raw_value: object,
) -> None:
    row_diagnostics.append(code)
    raw_value_json = (
        canonical_json_dumps("<missing>")
        if raw_value is _MISSING
        else canonical_json_dumps(raw_value)
    )
    _record_diagnostic(
        state,
        code=code,
        parent_asin=parent_asin,
        line_number=line_number,
        raw_value_json=raw_value_json,
    )


def _record_diagnostic(
    state: _State,
    *,
    code: str,
    parent_asin: str | None,
    line_number: int,
    raw_value_json: str,
) -> None:
    accumulator = state.diagnostics.get(code)
    if accumulator is None:
        accumulator = _DiagnosticAccumulator(
            sampler=_BoundedSampler[DiagnosticSample](state.config.sample_limit)
        )
        state.diagnostics[code] = accumulator
    accumulator.count += 1
    sample_hash, parent_token = _stable_sample_hash(
        catalog_hash=state.catalog_hash,
        seed=state.config.seed,
        parent_asin=parent_asin,
        line_number=line_number,
        key=f"diagnostic:{code}",
    )
    sample = DiagnosticSample(
        sample_hash=sample_hash,
        parent_asin=parent_asin,
        line_number=line_number,
        raw_value_json=raw_value_json,
    )
    rank: _SampleRank = (sample_hash, parent_token, line_number, raw_value_json)
    accumulator.sampler.consider(rank, sample)


def _stable_sample_hash(
    *,
    catalog_hash: str,
    seed: str,
    parent_asin: str | None,
    line_number: int,
    key: str,
) -> tuple[str, str]:
    parent_token = parent_asin if parent_asin is not None else f"@line:{line_number}"
    payload = canonical_json_dumps([catalog_hash, seed, parent_token, key]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), parent_token


def _finalize(state: _State, *, file_size: int) -> CatalogProfile:
    top_fields = tuple(
        TopLevelFieldProfile(
            field=field_name,
            present_count=accumulator.present_count,
            missing_count=state.product_row_count - accumulator.present_count,
            null_count=accumulator.null_count,
            empty_count=accumulator.empty_count,
            type_counts=_type_counts(accumulator.type_counts),
        )
        for field_name, accumulator in sorted(state.top_fields.items())
    )
    category_nodes = tuple(
        CategoryNode(
            category_id=category_id_for_path(path),
            path=path,
            parent_id=None if len(path) == 1 else category_id_for_path(path[:-1]),
            direct_support=state.direct_category_support[path],
            subtree_support=subtree_support,
        )
        for path, subtree_support in sorted(state.subtree_category_support.items())
    )
    detail_keys = tuple(
        _finalize_detail_key(raw_key, accumulator, top_value_limit=state.config.top_value_limit)
        for raw_key, accumulator in sorted(state.details.items())
    )
    category_detail_coverage = tuple(
        _finalize_category_coverage(state, path=path, raw_key=raw_key, present_count=count)
        for (path, raw_key), count in sorted(state.category_key_present.items())
    )
    diagnostics = tuple(
        DiagnosticProfile(
            code=code,
            count=accumulator.count,
            samples=accumulator.sampler.values(),
        )
        for code, accumulator in sorted(state.diagnostics.items())
    )
    return CatalogProfile(
        schema_version=_SCHEMA_VERSION,
        catalog_sha256=state.catalog_hash,
        file_size_bytes=file_size,
        physical_line_count=state.physical_line_count,
        product_row_count=state.product_row_count,
        invalid_record_count=state.invalid_record_count,
        category_assignment_count=state.category_assignment_count,
        valid_category_assignment_count=state.valid_category_assignment_count,
        product_row_with_diagnostics_count=state.product_row_with_diagnostics_count,
        unique_parent_asin_count=len(state.seen_parent_asins),
        seed=state.config.seed,
        sample_limit=state.config.sample_limit,
        top_value_limit=state.config.top_value_limit,
        top_level_fields=top_fields,
        category_nodes=category_nodes,
        detail_keys=detail_keys,
        category_detail_coverage=category_detail_coverage,
        diagnostics=diagnostics,
    )


def _finalize_detail_key(
    raw_key: str,
    accumulator: _DetailAccumulator,
    *,
    top_value_limit: int,
) -> DetailKeyProfile:
    ranked_values = sorted(
        accumulator.value_counts.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )[:top_value_limit]
    return DetailKeyProfile(
        raw_key=raw_key,
        support_count=accumulator.support_count,
        nonempty_count=accumulator.nonempty_count,
        null_count=accumulator.null_count,
        empty_count=accumulator.empty_count,
        type_counts=_type_counts(accumulator.type_counts),
        distinct_value_count=len(accumulator.value_counts),
        distinct_nonempty_value_count=len(accumulator.distinct_nonempty_values),
        top_values=tuple(
            ValueMass(
                canonical_value_json=identity[0],
                value_type=identity[1],
                count=count,
            )
            for identity, count in ranked_values
        ),
        samples=accumulator.sampler.values(),
    )


def _finalize_category_coverage(
    state: _State,
    *,
    path: tuple[str, ...],
    raw_key: str,
    present_count: int,
) -> CategoryDetailCoverage:
    product_count = state.subtree_category_support[path]
    nonempty_count = state.category_key_nonempty[(path, raw_key)]
    return CategoryDetailCoverage(
        category_id=category_id_for_path(path),
        raw_key=raw_key,
        product_count=product_count,
        present_count=present_count,
        nonempty_count=nonempty_count,
        presence_coverage=present_count / product_count,
        nonempty_coverage=nonempty_count / product_count,
    )


def _type_counts(counter: Counter[str]) -> tuple[TypeCount, ...]:
    return tuple(
        TypeCount(value_type=value_type, count=count)
        for value_type, count in sorted(counter.items())
    )


def _json_value_type(value: object) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        return "number"
    if type(value) is str:
        return "string"
    if type(value) is list:
        return "array"
    if type(value) is dict:
        return "object"
    raise TypeError(f"unsupported parsed JSON value type: {type(value).__name__}")


def _is_empty(value: object) -> bool:
    if type(value) is str:
        return not value.strip()
    if type(value) in (list, dict):
        return len(cast(Sequence[object], value)) == 0
    return False


def _reject_nonstandard_number(value: str) -> object:
    raise ValueError(f"non-standard JSON number: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number: {value}")
    return parsed


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result
