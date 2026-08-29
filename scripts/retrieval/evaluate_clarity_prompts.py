"""Evaluate target-free prompt discrimination for the fixed Dense clarity Probe.

This harness intentionally has no dataset argument and never imports the public
simulator.  Every hand-authored prompt is passed unchanged as ``q_sem`` to one
all-eligible Dense Top-80 search.  The fixed Probe then observes Top-20, Top-40,
and Top-80 prefixes of that same bound ranking without another score or sort.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import random
import re
import statistics
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar, cast

PROMPT_SUITE_SCHEMA = "shopping-copilot/clarity-prompt-suite/v0"
REPORT_SCHEMA = "shopping-copilot/clarity-prompt-evaluation/v0"
EXPECTED_LEVELS = ("vague", "focused", "specific")
PROBE_KS = (20, 40, 80)
PRIMARY_K = 40
MAX_PROBE_K = max(PROBE_KS)
BOOTSTRAP_REPLICATES = 5_000
BOOTSTRAP_SEED = 20_260_828

# Pre-registered engineering gate.  These values MUST NOT be changed after
# observing an evaluation run.
GATE_VS_K40_CONCORDANCE = 0.70
GATE_VS_K40_CONCORDANCE_BOOTSTRAP_LOWER = 0.60
GATE_VS_K40_MEDIAN_DELTA_BOOTSTRAP_LOWER = 0.0
GATE_VF_K40_CONCORDANCE = 0.60
GATE_FS_K40_CONCORDANCE = 0.60
GATE_FULL_CHAIN_NO_MATERIAL_REVERSAL = 0.65
GATE_LENGTH_K40_CONCORDANCE = 0.70
GATE_VS_K20_CONCORDANCE = 0.65
GATE_VS_K80_CONCORDANCE = 0.65
GATE_AUDIT_INVARIANCE_MATERIAL_CHANGE_RATE = 0.20
GATE_QUERY_AVAILABILITY = 1.0

_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")

# Support direct execution from a repository checkout as well as ``python -m``.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from shopping_copilot.retrieval.errors import RetrievalError  # noqa: E402


class DenseRetriever(Protocol):
    """Production Dense surface required by this evaluator."""

    def search_with_scores(self, q_sem: str, *, top_k: int) -> object:
        """Return one bound all-eligible ranking and its reusable score snapshot."""


@dataclass(frozen=True, slots=True)
class PromptFamily:
    identifier: str
    domain: str
    vague: str
    focused: str
    specific: str


@dataclass(frozen=True, slots=True)
class LengthControl:
    identifier: str
    domain: str
    lower_label: str
    lower_query: str
    higher_label: str
    higher_query: str


@dataclass(frozen=True, slots=True)
class InvarianceControl:
    identifier: str
    domain: str
    role: str
    left_query: str
    right_query: str
    reason: str


@dataclass(frozen=True, slots=True)
class DiagnosticPrompt:
    identifier: str
    kind: str
    query: str
    interpretation: str


@dataclass(frozen=True, slots=True)
class PromptSuite:
    language: str
    authorship: str
    families: tuple[PromptFamily, ...]
    length_controls: tuple[LengthControl, ...]
    invariance_controls: tuple[InvarianceControl, ...]
    diagnostics: tuple[DiagnosticPrompt, ...]


@dataclass(frozen=True, slots=True)
class QueryRequest:
    key: str
    q_sem: str


@dataclass(frozen=True, slots=True)
class ProbeEvidence:
    n: int
    available: bool
    reason: str | None
    resultant_length: float | None
    raw_g: float | None


@dataclass(frozen=True, slots=True)
class QueryObservation:
    q_sem: str
    evidence: tuple[ProbeEvidence, ...]

    def at(self, probe_k: int) -> ProbeEvidence:
        try:
            return self.evidence[PROBE_KS.index(probe_k)]
        except ValueError as error:
            raise ValueError(f"unsupported Probe K: {probe_k}") from error


@dataclass(frozen=True, slots=True)
class PairDatum:
    identifier: str
    cluster_id: str
    domain: str
    lower_label: str
    higher_label: str
    delta: float | None


@dataclass(frozen=True, slots=True)
class ChainDatum:
    identifier: str
    cluster_id: str
    domain: str
    vague_to_focused: float | None
    focused_to_specific: float | None


@dataclass(frozen=True, slots=True)
class PairedMetrics:
    pair_count: int
    available_count: int
    unavailable_count: int
    wins: int
    ties: int
    losses: int
    concordance: float | None
    median_delta: float | None


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class PairBootstrap:
    concordance: ConfidenceInterval | None
    median_delta: ConfidenceInterval | None
    by_domain: Mapping[str, tuple[ConfidenceInterval, ConfidenceInterval]]


@dataclass(frozen=True, slots=True)
class ChainMetrics:
    chain_count: int
    available_count: int
    unavailable_count: int
    no_material_reversal_count: int
    no_material_reversal_rate: float | None


class Clustered(Protocol):
    @property
    def cluster_id(self) -> str: ...

    @property
    def domain(self) -> str: ...


ClusteredT = TypeVar("ClusteredT", bound=Clustered)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("config/retrieval/clarity-prompts-v0.json"),
        help="strict target-free clarity prompt suite",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/catalog.jsonl"),
        help="frozen catalog JSONL used for Dense release binding",
    )
    parser.add_argument(
        "--dense-factory",
        required=True,
        metavar="MODULE:CALLABLE",
        help=(
            "factory called with catalog_path, index_path, and release_dir; "
            "the result must implement search_with_scores(q_sem, top_k=...)"
        ),
    )
    parser.add_argument(
        "--dense-index",
        type=Path,
        default=None,
        help="prebuilt Dense index path passed unchanged to the factory",
    )
    parser.add_argument(
        "--semantic-release",
        type=Path,
        default=Path("artifacts/catalog-semantic/release-v0"),
        help="active Catalog Semantic release used to bind the Dense index",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write JSON here instead of stdout",
    )
    return parser


def _load_factory(spec: str) -> Callable[..., object]:
    module_name, separator, attribute_name = spec.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("--dense-factory must use MODULE:CALLABLE syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise TypeError(f"dense factory is not callable: {spec}")
    return cast(Callable[..., object], factory)


def _build_dense_retriever(
    factory_spec: str,
    *,
    catalog_path: Path,
    index_path: Path | None,
    release_dir: Path,
) -> DenseRetriever:
    retriever = _load_factory(factory_spec)(
        catalog_path=catalog_path,
        index_path=index_path,
        release_dir=release_dir,
    )
    if not callable(getattr(retriever, "search_with_scores", None)):
        raise TypeError("dense factory result must provide search_with_scores")
    return cast(DenseRetriever, retriever)


def _load_prompt_suite(path: Path) -> PromptSuite:
    try:
        parsed: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"prompt suite is not valid JSON: {error}") from error
    root = _require_object(parsed, path="$")
    _require_exact_keys(
        root,
        {
            "schema",
            "language",
            "authorship",
            "levels",
            "families",
            "length_controls",
            "invariance_controls",
            "diagnostics",
        },
        path="$",
    )
    if _require_text(root["schema"], path="$.schema") != PROMPT_SUITE_SCHEMA:
        raise ValueError(f"$.schema must equal {PROMPT_SUITE_SCHEMA!r}")
    language = _require_text(root["language"], path="$.language")
    if language != "en":
        raise ValueError("$.language must equal 'en' for clarity-prompt-suite/v0")
    authorship = _require_text(root["authorship"], path="$.authorship")
    levels = tuple(
        _require_text(value, path=f"$.levels[{index}]")
        for index, value in enumerate(_require_array(root["levels"], path="$.levels"))
    )
    if levels != EXPECTED_LEVELS:
        raise ValueError(f"$.levels must equal {list(EXPECTED_LEVELS)!r}")

    seen_ids: dict[str, str] = {}
    families = tuple(
        _parse_family(value, index=index, seen_ids=seen_ids)
        for index, value in enumerate(_require_array(root["families"], path="$.families"))
    )
    length_controls = tuple(
        _parse_length_control(value, index=index, seen_ids=seen_ids)
        for index, value in enumerate(
            _require_array(root["length_controls"], path="$.length_controls")
        )
    )
    invariance_controls = tuple(
        _parse_invariance_control(value, index=index, seen_ids=seen_ids)
        for index, value in enumerate(
            _require_array(root["invariance_controls"], path="$.invariance_controls")
        )
    )
    diagnostics = tuple(
        _parse_diagnostic(value, index=index, seen_ids=seen_ids)
        for index, value in enumerate(_require_array(root["diagnostics"], path="$.diagnostics"))
    )
    if not families:
        raise ValueError("$.families must not be empty")
    if not length_controls:
        raise ValueError("$.length_controls must not be empty")
    if not diagnostics:
        raise ValueError("$.diagnostics must not be empty")
    roles = {control.role for control in invariance_controls}
    if roles != {"calibration", "audit"}:
        raise ValueError("$.invariance_controls must contain calibration and audit roles")
    family_domains = {family.domain for family in families}
    for index, length_control in enumerate(length_controls):
        if length_control.domain not in family_domains:
            raise ValueError(f"$.length_controls[{index}].domain is not a family domain")
    for index, invariance_control in enumerate(invariance_controls):
        if invariance_control.domain not in family_domains:
            raise ValueError(f"$.invariance_controls[{index}].domain is not a family domain")
    return PromptSuite(
        language=language,
        authorship=authorship,
        families=families,
        length_controls=length_controls,
        invariance_controls=invariance_controls,
        diagnostics=diagnostics,
    )


def _parse_family(
    value: object,
    *,
    index: int,
    seen_ids: dict[str, str],
) -> PromptFamily:
    path = f"$.families[{index}]"
    item = _require_object(value, path=path)
    _require_exact_keys(item, {"id", "domain", *EXPECTED_LEVELS}, path=path)
    identifier = _registered_identifier(item["id"], path=f"{path}.id", seen_ids=seen_ids)
    domain = _require_identifier(item["domain"], path=f"{path}.domain")
    prompts = tuple(_require_text(item[level], path=f"{path}.{level}") for level in EXPECTED_LEVELS)
    if len({prompt.casefold() for prompt in prompts}) != len(prompts):
        raise ValueError(f"{path} prompts must be distinct")
    return PromptFamily(identifier, domain, prompts[0], prompts[1], prompts[2])


def _parse_length_control(
    value: object,
    *,
    index: int,
    seen_ids: dict[str, str],
) -> LengthControl:
    path = f"$.length_controls[{index}]"
    item = _require_object(value, path=path)
    _require_exact_keys(
        item,
        {"id", "domain", "lower_label", "lower_query", "higher_label", "higher_query"},
        path=path,
    )
    identifier = _registered_identifier(item["id"], path=f"{path}.id", seen_ids=seen_ids)
    domain = _require_identifier(item["domain"], path=f"{path}.domain")
    lower_label = _require_identifier(item["lower_label"], path=f"{path}.lower_label")
    higher_label = _require_identifier(item["higher_label"], path=f"{path}.higher_label")
    lower_query = _require_text(item["lower_query"], path=f"{path}.lower_query")
    higher_query = _require_text(item["higher_query"], path=f"{path}.higher_query")
    if lower_label == higher_label or lower_query.casefold() == higher_query.casefold():
        raise ValueError(f"{path} lower and higher sides must be distinct")
    return LengthControl(
        identifier,
        domain,
        lower_label,
        lower_query,
        higher_label,
        higher_query,
    )


def _parse_invariance_control(
    value: object,
    *,
    index: int,
    seen_ids: dict[str, str],
) -> InvarianceControl:
    path = f"$.invariance_controls[{index}]"
    item = _require_object(value, path=path)
    _require_exact_keys(
        item,
        {"id", "domain", "role", "left_query", "right_query", "reason"},
        path=path,
    )
    identifier = _registered_identifier(item["id"], path=f"{path}.id", seen_ids=seen_ids)
    domain = _require_identifier(item["domain"], path=f"{path}.domain")
    role = _require_text(item["role"], path=f"{path}.role")
    if role not in {"calibration", "audit"}:
        raise ValueError(f"{path}.role must be 'calibration' or 'audit'")
    left_query = _require_text(item["left_query"], path=f"{path}.left_query")
    right_query = _require_text(item["right_query"], path=f"{path}.right_query")
    if left_query.casefold() == right_query.casefold():
        raise ValueError(f"{path} left_query and right_query must be distinct")
    reason = _require_text(item["reason"], path=f"{path}.reason")
    return InvarianceControl(identifier, domain, role, left_query, right_query, reason)


def _parse_diagnostic(
    value: object,
    *,
    index: int,
    seen_ids: dict[str, str],
) -> DiagnosticPrompt:
    path = f"$.diagnostics[{index}]"
    item = _require_object(value, path=path)
    _require_exact_keys(item, {"id", "kind", "query", "interpretation"}, path=path)
    identifier = _registered_identifier(item["id"], path=f"{path}.id", seen_ids=seen_ids)
    kind = _require_identifier(item["kind"], path=f"{path}.kind")
    query = _require_text(item["query"], path=f"{path}.query")
    interpretation = _require_text(item["interpretation"], path=f"{path}.interpretation")
    return DiagnosticPrompt(identifier, kind, query, interpretation)


def _require_object(value: object, *, path: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{path} must be an object")
    return cast(dict[str, object], value)


def _require_array(value: object, *, path: str) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"{path} must be an array")
    return cast(list[object], value)


def _require_exact_keys(item: Mapping[str, object], expected: set[str], *, path: str) -> None:
    observed = set(item)
    if observed == expected:
        return
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    raise ValueError(f"{path} has invalid keys; missing={missing!r}, extra={extra!r}")


def _require_text(value: object, *, path: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{path} must be a non-empty trimmed string")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError(f"{path} contains a lone surrogate")
    return value


def _require_identifier(value: object, *, path: str) -> str:
    observed = _require_text(value, path=path)
    if _IDENTIFIER_PATTERN.fullmatch(observed) is None:
        raise ValueError(f"{path} must be a canonical snake_case identifier")
    return observed


def _registered_identifier(
    value: object,
    *,
    path: str,
    seen_ids: dict[str, str],
) -> str:
    identifier = _require_identifier(value, path=path)
    previous = seen_ids.get(identifier)
    if previous is not None:
        raise ValueError(f"{path} duplicates ID {identifier!r} first seen at {previous}")
    seen_ids[identifier] = path
    return identifier


def _query_requests(suite: PromptSuite) -> tuple[QueryRequest, ...]:
    requests: list[QueryRequest] = []
    for family in suite.families:
        requests.extend(
            QueryRequest(f"family/{family.identifier}/{level}", getattr(family, level))
            for level in EXPECTED_LEVELS
        )
    for length_control in suite.length_controls:
        requests.extend(
            (
                QueryRequest(
                    f"length/{length_control.identifier}/lower", length_control.lower_query
                ),
                QueryRequest(
                    f"length/{length_control.identifier}/higher", length_control.higher_query
                ),
            )
        )
    for invariance_control in suite.invariance_controls:
        requests.extend(
            (
                QueryRequest(
                    f"invariance/{invariance_control.identifier}/left",
                    invariance_control.left_query,
                ),
                QueryRequest(
                    f"invariance/{invariance_control.identifier}/right",
                    invariance_control.right_query,
                ),
            )
        )
    requests.extend(
        QueryRequest(f"diagnostic/{diagnostic.identifier}", diagnostic.query)
        for diagnostic in suite.diagnostics
    )
    keys = [request.key for request in requests]
    if len(keys) != len(set(keys)):
        raise ValueError("prompt suite produced duplicate internal query IDs")
    return tuple(requests)


def _score_queries(
    requests: Sequence[QueryRequest],
    *,
    dense: DenseRetriever,
) -> tuple[dict[str, QueryObservation], int]:
    from shopping_copilot.retrieval import DenseIndex, DenseSearchResult, FixedDenseProbe

    dense_index = getattr(dense, "index", None)
    if not isinstance(dense_index, DenseIndex):
        raise TypeError("dense factory result must expose its verified DenseIndex")
    probe = FixedDenseProbe(dense_index)
    by_text: dict[str, QueryObservation] = {}
    by_key: dict[str, QueryObservation] = {}
    dense_search_calls = 0
    for request in requests:
        observation = by_text.get(request.q_sem)
        if observation is None:
            # The raw prompt is the q_sem.  Exactly one Top-80 ranking is built;
            # all Probe sizes below observe prefixes of this same result.
            result = dense.search_with_scores(request.q_sem, top_k=MAX_PROBE_K)
            dense_search_calls += 1
            if not isinstance(result, DenseSearchResult):
                raise TypeError("search_with_scores must return DenseSearchResult")
            if result.requested_top_k != MAX_PROBE_K:
                raise ValueError("Dense result does not preserve requested Top-80 depth")
            if result.eligible_mask is not None:
                raise ValueError("clarity prompt evaluation requires an all-eligible Dense result")
            evidence: list[ProbeEvidence] = []
            for probe_k in PROBE_KS:
                observed = probe.observe(result, probe_k=probe_k).coherence
                evidence.append(
                    ProbeEvidence(
                        n=observed.n,
                        available=observed.available,
                        reason=observed.reason,
                        resultant_length=observed.resultant_length,
                        raw_g=observed.debiased_pairwise_cosine,
                    )
                )
            observation = QueryObservation(request.q_sem, tuple(evidence))
            by_text[request.q_sem] = observation
        by_key[request.key] = observation
    return by_key, dense_search_calls


def _delta(lower: QueryObservation, higher: QueryObservation, probe_k: int) -> float | None:
    lower_g = lower.at(probe_k).raw_g
    higher_g = higher.at(probe_k).raw_g
    if lower_g is None or higher_g is None:
        return None
    return higher_g - lower_g


def _family_pairs(
    suite: PromptSuite,
    observations: Mapping[str, QueryObservation],
) -> dict[str, dict[int, list[PairDatum]]]:
    transitions = {
        "vague_to_focused": ("vague", "focused"),
        "focused_to_specific": ("focused", "specific"),
        "vague_to_specific": ("vague", "specific"),
    }
    output: dict[str, dict[int, list[PairDatum]]] = {
        transition: {probe_k: [] for probe_k in PROBE_KS} for transition in transitions
    }
    for family in suite.families:
        for transition, (lower_level, higher_level) in transitions.items():
            lower = observations[f"family/{family.identifier}/{lower_level}"]
            higher = observations[f"family/{family.identifier}/{higher_level}"]
            for probe_k in PROBE_KS:
                output[transition][probe_k].append(
                    PairDatum(
                        identifier=family.identifier,
                        cluster_id=family.identifier,
                        domain=family.domain,
                        lower_label=lower_level,
                        higher_label=higher_level,
                        delta=_delta(lower, higher, probe_k),
                    )
                )
    return output


def _length_pairs(
    suite: PromptSuite,
    observations: Mapping[str, QueryObservation],
) -> dict[int, list[PairDatum]]:
    output: dict[int, list[PairDatum]] = {probe_k: [] for probe_k in PROBE_KS}
    for control in suite.length_controls:
        lower = observations[f"length/{control.identifier}/lower"]
        higher = observations[f"length/{control.identifier}/higher"]
        for probe_k in PROBE_KS:
            output[probe_k].append(
                PairDatum(
                    identifier=control.identifier,
                    cluster_id=control.identifier,
                    domain=control.domain,
                    lower_label=control.lower_label,
                    higher_label=control.higher_label,
                    delta=_delta(lower, higher, probe_k),
                )
            )
    return output


def _calibration_epsilon(
    suite: PromptSuite,
    observations: Mapping[str, QueryObservation],
) -> tuple[float | None, list[float]]:
    absolute_deltas: list[float] = []
    calibration_count = 0
    for control in suite.invariance_controls:
        if control.role != "calibration":
            continue
        calibration_count += 1
        left = observations[f"invariance/{control.identifier}/left"]
        right = observations[f"invariance/{control.identifier}/right"]
        delta = _delta(left, right, PRIMARY_K)
        if delta is not None:
            absolute_deltas.append(abs(delta))
    if len(absolute_deltas) != calibration_count:
        return None, absolute_deltas
    return _quantile(sorted(absolute_deltas), 0.95), absolute_deltas


def _paired_metrics(records: Sequence[PairDatum], epsilon: float | None) -> PairedMetrics:
    available = [record.delta for record in records if record.delta is not None]
    if epsilon is None or not available:
        return PairedMetrics(
            pair_count=len(records),
            available_count=len(available),
            unavailable_count=len(records) - len(available),
            wins=0,
            ties=0,
            losses=0,
            concordance=None,
            median_delta=None if not available else statistics.median(available),
        )
    wins = sum(delta > epsilon for delta in available)
    losses = sum(delta < -epsilon for delta in available)
    ties = len(available) - wins - losses
    return PairedMetrics(
        pair_count=len(records),
        available_count=len(available),
        unavailable_count=len(records) - len(available),
        wins=wins,
        ties=ties,
        losses=losses,
        concordance=(wins + 0.5 * ties) / len(available),
        median_delta=statistics.median(available),
    )


def _bootstrap_pairs(
    records: Sequence[PairDatum],
    *,
    epsilon: float | None,
    label: str,
) -> PairBootstrap:
    available = [record for record in records if record.delta is not None]
    if epsilon is None or not available:
        return PairBootstrap(None, None, {})
    rng = _bootstrap_rng(label)
    concordances: list[float] = []
    medians: list[float] = []
    domains = sorted({record.domain for record in available})
    domain_concordances: dict[str, list[float]] = {domain: [] for domain in domains}
    domain_medians: dict[str, list[float]] = {domain: [] for domain in domains}
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_by_domain = _stratified_cluster_resample(available, rng)
        sampled = [record for domain in domains for record in sampled_by_domain[domain]]
        aggregate = _paired_metrics(sampled, epsilon)
        if aggregate.concordance is None or aggregate.median_delta is None:
            raise AssertionError("available bootstrap sample unexpectedly lacks paired metrics")
        concordances.append(aggregate.concordance)
        medians.append(aggregate.median_delta)
        for domain in domains:
            domain_metrics = _paired_metrics(sampled_by_domain[domain], epsilon)
            if domain_metrics.concordance is None or domain_metrics.median_delta is None:
                raise AssertionError("domain bootstrap sample unexpectedly lacks paired metrics")
            domain_concordances[domain].append(domain_metrics.concordance)
            domain_medians[domain].append(domain_metrics.median_delta)
    return PairBootstrap(
        concordance=_confidence_interval(concordances),
        median_delta=_confidence_interval(medians),
        by_domain={
            domain: (
                _confidence_interval(domain_concordances[domain]),
                _confidence_interval(domain_medians[domain]),
            )
            for domain in domains
        },
    )


def _stratified_cluster_resample(
    records: Sequence[ClusteredT],
    rng: random.Random,
) -> dict[str, list[ClusteredT]]:
    strata: dict[str, dict[str, list[ClusteredT]]] = defaultdict(lambda: defaultdict(list))
    cluster_domains: dict[str, str] = {}
    for record in records:
        previous_domain = cluster_domains.setdefault(record.cluster_id, record.domain)
        if previous_domain != record.domain:
            raise ValueError(f"cluster {record.cluster_id!r} spans multiple domains")
        strata[record.domain][record.cluster_id].append(record)
    output: dict[str, list[ClusteredT]] = {}
    for domain in sorted(strata):
        clusters = strata[domain]
        cluster_ids = sorted(clusters)
        sampled: list[ClusteredT] = []
        for _ in cluster_ids:
            sampled.extend(clusters[rng.choice(cluster_ids)])
        output[domain] = sampled
    return output


def _family_chain_records(
    family_pairs: Mapping[str, Mapping[int, Sequence[PairDatum]]],
) -> list[ChainDatum]:
    vague_focused = {
        record.identifier: record for record in family_pairs["vague_to_focused"][PRIMARY_K]
    }
    focused_specific = {
        record.identifier: record for record in family_pairs["focused_to_specific"][PRIMARY_K]
    }
    return [
        ChainDatum(
            identifier=identifier,
            cluster_id=record.cluster_id,
            domain=record.domain,
            vague_to_focused=record.delta,
            focused_to_specific=focused_specific[identifier].delta,
        )
        for identifier, record in vague_focused.items()
    ]


def _chain_metrics(records: Sequence[ChainDatum], epsilon: float | None) -> ChainMetrics:
    available = [
        record
        for record in records
        if record.vague_to_focused is not None and record.focused_to_specific is not None
    ]
    if epsilon is None or not available:
        return ChainMetrics(len(records), len(available), len(records) - len(available), 0, None)
    no_reversal = sum(
        cast(float, record.vague_to_focused) >= -epsilon
        and cast(float, record.focused_to_specific) >= -epsilon
        for record in available
    )
    return ChainMetrics(
        chain_count=len(records),
        available_count=len(available),
        unavailable_count=len(records) - len(available),
        no_material_reversal_count=no_reversal,
        no_material_reversal_rate=no_reversal / len(available),
    )


def _bootstrap_chain_rate(
    records: Sequence[ChainDatum],
    *,
    epsilon: float | None,
    label: str,
) -> ConfidenceInterval | None:
    available = [
        record
        for record in records
        if record.vague_to_focused is not None and record.focused_to_specific is not None
    ]
    if epsilon is None or not available:
        return None
    rng = _bootstrap_rng(label)
    rates: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        by_domain = _stratified_cluster_resample(available, rng)
        sample = [record for domain in sorted(by_domain) for record in by_domain[domain]]
        metrics = _chain_metrics(sample, epsilon)
        if metrics.no_material_reversal_rate is None:
            raise AssertionError("available bootstrap sample unexpectedly lacks chain rate")
        rates.append(metrics.no_material_reversal_rate)
    return _confidence_interval(rates)


def _audit_invariance_summary(
    suite: PromptSuite,
    observations: Mapping[str, QueryObservation],
    *,
    epsilon: float | None,
) -> tuple[dict[str, object], float | None]:
    controls: list[dict[str, object]] = []
    available_abs_deltas: list[float] = []
    audit_count = 0
    for control in suite.invariance_controls:
        if control.role != "audit":
            continue
        audit_count += 1
        left = observations[f"invariance/{control.identifier}/left"]
        right = observations[f"invariance/{control.identifier}/right"]
        delta = _delta(left, right, PRIMARY_K)
        absolute_delta = None if delta is None else abs(delta)
        if absolute_delta is not None:
            available_abs_deltas.append(absolute_delta)
        controls.append(
            {
                "id": control.identifier,
                "domain": control.domain,
                "reason": control.reason,
                "left": _observation_payload(left),
                "right": _observation_payload(right),
                "delta_at_40": _rounded(delta),
                "abs_delta_at_40": _rounded(absolute_delta),
                "material_change": (
                    None if epsilon is None or absolute_delta is None else absolute_delta > epsilon
                ),
            }
        )
    material_changes = (
        None
        if epsilon is None or len(available_abs_deltas) != audit_count
        else sum(delta > epsilon for delta in available_abs_deltas)
    )
    material_change_rate = (
        None if material_changes is None or audit_count == 0 else material_changes / audit_count
    )
    return (
        {
            "control_count": audit_count,
            "available_count": len(available_abs_deltas),
            "unavailable_count": audit_count - len(available_abs_deltas),
            "material_change_count": material_changes,
            "material_change_rate": _rounded(material_change_rate),
            "controls": controls,
        },
        material_change_rate,
    )


def _pair_report(
    records: Sequence[PairDatum],
    *,
    epsilon: float | None,
    label: str,
) -> tuple[dict[str, object], PairedMetrics, PairBootstrap]:
    point = _paired_metrics(records, epsilon)
    bootstrap = _bootstrap_pairs(records, epsilon=epsilon, label=label)
    by_domain: dict[str, object] = {}
    for domain in sorted({record.domain for record in records}):
        domain_records = [record for record in records if record.domain == domain]
        by_domain[domain] = _paired_metrics_payload(_paired_metrics(domain_records, epsilon))
    return (
        {
            "overall": _paired_metrics_payload(point),
            "by_domain": by_domain,
            "bootstrap_95_ci": _pair_bootstrap_payload(bootstrap),
        },
        point,
        bootstrap,
    )


def _paired_metrics_payload(metrics: PairedMetrics) -> dict[str, object]:
    return {
        "pair_count": metrics.pair_count,
        "available_count": metrics.available_count,
        "unavailable_count": metrics.unavailable_count,
        "wins": metrics.wins,
        "ties": metrics.ties,
        "losses": metrics.losses,
        "concordance": _rounded(metrics.concordance),
        "median_delta": _rounded(metrics.median_delta),
    }


def _pair_bootstrap_payload(bootstrap: PairBootstrap) -> dict[str, object]:
    return {
        "method": "domain-stratified cluster percentile bootstrap",
        "replicates": BOOTSTRAP_REPLICATES,
        "base_seed": BOOTSTRAP_SEED,
        "overall": {
            "concordance": _confidence_interval_payload(bootstrap.concordance),
            "median_delta": _confidence_interval_payload(bootstrap.median_delta),
        },
        "by_domain": {
            domain: {
                "concordance": _confidence_interval_payload(intervals[0]),
                "median_delta": _confidence_interval_payload(intervals[1]),
            }
            for domain, intervals in sorted(bootstrap.by_domain.items())
        },
    }


def _confidence_interval(values: Sequence[float]) -> ConfidenceInterval:
    ordered = sorted(values)
    return ConfidenceInterval(_quantile(ordered, 0.025), _quantile(ordered, 0.975))


def _confidence_interval_payload(interval: ConfidenceInterval | None) -> object:
    if interval is None:
        return None
    return {"lower": _rounded(interval.lower), "upper": _rounded(interval.upper)}


def _quantile(ordered: Sequence[float], probability: float) -> float:
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty sequence")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_rng(label: str) -> random.Random:
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}\0{label}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _observation_payload(observation: QueryObservation) -> dict[str, object]:
    return {
        "q_sem": observation.q_sem,
        "raw_g": {str(probe_k): _rounded(observation.at(probe_k).raw_g) for probe_k in PROBE_KS},
        "evidence": {
            str(probe_k): {
                "n": observation.at(probe_k).n,
                "available": observation.at(probe_k).available,
                "reason": observation.at(probe_k).reason,
                "resultant_length": _rounded(observation.at(probe_k).resultant_length),
            }
            for probe_k in PROBE_KS
        },
    }


def _family_values_payload(
    suite: PromptSuite,
    observations: Mapping[str, QueryObservation],
    *,
    epsilon: float | None,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for family in suite.families:
        values = {
            level: observations[f"family/{family.identifier}/{level}"] for level in EXPECTED_LEVELS
        }
        deltas = {
            "vague_to_focused": {
                str(probe_k): _rounded(_delta(values["vague"], values["focused"], probe_k))
                for probe_k in PROBE_KS
            },
            "focused_to_specific": {
                str(probe_k): _rounded(_delta(values["focused"], values["specific"], probe_k))
                for probe_k in PROBE_KS
            },
            "vague_to_specific": {
                str(probe_k): _rounded(_delta(values["vague"], values["specific"], probe_k))
                for probe_k in PROBE_KS
            },
        }
        vf_delta = _delta(values["vague"], values["focused"], PRIMARY_K)
        fs_delta = _delta(values["focused"], values["specific"], PRIMARY_K)
        output.append(
            {
                "id": family.identifier,
                "domain": family.domain,
                "levels": {level: _observation_payload(values[level]) for level in EXPECTED_LEVELS},
                "deltas": deltas,
                "k40_no_material_reversal": (
                    None
                    if epsilon is None or vf_delta is None or fs_delta is None
                    else vf_delta >= -epsilon and fs_delta >= -epsilon
                ),
            }
        )
    return output


def _length_values_payload(
    suite: PromptSuite,
    observations: Mapping[str, QueryObservation],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for control in suite.length_controls:
        lower = observations[f"length/{control.identifier}/lower"]
        higher = observations[f"length/{control.identifier}/higher"]
        output.append(
            {
                "id": control.identifier,
                "domain": control.domain,
                "lower_label": control.lower_label,
                "higher_label": control.higher_label,
                "lower": _observation_payload(lower),
                "higher": _observation_payload(higher),
                "delta": {
                    str(probe_k): _rounded(_delta(lower, higher, probe_k)) for probe_k in PROBE_KS
                },
            }
        )
    return output


def _calibration_payload(
    suite: PromptSuite,
    observations: Mapping[str, QueryObservation],
    *,
    epsilon: float | None,
    absolute_deltas: Sequence[float],
) -> dict[str, object]:
    controls: list[dict[str, object]] = []
    calibration_count = 0
    for control in suite.invariance_controls:
        if control.role != "calibration":
            continue
        calibration_count += 1
        left = observations[f"invariance/{control.identifier}/left"]
        right = observations[f"invariance/{control.identifier}/right"]
        delta = _delta(left, right, PRIMARY_K)
        controls.append(
            {
                "id": control.identifier,
                "domain": control.domain,
                "reason": control.reason,
                "left": _observation_payload(left),
                "right": _observation_payload(right),
                "delta_at_40": _rounded(delta),
                "abs_delta_at_40": _rounded(None if delta is None else abs(delta)),
            }
        )
    return {
        "definition": "P95 absolute K40 delta from role=calibration invariance controls",
        "quantile_method": "linear interpolation at (n - 1) * p",
        "control_count": calibration_count,
        "available_count": len(absolute_deltas),
        "unavailable_count": calibration_count - len(absolute_deltas),
        "abs_delta_at_40": [_rounded(value) for value in absolute_deltas],
        "epsilon": _rounded(epsilon),
        "controls": controls,
    }


def _diagnostics_payload(
    suite: PromptSuite,
    observations: Mapping[str, QueryObservation],
) -> list[dict[str, object]]:
    return [
        {
            "id": diagnostic.identifier,
            "kind": diagnostic.kind,
            "interpretation": diagnostic.interpretation,
            **_observation_payload(observations[f"diagnostic/{diagnostic.identifier}"]),
        }
        for diagnostic in suite.diagnostics
    ]


def _availability_payload(
    requests: Sequence[QueryRequest],
    observations: Mapping[str, QueryObservation],
    *,
    dense_search_calls: int,
) -> tuple[dict[str, object], float]:
    fully_available = sum(
        all(observations[request.key].at(probe_k).available for probe_k in PROBE_KS)
        for request in requests
    )
    available_observations = sum(
        observations[request.key].at(probe_k).available
        for request in requests
        for probe_k in PROBE_KS
    )
    query_count = len(requests)
    query_rate = 0.0 if query_count == 0 else fully_available / query_count
    return (
        {
            "logical_query_count": query_count,
            "unique_q_sem_count": len({request.q_sem for request in requests}),
            "dense_search_calls": dense_search_calls,
            "fully_available_query_count": fully_available,
            "fully_available_query_rate": _rounded(query_rate),
            "probe_observation_count": query_count * len(PROBE_KS),
            "available_probe_observation_count": available_observations,
            "all_queries_available": fully_available == query_count,
        },
        query_rate,
    )


def _gate_check(
    identifier: str,
    *,
    value: float | None,
    operator: str,
    threshold: float,
) -> dict[str, object]:
    if operator == ">=":
        passed = value is not None and value >= threshold
    elif operator == ">":
        passed = value is not None and value > threshold
    elif operator == "<=":
        passed = value is not None and value <= threshold
    else:
        raise AssertionError(f"unknown gate operator: {operator}")
    return {
        "id": identifier,
        "value": _rounded(value),
        "operator": operator,
        "threshold": threshold,
        "pass": passed,
    }


def _gate_payload(
    *,
    family_points: Mapping[str, Mapping[int, PairedMetrics]],
    family_bootstraps: Mapping[str, Mapping[int, PairBootstrap]],
    chain_metrics: ChainMetrics,
    length_points: Mapping[int, PairedMetrics],
    audit_material_change_rate: float | None,
    availability_rate: float,
) -> dict[str, object]:
    vague_specific_40 = family_points["vague_to_specific"][40]
    vague_specific_40_bootstrap = family_bootstraps["vague_to_specific"][40]
    checks = [
        _gate_check(
            "vague_to_specific_k40_concordance",
            value=vague_specific_40.concordance,
            operator=">=",
            threshold=GATE_VS_K40_CONCORDANCE,
        ),
        _gate_check(
            "vague_to_specific_k40_concordance_bootstrap_lower",
            value=(
                None
                if vague_specific_40_bootstrap.concordance is None
                else vague_specific_40_bootstrap.concordance.lower
            ),
            operator=">=",
            threshold=GATE_VS_K40_CONCORDANCE_BOOTSTRAP_LOWER,
        ),
        _gate_check(
            "vague_to_specific_k40_median_delta_bootstrap_lower",
            value=(
                None
                if vague_specific_40_bootstrap.median_delta is None
                else vague_specific_40_bootstrap.median_delta.lower
            ),
            operator=">",
            threshold=GATE_VS_K40_MEDIAN_DELTA_BOOTSTRAP_LOWER,
        ),
        _gate_check(
            "vague_to_focused_k40_concordance",
            value=family_points["vague_to_focused"][40].concordance,
            operator=">=",
            threshold=GATE_VF_K40_CONCORDANCE,
        ),
        _gate_check(
            "focused_to_specific_k40_concordance",
            value=family_points["focused_to_specific"][40].concordance,
            operator=">=",
            threshold=GATE_FS_K40_CONCORDANCE,
        ),
        _gate_check(
            "full_chain_k40_no_material_reversal_rate",
            value=chain_metrics.no_material_reversal_rate,
            operator=">=",
            threshold=GATE_FULL_CHAIN_NO_MATERIAL_REVERSAL,
        ),
        _gate_check(
            "length_control_k40_concordance",
            value=length_points[40].concordance,
            operator=">=",
            threshold=GATE_LENGTH_K40_CONCORDANCE,
        ),
        _gate_check(
            "vague_to_specific_k20_concordance",
            value=family_points["vague_to_specific"][20].concordance,
            operator=">=",
            threshold=GATE_VS_K20_CONCORDANCE,
        ),
        _gate_check(
            "vague_to_specific_k80_concordance",
            value=family_points["vague_to_specific"][80].concordance,
            operator=">=",
            threshold=GATE_VS_K80_CONCORDANCE,
        ),
        _gate_check(
            "audit_invariance_material_change_rate",
            value=audit_material_change_rate,
            operator="<=",
            threshold=GATE_AUDIT_INVARIANCE_MATERIAL_CHANGE_RATE,
        ),
        _gate_check(
            "all_query_availability",
            value=availability_rate,
            operator=">=",
            threshold=GATE_QUERY_AVAILABILITY,
        ),
    ]
    return {
        "status": "pass" if all(cast(bool, check["pass"]) for check in checks) else "fail",
        "rule": "all pre-registered checks must pass",
        "checks": checks,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 8)


def evaluate_clarity_prompts(
    *,
    prompts_path: Path,
    catalog_path: Path,
    dense_factory: str,
    dense_index: Path | None,
    semantic_release: Path,
) -> dict[str, object]:
    """Run the frozen target-free prompt discrimination evaluation."""

    suite = _load_prompt_suite(prompts_path)
    requests = _query_requests(suite)
    dense = _build_dense_retriever(
        dense_factory,
        catalog_path=catalog_path,
        index_path=dense_index,
        release_dir=semantic_release,
    )
    observations, dense_search_calls = _score_queries(requests, dense=dense)
    epsilon, calibration_abs_deltas = _calibration_epsilon(suite, observations)

    family_pairs = _family_pairs(suite, observations)
    family_reports: dict[str, dict[str, object]] = {}
    family_points: dict[str, dict[int, PairedMetrics]] = {}
    family_bootstraps: dict[str, dict[int, PairBootstrap]] = {}
    for transition in ("vague_to_focused", "focused_to_specific", "vague_to_specific"):
        family_reports[transition] = {}
        family_points[transition] = {}
        family_bootstraps[transition] = {}
        for probe_k in PROBE_KS:
            pair_payload, point, bootstrap = _pair_report(
                family_pairs[transition][probe_k],
                epsilon=epsilon,
                label=f"family/{transition}/k{probe_k}",
            )
            family_reports[transition][str(probe_k)] = pair_payload
            family_points[transition][probe_k] = point
            family_bootstraps[transition][probe_k] = bootstrap

    length_pairs = _length_pairs(suite, observations)
    length_reports: dict[str, object] = {}
    length_points: dict[int, PairedMetrics] = {}
    for probe_k in PROBE_KS:
        pair_payload, point, _ = _pair_report(
            length_pairs[probe_k],
            epsilon=epsilon,
            label=f"length/k{probe_k}",
        )
        length_reports[str(probe_k)] = pair_payload
        length_points[probe_k] = point

    chain_records = _family_chain_records(family_pairs)
    chain_metrics = _chain_metrics(chain_records, epsilon)
    chain_bootstrap = _bootstrap_chain_rate(
        chain_records,
        epsilon=epsilon,
        label="family/full_chain/k40",
    )
    audit_payload, audit_material_change_rate = _audit_invariance_summary(
        suite,
        observations,
        epsilon=epsilon,
    )
    availability_payload, availability_rate = _availability_payload(
        requests,
        observations,
        dense_search_calls=dense_search_calls,
    )

    dense_index_object = getattr(dense, "index", None)
    manifest = getattr(dense_index_object, "manifest", None)
    embedding = getattr(manifest, "embedding", None)
    chain_report_payload = _chain_metrics_payload(chain_metrics)
    chain_report_payload["bootstrap_95_ci"] = _confidence_interval_payload(chain_bootstrap)
    chain_report_payload["definition"] = "both adjacent deltas are >= -epsilon"
    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "prompt_suite": {
            "path": str(prompts_path),
            "sha256": _sha256(prompts_path),
            "schema": PROMPT_SUITE_SCHEMA,
            "language": suite.language,
            "authorship": suite.authorship,
            "family_count": len(suite.families),
            "length_control_count": len(suite.length_controls),
            "invariance_control_count": len(suite.invariance_controls),
            "diagnostic_count": len(suite.diagnostics),
        },
        "catalog": {"path": str(catalog_path), "sha256": _sha256(catalog_path)},
        "dense": {
            "factory": dense_factory,
            "index_path": None if dense_index is None else str(dense_index),
            "semantic_release": str(semantic_release),
            "index_id": getattr(dense_index_object, "index_id", None),
            "catalog_semantic_release_id": getattr(manifest, "catalog_semantic_release_id", None),
            "product_count": getattr(manifest, "product_count", None),
            "embedding_model_id": getattr(embedding, "model_id", None),
            "embedding_model_revision": getattr(embedding, "model_revision", None),
        },
        "evaluation_protocol": {
            "q_sem": "raw prompt unchanged",
            "eligibility": "all catalog products; eligible_mask=None",
            "probe_k": list(PROBE_KS),
            "ranking": "one Dense Top-80 sort per unique q_sem; K20/K40 are prefixes",
            "labels": "ordered prompt relations only; no target, recall, or public dataset",
            "tie_epsilon_source": "calibration invariance controls at K40",
            "bootstrap": {
                "method": "domain-stratified cluster percentile bootstrap",
                "replicates": BOOTSTRAP_REPLICATES,
                "base_seed": BOOTSTRAP_SEED,
            },
        },
        "availability": availability_payload,
        "calibration": _calibration_payload(
            suite,
            observations,
            epsilon=epsilon,
            absolute_deltas=calibration_abs_deltas,
        ),
        "family_discrimination": {
            "definition": "higher clarity is expected from vague to focused to specific",
            "paired_metrics": family_reports,
            "full_chain_k40": chain_report_payload,
            "families": _family_values_payload(suite, observations, epsilon=epsilon),
        },
        "length_control": {
            "definition": "higher side is shorter but intentionally more specific",
            "paired_metrics": length_reports,
            "controls": _length_values_payload(suite, observations),
        },
        "audit_invariance": audit_payload,
        "diagnostics": _diagnostics_payload(suite, observations),
    }
    report["pre_registered_gate"] = _gate_payload(
        family_points=family_points,
        family_bootstraps=family_bootstraps,
        chain_metrics=chain_metrics,
        length_points=length_points,
        audit_material_change_rate=audit_material_change_rate,
        availability_rate=availability_rate,
    )
    return report


def _chain_metrics_payload(metrics: ChainMetrics) -> dict[str, object]:
    return {
        "chain_count": metrics.chain_count,
        "available_count": metrics.available_count,
        "unavailable_count": metrics.unavailable_count,
        "no_material_reversal_count": metrics.no_material_reversal_count,
        "no_material_reversal_rate": _rounded(metrics.no_material_reversal_rate),
    }


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        report = evaluate_clarity_prompts(
            prompts_path=args.prompts,
            catalog_path=args.catalog,
            dense_factory=args.dense_factory,
            dense_index=args.dense_index,
            semantic_release=args.semantic_release,
        )
    except (ImportError, OSError, RetrievalError, TypeError, ValueError) as error:
        parser.error(str(error))

    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
