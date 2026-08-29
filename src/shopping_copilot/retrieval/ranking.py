"""Model-based relevance ranking and vector-only slate selection experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from .dense import DenseIndex
from .errors import RankingBackendUnavailableError
from .vector_diversity import VectorCandidate


class CrossEncoderScorer(Protocol):
    """Small backend-neutral query-document scoring boundary."""

    @property
    def model_id(self) -> str:
        """Return the stable model identifier used by the scorer."""

    def score(
        self,
        query: str,
        documents: Sequence[str],
        *,
        batch_size: int,
    ) -> tuple[float, ...]:
        """Return one finite raw score per document; larger is better."""


class _CrossEncoderModel(Protocol):
    def predict(self, sentences: Sequence[tuple[str, str]], **kwargs: object) -> object:
        """Run backend-specific pair scoring."""


class SentenceTransformerCrossEncoderScorer:
    """Lazy adapter for official Sentence Transformers cross-encoder models."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str | None = None,
        local_files_only: bool = False,
        max_length: int = 512,
        instruction: str | None = None,
        revision: str | None = None,
    ) -> None:
        if type(model_id) is not str or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if type(max_length) is not int or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        if instruction is not None and (type(instruction) is not str or not instruction.strip()):
            raise ValueError("instruction must be a non-empty string or None")
        if revision is not None and (type(revision) is not str or not revision.strip()):
            raise ValueError("revision must be a non-empty string or None")
        resolved_revision = None if revision is None else revision.strip()
        self._model_id = (
            model_id.strip()
            if resolved_revision is None
            else f"{model_id.strip()}@{resolved_revision}"
        )
        try:
            module = import_module("sentence_transformers")
            prompt_arguments: dict[str, object] = {}
            if instruction is not None:
                prompt_arguments = {
                    "prompts": {"shopping": instruction.strip()},
                    "default_prompt_name": "shopping",
                }
            self._model = cast(
                _CrossEncoderModel,
                module.CrossEncoder(
                    model_id.strip(),
                    revision=resolved_revision,
                    device=device,
                    trust_remote_code=False,
                    local_files_only=local_files_only,
                    max_length=max_length,
                    **prompt_arguments,
                ),
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise RankingBackendUnavailableError(
                f"cannot load cross-encoder ranking model {self._model_id!r}"
            ) from error

    @property
    def model_id(self) -> str:
        return self._model_id

    def score(
        self,
        query: str,
        documents: Sequence[str],
        *,
        batch_size: int,
    ) -> tuple[float, ...]:
        if type(query) is not str or not query.strip():
            raise ValueError("query must be a non-empty string")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        prepared: list[str] = []
        for document in documents:
            if type(document) is not str or not document.strip():
                raise ValueError("documents must contain non-empty strings")
            prepared.append(document.strip())
        if not prepared:
            return ()
        try:
            output = self._model.predict(
                [(query.strip(), document) for document in prepared],
                batch_size=batch_size,
                show_progress_bar=False,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise RankingBackendUnavailableError("cross-encoder ranking failed") from error
        scores = np.asarray(output, dtype=np.float64).reshape(-1)
        if scores.shape != (len(prepared),) or not np.isfinite(scores).all():
            raise RankingBackendUnavailableError("cross-encoder returned invalid scores")
        return tuple(float(value) for value in scores)


@dataclass(frozen=True, slots=True, kw_only=True)
class CrossEncoderRankingHit:
    """One auditable relevance score after blending model and route evidence."""

    parent_asin: str
    rank: int
    candidate_rank: int
    raw_model_score: float
    normalized_model_score: float
    prior_relevance: float
    relevance: float


@dataclass(frozen=True, slots=True, kw_only=True)
class CrossEncoderRankingResult:
    """Cross-encoder output retaining the original candidate-rank evidence."""

    model_id: str
    prior_weight: float
    hits: tuple[CrossEncoderRankingHit, ...]

    @property
    def candidates(self) -> tuple[VectorCandidate, ...]:
        return tuple(
            VectorCandidate(
                parent_asin=hit.parent_asin,
                candidate_rank=hit.rank,
                relevance=hit.relevance,
            )
            for hit in self.hits
        )


class CrossEncoderRelevanceReranker:
    """Rerank a bounded candidate pool without overriding hard eligibility."""

    def __init__(self, *, scorer: CrossEncoderScorer) -> None:
        self.scorer = scorer

    def rerank(
        self,
        query: str,
        candidates: tuple[VectorCandidate, ...],
        *,
        documents: Mapping[str, str],
        prior_weight: float = 0.25,
        batch_size: int = 16,
    ) -> CrossEncoderRankingResult:
        _validate_candidates(candidates)
        _require_unit_interval(prior_weight, name="prior_weight")
        texts: list[str] = []
        for candidate in candidates:
            try:
                text = documents[candidate.parent_asin]
            except KeyError as error:
                raise KeyError(f"missing product document: {candidate.parent_asin}") from error
            if type(text) is not str or not text.strip():
                raise ValueError("product documents must contain non-empty strings")
            texts.append(text)
        raw_scores = self.scorer.score(query, texts, batch_size=batch_size)
        if len(raw_scores) != len(candidates):
            raise ValueError("cross-encoder score count differs from candidates")
        normalized = _min_max_scores(raw_scores)
        values = tuple(
            prior_weight * candidate.relevance + (1.0 - prior_weight) * model_score
            for candidate, model_score in zip(candidates, normalized, strict=True)
        )
        order = sorted(
            range(len(candidates)),
            key=lambda index: (-values[index], candidates[index].parent_asin),
        )
        return CrossEncoderRankingResult(
            model_id=self.scorer.model_id,
            prior_weight=prior_weight,
            hits=tuple(
                CrossEncoderRankingHit(
                    parent_asin=candidates[index].parent_asin,
                    rank=rank,
                    candidate_rank=candidates[index].candidate_rank,
                    raw_model_score=raw_scores[index],
                    normalized_model_score=normalized[index],
                    prior_relevance=candidates[index].relevance,
                    relevance=float(values[index]),
                )
                for rank, index in enumerate(order, start=1)
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class VectorSlateHit:
    """One result chosen by a set-aware vector slate objective."""

    parent_asin: str
    rank: int
    candidate_rank: int
    relevance: float
    maximum_similarity_to_selected: float
    selection_score: float
    latent_aspect: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class VectorSlateResult:
    """Auditable output shared by DPP and latent-aspect xQuAD selectors."""

    method: str
    index_id: str
    candidate_count: int
    requested_top_k: int
    relevance_weight: float
    latent_aspect_count: int | None
    hits: tuple[VectorSlateHit, ...]


class GreedyDPPSelector:
    """Select a quality-diversity slate with greedy DPP MAP inference."""

    def __init__(self, *, index: DenseIndex, jitter: float = 1e-6) -> None:
        if type(index) is not DenseIndex:
            raise TypeError("index must be an exact DenseIndex")
        if type(jitter) is not float or not math.isfinite(jitter) or jitter <= 0.0:
            raise ValueError("jitter must be a positive finite float")
        self.index = index
        self.jitter = jitter

    def select(
        self,
        candidates: tuple[VectorCandidate, ...],
        *,
        top_k: int,
        relevance_weight: float,
    ) -> VectorSlateResult:
        _validate_candidates(candidates)
        _validate_selection(top_k, relevance_weight)
        if not candidates:
            return _empty_slate(
                method="greedy_dpp",
                index=self.index,
                top_k=top_k,
                relevance_weight=relevance_weight,
                latent_aspect_count=None,
            )

        vectors = _candidate_vectors(self.index, candidates).astype(np.float64)
        gram = np.clip(vectors @ vectors.T, -1.0, 1.0)
        count = len(candidates)
        diversity_strength = 1.0 - relevance_weight**2
        similarity_kernel = diversity_strength * gram + (1.0 - diversity_strength) * np.eye(
            count, dtype=np.float64
        )
        relevance = np.asarray([item.relevance for item in candidates], dtype=np.float64)
        quality_strength = 0.25 + 1.75 * relevance_weight
        quality = np.exp(quality_strength * relevance)
        kernel = quality[:, None] * similarity_kernel * quality[None, :]
        kernel += self.jitter * np.eye(count, dtype=np.float64)

        selected: list[int] = []
        available = set(range(count))
        current_log_determinant = 0.0
        hits: list[VectorSlateHit] = []
        for rank in range(1, min(top_k, count) + 1):
            scored: list[tuple[float, float, int]] = []
            for index in available:
                subset = (*selected, index)
                sign, log_determinant = np.linalg.slogdet(kernel[np.ix_(subset, subset)])
                if sign <= 0.0 or not math.isfinite(float(log_determinant)):
                    marginal = -math.inf
                else:
                    marginal = float(log_determinant) - current_log_determinant
                scored.append((marginal, candidates[index].relevance, -index))
            marginal, _, negative_index = max(scored)
            chosen = -negative_index
            maximum_similarity = _maximum_similarity(vectors, chosen, selected)
            selected.append(chosen)
            available.remove(chosen)
            current_log_determinant += marginal
            candidate = candidates[chosen]
            hits.append(
                VectorSlateHit(
                    parent_asin=candidate.parent_asin,
                    rank=rank,
                    candidate_rank=candidate.candidate_rank,
                    relevance=candidate.relevance,
                    maximum_similarity_to_selected=maximum_similarity,
                    selection_score=float(marginal),
                    latent_aspect=None,
                )
            )
        return VectorSlateResult(
            method="greedy_dpp",
            index_id=self.index.index_id,
            candidate_count=len(candidates),
            requested_top_k=top_k,
            relevance_weight=relevance_weight,
            latent_aspect_count=None,
            hits=tuple(hits),
        )


class LatentAspectXQuADSelector:
    """Apply xQuAD-style coverage to deterministic vector-derived aspects."""

    def __init__(self, *, index: DenseIndex, maximum_aspects: int = 6) -> None:
        if type(index) is not DenseIndex:
            raise TypeError("index must be an exact DenseIndex")
        if type(maximum_aspects) is not int or maximum_aspects <= 0:
            raise ValueError("maximum_aspects must be a positive integer")
        self.index = index
        self.maximum_aspects = maximum_aspects

    def select(
        self,
        candidates: tuple[VectorCandidate, ...],
        *,
        top_k: int,
        relevance_weight: float,
    ) -> VectorSlateResult:
        _validate_candidates(candidates)
        _validate_selection(top_k, relevance_weight)
        if not candidates:
            return _empty_slate(
                method="latent_xquad",
                index=self.index,
                top_k=top_k,
                relevance_weight=relevance_weight,
                latent_aspect_count=0,
            )
        vectors = _candidate_vectors(self.index, candidates).astype(np.float64)
        relevance = np.asarray([item.relevance for item in candidates], dtype=np.float64)
        aspect_count = min(self.maximum_aspects, len(candidates))
        centers = _farthest_first_centers(vectors, relevance, aspect_count)
        affinities = np.maximum(vectors @ vectors[list(centers)].T, 0.0)
        maxima = np.maximum(np.max(affinities, axis=0), 1e-12)
        affinities /= maxima[None, :]
        aspect_mass = np.sum(affinities * relevance[:, None], axis=0)
        if float(np.sum(aspect_mass)) <= 0.0:
            aspect_weights = np.full(aspect_count, 1.0 / aspect_count, dtype=np.float64)
        else:
            aspect_weights = aspect_mass / np.sum(aspect_mass)

        uncovered = np.ones(aspect_count, dtype=np.float64)
        available = np.ones(len(candidates), dtype=np.bool_)
        selected: list[int] = []
        hits: list[VectorSlateHit] = []
        for rank in range(1, min(top_k, len(candidates)) + 1):
            novelty = affinities @ (aspect_weights * uncovered)
            available_novelty = novelty[available]
            maximum_novelty = float(np.max(available_novelty))
            if maximum_novelty > 0.0:
                novelty /= maximum_novelty
            scores = relevance_weight * relevance + (1.0 - relevance_weight) * novelty
            scores[~available] = -np.inf
            chosen = int(np.argmax(scores))
            maximum_similarity = _maximum_similarity(vectors, chosen, selected)
            selected.append(chosen)
            available[chosen] = False
            uncovered *= 1.0 - affinities[chosen]
            candidate = candidates[chosen]
            hits.append(
                VectorSlateHit(
                    parent_asin=candidate.parent_asin,
                    rank=rank,
                    candidate_rank=candidate.candidate_rank,
                    relevance=candidate.relevance,
                    maximum_similarity_to_selected=maximum_similarity,
                    selection_score=float(scores[chosen]),
                    latent_aspect=int(np.argmax(affinities[chosen])),
                )
            )
        return VectorSlateResult(
            method="latent_xquad",
            index_id=self.index.index_id,
            candidate_count=len(candidates),
            requested_top_k=top_k,
            relevance_weight=relevance_weight,
            latent_aspect_count=aspect_count,
            hits=tuple(hits),
        )


def _validate_candidates(candidates: tuple[VectorCandidate, ...]) -> None:
    if type(candidates) is not tuple:
        raise TypeError("candidates must be a tuple")
    seen: set[str] = set()
    for expected_rank, candidate in enumerate(candidates, start=1):
        if type(candidate) is not VectorCandidate:
            raise TypeError("candidates must contain exact VectorCandidate values")
        if candidate.candidate_rank != expected_rank:
            raise ValueError("candidate ranks must be contiguous")
        if candidate.parent_asin in seen:
            raise ValueError("candidates must contain unique products")
        seen.add(candidate.parent_asin)
        _require_unit_interval(candidate.relevance, name="candidate relevance")


def _validate_selection(top_k: int, relevance_weight: float) -> None:
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    _require_unit_interval(relevance_weight, name="relevance_weight")


def _min_max_scores(scores: tuple[float, ...]) -> tuple[float, ...]:
    if not scores:
        return ()
    if any(type(value) is not float or not math.isfinite(value) for value in scores):
        raise ValueError("model scores must be finite floats")
    minimum = min(scores)
    maximum = max(scores)
    if maximum == minimum:
        return tuple(1.0 for _ in scores)
    return tuple(float((value - minimum) / (maximum - minimum)) for value in scores)


def _candidate_vectors(
    index: DenseIndex,
    candidates: tuple[VectorCandidate, ...],
) -> NDArray[np.float32]:
    rows = np.fromiter(
        (index.row_index(item.parent_asin) for item in candidates),
        dtype=np.int64,
        count=len(candidates),
    )
    return index.vectors[rows]


def _maximum_similarity(
    vectors: NDArray[np.float64],
    chosen: int,
    selected: list[int],
) -> float:
    if not selected:
        return 0.0
    return float(max(0.0, np.max(vectors[selected] @ vectors[chosen])))


def _farthest_first_centers(
    vectors: NDArray[np.float64],
    relevance: NDArray[np.float64],
    count: int,
) -> tuple[int, ...]:
    centers = [int(np.argmax(relevance))]
    maximum_similarity = vectors @ vectors[centers[0]]
    while len(centers) < count:
        distance = 1.0 - maximum_similarity
        distance[centers] = -np.inf
        chosen = int(np.argmax(distance))
        centers.append(chosen)
        maximum_similarity = np.maximum(maximum_similarity, vectors @ vectors[chosen])
    return tuple(centers)


def _empty_slate(
    *,
    method: str,
    index: DenseIndex,
    top_k: int,
    relevance_weight: float,
    latent_aspect_count: int | None,
) -> VectorSlateResult:
    return VectorSlateResult(
        method=method,
        index_id=index.index_id,
        candidate_count=0,
        requested_top_k=top_k,
        relevance_weight=relevance_weight,
        latent_aspect_count=latent_aspect_count,
        hits=(),
    )


def _require_unit_interval(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite float in [0, 1]")
    return value
