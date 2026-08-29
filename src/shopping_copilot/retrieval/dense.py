"""Exact dense scoring and mask-before-Top-K selection for a 50k catalog."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .embedding import FloatMatrix, FloatVector, TextEmbedder, normalize_vector
from .errors import DenseIndexIntegrityError, QueryEmbeddingError
from .models import DenseHit, DenseIndexManifest

BoolVector = NDArray[np.bool_]
ScoreVector = NDArray[np.float32]
_CONTENT_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True, kw_only=True)
class DenseScoreSnapshot:
    """One immutable full-catalog score vector bound to one loaded index."""

    index_id: str
    catalog_semantic_release_id: str
    values: ScoreVector
    _binding: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        observed = np.asarray(self.values)
        if observed.ndim != 1 or observed.dtype != np.float32:
            raise TypeError("DenseScoreSnapshot.values must be a float32 vector")
        if not np.isfinite(observed).all():
            raise ValueError("DenseScoreSnapshot.values must be finite")
        owned = np.array(observed, dtype=np.float32, order="C", copy=True)
        owned.setflags(write=False)
        object.__setattr__(self, "values", owned)


@dataclass(frozen=True, slots=True, kw_only=True)
class DenseEligibilityMask:
    """One immutable eligible set expressed in a specific index's row order."""

    index_id: str
    catalog_semantic_release_id: str
    values: BoolVector
    _binding: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        observed = np.asarray(self.values)
        if observed.ndim != 1 or observed.dtype != np.bool_:
            raise TypeError("DenseEligibilityMask.values must be a boolean vector")
        owned = np.array(observed, dtype=np.bool_, order="C", copy=True)
        owned.setflags(write=False)
        object.__setattr__(self, "values", cast(BoolVector, owned))

    @property
    def eligible_count(self) -> int:
        return int(np.count_nonzero(self.values))


@dataclass(frozen=True, slots=True, kw_only=True)
class DenseSearchResult:
    """A ranking and its reusable scores produced under one bound mask."""

    scores: DenseScoreSnapshot
    hits: tuple[DenseHit, ...]
    eligible_mask: DenseEligibilityMask | None
    requested_top_k: int
    _binding: object = field(repr=False, compare=False)


class DenseIndex:
    """A verified immutable matrix plus its exact row-to-product mapping."""

    def __init__(
        self,
        *,
        index_id: str,
        manifest: DenseIndexManifest,
        parent_asins: tuple[str, ...],
        vectors: FloatMatrix,
    ) -> None:
        if type(index_id) is not str or _CONTENT_ID_PATTERN.fullmatch(index_id) is None:
            raise DenseIndexIntegrityError("index_id is invalid")
        if not isinstance(manifest, DenseIndexManifest):
            raise DenseIndexIntegrityError("manifest is invalid")
        if type(parent_asins) is not tuple:
            raise DenseIndexIntegrityError("parent ASINs must be an immutable tuple")
        if len(parent_asins) != manifest.product_count:
            raise DenseIndexIntegrityError("parent ASIN count differs from manifest")
        if any(
            type(parent_asin) is not str or not parent_asin or parent_asin != parent_asin.strip()
            for parent_asin in parent_asins
        ):
            raise DenseIndexIntegrityError("parent ASINs contain an invalid ID")
        if parent_asins != tuple(sorted(parent_asins)) or len(set(parent_asins)) != len(
            parent_asins
        ):
            raise DenseIndexIntegrityError("parent ASINs must be sorted and unique")
        observed_vectors = np.asarray(vectors)
        if observed_vectors.shape != (
            manifest.product_count,
            manifest.embedding.dimension,
        ):
            raise DenseIndexIntegrityError("vector matrix shape differs from manifest")
        if observed_vectors.dtype != np.float32:
            raise DenseIndexIntegrityError("vector matrix dtype differs from manifest")
        if not np.isfinite(observed_vectors).all():
            raise DenseIndexIntegrityError("vector matrix contains a non-finite value")
        norms = np.linalg.norm(observed_vectors, axis=1)
        if not np.allclose(norms, 1.0, rtol=2e-4, atol=2e-4):
            raise DenseIndexIntegrityError("vector matrix rows are not L2-normalized")
        owned_vectors = np.array(
            observed_vectors,
            dtype=np.float32,
            order="C",
            copy=True,
        )
        owned_vectors.setflags(write=False)
        self.index_id = index_id
        self.manifest = manifest
        self.parent_asins = parent_asins
        self._vectors = owned_vectors
        self._binding = object()
        self._asin_sort_keys = np.asarray(parent_asins, dtype=np.str_)
        self._row_by_asin = {parent_asin: row for row, parent_asin in enumerate(parent_asins)}

    @property
    def vectors(self) -> FloatMatrix:
        """Expose a non-writeable view of the owned vector snapshot."""

        view = self._vectors.view()
        view.setflags(write=False)
        return view

    def row_index(self, parent_asin: str) -> int:
        """Return the immutable matrix row for one catalog product."""

        try:
            return self._row_by_asin[parent_asin]
        except KeyError as error:
            raise KeyError(f"unknown parent_asin: {parent_asin}") from error

    def score_vector(self, query_vector: FloatVector) -> DenseScoreSnapshot:
        """Return exact cosine scores for every product in index row order."""

        try:
            normalized = normalize_vector(
                np.asarray(query_vector, dtype=np.float32),
                expected_dimension=self.manifest.embedding.dimension,
                name="query vector",
            )
        except ValueError as error:
            raise QueryEmbeddingError(str(error)) from error
        scores = self._vectors @ normalized
        score_vector = np.asarray(scores, dtype=np.float32)
        if score_vector.shape != (self.manifest.product_count,):
            raise DenseIndexIntegrityError("dense scorer produced the wrong shape")
        if not np.isfinite(score_vector).all():
            raise DenseIndexIntegrityError("dense scorer produced a non-finite value")
        return DenseScoreSnapshot(
            index_id=self.index_id,
            catalog_semantic_release_id=self.manifest.catalog_semantic_release_id,
            values=score_vector,
            _binding=self._binding,
        )

    def make_eligibility_mask(
        self,
        eligible_parent_asins: Iterable[str],
    ) -> DenseEligibilityMask:
        """Safely map product IDs to this index's private row ordering."""

        if isinstance(eligible_parent_asins, (str, bytes)):
            raise TypeError("eligible_parent_asins must be an iterable of product IDs")
        values = np.zeros(self.manifest.product_count, dtype=np.bool_)
        try:
            iterator = iter(eligible_parent_asins)
        except TypeError as error:
            raise TypeError("eligible_parent_asins must be iterable") from error
        for parent_asin in iterator:
            if (
                type(parent_asin) is not str
                or not parent_asin
                or parent_asin != parent_asin.strip()
            ):
                raise ValueError("eligible_parent_asins contains an invalid ID")
            try:
                values[self._row_by_asin[parent_asin]] = True
            except KeyError as error:
                raise KeyError(f"unknown eligible parent_asin: {parent_asin}") from error
        return DenseEligibilityMask(
            index_id=self.index_id,
            catalog_semantic_release_id=self.manifest.catalog_semantic_release_id,
            values=values,
            _binding=self._binding,
        )

    def select_top_k(
        self,
        scores: DenseScoreSnapshot,
        *,
        top_k: int,
        eligible_mask: DenseEligibilityMask | None = None,
    ) -> tuple[DenseHit, ...]:
        """Apply eligibility before stable score ordering and truncation."""

        if type(top_k) is not int or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        self._require_score_snapshot(scores)
        observed = scores.values
        if top_k == 0:
            return ()

        candidate_indices: NDArray[np.int64]
        if eligible_mask is None:
            candidate_indices = np.arange(self.manifest.product_count, dtype=np.int64)
        else:
            self._require_eligibility_mask(eligible_mask)
            candidate_indices = cast(
                NDArray[np.int64],
                np.asarray(np.flatnonzero(eligible_mask.values), dtype=np.int64),
            )
        if candidate_indices.size == 0:
            return ()

        candidate_scores = observed[candidate_indices]
        candidate_asins = self._asin_sort_keys[candidate_indices]
        order = np.lexsort((candidate_asins, -candidate_scores))
        selected = candidate_indices[order[:top_k]]
        return tuple(
            DenseHit(
                parent_asin=self.parent_asins[int(index)],
                score=float(observed[int(index)]),
                rank=rank,
            )
            for rank, index in enumerate(selected, start=1)
        )

    def rank_scores(
        self,
        scores: DenseScoreSnapshot,
        *,
        top_k: int,
        eligible_mask: DenseEligibilityMask | None = None,
    ) -> DenseSearchResult:
        """Bind a stable ranking to the exact score snapshot and mask used."""

        hits = self.select_top_k(
            scores,
            top_k=top_k,
            eligible_mask=eligible_mask,
        )
        result = DenseSearchResult(
            scores=scores,
            hits=hits,
            eligible_mask=eligible_mask,
            requested_top_k=top_k,
            _binding=self._binding,
        )
        self._require_search_result(result)
        return result

    def _require_score_snapshot(self, scores: DenseScoreSnapshot) -> None:
        if not isinstance(scores, DenseScoreSnapshot):
            raise TypeError("scores must be a DenseScoreSnapshot")
        if (
            scores._binding is not self._binding
            or scores.index_id != self.index_id
            or scores.catalog_semantic_release_id != self.manifest.catalog_semantic_release_id
            or scores.values.shape != (self.manifest.product_count,)
        ):
            raise DenseIndexIntegrityError("score snapshot belongs to a different dense index")

    def _require_eligibility_mask(self, mask: DenseEligibilityMask) -> None:
        if not isinstance(mask, DenseEligibilityMask):
            raise TypeError("eligible_mask must be a DenseEligibilityMask")
        if (
            mask._binding is not self._binding
            or mask.index_id != self.index_id
            or mask.catalog_semantic_release_id != self.manifest.catalog_semantic_release_id
            or mask.values.shape != (self.manifest.product_count,)
        ):
            raise DenseIndexIntegrityError("eligibility mask belongs to a different dense index")

    def _require_search_result(self, result: DenseSearchResult) -> None:
        if not isinstance(result, DenseSearchResult) or result._binding is not self._binding:
            raise DenseIndexIntegrityError("search result belongs to a different dense index")
        self._require_score_snapshot(result.scores)
        if result.eligible_mask is not None:
            self._require_eligibility_mask(result.eligible_mask)
        if type(result.requested_top_k) is not int or result.requested_top_k < 0:
            raise DenseIndexIntegrityError("search result has an invalid requested_top_k")
        eligible_count = (
            self.manifest.product_count
            if result.eligible_mask is None
            else result.eligible_mask.eligible_count
        )
        if len(result.hits) != min(result.requested_top_k, eligible_count):
            raise DenseIndexIntegrityError("search result has an invalid hit count")
        for rank, hit in enumerate(result.hits, start=1):
            if hit.rank != rank:
                raise DenseIndexIntegrityError("search result ranks are not contiguous")
            row = self.row_index(hit.parent_asin)
            if hit.score != float(result.scores.values[row]):
                raise DenseIndexIntegrityError("search result score differs from its snapshot")


class DenseRetriever:
    """Online query encoder sharing one full score vector with all consumers."""

    def __init__(self, *, index: DenseIndex, embedder: TextEmbedder) -> None:
        if embedder.spec != index.manifest.embedding:
            raise DenseIndexIntegrityError("embedder specification differs from index manifest")
        self.index = index
        self.embedder = embedder

    def score(self, q_sem: str) -> DenseScoreSnapshot:
        """Encode one compiled semantic query and score the complete catalog once."""

        vector = self.embedder.encode_query(q_sem)
        return self.index.score_vector(vector)

    def search(
        self,
        q_sem: str,
        *,
        top_k: int,
        eligible_mask: DenseEligibilityMask | None = None,
    ) -> tuple[DenseHit, ...]:
        """Convenience wrapper for callers that do not need Probe score reuse."""

        return self.search_with_scores(
            q_sem,
            top_k=top_k,
            eligible_mask=eligible_mask,
        ).hits

    def search_with_scores(
        self,
        q_sem: str,
        *,
        top_k: int,
        eligible_mask: DenseEligibilityMask | None = None,
    ) -> DenseSearchResult:
        """Return one ranking plus its bound score snapshot for shadow consumers."""

        scores = self.score(q_sem)
        return self.index.rank_scores(
            scores,
            top_k=top_k,
            eligible_mask=eligible_mask,
        )
