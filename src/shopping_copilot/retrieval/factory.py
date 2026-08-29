"""Thin composition helpers for scripts and the future official API adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path

from shopping_copilot.catalog.semantic import CatalogSemanticError
from shopping_copilot.catalog.semantic.release import load_catalog_semantic_release

from .bundle import load_dense_index
from .calibration import BoundTransparencyCalibration
from .controller import FormalRetrievalPolicy, RetrievalController
from .dense import DenseRetriever
from .documents import load_product_documents
from .embedding import SentenceTransformerTextEmbedder
from .errors import DenseIndexIntegrityError
from .evidence import build_retrieval_evidence_index
from .hard_mask import HardMaskResolver
from .lexical import LEXICAL_PROBE_K, LexicalProbe
from .modes import DEFAULT_MODE_SIMILARITY_THRESHOLD
from .multi_probe import CompiledProbeRunner
from .resolved_probe import ResolvedCompiledProbeRunner
from .routing import FacetRoute
from .vector_diversity import VectorDiversityPolicy


def create_dense_retriever(
    *,
    index_path: str | Path | None,
    release_dir: str | Path | None,
    catalog_path: str | Path | None = None,
    device: str | None = None,
    local_files_only: bool = True,
) -> DenseRetriever:
    """Load the pinned local model and a verified, release-bound dense index."""

    if index_path is None:
        raise ValueError("index_path is required")
    if release_dir is None:
        raise ValueError("release_dir is required for semantic-release binding")
    release_path = Path(release_dir)
    try:
        release = load_catalog_semantic_release(release_path)
    except (CatalogSemanticError, OSError, RecursionError, TypeError, ValueError) as error:
        raise DenseIndexIntegrityError("active semantic release is invalid") from error
    observed_catalog_id = _catalog_content_id(
        release_path / "catalog.jsonl" if catalog_path is None else Path(catalog_path)
    )
    if observed_catalog_id != release.manifest.catalog_id:
        raise DenseIndexIntegrityError("catalog differs from the active semantic release")
    index = load_dense_index(
        index_path,
        expected_catalog_id=release.manifest.catalog_id,
        expected_release_id=release.release_id,
    )
    embedder = SentenceTransformerTextEmbedder(
        index.manifest.embedding,
        device=device,
        local_files_only=local_files_only,
    )
    return DenseRetriever(index=index, embedder=embedder)


def create_compiled_probe_runner(
    *,
    index_path: str | Path | None,
    release_dir: str | Path | None,
    calibration: BoundTransparencyCalibration,
    catalog_path: str | Path | None = None,
    device: str | None = None,
    local_files_only: bool = True,
    probe_k: int = LEXICAL_PROBE_K,
    mode_threshold: float = DEFAULT_MODE_SIMILARITY_THRESHOLD,
) -> CompiledProbeRunner:
    """Compose the verified dense index, catalog text, and fixed Probe policy."""

    if type(calibration) is not BoundTransparencyCalibration:
        raise TypeError("calibration must be a BoundTransparencyCalibration")
    retriever = create_dense_retriever(
        index_path=index_path,
        release_dir=release_dir,
        catalog_path=catalog_path,
        device=device,
        local_files_only=local_files_only,
    )
    calibration.validate_runtime(
        catalog_id=retriever.index.manifest.catalog_id,
        release_id=retriever.index.manifest.catalog_semantic_release_id,
        dense_index_id=retriever.index.index_id,
        probe_k=probe_k,
        mode_threshold=mode_threshold,
    )
    if release_dir is None:
        raise ValueError("release_dir is required")
    source = Path(release_dir) / "catalog.jsonl" if catalog_path is None else Path(catalog_path)
    documents = load_product_documents(
        source,
        expected_parent_asins=set(retriever.index.parent_asins),
    )
    return CompiledProbeRunner(
        retriever=retriever,
        lexical_probe=LexicalProbe(documents, probe_k=probe_k),
        calibration=calibration.calibration,
        probe_k=probe_k,
        mode_threshold=mode_threshold,
    )


def create_resolved_compiled_probe_runner(
    *,
    index_path: str | Path | None,
    release_dir: str | Path | None,
    calibration: BoundTransparencyCalibration,
    catalog_path: str | Path | None = None,
    device: str | None = None,
    local_files_only: bool = True,
    probe_k: int = LEXICAL_PROBE_K,
    mode_threshold: float = DEFAULT_MODE_SIMILARITY_THRESHOLD,
) -> ResolvedCompiledProbeRunner:
    """Compose hard-constraint resolution and the fixed Probe over one binding."""

    probe_runner = create_compiled_probe_runner(
        index_path=index_path,
        release_dir=release_dir,
        calibration=calibration,
        catalog_path=catalog_path,
        device=device,
        local_files_only=local_files_only,
        probe_k=probe_k,
        mode_threshold=mode_threshold,
    )
    if release_dir is None:
        raise ValueError("release_dir is required")
    release_path = Path(release_dir)
    try:
        release = load_catalog_semantic_release(release_path)
    except (CatalogSemanticError, OSError, RecursionError, TypeError, ValueError) as error:
        raise DenseIndexIntegrityError("active semantic release is invalid") from error
    source = release_path / "catalog.jsonl" if catalog_path is None else Path(catalog_path)
    evidence_index = build_retrieval_evidence_index(
        source,
        catalog_id=release.manifest.catalog_id,
        catalog_semantic_release_id=release.release_id,
        expected_parent_asins=set(probe_runner.dense_index.parent_asins),
    )
    resolver = HardMaskResolver(
        release=release,
        evidence_index=evidence_index,
        dense_index=probe_runner.dense_index,
    )
    return ResolvedCompiledProbeRunner(
        resolver=resolver,
        probe_runner=probe_runner,
    )


def create_retrieval_controller(
    *,
    index_path: str | Path | None,
    release_dir: str | Path | None,
    catalog_path: str | Path | None = None,
    device: str | None = None,
    local_files_only: bool = True,
    policy: FormalRetrievalPolicy | None = None,
    diversity_policy: VectorDiversityPolicy | None = None,
) -> RetrievalController:
    """Compose the release-bound formal retrieval stack over one 50k catalog."""

    if release_dir is None:
        raise ValueError("release_dir is required")
    resolved_policy = FormalRetrievalPolicy() if policy is None else policy
    if type(resolved_policy) is not FormalRetrievalPolicy:
        raise TypeError("policy must be an exact FormalRetrievalPolicy")
    if diversity_policy is not None and type(diversity_policy) is not VectorDiversityPolicy:
        raise TypeError("diversity_policy must be an exact VectorDiversityPolicy")

    release_path = Path(release_dir)
    retriever = create_dense_retriever(
        index_path=index_path,
        release_dir=release_path,
        catalog_path=catalog_path,
        device=device,
        local_files_only=local_files_only,
    )
    try:
        release = load_catalog_semantic_release(release_path)
    except (CatalogSemanticError, OSError, RecursionError, TypeError, ValueError) as error:
        raise DenseIndexIntegrityError("active semantic release is invalid") from error
    source = release_path / "catalog.jsonl" if catalog_path is None else Path(catalog_path)
    expected = set(retriever.index.parent_asins)
    documents = load_product_documents(source, expected_parent_asins=expected)
    evidence_index = build_retrieval_evidence_index(
        source,
        catalog_id=release.manifest.catalog_id,
        catalog_semantic_release_id=release.release_id,
        expected_parent_asins=expected,
    )
    resolver = HardMaskResolver(
        release=release,
        evidence_index=evidence_index,
        dense_index=retriever.index,
    )
    return RetrievalController(
        retriever=retriever,
        lexical_route=LexicalProbe(documents, probe_k=resolved_policy.route_k),
        facet_route=FacetRoute(evidence_index=evidence_index),
        hard_mask_resolver=resolver,
        policy=resolved_policy,
        diversity_policy=diversity_policy,
    )


def _catalog_content_id(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise DenseIndexIntegrityError("catalog is unavailable for index binding") from error
    return f"sha256:{digest.hexdigest()}"
