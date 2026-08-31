"""Atomic build and fail-closed loading for the dense-index bundle."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from shopping_copilot.catalog.semantic import (
    CanonicalJsonError,
    canonical_json_bytes,
    content_id_for_bytes,
)
from shopping_copilot.catalog.semantic.release import load_catalog_semantic_release

from .dense import DenseIndex
from .documents import ProductDocument, load_product_documents
from .embedding import FloatMatrix, TextEmbedder, normalize_rows
from .errors import DenseIndexBusyError, DenseIndexIntegrityError
from .models import (
    ARTIFACT_FILENAMES,
    DENSE_INDEX_BUILDER_VERSION,
    DENSE_INDEX_FILENAMES,
    DENSE_INDEX_MANIFEST_FILENAME,
    DENSE_INDEX_SCHEMA,
    PARENT_ASINS_FILENAME,
    PRODUCT_DOCUMENT_TEMPLATE_ID,
    VECTORS_FILENAME,
    ArtifactKind,
    DenseArtifactRef,
    DenseIndexManifest,
    EmbeddingSpec,
)

_DOCUMENT_CORPUS_DOMAIN = b"shopping-copilot/product-document-corpus/v0\0"
PARTIAL_DENSE_INDEX_BUILDER_VERSION = "dense_index_partial_reembed_v1"
PARTIAL_PRODUCT_DOCUMENT_TEMPLATE_ID = "product_document_partial_fact_card_v1"


def write_dense_index(
    release_dir: str | Path,
    output_dir: str | Path,
    *,
    embedder: TextEmbedder,
    batch_size: int = 128,
    expected_product_count: int = 50_000,
) -> DenseIndex:
    """Build one release-bound exact dense index and publish it atomically."""

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    release_path = Path(release_dir)
    target = Path(output_dir)
    release = load_catalog_semantic_release(
        release_path,
        expected_product_count=expected_product_count,
    )
    if target.exists():
        existing = load_dense_index(
            target,
            expected_catalog_id=release.manifest.catalog_id,
            expected_release_id=release.release_id,
        )
        if existing.manifest.embedding != embedder.spec:
            raise DenseIndexIntegrityError(
                "existing dense index uses a different embedding specification"
            )
        if (
            existing.manifest.builder_version != DENSE_INDEX_BUILDER_VERSION
            or existing.manifest.document_template_id != PRODUCT_DOCUMENT_TEMPLATE_ID
        ):
            raise DenseIndexIntegrityError(
                "existing dense index was built by a different document contract"
            )
        return existing

    expected_asins = tuple(
        assignment.parent_asin for assignment in release.product_category_assignments.assignments
    )
    documents = load_product_documents(
        release_path / "catalog.jsonl",
        expected_parent_asins=frozenset(expected_asins),
    )
    documents = tuple(sorted(documents, key=lambda item: item.parent_asin))
    if len(documents) != expected_product_count:
        raise DenseIndexIntegrityError("document count differs from expected product count")
    texts = [document.text for document in documents]
    try:
        encoded = embedder.encode_documents(texts, batch_size=batch_size)
        vectors = normalize_rows(
            np.asarray(encoded, dtype=np.float32),
            name="catalog document embeddings",
        )
    except ValueError as error:
        raise DenseIndexIntegrityError(str(error)) from error
    if vectors.shape != (len(documents), embedder.spec.dimension):
        raise DenseIndexIntegrityError("embedding backend returned the wrong matrix shape")

    _assert_catalog_identity(
        release_path / "catalog.jsonl",
        expected_catalog_id=release.manifest.catalog_id,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_index_writer(target):
        if target.exists():
            raise DenseIndexIntegrityError("dense index target appeared during publication")
        with TemporaryDirectory(prefix=".dense-index-", dir=target.parent) as temp:
            generation = Path(temp) / "generation"
            generation.mkdir()
            parent_asins = tuple(document.parent_asin for document in documents)
            (generation / PARENT_ASINS_FILENAME).write_bytes(canonical_json_bytes(parent_asins))
            with (generation / VECTORS_FILENAME).open("wb") as stream:
                np.save(stream, np.ascontiguousarray(vectors, dtype=np.float32), allow_pickle=False)
            refs = _artifact_refs(generation)
            manifest = DenseIndexManifest(
                schema=DENSE_INDEX_SCHEMA,
                builder_version=DENSE_INDEX_BUILDER_VERSION,
                catalog_id=release.manifest.catalog_id,
                catalog_semantic_release_id=release.release_id,
                document_template_id=PRODUCT_DOCUMENT_TEMPLATE_ID,
                document_corpus_id=document_corpus_id(documents),
                product_count=len(documents),
                embedding=embedder.spec,
                vector_dtype="float32",
                artifacts=refs,
            )
            (generation / DENSE_INDEX_MANIFEST_FILENAME).write_bytes(
                encode_dense_index_manifest(manifest)
            )
            load_dense_index(
                generation,
                expected_catalog_id=release.manifest.catalog_id,
                expected_release_id=release.release_id,
                mmap=False,
            )
            if target.exists():
                raise DenseIndexIntegrityError("dense index target appeared during publication")
            os.replace(generation, target)
    return load_dense_index(
        target,
        expected_catalog_id=release.manifest.catalog_id,
        expected_release_id=release.release_id,
    )


def write_partially_reembedded_dense_index(
    base_index_dir: str | Path,
    output_dir: str | Path,
    *,
    base_documents: Sequence[ProductDocument],
    replacement_documents: Mapping[str, ProductDocument],
    embedder: TextEmbedder,
    batch_size: int = 128,
) -> DenseIndex:
    """Copy a bound index and re-embed only explicitly replaced product documents."""

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(base_documents, Sequence):
        raise TypeError("base_documents must be a sequence")
    if not isinstance(replacement_documents, Mapping):
        raise TypeError("replacement_documents must be a mapping")
    base_path = Path(base_index_dir)
    target = Path(output_dir)
    if base_path.resolve() == target.resolve():
        raise ValueError("partial dense output must differ from the base index")

    base = load_dense_index(base_path)
    if embedder.spec != base.manifest.embedding:
        raise DenseIndexIntegrityError("embedding specification differs from the base index")
    documents = tuple(base_documents)
    if any(type(document) is not ProductDocument for document in documents):
        raise TypeError("base_documents must contain exact ProductDocument values")
    if tuple(item.parent_asin for item in documents) != base.parent_asins:
        raise DenseIndexIntegrityError("base documents differ from the dense index product order")
    if document_corpus_id(documents) != base.manifest.document_corpus_id:
        raise DenseIndexIntegrityError("base documents differ from the dense index corpus binding")

    replacements = _validate_replacement_documents(
        replacement_documents,
        parent_asins=base.parent_asins,
    )
    hybrid_documents = tuple(
        replacements.get(document.parent_asin, document) for document in documents
    )
    hybrid_corpus_id = document_corpus_id(hybrid_documents)
    if hybrid_corpus_id == base.manifest.document_corpus_id:
        raise DenseIndexIntegrityError("replacement documents did not change the document corpus")

    if target.exists():
        existing = load_dense_index(
            target,
            expected_catalog_id=base.manifest.catalog_id,
            expected_release_id=base.manifest.catalog_semantic_release_id,
        )
        if (
            existing.manifest.builder_version != PARTIAL_DENSE_INDEX_BUILDER_VERSION
            or existing.manifest.document_template_id != PARTIAL_PRODUCT_DOCUMENT_TEMPLATE_ID
            or existing.manifest.document_corpus_id != hybrid_corpus_id
            or existing.manifest.embedding != embedder.spec
        ):
            raise DenseIndexIntegrityError("existing partial dense index uses another contract")
        return existing

    ordered_ids = tuple(sorted(replacements))
    replacement_texts = [replacements[parent_asin].text for parent_asin in ordered_ids]
    try:
        encoded = embedder.encode_documents(replacement_texts, batch_size=batch_size)
        replacement_vectors = normalize_rows(
            np.asarray(encoded, dtype=np.float32),
            name="replacement document embeddings",
        )
    except ValueError as error:
        raise DenseIndexIntegrityError(str(error)) from error
    if replacement_vectors.shape != (len(ordered_ids), embedder.spec.dimension):
        raise DenseIndexIntegrityError("embedding backend returned the wrong replacement shape")

    vectors = np.array(base.vectors, dtype=np.float32, order="C", copy=True)
    row_by_asin = {parent_asin: row for row, parent_asin in enumerate(base.parent_asins)}
    for replacement_row, parent_asin in enumerate(ordered_ids):
        vectors[row_by_asin[parent_asin]] = replacement_vectors[replacement_row]

    target.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_index_writer(target):
        if target.exists():
            raise DenseIndexIntegrityError("partial dense target appeared during publication")
        with TemporaryDirectory(prefix=".dense-index-partial-", dir=target.parent) as temp:
            generation = Path(temp) / "generation"
            generation.mkdir()
            (generation / PARENT_ASINS_FILENAME).write_bytes(
                canonical_json_bytes(base.parent_asins)
            )
            with (generation / VECTORS_FILENAME).open("wb") as stream:
                np.save(stream, vectors, allow_pickle=False)
            manifest = DenseIndexManifest(
                schema=DENSE_INDEX_SCHEMA,
                builder_version=PARTIAL_DENSE_INDEX_BUILDER_VERSION,
                catalog_id=base.manifest.catalog_id,
                catalog_semantic_release_id=base.manifest.catalog_semantic_release_id,
                document_template_id=PARTIAL_PRODUCT_DOCUMENT_TEMPLATE_ID,
                document_corpus_id=hybrid_corpus_id,
                product_count=base.manifest.product_count,
                embedding=embedder.spec,
                vector_dtype="float32",
                artifacts=_artifact_refs(generation),
            )
            (generation / DENSE_INDEX_MANIFEST_FILENAME).write_bytes(
                encode_dense_index_manifest(manifest)
            )
            load_dense_index(
                generation,
                expected_catalog_id=base.manifest.catalog_id,
                expected_release_id=base.manifest.catalog_semantic_release_id,
                mmap=False,
            )
            if target.exists():
                raise DenseIndexIntegrityError("partial dense target appeared during publication")
            os.replace(generation, target)
    return load_dense_index(
        target,
        expected_catalog_id=base.manifest.catalog_id,
        expected_release_id=base.manifest.catalog_semantic_release_id,
    )


def load_dense_index(
    index_dir: str | Path,
    *,
    expected_catalog_id: str | None = None,
    expected_release_id: str | None = None,
    mmap: bool = True,
) -> DenseIndex:
    """Verify every byte and invariant before exposing an immutable dense index."""

    target = Path(index_dir)
    try:
        observed_names = {path.name for path in target.iterdir()}
    except OSError as error:
        raise DenseIndexIntegrityError("dense index directory is unavailable") from error
    if observed_names != DENSE_INDEX_FILENAMES:
        raise DenseIndexIntegrityError("dense index members are incomplete or unexpected")
    try:
        manifest_bytes = (target / DENSE_INDEX_MANIFEST_FILENAME).read_bytes()
        manifest = decode_dense_index_manifest(manifest_bytes)
    except OSError as error:
        raise DenseIndexIntegrityError("dense index manifest is unavailable") from error
    if expected_catalog_id is not None and manifest.catalog_id != expected_catalog_id:
        raise DenseIndexIntegrityError("dense index catalog ID differs from expected catalog")
    if (
        expected_release_id is not None
        and manifest.catalog_semantic_release_id != expected_release_id
    ):
        raise DenseIndexIntegrityError("dense index release ID differs from expected release")

    refs = {item.kind: item for item in manifest.artifacts}
    for kind, filename in ARTIFACT_FILENAMES.items():
        path = target / filename
        digest, byte_size = _hash_file(path)
        if digest != refs[kind].content_id or byte_size != refs[kind].byte_size:
            raise DenseIndexIntegrityError(f"dense index artifact bytes differ: {kind}")

    try:
        parent_asin_bytes = (target / PARENT_ASINS_FILENAME).read_bytes()
    except OSError as error:
        raise DenseIndexIntegrityError("parent ASIN artifact is unavailable") from error
    parent_asins = _decode_parent_asins(
        parent_asin_bytes,
        expected_count=manifest.product_count,
    )
    try:
        raw_vectors = np.load(
            target / VECTORS_FILENAME,
            mmap_mode="r" if mmap else None,
            allow_pickle=False,
        )
    except (OSError, TypeError, ValueError) as error:
        raise DenseIndexIntegrityError("vector artifact cannot be loaded") from error
    vectors = _validate_loaded_vectors(raw_vectors, manifest=manifest)
    index_id = content_id_for_bytes(manifest_bytes)
    index = DenseIndex(
        index_id=index_id,
        manifest=manifest,
        parent_asins=parent_asins,
        vectors=vectors,
    )
    # Recheck after materializing the owned snapshot. This closes the window
    # between the first integrity hash and np.load/read_bytes path access.
    for kind, filename in ARTIFACT_FILENAMES.items():
        digest, byte_size = _hash_file(target / filename)
        if digest != refs[kind].content_id or byte_size != refs[kind].byte_size:
            raise DenseIndexIntegrityError(f"dense index artifact changed while loading: {kind}")
    return index


def validate_dense_index(
    index_dir: str | Path,
    *,
    expected_catalog_id: str | None = None,
    expected_release_id: str | None = None,
) -> str:
    """Validate a bundle without loading an embedding model and return its ID."""

    return load_dense_index(
        index_dir,
        expected_catalog_id=expected_catalog_id,
        expected_release_id=expected_release_id,
    ).index_id


def encode_dense_index_manifest(manifest: DenseIndexManifest) -> bytes:
    """Encode one manifest as exact RFC 8785 JSON bytes."""

    return canonical_json_bytes(manifest)


def decode_dense_index_manifest(payload: bytes) -> DenseIndexManifest:
    """Strictly decode canonical manifest bytes without accepting extra fields."""

    try:
        decoded: object = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_nonfinite,
            object_pairs_hook=_object_without_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise DenseIndexIntegrityError("dense index manifest is not valid JSON") from error
    try:
        canonical = canonical_json_bytes(decoded)
    except (CanonicalJsonError, TypeError, ValueError) as error:
        raise DenseIndexIntegrityError("dense index manifest is not canonical JSON") from error
    if canonical != payload:
        raise DenseIndexIntegrityError("dense index manifest is not canonical")
    fields = _require_mapping(
        decoded,
        expected={
            "artifacts",
            "builder_version",
            "catalog_id",
            "catalog_semantic_release_id",
            "document_corpus_id",
            "document_template_id",
            "embedding",
            "product_count",
            "schema",
            "vector_dtype",
        },
        name="manifest",
    )
    embedding_fields = _require_mapping(
        fields["embedding"],
        expected={
            "backend",
            "backend_version",
            "dimension",
            "document_instruction",
            "max_sequence_length",
            "model_id",
            "model_revision",
            "normalization",
            "pooling",
            "query_instruction",
        },
        name="embedding",
    )
    raw_artifacts = fields["artifacts"]
    if type(raw_artifacts) is not list:
        raise DenseIndexIntegrityError("manifest artifacts must be an array")
    artifacts: list[DenseArtifactRef] = []
    for raw in raw_artifacts:
        item = _require_mapping(
            raw,
            expected={"byte_size", "content_id", "filename", "kind"},
            name="artifact",
        )
        try:
            artifacts.append(
                DenseArtifactRef(
                    kind=cast(ArtifactKind, item["kind"]),
                    filename=cast(str, item["filename"]),
                    content_id=cast(str, item["content_id"]),
                    byte_size=cast(int, item["byte_size"]),
                )
            )
        except (TypeError, ValueError) as error:
            raise DenseIndexIntegrityError(str(error)) from error
    try:
        embedding = EmbeddingSpec(
            backend=cast(str, embedding_fields["backend"]),
            backend_version=cast(str, embedding_fields["backend_version"]),
            model_id=cast(str, embedding_fields["model_id"]),
            model_revision=cast(str, embedding_fields["model_revision"]),
            dimension=cast(int, embedding_fields["dimension"]),
            max_sequence_length=cast(int, embedding_fields["max_sequence_length"]),
            query_instruction=cast(str, embedding_fields["query_instruction"]),
            document_instruction=cast(str, embedding_fields["document_instruction"]),
            pooling=cast(str, embedding_fields["pooling"]),
            normalization=cast(LiteralL2, embedding_fields["normalization"]),
        )
        return DenseIndexManifest(
            schema=cast(DenseSchema, fields["schema"]),
            builder_version=cast(str, fields["builder_version"]),
            catalog_id=cast(str, fields["catalog_id"]),
            catalog_semantic_release_id=cast(str, fields["catalog_semantic_release_id"]),
            document_template_id=cast(str, fields["document_template_id"]),
            document_corpus_id=cast(str, fields["document_corpus_id"]),
            product_count=cast(int, fields["product_count"]),
            embedding=embedding,
            vector_dtype=cast(LiteralFloat32, fields["vector_dtype"]),
            artifacts=tuple(artifacts),
        )
    except (TypeError, ValueError) as error:
        raise DenseIndexIntegrityError(str(error)) from error


def document_corpus_id(documents: Sequence[ProductDocument]) -> str:
    """Hash ordered rendered documents with an unambiguous length-prefix encoding."""

    digest = hashlib.sha256()
    digest.update(_DOCUMENT_CORPUS_DOMAIN)
    for document in documents:
        for value in (document.parent_asin, document.text):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
            digest.update(encoded)
    return f"sha256:{digest.hexdigest()}"


def _validate_replacement_documents(
    values: Mapping[str, ProductDocument],
    *,
    parent_asins: Sequence[str],
) -> dict[str, ProductDocument]:
    if not values:
        raise ValueError("replacement_documents must not be empty")
    allowed = frozenset(parent_asins)
    result: dict[str, ProductDocument] = {}
    for parent_asin, document in values.items():
        if type(parent_asin) is not str or not parent_asin.strip():
            raise ValueError("replacement_documents contains an invalid product ID")
        if type(document) is not ProductDocument:
            raise TypeError("replacement_documents must contain exact ProductDocument values")
        if document.parent_asin != parent_asin:
            raise ValueError("replacement document ID differs from its mapping key")
        if parent_asin not in allowed:
            raise KeyError(f"replacement document is outside the base index: {parent_asin}")
        result[parent_asin] = document
    return result


def _artifact_refs(generation: Path) -> tuple[DenseArtifactRef, ...]:
    refs: list[DenseArtifactRef] = []
    for kind in sorted(ARTIFACT_FILENAMES):
        filename = ARTIFACT_FILENAMES[kind]
        content_id, byte_size = _hash_file(generation / filename)
        refs.append(
            DenseArtifactRef(
                kind=kind,
                filename=filename,
                content_id=content_id,
                byte_size=byte_size,
            )
        )
    return tuple(refs)


def _decode_parent_asins(payload: bytes, *, expected_count: int) -> tuple[str, ...]:
    try:
        decoded: object = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise DenseIndexIntegrityError("parent ASIN artifact is not valid JSON") from error
    if canonical_json_bytes(decoded) != payload or type(decoded) is not list:
        raise DenseIndexIntegrityError("parent ASIN artifact is not canonical")
    values = cast(list[object], decoded)
    if len(values) != expected_count:
        raise DenseIndexIntegrityError("parent ASIN count differs from manifest")
    if any(type(value) is not str or not value or value != value.strip() for value in values):
        raise DenseIndexIntegrityError("parent ASIN artifact contains an invalid ID")
    parent_asins = tuple(cast(str, value) for value in values)
    if parent_asins != tuple(sorted(parent_asins)) or len(set(parent_asins)) != len(parent_asins):
        raise DenseIndexIntegrityError("parent ASIN artifact must be sorted and unique")
    return parent_asins


def _validate_loaded_vectors(
    raw: NDArray[np.generic],
    *,
    manifest: DenseIndexManifest,
) -> FloatMatrix:
    if raw.dtype != np.float32:
        raise DenseIndexIntegrityError("vector artifact dtype differs from manifest")
    if raw.shape != (manifest.product_count, manifest.embedding.dimension):
        raise DenseIndexIntegrityError("vector artifact shape differs from manifest")
    if not np.isfinite(raw).all():
        raise DenseIndexIntegrityError("vector artifact contains a non-finite value")
    norms = np.linalg.norm(raw, axis=1)
    if not np.allclose(norms, 1.0, rtol=2e-4, atol=2e-4):
        raise DenseIndexIntegrityError("vector artifact rows are not L2-normalized")
    return cast(FloatMatrix, raw)


def _assert_catalog_identity(path: Path, *, expected_catalog_id: str) -> None:
    actual, _ = _hash_file(path)
    if actual != expected_catalog_id:
        raise DenseIndexIntegrityError("catalog changed while the dense index was built")


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                byte_size += len(chunk)
    except OSError as error:
        raise DenseIndexIntegrityError(f"artifact is unavailable: {path.name}") from error
    return f"sha256:{digest.hexdigest()}", byte_size


@contextmanager
def _exclusive_index_writer(target: Path) -> Iterator[None]:
    resolved = target.resolve()
    lock = resolved.parent / f".{resolved.name}.write.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise DenseIndexBusyError(
            f"dense index publication is already running: {target}"
        ) from error
    try:
        os.close(descriptor)
        yield
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


class _DuplicateJsonKeyError(ValueError):
    pass


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_nonfinite(raw: str) -> object:
    raise ValueError(f"non-finite number: {raw}")


def _require_mapping(
    value: object,
    *,
    expected: set[str],
    name: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise DenseIndexIntegrityError(f"{name} must be an object")
    fields = cast(dict[str, object], value)
    if set(fields) != expected:
        raise DenseIndexIntegrityError(f"{name} has invalid fields")
    return fields


# Local aliases keep Literal casts readable on Python 3.10.
LiteralL2 = Literal["l2"]
LiteralFloat32 = Literal["float32"]
DenseSchema = Literal["shopping-copilot/dense-index-bundle/v0"]
