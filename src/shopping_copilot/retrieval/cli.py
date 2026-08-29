"""CLI for building, validating, querying, and inspecting the R0 dense index."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

from .bundle import validate_dense_index, write_dense_index
from .embedding import SentenceTransformerTextEmbedder
from .errors import RetrievalError
from .factory import create_dense_retriever
from .models import default_embedding_spec
from .probe import FixedDenseProbe


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="retrieval-dense")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="build one immutable dense index")
    build.add_argument("release_dir", type=Path)
    build.add_argument("output_dir", type=Path)
    build.add_argument("--batch-size", type=int, default=128)
    build.add_argument("--device", help="Sentence Transformers device, e.g. cuda or cpu")
    build.add_argument(
        "--offline",
        action="store_true",
        help="require the pinned model revision to exist in the local cache",
    )

    validate = commands.add_parser("validate", help="verify a dense bundle without a model")
    validate.add_argument("index_dir", type=Path)
    validate.add_argument("--catalog-id")
    validate.add_argument("--release-id")

    query = commands.add_parser("query", help="run exact dense search and shadow coherence")
    query.add_argument("index_dir", type=Path)
    query.add_argument("q_sem")
    query.add_argument(
        "--release-dir",
        type=Path,
        required=True,
        help="active Catalog Semantic release that the index must match",
    )
    query.add_argument("--top-k", type=int, default=10)
    query.add_argument("--probe-k", type=int, default=40)
    query.add_argument("--device")
    query.add_argument(
        "--allow-download",
        action="store_true",
        help="allow a missing model revision to be fetched at query time",
    )

    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            spec = default_embedding_spec()
            embedder = SentenceTransformerTextEmbedder(
                spec,
                device=args.device,
                local_files_only=args.offline,
                show_progress_bar=True,
            )
            started = time.perf_counter()
            index = write_dense_index(
                args.release_dir,
                args.output_dir,
                embedder=embedder,
                batch_size=args.batch_size,
            )
            elapsed = time.perf_counter() - started
            print(
                "retrieval-dense: built "
                f"{index.index_id} products={index.manifest.product_count} "
                f"dimension={index.manifest.embedding.dimension} seconds={elapsed:.3f}"
            )
        elif args.command == "validate":
            index_id = validate_dense_index(
                args.index_dir,
                expected_catalog_id=args.catalog_id,
                expected_release_id=args.release_id,
            )
            print(f"retrieval-dense: valid {index_id}")
        else:
            if args.top_k < 0:
                raise ValueError("top_k must be a non-negative integer")
            if args.probe_k <= 0:
                raise ValueError("probe_k must be a positive integer")
            retriever = create_dense_retriever(
                index_path=args.index_dir,
                release_dir=args.release_dir,
                catalog_path=args.release_dir / "catalog.jsonl",
                device=args.device,
                local_files_only=not args.allow_download,
            )
            index = retriever.index
            started = time.perf_counter()
            ranking_depth = max(args.top_k, args.probe_k)
            result = retriever.search_with_scores(args.q_sem, top_k=ranking_depth)
            hits = result.hits[: args.top_k]
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            observation = FixedDenseProbe(index).observe(result, probe_k=args.probe_k)
            payload = {
                "index_id": index.index_id,
                "q_sem": args.q_sem,
                "latency_ms": round(elapsed_ms, 3),
                "hits": [
                    {
                        "parent_asin": hit.parent_asin,
                        "rank": hit.rank,
                        "score": hit.score,
                    }
                    for hit in hits
                ],
                "shadow_probe": {
                    "probe_k": observation.probe_k,
                    "n": observation.coherence.n,
                    "available": observation.coherence.available,
                    "reason": observation.coherence.reason,
                    "resultant_length": observation.coherence.resultant_length,
                    "debiased_pairwise_cosine": (observation.coherence.debiased_pairwise_cosine),
                },
            }
            print(json.dumps(payload, indent=2, ensure_ascii=False))
    except (
        ImportError,
        OSError,
        RecursionError,
        RetrievalError,
        TypeError,
        ValueError,
    ) as error:
        parser.exit(1, f"retrieval-dense: error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
