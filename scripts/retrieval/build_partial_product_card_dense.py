#!/usr/bin/env python3
"""Build a hybrid Dense index by re-embedding only verified product-card targets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shopping_copilot.catalog.product_facts import load_product_fact_sidecar  # noqa: E402
from shopping_copilot.retrieval import (  # noqa: E402
    SentenceTransformerTextEmbedder,
    load_dense_index,
    load_product_documents,
    replace_product_documents,
    write_partially_reembedded_dense_index,
)

REPORT_SCHEMA = "shopping-copilot/partial-product-card-dense-build/v1"


def main() -> int:
    args = _parse_args()
    started = time.perf_counter()
    base = load_dense_index(args.base_index)
    cards = load_product_fact_sidecar(args.sidecar, catalog_path=args.catalog)
    base_documents = tuple(
        sorted(
            load_product_documents(
                args.catalog,
                expected_parent_asins=set(base.parent_asins),
            ),
            key=lambda item: item.parent_asin,
        )
    )
    projected = replace_product_documents(base_documents, cards)
    projected_by_asin = {item.parent_asin: item for item in projected}
    replacements = {parent_asin: projected_by_asin[parent_asin] for parent_asin in cards}

    embed_started = time.perf_counter()
    embedder = SentenceTransformerTextEmbedder(
        base.manifest.embedding,
        device=args.device,
        local_files_only=not args.allow_download,
        show_progress_bar=True,
    )
    index = write_partially_reembedded_dense_index(
        args.base_index,
        args.output,
        base_documents=base_documents,
        replacement_documents=replacements,
        embedder=embedder,
        batch_size=args.batch_size,
    )
    build_seconds = time.perf_counter() - embed_started

    replacement_rows = np.asarray(
        [base.parent_asins.index(parent_asin) for parent_asin in sorted(cards)],
        dtype=np.int64,
    )
    all_rows = np.arange(base.manifest.product_count, dtype=np.int64)
    unchanged_rows = np.setdiff1d(all_rows, replacement_rows, assume_unique=True)
    exact_unchanged = int(
        np.count_nonzero(
            np.all(base.vectors[unchanged_rows] == index.vectors[unchanged_rows], axis=1)
        )
    )
    changed_replacements = int(
        np.count_nonzero(
            np.any(base.vectors[replacement_rows] != index.vectors[replacement_rows], axis=1)
        )
    )
    report = {
        "schema": REPORT_SCHEMA,
        "base_index": str(args.base_index.resolve()),
        "base_index_id": base.index_id,
        "output_index": str(args.output.resolve()),
        "output_index_id": index.index_id,
        "sidecar": str(args.sidecar.resolve()),
        "product_count": index.manifest.product_count,
        "replacement_count": len(cards),
        "changed_replacement_vector_count": changed_replacements,
        "unchanged_product_count": len(unchanged_rows),
        "exactly_unchanged_vector_count": exact_unchanged,
        "embedding": {
            "model_id": index.manifest.embedding.model_id,
            "model_revision": index.manifest.embedding.model_revision,
            "dimension": index.manifest.embedding.dimension,
        },
        "document_corpus_id": index.manifest.document_corpus_id,
        "build_seconds": build_seconds,
        "total_seconds": time.perf_counter() - started,
    }
    report_path = args.output.parent / f"{args.output.name}-build.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-index",
        type=Path,
        default=ROOT / "artifacts/retrieval/dense-v0",
    )
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.jsonl")
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=ROOT / "data/benchmark_product_cards/public_200_v1/product-facts.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/retrieval/dense-public-200-replaced-v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
