"""Command-line interface for deterministic raw catalog profiling."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .bundle import ProfileBundleBusyError, write_profile_bundle
from .models import ProfileConfig
from .profiler import CatalogChangedError


def _nonnegative_integer(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser without reading process-global arguments."""

    parser = argparse.ArgumentParser(
        prog="catalog-profile",
        description="Read a JSONL catalog and write deterministic raw profiling reports.",
    )
    parser.add_argument("catalog", type=Path, help="input catalog JSONL path")
    parser.add_argument("output_dir", type=Path, help="generated report directory")
    parser.add_argument("--seed", default="raw-profile-v1", help="stable sample seed")
    parser.add_argument(
        "--sample-limit",
        type=_nonnegative_integer,
        default=20,
        help="maximum stable examples retained per key or diagnostic",
    )
    parser.add_argument(
        "--top-value-limit",
        type=_nonnegative_integer,
        default=50,
        help="maximum exact raw values retained per details key",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one profiling pass and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = ProfileConfig(
            seed=args.seed,
            sample_limit=args.sample_limit,
            top_value_limit=args.top_value_limit,
        )
        profile = write_profile_bundle(args.catalog, args.output_dir, config=config)
    except (
        CatalogChangedError,
        OSError,
        ProfileBundleBusyError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        parser.exit(1, f"catalog-profile: error: {error}\n")

    print(
        "catalog-profile: "
        f"{profile.product_row_count} rows, "
        f"{len(profile.category_nodes)} category nodes, "
        f"{len(profile.detail_keys)} details keys, "
        f"sha256={profile.catalog_sha256}"
    )
    return 0
