"""Command-line entry point for read-only CS3 resolution candidates."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ..errors import CatalogSemanticError, ResolutionBundleBusyError
from .resolution_bundle import (
    validate_resolution_candidate_bundle,
    write_resolution_candidate_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CS3 CLI parser without reading process-global arguments."""

    parser = argparse.ArgumentParser(
        prog="catalog-facet-resolution",
        description="Build auditable price evidence and a separate read-only query index.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="materialize CS3 evidence, index, and statistics")
    _add_inputs(build)
    build.add_argument("output_dir", type=Path, help="Git-ignored CS3 candidate directory")

    validate = commands.add_parser(
        "validate",
        help="rebuild from the frozen inputs and validate an existing CS3 bundle",
    )
    validate.add_argument("output_dir", type=Path, help="generated CS3 candidate directory")
    _add_inputs(validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one CS3 command and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            candidate = write_resolution_candidate_bundle(
                args.catalog,
                args.category_candidate,
                args.gate_a_candidate,
                args.output_dir,
            )
            root_rows = [
                row
                for row in candidate.stats.rows
                if row.scope_product_count == 50_000 and row.facet_id == "price"
            ]
            root = root_rows[0]
            print(
                "catalog-facet-resolution: candidate "
                f"{len(candidate.evidence_store.evidence)} evidence rows, "
                f"{root.known_count} known prices, {root.unknown_count} unknown prices, "
                "catalog unchanged"
            )
        else:
            validate_resolution_candidate_bundle(
                args.output_dir,
                catalog_path=args.catalog,
                category_candidate_dir=args.category_candidate,
                gate_a_candidate_dir=args.gate_a_candidate,
            )
            print(f"catalog-facet-resolution: valid read-only bundle {args.output_dir}")
    except (
        CatalogSemanticError,
        ResolutionBundleBusyError,
        OSError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        parser.exit(1, f"catalog-facet-resolution: error: {error}\n")
    return 0


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("catalog", type=Path, help="exact read-only raw catalog JSONL path")
    parser.add_argument(
        "category_candidate",
        type=Path,
        help="validated CS1 category candidate directory",
    )
    parser.add_argument(
        "gate_a_candidate",
        type=Path,
        help="validated reviewed Gate-A candidate directory",
    )


if __name__ == "__main__":
    raise SystemExit(main())
