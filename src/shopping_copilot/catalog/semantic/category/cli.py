"""Command-line entry point for CS1 category proposal and candidate builds."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ..errors import CatalogSemanticError, CategoryBundleBusyError
from .bundle import (
    validate_category_bundle,
    write_category_candidate_bundle,
    write_category_graph_proposal_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser without consulting process-global arguments."""

    parser = argparse.ArgumentParser(
        prog="catalog-category",
        description="Build and validate deterministic CS1 category candidates.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    propose = commands.add_parser(
        "propose",
        help="strictly scan the 50k catalog and write the Pass-A graph proposal",
    )
    propose.add_argument("catalog", type=Path, help="exact raw catalog JSONL path")
    propose.add_argument("output_dir", type=Path, help="Git-ignored proposal bundle directory")

    build = commands.add_parser(
        "build",
        help="materialize reviewed scopes and write Pass-B category candidates",
    )
    build.add_argument("catalog", type=Path, help="exact raw catalog JSONL path")
    build.add_argument(
        "scope_selection",
        type=Path,
        help="source-controlled category-scope-selection/v0 JSON path",
    )
    build.add_argument("output_dir", type=Path, help="Git-ignored candidate bundle directory")

    validate = commands.add_parser(
        "validate",
        help="reload and independently validate a proposal or candidate bundle",
    )
    validate.add_argument("output_dir", type=Path, help="generated category bundle directory")
    validate.add_argument(
        "--catalog",
        type=Path,
        help="exact raw catalog; required when validating a candidate bundle",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one CS1 command and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "propose":
            proposal = write_category_graph_proposal_bundle(args.catalog, args.output_dir)
            print(
                "catalog-category: proposal "
                f"{proposal.product_count} products, "
                f"{len(proposal.nodes)} canonical nodes, "
                f"{len(proposal.collisions)} collision groups, "
                f"graph={proposal.category_graph_id}"
            )
        elif args.command == "build":
            candidate = write_category_candidate_bundle(
                args.catalog,
                args.scope_selection,
                args.output_dir,
            )
            print(
                "catalog-category: candidate "
                f"{len(candidate.registry.scopes)} scopes, "
                f"{len(candidate.assignments.assignments)} assignments, "
                f"graph={candidate.registry.category_graph_id}"
            )
        else:
            validate_category_bundle(args.output_dir, catalog_path=args.catalog)
            print(f"catalog-category: valid bundle {args.output_dir}")
    except (
        CatalogSemanticError,
        CategoryBundleBusyError,
        OSError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        parser.exit(1, f"catalog-category: error: {error}\n")
    return 0
