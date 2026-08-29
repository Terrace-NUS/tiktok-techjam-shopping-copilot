"""Command-line entry point for owner-approved Gate-B capability candidates."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ..errors import CatalogSemanticError, GateBBundleBusyError
from .gate_b_approval_bundle import (
    validate_gate_b_candidate_bundle,
    write_gate_b_candidate_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the approved Gate-B CLI parser."""

    parser = argparse.ArgumentParser(
        prog="catalog-facet-gate-b",
        description="Publish exact-scope capabilities from an owner-approved review proposal.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="materialize approved Gate-B capabilities")
    _add_inputs(build)
    build.add_argument("output_dir", type=Path, help="Git-ignored Gate-B candidate directory")
    validate = commands.add_parser(
        "validate",
        help="rebuild from every pinned input and validate an existing candidate",
    )
    validate.add_argument("output_dir", type=Path, help="generated Gate-B candidate directory")
    _add_inputs(validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one approved Gate-B command and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            candidate = write_gate_b_candidate_bundle(
                args.catalog,
                args.category_candidate,
                args.gate_a_candidate,
                args.resolution_candidate,
                args.public_set,
                args.gate_b_review,
                args.gate_b_selection,
                args.output_dir,
            )
            print(
                "catalog-facet-gate-b: owner approval recorded; "
                f"{len(candidate.capabilities.entries)} exact-scope capabilities published; "
                "runtime integration pending"
            )
        else:
            validate_gate_b_candidate_bundle(
                args.output_dir,
                catalog_path=args.catalog,
                category_candidate_dir=args.category_candidate,
                gate_a_candidate_dir=args.gate_a_candidate,
                resolution_candidate_dir=args.resolution_candidate,
                public_set_path=args.public_set,
                gate_b_review_dir=args.gate_b_review,
                gate_b_selection_path=args.gate_b_selection,
            )
            print(f"catalog-facet-gate-b: valid approved candidate {args.output_dir}")
    except (
        CatalogSemanticError,
        GateBBundleBusyError,
        OSError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        parser.exit(1, f"catalog-facet-gate-b: error: {error}\n")
    return 0


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("catalog", type=Path, help="exact read-only raw catalog JSONL")
    parser.add_argument("category_candidate", type=Path, help="validated CS1 candidate")
    parser.add_argument("gate_a_candidate", type=Path, help="validated Gate-A candidate")
    parser.add_argument("resolution_candidate", type=Path, help="validated CS3 candidate")
    parser.add_argument("public_set", type=Path, help="exact official public JSONL")
    parser.add_argument("gate_b_review", type=Path, help="validated Gate-B review packet")
    parser.add_argument("gate_b_selection", type=Path, help="owner-approved Gate-B selection")


if __name__ == "__main__":
    raise SystemExit(main())
