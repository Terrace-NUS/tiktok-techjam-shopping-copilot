"""Command-line entry point for reviewed CS2 Gate-A candidates."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ..errors import CatalogSemanticError, GateABundleBusyError
from .gate_a_bundle import (
    validate_gate_a_candidate_bundle,
    write_gate_a_candidate_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser without reading process-global arguments."""

    parser = argparse.ArgumentParser(
        prog="catalog-facet-gate-a",
        description="Build and validate reviewed normative Gate-A candidates.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="materialize reviewed Gate-A artifacts")
    _add_inputs(build)
    build.add_argument("output_dir", type=Path, help="Git-ignored candidate directory")

    validate = commands.add_parser(
        "validate",
        help="rebuild from all upstream truth and validate a candidate bundle",
    )
    validate.add_argument("output_dir", type=Path, help="generated candidate directory")
    _add_inputs(validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one Gate-A command and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            candidate = write_gate_a_candidate_bundle(
                args.catalog,
                args.category_candidate,
                args.profile_selection,
                args.source_profile,
                args.gate_a_selection,
                args.output_dir,
            )
            audit = candidate.price_audits[0]
            print(
                "catalog-facet-gate-a: candidate "
                f"{len(candidate.facet_schema.facets)} facets, "
                f"{len(candidate.bindings.bindings)} bindings, "
                f"{audit.valid_count} valid price facts, "
                f"catalog={candidate.catalog_id}"
            )
        else:
            validate_gate_a_candidate_bundle(
                args.output_dir,
                catalog_path=args.catalog,
                category_candidate_dir=args.category_candidate,
                profile_selection_path=args.profile_selection,
                source_profile_dir=args.source_profile,
                gate_a_selection_path=args.gate_a_selection,
            )
            print(f"catalog-facet-gate-a: valid bundle {args.output_dir}")
    except (
        CatalogSemanticError,
        GateABundleBusyError,
        OSError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        parser.exit(1, f"catalog-facet-gate-a: error: {error}\n")
    return 0


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("catalog", type=Path, help="exact raw catalog JSONL path")
    parser.add_argument(
        "category_candidate",
        type=Path,
        help="validated CS1 category candidate directory",
    )
    parser.add_argument(
        "profile_selection",
        type=Path,
        help="source-controlled Gate-A profile selection JSON",
    )
    parser.add_argument(
        "source_profile",
        type=Path,
        help="validated CS2 source-profile bundle directory",
    )
    parser.add_argument(
        "gate_a_selection",
        type=Path,
        help="source-controlled reviewed Gate-A selection JSON",
    )


if __name__ == "__main__":
    raise SystemExit(main())
