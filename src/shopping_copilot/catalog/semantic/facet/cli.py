"""Command-line entry point for the CS2 Gate-A source-profile checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ..errors import CatalogSemanticError, FacetProfileBundleBusyError
from .bundle import (
    validate_gate_a_source_profile_bundle,
    write_gate_a_source_profile_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser without reading process-global arguments."""

    parser = argparse.ArgumentParser(
        prog="catalog-facet-profile",
        description="Build and validate deterministic CS2 Gate-A source profiles.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser(
        "build",
        help="profile exact structured sources inside reviewed category scopes",
    )
    _add_inputs(build)
    build.add_argument("output_dir", type=Path, help="Git-ignored profile bundle directory")

    validate = commands.add_parser(
        "validate",
        help="rebuild from upstream truth and validate a generated profile bundle",
    )
    validate.add_argument("output_dir", type=Path, help="generated profile bundle directory")
    _add_inputs(validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one CS2 profiling command and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            profile = write_gate_a_source_profile_bundle(
                args.catalog,
                args.category_candidate,
                args.profile_selection,
                args.output_dir,
            )
            print(
                "catalog-facet-profile: proposal "
                f"{len(profile.sources)} sources, "
                f"{len(profile.scopes)} scopes, "
                f"{len(profile.scope_source_profiles)} scope-source rows, "
                f"catalog={profile.catalog_id}"
            )
        else:
            validate_gate_a_source_profile_bundle(
                args.output_dir,
                catalog_path=args.catalog,
                category_candidate_dir=args.category_candidate,
                selection_path=args.profile_selection,
            )
            print(f"catalog-facet-profile: valid bundle {args.output_dir}")
    except (
        CatalogSemanticError,
        FacetProfileBundleBusyError,
        OSError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        parser.exit(1, f"catalog-facet-profile: error: {error}\n")
    return 0


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("catalog", type=Path, help="exact raw catalog JSONL path")
    parser.add_argument(
        "category_candidate",
        type=Path,
        help="validated CS1 category candidate bundle directory",
    )
    parser.add_argument(
        "profile_selection",
        type=Path,
        help="source-controlled gate-a-profile-selection/v0 JSON path",
    )
