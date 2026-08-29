"""Command-line entry point for CS5A runtime projection candidates."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ..errors import CatalogSemanticError, RuntimeProjectionBundleBusyError
from .bundle import validate_runtime_projection_bundle, write_runtime_projection_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="catalog-runtime-projection",
        description="Project approved Gate-B price semantics into session runtime artifacts.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="materialize the CS5 runtime candidate")
    _add_inputs(build)
    build.add_argument("output_dir", type=Path, help="Git-ignored CS5 candidate directory")
    validate = commands.add_parser(
        "validate",
        help="rebuild the approval chain and validate an existing CS5 candidate",
    )
    validate.add_argument("output_dir", type=Path, help="generated CS5A candidate directory")
    _add_inputs(validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            build = write_runtime_projection_bundle(
                args.catalog,
                args.category_candidate,
                args.gate_a_candidate,
                args.resolution_candidate,
                args.public_set,
                args.gate_b_review,
                args.gate_b_selection,
                args.gate_b_candidate,
                args.output_dir,
            )
            print(
                "catalog-runtime-projection: "
                f"{len(build.runtime_registry.entries)} session facets projected; "
                "grounding implemented; retrieval and gateway not integrated"
            )
        else:
            validate_runtime_projection_bundle(
                args.output_dir,
                catalog_path=args.catalog,
                category_candidate_dir=args.category_candidate,
                gate_a_candidate_dir=args.gate_a_candidate,
                resolution_candidate_dir=args.resolution_candidate,
                public_set_path=args.public_set,
                gate_b_review_dir=args.gate_b_review,
                gate_b_selection_path=args.gate_b_selection,
                gate_b_candidate_dir=args.gate_b_candidate,
            )
            print(f"catalog-runtime-projection: valid CS5 candidate {args.output_dir}")
    except (
        CatalogSemanticError,
        RuntimeProjectionBundleBusyError,
        OSError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        parser.exit(1, f"catalog-runtime-projection: error: {error}\n")
    return 0


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("catalog", type=Path, help="exact read-only raw catalog JSONL")
    parser.add_argument("category_candidate", type=Path, help="validated CS1 candidate")
    parser.add_argument("gate_a_candidate", type=Path, help="validated Gate-A candidate")
    parser.add_argument("resolution_candidate", type=Path, help="validated CS3 candidate")
    parser.add_argument("public_set", type=Path, help="exact official public JSONL")
    parser.add_argument("gate_b_review", type=Path, help="validated Gate-B review packet")
    parser.add_argument("gate_b_selection", type=Path, help="owner-approved Gate-B selection")
    parser.add_argument("gate_b_candidate", type=Path, help="validated approved Gate-B candidate")


if __name__ == "__main__":
    raise SystemExit(main())
