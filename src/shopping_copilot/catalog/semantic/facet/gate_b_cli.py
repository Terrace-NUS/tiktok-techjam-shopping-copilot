"""Command-line entry point for non-authoritative Gate-B review packets."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ..errors import CatalogSemanticError, GateBReviewBundleBusyError
from .gate_b_bundle import validate_gate_b_review_bundle, write_gate_b_review_bundle


def build_parser() -> argparse.ArgumentParser:
    """Create the Gate-B review CLI parser."""

    parser = argparse.ArgumentParser(
        prog="catalog-facet-gate-b-review",
        description="Build review evidence for price capabilities without enabling runtime use.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="materialize the Gate-B owner-review packet")
    _add_inputs(build)
    build.add_argument("output_dir", type=Path, help="Git-ignored review packet directory")
    validate = commands.add_parser(
        "validate",
        help="rebuild from all pinned inputs and validate an existing packet",
    )
    validate.add_argument("output_dir", type=Path, help="generated review packet directory")
    _add_inputs(validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one Gate-B review command and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = write_gate_b_review_bundle(
                args.catalog,
                args.category_candidate,
                args.gate_a_candidate,
                args.resolution_candidate,
                args.public_set,
                args.output_dir,
            )
            audit = result.public_target_audit
            print(
                "catalog-facet-gate-b-review: awaiting owner approval; "
                f"{len(result.proposal.proposed_capabilities)} exact-scope rows proposed, "
                f"public targets {audit.known_count} known / {audit.unknown_count} unknown, "
                "runtime unchanged"
            )
        else:
            validate_gate_b_review_bundle(
                args.output_dir,
                catalog_path=args.catalog,
                category_candidate_dir=args.category_candidate,
                gate_a_candidate_dir=args.gate_a_candidate,
                resolution_candidate_dir=args.resolution_candidate,
                public_set_path=args.public_set,
            )
            print(f"catalog-facet-gate-b-review: valid review packet {args.output_dir}")
    except (
        CatalogSemanticError,
        GateBReviewBundleBusyError,
        OSError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        parser.exit(1, f"catalog-facet-gate-b-review: error: {error}\n")
    return 0


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("catalog", type=Path, help="exact read-only raw catalog JSONL")
    parser.add_argument("category_candidate", type=Path, help="validated CS1 candidate")
    parser.add_argument("gate_a_candidate", type=Path, help="validated Gate-A candidate")
    parser.add_argument("resolution_candidate", type=Path, help="validated CS3 candidate")
    parser.add_argument("public_set", type=Path, help="exact official public JSONL")


if __name__ == "__main__":
    raise SystemExit(main())
