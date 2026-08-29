"""Command-line entry point for CS6 release assembly and verification."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ..errors import CatalogSemanticError, ReleaseBundleBusyError
from .bundle import validate_catalog_semantic_release, write_catalog_semantic_release


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="catalog-semantic-release")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="assemble one immutable 13-artifact release")
    _add_build_inputs(build)
    build.add_argument("output_dir", type=Path, help="new immutable release directory")
    validate = commands.add_parser("validate", help="verify a self-contained release")
    validate.add_argument("release_dir", type=Path)
    validate.add_argument("--release-id", dest="release_id")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            release = write_catalog_semantic_release(
                args.catalog,
                args.category_candidate,
                args.gate_a_candidate,
                args.resolution_candidate,
                args.public_set,
                args.gate_b_review,
                args.gate_b_selection,
                args.gate_b_candidate,
                args.runtime_projection,
                args.output_dir,
            )
            print(
                "catalog-semantic-release: published "
                f"{release.release_id} with {len(release.manifest.artifacts)} artifacts"
            )
        else:
            release_id = validate_catalog_semantic_release(
                args.release_dir,
                expected_release_id=args.release_id,
            )
            print(f"catalog-semantic-release: valid {release_id}")
    except (
        CatalogSemanticError,
        ReleaseBundleBusyError,
        OSError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        parser.exit(1, f"catalog-semantic-release: error: {error}\n")
    return 0


def _add_build_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("catalog", type=Path, help="exact frozen raw catalog JSONL")
    parser.add_argument("category_candidate", type=Path)
    parser.add_argument("gate_a_candidate", type=Path)
    parser.add_argument("resolution_candidate", type=Path)
    parser.add_argument("public_set", type=Path)
    parser.add_argument("gate_b_review", type=Path)
    parser.add_argument("gate_b_selection", type=Path)
    parser.add_argument("gate_b_candidate", type=Path)
    parser.add_argument("runtime_projection", type=Path)


if __name__ == "__main__":
    raise SystemExit(main())
