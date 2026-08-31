"""Export the 200 public benchmark product cards as one shareable JSON file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPOSITORY_ROOT / "data" / "public_set.jsonl"
DEFAULT_CARD_BUNDLE = (
    REPOSITORY_ROOT / "data" / "benchmark_product_cards" / "public_200_v1" / "product-facts.jsonl"
)
DEFAULT_MANIFEST = DEFAULT_CARD_BUNDLE.with_name("manifest.json")
DEFAULT_OUTPUT = REPOSITORY_ROOT / "artifacts" / "benchmark" / "product-cards-public-200-v1.json"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        value = json.loads(raw_line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_cards(
    *,
    dataset_path: Path,
    card_bundle_path: Path,
    manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    samples = _load_jsonl(dataset_path)
    cards = _load_jsonl(card_bundle_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if len(samples) != 200:
        raise ValueError(f"Expected 200 public samples, found {len(samples)}")
    if len(cards) != 200:
        raise ValueError(f"Expected 200 product cards, found {len(cards)}")

    sample_ids = [str(sample["sample_id"]) for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Public dataset contains duplicate sample_id values")

    card_by_asin: dict[str, dict[str, Any]] = {}
    for card in cards:
        parent_asin = str(card["parent_asin"])
        if parent_asin in card_by_asin:
            raise ValueError(f"Duplicate product card for parent_asin {parent_asin}")
        card_by_asin[parent_asin] = card

    target_asins = [str(sample["ground_truth"]["parent_asin"]) for sample in samples]
    if len(set(target_asins)) != len(target_asins):
        raise ValueError("Public dataset contains duplicate target products")

    missing = sorted(set(target_asins) - set(card_by_asin))
    extra = sorted(set(card_by_asin) - set(target_asins))
    if missing or extra:
        raise ValueError(f"Product-card coverage mismatch: missing={missing}, extra={extra}")

    entries = []
    for sample, target_asin in zip(samples, target_asins, strict=True):
        entries.append(
            {
                "sample_id": sample["sample_id"],
                "scenario_type": sample["scenario_type"],
                "category_bucket": sample["category_bucket"],
                "difficulty_bucket": sample["difficulty_bucket"],
                "target_parent_asin": target_asin,
                "product_card": card_by_asin[target_asin],
            }
        )

    payload: dict[str, Any] = {
        "schema": "shopping-copilot/public-benchmark-product-card-collection/v1",
        "description": (
            "All 200 grounded product cards used by the public benchmark disclosure "
            "experiment, ordered by public benchmark sample_id."
        ),
        "card_count": len(entries),
        "scope": "known_public_benchmark_target_pool",
        "score_comparability": "diagnostic_only_target_pool_enrichment",
        "warning": manifest["warning"],
        "provenance": {
            "dataset": {
                "path": dataset_path.relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": _sha256(dataset_path),
            },
            "product_card_bundle": {
                "path": card_bundle_path.relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": _sha256(card_bundle_path),
            },
            "bundle_manifest": manifest,
        },
        "cards": entries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cards", type=Path, default=DEFAULT_CARD_BUNDLE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = export_cards(
        dataset_path=args.dataset.resolve(),
        card_bundle_path=args.cards.resolve(),
        manifest_path=args.manifest.resolve(),
        output_path=args.output.resolve(),
    )
    print(f"Exported {payload['card_count']} product cards to {args.output.resolve()}")


if __name__ == "__main__":
    main()
