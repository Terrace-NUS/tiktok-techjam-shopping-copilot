"""Capture four visible turns for every buying/browsing task by always asking ``other``."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.query_understanding.generate_simulator_prompts import (  # noqa: E402
    _classify_visible_response,
)
from scripts.query_understanding.suites import (  # noqa: E402
    SIMULATOR_OTHER_SUITE_ID,
    load_prompt_suite,
)

SCHEMA = "shopping-copilot/query-understanding-simulator-suite/v0"
SUITE_VERSION = "v1-other-only"
SOURCE = "official_conversation_simulator"
SCENARIOS = ("buying", "browsing")
VISIBLE_TURNS = 4
ASK_ATTRIBUTE = "other"
ASSISTANT_QUESTION = "What other requirements or preferences matter to you?"

_evaluator = importlib.import_module("evaluator.local_evaluator")


class OtherOnlyCaptureAgent:
    """Record participant-visible messages and always ask the simulator for ``other``."""

    def __init__(self) -> None:
        self._last_assistant_message: str | None = None
        self._last_question: str | None = None
        self.turns: list[dict[str, object]] = []

    def reset(self, session_id: str, user_profile: dict[str, object]) -> None:
        del session_id, user_profile
        self._last_assistant_message = None
        self._last_question = None
        self.turns.clear()

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict[str, object]:
        del session_id, top_k
        self.turns.append(
            {
                "turn": turn,
                "user_message": user_message,
                "last_assistant_message": self._last_assistant_message,
                "last_question": self._last_question,
                "response_shape": _classify_visible_response(user_message),
                "ask_attribute": ASK_ATTRIBUTE,
            }
        )
        self._last_assistant_message = ASSISTANT_QUESTION
        self._last_question = ASSISTANT_QUESTION
        return {
            "message": ASSISTANT_QUESTION,
            "ask_attribute": ASK_ATTRIBUTE,
            "recommendations": [],
        }


def build_suite(*, dataset_path: Path, catalog_path: Path) -> dict[str, object]:
    samples = cast(list[dict[str, Any]], _evaluator.load_jsonl(dataset_path))
    catalog_ids, categories, products = _evaluator.catalog_index(catalog_path)
    counts: Counter[str] = Counter()
    conversations: list[dict[str, object]] = []

    for source_ordinal, sample in enumerate(samples, start=1):
        scenario = sample.get("scenario_type")
        if scenario not in SCENARIOS:
            continue
        sample_id = sample.get("sample_id")
        difficulty = sample.get("difficulty_bucket")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"public sample {source_ordinal} has an invalid sample_id")
        if not isinstance(difficulty, str) or not difficulty:
            raise ValueError(f"public sample {sample_id} has an invalid difficulty_bucket")

        capture = OtherOnlyCaptureAgent()
        _evaluator.evaluate(capture, [sample], catalog_ids, categories, products)
        if len(capture.turns) != _evaluator.MAX_TURNS:
            raise AssertionError(
                f"capture for {sample_id} produced {len(capture.turns)} turns; "
                f"expected {_evaluator.MAX_TURNS}"
            )

        counts[cast(str, scenario)] += 1
        rank = counts[cast(str, scenario)]
        conversations.append(
            {
                "id": f"official_simulator_other_{scenario}_{rank:03d}",
                "tier": "smoke" if rank == 1 else "full",
                "turns": capture.turns[:VISIBLE_TURNS],
                "provenance": {
                    "sample_id": sample_id,
                    "scenario_type": scenario,
                    "difficulty_bucket": difficulty,
                    "source_ordinal": source_ordinal,
                },
            }
        )

    if tuple(counts[scenario] for scenario in SCENARIOS) != (80, 80):
        raise ValueError(f"unexpected public scenario counts: {dict(counts)}")

    evaluator_path = Path(cast(str, _evaluator.__file__)).resolve()
    return {
        "schema": SCHEMA,
        "suite_id": SIMULATOR_OTHER_SUITE_ID,
        "source": SOURCE,
        "description": (
            "Every public buying and browsing task captured from the official conversation simulator; "
            "the agent always returns ask_attribute='other' and stores no evaluator-only state."
        ),
        "generator": {
            "script": "scripts/query_understanding/generate_simulator_other_prompts.py",
            "suite_version": SUITE_VERSION,
            "dataset_sha256": _sha256(dataset_path),
            "catalog_sha256": _sha256(catalog_path),
            "evaluator_sha256": _sha256(evaluator_path),
            "selection_method": "all public buying and browsing samples in source order",
            "selected_per_scenario": 80,
            "visible_turns_per_conversation": VISIBLE_TURNS,
            "base_ask_schedule": [ASK_ATTRIBUTE],
        },
        "conversations": conversations,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _encoded_suite(suite: dict[str, object]) -> bytes:
    return (json.dumps(suite, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("config/query_understanding/simulator-other-prompts-v1.json"),
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    suite = build_suite(dataset_path=args.dataset, catalog_path=args.catalog)
    payload = _encoded_suite(suite)
    if args.check:
        try:
            existing = args.output.read_bytes()
        except FileNotFoundError as error:
            raise SystemExit(f"simulator prompt suite does not exist: {args.output}") from error
        if existing != payload:
            raise SystemExit(f"simulator prompt suite is stale: {args.output}")
        print(f"simulator prompt suite is current: {args.output} ({len(payload)} bytes)")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    loaded = load_prompt_suite(args.output)
    turn_count = sum(len(item.turns) for item in loaded.conversations)
    print(
        f"wrote {len(payload)} bytes to {args.output}: "
        f"{len(loaded.conversations)} conversations / {turn_count} turns"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
