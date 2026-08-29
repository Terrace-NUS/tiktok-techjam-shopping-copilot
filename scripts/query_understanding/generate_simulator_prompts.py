"""Generate a target-isolated Query Understanding suite from the official simulator.

The generator deliberately delegates all customer-message construction to
``evaluator.local_evaluator.evaluate``.  Its capture agent sees exactly the
same ``user_message`` values as a participant Agent and never stores reset
profiles, recommendations, target identifiers, intent cards, or catalog rows.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

SCHEMA = "shopping-copilot/query-understanding-simulator-suite/v0"
SUITE_ID = "official-simulator-prompts-v0"
SUITE_VERSION = "v0"
SOURCE = "official_toy_simulator"
SCENARIOS = ("buying", "browsing", "intent_override", "boundary")
SELECTED_PER_SCENARIO = 8
VISIBLE_TURNS = 4
BASE_ASK_SCHEDULE: tuple[str | None, ...] = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
    None,
)

TOP_LEVEL_KEYS = {
    "schema",
    "suite_id",
    "source",
    "description",
    "generator",
    "conversations",
}
GENERATOR_KEYS = {
    "script",
    "suite_version",
    "dataset_sha256",
    "catalog_sha256",
    "evaluator_sha256",
    "selection_method",
    "selected_per_scenario",
    "visible_turns_per_conversation",
    "base_ask_schedule",
}
CONVERSATION_KEYS = {"id", "tier", "turns", "provenance"}
PROVENANCE_KEYS = {"sample_id", "scenario_type", "difficulty_bucket", "source_ordinal"}
TURN_KEYS = {
    "turn",
    "user_message",
    "last_assistant_message",
    "last_question",
    "response_shape",
    "ask_attribute",
}
RESPONSE_SHAPES = {
    "initial_requirement",
    "initial_exploration",
    "initial_preference",
    "attribute_disclosure",
    "explicit_no_preference",
    "no_additional_preference",
    "negative_feedback",
    "explicit_override",
}

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
_evaluator = importlib.import_module("evaluator.local_evaluator")


class CaptureAgent:
    """A deterministic Agent that records only participant-visible turn data."""

    def __init__(self, *, sample_id: str) -> None:
        self._ask_schedule = _rotated_ask_schedule(sample_id)
        self._last_assistant_message: str | None = None
        self._last_question: str | None = None
        self.turns: list[dict[str, object]] = []

    def reset(self, session_id: str, user_profile: dict[str, object]) -> None:
        # The official API supplies both fields, but this fixture intentionally
        # persists neither of them.  They cannot influence the fixed ask policy.
        del session_id, user_profile
        self._last_assistant_message = None
        self._last_question = None
        self.turns.clear()

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict[str, object]:
        del session_id, top_k
        ask_attribute = self._ask_schedule[(turn - 1) % len(self._ask_schedule)]
        self.turns.append(
            {
                "turn": turn,
                "user_message": user_message,
                "last_assistant_message": self._last_assistant_message,
                "last_question": self._last_question,
                "response_shape": _classify_visible_response(user_message),
                "ask_attribute": ask_attribute,
            }
        )
        if ask_attribute is None:
            message = "I'll keep refining the options."
            question = None
        else:
            message = f"What is your preference for {ask_attribute}?"
            question = message
        self._last_assistant_message = message
        self._last_question = question
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": [],
        }


def _rotated_ask_schedule(sample_id: str) -> tuple[str | None, ...]:
    digest = hashlib.sha256(f"{sample_id}{SUITE_ID}:ask-schedule".encode()).digest()
    offset = int.from_bytes(digest[:8], "big") % len(BASE_ASK_SCHEDULE)
    return BASE_ASK_SCHEDULE[offset:] + BASE_ASK_SCHEDULE[:offset]


def _selection_digest(sample_id: str) -> str:
    return hashlib.sha256(f"{sample_id}{SUITE_VERSION}".encode()).hexdigest()


def _classify_visible_response(message: str) -> str:
    """Classify a turn using its visible text and no simulator metadata."""

    if message.startswith("Actually, ignore my earlier preference. What I need is:"):
        return "explicit_override"
    if message.startswith("For that, what matters is:"):
        return "attribute_disclosure"
    if message.startswith("I don't have a preference for "):
        return "explicit_no_preference"
    if message.startswith("I don't have an additional preference for "):
        return "no_additional_preference"
    if message == "Those options are not quite right yet. Ask me about one specific attribute.":
        return "negative_feedback"
    if message.startswith("I'm looking for ") and ". A key requirement is:" in message:
        return "initial_requirement"
    if message.startswith("I'm looking for ") and message.endswith(", but I'm still exploring."):
        return "initial_exploration"
    if message.startswith("I'm looking for "):
        return "initial_preference"
    raise ValueError(f"unrecognized official simulator message shape: {message!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_samples(samples: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    buckets: dict[str, list[tuple[str, int, dict[str, Any]]]] = defaultdict(list)
    seen_sample_ids: set[str] = set()
    for ordinal, sample in enumerate(samples, start=1):
        sample_id = sample.get("sample_id")
        scenario = sample.get("scenario_type")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"public sample {ordinal} has an invalid sample_id")
        if sample_id in seen_sample_ids:
            raise ValueError(f"duplicate public sample_id: {sample_id}")
        seen_sample_ids.add(sample_id)
        if scenario not in SCENARIOS:
            raise ValueError(f"public sample {sample_id} has an unexpected scenario: {scenario!r}")
        buckets[cast(str, scenario)].append((_selection_digest(sample_id), ordinal, sample))

    selected: list[tuple[int, dict[str, Any]]] = []
    for scenario in SCENARIOS:
        candidates = sorted(buckets[scenario], key=lambda item: (item[0], item[1]))
        if len(candidates) < SELECTED_PER_SCENARIO:
            raise ValueError(
                f"scenario {scenario!r} has {len(candidates)} samples; need {SELECTED_PER_SCENARIO}"
            )
        selected.extend(
            (ordinal, sample) for _, ordinal, sample in candidates[:SELECTED_PER_SCENARIO]
        )
    return selected


def build_suite(*, dataset_path: Path, catalog_path: Path) -> dict[str, object]:
    samples = cast(list[dict[str, Any]], _evaluator.load_jsonl(dataset_path))
    catalog_ids, categories, products = _evaluator.catalog_index(catalog_path)
    selected = _select_samples(samples)
    per_scenario_rank: Counter[str] = Counter()
    conversations: list[dict[str, object]] = []

    for source_ordinal, sample in selected:
        sample_id = cast(str, sample["sample_id"])
        scenario = cast(str, sample["scenario_type"])
        difficulty = sample.get("difficulty_bucket")
        if not isinstance(difficulty, str) or not difficulty:
            raise ValueError(f"public sample {sample_id} has an invalid difficulty_bucket")

        capture = CaptureAgent(sample_id=sample_id)
        _evaluator.evaluate(capture, [sample], catalog_ids, categories, products)
        if len(capture.turns) != _evaluator.MAX_TURNS:
            raise AssertionError(
                f"capture for {sample_id} produced {len(capture.turns)} turns; "
                f"expected {_evaluator.MAX_TURNS}"
            )

        rank = per_scenario_rank[scenario]
        per_scenario_rank[scenario] += 1
        conversations.append(
            {
                "id": f"official_simulator_{scenario}_{rank + 1:02d}",
                "tier": "smoke" if rank == 0 else "full",
                "turns": capture.turns[:VISIBLE_TURNS],
                "provenance": {
                    "sample_id": sample_id,
                    "scenario_type": scenario,
                    "difficulty_bucket": difficulty,
                    "source_ordinal": source_ordinal,
                },
            }
        )

    evaluator_path = Path(cast(str, _evaluator.__file__)).resolve()
    suite: dict[str, object] = {
        "schema": SCHEMA,
        "suite_id": SUITE_ID,
        "source": SOURCE,
        "description": (
            "Participant-visible user messages captured by running the official toy simulator; "
            "all evaluator-only state is excluded."
        ),
        "generator": {
            "script": "scripts/query_understanding/generate_simulator_prompts.py",
            "suite_version": SUITE_VERSION,
            "dataset_sha256": _sha256(dataset_path),
            "catalog_sha256": _sha256(catalog_path),
            "evaluator_sha256": _sha256(evaluator_path),
            "selection_method": "lowest sha256(sample_id + suite_version), eight per scenario",
            "selected_per_scenario": SELECTED_PER_SCENARIO,
            "visible_turns_per_conversation": VISIBLE_TURNS,
            "base_ask_schedule": list(BASE_ASK_SCHEDULE),
        },
        "conversations": conversations,
    }
    _validate_safe_suite(suite)
    return suite


def _validate_safe_suite(suite: dict[str, object]) -> None:
    if set(suite) != TOP_LEVEL_KEYS:
        raise AssertionError("simulator suite has unexpected top-level fields")
    if suite["schema"] != SCHEMA or suite["suite_id"] != SUITE_ID or suite["source"] != SOURCE:
        raise AssertionError("simulator suite identity is invalid")
    generator = suite["generator"]
    if not isinstance(generator, dict) or set(generator) != GENERATOR_KEYS:
        raise AssertionError("simulator suite generator fields are invalid")
    conversations = suite["conversations"]
    if (
        not isinstance(conversations, list)
        or len(conversations) != len(SCENARIOS) * SELECTED_PER_SCENARIO
    ):
        raise AssertionError("simulator suite conversation count is invalid")

    scenario_counts: Counter[str] = Counter()
    smoke_counts: Counter[str] = Counter()
    conversation_ids: set[str] = set()
    sample_ids: set[str] = set()
    allowed_attributes = set(cast(set[str], _evaluator.ALLOWED_ATTRIBUTES))
    if allowed_attributes != {item for item in BASE_ASK_SCHEDULE if item is not None}:
        raise AssertionError("base ask schedule differs from the official allowed attributes")

    for conversation in conversations:
        if not isinstance(conversation, dict) or set(conversation) != CONVERSATION_KEYS:
            raise AssertionError("simulator conversation fields are invalid")
        conversation_id = conversation["id"]
        tier = conversation["tier"]
        provenance = conversation["provenance"]
        turns = conversation["turns"]
        if not isinstance(conversation_id, str) or conversation_id in conversation_ids:
            raise AssertionError("simulator conversation id is invalid or duplicated")
        conversation_ids.add(conversation_id)
        if tier not in {"smoke", "full"}:
            raise AssertionError("simulator conversation tier is invalid")
        if not isinstance(provenance, dict) or set(provenance) != PROVENANCE_KEYS:
            raise AssertionError("simulator provenance fields are invalid")
        sample_id = provenance["sample_id"]
        scenario = provenance["scenario_type"]
        if not isinstance(sample_id, str) or sample_id in sample_ids:
            raise AssertionError("simulator provenance sample_id is invalid or duplicated")
        sample_ids.add(sample_id)
        if scenario not in SCENARIOS:
            raise AssertionError("simulator provenance scenario is invalid")
        scenario_name = cast(str, scenario)
        scenario_counts[scenario_name] += 1
        smoke_counts[scenario_name] += int(tier == "smoke")
        if not isinstance(provenance["difficulty_bucket"], str):
            raise AssertionError("simulator difficulty bucket is invalid")
        if not isinstance(provenance["source_ordinal"], int) or provenance["source_ordinal"] < 1:
            raise AssertionError("simulator source ordinal is invalid")
        if not isinstance(turns, list) or len(turns) != VISIBLE_TURNS:
            raise AssertionError("simulator visible turn count is invalid")
        for expected_turn, turn in enumerate(turns, start=1):
            if not isinstance(turn, dict) or set(turn) != TURN_KEYS:
                raise AssertionError("simulator turn fields are invalid")
            if turn["turn"] != expected_turn:
                raise AssertionError("simulator turns are not consecutive")
            if not isinstance(turn["user_message"], str) or not turn["user_message"]:
                raise AssertionError("simulator user message is invalid")
            for nullable_field in ("last_assistant_message", "last_question"):
                if turn[nullable_field] is not None and not isinstance(turn[nullable_field], str):
                    raise AssertionError(f"simulator {nullable_field} is invalid")
            if turn["response_shape"] not in RESPONSE_SHAPES:
                raise AssertionError("simulator response shape is invalid")
            if (
                turn["ask_attribute"] is not None
                and turn["ask_attribute"] not in allowed_attributes
            ):
                raise AssertionError("simulator ask_attribute is invalid")

    expected_counts = Counter({scenario: SELECTED_PER_SCENARIO for scenario in SCENARIOS})
    expected_smoke = Counter({scenario: 1 for scenario in SCENARIOS})
    if scenario_counts != expected_counts or smoke_counts != expected_smoke:
        raise AssertionError("simulator scenario or smoke-tier balance is invalid")


def _encoded_suite(suite: dict[str, object]) -> bytes:
    return (json.dumps(suite, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate target-isolated QU prompts by running the official toy simulator."
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("config/query_understanding/simulator-prompts-v0.json"),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail unless the existing output is byte-identical to a fresh generation",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = _encoded_suite(build_suite(dataset_path=args.dataset, catalog_path=args.catalog))
    if args.check:
        try:
            existing = args.output.read_bytes()
        except FileNotFoundError as exc:
            raise SystemExit(f"simulator prompt suite does not exist: {args.output}") from exc
        if existing != payload:
            raise SystemExit(f"simulator prompt suite is stale: {args.output}")
        print(f"simulator prompt suite is current: {args.output} ({len(payload)} bytes)")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(f"wrote {len(payload)} bytes to {args.output}")


if __name__ == "__main__":
    main()
