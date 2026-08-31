from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.simulator.evaluate_full_pipeline_other import (
    _apply_scripted_disclosure,
    _load_scripted_conversations,
)


def test_loads_only_user_turns_and_preserves_grounded_fact_payload(tmp_path: Path) -> None:
    path = tmp_path / "conversations.jsonl"
    path.write_text(
        json.dumps(
            {
                "sample_id": "public_0001",
                "transcript": [
                    {
                        "turn": 1,
                        "role": "user",
                        "kind": "initial",
                        "message": "I need a cotton shirt.",
                        "disclosed_fact_ids": ["fact_1"],
                        "disclosed_facts": [
                            {"id": "fact_1", "facet": "material", "value": "cotton"}
                        ],
                        "withdrawn_fact_ids": [],
                    },
                    {
                        "turn": 1,
                        "role": "assistant",
                        "kind": "clarification",
                        "message": "What else?",
                    },
                    {
                        "turn": 2,
                        "role": "user",
                        "kind": "intent_override",
                        "message": "Actually, not cotton.",
                        "disclosed_fact_ids": ["fact_2"],
                        "disclosed_facts": [
                            {"id": "fact_2", "facet": "material", "value": "cotton"}
                        ],
                        "withdrawn_fact_ids": ["fact_1"],
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    turns = _load_scripted_conversations(path)["public_0001"]

    assert [item.turn for item in turns] == [1, 2]
    assert turns[1].kind == "intent_override"
    assert turns[0].disclosed_facts[0]["facet"] == "material"
    disclosed: set[str] = set()
    withdrawn: set[str] = set()
    _apply_scripted_disclosure(turns[0], disclosed=disclosed, withdrawn=withdrawn)
    _apply_scripted_disclosure(turns[1], disclosed=disclosed, withdrawn=withdrawn)
    assert disclosed == {"fact_1", "fact_2"}
    assert withdrawn == {"fact_1"}


def test_rejects_nonconsecutive_user_turns(tmp_path: Path) -> None:
    path = tmp_path / "conversations.jsonl"
    path.write_text(
        json.dumps(
            {
                "sample_id": "public_0001",
                "transcript": [
                    {
                        "turn": 2,
                        "role": "user",
                        "kind": "initial",
                        "message": "I need a shirt.",
                        "disclosed_fact_ids": [],
                        "disclosed_facts": [],
                        "withdrawn_fact_ids": [],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="consecutive user turns"):
        _load_scripted_conversations(path)
