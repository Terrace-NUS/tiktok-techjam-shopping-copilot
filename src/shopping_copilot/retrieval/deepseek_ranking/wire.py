"""Native tool schema and exact-batch decoder for DeepSeek ranking."""

from __future__ import annotations

import json
from typing import cast

from .errors import DeepSeekRankingError, DeepSeekRankingErrorCode
from .models import CandidateJudgement, CandidateVerdict, DeepSeekRankingRequest

TOOL_NAME = "submit_candidate_judgements"


def candidate_judgement_tool(*, strict: bool = False) -> dict[str, object]:
    tool: dict[str, object] = {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": (
                "Submit one evidence-based individual-fit judgement for every candidate."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "judgements": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "candidate_id": {"type": "string"},
                                "fit_score": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 100,
                                },
                                "verdict": {
                                    "type": "string",
                                    "enum": [item.value for item in CandidateVerdict],
                                },
                                "matched_preference_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "unsupported_preference_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "conflict_preference_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "concerns": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "reason": {"type": "string"},
                            },
                            "required": [
                                "candidate_id",
                                "fit_score",
                                "verdict",
                                "matched_preference_ids",
                                "unsupported_preference_ids",
                                "conflict_preference_ids",
                                "concerns",
                                "reason",
                            ],
                        },
                    }
                },
                "required": ["judgements"],
            },
        },
    }
    if strict:
        cast(dict[str, object], tool["function"])["strict"] = True
    return tool


def decode_candidate_judgements(
    arguments: str,
    request: DeepSeekRankingRequest,
) -> tuple[CandidateJudgement, ...]:
    if type(arguments) is not str:
        raise TypeError("arguments must be a string")
    if type(request) is not DeepSeekRankingRequest:
        raise TypeError("request must be an exact DeepSeekRankingRequest")
    try:
        decoded: object = json.loads(
            arguments,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise DeepSeekRankingError(
            DeepSeekRankingErrorCode.INVALID_JUDGEMENTS,
            "tool arguments are not valid unique-key JSON",
        ) from error
    if type(decoded) is not dict or set(decoded) != {"judgements"}:
        raise DeepSeekRankingError(
            DeepSeekRankingErrorCode.INVALID_JUDGEMENTS,
            "tool arguments must contain only judgements",
        )
    raw_items = cast(dict[str, object], decoded)["judgements"]
    if type(raw_items) is not list:
        raise DeepSeekRankingError(
            DeepSeekRankingErrorCode.INVALID_JUDGEMENTS,
            "judgements must be an array",
        )

    candidate_ids = tuple(card.parent_asin for card in request.shortlist.cards)
    allowed_candidates = set(candidate_ids)
    allowed_preferences = {item.id for item in request.intent.preferences}
    by_asin: dict[str, CandidateJudgement] = {}
    try:
        for raw in raw_items:
            if type(raw) is not dict:
                raise ValueError("judgement must be an object")
            item = cast(dict[str, object], raw)
            required = {
                "candidate_id",
                "fit_score",
                "verdict",
                "matched_preference_ids",
                "unsupported_preference_ids",
                "conflict_preference_ids",
                "concerns",
                "reason",
            }
            if set(item) != required:
                raise ValueError("judgement fields differ from contract")
            parent_asin = _text(item["candidate_id"], name="candidate_id")
            if parent_asin not in allowed_candidates:
                raise ValueError(f"unknown candidate_id: {parent_asin}")
            if parent_asin in by_asin:
                raise ValueError(f"duplicate candidate_id: {parent_asin}")
            matched = _texts(item["matched_preference_ids"], name="matched_preference_ids")
            unsupported = _texts(
                item["unsupported_preference_ids"],
                name="unsupported_preference_ids",
            )
            conflicts = _texts(
                item["conflict_preference_ids"],
                name="conflict_preference_ids",
            )
            observed_preferences = set((*matched, *unsupported, *conflicts))
            if not observed_preferences <= allowed_preferences:
                unknown = min(observed_preferences - allowed_preferences)
                raise ValueError(f"unknown preference_id: {unknown}")
            fit_score = item["fit_score"]
            if type(fit_score) is not int:
                raise ValueError("fit_score must be an integer")
            verdict_value = _text(item["verdict"], name="verdict")
            try:
                verdict = CandidateVerdict(verdict_value)
            except ValueError as error:
                raise ValueError("verdict is invalid") from error
            by_asin[parent_asin] = CandidateJudgement(
                parent_asin=parent_asin,
                fit_score=fit_score,
                verdict=verdict,
                matched_preference_ids=matched,
                unsupported_preference_ids=unsupported,
                conflict_preference_ids=conflicts,
                concerns=_texts(item["concerns"], name="concerns"),
                reason=_text(item["reason"], name="reason"),
            )
    except (TypeError, ValueError) as error:
        raise DeepSeekRankingError(
            DeepSeekRankingErrorCode.INVALID_JUDGEMENTS,
            str(error),
        ) from error
    if set(by_asin) != allowed_candidates:
        missing = sorted(allowed_candidates - set(by_asin))
        raise DeepSeekRankingError(
            DeepSeekRankingErrorCode.INVALID_JUDGEMENTS,
            f"missing candidate judgements: {missing[:3]}",
        )
    return tuple(by_asin[parent_asin] for parent_asin in candidate_ids)


def _texts(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise ValueError(f"{name} must be an array")
    result = tuple(_text(item, name=f"{name}[]") for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")
