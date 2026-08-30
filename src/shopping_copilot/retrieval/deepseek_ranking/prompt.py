"""Model-facing prompt for globally calibrated candidate judgement."""

from __future__ import annotations

import hashlib

from shopping_copilot.session_context import Preference

from .models import (
    RANKING_CONTRACT_VERSION,
    DeepSeekRankingRequest,
    RankingCandidateCard,
    canonical_json,
)

SYSTEM_PROMPT = """You judge how well each shopping candidate fits the user's CURRENT resolved intent.

The application has already applied hard retrieval eligibility. Do not choose a final slate, enforce diversity, or infer a target product. Judge every candidate's individual fit, while calibrating scores relative to the complete batch.

Evidence and precedence rules:
1. The current session intent is authoritative.
2. An optional long-term user profile is secondary. If it conflicts with an explicit current-session preference, the current session wins.
3. Use only product evidence shown in the candidate card. Unknown evidence is unsupported, not a confirmed conflict.
   An explicitly incompatible value is a conflict: for example, size 13 conflicts with a current size 10 preference.
4. Do not reward polished wording, retrieval route, candidate order, or imagined product properties.
5. Do not use intent transparency or product-to-product diversity when assigning fit scores.
6. Hard current-session preferences matter more than soft preferences. Every preference ID may appear in at most one judgement array.

Score bands are fixed:
- 75-100: strong_match. The product directly serves the goal and has strong support for important active preferences.
- 40-74: possible_match. It serves the goal but has meaningful unsupported preferences, ambiguity, or a soft conflict.
- 0-39: weak_match. It is peripheral to the goal or conflicts with important current intent.

For every input candidate, call submit_candidate_judgements with exactly one judgement. Preference ID arrays may contain only IDs shown in current_intent.preferences. Keep concerns and reason short and evidence-based."""


def build_messages(
    request: DeepSeekRankingRequest,
    *,
    repair_instruction: str | None = None,
) -> tuple[dict[str, str], ...]:
    if type(request) is not DeepSeekRankingRequest:
        raise TypeError("request must be an exact DeepSeekRankingRequest")
    user_payload = {
        "contract": RANKING_CONTRACT_VERSION,
        "request_id": request.request_id,
        "current_intent": {
            "version": request.intent.version,
            "goal": request.intent.goal,
            "preferences": [_preference_payload(item) for item in request.intent.preferences],
            "dont_care_facets": sorted(request.intent.dont_care_facets),
            "compiled_semantic_query": request.compiled_query.q_sem,
            "compiled_lexical_query": request.compiled_query.q_lex,
        },
        "user_profile": (
            None if request.user_profile is None else request.user_profile.as_payload()
        ),
        "candidates": [
            {
                "candidate_id": card.parent_asin,
                "product_evidence": card.product_text,
            }
            for card in _model_order(request)
        ],
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": canonical_json(user_payload)},
    ]
    if repair_instruction is not None:
        if type(repair_instruction) is not str or not repair_instruction.strip():
            raise ValueError("repair_instruction must be non-empty or None")
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous tool arguments were invalid. Regenerate the complete tool "
                    f"call. Repair requirement: {repair_instruction.strip()}"
                ),
            }
        )
    return tuple(messages)


def _model_order(request: DeepSeekRankingRequest) -> tuple[RankingCandidateCard, ...]:
    return tuple(
        sorted(
            request.shortlist.cards,
            key=lambda card: hashlib.sha256(
                f"{request.request_id}:{card.parent_asin}".encode()
            ).digest(),
        )
    )


def _preference_payload(preference: Preference) -> dict[str, object]:
    value: object = preference.value
    if isinstance(value, tuple):
        value = list(value)
    return {
        "preference_id": preference.id,
        "facet": preference.facet,
        "operator": None if preference.operator is None else preference.operator.value,
        "value": value,
        "semantic_text": preference.semantic_text,
        "semantic_polarity": (
            None
            if preference.semantic_polarity is None
            else preference.semantic_polarity.value
        ),
        "commitment": preference.commitment.value,
        "source": preference.source.value,
        "evidence_text": preference.evidence_text,
    }
