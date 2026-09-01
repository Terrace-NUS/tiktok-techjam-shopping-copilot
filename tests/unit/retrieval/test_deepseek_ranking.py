from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pytest

from shopping_copilot.providers import HttpResponse
from shopping_copilot.query_compiler import (
    COMPILED_QUERY_SCHEMA,
    QUERY_COMPILER_VERSION,
    CompiledDirectives,
    CompiledQuery,
    DiversityDirective,
)
from shopping_copilot.retrieval import (
    CrossEncoderRelevanceReranker,
    DenseIndex,
    DenseIndexManifest,
    EmbeddingSpec,
    VectorCandidate,
)
from shopping_copilot.retrieval.deepseek_ranking import (
    TOOL_NAME,
    CandidateJudgement,
    CandidateVerdict,
    DeepSeekJudgementResult,
    DeepSeekQualityPipeline,
    DeepSeekQualityRanker,
    DeepSeekRankingConfig,
    DeepSeekRankingError,
    DeepSeekRankingErrorCode,
    DeepSeekRankingProvider,
    DeepSeekRankingRequest,
    DeepSeekRankingTrace,
    DirectionAwareShortlister,
    QualityRankingMode,
    RankingCandidateCard,
    RankingShortlist,
    RankingUserProfile,
    TransparencyAwareDPPFinalizer,
    candidate_judgement_tool,
    decode_candidate_judgements,
)
from shopping_copilot.retrieval.deepseek_ranking.prompt import build_messages
from shopping_copilot.retrieval.models import DENSE_INDEX_SCHEMA, DenseArtifactRef
from shopping_copilot.retrieval.ranking import (
    CrossEncoderRankingHit,
    CrossEncoderRankingResult,
)
from shopping_copilot.retrieval.transparency_recall import (
    TRANSPARENCY_RECALL_POLICY_ID,
    MultiCenterRecallTimings,
    RecallBudgets,
    RecallDirection,
    TransparencyRecallTrace,
)
from shopping_copilot.session_context import (
    Commitment,
    IntentState,
    Operator,
    Preference,
    PreferenceSource,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _RecordedCall:
    url: str
    headers: dict[str, str]
    body: bytes
    timeout_seconds: float


class _RecordingTransport:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[_RecordedCall] = []

    def post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
    ) -> HttpResponse:
        self.calls.append(
            _RecordedCall(
                url=url,
                headers=dict(headers),
                body=body,
                timeout_seconds=timeout_seconds,
            )
        )
        return self.responses.pop(0)


class _SequenceJudge:
    def __init__(self, *outcomes: DeepSeekJudgementResult | DeepSeekRankingError) -> None:
        self.outcomes = list(outcomes)
        self.repair_instructions: list[str | None] = []

    def judge(
        self,
        request: DeepSeekRankingRequest,
        *,
        repair_instruction: str | None = None,
    ) -> DeepSeekJudgementResult:
        self.repair_instructions.append(repair_instruction)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, DeepSeekRankingError):
            raise outcome
        return outcome


class _FakeBGE:
    @property
    def model_id(self) -> str:
        return "fake/bge"

    def score(
        self,
        query: str,
        documents: Sequence[str],
        *,
        batch_size: int,
    ) -> tuple[float, ...]:
        assert query == "Winter boots in black."
        return tuple(float("Black" in item) for item in documents)


def _preference() -> Preference:
    return Preference(
        id="p_color",
        facet="color",
        operator=Operator.EQ,
        value="black",
        semantic_text=None,
        semantic_polarity=None,
        commitment=Commitment.HARD,
        source=PreferenceSource.USER_EXPLICIT,
        source_turn=1,
        evidence_text="make it black",
        interpretation_confidence=0.99,
    )


def _intent() -> IntentState:
    return IntentState(
        goal="Find winter boots",
        preferences=(_preference(),),
        dont_care_facets=frozenset(),
        version=1,
    )


def _compiled_query() -> CompiledQuery:
    return CompiledQuery(
        schema=COMPILED_QUERY_SCHEMA,
        compiler_version=QUERY_COMPILER_VERSION,
        catalog_id="sha256:" + "1" * 64,
        catalog_semantic_release_id="sha256:" + "2" * 64,
        category_graph_id="sha256:" + "3" * 64,
        intent_version=1,
        q_lex="black winter boots",
        q_sem="Winter boots in black.",
        search_ready=True,
        hard_constraints=(),
        ranking_preferences=(),
        dont_care_facets=(),
        directives=CompiledDirectives(
            diversity=DiversityDirective.AUTO,
            comparison_requested=False,
            explanation_requested=False,
        ),
        requires_clarification=False,
        clarification_reason=None,
        trace=(),
    )


def _card(
    parent_asin: str,
    rank: int,
    bge_relevance: float,
    *,
    text: str,
) -> RankingCandidateCard:
    return RankingCandidateCard(
        parent_asin=parent_asin,
        shortlist_rank=rank,
        original_candidate_rank=rank,
        bge_relevance=bge_relevance,
        normalized_bge_score=bge_relevance,
        direction_id=None,
        routes=("dense",),
        product_text=text,
    )


def _request(*, user_profile: RankingUserProfile | None = None) -> DeepSeekRankingRequest:
    return DeepSeekRankingRequest(
        request_id="ranking-test-1",
        intent=_intent(),
        compiled_query=_compiled_query(),
        shortlist=RankingShortlist(
            model_id="fake/bge",
            requested_top_k=2,
            protected_per_direction=0,
            cards=(
                _card("A", 1, 0.9, text="title: Black insulated winter boot"),
                _card("B", 2, 0.8, text="title: Red summer sandal"),
            ),
        ),
        user_profile=user_profile,
    )


def _judgement(
    parent_asin: str,
    score: int,
    verdict: CandidateVerdict,
) -> CandidateJudgement:
    matched = ("p_color",) if parent_asin == "A" else ()
    conflicts = ("p_color",) if parent_asin == "B" else ()
    return CandidateJudgement(
        parent_asin=parent_asin,
        fit_score=score,
        verdict=verdict,
        matched_preference_ids=matched,
        unsupported_preference_ids=(),
        conflict_preference_ids=conflicts,
        concerns=(),
        reason="Evidence-based unit-test judgement.",
    )


def _arguments() -> str:
    return json.dumps(
        {
            "judgements": [
                {
                    "candidate_id": "B",
                    "fit_score": 20,
                    "verdict": "weak_match",
                    "matched_preference_ids": [],
                    "unsupported_preference_ids": [],
                    "conflict_preference_ids": ["p_color"],
                    "concerns": ["Wrong season and color."],
                    "reason": "It conflicts with the current request.",
                },
                {
                    "candidate_id": "A",
                    "fit_score": 95,
                    "verdict": "strong_match",
                    "matched_preference_ids": ["p_color"],
                    "unsupported_preference_ids": [],
                    "conflict_preference_ids": [],
                    "concerns": [],
                    "reason": "Directly supported by the product evidence.",
                },
            ]
        },
        separators=(",", ":"),
    )


def _chat_response(*, arguments: str | None = None) -> HttpResponse:
    return HttpResponse(
        status=200,
        body=json.dumps(
            {
                "id": "ranking-response-1",
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": TOOL_NAME,
                                        "arguments": arguments or _arguments(),
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 600,
                    "completion_tokens": 200,
                    "total_tokens": 800,
                },
            },
            separators=(",", ":"),
        ).encode(),
    )


def test_prompt_gives_current_session_priority_and_hides_ranking_anchors() -> None:
    profile = RankingUserProfile(
        schema="shopping-copilot/user-profile/v0",
        version=1,
        payload={"favorite_colors": ["red"]},
    )

    messages = build_messages(_request(user_profile=profile))
    system = messages[0]["content"]
    payload = json.loads(messages[1]["content"])

    assert "current session intent is authoritative" in system.casefold()
    assert payload["user_profile"]["payload"] == {"favorite_colors": ["red"]}
    assert {item["candidate_id"] for item in payload["candidates"]} == {"A", "B"}
    serialized_candidates = json.dumps(payload["candidates"])
    assert "bge" not in serialized_candidates.casefold()
    assert "direction" not in serialized_candidates.casefold()
    with pytest.raises(TypeError):
        profile.payload["new"] = "value"  # type: ignore[index]


def test_decoder_requires_exact_batch_and_restores_shortlist_order() -> None:
    judgements = decode_candidate_judgements(_arguments(), _request())
    assert [item.parent_asin for item in judgements] == ["A", "B"]

    missing = json.loads(_arguments())
    missing["judgements"].pop()
    with pytest.raises(DeepSeekRankingError) as caught:
        decode_candidate_judgements(json.dumps(missing), _request())
    assert caught.value.code is DeepSeekRankingErrorCode.INVALID_JUDGEMENTS


def test_provider_forces_native_tool_and_decodes_usage() -> None:
    transport = _RecordingTransport(_chat_response())
    provider = DeepSeekRankingProvider(
        api_key="unit-test-secret",
        config=DeepSeekRankingConfig(timeout_seconds=7.5, max_tokens=777),
        transport=transport,
    )

    result = provider.judge(_request())

    payload = json.loads(transport.calls[0].body)
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["tool_choice"] == {
        "type": "function",
        "function": {"name": TOOL_NAME},
    }
    assert payload["tools"] == [candidate_judgement_tool(strict=False)]
    assert payload["max_tokens"] == 777
    assert result.trace.total_tokens == 800
    assert [item.parent_asin for item in result.judgements] == ["A", "B"]


def test_quality_ranker_repairs_once_then_blends_80_20() -> None:
    successful = DeepSeekJudgementResult(
        judgements=(
            _judgement("A", 95, CandidateVerdict.STRONG_MATCH),
            _judgement("B", 20, CandidateVerdict.WEAK_MATCH),
        ),
        trace=DeepSeekRankingTrace(
            response_id="ok",
            model="deepseek-v4-flash",
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
        ),
    )
    judge = _SequenceJudge(
        DeepSeekRankingError(DeepSeekRankingErrorCode.INVALID_JUDGEMENTS),
        successful,
    )

    result = DeepSeekQualityRanker(provider=judge).rank(_request())

    assert result.mode is QualityRankingMode.DEEPSEEK
    assert result.attempts == 2
    assert judge.repair_instructions[0] is None
    assert judge.repair_instructions[1]
    assert [item.parent_asin for item in result.hits] == ["A", "B"]
    assert result.hits[0].quality == pytest.approx(0.8 * 0.95 + 0.2 * 0.9)


def test_quality_ranker_falls_back_to_bge_on_provider_failure() -> None:
    judge = _SequenceJudge(DeepSeekRankingError(DeepSeekRankingErrorCode.PROVIDER_RATE_LIMIT))

    result = DeepSeekQualityRanker(provider=judge).rank(_request())

    assert result.mode is QualityRankingMode.BGE_FALLBACK
    assert result.attempts == 1
    assert result.fallback_reason == "provider_rate_limit"
    assert [item.parent_asin for item in result.hits] == ["A", "B"]
    assert all(item.deepseek_fit is None for item in result.hits)


def test_pipeline_connects_bge_shortlist_to_deepseek_quality() -> None:
    successful = DeepSeekJudgementResult(
        judgements=(
            _judgement("A", 95, CandidateVerdict.STRONG_MATCH),
            _judgement("B", 20, CandidateVerdict.WEAK_MATCH),
        ),
        trace=DeepSeekRankingTrace(
            response_id="ok",
            model="deepseek-v4-flash",
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
        ),
    )
    pipeline = DeepSeekQualityPipeline(
        index=_index(),
        bge_reranker=CrossEncoderRelevanceReranker(scorer=_FakeBGE()),
        deepseek_ranker=DeepSeekQualityRanker(
            provider=_SequenceJudge(successful),
        ),
        shortlist_k=2,
        protected_per_direction=0,
    )

    result = pipeline.rank(
        request_id="pipeline-test",
        intent=_intent(),
        compiled_query=_compiled_query(),
        candidates=(
            VectorCandidate(parent_asin="A", candidate_rank=1, relevance=1.0),
            VectorCandidate(parent_asin="B", candidate_rank=2, relevance=0.8),
        ),
        documents={
            "A": "title: Black winter boot",
            "B": "title: Red summer sandal",
        },
        recall_trace=None,
    )

    assert [item.parent_asin for item in result.shortlist.cards] == ["A", "B"]
    assert result.quality_ranking.mode is QualityRankingMode.DEEPSEEK
    assert result.quality_ranking.hits[0].parent_asin == "A"


def test_final_dpp_maps_transparency_only_after_quality_ranking() -> None:
    successful = DeepSeekJudgementResult(
        judgements=(
            _judgement("A", 95, CandidateVerdict.STRONG_MATCH),
            _judgement("B", 20, CandidateVerdict.WEAK_MATCH),
        ),
        trace=DeepSeekRankingTrace(
            response_id="ok",
            model="deepseek-v4-flash",
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
        ),
    )
    quality = DeepSeekQualityRanker(provider=_SequenceJudge(successful)).rank(_request())
    finalizer = TransparencyAwareDPPFinalizer(index=_index())

    broad = finalizer.select(quality, transparency=0.0, top_k=2)
    focused = finalizer.select(quality, transparency=1.0, top_k=2)

    assert broad.relevance_weight == pytest.approx(0.30)
    assert focused.relevance_weight == pytest.approx(0.90)
    assert len(broad.result.hits) == len(focused.result.hits) == 2


def test_shortlist_protects_each_semantic_direction() -> None:
    index = _index()
    ranking = CrossEncoderRankingResult(
        model_id="fake/bge",
        prior_weight=0.0,
        hits=tuple(
            CrossEncoderRankingHit(
                parent_asin=parent_asin,
                rank=rank,
                candidate_rank=rank,
                raw_model_score=float(7 - rank),
                normalized_model_score=float(7 - rank) / 6.0,
                prior_relevance=float(7 - rank) / 6.0,
                relevance=float(7 - rank) / 6.0,
            )
            for rank, parent_asin in enumerate(("A", "B", "C", "D", "E", "F"), start=1)
        ),
    )
    trace = TransparencyRecallTrace(
        policy_id=TRANSPARENCY_RECALL_POLICY_ID,
        transparency=0.2,
        candidate_pool_k=6,
        candidate_pool_count=6,
        requested_direction_count=2,
        actual_direction_count=2,
        frontier_requested_k=6,
        frontier_count=6,
        budgets=RecallBudgets(dense=2, lexical=2, facet=2),
        actual_dense_count=2,
        actual_lexical_count=2,
        actual_facet_count=2,
        dense_refill_count=0,
        directions=(
            RecallDirection(
                direction_id="direction_0",
                center_parent_asin="A",
                query_similarity=0.9,
                maximum_similarity_to_previous_centers=None,
            ),
            RecallDirection(
                direction_id="direction_1",
                center_parent_asin="D",
                query_similarity=0.8,
                maximum_similarity_to_previous_centers=0.0,
            ),
        ),
        dense_candidates=(),
        planner_timings=MultiCenterRecallTimings(
            frontier_ms=0.0,
            center_selection_ms=0.0,
            direction_expansion_ms=0.0,
            total_ms=0.0,
        ),
    )
    documents = {item: f"title: Product {item}" for item in index.parent_asins}

    shortlist = DirectionAwareShortlister(
        index=index,
        top_k=3,
        protected_per_direction=1,
    ).select(ranking, documents=documents, recall_trace=trace)

    assert [item.parent_asin for item in shortlist.cards] == ["A", "B", "D"]
    assert {item.direction_id for item in shortlist.cards} == {
        "direction_0",
        "direction_1",
    }


def _index() -> DenseIndex:
    spec = EmbeddingSpec(
        backend="fake",
        backend_version="1.0",
        model_id="example/model",
        model_revision="revision",
        dimension=3,
        max_sequence_length=32,
        query_instruction="",
        document_instruction="",
        pooling="cls",
    )
    manifest = DenseIndexManifest(
        schema=DENSE_INDEX_SCHEMA,
        builder_version="dense_index_v0",
        catalog_id="sha256:" + "1" * 64,
        catalog_semantic_release_id="sha256:" + "2" * 64,
        document_template_id="product_document_v1",
        document_corpus_id="sha256:" + "3" * 64,
        product_count=6,
        embedding=spec,
        vector_dtype="float32",
        artifacts=(
            DenseArtifactRef(
                kind="parent_asins",
                filename="parent-asins.json",
                content_id="sha256:" + "4" * 64,
                byte_size=1,
            ),
            DenseArtifactRef(
                kind="vectors",
                filename="vectors.npy",
                content_id="sha256:" + "5" * 64,
                byte_size=1,
            ),
        ),
    )
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.98, 0.02, 0.0],
            [0.0, 1.0, 0.0],
            [0.01, 0.99, 0.0],
            [0.02, 0.98, 0.0],
        ],
        dtype=np.float32,
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return DenseIndex(
        index_id="sha256:" + "a" * 64,
        manifest=manifest,
        parent_asins=("A", "B", "C", "D", "E", "F"),
        vectors=vectors,
    )
