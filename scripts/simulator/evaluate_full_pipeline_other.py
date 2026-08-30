"""Run the real shopping pipeline against the official public toy simulator.

This file is deliberately an adapter, not a second retrieval implementation.  It
replays the organizer's public evaluator protocol while keeping the target ASIN on
the evaluator side of the boundary.  The agent side always asks ``other`` and uses
the repository's Query Understanding, Session Context, Intent Transparency,
multi-route retrieval, BGE reranking, and vector DPP selection components.

Every turn and every completed session is appended to JSONL before the next one
starts, so a long experiment leaves useful evidence even if it is interrupted.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for source_path in (ROOT, SRC):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    metric_summary,
    normalize_recommendations,
)
from shopping_copilot.catalog.semantic import CatalogSemanticGateway  # noqa: E402
from shopping_copilot.catalog.semantic.release import (  # noqa: E402
    load_catalog_semantic_release,
)
from shopping_copilot.catalog.semantic.runtime import (  # noqa: E402
    SYSTEM_PRODUCT_CATEGORY_FACET_ID,
)
from shopping_copilot.query_compiler import QueryCompiler  # noqa: E402
from shopping_copilot.query_understanding import (  # noqa: E402
    IntentMaterializer,
    QueryUnderstandingError,
    QueryUnderstandingService,
    ResolvedTurnIntent,
    ShownProductView,
    build_reconcile_request,
    category_options_from_registry,
    request_payload,
)
from shopping_copilot.query_understanding.deepseek import (  # noqa: E402
    DeepSeekConfig,
    DeepSeekProvider,
)
from shopping_copilot.query_understanding.errors import (  # noqa: E402
    QueryUnderstandingErrorCode,
)
from shopping_copilot.retrieval import (  # noqa: E402
    CrossEncoderRelevanceReranker,
    GreedyDPPSelector,
    IntentTransparencyEstimate,
    IntentVolumeEstimator,
    IntentVolumePolicy,
    SentenceTransformerCrossEncoderScorer,
    VectorCandidate,
    create_retrieval_controller,
    load_catalog_density,
    load_product_documents,
    normalized_fusion_relevance,
)
from shopping_copilot.session_context import (  # noqa: E402
    InteractionContext,
    ProfilePrior,
    SessionContext,
    SessionState,
    TurnRecord,
    encode_snapshot,
)
from shopping_copilot.session_context.models import IntentState  # noqa: E402
from shopping_copilot.simulator import DeepSeekSurfaceRealizer  # noqa: E402

REPORT_SCHEMA = "shopping-copilot/official-simulator-full-pipeline/v0"
TURN_SCHEMA = "shopping-copilot/official-simulator-turn-audit/v0"
SESSION_SCHEMA = "shopping-copilot/official-simulator-session-result/v0"
BGE_MODEL = "BAAI/bge-reranker-v2-m3"
BGE_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
ASK_ATTRIBUTE = "other"
ASSISTANT_QUESTION = "What other requirements or preferences matter to you?"
TRANSIENT_QU_ERRORS = frozenset(
    {
        QueryUnderstandingErrorCode.PROVIDER_RATE_LIMIT,
        QueryUnderstandingErrorCode.PROVIDER_TIMEOUT,
        QueryUnderstandingErrorCode.PROVIDER_UNAVAILABLE,
    }
)


@dataclass(slots=True)
class _SessionRuntime:
    context: SessionContext
    raw_profile: dict[str, object]
    previous_transparency: IntentTransparencyEstimate | None = None
    last_response: dict[str, object] | None = None
    last_turn_audit: dict[str, object] | None = None
    last_pipeline: dict[str, object] | None = None
    reusable_key: tuple[str, int] | None = None
    reusable_from_turn: int | None = None


class FullPipelineOtherAgent:
    """A target-blind adapter exposing the official ``reset/respond`` shape."""

    def __init__(
        self,
        *,
        service: QueryUnderstandingService,
        compiler: QueryCompiler,
        estimator: IntentVolumeEstimator,
        controller: Any,
        reranker: CrossEncoderRelevanceReranker | None,
        selector: GreedyDPPSelector,
        documents: dict[str, str],
        product_metadata: dict[str, dict[str, object]],
        category_options: tuple[Any, ...],
        allowed_dont_care_facets: tuple[str, ...],
        facet_registry: Any,
        qu_retry_count: int,
        repeat_noop_cache: bool,
    ) -> None:
        self._service = service
        self._compiler = compiler
        self._estimator = estimator
        self._controller = controller
        self._reranker = reranker
        self._selector = selector
        self._documents = documents
        self._metadata = product_metadata
        self._category_options = category_options
        self._allowed_dont_care_facets = allowed_dont_care_facets
        self._facet_registry = facet_registry
        self._qu_retry_count = qu_retry_count
        self._repeat_noop_cache = repeat_noop_cache
        self._sessions: dict[str, _SessionRuntime] = {}
        self._local_model_lock = threading.Lock()

    def reset(self, session_id: str, user_profile: dict[str, object]) -> None:
        """Start a clean typed Session Context without seeing evaluator truth."""

        profile = _profile_prior(user_profile)
        context = SessionContext(
            session_id=session_id,
            profile=profile,
            state=SessionState(
                intent=IntentState(
                    goal=None,
                    preferences=(),
                    dont_care_facets=frozenset(),
                    version=0,
                ),
                interaction=InteractionContext(turns=()),
                search_belief=None,
            ),
        )
        _context_payload(context, self._facet_registry)
        self._sessions[session_id] = _SessionRuntime(
            context=context,
            raw_profile=dict(user_profile),
        )

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict[str, object]:
        """Run one target-blind turn and always ask the simulator for ``other``."""

        if top_k != TOP_K:
            raise ValueError(f"this audit expects official top_k={TOP_K}")
        session = self._sessions[session_id]
        if turn != len(session.context.state.interaction.turns) + 1:
            raise ValueError("turn is not contiguous with Session Context")

        started = time.perf_counter()
        context_before = _context_payload(session.context, self._facet_registry)
        shown = _shown_product_views(session.last_response, self._metadata)
        request = build_reconcile_request(
            turn=turn,
            latest_utterance=user_message,
            current_intent=session.context.state.intent,
            category_options=self._category_options,
            shown_products=shown,
            last_assistant_message=(
                None
                if session.last_response is None
                else cast(str, session.last_response["message"])
            ),
            last_question=(None if session.last_response is None else ASSISTANT_QUESTION),
            allowed_dont_care_facets=self._allowed_dont_care_facets,
        )
        exact_request = request_payload(request)
        reuse_key = (_normalized_message(user_message), session.context.state.intent.version)
        if (
            self._repeat_noop_cache
            and session.reusable_key == reuse_key
            and session.last_pipeline is not None
            and session.last_response is not None
        ):
            return self._reuse_noop_turn(
                session=session,
                turn=turn,
                user_message=user_message,
                exact_request=exact_request,
                context_before=context_before,
                started=started,
            )

        timings: dict[str, float] = {}
        retry_errors: list[dict[str, object]] = []
        qu_started = time.perf_counter()
        try:
            resolved = self._resolve_with_retry(
                current=session.context.state.intent,
                request=request,
                retry_errors=retry_errors,
            )
        except Exception as error:
            timings["query_understanding_ms"] = _elapsed_ms(qu_started)
            return self._failed_turn(
                session=session,
                turn=turn,
                user_message=user_message,
                context_before=context_before,
                exact_request=exact_request,
                stage="query_understanding",
                error=error,
                retry_errors=retry_errors,
                timings=timings,
                started=started,
                resolved=None,
            )
        timings["query_understanding_ms"] = _elapsed_ms(qu_started)

        compile_started = time.perf_counter()
        try:
            compiled = self._compiler.compile(resolved)
        except Exception as error:
            timings["compile_ms"] = _elapsed_ms(compile_started)
            return self._failed_turn(
                session=session,
                turn=turn,
                user_message=user_message,
                context_before=context_before,
                exact_request=exact_request,
                stage="query_compiler",
                error=error,
                retry_errors=retry_errors,
                timings=timings,
                started=started,
                resolved=resolved,
            )
        timings["compile_ms"] = _elapsed_ms(compile_started)

        volume_started = time.perf_counter()
        try:
            with self._local_model_lock:
                transparency = self._estimator.estimate(
                    session_id=session.context.session_id,
                    intent=resolved.final_intent,
                    compiled=compiled,
                    previous=session.previous_transparency,
                    goal_switched=_explicit_goal_switch(resolved),
                )
            applied_transparency = (
                0.0 if transparency.transparency is None else float(transparency.transparency)
            )
            transparency_error = None
        except Exception as error:
            transparency = None
            applied_transparency = 0.5
            transparency_error = _error_payload(error)
        timings["intent_transparency_ms"] = _elapsed_ms(volume_started)

        retrieval_started = time.perf_counter()
        try:
            with self._local_model_lock:
                retrieval = self._controller.search(
                    compiled,
                    transparency=float(applied_transparency),
                )
        except Exception as error:
            timings["retrieval_ms"] = _elapsed_ms(retrieval_started)
            return self._failed_turn(
                session=session,
                turn=turn,
                user_message=user_message,
                context_before=context_before,
                exact_request=exact_request,
                stage="retrieval",
                error=error,
                retry_errors=retry_errors,
                timings=timings,
                started=started,
                resolved=resolved,
                compiled=compiled,
                transparency=transparency,
                applied_transparency=applied_transparency,
                transparency_error=transparency_error,
            )
        timings["retrieval_ms"] = _elapsed_ms(retrieval_started)

        ranking_started = time.perf_counter()
        ranking_error: dict[str, object] | None = None
        bge_result: Any | None = None
        dpp_result: Any | None = None
        if self._reranker is not None and retrieval.fused_candidates:
            try:
                relevance = normalized_fusion_relevance(retrieval.fused_candidates)
                candidates = tuple(
                    VectorCandidate(
                        parent_asin=item.parent_asin,
                        candidate_rank=item.rank,
                        relevance=item_relevance,
                    )
                    for item, item_relevance in zip(
                        retrieval.fused_candidates,
                        relevance,
                        strict=True,
                    )
                )
                with self._local_model_lock:
                    bge_result = self._reranker.rerank(
                        compiled.q_sem,
                        candidates,
                        documents=self._documents,
                        prior_weight=0.25,
                        batch_size=32,
                    )
                    dpp_result = self._selector.select(
                        bge_result.candidates,
                        top_k=top_k,
                        relevance_weight=float(retrieval.relevance_weight),
                    )
                recommendations = [item.parent_asin for item in dpp_result.hits]
                ranking_mode = "bge_dpp"
            except Exception as error:
                ranking_error = _error_payload(error)
                recommendations = [item.parent_asin for item in retrieval.hits[:top_k]]
                ranking_mode = "formal_mmr_fallback"
        else:
            recommendations = [item.parent_asin for item in retrieval.hits[:top_k]]
            ranking_mode = "formal_mmr"
        timings["ranking_ms"] = _elapsed_ms(ranking_started)

        response = _agent_response(recommendations, resolved=resolved)
        session.context = _advance_context(
            session.context,
            turn=turn,
            user_message=user_message,
            resolved=resolved,
            response=response,
        )
        context_after = _context_payload(session.context, self._facet_registry)
        if transparency is not None:
            session.previous_transparency = transparency
        session.last_response = response
        session.reusable_key = (
            (_normalized_message(user_message), session.context.state.intent.version)
            if resolved.update is None
            else None
        )
        session.reusable_from_turn = turn if session.reusable_key is not None else None

        pipeline = {
            "compiled_query": _json_value(compiled),
            "intent_transparency": {
                "estimate": None if transparency is None else transparency.as_payload(),
                "applied_transparency": applied_transparency,
                "error": transparency_error,
            },
            "retrieval": _retrieval_payload(retrieval),
            "ranking": _ranking_payload(
                mode=ranking_mode,
                bge_result=bge_result,
                dpp_result=dpp_result,
                fallback_hits=retrieval.hits,
                error=ranking_error,
            ),
            "recommendation_products": _recommendation_products(
                recommendations,
                self._metadata,
            ),
        }
        session.last_pipeline = pipeline
        timings["total_agent_ms"] = _elapsed_ms(started)
        session.last_turn_audit = {
            "schema": TURN_SCHEMA,
            "session_id": session.context.session_id,
            "turn": turn,
            "user_message": user_message,
            "raw_user_profile": session.raw_profile,
            "session_context_before": context_before,
            "query_understanding": {
                "status": "success",
                "request": exact_request,
                "retry_errors": retry_errors,
                "resolved_turn": _json_value(resolved),
            },
            **pipeline,
            "agent_response": response,
            "session_context_after": context_after,
            "timings": timings,
        }
        return response

    def last_audit(self, session_id: str) -> dict[str, object]:
        audit = self._sessions[session_id].last_turn_audit
        if audit is None:
            raise LookupError("the session has no completed turn")
        return audit

    def _resolve_with_retry(
        self,
        *,
        current: IntentState,
        request: Any,
        retry_errors: list[dict[str, object]],
    ) -> ResolvedTurnIntent:
        for attempt in range(1, self._qu_retry_count + 1):
            try:
                return self._service.resolve(current=current, request=request)
            except QueryUnderstandingError as error:
                retry_errors.append(
                    {
                        "attempt": attempt,
                        **_error_payload(error),
                    }
                )
                if error.code not in TRANSIENT_QU_ERRORS or attempt == self._qu_retry_count:
                    raise
                time.sleep(float(2 ** (attempt - 1)))
        raise AssertionError("retry loop must return or raise")

    def _reuse_noop_turn(
        self,
        *,
        session: _SessionRuntime,
        turn: int,
        user_message: str,
        exact_request: dict[str, object],
        context_before: dict[str, object],
        started: float,
    ) -> dict[str, object]:
        assert session.last_response is not None
        assert session.last_pipeline is not None
        response = dict(session.last_response)
        response["usage"] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        session.context = _advance_context(
            session.context,
            turn=turn,
            user_message=user_message,
            resolved=None,
            response=response,
        )
        context_after = _context_payload(session.context, self._facet_registry)
        session.last_response = response
        pipeline = _json_round_trip(session.last_pipeline)
        timings = {"total_agent_ms": _elapsed_ms(started)}
        session.last_turn_audit = {
            "schema": TURN_SCHEMA,
            "session_id": session.context.session_id,
            "turn": turn,
            "user_message": user_message,
            "raw_user_profile": session.raw_profile,
            "session_context_before": context_before,
            "query_understanding": {
                "status": "skipped_repeated_noop",
                "request": exact_request,
                "retry_errors": [],
                "resolved_turn": None,
                "reused_from_turn": session.reusable_from_turn,
            },
            **pipeline,
            "agent_response": response,
            "session_context_after": context_after,
            "timings": timings,
        }
        return response

    def _failed_turn(
        self,
        *,
        session: _SessionRuntime,
        turn: int,
        user_message: str,
        context_before: dict[str, object],
        exact_request: dict[str, object],
        stage: str,
        error: Exception,
        retry_errors: list[dict[str, object]],
        timings: dict[str, float],
        started: float,
        resolved: ResolvedTurnIntent | None,
        compiled: Any | None = None,
        transparency: IntentTransparencyEstimate | None = None,
        applied_transparency: float | None = None,
        transparency_error: dict[str, object] | None = None,
    ) -> dict[str, object]:
        previous = session.last_response
        recommendations = (
            []
            if previous is None
            else [str(item) for item in cast(list[object], previous["recommendations"])]
        )
        response = _agent_response(recommendations, resolved=resolved)
        session.context = _advance_context(
            session.context,
            turn=turn,
            user_message=user_message,
            resolved=resolved,
            response=response,
        )
        context_after = _context_payload(session.context, self._facet_registry)
        session.last_response = response
        session.reusable_key = None
        session.reusable_from_turn = None
        timings["total_agent_ms"] = _elapsed_ms(started)
        session.last_turn_audit = {
            "schema": TURN_SCHEMA,
            "session_id": session.context.session_id,
            "turn": turn,
            "user_message": user_message,
            "raw_user_profile": session.raw_profile,
            "session_context_before": context_before,
            "query_understanding": {
                "status": "error" if stage == "query_understanding" else "success",
                "request": exact_request,
                "retry_errors": retry_errors,
                "resolved_turn": None if resolved is None else _json_value(resolved),
            },
            "compiled_query": None if compiled is None else _json_value(compiled),
            "intent_transparency": {
                "estimate": None if transparency is None else transparency.as_payload(),
                "applied_transparency": applied_transparency,
                "error": transparency_error,
            },
            "retrieval": None,
            "ranking": None,
            "recommendation_products": _recommendation_products(
                recommendations,
                self._metadata,
            ),
            "agent_response": response,
            "session_context_after": context_after,
            "failure": {"stage": stage, "error": _error_payload(error)},
            "timings": timings,
        }
        return response


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    turns_path = output_dir / "turns.jsonl"
    sessions_path = output_dir / "sessions.jsonl"
    if not args.resume and (turns_path.exists() or sessions_path.exists()):
        raise SystemExit(
            f"output already contains logs; use --resume or choose another: {output_dir}"
        )

    samples = load_jsonl(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]
    catalog_ids, categories, products = catalog_index(args.catalog)
    completed_rows = _load_existing_jsonl(sessions_path) if args.resume else []
    completed_ids = {str(item["sample_id"]) for item in completed_rows}
    selected = [item for item in samples if str(item["sample_id"]) not in completed_ids]
    if not selected:
        raise SystemExit("no unfinished simulator sessions selected")

    api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise SystemExit("DeepSeek API key file is empty")

    surface_realizer: DeepSeekSurfaceRealizer | None = None
    if args.reply_model == "deepseek":
        reply_cache = args.reply_cache or output_dir / "deepseek-reply-cache.json"
        surface_realizer = DeepSeekSurfaceRealizer(
            api_key=api_key,
            cache_path=reply_cache,
            model=args.model,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
        )
        surface_requests = _fixed_other_surface_requests(
            samples=selected,
            categories=categories,
            products=products,
            max_turns=args.max_turns,
        )
        print(
            f"prewarming {len(set(surface_requests))} unique DeepSeek customer messages...",
            flush=True,
        )
        surface_realizer.prewarm(surface_requests, max_workers=args.reply_workers)

    initialization_started = time.perf_counter()
    release = load_catalog_semantic_release(args.semantic_release)
    gateway = CatalogSemanticGateway(release)
    service = QueryUnderstandingService(
        provider=DeepSeekProvider(
            api_key=api_key,
            config=DeepSeekConfig(
                model=args.model,
                base_url=args.base_url,
                timeout_seconds=args.timeout_seconds,
                max_tokens=args.max_tokens,
                temperature=0.0,
                strict_tools=False,
                disable_thinking=True,
            ),
        ),
        materializer=IntentMaterializer(gateway=gateway, grounder=release.grounder),
    )
    compiler = QueryCompiler(
        catalog_semantic_release_id=release.release_id,
        category_registry=release.category_registry,
    )
    print("initializing formal retrieval and intent-volume runtime...", flush=True)
    controller = create_retrieval_controller(
        index_path=args.dense_index,
        release_dir=args.semantic_release,
        catalog_path=args.catalog,
        device=args.device,
        local_files_only=True,
    )
    policy = IntentVolumePolicy()
    density = load_catalog_density(
        args.density_cache,
        dense_index=controller.retriever.index,
        temperature=policy.density_temperature,
    )
    estimator = IntentVolumeEstimator(
        dense_index=controller.retriever.index,
        embedder=controller.retriever.embedder,
        hard_mask_resolver=controller.hard_mask_resolver,
        density=density,
        policy=policy,
    )
    print("loading product documents...", flush=True)
    loaded_documents = load_product_documents(
        args.catalog,
        expected_parent_asins=set(controller.retriever.index.parent_asins),
    )
    documents = {item.parent_asin: _compact_document(item.text) for item in loaded_documents}
    metadata = _product_metadata(products)
    reranker = None
    if not args.disable_cross_encoder:
        print("loading pinned BGE reranker...", flush=True)
        reranker = CrossEncoderRelevanceReranker(
            scorer=SentenceTransformerCrossEncoderScorer(
                BGE_MODEL,
                revision=BGE_REVISION,
                device=args.device,
                local_files_only=True,
                max_length=384,
            )
        )
    agent = FullPipelineOtherAgent(
        service=service,
        compiler=compiler,
        estimator=estimator,
        controller=controller,
        reranker=reranker,
        selector=GreedyDPPSelector(index=controller.retriever.index),
        documents=documents,
        product_metadata=metadata,
        category_options=category_options_from_registry(release.category_registry),
        allowed_dont_care_facets=tuple(
            spec.id for spec in gateway.registry if spec.id != SYSTEM_PRODUCT_CATEGORY_FACET_ID
        ),
        facet_registry=gateway.registry,
        qu_retry_count=args.qu_retry_count,
        repeat_noop_cache=not args.disable_repeat_noop_cache,
    )
    initialization_seconds = time.perf_counter() - initialization_started

    manifest = {
        "schema": REPORT_SCHEMA,
        "created_at": _utc_now(),
        "inputs": {
            "dataset": str(args.dataset.resolve()),
            "catalog": str(args.catalog.resolve()),
            "semantic_release": str(args.semantic_release.resolve()),
            "dense_index": str(args.dense_index.resolve()),
            "density_cache": str(args.density_cache.resolve()),
            "selected_sample_count": len(samples),
            "remaining_sample_count": len(selected),
        },
        "adapter_policy": {
            "ask_attribute": ASK_ATTRIBUTE,
            "top_k": TOP_K,
            "max_turns": args.max_turns,
            "session_workers": args.workers,
            "surface_workers": args.reply_workers,
            "continue_after_hit": args.continue_after_hit,
            "target_visible_to_agent": False,
            "repeat_noop_cache": not args.disable_repeat_noop_cache,
        },
        "models": {
            "query_understanding": args.model,
            "simulator_reply": None if surface_realizer is None else args.model,
            "dense": controller.retriever.index.manifest.embedding.model_id,
            "cross_encoder": None if reranker is None else reranker.scorer.model_id,
        },
        "initialization_seconds": initialization_seconds,
    }
    _write_json(output_dir / "run.json", manifest)

    run_started = time.perf_counter()
    new_session_rows: list[dict[str, object]] = []
    turn_log_lock = threading.Lock()
    with (
        turns_path.open("a", encoding="utf-8", buffering=1) as turn_log,
        sessions_path.open("a", encoding="utf-8", buffering=1) as session_log,
    ):
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _run_session,
                    agent=agent,
                    sample=sample,
                    catalog_ids=catalog_ids,
                    categories=categories,
                    products=products,
                    max_turns=args.max_turns,
                    continue_after_hit=args.continue_after_hit,
                    turn_log=turn_log,
                    turn_log_lock=turn_log_lock,
                    surface_realizer=surface_realizer,
                ): sample
                for sample in selected
            }
            for ordinal, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                new_session_rows.append(row)
                session_log.write(_json_line(row))
                session_log.flush()
                print(
                    f"session {ordinal}/{len(selected)} {row['sample_id']} "
                    f"scenario={row['scenario_type']} hit={row['hit']} "
                    f"turn={row['first_hit_turn']}",
                    flush=True,
                )

    all_session_rows = [*completed_rows, *new_session_rows]
    summary = _summary_payload(
        sessions=all_session_rows,
        manifest=manifest,
        run_seconds=time.perf_counter() - run_started,
    )
    summary["simulator_reply_model"] = {
        "mode": args.reply_model,
        "usage": (
            None if surface_realizer is None else surface_realizer.usage.as_payload()
        ),
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(_render_summary(summary), encoding="utf-8")
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2), flush=True)
    print(f"logs: {output_dir.resolve()}", flush=True)
    return 0


def _run_session(
    *,
    agent: FullPipelineOtherAgent,
    sample: dict[str, object],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict[str, object]],
    max_turns: int,
    continue_after_hit: bool,
    turn_log: Any,
    turn_log_lock: threading.Lock,
    surface_realizer: DeepSeekSurfaceRealizer | None,
) -> dict[str, object]:
    sample_id = str(sample["sample_id"])
    session_id = f"official-public/{sample_id}"
    profile = cast(dict[str, object], sample["user_profile"])
    agent.reset(session_id, profile)

    target = str(cast(dict[str, object], sample["ground_truth"])["parent_asin"])
    intent_card, behavior = materialize_hidden_fields(sample, products)
    effective_sample = {**sample, "intent_card": intent_card, "behavior": behavior}
    disclosed: set[str] = set()
    boundary_used = False
    scenario = str(sample["scenario_type"])
    override_applied = scenario != "intent_override"
    canonical_user_message = initial_message(
        effective_sample,
        coarse_category(categories.get(target, [])),
        disclosed,
    )
    user_message = _surface_message(
        surface_realizer,
        canonical_user_message,
        "initial message",
    )
    hit_turn: int | None = None
    best_rank: int | None = None
    prompt_tokens = 0
    completion_tokens = 0
    error_turns = 0

    for turn in range(1, max_turns + 1):
        disclosed_before = sorted(disclosed)
        response = agent.respond(session_id, user_message, turn, TOP_K)
        if response.get("ask_attribute") != ASK_ATTRIBUTE:
            raise AssertionError("fixed-other adapter returned a different ask_attribute")
        audit = agent.last_audit(session_id)
        usage = response.get("usage")
        if type(usage) is dict:
            prompt_tokens += _non_negative_int(cast(dict[str, object], usage).get("prompt_tokens"))
            completion_tokens += _non_negative_int(
                cast(dict[str, object], usage).get("completion_tokens")
            )
        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        target_rank = ranked.index(target) + 1 if target in ranked else None
        scored_hit = override_applied and target_rank is not None
        if audit.get("failure") is not None:
            error_turns += 1
        audit["evaluator"] = {
            "sample_id": sample_id,
            "scenario_type": scenario,
            "category_bucket": sample.get("category_bucket"),
            "difficulty_bucket": sample.get("difficulty_bucket"),
            "target_parent_asin": target,
            "target_was_not_passed_to_agent": True,
            "override_applied_before_scoring": override_applied,
            "normalized_top_10": ranked,
            "target_rank": target_rank,
            "scored_hit": scored_hit,
            "simulator_disclosed_before": disclosed_before,
        }
        with turn_log_lock:
            turn_log.write(_json_line(audit))
            turn_log.flush()
        if scored_hit:
            best_rank = target_rank if best_rank is None else min(best_rank, target_rank)
            if hit_turn is None:
                hit_turn = turn
            if not continue_after_hit:
                break
        if turn == max_turns:
            break

        override = cast(dict[str, object], effective_sample.get("behavior", {})).get("override")
        override_object = cast(dict[str, object], override) if type(override) is dict else {}
        if not override_applied and turn + 1 == int(override_object.get("turn", 3)):
            override_applied = True
            new_value = str(override_object.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            canonical_user_message = str(
                override_object.get(
                    "message",
                    "Actually, please ignore my earlier preference.",
                )
            )
            user_message = _surface_message(
                surface_realizer,
                canonical_user_message,
                "intent-override customer reply",
            )
        else:
            canonical_user_message, boundary_used = customer_reply(
                effective_sample,
                ASK_ATTRIBUTE,
                disclosed,
                boundary_used,
            )
            user_message = _surface_message(
                surface_realizer,
                canonical_user_message,
                "follow-up customer reply",
            )

    return {
        "schema": SESSION_SCHEMA,
        "sample_id": sample_id,
        "session_id": session_id,
        "scenario_type": scenario,
        "hit": hit_turn is not None,
        "first_hit_turn": hit_turn,
        "best_rank": best_rank,
        "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        "turns_executed": turn,
        "error_turn_count": error_turns,
        "reported_token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _summary_payload(
    *,
    sessions: list[dict[str, object]],
    manifest: dict[str, object],
    run_seconds: float,
) -> dict[str, object]:
    overall = metric_summary(cast(list[dict], sessions))
    mttc = overall["mttc"]
    efficiency = 0.0 if mttc is None else max(0.0, min(1.0, (11.0 - float(mttc)) / 10.0))
    technical_score = (
        0.50 * float(overall["hit_rate_at_10"]) + 0.30 * float(overall["mrr"]) + 0.20 * efficiency
    )
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for session in sessions:
        grouped[str(session["scenario_type"])].append(session)
    prompt_tokens = sum(
        int(cast(dict[str, object], item["reported_token_usage"])["prompt_tokens"])
        for item in sessions
    )
    completion_tokens = sum(
        int(cast(dict[str, object], item["reported_token_usage"])["completion_tokens"])
        for item in sessions
    )
    return {
        "schema": REPORT_SCHEMA,
        "completed_at": _utc_now(),
        "run": manifest,
        "runtime": {
            "evaluation_seconds": run_seconds,
            "mean_seconds_per_session": run_seconds / max(1, len(sessions)),
            "error_turn_count": sum(int(item["error_turn_count"]) for item in sessions),
        },
        "metrics": {
            **overall,
            "efficiency": round(efficiency, 6),
            "recommended_technical_score": round(technical_score, 6),
            "scenario_metrics": {
                name: metric_summary(cast(list[dict], rows))
                for name, rows in sorted(grouped.items())
            },
            "reported_token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        },
        "sessions": sessions,
    }


def _advance_context(
    context: SessionContext,
    *,
    turn: int,
    user_message: str,
    resolved: ResolvedTurnIntent | None,
    response: dict[str, object],
) -> SessionContext:
    before = context.state.intent
    after = before if resolved is None else resolved.final_intent
    record = TurnRecord(
        turn=turn,
        user_message=user_message,
        intent_version_before=before.version,
        accepted_update=None if resolved is None else resolved.update,
        intent_version_after=after.version,
        assistant_message=str(response["message"]),
        question=ASSISTANT_QUESTION,
        question_key=f"ask_attribute:{ASK_ATTRIBUTE}",
        ask_attribute=ASK_ATTRIBUTE,
        shown_product_ids=tuple(
            str(item) for item in cast(list[object], response["recommendations"])
        ),
        feedback=() if resolved is None else resolved.feedback,
        search_belief_probe_id=None,
    )
    return SessionContext(
        session_id=context.session_id,
        profile=context.profile,
        state=SessionState(
            intent=after,
            interaction=InteractionContext(turns=(*context.state.interaction.turns, record)),
            search_belief=None,
        ),
    )


def _agent_response(
    recommendations: list[str],
    *,
    resolved: ResolvedTurnIntent | None,
) -> dict[str, object]:
    traces = () if resolved is None else resolved.trace.attempts
    prompt_tokens = sum(item.prompt_tokens or 0 for item in traces)
    completion_tokens = sum(item.completion_tokens or 0 for item in traces)
    return {
        "message": (
            "Here are my current best options. "
            "What other requirements or preferences matter to you?"
        ),
        "ask_attribute": ASK_ATTRIBUTE,
        "recommendations": recommendations[:TOP_K],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _shown_product_views(
    response: dict[str, object] | None,
    metadata: dict[str, dict[str, object]],
) -> tuple[ShownProductView, ...]:
    if response is None:
        return ()
    recommendations = cast(list[object], response["recommendations"])
    return tuple(
        ShownProductView(
            ref=f"product_{index}",
            product_ids=(str(parent_asin),),
            label=str(metadata.get(str(parent_asin), {}).get("title") or "product"),
        )
        for index, parent_asin in enumerate(recommendations)
    )


def _profile_prior(raw: dict[str, object]) -> ProfilePrior:
    tags = raw.get("preference_tags")
    if type(tags) is not list or any(type(item) is not str for item in tags):
        raise ValueError("user_profile.preference_tags must be strings")
    rating = raw.get("average_prior_rating")
    if rating is not None and type(rating) not in (int, float):
        raise ValueError("user_profile.average_prior_rating must be numeric or null")
    return ProfilePrior(
        purchase_frequency=str(raw.get("purchase_frequency", "")),
        average_prior_rating=None if rating is None else float(rating),
        rating_style=str(raw.get("rating_style", "")),
        preference_tags=tuple(cast(list[str], tags)),
        summary=str(raw.get("summary", "")),
    )


def _explicit_goal_switch(resolved: ResolvedTurnIntent) -> bool:
    update = resolved.update
    if update is None:
        return False
    return any(
        getattr(operation, "op", None) == "switch_goal"
        and not getattr(operation, "carry_preference_ids", ())
        for operation in update.operations
    )


def _retrieval_payload(result: Any) -> dict[str, object]:
    return {
        "transparency": result.transparency,
        "relevance_weight": result.relevance_weight,
        "hard_mask": {
            "eligible_count": len(result.hard_mask.eligible_parent_asins),
            "hard_filter_relaxed": result.hard_mask.hard_filter_relaxed,
            "relaxed_preference_ids": [
                item.preference_id for item in result.hard_mask.relaxed_constraints
            ],
            "trace": [_json_value(item) for item in result.hard_mask.trace],
        },
        "routes": [
            {
                "route": route.route.value,
                "requested_top_k": route.requested_top_k,
                "available": route.available,
                "reason": route.reason,
                "hits": [_json_value(item) for item in route.hits],
            }
            for route in result.routes
        ],
        "fused_candidates": [_json_value(item) for item in result.fused_candidates],
        "formal_mmr_hits": [_json_value(item) for item in result.hits],
    }


def _ranking_payload(
    *,
    mode: str,
    bge_result: Any | None,
    dpp_result: Any | None,
    fallback_hits: tuple[Any, ...],
    error: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "mode": mode,
        "error": error,
        "cross_encoder": None if bge_result is None else _json_value(bge_result),
        "dpp": None if dpp_result is None else _json_value(dpp_result),
        "formal_mmr_fallback_hits": [_json_value(item) for item in fallback_hits],
    }


def _recommendation_products(
    parent_asins: list[str],
    metadata: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {"rank": rank, "parent_asin": parent_asin, **metadata.get(parent_asin, {})}
        for rank, parent_asin in enumerate(parent_asins, start=1)
    ]


def _product_metadata(
    products: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    return {
        parent_asin: {
            "title": product.get("title"),
            "categories": product.get("categories"),
            "price": product.get("price"),
            "store": product.get("store"),
        }
        for parent_asin, product in products.items()
    }


def _context_payload(context: SessionContext, registry: Any) -> dict[str, object]:
    decoded = json.loads(encode_snapshot(context, registry).decode("utf-8"))
    if type(decoded) is not dict:
        raise TypeError("encoded Session Context must be an object")
    return cast(dict[str, object], decoded)


def _json_value(value: object) -> object:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("audit payload contains a non-finite float")
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        payload = {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
        payload["type"] = type(value).__name__
        operation = getattr(value, "op", None)
        if type(operation) is str:
            payload["op"] = operation
        return payload
    if type(value) in (tuple, list):
        return [_json_value(item) for item in cast(Any, value)]
    if type(value) in (set, frozenset):
        return [_json_value(item) for item in sorted(cast(Any, value))]
    if type(value) is dict:
        return {
            str(key): _json_value(item) for key, item in cast(dict[object, object], value).items()
        }
    raise TypeError(f"unsupported audit value: {type(value).__name__}")


def _error_payload(error: Exception) -> dict[str, object]:
    payload: dict[str, object] = {"type": type(error).__name__, "message": str(error)}
    if isinstance(error, QueryUnderstandingError):
        payload.update(
            {
                "code": error.code.value,
                "path": list(error.path),
                "details": {key: value for key, value in error.details},
            }
        )
    return payload


def _compact_document(text: str) -> str:
    kept = []
    for line in text.splitlines():
        label = line.partition(":")[0]
        if label in {"title", "categories", "store", "features", "details"}:
            kept.append(line)
    return "\n".join(kept)[:2400]


def _render_summary(summary: dict[str, object]) -> str:
    metrics = cast(dict[str, object], summary["metrics"])
    runtime = cast(dict[str, object], summary["runtime"])
    scenario_metrics = cast(dict[str, dict[str, object]], metrics["scenario_metrics"])
    lines = [
        "# Official public simulator · full-pipeline fixed-other run",
        "",
        "This is an integration smoke test of the real architecture behind a thin official-API adapter.",
        "The target ASIN stayed in the evaluator and was never passed to Query Understanding or retrieval.",
        "Every response used `ask_attribute = other`.",
        "",
        "## Overall",
        "",
        f"- Sessions: **{metrics['sample_count']}**",
        f"- Hit@10: **{float(metrics['hit_rate_at_10']):.3f}**",
        f"- MRR: **{float(metrics['mrr']):.3f}**",
        f"- MTTC: **{float(metrics['mttc']):.3f}**",
        f"- Suggested technical score: **{float(metrics['recommended_technical_score']):.3f}**",
        f"- Error turns: **{runtime['error_turn_count']}**",
        "",
        "## By scenario",
        "",
        "| scenario | n | Hit@10 | MRR | MTTC |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, item in sorted(scenario_metrics.items()):
        lines.append(
            f"| {name} | {item['sample_count']} | {float(item['hit_rate_at_10']):.3f} | "
            f"{float(item['mrr']):.3f} | {float(item['mttc']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `run.json`: frozen inputs, policies, and model IDs",
            "- `turns.jsonl`: complete per-turn Session Context, QU request/result, T_t, retrieval, ranking, response, and evaluator observation",
            "- `sessions.jsonl`: incrementally checkpointed per-session outcomes",
            "- `summary.json`: metrics and all session outcomes",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/public_set.jsonl")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.jsonl")
    parser.add_argument(
        "--semantic-release",
        type=Path,
        default=ROOT / "artifacts/catalog-semantic/release-v0",
    )
    parser.add_argument(
        "--dense-index",
        type=Path,
        default=ROOT / "artifacts/retrieval/dense-v0",
    )
    parser.add_argument(
        "--density-cache",
        type=Path,
        default=ROOT / "artifacts/retrieval/intent-volume-density-v0.npz",
    )
    parser.add_argument("--api-key-file", type=Path, default=ROOT / "dpskapi")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--reply-model",
        choices=("template", "deepseek"),
        default="template",
        help="surface wording for the official simulator customer",
    )
    parser.add_argument(
        "--reply-cache",
        type=Path,
        default=None,
        help="persistent DeepSeek customer-message cache",
    )
    parser.add_argument("--reply-workers", type=int, default=16)
    parser.add_argument("--qu-retry-count", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="concurrent sessions; local CUDA model calls remain serialized",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument("--disable-cross-encoder", action="store_true")
    parser.add_argument("--disable-repeat-noop-cache", action="store_true")
    parser.add_argument(
        "--continue-after-hit",
        action="store_true",
        help="keep replaying visible turns after the first hit for state and mask audits",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "artifacts/simulator"
        / f"full-pipeline-other-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not 1 <= args.max_turns <= MAX_TURNS:
        parser.error(f"--max-turns must be in [1, {MAX_TURNS}]")
    if args.qu_retry_count < 1:
        parser.error("--qu-retry-count must be positive")
    if args.reply_workers < 1:
        parser.error("--reply-workers must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def _surface_message(
    realizer: DeepSeekSurfaceRealizer | None,
    canonical_message: str,
    reply_type: str,
) -> str:
    if realizer is None:
        return canonical_message
    return realizer.rewrite(canonical_message, reply_type)


def _fixed_other_surface_requests(
    *,
    samples: list[dict[str, object]],
    categories: dict[str, list[str]],
    products: dict[str, dict[str, object]],
    max_turns: int,
) -> list[tuple[str, str]]:
    requests: list[tuple[str, str]] = []
    for sample in samples:
        target = str(cast(dict[str, object], sample["ground_truth"])["parent_asin"])
        intent_card, behavior = materialize_hidden_fields(sample, products)
        effective_sample = {**sample, "intent_card": intent_card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        scenario = str(sample["scenario_type"])
        override_applied = scenario != "intent_override"
        canonical = initial_message(
            effective_sample,
            coarse_category(categories.get(target, [])),
            disclosed,
        )
        requests.append((canonical, "initial message"))
        for turn in range(1, max_turns):
            override = cast(dict[str, object], effective_sample.get("behavior", {})).get(
                "override"
            )
            override_object = cast(dict[str, object], override) if type(override) is dict else {}
            if not override_applied and turn + 1 == int(override_object.get("turn", 3)):
                override_applied = True
                new_value = str(override_object.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                canonical = str(
                    override_object.get(
                        "message",
                        "Actually, please ignore my earlier preference.",
                    )
                )
                requests.append((canonical, "intent-override customer reply"))
            else:
                canonical, boundary_used = customer_reply(
                    effective_sample,
                    ASK_ATTRIBUTE,
                    disclosed,
                    boundary_used,
                )
                requests.append((canonical, "follow-up customer reply"))
    return requests


def _load_existing_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [cast(dict[str, object], item) for item in load_jsonl(path)]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _json_line(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"


def _json_round_trip(payload: dict[str, object]) -> dict[str, object]:
    observed = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    return cast(dict[str, object], observed)


def _normalized_message(message: str) -> str:
    return " ".join(message.split()).casefold()


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _non_negative_int(value: object) -> int:
    return value if type(value) is int and value >= 0 else 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
