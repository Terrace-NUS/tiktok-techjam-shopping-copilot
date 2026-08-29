"""Replay QU prompt suites through DeepSeek, compilation, hard mask, and Probe."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.query_understanding.evaluate_prompts import (  # noqa: E402
    evaluate_critical_assertions,
)
from scripts.query_understanding.suites import (  # noqa: E402
    PromptConversation,
    PromptSuite,
    PromptTurn,
    load_prompt_suite,
)
from shopping_copilot.catalog.semantic import (  # noqa: E402
    CatalogSemanticGateway,
)
from shopping_copilot.catalog.semantic.release import (  # noqa: E402
    load_catalog_semantic_release,
)
from shopping_copilot.catalog.semantic.runtime import (  # noqa: E402
    SYSTEM_PRODUCT_CATEGORY_FACET_ID,
)
from shopping_copilot.query_compiler import QueryCompiler  # noqa: E402
from shopping_copilot.query_understanding import (  # noqa: E402
    IntentMaterializer,
    ProviderTrace,
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
from shopping_copilot.retrieval import (  # noqa: E402
    create_resolved_compiled_probe_runner,
    load_bound_transparency_calibration,
)
from shopping_copilot.session_context import (  # noqa: E402
    IntentState,
    InteractionContext,
    Preference,
    ProductFeedback,
    SearchBelief,
    SessionContext,
    SessionState,
    TurnRecord,
    encode_snapshot,
)

REPORT_SCHEMA = "shopping-copilot/qu-to-probe-evaluation/v2"
DEFAULT_NATURAL_SUITE = Path("config/query_understanding/natural-prompts-v0.json")
DEFAULT_SIMULATOR_SUITE = Path("config/query_understanding/simulator-prompts-v0.json")
DEFAULT_RELEASE = Path("artifacts/catalog-semantic/release-v0")
DEFAULT_DENSE_INDEX = Path("artifacts/retrieval/dense-v0")
DEFAULT_CALIBRATION = Path("config/retrieval/transparency-calibration-v1.json")


def main() -> int:
    args = _parse_args()
    suites = _load_suites(args)
    selected = tuple(
        (
            suite,
            _select_conversations(
                suite,
                tier=args.tier,
                limit=args.limit,
                simulator_limit_per_scenario=args.simulator_limit_per_scenario,
            ),
        )
        for suite in suites
    )
    selected_turn_count = sum(
        len(_selected_turns(conversation, max_turn=args.max_turn))
        for _, conversations in selected
        for conversation in conversations
    )
    if selected_turn_count == 0:
        raise SystemExit("no turns selected")

    api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise SystemExit("API key file is empty")

    release = load_catalog_semantic_release(args.release)
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
    calibration = load_bound_transparency_calibration(args.calibration)
    print("building bound Retrieval Evidence + Probe runtime...", flush=True)
    probe_runner = create_resolved_compiled_probe_runner(
        index_path=args.dense_index,
        release_dir=args.release,
        calibration=calibration,
        catalog_path=args.catalog,
        device=args.device,
        local_files_only=True,
        probe_k=calibration.probe_k,
        mode_threshold=calibration.mode_similarity_threshold,
    )
    category_options = category_options_from_registry(release.category_registry)
    category_labels = {scope.id: scope.label for scope in release.category_registry.scopes}
    allowed_dont_care_facets = tuple(
        spec.id for spec in gateway.registry if spec.id != SYSTEM_PRODUCT_CATEGORY_FACET_ID
    )

    records: list[dict[str, object]] = []
    completed = 0
    started_all = time.perf_counter()
    for suite, conversations in selected:
        for conversation in conversations:
            current = IntentState(
                goal=conversation.initial_goal,
                preferences=(),
                dont_care_facets=frozenset(),
                version=0,
            )
            session_context = SessionContext(
                session_id=f"evaluation/{suite.suite_id}/{conversation.identifier}",
                profile=None,
                state=SessionState(
                    intent=current,
                    interaction=InteractionContext(turns=()),
                    search_belief=None,
                ),
            )
            blocked = False
            for turn in _selected_turns(conversation, max_turn=args.max_turn):
                if blocked:
                    records.append(_skipped_record(suite, conversation, turn))
                    completed += 1
                    continue
                shown = _shown_product_views(suite, conversation, turn)
                request = build_reconcile_request(
                    turn=turn.turn,
                    latest_utterance=turn.user_message,
                    current_intent=current,
                    category_options=category_options,
                    shown_products=shown,
                    last_assistant_message=turn.last_assistant_message,
                    last_question=turn.last_question,
                    allowed_dont_care_facets=allowed_dont_care_facets,
                )
                before = current
                context_before = _session_context_payload(session_context, gateway.registry)
                exact_request = request_payload(request)
                qu_started = time.perf_counter()
                try:
                    resolved = service.resolve(current=current, request=request)
                except Exception as error:
                    records.append(
                        _error_record(
                            suite,
                            conversation,
                            turn,
                            stage="query_understanding",
                            elapsed_ms=(time.perf_counter() - qu_started) * 1000.0,
                            error=error,
                            session_context_before=context_before,
                            deepseek_request_payload=exact_request,
                        )
                    )
                    blocked = True
                    completed += 1
                    _progress(completed, selected_turn_count, started_all)
                    continue

                qu_elapsed_ms = (time.perf_counter() - qu_started) * 1000.0
                outcomes = (
                    evaluate_critical_assertions(
                        turn.critical_assertions,
                        before=before,
                        resolved=resolved,
                        shown_products=shown,
                        category_labels=category_labels,
                    )
                    if suite.cohort == "natural"
                    else ()
                )
                current = resolved.final_intent
                try:
                    compiled = compiler.compile(resolved)
                except Exception as error:
                    session_context = _advance_session_context(
                        session_context,
                        turn=turn,
                        shown=shown,
                        resolved=resolved,
                        search_belief=None,
                    )
                    records.append(
                        _error_record(
                            suite,
                            conversation,
                            turn,
                            stage="query_compiler",
                            elapsed_ms=qu_elapsed_ms,
                            error=error,
                            resolved_intent=_intent_payload(resolved.final_intent),
                            session_context_before=context_before,
                            session_context_after=_session_context_payload(
                                session_context,
                                gateway.registry,
                            ),
                            deepseek_request_payload=exact_request,
                            resolved_turn=_resolved_payload(resolved),
                        )
                    )
                    completed += 1
                    _progress(completed, selected_turn_count, started_all)
                    continue

                base = _base_record(suite, conversation, turn)
                if not compiled.search_ready:
                    session_context = _advance_session_context(
                        session_context,
                        turn=turn,
                        shown=shown,
                        resolved=resolved,
                        search_belief=None,
                    )
                    records.append(
                        {
                            **base,
                            "status": "not_searchable",
                            "qu_latency_ms": round(qu_elapsed_ms, 3),
                            "qu_attempts": _attempt_payload(resolved.trace.attempts),
                            "critical_semantic_pass": (
                                None if not outcomes else all(item.passed for item in outcomes)
                            ),
                            "session_context_before": context_before,
                            "session_context_after": _session_context_payload(
                                session_context,
                                gateway.registry,
                            ),
                            "deepseek_request_payload": exact_request,
                            "resolved_turn": _resolved_payload(resolved),
                            "final_intent": _intent_payload(resolved.final_intent),
                            "compiled": _compiled_payload(compiled),
                            "mask": None,
                            "probe": None,
                            "error": None,
                        }
                    )
                    completed += 1
                    _progress(completed, selected_turn_count, started_all)
                    continue

                probe_started = time.perf_counter()
                try:
                    run = probe_runner.run(compiled)
                except Exception as error:
                    session_context = _advance_session_context(
                        session_context,
                        turn=turn,
                        shown=shown,
                        resolved=resolved,
                        search_belief=None,
                    )
                    records.append(
                        _error_record(
                            suite,
                            conversation,
                            turn,
                            stage="hard_mask_or_probe",
                            elapsed_ms=qu_elapsed_ms,
                            error=error,
                            resolved_intent=_intent_payload(resolved.final_intent),
                            compiled=_compiled_payload(compiled),
                            session_context_before=context_before,
                            session_context_after=_session_context_payload(
                                session_context,
                                gateway.registry,
                            ),
                            deepseek_request_payload=exact_request,
                            resolved_turn=_resolved_payload(resolved),
                        )
                    )
                    completed += 1
                    _progress(completed, selected_turn_count, started_all)
                    continue

                session_context = _advance_session_context(
                    session_context,
                    turn=turn,
                    shown=shown,
                    resolved=resolved,
                    search_belief=run.probe_run.search_belief,
                )
                records.append(
                    {
                        **base,
                        "status": "success",
                        "qu_latency_ms": round(qu_elapsed_ms, 3),
                        "probe_latency_ms": round(
                            (time.perf_counter() - probe_started) * 1000.0,
                            3,
                        ),
                        "qu_attempts": _attempt_payload(resolved.trace.attempts),
                        "critical_semantic_pass": (
                            None if not outcomes else all(item.passed for item in outcomes)
                        ),
                        "session_context_before": context_before,
                        "session_context_after": _session_context_payload(
                            session_context,
                            gateway.registry,
                        ),
                        "deepseek_request_payload": exact_request,
                        "resolved_turn": _resolved_payload(resolved),
                        "assertions": [
                            {
                                "kind": item.kind,
                                "pass": item.passed,
                                "reason": item.reason,
                            }
                            for item in outcomes
                        ],
                        "final_intent": _intent_payload(resolved.final_intent),
                        "compiled": _compiled_payload(compiled),
                        "mask": _mask_payload(run.mask_resolution),
                        "probe": _probe_payload(run.probe_run),
                        "error": None,
                    }
                )
                completed += 1
                _progress(completed, selected_turn_count, started_all)

    report = {
        "schema": REPORT_SCHEMA,
        "selection": {
            "tier": args.tier,
            "cohort": args.cohort,
            "limit_per_cohort": args.limit,
            "simulator_limit_per_scenario": args.simulator_limit_per_scenario,
        },
        "runtime": {
            "model": args.model,
            "catalog_id": release.manifest.catalog_id,
            "catalog_semantic_release_id": release.release_id,
            "dense_index_id": calibration.dense_index_id,
            "probe_k": calibration.probe_k,
            "mode_similarity_threshold": calibration.mode_similarity_threshold,
            "low_anchor": calibration.low_anchor,
            "high_anchor": calibration.high_anchor,
        },
        "suite_inventory": [
            {
                "suite_id": suite.suite_id,
                "cohort": suite.cohort,
                "conversation_count": len(conversations),
                "turn_count": sum(len(item.turns) for item in conversations),
            }
            for suite, conversations in selected
        ],
        "summary": _summary(records, selected_turn_count=selected_turn_count),
        "turns": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {args.output}", flush=True)
    return 0


def _summary(records: list[dict[str, object]], *, selected_turn_count: int) -> dict[str, object]:
    successful = [item for item in records if item["status"] == "success"]
    probe_payloads = [cast(dict[str, object], item["probe"]) for item in successful]
    available = [
        cast(float, item["certainty"]) for item in probe_payloads if item["certainty"] is not None
    ]
    diagnostics = Counter(str(item["diagnostic_status"]) for item in probe_payloads)
    bins = Counter(_certainty_bin(value) for value in available)
    qu_success = sum(
        item["status"] != "skipped_after_failure" and item.get("final_intent") is not None
        for item in records
    )
    token_totals: Counter[str] = Counter()
    for item in records:
        attempts = item.get("qu_attempts")
        if type(attempts) is list:
            for attempt in cast(list[dict[str, object]], attempts):
                for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = attempt[field]
                    if type(value) is int:
                        token_totals[field] += value
    clarity_pairs = _clarity_pairs(records)
    return {
        "selected_turn_count": selected_turn_count,
        "qu_success_count": qu_success,
        "pipeline_success_count": len(successful),
        "not_searchable_count": sum(item["status"] == "not_searchable" for item in records),
        "error_count": sum(item["status"] == "error" for item in records),
        "ct_available_count": len(available),
        "ct_unavailable_count": len(successful) - len(available),
        "ct_min": min(available) if available else None,
        "ct_median": statistics.median(available) if available else None,
        "ct_mean": statistics.fmean(available) if available else None,
        "ct_max": max(available) if available else None,
        "ct_bins": dict(sorted(bins.items())),
        "diagnostic_status_counts": dict(sorted(diagnostics.items())),
        "hard_filter_relaxed_count": sum(
            cast(dict[str, object], item["mask"])["hard_filter_relaxed"] is True
            for item in successful
        ),
        "critical_turn_pass_count": sum(
            item.get("critical_semantic_pass") is True for item in records
        ),
        "critical_turn_fail_count": sum(
            item.get("critical_semantic_pass") is False for item in records
        ),
        "successful_turn_token_usage": dict(sorted(token_totals.items())),
        "clarity_story_pairs": clarity_pairs,
    }


def _clarity_pairs(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for item in records:
        conversation_id = str(item["conversation_id"])
        if conversation_id.startswith("c0") and item["cohort"] == "natural":
            grouped.setdefault(conversation_id, []).append(item)
    result: list[dict[str, object]] = []
    for conversation_id, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: cast(int, item["turn"]))
        if len(ordered) != 2:
            continue
        values = []
        for item in ordered:
            probe = item.get("probe")
            values.append(
                None if type(probe) is not dict else cast(dict[str, object], probe)["certainty"]
            )
        delta = (
            cast(float, values[1]) - cast(float, values[0])
            if values[0] is not None and values[1] is not None
            else None
        )
        result.append(
            {
                "conversation_id": conversation_id,
                "vague_ct": values[0],
                "specific_ct": values[1],
                "delta": delta,
            }
        )
    return result


def _certainty_bin(value: float) -> str:
    if value < 0.2:
        return "[0.0,0.2)"
    if value < 0.4:
        return "[0.2,0.4)"
    if value < 0.6:
        return "[0.4,0.6)"
    if value < 0.8:
        return "[0.6,0.8)"
    return "[0.8,1.0]"


def _probe_payload(run: object) -> dict[str, object]:
    from shopping_copilot.retrieval import CompiledProbeRun

    assert type(run) is CompiledProbeRun
    semantic = run.snapshot.semantic
    diagnostics = run.estimate.diagnostics
    return {
        "probe_id": run.snapshot.probe_id,
        "certainty": run.estimate.certainty,
        "mode_coherence": semantic.equal_mode_coherence.debiased_pairwise_cosine,
        "listing_coherence": semantic.raw_listing_coherence.debiased_pairwise_cosine,
        "mode_count": len(semantic.modes),
        "largest_mode_share": semantic.largest_mode_share,
        "effective_mode_count": semantic.effective_mode_count,
        "dense_count": len(semantic.hits),
        "lexical_count": len(run.snapshot.lexical.hits),
        "lexical_available": run.snapshot.lexical.available,
        "lexical_token_coverage": diagnostics.lexical_token_coverage,
        "diagnostic_status": diagnostics.status.value,
        "diagnostic_reasons": list(diagnostics.reason_codes),
        "ranking_hits": [
            {
                "parent_asin": hit.parent_asin,
                "score": hit.score,
                "rank": hit.rank,
            }
            for hit in run.ranking.hits
        ],
        "lexical": {
            "probe_k": run.snapshot.lexical.probe_k,
            "tokens": list(run.snapshot.lexical.tokens),
            "eligible_count": run.snapshot.lexical.eligible_count,
            "matched_count": run.snapshot.lexical.matched_count,
            "matched_token_count": run.snapshot.lexical.matched_token_count,
            "mean_normalized_idf": run.snapshot.lexical.mean_normalized_idf,
            "available": run.snapshot.lexical.available,
            "reason": run.snapshot.lexical.reason,
            "hits": [
                {
                    "parent_asin": hit.parent_asin,
                    "raw_bm25": hit.raw_bm25,
                    "rank": hit.rank,
                }
                for hit in run.snapshot.lexical.hits
            ],
        },
        "semantic": {
            "probe_k": semantic.probe_k,
            "threshold": semantic.threshold,
            "memberships": [
                {
                    "parent_asin": item.parent_asin,
                    "dense_rank": item.dense_rank,
                    "mode_id": item.mode_id,
                    "similarity_to_leader": item.similarity_to_leader,
                }
                for item in semantic.memberships
            ],
            "modes": [
                {
                    "id": mode.id,
                    "leader_id": mode.leader_id,
                    "size": mode.size,
                    "best_score": mode.best_score,
                    "representative_ids": list(mode.representative_ids),
                }
                for mode in semantic.modes
            ],
            "raw_listing_coherence": _coherence_payload(semantic.raw_listing_coherence),
            "equal_mode_coherence": _coherence_payload(semantic.equal_mode_coherence),
            "largest_mode_share": semantic.largest_mode_share,
            "effective_mode_count": semantic.effective_mode_count,
            "duplicate_concentration_warning": (semantic.duplicate_concentration_warning),
        },
        "snapshot_identity": {
            "schema": run.snapshot.schema,
            "probe_policy_id": run.snapshot.probe_policy_id,
            "compiled_query_digest": run.snapshot.compiled_query_digest,
            "eligibility_digest": run.snapshot.eligibility_digest,
            "catalog_id": run.snapshot.catalog_id,
            "catalog_semantic_release_id": run.snapshot.catalog_semantic_release_id,
            "dense_index_id": run.snapshot.dense_index_id,
        },
        "estimate": {
            "policy_id": run.estimate.policy_id,
            "raw_mode_coherence": run.estimate.raw_mode_coherence,
            "controller_fallback": run.estimate.controller_fallback,
            "diagnostics": {
                "status": diagnostics.status.value,
                "reason_codes": list(diagnostics.reason_codes),
                "probe_k": diagnostics.probe_k,
                "eligible_count": diagnostics.eligible_count,
                "dense_count": diagnostics.dense_count,
                "lexical_count": diagnostics.lexical_count,
                "fill_ratio": diagnostics.fill_ratio,
                "lexical_available": diagnostics.lexical_available,
                "route_overlap_count": diagnostics.route_overlap_count,
                "route_overlap": diagnostics.route_overlap,
                "listing_coherence": diagnostics.listing_coherence,
                "mode_coherence": diagnostics.mode_coherence,
                "mode_count": diagnostics.mode_count,
                "largest_mode_share": diagnostics.largest_mode_share,
                "effective_mode_count": diagnostics.effective_mode_count,
                "duplicate_warning": diagnostics.duplicate_warning,
                "lexical_token_coverage": diagnostics.lexical_token_coverage,
                "lexical_mean_normalized_idf": (diagnostics.lexical_mean_normalized_idf),
                "hard_filter_relaxed": diagnostics.hard_filter_relaxed,
            },
        },
        "search_belief": _search_belief_payload(run.search_belief),
    }


def _coherence_payload(value: object) -> dict[str, object]:
    return {
        "n": value.n,
        "resultant_length": value.resultant_length,
        "debiased_pairwise_cosine": value.debiased_pairwise_cosine,
        "available": value.available,
        "reason": value.reason,
    }


def _mask_payload(resolution: object) -> dict[str, object]:
    from shopping_copilot.retrieval import ResolvedHardMask

    assert type(resolution) is ResolvedHardMask
    return {
        "eligible_count": len(resolution.eligible_parent_asins),
        "hard_filter_relaxed": resolution.hard_filter_relaxed,
        "relaxed_preference_ids": [item.preference_id for item in resolution.relaxed_constraints],
        "trace": [
            {
                "preference_id": item.preference_id,
                "facet": item.facet,
                "operator": item.operator.value,
                "before_count": item.before_count,
                "matched_count": item.matched_count,
                "after_count": item.after_count,
                "disposition": item.disposition.value,
                "reason": item.reason,
            }
            for item in resolution.trace
        ],
    }


def _compiled_payload(compiled: object) -> dict[str, object]:
    from shopping_copilot.query_compiler import CompiledQuery

    assert type(compiled) is CompiledQuery
    return {
        "schema": compiled.schema,
        "compiler_version": compiled.compiler_version,
        "catalog_id": compiled.catalog_id,
        "catalog_semantic_release_id": compiled.catalog_semantic_release_id,
        "category_graph_id": compiled.category_graph_id,
        "intent_version": compiled.intent_version,
        "search_ready": compiled.search_ready,
        "q_lex": compiled.q_lex,
        "q_sem": compiled.q_sem,
        "hard_constraints": [
            {
                "preference_id": item.preference_id,
                "facet": item.facet,
                "operator": item.operator.value,
                "value": list(item.value) if type(item.value) is tuple else item.value,
                "policy": item.policy.value,
            }
            for item in compiled.hard_constraints
        ],
        "ranking_preferences": [
            {
                "preference_id": item.preference_id,
                "facet": item.facet,
                "operator": None if item.operator is None else item.operator.value,
                "value": list(item.value) if type(item.value) is tuple else item.value,
                "semantic_text": item.semantic_text,
                "semantic_polarity": (
                    None if item.semantic_polarity is None else item.semantic_polarity.value
                ),
                "commitment": item.commitment.value,
                "source": item.source.value,
                "reason": item.reason.value,
            }
            for item in compiled.ranking_preferences
        ],
        "dont_care_facets": list(compiled.dont_care_facets),
        "directives": {
            "diversity": compiled.directives.diversity.value,
            "comparison_requested": compiled.directives.comparison_requested,
            "explanation_requested": compiled.directives.explanation_requested,
        },
        "requires_clarification": compiled.requires_clarification,
        "clarification_reason": compiled.clarification_reason,
        "trace": [
            {
                "preference_id": item.preference_id,
                "targets": [target.value for target in item.targets],
                "reason": item.reason,
            }
            for item in compiled.trace
        ],
    }


def _session_context_payload(
    context: SessionContext,
    registry: object,
) -> dict[str, object]:
    """Encode the actual typed SessionContext used by the audit runner."""

    payload = json.loads(encode_snapshot(context, registry).decode("utf-8"))
    assert type(payload) is dict
    return cast(dict[str, object], payload)


def _advance_session_context(
    context: SessionContext,
    *,
    turn: PromptTurn,
    shown: tuple[ShownProductView, ...],
    resolved: ResolvedTurnIntent,
    search_belief: SearchBelief | None,
) -> SessionContext:
    question = _assistant_question(turn.ask_attribute)
    record = TurnRecord(
        turn=turn.turn,
        user_message=turn.user_message,
        intent_version_before=context.state.intent.version,
        accepted_update=resolved.update,
        intent_version_after=resolved.final_intent.version,
        assistant_message=(
            question
            if question is not None
            else "[evaluation stops after Query Understanding and Probe]"
        ),
        question=question,
        question_key=(
            None if turn.ask_attribute is None else f"ask_attribute:{turn.ask_attribute}"
        ),
        ask_attribute=turn.ask_attribute,
        shown_product_ids=tuple(product_id for item in shown for product_id in item.product_ids),
        feedback=resolved.feedback,
        search_belief_probe_id=(
            None if search_belief is None else search_belief.certainty_evidence.probe_id
        ),
    )
    return SessionContext(
        session_id=context.session_id,
        profile=context.profile,
        state=SessionState(
            intent=resolved.final_intent,
            interaction=InteractionContext(turns=(*context.state.interaction.turns, record)),
            search_belief=search_belief,
        ),
    )


def _assistant_question(ask_attribute: str | None) -> str | None:
    if ask_attribute is None:
        return None
    if ask_attribute == "other":
        return "What other requirements or preferences matter to you?"
    return f"What {ask_attribute} requirements or preferences matter to you?"


def _resolved_payload(resolved: ResolvedTurnIntent) -> dict[str, object]:
    return {
        "update": _state_update_payload(resolved.update),
        "final_intent": _intent_payload(resolved.final_intent),
        "feedback": [_feedback_payload(item) for item in resolved.feedback],
        "directives": {
            "diversity": resolved.directives.diversity.value,
            "comparison_requested": resolved.directives.comparison_requested,
            "explanation_requested": resolved.directives.explanation_requested,
        },
        "clarification": {
            "needed": resolved.clarification.needed,
            "reason": resolved.clarification.reason,
            "alternatives": list(resolved.clarification.alternatives),
        },
        "trace": {
            "attempts": _attempt_payload(resolved.trace.attempts),
            "interpretation_summary": resolved.trace.interpretation_summary,
            "semantic_fallback_facets": list(resolved.trace.semantic_fallback_facets),
            "ignored_dont_care_facets": list(resolved.trace.ignored_dont_care_facets),
        },
    }


def _state_update_payload(update: object | None) -> dict[str, object] | None:
    if update is None:
        return None
    return {
        "turn": update.turn,
        "base_intent_version": update.base_intent_version,
        "operations": [_operation_payload(item) for item in update.operations],
    }


def _operation_payload(operation: object) -> dict[str, object]:
    op = operation.op
    if op == "add_preference":
        return {"op": op, "preference": _preference_payload(operation.preference)}
    if op == "replace_facet":
        return {
            "op": op,
            "facet": operation.facet,
            "preferences": [_preference_payload(item) for item in operation.preferences],
        }
    if op == "remove_preference":
        return {"op": op, "preference_ids": list(operation.preference_ids)}
    if op == "clear_facet":
        return {"op": op, "facet": operation.facet}
    if op == "set_dont_care":
        return {"op": op, "facet": operation.facet}
    if op == "switch_goal":
        return {
            "op": op,
            "new_goal": operation.new_goal,
            "carry_preference_ids": list(operation.carry_preference_ids),
        }
    raise AssertionError(f"unknown state operation: {op!r}")


def _feedback_payload(item: ProductFeedback) -> dict[str, object]:
    return {
        "product_ids": list(item.product_ids),
        "signal": item.signal.value,
        "compared_to_ids": list(item.compared_to_ids),
        "evidence_text": item.evidence_text,
    }


def _search_belief_payload(belief: SearchBelief) -> dict[str, object]:
    evidence = belief.certainty_evidence
    return {
        "based_on_intent_version": belief.based_on_intent_version,
        "certainty": belief.certainty,
        "certainty_method": belief.certainty_method,
        "certainty_evidence": {
            "probe_id": evidence.probe_id,
            "probe_size": evidence.probe_size,
            "raw_concentration": evidence.raw_concentration,
            "quality_status": evidence.quality_status.value,
            "quality_reasons": list(evidence.quality_reasons),
        },
        "candidate_modes": [
            {
                "id": item.id,
                "label": item.label,
                "mass": item.mass,
                "representative_ids": list(item.representative_ids),
            }
            for item in belief.candidate_modes
        ],
        "facet_stats": [
            {
                "facet": item.facet,
                "entropy": item.entropy,
                "coverage": item.coverage,
                "top_values": [
                    {"value": value.value, "mass": value.mass} for value in item.top_values
                ],
            }
            for item in belief.facet_stats
        ],
    }


def _intent_payload(intent: IntentState) -> dict[str, object]:
    return {
        "goal": intent.goal,
        "version": intent.version,
        "preferences": [_preference_payload(item) for item in intent.preferences],
        "dont_care_facets": sorted(_facet_alias(item) for item in intent.dont_care_facets),
    }


def _preference_payload(item: Preference) -> dict[str, object]:
    relation = (
        item.operator.value
        if item.operator is not None
        else f"semantic_{item.semantic_polarity.value}"
    )
    return {
        "id": item.id,
        "facet": _facet_alias(item.facet),
        "canonical_facet": item.facet,
        "relation": relation,
        "operator": None if item.operator is None else item.operator.value,
        "value": list(item.value) if type(item.value) is tuple else item.value,
        "semantic_text": item.semantic_text,
        "semantic_polarity": (
            None if item.semantic_polarity is None else item.semantic_polarity.value
        ),
        "strength": item.commitment.value,
        "source": item.source.value,
        "source_turn": item.source_turn,
        "evidence_text": item.evidence_text,
        "interpretation_confidence": item.interpretation_confidence,
    }


def _facet_alias(facet: str | None) -> str | None:
    return "category" if facet == SYSTEM_PRODUCT_CATEGORY_FACET_ID else facet


def _attempt_payload(attempts: tuple[ProviderTrace, ...]) -> list[dict[str, object]]:
    return [
        {
            "response_id": item.response_id,
            "model": item.model,
            "prompt_tokens": item.prompt_tokens,
            "completion_tokens": item.completion_tokens,
            "total_tokens": item.total_tokens,
        }
        for item in attempts
    ]


def _base_record(
    suite: PromptSuite,
    conversation: PromptConversation,
    turn: PromptTurn,
) -> dict[str, object]:
    scenario = (
        None if conversation.provenance is None else conversation.provenance.get("scenario_type")
    )
    return {
        "suite_id": suite.suite_id,
        "cohort": suite.cohort,
        "conversation_id": conversation.identifier,
        "turn": turn.turn,
        "tier": conversation.tier,
        "tags": list(conversation.tags),
        "scenario_type": scenario,
        "response_shape": turn.response_shape,
        "user_message": turn.user_message,
    }


def _error_record(
    suite: PromptSuite,
    conversation: PromptConversation,
    turn: PromptTurn,
    *,
    stage: str,
    elapsed_ms: float,
    error: Exception,
    resolved_intent: dict[str, object] | None = None,
    compiled: dict[str, object] | None = None,
    session_context_before: dict[str, object] | None = None,
    session_context_after: dict[str, object] | None = None,
    deepseek_request_payload: dict[str, object] | None = None,
    resolved_turn: dict[str, object] | None = None,
) -> dict[str, object]:
    if isinstance(error, QueryUnderstandingError):
        error_payload: dict[str, object] = {
            "type": type(error).__name__,
            "code": error.code.value,
            "path": list(error.path),
            "details": {key: value for key, value in error.details},
        }
    else:
        error_payload = {
            "type": type(error).__name__,
            "message": str(error),
        }
    return {
        **_base_record(suite, conversation, turn),
        "status": "error",
        "stage": stage,
        "qu_latency_ms": round(elapsed_ms, 3),
        "qu_attempts": [],
        "critical_semantic_pass": False if suite.cohort == "natural" else None,
        "session_context_before": session_context_before,
        "session_context_after": session_context_after,
        "deepseek_request_payload": deepseek_request_payload,
        "resolved_turn": resolved_turn,
        "final_intent": resolved_intent,
        "compiled": compiled,
        "mask": None,
        "probe": None,
        "error": error_payload,
    }


def _skipped_record(
    suite: PromptSuite,
    conversation: PromptConversation,
    turn: PromptTurn,
) -> dict[str, object]:
    return {
        **_base_record(suite, conversation, turn),
        "status": "skipped_after_failure",
        "critical_semantic_pass": False if suite.cohort == "natural" else None,
        "final_intent": None,
        "compiled": None,
        "mask": None,
        "probe": None,
        "error": None,
    }


def _shown_product_views(
    suite: PromptSuite,
    conversation: PromptConversation,
    turn: PromptTurn,
) -> tuple[ShownProductView, ...]:
    return tuple(
        ShownProductView(
            ref=f"product_{index}",
            product_ids=(
                f"fixture/{suite.suite_id}/{conversation.identifier}/{turn.turn}/{index}",
            ),
            label=product.label,
        )
        for index, product in enumerate(turn.shown_products)
    )


def _select_conversations(
    suite: PromptSuite,
    *,
    tier: str,
    limit: int | None,
    simulator_limit_per_scenario: int | None,
) -> tuple[PromptConversation, ...]:
    selected = (
        suite.conversations
        if tier == "full"
        else tuple(item for item in suite.conversations if item.tier == "smoke")
    )
    if simulator_limit_per_scenario is not None and suite.cohort == "simulator":
        scenario_counts: Counter[str] = Counter()
        balanced: list[PromptConversation] = []
        for conversation in selected:
            scenario = (
                None
                if conversation.provenance is None
                else conversation.provenance.get("scenario_type")
            )
            if not isinstance(scenario, str):
                raise ValueError(
                    f"simulator conversation has no scenario: {conversation.identifier}"
                )
            if scenario_counts[scenario] >= simulator_limit_per_scenario:
                continue
            scenario_counts[scenario] += 1
            balanced.append(conversation)
        selected = tuple(balanced)
    return selected if limit is None else selected[:limit]


def _selected_turns(
    conversation: PromptConversation,
    *,
    max_turn: int | None,
) -> tuple[PromptTurn, ...]:
    if max_turn is None:
        return conversation.turns
    return tuple(turn for turn in conversation.turns if turn.turn <= max_turn)


def _load_suites(args: argparse.Namespace) -> tuple[PromptSuite, ...]:
    paths = []
    if args.cohort in {"natural", "all"}:
        paths.append(args.natural_suite)
    if args.cohort in {"simulator", "all"}:
        paths.append(args.simulator_suite)
    return tuple(load_prompt_suite(path) for path in paths)


def _progress(completed: int, total: int, started: float) -> None:
    if completed == total or completed % 5 == 0:
        elapsed = time.perf_counter() - started
        print(f"progress {completed}/{total} ({elapsed:.1f}s)", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural-suite", type=Path, default=DEFAULT_NATURAL_SUITE)
    parser.add_argument("--simulator-suite", type=Path, default=DEFAULT_SIMULATOR_SUITE)
    parser.add_argument("--cohort", choices=("natural", "simulator", "all"), default="all")
    parser.add_argument("--tier", choices=("smoke", "full"), default="full")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--max-turn",
        type=int,
        default=None,
        help="run at most this many turns from each selected conversation",
    )
    parser.add_argument(
        "--simulator-limit-per-scenario",
        type=int,
        default=None,
        help="select at most this many simulator conversations from each scenario",
    )
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--dense-index", type=Path, default=DEFAULT_DENSE_INDEX)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/retrieval/qu-to-probe-full-v1.json"),
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_turn is not None and args.max_turn < 1:
        parser.error("--max-turn must be positive")
    if args.simulator_limit_per_scenario is not None and args.simulator_limit_per_scenario < 1:
        parser.error("--simulator-limit-per-scenario must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
