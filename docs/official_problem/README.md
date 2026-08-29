# Formal Track 4 Reference

This directory contains the public TikTok TechJam 2026 information document.
Track 4, **Shopping Copilot: AI Conversational Search and Recommendations**, is
on PDF pages 38–43.

## Source precedence

When project documents differ, use this order:

1. the formal public problem statement for challenge scope, limits,
   deliverables, and judging criteria;
2. the machine-readable participant kit for the Agent API and deterministic
   evaluator behavior;
3. the early-bird export only as historical context;
4. project research and design documents for our implementation choices.

The formal document points to the same
`TechJam2026/techjam-conversational-search` repository and `participant-kit`
release already reproduced locally. The release remains the compatibility
baseline. Later documentation-only changes on the official `main` branch
clarify that:

- `TechnicalScore` is an objective input to the 35% `Technical Execution`
  criterion, not the whole criterion or the whole judging result; and
- the organizer does not provide or reimburse optional model API credits.

## Formal Track 4 requirements

The system is expected to demonstrate:

- separate Buying and Browsing routes;
- in-memory multi-route retrieval combining keyword, category, and vector
  similarity, followed by semantic ranking;
- multi-turn accumulation plus explicit intent-override handling;
- proactive clarification when an over-broad request leaves too many
  candidates;
- short-term session state and safe use of the provided aggregate profile;
- adaptive orchestration that can change strategy as the conversation evolves;
- evaluation through Hit Rate@10, MRR, MTTC, Efficiency, and the combined
  `TechnicalScore`.

Hard boundaries include:

- at most 10 turns per session; exceeding the limit forces termination and a
  zero score;
- a strictly read-only catalog with no mock ASIN injection;
- text dialogs and structured/text catalog data only;
- an entirely in-memory, lightweight retrieval path rather than a heavy
  external vector database cluster;
- backend/headless evaluation, so UI work is out of scope; and
- static catalog, pricing, and category trees during the competition.

## Engineering interpretation

The official simulator remains an API and score regression harness, not the
definition of the product architecture. Domain state, catalog semantics,
retrieval, routing, and ranking live under `src/shopping_copilot/`; the adapter
under `starter/` stays thin.

Missing catalog fields are unknown evidence, not negative facts. In
particular, a missing price must not by itself mean that a product violates a
budget. Whether a reviewed price fact may become a hard retrieval constraint
is decided later from resolved statistics and false-negative tests.
