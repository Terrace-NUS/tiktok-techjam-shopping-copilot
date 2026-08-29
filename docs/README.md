# Documentation

This directory separates frozen competition material, project research, and
normative engineering design. The official simulator is kept for API and score
regression; it is not treated as a complete model of conversational shopping.

## Team briefing

- [`team_briefing/README.md`](team_briefing/README.md): presentation-oriented
  walkthrough of Session Context, facet construction/extraction, Query
  Understanding, and the Fuzzy Intent Volume / $T_t$ / $D_t$ stage.

## Engineering design

| Area | Module page |
| --- | --- |
| Catalog semantic layer | [`design/catalog_semantic/README.md`](design/catalog_semantic/README.md) |
| Session context | [`design/session_context/README.md`](design/session_context/README.md) |
| Query understanding | [`design/query_understanding/README.md`](design/query_understanding/README.md) |
| Query compiler | [`design/query_compiler/README.md`](design/query_compiler/README.md) |
| Retrieval working contract | [`design/retrieve/contract-v0.md`](design/retrieve/contract-v0.md) |
| Dense Retrieval R0 experiment | [`design/retrieve/dense-r0.md`](design/retrieve/dense-r0.md) |
| Real-world clarity audit V0 | [`design/retrieve/clarity-evaluation-v0.md`](design/retrieve/clarity-evaluation-v0.md) |
| Fixed multi-view Probe and C_t/D_t | [`design/retrieve/probe-v1.md`](design/retrieve/probe-v1.md) |
| Retrieval evidence and hard-mask resolver | [`design/retrieve/evidence-hard-mask-v0.md`](design/retrieve/evidence-hard-mask-v0.md) |
| Intent-transparency held-out audit | [`design/retrieve/transparency-evaluation-v1.md`](design/retrieve/transparency-evaluation-v1.md) |
| Fuzzy Intent Volume runtime v1 | [`design/intent_purity/runtime-contract-v1.md`](design/intent_purity/runtime-contract-v1.md) |

## Project research

- [`意图确定度.md`](意图确定度.md): the result-aware intent-certainty thesis.
- [`8_26.md`](8_26.md): system architecture and evaluation research.

Research notes explain design motivation. They are not normative runtime
contracts.

## Official compatibility material

- [`official_problem/README.md`](official_problem/README.md): formal Track 4
  scope, constraints, source precedence, and local engineering implications.
- [`official_problem/TikTok TechJam 2026 Information Document.pdf`](<official_problem/TikTok TechJam 2026 Information Document.pdf>):
  formal public information document; Track 4 is on PDF pages 38–43.
- [`early_bird_problem_set.md`](early_bird_problem_set.md)
- [`competition_specification.md`](competition_specification.md)
- [`agent_api_contract.json`](agent_api_contract.json)
- [`evaluation_config.json`](evaluation_config.json)
- [`submission_rules.md`](submission_rules.md)
- [`baseline_results.json`](baseline_results.json)
- [`official_kit_reproduction.md`](official_kit_reproduction.md)

The official adapter remains under `starter/`; the official evaluator remains
under `evaluator/`. Product-domain code must not be implemented inside either
directory.
