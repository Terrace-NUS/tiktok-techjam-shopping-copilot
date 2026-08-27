# Documentation

This directory separates frozen competition material, project research, and
normative engineering design. The official simulator is kept for API and score
regression; it is not treated as a complete model of conversational shopping.

## Engineering design

| Area | Module page |
| --- | --- |
| Session context | [`design/session_context/README.md`](design/session_context/README.md) |

## Project research

- [`意图确定度.md`](意图确定度.md): the result-aware intent-certainty thesis.
- [`8_26.md`](8_26.md): system architecture and evaluation research.

Research notes explain design motivation. They are not normative runtime
contracts.

## Official compatibility material

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
