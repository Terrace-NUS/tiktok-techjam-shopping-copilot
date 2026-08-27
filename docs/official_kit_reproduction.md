# Official Participant Kit Reproduction

## Purpose

The official participant kit is kept in this repository as a compatibility
harness for the published Agent API, catalog, session protocol, and scoring
formula.

It is not treated as evidence that a system satisfies the complete design
intent of Track 4. The local evaluator uses deterministic, synthetic customer
messages derived from target-product metadata. Results on the 200 public
sessions are regression results, not a general conversational-commerce
benchmark.

## Frozen provenance

- Repository: `TechJam2026/techjam-conversational-search`
- Git tag: `participant-kit`
- Git commit: `2a6cc8e776da66ce69b1cbd237838fbc43f32587`
- Release publication time: `2026-08-24T08:49:52Z`
- Participant ZIP SHA-256:
  `b3d7e283b835343b42c4919ea2ca90f2fb5a2aa2b10537f14dcf42f03e5b38ae`
- Catalog gzip SHA-256:
  `07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8`
- Locally decompressed catalog SHA-256:
  `da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`

The release ZIP omits `.gitignore` and `tests/`. Those files were restored
from the same frozen Git tag. At initial reproduction time, all official
tracked files were semantically identical to the tagged versions; byte-level
differences were Windows versus Unix line endings only. Project files may
subsequently evolve while this document preserves the imported baseline's
provenance and reproduced metrics.

## Local assets

The following official data is installed locally:

- `data/catalog.jsonl`: 50,000 products, 60,546,327 bytes
- `data/public_set.jsonl`: 200 labeled development sessions
- `data/releases/catalog.jsonl.gz`: verified release asset
- `data/releases/SHA256SUMS`: official checksums

`data/catalog.jsonl`, `data/releases/`, and evaluator output `results.json`
are intentionally ignored by Git.

## Reproduction commands

The official starter has no third-party Python dependencies.

```powershell
python -m unittest discover -s tests -v
python -m evaluator.local_evaluator
```

The evaluator writes full per-session output to `results.json`.

## Reproduced baseline

Verified on Windows with Python 3.13.12:

| Metric | Reproduced value |
| --- | ---: |
| Sessions | 200 |
| Hit Rate@10 | 0.125 |
| MRR | 0.068034 |
| MTTC | 9.81 |
| Efficiency | 0.119 |
| TechnicalScore | 0.106710 |

All three official evaluator unit tests pass. The complete local run took
18.303 seconds on the current machine; elapsed time is an environment
observation, not an official benchmark result.

## Compatibility boundary

Future implementation work must preserve these externally visible rules:

- Export `starter.agent.Agent`.
- Call `reset(session_id, user_profile)` before `respond(...)`.
- Support turns 1 through 10 with `top_k=10`.
- Return a string `message`, an allowed `ask_attribute` or `null`, and ordered
  recommendations.
- Recommend only catalog-valid `parent_asin` values. Only the first 10 unique,
  valid values are scored, using exact equality.
- Keep an offline-capable path because final scoring may disable network
  access.

The intended project structure is to keep `starter/agent.py` as a thin
TechJam adapter. State tracking, retrieval, routing, ranking, and evaluation
logic should live in independent project modules and should also be tested on
design-intent stress scenarios that do not copy the toy simulator's metadata
templates.
