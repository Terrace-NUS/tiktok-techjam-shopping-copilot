# TechJam Conversational E-Commerce Search Challenge

> Project development note: this repository vendors the frozen official kit as
> a compatibility harness. The project architecture, research notes, and
> implementation status are indexed in [`docs/README.md`](docs/README.md).
> Passing the local simulator is a regression requirement, not the definition
> of the product architecture.

## Project Development

The project package uses a Python 3.10+ `src` layout. Create an isolated
development environment and install the package in editable mode:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --require-hashes -r requirements/runtime.lock
python -m pip install -e ".[dev]"
```

The runtime lock pins the exact universal wheels and SHA-256 hashes used by
the catalog semantic builder. Development tools remain optional dependencies.

Run the development quality gates with:

```powershell
python -m pytest
python -m ruff check src tests/unit
python -m ruff format --check src tests/unit
python -m mypy
```

Profile the downloaded catalog without modifying it:

```powershell
python -m shopping_copilot.catalog.profiling `
  data/catalog.jsonl `
  artifacts/catalog-profile
```

The profiler and its generated artifacts are documented in
[`docs/design/catalog_semantic/README.md`](docs/design/catalog_semantic/README.md).
The same page documents the reviewed category/facet pipeline, the read-only
CS3 price index, the owner-approved Gate-B contract, the CS5 runtime grounding
boundary, and the CS6 self-contained release. CS7 now binds that release to an
atomic session Gateway/store and a strict release-pinned snapshot envelope.
An experimental full-catalog Dense R0 route is implemented. Query Understanding
now has a complete-state typed DeepSeek function-call adapter, wide structured
facets, deterministic materialization, and Gateway preview. A deterministic
Query Compiler produces lexical, semantic, hard-constraint, ranking, directive,
and trace views. The earlier fixed Top-80 mode-coherence `C_t` remains as a
compatibility probe, while the story-facing runtime now measures full-catalog
Fuzzy Intent Volume: structured evidence and semantic preferences form a
Product of Experts, duplicate listings are density-discounted, and the result
is exposed as `T_t` with separate health diagnostics `D_t`. Hard-mask-first
Dense/Lexical/Facet retrieval, RRF, and `T_t`-aware vector MMR are implemented.
Relative-score fusion, Qwen/BGE cross-encoders, vector DPP, and latent xQuAD
now have a same-pool 50k evaluation; selecting a new production ranking policy,
application orchestration, and asking remain downstream steps. See
[`docs/design/query_understanding/README.md`](docs/design/query_understanding/README.md).

Install the optional local embedding stack, build the 50k dense index, and run
the component evaluation with:

```powershell
python -m pip install --require-hashes -r requirements/retrieval.lock
python -m pip install -e ".[dev,retrieval]"

retrieval-dense build `
  artifacts/catalog-semantic/release-v0 `
  artifacts/retrieval/dense-v0

python scripts/retrieval/evaluate_first_turn.py `
  --dense-factory shopping_copilot.retrieval:create_dense_retriever `
  --dense-index artifacts/retrieval/dense-v0 `
  --semantic-release artifacts/catalog-semantic/release-v0 `
  --output artifacts/retrieval/first-turn-evaluation.json

python scripts/retrieval/evaluate_clarity_prompts.py `
  --dense-factory shopping_copilot.retrieval:create_dense_retriever `
  --dense-index artifacts/retrieval/dense-v0 `
  --semantic-release artifacts/catalog-semantic/release-v0 `
  --output artifacts/retrieval/clarity-evaluation-v0.json

python scripts/retrieval/evaluate_transparency_v1.py

python scripts/retrieval/evaluate_intent_volume_runtime_v1.py --device cpu
```

The design boundary, pinned encoder, measured recall, and current Probe
limitations are recorded in
[`docs/design/retrieve/dense-r0.md`](docs/design/retrieve/dense-r0.md). The
target-free real-world clarity prompts, fixed statistical gate, and first
result are recorded in
[`docs/design/retrieve/clarity-evaluation-v0.md`](docs/design/retrieve/clarity-evaluation-v0.md).
The legacy Probe/C_t/D_t contract and its held-out V1 audit are recorded
in [`docs/design/retrieve/probe-v1.md`](docs/design/retrieve/probe-v1.md) and
[`docs/design/retrieve/transparency-evaluation-v1.md`](docs/design/retrieve/transparency-evaluation-v1.md).
The current story-facing `T_t` runtime contract, frozen hackathon parameters,
unavailable semantics, transition labels, and real-session demo are documented
in [`docs/design/intent_purity/runtime-contract-v1.md`](docs/design/intent_purity/runtime-contract-v1.md).

The organizer-facing evaluator remains a separate compatibility regression
and continues to run with `python -m evaluator.local_evaluator`.

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Starter

Python 3.10 or later is recommended. The starter uses only the Python standard library.

```bash
python3 -m evaluator.local_evaluator
```

Keep product and domain logic under the planned `src/shopping_copilot/`
package. `starter/agent.py` is the official API adapter and composition root;
keep it thin. Do not edit the evaluator or public labels when reporting your
local score.
The command writes per-session results and aggregate metrics to `results.json`.

The included weak BM25 starter scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

`TechnicalScore` is an objective input to the `Technical Execution` judging
criterion. It is not a separate judging criterion and does not represent the
entire `Technical Execution` score.

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer does not provide or reimburse model API credits; teams are responsible for costs from optional external services.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  official adapter and current weak baseline
evaluator/local_evaluator.py      public-set simulator and scorer
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`

Organizer-only runbooks and private-evaluation material are intentionally not
vendored in this participant repository.

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
