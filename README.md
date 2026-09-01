# APERTURE

### Retrieval as Sensing for Conversational Shopping

APERTURE is a catalogue-grounded shopping copilot that treats a conversation as one
evolving decision. It turns natural shopper language into an explicit Session Context,
probes the live catalogue to compute **Intent Transparency**, and uses that signal to
control multi-route retrieval and final-set composition.

> **Shopping intent is a journey, not a label.**

<p align="center">
  <img src="assets/aperture-architecture.png" width="760" alt="APERTURE architecture from natural language through explicit session state, catalogue probing, Intent Transparency, adaptive multi-route recall, evidence-grounded ranking, and profile precedence">
</p>

## Results at a glance

| Evaluation | Result |
|---|---:|
| Official 200-session Technical Score | **0.9624** |
| Official 200-session Hit@10 | **1.0000** |
| 2,000-session public-like stress test | **1,997 / 2,000 targets recalled** |
| Natural-language benchmark Recall@10 | **1.000** |
| Exploration Satisfaction | **87.4%** |
| Intent Transparency ablation lift | **+36.5 pp Recall@10** |

The 2,000-session suite is an internal public-like stress test, not organizer ground
truth or an independent holdout. Full methodology and interpretation are provided in
the project submission; this README focuses on running and verifying the repository.

## Reproduce the repository

### 1. Requirements

- Python 3.10 or later
- the official 50K catalogue at `data/catalog.jsonl`
- a CUDA-capable GPU for the full retrieval pipeline
- a DeepSeek API key for Query Understanding and product judgement
- locally available BGE embedding and reranking model weights

The default `Agent()` path is model-free and implements the official simulator
interface. The complete APERTURE runtime is an explicit opt-in and never silently
falls back to the simulator path.

### 2. Install

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/Terrace-NUS/tiktok-techjam-shopping-copilot.git
cd tiktok-techjam-shopping-copilot
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
```

Install the locked runtime and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install --require-hashes -r requirements/runtime.lock
python -m pip install -e ".[dev]"
```

For the full APERTURE retrieval pipeline, also install the locked local-model stack:

```bash
python -m pip install --require-hashes -r requirements/retrieval.lock
python -m pip install -e ".[dev,retrieval]"
```

### 3. Add the competition catalogue

Place the immutable organizer catalogue here:

```text
data/catalog.jsonl
```

The expected file contains 50,000 product rows. APERTURE never rewrites it; product
facts, embeddings, semantic facets, indexes, and density caches are separate sidecar
assets under the ignored `artifacts/` tree.

### 4. Run the official compatibility evaluator

The released 200-session development set is checked in at `data/public_set.jsonl`.
Run the organizer-compatible, model-free path with:

```bash
python -m evaluator.local_evaluator
```

The evaluator writes `results.json` in the repository root. Override paths when
needed:

```bash
python -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --output results.json
```

## Official Agent interface

The competition entry point is `starter/agent.py` and exposes the required
`reset(...)` and `respond(...)` methods.

### Model-free official-simulator path

```python
from starter.agent import Agent

agent = Agent()
agent.reset("demo-session", {})
response = agent.respond(
    "demo-session",
    "I need something understated for commuting.",
    turn=1,
    top_k=10,
)
```

### Full APERTURE path

```python
import os

from starter.agent import Agent

agent = Agent(
    mode="real_world",
    deepseek_api_key=os.environ["DEEPSEEK_API_KEY"],
)

agent.reset("demo-session", {})
response = agent.respond(
    "demo-session",
    "I need something understated for commuting with a 15-inch laptop.",
    turn=1,
    top_k=10,
)
audit = agent.last_audit("demo-session")
```

Both modes return the organizer response contract:

```text
message
ask_attribute
recommendations[].parent_asin
recommendations[].score
usage.prompt_tokens
usage.completion_tokens
```

## Full-pipeline assets

The real-world constructor validates every dependency before serving a turn. Its
default paths are:

| Asset | Default path |
|---|---|
| Immutable catalogue | `data/catalog.jsonl` |
| Semantic catalogue release | `artifacts/catalog-semantic/release-v0/` |
| Dense BGE index | `artifacts/retrieval/dense-v0/` |
| Intent-volume density cache | `artifacts/retrieval/intent-volume-density-v0.npz` |
| Optional product cards | `data/product_fact_cards/deepseek_7011_v1/product-facts.jsonl.gz` |

To rerun the complete multi-turn public-simulator protocol after preparing those
assets:

```bash
python scripts/simulator/evaluate_full_pipeline_other.py \
  --catalog data/catalog.jsonl \
  --semantic-release artifacts/catalog-semantic/release-v0 \
  --dense-index artifacts/retrieval/dense-v0 \
  --density-cache artifacts/retrieval/intent-volume-density-v0.npz \
  --product-card-sidecar data/product_fact_cards/deepseek_7011_v1/product-facts.jsonl.gz \
  --api-key-file path/to/deepseek-key.txt \
  --device cuda \
  --output-dir artifacts/evaluations/full-pipeline
```

The target ASIN remains on the evaluator side of the boundary. Every turn and completed
session is appended to JSONL, and interrupted runs can be continued with `--resume`.
Use `--limit` for a smoke run before starting the full evaluation.

## Query Understanding evaluation

Validate the checked-in prompt suites without making API calls:

```bash
python scripts/query_understanding/evaluate_prompts.py \
  --cohort all \
  --tier full \
  --validate-only
```

Replay the same suites against DeepSeek:

```bash
python scripts/query_understanding/evaluate_prompts.py \
  --cohort all \
  --tier full \
  --api-key-env DEEPSEEK_API_KEY \
  --output artifacts/evaluations/query-understanding.json
```

## Test and verify

Run the complete offline verification suite:

```bash
python -m pytest
python -m ruff check src tests/unit
python -m ruff format --check src tests/unit
python -m mypy
```

The release commit passes **1,139 tests**, Ruff linting and formatting, and strict mypy
checking across 172 source files.

## Repository layout

| Path | Purpose |
|---|---|
| `src/shopping_copilot/` | Session state, Query Understanding, catalogue semantics, Probe, retrieval, ranking, response, and benchmark logic |
| `starter/agent.py` | Thin adapter for the official Agent API |
| `evaluator/` | Frozen local compatibility evaluator |
| `scripts/` | Reproducible builders, evaluations, and benchmark runners |
| `config/` | Frozen runtime policies and prompt/evaluation fixtures |
| `data/public_set.jsonl` | 200 released development sessions |
| `data/product_fact_cards/` | Source-grounded product-fact sidecar assets |
| `requirements/` | Locked runtime and retrieval dependencies |
| `assets/` | Public project visuals |

## Models and technologies

- **DeepSeek API (`deepseek-v4-flash`)** for native-tool Query Understanding and
  evidence-grounded product judgement
- **BAAI/bge-small-en-v1.5** for dense catalogue embeddings
- **BAAI/bge-reranker-v2-m3** for cross-encoder relevance
- **PyTorch + CUDA** for full-catalogue vector computation
- **NumPy** for fusion, density correction, and DPP selection
- **Sentence Transformers** for local embedding and reranking models

## Data, privacy, and provenance

- Competition catalogue rows are never rewritten.
- Product facts retain source references and quoted supporting evidence.
- Provider response IDs, API credentials, and private model metadata are excluded from
  released sidecars.
- Browsing alone does not mutate the supplied cross-session profile.
- Current Session Context always takes precedence over profile preferences.

The catalogue is derived from
[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) by McAuley Lab, UCSD.
See [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for redistribution and dependency
attribution.
