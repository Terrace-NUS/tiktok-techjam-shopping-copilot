# APERTURE

### Retrieval as Sensing for Conversational Shopping

APERTURE is a catalogue-grounded shopping copilot that treats a conversation as one
evolving decision. It uses retrieval not only to find products, but to sense how much
meaningful product space remains open—exploring while intent is uncertain and moving
toward precision as the decision takes shape.

> **Shopping intent is a journey, not a label.**

| Official evaluation | APERTURE | Official starter |
|---|---:|---:|
| Technical Score @200 | **0.9624** | 0.1067 |
| Hit@10 @200 | **1.0000** | 0.1250 |
| MRR @200 | **0.9842** | 0.0680 |

On our complementary natural-language benchmark, the full system reaches **1.000
Recall@10**, finds the target in **2.195 turns**, and achieves **87.4% Exploration
Satisfaction**. With the rest of APERTURE held fixed, our Intent Transparency algorithm
adds **36.5 percentage points** to Recall@10.

**[Why APERTURE](#why-aperture)** · **[System architecture](#system-architecture)** ·
**[Intent Transparency](#intent-transparency)** · **[Results](#evaluation)** ·
**[Run locally](#run-aperture)**

## Why APERTURE

Shopping rarely begins with a perfect query. A shopper may start with an occasion,
discover new possibilities, reject colours or materials, change categories, and only
gradually realize what they want. A language model can interpret those words, but it
cannot know from language alone whether the live catalogue contains one coherent
direction, several plausible directions, or mostly near-duplicate listings.

APERTURE closes that gap with three ideas:

1. **Explicit session state.** DeepSeek proposes typed intent operations; validated,
   deterministic code owns every state transition.
2. **Retrieval as sensing.** A fixed Catalogue Probe measures the product landscape,
   and our original Intent Transparency algorithm turns that evidence into a live
   search-control signal.
3. **Decision-aware retrieval.** The system adapts recall breadth and final-set
   diversity as intent evolves, while the current session always takes precedence over
   any long-term profile.

## System architecture

<p align="center">
  <img src="assets/aperture-architecture.png" width="760" alt="APERTURE architecture from natural language through explicit session state, catalogue probing, Intent Transparency, adaptive multi-route recall, evidence-grounded ranking, and profile precedence">
</p>

One turn moves through five auditable stages:

1. **Query Understanding** translates natural language into `add`, `revise`, `remove`,
   or `override` operations.
2. A validator and deterministic reducer commit one complete **Session Context**.
3. A fixed **Catalogue Probe** evaluates that state against the 50K-product catalogue.
4. **Intent Transparency** controls multi-route recall and final-set composition.
5. BGE and DeepSeek judge candidate fit; a transparency-aware selector returns the
   final Top-10 and the response is composed from grounded evidence.

The Session Context is authoritative. A supplied cross-session profile is treated only
as a weak prior, and individual product judgement remains grounded in current intent
and product evidence.

## Core system

### Explicit, inspectable session state

DeepSeek V4 Flash interprets natural language through native tool calls that propose
typed `add`, `revise`, `remove`, and `override` operations. The tool schema constrains
what may change; local validation checks whether the operation is legal; and a
deterministic reducer commits the result. Every turn produces a complete, replayable
Session Context instead of another opaque transcript summary.

### Probe before planning

Before choosing a retrieval strategy, APERTURE runs a fixed sensing pass over the same
catalogue on every turn. The Probe combines structured constraints, semantic
preferences, and catalogue density so the planner sees the product landscape—not just
the shopper's words. The complete query-to-catalogue similarity pass takes **39 ms for
50K products** on an RTX 4070 Ti.

### Intent Transparency

APERTURE generalizes `BUYING` and `BROWSING` from two fixed modes into a continuous,
catalogue-grounded control signal: **Intent Transparency** (`T_t`). It is our original
algorithm for measuring how much meaningful product space remains compatible with the
current Session Context.

For every product `i` and preference `c`, the Catalogue Probe computes fuzzy membership
`m_ic`. A Product of Experts combines active preferences, while local catalogue density
prevents near-duplicate listings from masquerading as genuinely different choices:

$$
N_t = \sum_{i \in \mathrm{catalog}}
\frac{\prod_c m_{ic}^{\lambda_c}}{d_i}
$$

The remaining intent volume is normalized against the complete catalogue:

$$
T_t = 1 - \frac{\log(1 + N_t)}{\log(1 + N_{\mathrm{catalog}})}
$$

`T_t` is a search policy signal, not an LLM confidence score. Low transparency opens
distinct recall directions and rewards meaningful difference across the final set.
Rising transparency searches more deeply around fewer directions and progressively
favours precise relevance.

### Adaptive recall and evidence-grounded ranking

The full runtime combines:

- hard-constraint masking before candidate generation;
- multi-centre semantic, lexical, and structured-facet recall;
- round-robin fusion that protects smaller semantic directions;
- a local BGE cross-encoder that reduces 300 candidates to 48;
- evidence-grounded DeepSeek product judgement; and
- a `T_t`-aware greedy DPP selector for the final Top-10.

Intent Transparency controls recall policy and final-set composition. BGE and DeepSeek
score individual product fit from the complete Session Context and grounded product
cards; the response layer may also use `T_t` to communicate whether the search is still
broad, narrowing, or focused.

### Profile precedence

APERTURE accepts a compact cross-session profile as optional context, but never lets it
overwrite the current decision. Browsing does not mutate the profile, and current
Session Context always wins when the two conflict. This keeps useful continuity
subordinate to what the shopper is asking for now.

## Evaluation

### Official benchmark: near-perfect target recovery

We evaluate APERTURE against the official BM25 starter on all **200 released
development sessions**, then repeat the same protocol on a **2,000-session public-like
stress test** built with the official data-generation method to check whether the
result holds at 10× scale.

| System | Hit@10 ↑ | MRR ↑ | MTTC ↓ | Technical Score ↑ |
|---|---:|---:|---:|---:|
| Baseline (official starter) @200 | 0.1250 | 0.0680 | 9.8100 | 0.1067 |
| **APERTURE @200** | **1.0000** | **0.9842** | **2.6450** | **0.9624** |
| Baseline (official starter) @2000 | 0.1400 | 0.0776 | 9.6405 | 0.1205 |
| **APERTURE @2000** | **0.9985** | **0.9565** | **2.9515** | **0.9472** |

On the released 200 sessions, APERTURE lifts Technical Score from **0.1067 to 0.9624**
and places every target in the Top-10. At 10× scale, it still reaches **1,997 / 2,000
targets** with a **0.9472** Technical Score. The expanded suite is an internal
public-like stress test—not organizer ground truth or an independent holdout.

### Beyond target recovery

The official simulator centres on recovering one hidden purchased item. We built a
complementary **200-journey, catalogue-grounded benchmark** to test natural shopper
language, rich product-specific semantics, and intent that evolves through
interaction.

Instead of reducing every product to a small predefined schema, each journey draws
**6–8 auditable, source-grounded preferences from an actual listing**—material,
construction, aesthetic, use case, capacity, fit, or whatever matters for that product.
DeepSeek turns progressively disclosed evidence into varied shopper language,
including refinements, exclusions, and changes of mind. Deterministic code owns the
hidden state, disclosure policy, and scoring.

During exploration, the Top-10 must remain relevant to what the shopper has expressed
and meaningfully diverse before another preference is revealed. The target ASIN is
never exposed to the evaluator.

| System | Recall@10 ↑ | Turns to Recall ↓ | Exploration Satisfaction ↑ |
|---|---:|---:|---:|
| Baseline (official starter) | 0.575 | 3.895 | 48.4% |
| APERTURE (without Intent Transparency) | 0.635 | 3.295 | 64.8% |
| **APERTURE** | **1.000** | **2.195** | **87.4%** |

Moving from the official starter to APERTURE without Intent Transparency already
improves performance under richer language, product semantics, and multi-turn state.
The final two rows isolate our core contribution: with the rest of APERTURE unchanged,
Intent Transparency adds **36.5 percentage points** to Recall@10, reaches the target
**1.10 turns earlier**, and raises Exploration Satisfaction by **22.6 percentage
points**.

It does not merely improve final retrieval; it changes how effectively the conversation
progresses from exploration to decision.

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

The 50,000-row catalogue and generated semantic/index artifacts are intentionally not
committed. The source catalogue remains immutable; embeddings, facets, indexes,
product cards, and density caches are reproducible sidecars.

## Run APERTURE

### Requirements

- Python 3.10 or later
- the official 50K catalogue at `data/catalog.jsonl`
- a CUDA-capable GPU for the measured real-world configuration
- a DeepSeek API key for Query Understanding and product judgement
- locally cached BGE embedding and reranking models for offline model loading

### Install

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install --require-hashes -r requirements/runtime.lock
python -m pip install -e ".[dev]"
```

Install the optional local retrieval stack for the full APERTURE pipeline:

```bash
python -m pip install --require-hashes -r requirements/retrieval.lock
python -m pip install -e ".[dev,retrieval]"
```

### Official Agent API

The default constructor selects the model-free official-simulator compatibility path.
The real system is an explicit opt-in and never silently falls back to the toy path.

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
```

The default real-world configuration expects these generated assets:

```text
artifacts/catalog-semantic/release-v0/
artifacts/retrieval/dense-v0/
artifacts/retrieval/intent-volume-density-v0.npz
```

To run the official local compatibility evaluator:

```bash
python -m evaluator.local_evaluator
```

## Verification

```bash
python -m pytest
python -m ruff check src tests/unit
python -m ruff format --check src tests/unit
python -m mypy
```

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
- Product facts must retain source references and quoted supporting evidence.
- Provider response IDs, API credentials, and private model metadata are excluded from
  released sidecars.
- Browsing alone does not mutate the supplied cross-session profile.
- Profile preferences remain subordinate to the current Session Context.

The catalogue is derived from
[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) by McAuley Lab, UCSD.
See [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for redistribution and dependency
attribution.
