# Runtime routing v1

## 1. Why there are two modes

The official simulator and the real shopping demo expose similar `reset` /
`respond` calls, but they are different environments:

- the official simulator follows fixed disclosure templates and rewards a small,
  deterministic retrieval policy;
- the real system accepts natural language and needs Query Understanding, Session
  Context, intent transparency, multi-route retrieval, and semantic ranking.

The repository therefore has two explicit execution modes behind one response
contract. It does not inspect prompts, sample IDs, target ASINs, or evaluator files
to guess which environment is running.

## 2. Selection contract

`starter.Agent()` defaults to the offline simulator specialist:

```python
from starter.agent import Agent

agent = Agent()
assert agent.mode == "official_simulator"
```

The full system requires an explicit mode and API credential:

```python
agent = Agent(
    mode="real_world",
    deepseek_api_key="...",
)
```

For non-default paths or model settings, construct `RealWorldConfig`:

```python
from shopping_copilot.application import RealWorldConfig
from starter.agent import Agent

agent = Agent(
    mode="real_world",
    real_world_config=RealWorldConfig(
        api_key="...",
        device="cuda",
        cross_encoder=True,
    ),
)
```

The API key is never read implicitly from an environment variable. Real-world mode
fails during construction when neither `deepseek_api_key` nor `real_world_config`
is supplied. It never silently falls back to the simulator strategy.

## 3. Runtime paths

```text
starter.Agent
├── official_simulator (default)
│   └── model-free catalog/state/ranking policy
└── real_world (explicit)
    └── DeepSeek QU
        → Session Context
        → query compilation
        → intent transparency
        → adaptive multi-route retrieval
        → optional BGE reranking
        → DPP selection
```

The default path imports no Torch, sentence-transformers, CUDA runtime, or API
transport. The expensive dependencies are loaded lazily only after `real_world`
has been selected.

Both delegates are normalized to the official response keys:

- `message`;
- `ask_attribute`;
- `recommendations`, represented as `{ "parent_asin": ... }` objects; and
- `usage.prompt_tokens` / `usage.completion_tokens`.

## 4. Product-card boundary

Real-world mode uses the raw 50k catalog projection by default. The enriched public
200 target bundle is diagnostic data and is never selected automatically. A
product-card sidecar can be supplied only through an explicit `RealWorldConfig`.

Neither runtime path may read public-set labels, hidden intent cards, evaluator
state, or a known target-ASIN pool.

## 5. Current implementation boundary

The model-free strategy lives under
`src/shopping_copilot/application/toy_simulator/`. The full runtime builder lives in
`src/shopping_copilot/application/real_world.py` and reuses the already-tested
`FullPipelineOtherAgent` implementation from the simulator integration runner.

The next architecture cleanup can move that agent class from the runner into the
application package. This does not change the public mode-selection contract.
