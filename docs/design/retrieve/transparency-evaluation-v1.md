# Intent Transparency V1 Evaluation

- Date: **2026-08-29**
- Result: **passed the frozen V1 hackathon gate**
- Probe: fixed Top-80, semantic-mode leader threshold `0.94`
- Data: 24 target-free prompt families, 12 calibration and 12 held-out audit

> Scope note (2026-08-29): this gate used hand-written `q_lex` / `q_sem` pairs
> to test Probe in isolation. The later 200-turn live DeepSeek QU evaluation
> found that the full system does **not** yet produce monotonic `C_t` growth as
> users add details. See
> [QU → Probe 真实全链路评测](./qu-to-probe-evaluation-v1.md). Therefore
> “passed” below applies only to this isolated Probe gate, not to the stronger
> end-to-end intent-transparency claim.

## What was tested

Each family describes the same shopping task twice:

- a vague request that still leaves several product directions open;
- a specific request that narrows the task to one concrete product region.

The prompts were written before retrieval scores were run. They contain no
official hidden target, target ASIN, expected product, simulator label, or
ranking annotation. The calibration and audit families use different prompt
wording.

For every prompt the evaluator ran:

```text
hand-written q_lex -> fixed Lexical Top-80 observation
hand-written q_sem -> one exact Dense Top-80
                   -> fixed-leader semantic modes
                   -> equal-mode coherence G_mode
```

## Calibration correction discovered by the blind run

The first attempted calibration used the median vague score as `g_low` and the
median specific score as `g_high`. It was rejected: unrelated product domains
have different natural vector geometry, so cross-category marginal medians can
reverse even when most within-task pairs move in the correct direction.

The Probe algorithm and audit prompts were not edited. The scale definition was
corrected to use the 10th and 90th percentiles of all available calibration
coherences:

| Anchor | Value |
| --- | ---: |
| `g_low` (P10) | `0.2569630265` |
| `g_high` (P90) | `0.4483984915` |

This calibrates the observed concentration range. The vague/specific labels are
used only for paired directional falsification.

## Held-out audit result

| Check | Result | Gate |
| --- | ---: | ---: |
| Available paired mode coherence | `12 / 12` | `100%` required |
| Specific greater than vague | `9 / 12` | `>= 70%` required |
| Strict paired direction rate | `0.75` | pass |
| Median paired improvement | `+0.0213248041` | `> 0` required |
| Overall gate | **pass** | all checks |

The result does not claim that every extra adjective must increase `C_t`, nor
that `C_t` is a probability. It supports the hackathon claim that fixed
semantic-mode dispersion usually detects a real narrowing of the catalog
region, while preserving visible counterexamples instead of hiding them.

## Frozen runtime mapping

```text
C_t = clip((G_mode - 0.2569630265)
           / (0.4483984915 - 0.2569630265), 0, 1)
```

Counts, lexical coverage, route overlap, filter relaxation, and duplicate
warnings remain in `D_t`; none enters this formula.

The machine-readable report is reproducibly generated at
`artifacts/retrieval/transparency-v1/report.json` and deliberately ignored by
Git. The bound runtime configuration is
[`config/retrieval/transparency-calibration-v1.json`](../../../config/retrieval/transparency-calibration-v1.json).

Reproduce the run with:

```powershell
.\.venv-3.10\Scripts\python.exe scripts\retrieval\evaluate_transparency_v1.py
```
