# Simulator `other`-only C_t 实测（v1）

日期：2026-08-29

## 正确的 toy simulator 交互协议

官方 toy simulator 不理解 assistant 的自然语言内容。它生成下一条用户消息时，真正读取的是返回对象中的：

```json
{"ask_attribute": "other"}
```

当值为 `other` 时，simulator 会从尚未披露的 hard constraints 和 soft preferences 中最多返回两条。询问具体 attribute 时，它只依赖一个非常粗糙的关键词分类器，容易错误地返回“没有额外偏好”。

因此用于测试 `C_t` 收敛的主要交互固定为：

```text
buying / browsing task
  → ask_attribute="other"
  → simulator 披露最多两条剩余约束
  → ask_attribute="other"
  → simulator 再披露最多两条
  → ask_attribute="other"
  → 剩余信息耗尽时返回 no additional preference
```

不使用 `intent_override` 和 `boundary`。

## 生成的数据

我们重新运行官方 local evaluator，覆盖 public set 中全部：

| scenario | task | 可见 turn |
|---|---:|---:|
| buying | 80 | 320 |
| browsing | 80 | 320 |
| 合计 | 160 | 640 |

640 turn 的消息形状为：

| response shape | 数量 |
|---|---:|
| initial requirement | 80 |
| initial exploration | 80 |
| attribute disclosure | 320 |
| no additional preference | 160 |

完整 fixture 是 `config/query_understanding/simulator-other-prompts-v1.json`。它只保存参赛 Agent 可见的信息，不保存 target ASIN、intent card、user profile 或隐藏约束。

## 真实 QU → Probe 抽样

先从 buying 和 browsing 各取 8 个 task，执行 16 × 4 = 64 个真实 turn。每段对话共享连续的 Session Context。

| 项目 | 实测值 |
|---|---:|
| QU 成功 | 64 / 64 |
| 完整走到 Probe | 64 / 64 |
| `C_t` 可计算 | 59 / 64 |
| `C_t` unavailable | 5 / 64 |
| token | 203,970 |

`C_t` 的总体平均为 `0.341`，中位数为 `0.361`。总体值不是本次最重要的检查，关键是同一个 task 内新增信息后的变化方向。

## 收敛结果

### 初始状态 → 第一次 `other` 披露

15 个 task 的前后 `C_t` 都可计算：

| 变化 | task |
|---|---:|
| 上升 | 4 |
| 不变 | 0 |
| 下降 | 11 |
| 平均变化 | `-0.173` |
| 中位变化 | `-0.181` |

按 scenario：

| scenario | 可比较 | 上升 | 下降 | 平均变化 |
|---|---:|---:|---:|---:|
| buying | 7 | 3 | 4 | `-0.123` |
| browsing | 8 | 1 | 7 | `-0.217` |

### 第一次披露 → 第二次 `other` 披露

14 个 task 可比较：

| 变化 | task |
|---|---:|
| 上升 | 5 |
| 不变 | 5 |
| 下降 | 4 |
| 平均变化 | `-0.014` |

### 信息耗尽

第二次披露后，simulator 返回 “I don't have an additional preference for other.”：

| 变化 | task |
|---|---:|
| 上升 | 0 |
| 不变 | 14 |
| 下降 | 0 |

这一段行为正确：Session Context 没有变化，`q_sem` 和 `C_t` 也保持不变。

## 排除 QU 没有写入信息的可能

第一次披露：

- 16 / 16 的 `q_sem` 发生变化；
- 每个 task 增加 1–3 条 preference；
- 15 / 16 增加至少一条 hard constraint。

第二次披露：

- 15 / 16 的 `q_sem` 发生变化；
- 15 / 16 增加至少一条 preference；
- 8 / 16 增加至少一条 hard constraint。

信息耗尽轮：

- 16 / 16 的 `q_sem` 不变；
- 16 / 16 的 preference 数不变；
- 16 / 16 的 hard constraint 数不变。

因此第一次披露后 `C_t` 大量下降不能归因于 QU 忽略了 simulator 信息。新增信息确实进入了 Session Context 和 compiled query，失败发生在当前 `C_t` 的检索与度量方法。

## 结论

在符合官方 toy simulator 行为的交互协议下，当前 `C_t v1` 仍然没有体现故事要求的逐步收敛。特别是第一次得到明确的新约束后，11 / 15 的可比较 task 反而下降。

故事与 contract 不变：`C_t` 必须表示同一个购物目标下的意图透明度。下一版算法的验收必须至少重跑这 16 个 task，并将两次真实 disclosure 的方向性作为核心 gate。

## 结构性问题定位

我们用相同的已保存 QU 状态做了一个 2×2 反事实矩阵：

| 运行 | query | hard mask |
|---|---|---|
| previous | 上一轮 | 上一轮 |
| query-only | 当前轮 | 上一轮 |
| mask-only | 上一轮 | 当前轮 |
| full | 当前轮 | 当前轮 |

29 个 `G_mode` 可比较的 disclosure 中，full 链路有 15 次下降。对这 15 次下降：

| 诊断信号 | 次数 |
|---|---:|
| 只换新 query 也下降 | 13 |
| 只换新 hard mask 也下降 | 13 |
| 两者单独都会下降 | 11 |
| 两者单独都不下降、只有组合下降 | 0 |
| semantic-mode 合并把非下降方向翻成下降 | 0 |
| Top-K Jaccard `< 0.25` | 10 |

下降样本的平均 Top-K Jaccard 只有 `0.168`；当前 hard mask 后的候选数平均只剩上一轮的 `22.6%`。

因此问题不是标定、0/1 截断或 semantic-mode 去重。新增信息同时通过 q_sem 改写和 hard mask 大幅更换 Top-K；固定 Top-80 会从新的候选区域重新补满，而 `G_mode` 测量的是这批新商品自身的绝对相似度。前后两轮没有共享固定参照系，所以新增约束不保证 `G_mode` 收敛。

逐 transition 反事实结果：

- `artifacts/retrieval/transparency-transition-diagnosis-v1.json`
- `artifacts/retrieval/transparency-transition-diagnosis-v1.md`

## Top-K 深度排查

为了区分“Top-80 这个数字不合适”和“每轮重新截取 Top-K 的结构不合适”，我们保持同一批 QU 状态、hard mask、dense score 和 mode threshold，只改变 K：

| K | 可比较 disclosure | 上升 | 不变 | 下降 | 平均 `ΔG_mode` |
|---:|---:|---:|---:|---:|---:|
| 20 | 29 | 9 | 2 | 18 | `-0.0243` |
| 40 | 29 | 12 | 2 | 15 | `-0.0229` |
| 80 | 29 | 12 | 2 | 15 | `-0.0265` |
| 160 | 29 | 11 | 2 | 16 | `-0.0259` |
| 320 | 29 | 12 | 3 | 14 | `-0.0199` |

最关键的第一次 `other` 披露：

| K | 下降 / 可比较 | 平均 `ΔG_mode` | 平均前后 Top-K Jaccard |
|---:|---:|---:|---:|
| 20 | 12 / 15 | `-0.0560` | `0.096` |
| 40 | 10 / 15 | `-0.0530` | `0.117` |
| 80 | 11 / 15 | `-0.0546` | `0.140` |
| 160 | 11 / 15 | `-0.0502` | `0.175` |
| 320 | 10 / 15 | `-0.0369` | `0.189` |

增加 K 会提高一点候选重叠并减轻平均下降，但所有 K 仍然失败。结论是：**80 这个具体数值不是根因；每轮根据新 query 和新 mask 重新构造一个 Top-K 移动窗口，才是不可比性的来源。** 单纯把 80 调大或调小不能修复 `C_t`。

完整 K-sweep：

- `artifacts/retrieval/transparency-probe-depth-sweep-v1.json`
- `artifacts/retrieval/transparency-probe-depth-sweep-v1.md`

## Lexical Top-80 的 Dense coherence

为了检查 Dense 最近邻的选择偏差是否是唯一根因，我们进行了另一条 route 实验：

```text
q_lex
→ hard mask 后的 BM25 Top-80
→ 读取这些商品现有的 Dense document vectors
→ 使用相同的 0.94 semantic-mode 合并
→ 计算相同的 G_mode
```

绝对 `C_t` 没有直接作为结论，因为冻结的 anchors 是在 Dense-route selection 上标定的；这里只比较标定前 `G_mode` 的方向。

| route | 可比较 disclosure | 上升 | 不变 | 下降 | 平均 `ΔG_mode` |
|---|---:|---:|---:|---:|---:|
| Dense Top-80 | 29 | 12 | 2 | 15 | `-0.0265` |
| Lexical Top-80 + Dense vectors | 29 | 7 | 8 | 14 | `-0.0190` |

第一次 `other` 披露：

| route | 上升 | 不变 | 下降 | 平均 `ΔG_mode` |
|---|---:|---:|---:|---:|
| Dense | 4 | 0 | 11 | `-0.0546` |
| Lexical | 3 | 1 | 11 | `-0.0509` |

Lexical 结果前后平均 Jaccard 为 `0.414`，高于 Dense 的约 `0.321`，但没有恢复收敛。21 次 `q_lex` 发生变化的可比较 disclosure 中，7 次上升、14 次下降；8 次 `q_lex` 不变的 disclosure 全部保持不变。

因此 Dense 最近邻“天然内部相似”确实会影响绝对分数，但不是这次方向失败的唯一根因。BM25 换成另一批结果后仍然失败，说明更根本的问题是：**任何随当前 query 和 hard mask 改变的召回窗口，其内部绝对 coherence 都不保证与意图透明度同方向变化。**

完整 Lexical-route 实验：

- `artifacts/retrieval/lexical-semantic-coherence-v1.json`
- `artifacts/retrieval/lexical-semantic-coherence-v1.md`

## 复现

生成或校验全部 640-turn fixture：

```powershell
.\.venv-3.10\Scripts\python.exe scripts/query_understanding/generate_simulator_other_prompts.py
.\.venv-3.10\Scripts\python.exe scripts/query_understanding/generate_simulator_other_prompts.py --check
```

重跑当前 16-task 抽样：

```powershell
.\.venv-3.10\Scripts\python.exe scripts/retrieval/evaluate_qu_to_probe.py `
  --tier full `
  --cohort simulator `
  --simulator-suite config/query_understanding/simulator-other-prompts-v1.json `
  --simulator-limit-per-scenario 8 `
  --api-key-file dpskapi `
  --output artifacts/retrieval/qu-to-probe-simulator-other-16x4-v1.json
```

重跑 2×2 结构诊断：

```powershell
.\.venv-3.10\Scripts\python.exe scripts/retrieval/diagnose_transparency_transitions.py
```

重跑 Top-K 深度扫描：

```powershell
.\.venv-3.10\Scripts\python.exe scripts/retrieval/sweep_transparency_probe_depth.py
```

重跑 Lexical-result Dense coherence：

```powershell
.\.venv-3.10\Scripts\python.exe scripts/retrieval/evaluate_lexical_semantic_coherence.py
```

逐 turn 结果：

- `artifacts/retrieval/qu-to-probe-simulator-other-16x4-v1.json`
- `artifacts/retrieval/qu-to-probe-simulator-other-16x4-v1.md`
