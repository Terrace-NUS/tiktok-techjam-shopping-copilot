# Intent Transparency 运行时协议 v1

- 状态：**Hackathon runtime v1 已实现**
- 参数入口：[`../../../config/retrieval/intent-volume-v1.json`](../../../config/retrieval/intent-volume-v1.json)
- 实现入口：[`../../../src/shopping_copilot/retrieval/intent_volume.py`](../../../src/shopping_copilot/retrieval/intent_volume.py)

## 1. 这个组件做什么

Query Understanding 已经把自然语言修复成完整 Session Context。Intent Volume 不再猜一次用户原话，
而是回答：

> 在当前完整意图下，50k catalog 中还剩下多少有意义的购物空间？

```text
IntentState + CompiledQuery
    → structured hard evidence：满足为 1，不满足为 ε
    → goal / soft / semantic preference：embedding membership
    → Product of Experts 合并条件
    → inverse-density weight 给重复 listing 降权
    → 剩余体积 N_t
    → 0–1 Intent Transparency T_t
    → 独立诊断 D_t
```

它不修改 Session Context、catalog、semantic release 或 dense index。

## 2. v1 参数

当前选择与 60 段、130 turn 全量实验中排名第一的配置一致：

| 参数 | v1 |
| --- | ---: |
| density temperature | 0.025 |
| semantic membership quantile | 0.85 |
| semantic membership temperature | 0.06 |
| hard mismatch floor `ε` | 0.01 |
| soft preference exponent | 0.5 |
| stable relative tolerance | 0.10 |
| diagnostic Top-K | 20 |

固定 policy ID 为 `soft_hybrid_intent_volume_v1`。

## 3. `T_t` 怎样从体积得到

令：

- `N_t`：当前完整 Session Context 剩余的密度校正体积；
- `N_catalog`：整个 catalog 在相同 density kernel 下的有效参考体积。

v1 映射为：

$$
T_t = 1 - \frac{\log(1+N_t)}{\log(1+N_{catalog})}
$$

并限制在 `[0, 1]`：

- `T_t` 越低，仍有更多合理方向；
- `T_t` 越高，当前条件留下的空间越小；
- 撤销条件可以让 `T_t` 下降；
- 换目标时仍计算新值，但 transition 标记为 `moved`，不把涨跌解释成继续收敛。

### 为什么不用“当前 goal 的体积”作为零点

goal 本身可能已经很具体，例如 `pearl stud earrings`。如果每次都把 goal-only 空间强制定为透明度 0，
这一整段已经表达出来的信息会消失，而且相同意图会因 session 起点不同而得到不同分数。

因此 v1 使用 release-bound catalog reference。输出仍保留 `goal_reference_volume`，以后可以做分品类展示
或实验，但它不进入 v1 的 `T_t`。

## 4. 变化方向

方向依据原始 `N_t` 判断，而不是依据已经压缩到 0–1 的展示值：

| `direction` | 含义 |
| --- | --- |
| `initial` | 当前 session 第一笔可用测量 |
| `narrower` | 同一目标下剩余体积明显下降 |
| `broader` | 同一目标下剩余体积明显上升 |
| `stable` | 相对变化不超过 10% |
| `moved` | 上游明确表示换商品目标；不要求数值单调 |
| `unavailable` | 当前没有可信的商品空间可测 |

这明确拒绝 `T_t = max(T_t, T_{t-1})`。系统不会把用户改变主意伪装成理解不断增长。

运行时不会用两个不相等的 goal 字符串自动推断 `moved`：`footwear → boots` 也可能只是细化。调用方
只有在 QU / session update 明确表达跨商品切换时才传入 `goal_switched=true`；其他 goal 修订仍按实际
体积判断 narrower、broader 或 stable。

## 5. JSON 输出

```json
{
  "schema": "shopping-copilot/intent-transparency/v1",
  "session_id": "demo-running",
  "dense_index_id": "sha256:...",
  "catalog_semantic_release_id": "sha256:...",
  "policy_id": "soft_hybrid_intent_volume_v1",
  "mapping_id": "catalog_log_volume_v1",
  "intent_version": 3,
  "goal": "trail running shoes",
  "transparency": 0.91,
  "change": 0.18,
  "direction": "narrower",
  "remaining_intent_volume": 1.29,
  "catalog_reference_volume": 38123.16,
  "goal_reference_volume": 249.46,
  "diagnostics": {
    "status": "healthy",
    "reason_codes": [],
    "semantic_factor_count": 2,
    "hard_factor_count": 2,
    "relaxed_hard_preference_ids": [],
    "top_all_hard_compliance": 0.8,
    "top_mean_hard_factor_compliance": 0.9,
    "active_facets": ["feature", "use_case"],
    "dont_care_facets": [],
    "open_facets": ["price", "weight"]
  }
}
```

`open_facets` 不是 Intent Volume 自己猜的。若 clarification planner 已经知道仍值得追问的 facet，可以
显式传入；否则保持空数组。

## 6. `D_t` 与 unavailable

`T_t` 只由体积决定。以下信号只能改变诊断状态，不能偷偷修改分数：

- hard constraint 是否因空结果降级成 semantic preference；
- Top 商品同时满足全部可验证 hard facet 的比例；
- Top 商品平均满足多少 hard facets；
- 本轮使用了多少 semantic / structured factors；
- 哪些 facet 已明确、明确不在乎或仍待 clarification planner 决定。

当前 compiled query 不可检索，或既没有 goal 也没有任何可测语义因子时：

```json
{
  "transparency": null,
  "change": null,
  "direction": "unavailable",
  "diagnostics": {
    "status": "unavailable",
    "reason_codes": ["intent_not_searchable"]
  }
}
```

不使用 0 或 0.5 冒充一次真实测量。

## 7. 与旧 `C_t` 的关系

`retrieval/transparency.py` 中的旧 Top-80 mode-coherence `C_t` 暂时保留，避免破坏已有 Probe 和测试。
新故事主线只使用本协议的 Fuzzy Intent Volume `T_t`。两者不能在 UI 或控制器里混成同一个字段。

后续接入时应新增 Intent Transparency sidecar；等调用方迁移完成，再单独决定是否删除旧接口。

## 8. 当前证据边界

全量 v1.3 运行得到：

- narrower 33/33；
- broader 10/10；
- stable 7/7；
- 三轮渐进收紧的相邻变化 20/20；
- override 只观察迁移，不规定升降。

这些结果说明状态方向和演示链路健康，但方向正确部分也来自 Product of Experts 的状态代数。商品
相关性必须继续看 `D_t`，不能用方向满分代替最终召回质量。

## 9. 运行时一致性与性能

正式组件已重放全部 127 个可检索状态，并逐项对比参数扫描 artifact 中的同一配置：

- parity failure：**0 / 127**；
- transparency 最大绝对误差：小于 **`4e-7`**；
- remaining volume 最大相对误差：小于 **`5e-6`**；
- CPU 热路径平均约 **94ms/turn**，最慢约 **308ms**。

完整 catalog、semantic release、retrieval evidence 和 embedding 模型的冷启动约 77 秒，其中绝大部分
是一次性加载与 evidence index 构建。应用必须把这些对象作为进程级依赖预热并复用；不能在每次
用户消息中重新构造 estimator。

演示轨迹：

- [`../../../artifacts/retrieval/intent-transparency-runtime-v1-demo.md`](../../../artifacts/retrieval/intent-transparency-runtime-v1-demo.md)
- [`../../../artifacts/retrieval/intent-transparency-runtime-v1-demo.json`](../../../artifacts/retrieval/intent-transparency-runtime-v1-demo.json)
- [`../../../artifacts/retrieval/intent-transparency-runtime-v1-full.json`](../../../artifacts/retrieval/intent-transparency-runtime-v1-full.json)
