能查到不少，但有一个关键点：**去年大部分队伍并没有单独公开一份“technical report.pdf”**。2025 官方统一要求的核心材料其实是 **Devpost 文字说明 + public GitHub/README + 3 分钟以内 demo video**，并没有要求所有 track 都额外交一篇正式技术报告。([TikTok TechJam 2025][1]) 今年我们 Track 4 的要求其实也延续了这个结构：Written Project Description、GitHub README、Demo Video。

所以真正值得看的“技术报告”主要藏在 **Devpost Story 和 GitHub README/docs** 里。我把 Top 5 都核了一遍：

| 名次 | 项目                       | 能找到的技术材料                          | 信息量    |
| -- | ------------------------ | --------------------------------- | ------ |
| 🥇 | **PrivaStream**          | Devpost + README + **完整 docs 目录** | **很高** |
| 🥈 | **Denoising With Mamba** | Devpost + README，基本就是一篇小论文        | **很高** |
| 🥉 | **Compliance Sentinel**  | **Devpost 本身就是长技术报告** + repo      | **极高** |
| 4  | **Privify**              | Devpost + README                  | 中等     |
| 5  | **ARC**                  | Devpost + README + 后续 Medium 技术复盘 | **很高** |

### 🥇 PrivaStream

这个其实比我预想中公开得多。

[PrivaStream Devpost submission](https://devpost.com/software/live-privacy-shield)
[PrivaStream GitHub repository](https://github.com/Saximn/privastream)
[PrivaStream technical docs directory](https://github.com/Saximn/privastream/tree/main/docs)

Devpost 里面已经写了数据、模型、指标和 latency，例如 license plate mAP50、DeBERTa audio PII classifier、OCR classifier，以及 production scalability。([Devpost - The home for hackathons][2])

更值得看的是 GitHub 的 `docs/`：

`ARCHITECTURE.md`
`ML_PRODUCTION_PLAN.md`
`BACKEND_FRAMEWORK_ANALYSIS.md`
`API.md`
`DEPLOYMENT.md`
`IMPROVEMENTS.md`

也就是说它虽然没有叫 **Technical Report.pdf**，但实际上已经有一套相当完整的 engineering design docs。([GitHub][3])

对我们最有参考意义的是：**他们非常努力把一个 72-hour hackathon 项目包装成“production system”**。README 直接给 system requirements、API、Docker deployment、benchmark、fail-safe 等工程信息。([GitHub][4])

---

### 🥈 Denoising Reviews With Mamba

这个可能是五个里面最像**学术技术报告**的。

[Denoising With Mamba GitHub / technical README](https://github.com/Buxt-Codes/Denoising-With-Mamba/blob/main/README.md?utm_source=chatgpt.com)
[Denoising With Mamba Devpost](https://devpost.com/software/denoising-reviews-with-mamba)

README 的结构直接就是：

> Overview
>
> 1. Model Architecture
> 2. Data Collection and Labelling
> 3. Training
> 4. Results
> 5. Additional Tests
> 6. Key Contributions
> 7. Conclusion

而且不是空泛介绍。它真的放了 ablation：

| Model                | Accuracy |       F1 |  ROC-AUC |
| -------------------- | -------: | -------: | -------: |
| BERT–Mamba FiLM      |      .83 |     .827 |     .949 |
| Nomic–Transformer    |     .915 |     .914 |     .984 |
| Nomic–Mamba w/o FiLM |     .791 |     .741 |     .914 |
| **Nomic–Mamba FiLM** | **.929** | **.929** | **.989** |

他们还明确用 ablation 去支撑两个 claim：Nomic encoder 有贡献，以及 FiLM context injection 有贡献。([GitHub][5])

**这个我非常建议我们仔细读。** 它说明 TechJam 的获奖 submission 完全可以用一种类似 mini-paper 的方式写 Devpost/README，而不是普通 hackathon 式“我们用了 GPT + React + FastAPI”。

---

### 🥉 Compliance Sentinel

这个是最有意思的：**它的 Devpost 页面本身几乎就是完整 technical report。**

[Compliance Sentinel full Devpost report](https://devpost.com/software/compliance-sentinel)
[Compliance Sentinel GitHub](https://github.com/Joshyxwa/AC-Acai)

里面有：

**Executive Summary → Problem → Multi-Agent Architecture → Multi-Agent Workflow → Key Innovations → Technical Implementation → Dataset → Impact & Business Value → Sample Output**

而且技术细节很具体，例如：

Legal Analyst 使用 Legal-BERT，按**法律条文而不是 token 数** chunk；Supabase HNSW + cosine similarity；使用 **HyDE RAG**。Adversarial Strategist 则用 Qwen3-Embedding-8B，加 Full Text Search、普通 dense、HyDE dense，然后 **RRF fusion**。后面再接 Defence Auditor → Adjudicator → Compliance Report Agent。([Devpost - The home for hackathons][6])

这篇对我们的意义可能比他们代码还大，因为它特别体现了一件事：

**他们不是简单汇报“我们实现了什么 module”，而是先创造一个 central framing——AI Red Team——再让所有技术模块成为这个故事的组成部分。**

这和我们现在想讲的 **Retrieval as Sensing** 非常同构。

---

### 4th Privify

[Privify Devpost](https://devpost.com/software/privify-wl1khb)
[Privify GitHub README](https://github.com/tohhylucas/Privify/blob/main/README.md)

没有找到单独 report。

不过 README 对系统实现写得还比较完整：两个 server、client-side FHE、Concrete-ML encrypted inference、local LLM explanation、数据流和 API 都有。([GitHub][7])

相比前三名，它的技术 narrative 明显薄一点。Devpost 更偏产品 framing：

**Grammarly-style privacy coach → FHE inference → on-device SLM → Privacy Health Dashboard。** ([Devpost - The home for hackathons][8])

---

### 5th ARC

这个也非常值得我们看。

[ARC Devpost](https://devpost.com/software/arc-automated-review-checking-with-machine-learning)
[ARC GitHub README](https://github.com/frznprograms/ARC/blob/main/README.md)
[ARC 后续技术复盘文章：Building a Multi-Tier Review Classifier](https://medium.com/mitb-for-all/building-a-multi-tier-review-classifier-9b2b7100a161?utm_source=chatgpt.com)

它的核心故事非常干净：

**TF-IDF + Logistic Regression → FastText → LoRA-tuned DistilBERT**

按计算成本从低到高 cascade，容易样本提前退出，困难样本才进入重模型。README 直接说目标是 amortize prediction cost，同时保证 accuracy。([GitHub][9])

后来的 Medium 文章把它总结成：

> 用 “right level of intelligence at the right point in the pipeline” 解决 **speed / cost / accuracy iron triangle**。

这篇虽然是赛后写的，**不是原始 submission technical report**，但反而特别适合研究他们后来是怎么提炼获奖项目故事的。([Medium][10])

---

所以如果你的目的是**给我们今年的 Devpost 技术说明找模板**，我会优先研究：

**Compliance Sentinel → Denoising With Mamba → PrivaStream → ARC。**

这四个分别代表四种很有价值的写法：

**Compliance Sentinel：故事驱动型。** 一个 central thesis 统领所有模块。
**Denoising：论文型。** hypothesis → architecture → experiment → ablation → contribution。
**PrivaStream：工业系统型。** architecture → metrics → latency → scalability → fail-safe。
**ARC：工程 trade-off 型。** 明确优化 accuracy / latency / cost，并用 routing 解释系统设计。

尤其对我们现在的项目，我觉得最应该混合的是 **Compliance Sentinel + Denoising + PrivaStream**：
用 **Retrieval as Sensing / Intent Transparency** 做 central thesis；用 ablation/benchmark 证明 probe 和 adaptive retrieval 真有效；再用 latency、fallback、memory、tool-calling 等说明不是纸面算法，而是能落地的 shopping system。

[1]: https://tiktoktechjam2025.devpost.com/rules?utm_source=chatgpt.com "TikTok TechJam 2025: Build With Joy, Code For Change - Devpost"
[2]: https://devpost.com/software/live-privacy-shield "PrivaStream | Devpost"
[3]: https://github.com/Saximn/privastream/tree/main/docs "privastream/docs at main · Saximn/privastream · GitHub"
[4]: https://github.com/Saximn/privastream "GitHub - Saximn/privastream: TikTok TechJam 2025 - 1st Place · GitHub"
[5]: https://github.com/Buxt-Codes/Denoising-With-Mamba/blob/main/README.md?utm_source=chatgpt.com "Denoising-With-Mamba/README.md at main · Buxt-Codes/Denoising-With-Mamba · GitHub"
[6]: https://devpost.com/software/compliance-sentinel "Compliance Sentinel | Devpost"
[7]: https://github.com/tohhylucas/Privify "GitHub - tohhylucas/Privify · GitHub"
[8]: https://devpost.com/software/privify-wl1khb?utm_source=chatgpt.com "Privify | Devpost"
[9]: https://github.com/frznprograms/ARC "GitHub - frznprograms/ARC: Automatic Review Detectection for Google location reviews · GitHub"
[10]: https://medium.com/mitb-for-all/building-a-multi-tier-review-classifier-9b2b7100a161?utm_source=chatgpt.com "Medium"
