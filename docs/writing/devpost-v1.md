## 🛍️ Shopping Intent Is a Journey, Not a Label

Shopping rarely begins with a perfect query. A shopper may start with a vague occasion, discover new possibilities, reject colours or materials, switch product categories, and only gradually realise what they actually want. Traditional search treats every turn as another keyword request; our copilot treats the entire conversation as **one evolving shopping decision**. It explores when the product direction is still open, focuses when that direction becomes coherent, and adapts when the shopper changes their mind. **This is not a chatbot wrapped around search—it is a shopping system that uses retrieval to understand.**

<!-- Next: place the main story visual here. -->

---

## ✨ What Makes Our Copilot Different

Our goal is not only to retrieve what shoppers already know to ask for. It is to create momentum: reveal genuinely different possibilities while intent is still forming, help curiosity grow into genuine purchase intent, and then narrow the path towards a confident decision. The three capabilities below make that journey possible.

### 📡 Beyond Buying or Browsing

Instead of forcing shoppers into a binary label, we measure **intent transparency** from the shape of the product space itself. This signal tells the copilot when to explore broadly and when to focus—and directly changes both retrieval and ranking.

### 🧠 Intent That Evolves With You

DeepSeek uses native tool calls to add, revise, remove, or override typed shopping preferences. The result is a precise, inspectable short-term memory rather than an opaque summary of the conversation.

### 🌱 A Memory That Knows What to Forget

After each completed purchase, the copilot re-evaluates product similarity and re-distils its long-term preference memory. It learns useful shopping continuity without storing the private story behind every purchase.

---

## 🪄 Behind the Magic

### 🔎 Probe Before We Plan

Language models understand what words *could* mean, but they cannot see what the live catalogue actually contains—whether a request spans several product directions, is dominated by near-duplicate listings, or has little inventory support. Before choosing a search strategy, our copilot therefore runs a fast, fixed sensing pass:

```text
Conversation → Fixed Catalogue Probe → Observe the Product Landscape → Plan Retrieval
```

JD.com's SIGIR 2026 **Probe-then-Plan** system demonstrated that an initial retrieval snapshot can overcome the blindness of language-only planning in deployed e-commerce search ([Chen, Zhai, and Li, 2026](https://arxiv.org/html/2603.15262v2)). We extend this idea from grounding a search plan to sensing an evolving shopping decision. The Probe stays identical across turns, and its observation becomes the evidence for our **Intent Transparency** algorithm.

### 📡 Intent Transparency: Beyond Buying or Browsing

“Is this shopper buying or browsing?” is the wrong question. A shopper ready to buy may still have a scattered idea of what they want, while a casual browser may already describe one highly specific product. The question that matters is whether we can help them **discover—not merely retrieve—the one product that feels meant for them**. Search does not need another behavioural label. It needs a live measure of **how much meaningful product space remains open**—so every search, question, and recommendation can turn uncertainty into progress towards a purchase the shopper feels confident about.

Inspired by research showing that the concentration of engaged product vectors can reveal query coherence ([BoDS, 2026](https://sigir-ecom.github.io/eCom26Papers/paper_753.pdf)), we designed a new multi-turn, catalogue-grounded algorithm: **Intent Transparency** (\\(T_t\\)).

The Probe efficiently evaluates the complete Session Context across our 50K-product catalogue using structured constraints and semantic preferences. For every product \\(i\\) and preference \\(c\\), it computes a fuzzy membership \\(m_{ic}\in[0,1]\\). We combine these conditions as a Product of Experts and discount dense regions of near-duplicate listings:

$$
N_t = \sum_{i \in \mathrm{catalog}} \frac{\prod_c m_{ic}^{\lambda_c}}{d_i}
$$

Here, \\(\lambda_c\\) controls the strength of each preference and \\(d_i\\) is the local catalogue density around product \\(i\\). As a result, many sellers listing nearly the same product do not masquerade as many genuinely different possibilities.

> ⚡ **50K products in under 40 ms.** The full query-to-catalogue similarity pass is executed as a single CUDA matrix multiplication, measured at **39 ms** on our RTX 4070 Ti.

We then map this remaining intent volume against the complete-catalogue reference volume:

$$
T_t = 1 - \frac{\log(1 + N_t)}{\log(1 + N_{\mathrm{catalog}})}
$$

\\(T_t\\) is a catalogue-grounded control signal, not an LLM confidence score. For “something for a summer wedding,” a low \\(T_t\\) keeps several plausible product directions open; after “simple silver earrings under USD 80,” \\(T_t\\) rises as the relevant space becomes more concentrated.

### 🎯 From Possibility Space to the Right Ten

Intent Transparency changes both **what the system retrieves** and **what the shopper finally sees**.

**Adaptive recall.** After confirmed violations are removed, we build a 300-product pool through multi-centre semantic, lexical, and structured-facet recall. Low \\(T_t\\) opens as many as six semantic directions and spends more of the pool on discovery; high \\(T_t\\) searches more deeply around one direction and gives exact evidence more room. Round-robin fusion protects smaller directions from being erased by one popular product cluster. In a controlled 50K-catalogue sweep, changing only \\(T_t\\) produced **6 → 5 → 4 → 2 → 1** active directions.

**Intent-aware ranking.** A local BGE cross-encoder reduces 300 candidates to 48 while protecting every active recall direction. DeepSeek then reads the complete Session Context and grounded product cards to judge each product's individual fit; its score contributes 80%, with BGE providing a 20% relevance anchor. We deliberately keep \\(T_t\\) out of that judgement. Instead, a final set selector uses \\(w_{\mathrm{relevance}} = 0.30 + 0.60T_t\\) to choose ten products: low transparency rewards meaningful difference across the set, while high transparency rewards precise fit.

### 🧠 Intent That Evolves With You

**DeepSeek interprets. Tools constrain. Code remembers.**

Most conversational systems remember shopping by accumulating a transcript or repeatedly asking an LLM to rewrite a summary. That summary may sound fluent, but it can silently drop an old preference, confuse a refinement with a replacement, or rewrite facts the shopper never changed.

We let DeepSeek V4 Flash understand human language—but we never give the model ownership of memory. It reads the new message alongside the current Session Context, then **uses native tool calls to propose precise state operations**:

```text
User Message + Current Context
        → DeepSeek Native Tool Call
        → add / revise / remove / override
        → Validate + Deterministic Reducer
        → Complete New Session Context
```

In a live stress test, the current intent was understated black waterproof commuter sneakers under USD 120—lightweight, with no leather. Then the shopper said: **“Actually, the shoes are sorted. What I need now is something I can throw a 15-inch laptop, charger, and gym clothes into. Keep the same USD 120 limit and that quiet look—not necessarily black—but none of the shoe-specific stuff. It should pass at work without looking like one of those rigid briefcases.”**

Without being told which fields to edit, one native tool call changed the goal to a bag, carried over only the budget and quiet aesthetic, removed every shoe-specific constraint, added the new capacity and work context, and excluded the rigid-briefcase look. No repair was needed.

The tool schema defines what may change; local validation decides whether the proposed operation is legal; and deterministic code commits the new state. Every change is inspectable and replayable, while the downstream Probe always receives one complete, repaired intent—not another guess from the raw transcript.

> ✅ **130 / 130 valid native tool calls** in our fixed Query Understanding evaluation suite.

The LLM may reason about memory, but only the application is allowed to own and mutate it.

### 🌱 A Memory That Knows What to Forget

**It learns your taste—not your identity.**

A single purchase is evidence, not a permanent label. Psychology-aware recommendation research treats preferences as constructed and revisable ([Atas et al., 2021](https://doi.org/10.1007/s10844-021-00674-5)); cross-session systems such as MemoCRS likewise show why useful history should be distilled instead of replaying every old conversation ([Xi et al., 2024](https://arxiv.org/abs/2407.04960)).

A completed purchase—not casual browsing—creates new behavioural evidence. For each product category \\(c\\), let \\(s_{ij}\\) be the cosine similarity between two purchased product vectors. Their mean pairwise similarity is:

$$
S_c = \frac{2}{n_c(n_c - 1)} \sum_{i<j} s_{ij}
$$

$$
C_c = \frac{n_c - 1}{n_c - 1 + \kappa} \mathrm{clip}(\frac{S_c - b_c}{1 - b_c}, 0, 1)
$$

We set \\(C_c=0\\) when fewer than two purchases exist; otherwise, \\(b_c\\) removes the category's natural similarity and \\(\kappa\\) prevents sparse evidence from becoming an identity. Similar choices raise the **Shopping Continuity Score** \\(C_c\\); a divergent purchase lowers it and weakens memory's influence. The system then re-distils the evidence into a compact natural-language preference instead of endlessly appending history.

```text
Completed Purchase → Category Similarity Recalculation
                   → Preference Re-distillation
                   → Weak Prior for the Next Ranking
```

We retain what can improve the next recommendation—product evidence, category continuity, and editable preference statements—not who the purchase was for, why it happened, or the private story around it. The current Session Context always outranks long-term memory, and every remembered field can be inspected, corrected, or deleted by the shopper.

> 🔐 **A purchase can guide the next search without becoming a permanent identity.**

---

## 🧩 How It All Comes Together

<!--
Place the full-system architecture visual here.

Main path:
User Language → DeepSeek Query Understanding → Session Context → Catalogue Probe
→ Intent Transparency → Adaptive Multi-Route Recall → BGE Shortlist
→ DeepSeek Product Judgement → T-aware Final Set → Natural-Language Response

Memory loop:
Completed Purchase → Shopping Continuity Recalculation → Preference Re-distillation
→ Privacy-Aware Long-Term Memory → Weak Prior for Future Ranking

The visual should make two precedence rules obvious:
1. Session Context always overrides Long-Term Memory.
2. Intent Transparency controls recall breadth and final-set diversity, not DeepSeek's
   individual product judgement.
-->

---

## 🛠️ Built With

| Area | What we used |
|---|---|
| **Development tools** | VS Code on Windows, PowerShell, Python 3.10 virtual environments, Git and GitHub. OpenAI Codex and ChatGPT supported implementation, testing, design review, and documentation. Experiments ran on an NVIDIA RTX 4070 Ti with CUDA. |
| **APIs and models** | The **DeepSeek API (`deepseek-v4-flash`)** powers native-tool Query Understanding and evidence-aware product judgement. The official **TikTok TechJam Agent interface and evaluation harness** provide the competition boundary. Local Hugging Face models include `BAAI/bge-small-en-v1.5` for dense product embeddings and `BAAI/bge-reranker-v2-m3` for cross-encoder relevance. |
| **Libraries and frameworks** | PyTorch and CUDA for full-catalogue vector computation; Sentence Transformers for embedding and reranking models; NumPy for retrieval, fusion, and DPP set selection; scikit-learn for dataset-distribution diagnostics; pytest, Ruff, and mypy for verification and code quality. |
| **Datasets and assets** | The official **50,000-product catalogue**, derived from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) `Clothing_Shoes_and_Jewelry`; 200 public development sessions and the official evaluation harness; source-grounded product fact cards; and hand-authored natural-language, override, and adversarial Query Understanding tests. |

> 📦 **The competition catalogue remains immutable.** Embeddings, semantic facets, indexes, and product cards are stored as reproducible sidecar assets; no source row is rewritten.
