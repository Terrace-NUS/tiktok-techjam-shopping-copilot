# Fixed Multi-view Probe and Intent Transparency v1

- Status: **implementation working contract**
- Date: **2026-08-29**
- Depends on: Retrieval Contract v0, Compiled Query v0, Session Context v1

This document freezes the first implementable Probe boundary. The Probe is a
small, fixed observation step that runs before adaptive retrieval. It answers:

> When the current compiled intent is projected into this catalog, do the
> plausible results form one compact semantic direction or remain dispersed?

It does not predict whether the user will buy. It does not use a hidden target,
and it does not change its own behavior based on the score it produces.

## 1. Fixed execution

For one `CompiledQuery`, the Probe uses:

```text
q_lex + q_sem + resolved eligible products
    ├── Lexical view: fixed Top-K from q_lex
    └── Dense view:   one full score vector, fixed Top-K from q_sem
                         ├── listing geometry
                         └── semantic-mode geometry
```

All views use the same resolved hard eligibility set before their Top-K
truncation. V1 fixes `probe_k=80`; it is not a function of `C_t`.

The first implementation uses the dense view to calculate transparency. The
lexical view is supporting evidence and a retrieval-health diagnostic. A later
facet view may explain which attributes divide the semantic modes, but the
number of populated facets is not a transparency feature.

## 2. Why semantic modes exist

Raw Top-K products are listings, not necessarily independent choices. Several
near-identical listings of the same popular product can make a result bag look
artificially concentrated.

The Mode Probe therefore groups near-duplicate dense vectors deterministically:

1. process candidates in dense-rank order;
2. compare a candidate with the fixed leader of every existing mode;
3. join the best matching mode when leader cosine, rounded to six decimal
   places, is at least the frozen V1 threshold `0.94`;
4. otherwise start a new mode whose leader is this candidate; and
5. never update a leader after creation.

The fixed-leader rule avoids a chain in which `A` is close to `B`, `B` is close
to `C`, but `A` and `C` are not actually the same product direction.

Each mode receives one normalized centroid. Transparency geometry is measured
over these centroids with equal mode weight. A mode with twenty duplicate
listings therefore does not count twenty times.

## 3. Raw concentration evidence

For candidate or mode vectors `x_i`, the existing coherence operator first
subtracts the whole-catalog mean and L2-normalizes the centered vectors. Let
`R` be the length of their mean direction and `n` their count. The raw statistic
is:

$$
G = \frac{nR^2 - 1}{n - 1}.
$$

`G` is the mean pairwise cosine between different centered directions. Higher
means the result directions agree more strongly. It is signed raw evidence,
not yet a probability.

The snapshot records both:

- listing coherence: useful for comparison with the old Dense Probe;
- equal-mode coherence: the primary duplicate-resistant transparency signal.

If fewer than two semantic modes exist, mode coherence is unavailable rather
than automatically treated as perfect certainty.

## 4. From raw evidence to `C_t`

`C_t` is produced by a versioned monotonic calibrator:

$$
C_t = \operatorname{clip}\left(
\frac{G_{mode}-g_{low}}{g_{high}-g_{low}}, 0, 1
\right).
$$

`g_low` and `g_high` are frozen catalog/model-specific anchors obtained from a
target-free natural-language prompt suite. V1 uses the 10th and 90th
percentiles of all available calibration-pair mode coherences. The paired
`vague -> specific` labels are used only for a held-out directional audit,
because unrelated product domains have different natural baseline geometry.
The anchors belong to a calibration artifact and must be bound to the same
dense index and Probe policy.

Important exclusions:

- eligible-product count does not enter `C_t`;
- the number of extracted preferences does not enter `C_t`;
- lexical/dense route agreement does not enter `C_t`;
- Buying/Browsing wording does not enter `C_t`; and
- an explicit request for more diversity changes presentation policy, not the
  underlying transparency score.

When calibration or geometric evidence is unavailable, the stored certainty
is `None`. A downstream controller may use a recorded neutral fallback of
`0.5`, but it must not save that fallback as measured user certainty.

## 5. `D_t`: retrieval-health diagnostics

`D_t` is a sidecar diagnostic object for the current run. It explains whether
the Probe evidence is trustworthy and why a fallback was used. It is not a
second exploration axis and is not user intent.

The implementation records:

- eligible and observed counts;
- dense and lexical availability;
- lexical query-token coverage and mean normalized IDF;
- overlap between lexical and dense Top-K;
- listing versus mode coherence;
- largest mode share and effective mode count;
- duplicate-concentration warning;
- hard-filter relaxation supplied by the mask resolver; and
- canonical reason codes for degraded or unavailable evidence.

Counts can mark an empty or under-filled observation as low quality. They never
raise or lower a valid `C_t` value.

An under-filled Probe therefore records a degraded diagnostic but may still
emit calibrated `C_t` when at least two valid semantic modes exist. An empty
Probe, fewer than two modes, invalid geometry, or an unapproved calibration
emits `C_t=None`.

## 6. Runtime DTO boundary

The Probe produces one immutable snapshot containing raw evidence. A separate
estimator consumes that snapshot and a calibration policy, then produces:

```text
TransparencyEstimate
├── certainty: float | None
├── raw_mode_coherence: float | None
├── quality
├── reason_codes
└── SearchBelief projection
```

`SearchBelief` remains compatible with Session Context v1:

- `certainty` stores measured `C_t` or `None`;
- `certainty_evidence.raw_concentration` stores a clipped display copy of the
  raw concentration;
- `candidate_modes` stores the observed modes and representative product IDs;
- `facet_stats` may remain empty until a catalog-grounded facet observer is
  connected; and
- `D_t` remains ephemeral instead of being written as user preference.

The score vector is reused in memory by the later dense route; it is not copied
into Session Context.

## 7. Determinism and invariants

- Probe depth and mode threshold are construction-time constants.
- Dense scoring happens once per compiled semantic query.
- The hard mask is applied before every Probe Top-K.
- Dense ties are resolved by `parent_asin`.
- Lexical ties are resolved by `parent_asin`; SQLite BM25 is ordered ascending.
- Mode creation is rank ordered; leaders never move.
- The same Top-K geometry with a different eligible count yields the same
  `C_t`.
- An empty or invalid observation never becomes high transparency.
- `C_t` cannot control the Probe that produces `C_t`.

## 8. Frozen V1 calibration

The score-blind suite contains 24 target-free families: 12 calibration and 12
held-out audit families across 12 product domains. The audit achieved full
availability, strict `specific > vague` direction in 9/12 pairs (`0.75`), and
a positive median paired delta (`+0.0213248041`). The frozen hackathon gate
passed.

The runtime anchors are:

```text
g_low  = 0.256963026520931
g_high = 0.4483984914520624
```

The bound configuration is
[`config/retrieval/transparency-calibration-v1.json`](../../../config/retrieval/transparency-calibration-v1.json),
and the reviewed evaluation summary is
[`transparency-evaluation-v1.md`](transparency-evaluation-v1.md).

## 9. Deliberately deferred

The Probe runner itself remains independent of hard-constraint interpretation.
The implemented `ResolvedCompiledProbeRunner` first invokes the upstream
Retrieval Evidence and hard-mask layer, then passes its one release-bound mask
and relaxation flag into the fixed Probe. The Probe derives the exact
parent-ASIN set from that bound mask and applies it to both views. The upstream
contract is [`evidence-hard-mask-v0.md`](evidence-hard-mask-v0.md).

It does not make facet counts part of the scalar score, does not use an
LLM to merge products, and does not create a second fallback candidate pool.
Facet distributions remain valuable for explanation and clarification, but
they should be added only when their evidence source and normalization are
explicit.
