# 03: jaredpalmer/kev

Research note for MirethSTM1 Phase 1. Date checked: 2026-09-30. Code read at commit `6b1da9dc202a9bed515e63f4263b6aab4bc83966` of https://github.com/jaredpalmer/kev (all paths below are in that repo).

Legend: VERIFIED = read in code, docs or API output. UNVERIFIED = secondary source, or not run or accessed by us. CONTRADICTED = the brief says X, reality is Y.

## 1. Repo facts

| Item | Value | Status | Source |
|---|---|---|---|
| Stars | 8,059 | VERIFIED | `gh api repos/jaredpalmer/kev` (2026-09-30) |
| Last push | 2026-09-30T22:35:27Z | VERIFIED | same call |
| Created | 2026-09-17 (first commit d0e2b1f, "Prototype of a Jev-style decision model") | VERIFIED | git log, gh api |
| Activity | 320 commits on the default branch, 511 forks, 28 open issues | VERIFIED | git log, gh api |
| License | Apache-2.0 (code, adapters, heads). Datasets keep their own licenses | VERIFIED | `LICENSE`, repo metadata, model card front matter |
| Stack | Python 3.12/3.13, PyTorch + PEFT, MLX backend on Apple Silicon, HTTP server (`kev/serve.py`), Modal for training and deploy | VERIFIED | `README.md`, `pyproject.toml` |
| Author framing | Built by Jared Palmer "with Devin (Cognition)"; based on Archer Hume's post "Jev's Architecture Unmasked" (archerhume.com/posts/jevs-architecture-unmasked) | VERIFIED (that the README says so) | `docs/model-cards/kev-0.5b.md`, `README.md` |

The repo is about two weeks old and moves fast: released checkpoints have changed several times (older versions are kept as Hub tags). Re-check numbers before quoting them.

## 2. Request / response schema (TypeSafe System One compatible)

Endpoint: `POST /v1/systemone` (VERIFIED, `README.md` section "POST /v1/systemone"; pydantic models in `kev/api.py`). Extras: `GET /v1/models`, `POST /v1/systemone/permute`, `POST /v1/systemone/separate`.

Request (`kev/api.py` lines 17-46):

```jsonc
{ "state": "string | object | array",
  "model": "kev-latest",
  "questions": { "<id>": { "type": "noul" | "choice" | "score",
                           "instructions": "string | object | array (optional)",
                           "criteria": "noul: {true?, false?}; choice: {name: description|null}, 1..255; score: [level, ...], 1..255" } } }
```

Response (README example, Kev-4B on Apple M5):

```jsonc
{ "model": "kev-latest",
  "answers": {
    "department":  {"type":"choice","choice":"returns","confidence":0.21,"probabilities":{"returns":0.47,"shipping":0.28,"billing":0.25}},
    "escalate":    {"type":"noul","noul":0.93},
    "frustration": {"type":"score","score":1.44,"confidence":0.34,"legend":{"0":"Calm","1":"Frustrated","2":"Very angry"},"probabilities":{"0":0.00,"1":0.56,"2":0.44}} },
  "usage": {"input_tokens":101,"output_tokens":161}, "latency_ms": 495 }
```

Details, all VERIFIED in `kev/api.py`:

- `noul` becomes 2 options `[false, true]`; the answer is `p(true)` (`to_answers`, lines 152-153).
- `choice` confidence is `(p_max - 1/K) / (1 - 1/K)` (line 130), 1.0 when K=1. This is a normalized margin, NOT the top probability. For ECE use `max(p)`, not this field.
- `score` answer is the expected level index `sum(i * p_i)`; confidence is `max(0, 1 - E|level - mode| / D)`, D the mean absolute deviation of a uniform distribution over L levels (lines 133-140).
- Probabilities are rounded to 4 decimals so a 255-option distribution still sums within 0.02 of 1 (line 146).
- `usage.output_tokens` is the token count of the serialized answers, not generated tokens.
- Delimiter-like strings in user text are rewritten before tokenization so option boundaries cannot be forged (`kev/model.py` `user_tokens`, lines 78-81). States over 65,536 tokens return 422 with the token count (README).
- Kev says both confidence formulas mirror TypeSafe's `system-one-adapter` 0.2.1 (`_utils/confidence_metrics.py`). UNVERIFIED by us (we did not read that package).
- Kev says the official TypeSafe Python SDK (`typesafe_sdk`: `TypeSafeClient`, `Noul`, `Choice`, `Score`) works against it unchanged with `base_url` set. VERIFIED that the README says so; we did not run it.

The brief's wire API matches for `noul` and `choice` (up to 255). CONTRADICTED in one detail: Kev's `score` has no min/max; it takes an ordered list of level descriptions (1..255) and returns the expected level index. The brief's "score: min/max, handled via bins" must be reconciled with the TypeSafe spec note before the wire format is frozen.

## 3. Architecture

The brief says "LoRA + pointer head". VERIFIED, with an important consequence: this is a trained architecture, not zero-shot label scoring.

| Component | Detail | Source |
|---|---|---|
| Base | Current: Qwen3.5-0.8B-Base, 4B-Base, 9B-Base; 27B from Qwen3.8 post-trained (full fine-tune, 51 GB). Older: Qwen2.5-0.5B (prototype), Qwen3-0.6B/4B/8B | `README.md` Models table; `docs/model-cards/*.md` |
| LoRA | r=16 in released models, alpha = 2r, dropout 0.05, targets q,k,v,o,gate,up,down (plus Gated DeltaNet projections on Qwen3.5 hybrids). Kev-4B: 33.8M trainable parameters | `kev/model.py` lines 265-275; `docs/model-cards/kev-4b.md` |
| Pointer head | Two linear maps `d -> 256`: query from the `<decide>` token hidden state, key from each option's closing `</opt>` token hidden state; scaled dot product `z = (K h_opt . Q h_decide) / sqrt(256)`; softmax over the options of one question | `kev/model.py` lines 205-224 |
| Input packing | One sequence: `<state> ...state... <q> instr <opt> o1 </opt> <opt> o2 </opt> ... <decide> <q> ... <decide>` | `kev/model.py` `encode`, lines 87-127 |
| Attention mask | Block-causal: a question sees the state and its own branch only, never sibling questions; each branch restarts position ids after the state | `kev/model.py` `branch_mask`; 0.5B card |
| Delimiters | Existing reserved Qwen special tokens; user text sanitized | 0.5B card |

How multi-token labels are handled: there is no label-token log-probability scoring at all. Each option's full text (name plus optional description) sits inside `<opt> ... </opt>` in the prompt, and the score is read from the hidden state at the closing delimiter. A multi-token label is natural (the closing-token state has attended over the whole option) and first-token collisions cannot occur. The price: the LoRA and the pointer head must be trained (`kev/train.py`); the pretrained LM head is never used. This is a different mechanism from MirethSTM1's (sum of label-token log-probs from the LM head), so Kev is NOT a code source for our scorer, only for the API shape, the metric protocol and the calibration protocol.

Order sensitivity: options are visible to each other inside one branch, so position matters. Kev measured permutation flips (0.5B prototype: 7.4% argmax flips; Kev-4B Qwen3: 6%; Jev 0%, per its card table) and added a permutation-consistency KL loss (`--perm_kl`) plus an optional `option_isolation` mask (`encode` docstring). Scoring each label in its own suffix (our design) removes option-order effects on the scored labels by construction, if labels are not also listed in the shared prompt; if we list them in the prompt there is still some order dependence. Worth a line in our README and a permutation test in our benchmark.

## 4. Training data and procedure

VERIFIED from the model cards (0.5B prototype, 4B Qwen3, 4B Qwen3.5) and `kev/train.py` flags.

- Public datasets converted to TypeSafe-shaped requests with the serving code path (`api.to_record`): Banking77, BoolQ, AG News, MNLI, SST-5, Yelp Review Full, then TREC, DBpedia-14, Amazon reviews, IMDB, plus programmatic policy/rule records, later CFPB complaints and others.
- Prototype (Kev-0.5B): 1,500 records per source x 6 sources = 9,000 records, 13,500 questions. Augmentation: shuffled option order, 10% true option replaced by "other: None of the above", 15% added distractor option, about 30% null option descriptions, varied state wrappers.
- Objective: cross-entropy over options, plus optional ordinal term for Score (`--ord_w`) and permutation KL (`--perm_kl`). AdamW, OneCycle, 2 epochs, effective batch 8. Prototype lr 2e-4; later recipes lr 5e-5 or 2e-5 (Kev's stated finding: lower lr preserves base capability and was the largest single recipe gain).
- Hardware: prototype on a laptop (Apple M5, about 1h45m); 4B on one H100, about 40 minutes.
- Evaluation discipline: frozen checksummed suites, development sets for selection, locked test read once per model. Suites are published at huggingface.co/datasets/jaredpalmer/kev-suites (linked from the README; not fetched by us).

## 5. The "ECE about 0.065" claim: what it actually is

The number appears in exactly two model cards, for two different models, and neither is the current Kev-4B:

| Where | Model | What the 0.065 is | Status |
|---|---|---|---|
| `docs/model-cards/kev-0.5b.md` front matter and results table | Kev-0.5B prototype (Qwen2.5-0.5B, LoRA + pointer head), superseded | Raw (T=1) ECE, labeled "ECE (10 bins)", pooled over all 1,350 questions of six held-out in-distribution sources (150 records per source, seed 1). Card text: "ECE uses 10 equal-width bins on the top probability." | VERIFIED |
| same card, "Temperature scaling" | same | T fitted on even-indexed records, tested on odd-indexed: T = 1.47, NLL 0.505 to 0.481, ECE 0.057 to **0.031** (on the odd half) | VERIFIED |
| `docs/model-cards/kev-4b-qwen3.md` front matter | Kev-4B on Qwen3-4B-Base (previous generation, Hub tag `qwen3`) | "ECE, raw probabilities" = 0.065 on decision-v4 development (1,204 records, in-distribution), accuracy 0.854 | VERIFIED |
| same card, "Known limits" | same | Out-of-domain raw ECE 0.096; "temperature fitted in-domain does not transfer" | VERIFIED |

Verdict: PARTLY CONTRADICTED. "Reported ECE about 0.065" is true only for the raw, uncalibrated, in-distribution readout of two older checkpoints. It is a BEFORE number. After temperature scaling the prototype reports 0.031. The current Kev-4B card reports ECE 0.013 as served on in-distribution development data, 0.042 out-of-domain (Jev 0.049 on the same set), and 0.084 to 0.095 on its hard-v1 set (`docs/model-cards/kev-4b.md` results table). Do not cite 0.065 as Kev's headline number.

Exactly how Kev measures ECE (`kev/metrics.py` lines 15-21, VERIFIED):

- Top-label (confidence = max probability; correct = argmax equals label), not classwise.
- 10 equal-width bins on [0, 1], left-closed `[lo, hi)`, last bin closed. Default `bins=10`.
- Sum over non-empty bins of (bin share of samples) x |accuracy in bin - mean confidence in bin|.
- Pooled over all question types (choice, noul, score) in one population; for `noul` the confidence is `max(p, 1-p)`. No classwise or equal-mass variant exists.
- Also reported per source/task (`grouped_metrics`), per state-length bucket, and with bootstrap 95% CIs.

Excerpt (Kev, Apache-2.0, `kev/metrics.py`):

```python
edges = np.linspace(0, 1, bins + 1); e = 0.0
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (conf >= lo) & (conf < hi) if hi < 1 else (conf >= lo) & (conf <= hi)
    if m.any(): e += m.mean() * abs(correct[m].mean() - conf[m].mean())
```

## 6. Other metric definitions (for comparability)

All VERIFIED in `kev/metrics.py`. Rows are dicts `{p, label, type, keys, task, source, group, variant, logits?}`.

| Metric | Definition in Kev | Lines |
|---|---|---|
| NLL | Mean of `logsumexp(z/T) - (z/T)[label]`, exact from recorded logits; fallback `-log(max(p[label], 1e-9))` | 29-53 |
| Brier | Multiclass, SUMMED over options: `sum_k (p_k - onehot_k)^2`, then mean over questions. Range 0 to 2 (not divided by K). For a binary question this equals `2 (p - y)^2`, twice the usual binary Brier | 74 |
| Accuracy | `argmax(p) == label` | 73 |
| Score extras | MAE of expected level vs label; ranked probability score (mean squared gap of the CDFs) | 75-77 |
| Confident error rate | share of questions with `max p >= 0.9` that are wrong; also coverage and accuracy at 0.9 | 82-83 |
| Selective prediction | coverage at <= 5% and <= 1% error (accept in descending confidence while empirical error among accepted <= budget), risk-coverage curve, AURC, threshold selection | 103-164 |
| Confidence bias | mean max-p minus accuracy (signed overconfidence) | 89 |
| Temperature fit | Grid search: minimum mean NLL over a 121-point log-spaced grid on T in [0.25, 4], every question weighted equally ("micro"); "macro" weights per task. Fits on RAW logits and refuses rows already calibrated | 272-294 |
| Out-of-fold calibration | 5-fold, folds disjoint in (source, record group) so sibling questions never straddle train and test; per-fold T fitted on the other folds and applied to the held-out fold; bootstrap CI on raw vs out-of-fold ECE | 317-369 |
| Bootstrap | Record-clustered, source-stratified percentile bootstrap (1,000 samples), paired for model-vs-model deltas | 308-426 |

Calibration protocol lessons worth copying (`scripts/calibrate_checkpoint.py` docstring, VERIFIED):

- Fit T on held-out DATASETS, not on held-out items of the training sources. Kev's round 19 fitted T on in-distribution rows and then missed an out-of-distribution set (ECE 0.059); a pool of held-out datasets gave 0.0085 on the same checkpoint (round 20).
- Argmax and accuracy never change under temperature scaling; only ECE, NLL, Brier and the selective metrics move.
- The shipped temperature is stored in the head file and applied as `z / T` at inference only; `KEV_TEMPERATURE=1.0` restores raw probabilities. Current Kev-4B T = 2.41 (round-8 version 2.96): raw logits are heavily overconfident, unlike the 0.5B prototype (T = 1.47). VERIFIED from card text.
- Even calibrated, ECE varies across datasets (Kev-4B: 0.013 in-distribution, 0.042 to 0.095 on harder sets). Report ECE per dataset, never one pooled figure alone.

Caveats (VERIFIED): Kev's ECE pools heterogeneous tasks, so an easy high-accuracy source can mask miscalibration on a hard one. Top-label equal-width ECE is noisy with few samples per bin; Kev's own 27B card notes that raw (0.048) and out-of-fold (0.038) ECE intervals overlap. Kev's release gates are accuracy, Brier, confident-error rate and coverage at a fixed error budget, with ECE secondary; ECE alone is easy to game.

## 7. Benchmark numbers and datasets (as reported by Kev, UNVERIFIED by rerun)

We ran no Kev model (no GPU here, weights not downloaded). Numbers below are as printed in the README; each cell is development / locked test.

| Model | Accuracy, new sources | Accuracy, trained sources | Brier, new sources |
|---|---|---|---|
| Kev-0.8B | 0.648 / 0.697 | 0.827 / 0.838 | 0.481 / 0.416 |
| Kev-4B | 0.817 / 0.838 | 0.873 / 0.865 | 0.269 / 0.242 |
| Kev-9B | 0.820 / 0.852 | 0.874 / 0.873 | 0.289 / 0.217 |
| Kev-27B | 0.851 / 0.889 | 0.865 / 0.866 | 0.225 / 0.156 |
| Jev (development only) | 0.857 | 0.845 | 0.211 |

Latency (README): Kev-4B, six questions on a new short text: 18.1 ms model time on H100, 41.5 ms on L40S; Apple M5 721 ms for five questions, 136 ms when the text repeats and is cached.

Dataset overlap with our benchmark plan: AG News (`fancyzhx/ag_news`), Banking77 (`legacy-datasets/banking77`), Yelp Review Full (`Yelp/yelp_review_full`), SST-5 (`SetFit/sst5`), BoolQ, MNLI. Kev TRAINS on the train splits of these, so they are in-distribution for Kev and zero-shot for MirethSTM1: Kev's numbers on them are not comparable to ours. Kev's never-trained ("new source") sets are QNLI, SciQ, TweetEval-offensive, PAWS, MMLU, Emotion, MMLU-Pro and held-out policy structures (4B Qwen3 card, VERIFIED).

Base-model baseline precedent (0.5B card, VERIFIED): zero-shot Qwen2.5-0.5B with next-token logits over option letters A-H, per-source accuracy / ECE (AG News 0.813 / 0.069 base, 0.787 / 0.160 instruct; BoolQ 0.427 / 0.274 base; MNLI 0.460 / 0.225 base; SST-5 0.373 / 0.083 base); K=77 (Banking77) was not run. That "letter readout" is the single-token approach our full-label scoring is meant to beat.

## 8. Reliability of Kev's own claims

- Kev retired several of its own eval sets (scienthoon, WANLI, TypeSafe-gold) on 2026-09-27 and 2026-09-30 for label-quality problems (`docs/model-cards/kev-4b.md`). The repo carries a `PLAN.md` of about 294 KB and an `AGENTS.md` of about 67 KB; methods and results change between rounds. Quote a specific commit and model revision.
- Jev numbers in Kev's tables come from running Jev through the Vercel AI Gateway on development sets only (`kev/jev.py`, README). The README itself says it is not a controlled comparison.
- The mechanism attribution rests on Archer Hume's blog post, a third-party inference about Jev, not TypeSafe documentation. We have not read that post (UNVERIFIED); do not repeat claims about Jev internals.

## 9. What this means for MirethSTM1

### Spec implications

1. Wire API: Kev is a second implementation of the `POST /v1/systemone` shape with `noul`, `choice` (1 to 255 options, name to description-or-null map) and `score` (ordered list of level descriptions, 1 to 255). Match field names exactly (`answers.<id>.noul | choice + confidence + probabilities | score + legend + probabilities + confidence`, `usage`, `latency_ms`, `model`). `POST /v1/decide` can be a superset, with `/v1/systemone` as an alias so the TypeSafe SDK works unchanged. Settle `score` (level list vs min/max + bins) against the TypeSafe API note before freezing.
2. Emit two numbers per question: API-compatible `confidence` (normalized margin) and `p = max(probs)`. The frozen JSONL `p` must be the top-label probability so ECE is computed on it.
3. Round probabilities to 4 decimals in API output (sum tolerance 0.02 at 255 options); keep full precision in the JSONL log.
4. Temperature: apply `z / T` to the per-label summed log-probs before softmax; store T with the model config; provide an env or flag override to get raw probabilities. Record `T` and a raw/served flag in every JSONL event (T is already in the frozen schema).
5. Reject over-long contexts explicitly with token counts (Kev returns 422); never truncate silently.

### Calibration and evaluation code shape (for comparable numbers)

6. `ece(conf, correct, bins=10)`: top-label, 10 equal-width bins, left-closed with a closed last bin, pooled. Use this as the headline ECE so numbers are comparable with Kev's and most of the literature, and also report it per dataset. An equal-mass variant may be added but must be labeled and never mixed.
7. `brier`: state the convention. Use the summed multiclass form `sum_k (p_k - y_k)^2` (range 0 to 2) to match Kev, and note that for binary questions it is twice the textbook binary Brier. Report any other form only with a label.
8. `nll`: `logsumexp(z/T) - (z/T)[y]` from logits; probability floor 1e-9 only when logits are missing. Store raw logits (our per-label summed log-probs) in benchmark rows so T can be re-fitted and re-applied without rerunning the model (the idea behind Kev's `raw_row` and `tempered_row`).
9. `fit_temperature`: minimum mean NLL over a log-spaced grid (Kev: 121 points on [0.25, 4]). Simple, deterministic, dependency-free; enough for one scalar. Kev's fitted values (1.47 to 2.96) sit inside that range, but our summed-log-prob logits may differ, so use a wider range (for example 0.25 to 16) and assert the fitted T is not at a grid edge.
10. Fit T on a split that is not the evaluation split and, per Kev's round 19 to 20 finding, preferably on different datasets from those reported. Plan: fit on a calibration slice, report ECE before and after on every dataset's test split including at least one dataset not used for fitting, plus a 5-fold out-of-fold ECE with a bootstrap CI for the fitting dataset.
11. Add the cheap, honest selective metrics Kev uses: confident-error rate at `p >= 0.9`, coverage at 5% error, AURC, and signed confidence bias (mean max-p minus accuracy).
12. Reliability diagram: write our own (10 equal-width bins; bar = accuracy per bin; diagonal = perfect; annotate bin counts and grey out bins with n < 10; before and after T). Kev's repo has `kev/plot.py`, but we only checked that it reads a log and did not find a reliability plotter; there is nothing we verified to reuse.
13. Bootstrap 95% CIs on ECE, NLL and Brier (1,000 resamples of records). Our datasets have one question per record, so a plain record bootstrap is the right unit; Kev clusters because it has sibling questions per record.
14. Permutation test: shuffle the option order and report argmax flip rate and spread of p(correct), as Kev does. This shows the benefit of independent full-label scoring.

### What to reuse (ideas, with attribution)

- Wire-API shape and confidence formulas (Kev `kev/api.py`, Apache-2.0; TypeSafe reference adapter). Re-implement, do not copy files. README credit line: "API shape and metric protocol inspired by jaredpalmer/kev (Apache-2.0)".
- The metric protocol: top-label 10-bin ECE, summed Brier, NLL from logits, coverage at an error budget, group-disjoint out-of-fold temperature fit, paired bootstrap. These are definitions and short functions; re-implement from the definitions above. If any Kev function is copied verbatim, keep the Apache-2.0 notice and mark the change.
- The discipline: frozen dev and test splits, read the test once, fit calibration on held-out datasets, publish per-source rows.

### What to avoid

- Do not copy Kev's architecture (LoRA, pointer head, packed block-causal mask). It needs trained weights and is out of scope (the brief defers fine-tuning). Our scorer uses the pretrained LM head.
- Do not cite "ECE 0.065" as Kev's result, and do not compare our calibrated ECE with Kev's raw 0.065. The fair reference points are post-scaling figures (prototype 0.031; Kev-4B 0.013 in-distribution and 0.042 out-of-domain), on Kev's sets, not ours.
- Do not report a single pooled ECE across datasets; report per dataset.
- Do not set our zero-shot AG News, Banking77, SST-5 or Yelp numbers beside Kev's without saying Kev trained on them.
- Do not repeat Kev's Jev comparisons as fact; our framing stays "free approximation, not equivalent".
- Do not depend on Kev's package (torch + peft + MLX + Modal, huge planning files). Keep our dependencies small.
- Do not repeat the Archer Hume post's claims about Jev internals (not read by us).

### Open items

- Confirm the TypeSafe `score` request shape and whether the SDK hard-codes `/v1/systemone` (see the TypeSafe API note).
- Optional, desktop GPU later: run Kev-0.8B or Kev-4B via `kev.serve` on our benchmark questions as a trained-baseline row. Kev needs Python 3.12 or 3.13 (our venv is 3.11), so use a separate environment.
