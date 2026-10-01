# 07. Benchmark design

Date checked: 2026-09-30. Status tags: VERIFIED (read or ran it myself), UNVERIFIED (estimate or secondary source), CONTRADICTED (brief says X, reality is Y).

## 1. Dataset IDs that load today

Tested with `datasets` 5.0.1 (current at check time) via `load_dataset_builder` plus a streamed first row of every split. Dataset scripts are refused by this version.

| Dataset | ID that works | Result | Status |
|---|---|---|---|
| AG News | `fancyzhx/ag_news` | Loads (parquet files). Bare `ag_news` fails in datasets 5 (legacy name is not a `namespace/name` id). | VERIFIED |
| Banking77 | `mteb/banking77` | Loads (parquet, MIT). | VERIFIED |
| Banking77 (legacy mirror, added by reviewer) | `legacy-datasets/banking77` | HF API lists `data/train-00000-of-00001.parquet` and `data/test-00000-of-00001.parquet`, license cc-by-4.0 (file listing only; loading not run by the reviewer). An alternative to `mteb/banking77` if the original CC-BY-4.0 terms are preferred. | VERIFIED (listing) |
| Banking77 (original) | `PolyAI/banking77` | CONTRADICTED as a usable source: fails with `RuntimeError: Dataset scripts are no longer supported, but found banking77.py`. The repo holds only `banking77.py` and `dataset_infos.json`. `revision="refs/convert/parquet"` does not exist for it either, and the `/parquet` API returns an error. | VERIFIED |
| SST-2 | `stanfordnlp/sst2` or `nyu-mll/glue` config `sst2` | Both load, identical splits. | VERIFIED |
| Yelp 5-star | `Yelp/yelp_review_full` | Loads (parquet). Bare `yelp_review_full` fails like `ag_news`. | VERIFIED |

Sources: https://huggingface.co/datasets/fancyzhx/ag_news, https://huggingface.co/datasets/mteb/banking77, https://huggingface.co/datasets/PolyAI/banking77, https://huggingface.co/datasets/stanfordnlp/sst2, https://huggingface.co/datasets/nyu-mll/glue, https://huggingface.co/datasets/Yelp/yelp_review_full (HF API output, 2026-09-30).

### Splits, sizes, labels, license (all counts from `info.splits`, class counts from loading the data)

| Dataset | Splits | Label names, index order | License (card metadata) |
|---|---|---|---|
| AG News | train 120000, test 7600 (1900 per class) | 0 World, 1 Sports, 2 Business, 3 Sci/Tech | card says `unknown`; card text restricts use to non-commercial research and similar. Do not redistribute the data; download by ID. |
| Banking77 (mteb) | train 9993, test 3076. Per class: train 35 to 187, test 39 or 40 | 77 intents in `label_text` (see note) | MIT (mteb repo card). The original PolyAI repo says CC-BY-4.0. |
| SST-2 | train 67349, validation 872 (428 neg, 444 pos), test 1821 | 0 negative, 1 positive | card `unknown`. `nyu-mll/glue` is tagged `other`. |
| Yelp full | train 650000, test 50000 (10000 per star) | 0 "1 star", 1 "2 star", 2 "3 stars", 3 "4 stars", 4 "5 stars" | `other`, `license_details: yelp-licence` (Yelp dataset terms, academic use). Download by ID only. |

Notes (VERIFIED by running):
- **SST-2 test labels are hidden** (all `-1`). Use `validation` (872 rows) as the only labelled held-out set.
- **SST-2 train is phrase-level** (first row: "hide new secretions from the parental units"), validation is full sentences. Calibrating T on train phrases and evaluating on validation sentences is a distribution shift. Recommendation: fit T on train rows with at least 8 words, or cross-fit on validation (fit on one stratified half, evaluate on the other, then swap). Report the train-fitted number as a secondary row.
- **Banking77 label names in `mteb/banking77`** are lowercase snake_case with two oddities: `Refund_not_showing_up` (capital R) and `reverted_card_payment?` (trailing question mark). Normalize these (lowercase, strip `?`) in the benchmark label map, and keep the raw name in a comment. The `mteb` dataset has no `ClassLabel` names in its features; `label_text` carries them. Index 0 is `activate_my_card` and index 76 is `wrong_exchange_rate_for_cash_withdrawal` (the order is the dataset's own, verified by sorting (label, label_text) pairs). Build the list from the train split at runtime with `sorted({(label, label_text)})` instead of hard-coding it.
- Banking77 labels are multi-token in Qwen tokenizers and share prefixes (`card_payment_fee_charged`, `card_payment_not_recognised`, `card_payment_wrong_exchange_rate`; `top_up_*`; `verify_*`). This is exactly the case full-label scoring targets, so Banking77 is the headline dataset for differentiator 1.

### Which split for what

| Dataset | Fit T (calibration) | Evaluate |
|---|---|---|
| AG News | 1000 random rows from train (stratified 250 per class, seed 0) | 2000 random rows from test (seed 0, stratified 500 per class) |
| Banking77 | 1000 random rows from train (seed 0; at least 10 per class where possible) | full test, 3076 |
| SST-2 | 1000 train rows with at least 8 words; also report 2-fold cross-fit on validation | full validation, 872 |
| Yelp | 1000 random rows from train (200 per star) | 2000 random rows from test (400 per star) |

Calibration rows never appear in the evaluation set (different split), so no leakage. Truncate Yelp reviews to 512 tokens (keep the head), identically for scorer and baseline.

## 2. Metric definitions

Let there be N examples, K classes, model distribution p_i (length K, sums to 1), true class y_i, prediction c_i = argmax_k p_ik, confidence q_i = max_k p_ik.

| Metric | Formula | Notes |
|---|---|---|
| Accuracy | (1/N) sum 1[c_i = y_i] | Invalid baseline output counts as wrong. |
| Macro-F1 | mean over k of F1_k, F1_k = 2 P_k R_k / (P_k + R_k) | Classes with no predictions and no support: F1 = 0 if any support, skip if none. Use `sklearn.metrics.f1_score(average="macro", zero_division=0)`. Invalid outputs map to an extra "invalid" prediction, so they hurt recall. |
| NLL | -(1/N) sum log p_i,y_i | Clip p at 1e-12. Scorer only (baseline has no distribution). |
| Brier (multi-class, chosen) | (1/N) sum_i sum_k (p_ik - 1[k = y_i])^2 | Range 0 to 2. |
| ECE | sum_{b=1..15} (n_b / N) * abs(acc_b - conf_b) | 15 equal-width bins on q_i, top-label. |

**Brier choice: sum over all classes.** It is the original Brier (1950) multi-class form, a strictly proper scoring rule over the whole distribution, so it rewards getting the non-top probabilities right, which matters for a product that returns a distribution. The top-label variant, (q_i - 1[c_i = y_i])^2, only scores the confidence of the argmax and is the same thing as a binary Brier on the correctness event; it is not proper for the full distribution. Report the sum-over-classes value as "Brier". Optionally also print Brier / 2 so the scale is 0 to 1. Note that the value depends on K (more classes, smaller typical value), so compare within a dataset only. (Design reasoning; no external source needed.)

**ECE.** Guo et al. use M = 15 equal-width bins on the top-label confidence and the weighted absolute gap between accuracy and confidence: "ECE (with M = 15 bins)" in Table 1 (VERIFIED, read the paper text: https://arxiv.org/abs/1706.04599, Guo, Pleiss, Sun, Weinberger, ICML 2017). Reference implementation to match: bin edges `linspace(0, 1, 16)`, left-closed bins, last bin closed on the right so q = 1.0 is counted. Alternatives, to be reported in an appendix only, not the headline: 10 bins (kev uses 10, see section 7), equal-mass (adaptive) bins, classwise ECE, debiased ECE. Known caveat (UNVERIFIED as a general statement, standard in the literature): ECE is biased upward at small N because each bin has noise; at N = 1000 with 15 bins (about 67 per bin on average, far fewer in low-confidence bins) expect a noise floor of roughly 0.01 to 0.02. So always give a bootstrap 95% CI (1000 resamples of the evaluation rows) next to every ECE, and do not claim differences smaller than the CI.

**Reliability diagram.** Same 15 bins. x = mean confidence in bin, y = accuracy in bin, diagonal = perfect. Draw bars or points only for non-empty bins, size or opacity by n_b, and show a confidence histogram below. Plot "before" (T = 1) and "after" (fitted T) on one figure per dataset, from the same bin edges. Save PNG plus the raw bin table as JSON so the README can be regenerated.

**Temperature scaling.** One scalar T > 0 on the score vector z_i (the summed label log-probs, before softmax): p = softmax(z / T). Fit on the calibration split by minimizing mean NLL. Implementation choice: optimize u = log T (keeps T positive) with `torch.optim.LBFGS` (strong Wolfe line search, about 50 iterations) or `scipy.optimize.minimize_scalar(bounds=(log 0.05, log 20), method="bounded")`. NLL in log T is smooth and unimodal in practice, so a bounded scalar search is sufficient and has no learning-rate to tune. Check the optimizer against a 121-point log-grid on 0.25 to 4 in a unit test (kev does exactly the grid, see section 7). Argmax does not change with T, so accuracy and macro-F1 are identical before and after; only NLL, Brier and ECE move. Fit one T per dataset and per model, and additionally report a single pooled T across datasets, to show how portable it is. Store T with the model id in the event stream (`T` field of the frozen JSONL schema).

## 3. Baseline: same model generates JSON

Purpose: the fair "what would you do without this engine" comparison.

- Same model and weights, same dtype (bf16), `attn_implementation="sdpa"`, same chat template, same context text, same schema rendered as instructions.
- Prompt: system message gives the field names, allowed values for each, and "reply with a JSON object only". Use the same allowed-label strings as the scorer.
- Decoding: greedy (`do_sample=False`, no temperature), `use_cache=True`, `max_new_tokens = 24 + 16 * n_fields` (covers a short key, a label of up to about 10 tokens and punctuation per field; Banking77 labels are the longest, so use `32 + 24 * n_fields` for that dataset). Stop on EOS. Record the cap hit rate.
- Validity check: `json.loads` of the decoded text (after stripping a markdown fence if present), then `jsonschema.validate` with `enum` per field and `required` for every field, `additionalProperties: false`. Report **schema-validity rate** = valid / N.
- Counting invalid outputs: **as wrong**. Accuracy is over all N rows (denominator is N, not the valid count). Also report "accuracy among valid" as a secondary number so the reader sees the two effects separately. An output that is valid JSON but has an out-of-enum value is invalid. No retry, no repair, no partial credit (a per-field "salvaged accuracy" column is allowed as a secondary number: parse what can be parsed field by field with a tolerant regex).
- Confidence: **none by default**. Generated JSON has no probabilities, so the baseline has accuracy, macro-F1, validity and latency only; NLL, Brier, ECE are "n/a". Optional extra row "baseline + token prob": take the softmax probability of the first label token from `output_scores=True` as a confidence for the chosen label and report ECE for it. This is a separate, clearly labelled row; it is not a distribution, so NLL and Brier stay n/a. Mark it optional; skip first if time is short.
- Stronger baseline (deferred): grammar-constrained decoding (for example the `outlines` or `lm-format-enforcer` libraries). It removes invalid JSON but not the per-token generation cost. Say in the README that the plain `generate()` baseline is the one measured.

## 4. Latency protocol

Hardware facts to print in every report: GPU name, driver, torch and CUDA versions, dtype, model id, `nvidia-smi` temperature and clocks at start and end.

- Batch size 1, single stream, one request at a time, model already loaded and on GPU, no other GPU load.
- Timed span: from receiving the raw context string and schema to having the typed result in Python (tokenization, prefill, all fields, softmax, result dict). For the baseline: tokenization through `json.loads` plus schema check. Excluded: model load, dataset load, writing the JSONL.
- Warmup: 10 untimed runs per field count and per system, with the same shapes as the timed runs (first calls trigger CUDA context, kernel selection and allocator growth).
- Timing: `torch.cuda.synchronize()` immediately before starting the clock and again before stopping it, `time.perf_counter()` in between. (CUDA events are an acceptable alternative for the GPU part, but wall time with synchronize is what a user sees.)
- Runs: N = 200 timed runs per (system, field count), each on a different context (cycle through 200 distinct evaluation texts), so no result or prefix is reused between runs. Report p50 and p95 (`numpy.percentile`, linear interpolation) plus mean, min and max. With N = 200, p95 rests on the 10 slowest samples, which is noisy; say so, and report the bootstrap 95% CI for p50 and p95.
- Field counts: 1, 5, 10, 20. Interleave systems (scorer run, baseline run, scorer run, ...) per field count so thermal drift affects both equally. Run the whole protocol twice (cold GPU and warm GPU) only if time allows.
- Also report: tokens prefilled, candidate suffix tokens scored, tokens generated by the baseline, peak memory (`torch.cuda.max_memory_allocated`).

### Building 1/5/10/20-field schemas from these datasets

Use one context text per run (an AG News article, or an SST-2/Yelp review) and several questions about that same text, rendered into both the scorer schema and the baseline prompt identically. Field pool (labels known without a human):

| Field | Type | Options | Ground truth |
|---|---|---|---|
| topic | choice | World, Sports, Business, Sci/Tech | AG News label (only when the text is an AG News row) |
| sentiment | noul | yes/no "is the sentiment positive" | SST-2 label (SST-2 rows only) |
| intent | choice | 77 Banking77 intents | Banking77 label (only for Banking77 rows) |
| stars | score | 1 to 5 | Yelp label (Yelp rows only) |
| synthetic noul fields | noul | yes/no | computed by rules, see below |

Synthetic yes/no fields with deterministic ground truth (so latency runs still produce checkable answers): "Does the text contain a digit?", "Does the text contain a question mark?", "Is the text longer than 200 characters?", "Does the text contain a quotation mark?", "Does the text start with an uppercase letter?", "Does the text contain the word 'the'?", "Does the text mention a person's name?" (this last one has no rule; skip), and so on. Generate these from a fixed list of about 25 rule-based questions; take the first n_fields minus the real fields.

Schemas per field count (AG News context as the running example):
- 1 field: topic.
- 5 fields: topic + 4 synthetic noul.
- 10 fields: topic + 9 synthetic noul (or 2 real choice fields plus 8 noul when two datasets are concatenated).
- 20 fields: topic + 19 synthetic noul.
This measures the effect of field count at constant context, which is the claim (one prefill, N cheap field scores versus N+ tokens of generation). Keep a second family "mixed" (topic, sentiment, intent, stars using a concatenated multi-source context) only if time allows. Fairness rules: identical context and schema text in both systems; identical label strings; schema text length grows with field count in both prompts (report prompt tokens per field count, since prefill grows too); baseline JSON output length grows with field count, scorer's candidate scoring grows with options, and both effects are the thing being measured, not hidden.

## 5. Sample sizes and arithmetic (Thursday to Sunday, RTX 5070 12 GB, 4B bf16)

All per-document costs below are my ESTIMATES (UNVERIFIED). They must be replaced with measured numbers after the first GPU smoke test; the plan below has slack for a 2x miss.

Assumptions: Qwen3-4B bf16 weights are about 8 GB. Decode is memory-bandwidth bound: 8 GB at roughly 672 GB/s (RTX 5070 spec sheet figure, UNVERIFIED) gives a floor near 12 ms/token, so assume about 30 ms/token in HF eager Python. Prefill of about 150 to 500 tokens takes roughly 50 to 150 ms.

Per-document estimate (seconds):

| Workload | Scorer | Baseline (generate) |
|---|---|---|
| AG News, 1 field | 0.15 | 0.8 (about 20 tokens x 30 ms + prefill) |
| Banking77, 1 field, 77 candidates | 0.4 | 0.6 to 0.8 (about 12 tokens, long prompt listing 77 labels) |
| SST-2, 1 field | 0.12 | 0.8 |
| Yelp, 1 field, 5 bins | 0.25 | 0.5 (1 to 3 tokens but long prompt) |

Accuracy and calibration runs:

| Item | Docs | Scorer time | Baseline time |
|---|---|---|---|
| AG News eval | 2000 | 2000 x 0.15 = 300 s = 5 min | 2000 x 0.8 = 1600 s = 27 min |
| Banking77 eval | 3076 | 3076 x 0.4 = 1230 s = 21 min | 3076 x 0.6 = 1850 s = 31 min |
| SST-2 eval | 872 | 872 x 0.12 = 105 s = 2 min | 872 x 0.8 = 700 s = 12 min |
| Yelp eval | 2000 | 2000 x 0.25 = 500 s = 8 min | 2000 x 0.5 = 1000 s = 17 min |
| Calibration fits (scorer only) | 4 x 1000 | 4000 x 0.25 = 1000 s = 17 min | none |
| Subtotal | | about 53 min | about 87 min |

Latency runs (210 calls per cell: 10 warmup + 200 timed, batch 1):

| Fields | Baseline per call (output tokens x 30 ms) | Baseline 210 calls | Scorer per call | Scorer 210 calls |
|---|---|---|---|---|
| 1 | about 20 tokens = 0.7 s | 2.5 min | 0.15 s | 0.5 min |
| 5 | about 80 tokens = 2.4 s | 8.4 min | 0.3 s | 1 min |
| 10 | about 150 tokens = 4.5 s | 15.8 min | 0.5 s | 1.8 min |
| 20 | about 300 tokens = 9 s | 31.5 min | 0.9 s | 3.2 min |
| Subtotal | | about 58 min | | about 7 min |

Total compute for the 4B model: 53 + 87 + 58 + 7 = 205 min, about 3.4 hours. Fast mode (1.7B) repeat of the latency table and Banking77/AG News only: about 40 percent of that, so about 1.4 hours if the schedule allows. Suggested placement: Friday evening smoke test (20 docs per dataset, 10 latency runs) and fixing; Saturday run the full accuracy/calibration pass and the latency pass in the background (about 3.4 h, unattended); Sunday morning regenerate tables and reliability diagrams from saved JSONL, write the README table. If the 2x slack is consumed, cut in this order: baseline eval down to 1000 docs on AG News and Yelp, latency N from 200 to 100, drop the 1.7B repeat. Keep Banking77 at the full 3076 test rows for the scorer (headline) and at least 1000 for the baseline.

Statistical sense check: with N = 2000 the standard error of an accuracy near 0.9 is sqrt(0.9 x 0.1 / 2000) = 0.0067, so differences under about 1.3 points (2 SE) are noise. Banking77 at 3076 rows gives about 0.005 at 0.9. SST-2 at 872 rows gives 0.010. Report bootstrap CIs, not bare point values.

## 6. Yelp 5-star via score bins

Yelp is a score field: min 1, max 5, handled as 5 bins, each bin label the string "1".."5" (single tokens, so the wire `score` type also works with 0 to 10 style ranges where "10" is multi-token and full-label scoring matters). Ground truth: `label + 1` (HF label 0 is 1 star).

Distribution p over the 5 bins from the scorer; T fitted on the 1000 train rows by NLL on the bin distribution (same routine as the choice fields).

Report both decision rules:

| Rule | Prediction | Why |
|---|---|---|
| Argmax | mode bin, integer star | accuracy, macro-F1 (5 classes), ECE on top-label confidence |
| Expected value | E = sum_k k p_k, real number 1 to 5 | MAE and RMSE against the true star; minimizes squared error in theory, so expect lower RMSE than argmax |

Metrics for the bin distribution: MAE = mean |E - y|, RMSE = sqrt(mean (E - y)^2), same two for the argmax prediction, off-by-one accuracy (|c - y| <= 1), NLL, multi-class Brier, and ECE (15 bins, top-label) on the bin distribution, before and after T. Because stars are ordinal, add the Ranked Probability Score (mean over examples of sum_{k=1..4} (CDF_pred(k) - CDF_true(k))^2) as an ordinal proper score; this is optional and cheap. Expect the expected-value RMSE to beat argmax RMSE and argmax accuracy to be modest (5-way Yelp is hard; fine-tuned models are in the high 60s percent, UNVERIFIED, a zero-shot 4B should be lower). State plainly that Yelp is a bin-mechanism test, not a leaderboard claim. The baseline for Yelp: generate `{"stars": 1..5}` with the same validity check; only accuracy, macro-F1, MAE and RMSE apply (from the integer).

Caveat: Yelp review text has a restrictive license; the repo holds code and result tables only, never a copy of the data.

## 7. Repos whose metric code we could match

| Repo | Stars / license / last push (gh api, 2026-09-30) | What it publishes | Status |
|---|---|---|---|
| razorback16/openjev | 549 stars, Apache-2.0, pushed 2026-09-29 (HEAD dcd2094) | No ECE or Brier code anywhere (a grep for ece, brier, reliability found only calibration-temperature plumbing: `config.py` line 85, `encoders.py` lines 266 and 435, which load a stored temperature, "1.532 for JevK5 v0.2"). No benchmark to match. | VERIFIED (clone at https://github.com/razorback16/openjev) |
| jaredpalmer/kev | 8059 stars, Apache-2.0, pushed 2026-09-30 (HEAD 6b1da9d) | `kev/metrics.py`: `ece(conf, correct, bins=10)` (equal-width, top-label, last bin right-closed, line 15), `fit_temperature` (min mean NLL over a log grid 0.25 to 4, 121 points for released fits: `TEMPERATURE_FIT = {"aggregation": "micro", "points": 121}`, lines 272 to 300), held-out rules (fit on development rows, evaluate on others), bootstrap CIs, selective-prediction metrics. | VERIFIED (https://github.com/jaredpalmer/kev, kev/metrics.py) |

The brief's kev "ECE ~0.065": not in `kev/metrics.py`. Doc 03 located it in two older model cards as a RAW (before temperature scaling) in-distribution ECE, 10 bins (0.5B prototype and the Qwen3-4B generation). It is a before number, not a calibrated result.

Matching decision: kev uses 10 bins and a grid over T; the brief and Guo et al. use 15 bins. Headline = 15 bins (Guo convention). Also print the 10-bin ECE in the JSON output so a reader can compare with kev numbers. Our `ece(conf, correct, bins=15)` and `fit_temperature` (grid, then bounded refine) should be API-shaped like kev's so results are comparable; reimplement from the formulas above (do not copy code; Apache-2.0 attribution would apply to any copied excerpt).

## 8. What this means for MirethSTM1

For SPEC:
1. Benchmark datasets by ID: `fancyzhx/ag_news`, `mteb/banking77` (not `PolyAI/banking77`), `stanfordnlp/sst2` (validation only), `Yelp/yelp_review_full`. Pin `datasets>=4` (tested 5.0.1) and never rely on scripts.
2. Headline calibration settings: 15 equal-width bins, top-label ECE, sum-over-classes Brier, NLL, one scalar T fit on a disjoint calibration split by minimizing NLL on log T. Every metric reported with a bootstrap 95% CI.
3. `fit_temperature(scores, labels) -> T` and `ece(conf, correct, bins=15)` signatures; `reliability_bins(...)` returning the per-bin table that the figure code and the README both consume. Accuracy and macro-F1 must be unchanged by T (add that as a unit test).
4. Frozen event schema already has `T` and `model`; the benchmark harness writes the same JSONL so jevscope can replay a benchmark run.
5. Baseline is `generate()` greedy plus `jsonschema` validation, invalid counted wrong, no confidence. Output of the harness must include validity rate next to accuracy.
6. Latency protocol lives in code as defaults: warmup 10, N 200, batch 1, synchronize both ends, field counts 1/5/10/20, p50 and p95 with bootstrap CI.
7. Score fields report both the distribution and the expected value; benchmark reports MAE and RMSE for both rules.

For the scaffold:
- `mirethstm/calibration.py`: `fit_temperature`, `ece`, `brier`, `nll`, `reliability_bins`, `bootstrap_ci`. Pure numpy/torch, CPU-testable with synthetic logits (known-T recovery test: sample logits, apply a known T, check recovery within 5 percent).
- `bench/` script with subcommands `accuracy`, `latency`, `report`; dataset loader table `{name: (hf_id, config, split_fit, split_eval, text_col, label_col, label_names)}`; Banking77 label normalization (lowercase, strip `?`) and runtime-derived label order.
- Synthetic rule-based noul field list (about 25 questions with deterministic answers) for the 5/10/20-field schemas.
- Datasets cached out of the repo, never committed (license terms for AG News, Yelp, SST-2).
- Per-sample rows saved (JSONL) so all tables and diagrams are regenerable without rerunning the GPU.

Open items for the lead engineer:
- Replace every per-document time estimate in section 5 with measured values after the first GPU smoke run, and recompute the schedule.
- Decide whether the optional "baseline + token prob" row is worth the time (default: skip).
- Confirm on the desktop that the Windows symlink warning from `huggingface_hub` (cache falls back to copies without Developer Mode) does not matter; it only affects disk usage. (It appeared when I ran the loaders here; harmless.)
