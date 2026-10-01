# 11. AnyJev deep read

Date checked: 2026-09-30. Repo: https://github.com/nokia-applied-research/AnyJev, read at commit 10d5db9 (pushed 2026-09-28). Paths below are relative to that repo.
Tags: VERIFIED (read in code, docs or API output), UNVERIFIED (not checked or secondary), CONTRADICTED (a claim that does not hold).
Scope: what AnyJev is, what its hidden-state heads cost and buy, its order-flip and calibration numbers, and what MirethSTM1 should take from it. No GPU code was run, nothing installed, no paid calls. Every number below is the authors' own, read from their committed docs; none was re-run here.

## 1. Summary for the lead engineer

- AnyJev is the strongest open project we have read on method rigor: a ladder of readouts (raw, L0, L1, L2), confidence intervals, committed result JSON, a second-run reproduction, and a research log that keeps its negative results. Its 988 stars (as of 2026-09-30 the API says 990) are earned.
- It does not threaten the full-label headline today. It reads single tokens only (letters, digits, Yes/No) and refuses anything else. But its roadmap lists "span readout for more than 26 options" with a teacher-forced scorer and a PMI correction, so the gap is open, not permanent (VERIFIED, see section 9).
- The hidden-state heads (L2) are real and well measured, but they are per question and per model and need 100 to 300 labels each. They cannot serve an arbitrary TypeSafe-style request. The 23 shipped head sets only cover the 20 workflow questions of one dataset plus three benchmark tasks. For a generic engine like ours they buy nothing out of the box.
- Their speed numbers are all one H100 NVL, bf16, eager transformers or vLLM 0.7.0. There is no consumer GPU, Windows or Mac number anywhere. Nothing is comparable to the 68 to 270 ms M4 Max demo figures without a re-run.
- Most reusable for us: the measurement discipline (reverse-order flip rate, ECE definition, median of repeats with spread, unseen states), the canonical option listing, the certified label-free stopping idea, and the layout findings (what a prefix cache can and cannot keep).

## 2. What it is

- Python package `anyjev` 0.2.0, "turn any LLM into a Jev-style decision model", typed decisions (`choice`, `noul`, `score`) with probabilities from one prefill (README.md:101-107; pyproject.toml). VERIFIED.
- Authors list Nokia (Sunnyvale) and Tencent Hunyuan (README.md:14-18). Pyproject classifier is "Development Status :: 2 - Pre-Alpha". VERIFIED.
- Not affiliated with TypeSafe; says so (README.md:190-192). VERIFIED.
- Four levels (README.md:164-169; docs/levels.md):
  - `raw`: softmax over label-token logits at the answer position. "exactly max_tokens=1 plus logprobs" (docs/levels.md:7-21).
  - `L0`: zero labels. K cyclic rotations of the option list, combined by geometric mean (log space), plus a batch label-prior correction at strength 0.75 (docs/levels.md:23-86).
  - `L1`: temperature scaling on 100 to 500 labels, on top of L0 (docs/levels.md:92-114).
  - `L2`: a closed-form linear head on a hidden state about two thirds down the model, 100 to 300 labels per question (docs/levels.md:116-151).
- Status: a package plus benchmark harness, not a server. There is no HTTP server, no MCP, no UI beyond a Gradio Space entry (`space/app.py`) and a demo script. "Jev-compatible HTTP server" is an unticked roadmap item (ROADMAP.md, "Next"). VERIFIED.
- Stars 990, forks 127, created 2026-09-21, last push 2026-09-28, 10 open issues, not archived (gh api, 2026-09-30). VERIFIED.

## 3. License and openness

- LICENSE is the full Apache License 2.0 text (169 lines); `pyproject.toml` has `license = "Apache-2.0"`; GitHub reports `Apache-2.0`. VERIFIED.
- Code: open. Heads: five JSON files in `anyjev-heads/` (1.8 to 4.4 MB each), open under the repo license. Base model weights are not shipped; they are Qwen3 checkpoints from the Hub (Apache-2.0, see file 06). VERIFIED.
- Data: not vendored. `THIRD_PARTY.md` lists each dataset with its license; LocalLLaMA/typed-decisions is Apache-2.0, banking77 CC-BY-4.0, CLINC150 CC-BY-3.0, MASSIVE CC-BY-4.0, 20 Newsgroups "see dataset card". VERIFIED.
- The typed-decisions gold is "one teacher model's soft label per decision, not a human judgment" (THIRD_PARTY.md). README.md:209 adds that a fresh sample of the teacher agrees with its own gold 0.735 of the time. This matters when reading any "accuracy" on that set. VERIFIED.
- Shipped head files are tied to the 23 dataset questions (`anyjev-heads/Qwen__Qwen3-4B.json`: `heads` has 23 entries keyed by question hash, with ids such as `action`, `needs_review`, `risk`, `urgency`, `churn_risk`). VERIFIED.

## 4. Models and plug-in-ability

- Headline models: Qwen3-1.7B, 4B, 8B, 30B-A3B-Instruct-2507, 32B for L2 (README.md:134-140). VERIFIED.
- L0 and L1 are measured on 11 models of seven architectures: Qwen2.5-7B, Qwen3 (1.7B to 32B), SmolLM2-1.7B, OLMo-2-7B, granite-3.3-8B, Phi-4-mini, Mistral-7B-v0.3 (docs/results_small_models.md:5-18). VERIFIED.
- The smallest model measured is 1.7B. There is no Qwen2.5-1.5B or Qwen3-0.6B row (the demo says the 0.6B is used on CPU, space/app.py docstring). So the founder's reference model (Qwen2.5-1.5B) was never run through AnyJev. VERIFIED.
- Stock weights, no fine-tuning, no LoRA. L2 adds a closed-form head (not gradient trained). The research log tried a gradient-trained logistic regression on the same features: +0.002 accuracy, so the closed form loses nothing (docs/research_log.md, entry 0, E0.3). LoRA at the answer position is listed as "Later" (ROADMAP.md). VERIFIED.
- Plugging in other models: yes for raw/L0/L1. `HFBackend(model_name)` takes any causal LM with a chat template; `n_layers` and the backbone are discovered from config and `base_model_prefix` (`anyjev/backends/hf.py:19-55`). Backend protocol is one method, `next_token_logprobs(prompts, token_ids)` (`anyjev/backends/base.py:11-21`). New engines (SGLang, llama.cpp, MLX, Ollama) are listed as help wanted (ROADMAP.md). For L2 the backend must expose hidden states; only local transformers and a vLLM embed server do. VERIFIED.
- Heads do not transfer between models or questions (README.md:210; docs/levels.md:137-141). A question-agnostic head was tried six ways and stayed at or below L0 on every held-out question (ROADMAP.md, "Closed"; best 0.580 vs L0 0.635 on Qwen3-8B). VERIFIED.

## 5. Scoring mechanism

### 5.1 Readout (single token, always)

- Choice options are shown as `A. option text`, `B. ...` and the model is told "Answer with the letter only." The scored tokens are the single letter tokens at the answer position (`anyjev/readout.py:98-127`; labels from `answer_labels`, lines 24-32). VERIFIED.
- `noul` reads the tokens `Yes` and `No`, which stay bound to the option when the phrasing order flips ("Answer Yes or No." vs "Answer No or Yes.") (`readout.py:28-29, 55-60, 108-111`). VERIFIED.
- `score` reads digit tokens `1..n`; if the tokenizer cannot emit single-token digits (sentencepiece), it falls back to letters (`readout.py:35-48`; CHANGELOG 0.1.0 names Mistral). VERIFIED.
- Multi-token labels and collisions raise `LabelTokenError` (`readout.py:63-85`). There is no full-label path. VERIFIED.
- Option limit: `MAX_OPTIONS = 26` in `anyjev/question.py` ("letter-readout limit; span readout will lift this"); score is 2 to 10 levels (the TypeSafe docs say 2 to 10 as well). The README states it plainly: "at most 26 options in the letter readout (a span readout is on the roadmap, not in the code)" (README.md:214). VERIFIED. TypeSafe allows 255; AnyJev cannot run the full 77-way BANKING77 (ROADMAP.md, span readout item). VERIFIED.
- Prompt shape: system line "You are a decision function ...", then `State:`, `Question:`, `Options:` lines, then the instruction. One question per prompt; a state with several questions is several prompts sharing the state prefix (`build_prompt` splits the user text at a `split` index so a backend can reuse the prefix KV, `readout.py:91-127`). VERIFIED.
- Qwen3 thinking is turned off through the chat template (`enable_thinking=False`, `readout.py:130-139`). VERIFIED.

### 5.2 How it batches

- `HFBackend.score_shared` computes the state prefix once, repeats the cache K times (`_repeat_cache`), then runs the K short option-layout suffixes as a batch of K sequences against the cached KV, taking the last hidden state and projecting only that row through `lm_head` (`anyjev/backends/hf.py:67-168`). It checks that tokenizing prefix and suffix separately equals tokenizing the joined text, and falls back to the plain path when not (`hf.py:105-114`). VERIFIED.
- The Decider only uses it when the prefix is at least 256 tokens (`shared_min_prefix_tokens`); on short states "a single batched forward over the full prompts is cheaper than two forwards" (docs/results_latency.md, "transformers backend"). VERIFIED.
- `logits_to_keep=1` avoids projecting every position (`hf.py:57-65`). VERIFIED.
- vLLM: one HTTP request per prompt, `max_tokens=1`, `allowed_token_ids` set to the label ids, `logprobs=K`; relies on `--enable-prefix-caching` (`anyjev/backends/vllm.py`, `_one`). A thread pool of 16 sends requests. VERIFIED.
- No tree mask, no packed single pass in the shipped code. Research entries 21 and 22 tested both ideas; see section 8.

### 5.3 The L2 hidden-state head

- Feature: residual stream at the last prompt position after block b*, where b* is fixed per model at the shallowest block within 0.5 points of the best pooled out-of-fold accuracy (docs/method_v3.md section 2.4). Shipped: 1.7B block 18 of 28, 4B 24 of 36, 8B 24 of 36, 30B-A3B 40 of 48, 32B 52 of 64 (docs/results_exit.md:13-19). No final norm, no `lm_head` is applied. VERIFIED.
- Solvers (`anyjev/heads.py`): diff-of-means, shrunk LDA (Woodbury so the solve is n by n), ridge in dual form, reduced-rank regression. `fit_head` picks layer and hyperparameters by 5-fold out-of-fold NLL after temperature scaling, then refits on everything; the temperature kept is the out-of-fold one. VERIFIED.
- Cost to fit: one forward of the labelled states plus 2 to 8 s CPU on 1.7B to 8B, 7 to 27 s on the 32B (docs/method_v3.md section 2.2, from `bench/results_exit/2026-09-22/<model>.artifact.json`). VERIFIED (docs only).
- Head file size: 23 heads per model in 1.8 to 4.4 MB (README.md:150; file sizes 1.77, 2.21, 3.51, 4.38 MB on disk, with the 30B-A3B at 1.77 MB). VERIFIED.
- Routing: a head is found by exact question layout, then the same options under another wording, then the same option set in another order (`Decider.route`, docs/levels.md:141-145). Label-free adaptation re-estimates only the feature mean and scale from unlabelled traffic (30 requests by default) (`_adapted`, docs/jev_mode.md). VERIFIED.
- Online labelling: `observe(q, state, label)` solves the head at 30 labels, re-solves at 60, 120, and so on (README.md:174-175). VERIFIED.
- On vLLM: an embed server with pooler `LAST`, no normalize, no softmax returns the final-layer hidden state; early depth is obtained by writing a truncated checkpoint (`anyjev/truncate.py`, `anyjev/pipeline.py`). A head fit on a truncated model's output (which has the final norm applied) must be fit on that same truncated model, since it differs from the raw block-b residual (`truncate.py:17-20`). The vLLM backend can only serve the last layer (`vllm.py`, `hidden_states`). VERIFIED.

## 6. What the heads cost and buy

All numbers: LocalLLaMA/typed-decisions, 20 questions, 300 labelled decisions per question to fit, 100 held out per question (2000 pooled decisions). "Accuracy" is agreement with a teacher LLM's gold. VERIFIED from docs/results_exit.md:13-19; not re-run.

| model | raw | L0 | L1 | L2 head | 95% CI | ECE | flip | block |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-1.7B | 0.468 | 0.494 | 0.499 | 0.730 | [0.710, 0.750] | 0.028 | 0.113 | 18 of 28 |
| Qwen3-4B | 0.547 | 0.564 | 0.567 | 0.786 | [0.767, 0.804] | 0.034 | 0.073 | 24 of 36 |
| Qwen3-8B | 0.626 | 0.647 | 0.648 | 0.771 | [0.752, 0.789] | 0.034 | 0.069 | 24 of 36 |
| Qwen3-30B-A3B | 0.599 | 0.630 | 0.630 | 0.799 | [0.780, 0.817] | 0.029 | 0.069 | 40 of 48 |
| Qwen3-32B | 0.684 | 0.700 | 0.699 | 0.798 | [0.780, 0.815] | 0.046 | 0.065 | 52 of 64 |

Comparison rows in the same table: Jev 0.727 "as published, not measured here" and a Laya model fine-tuned on this set's train split 0.768 (measured by them). Jev's number comes from Laya's benchmark file, not from TypeSafe (docs/results_typed.md, last rows). VERIFIED.

### What the heads buy

- Accuracy: +0.24 over L1 on the 1.7B, +0.21 on the 4B, +0.12 on the 8B (docs/research_log.md entry 6). The gain is biggest for small models. VERIFIED (docs).
- Label efficiency on Qwen3-8B (entry 5, `Qwen__Qwen3-8B.labels.json`): 20 labels 0.654, 50 labels 0.707, 100 labels 0.740, 200 labels 0.754, 300 labels 0.772, against 0.626 for L1 at all budgets. ECE falls from 0.11 at 50 labels to 0.034 at 300. VERIFIED (docs).
- Calibration: pooled ECE 0.028 to 0.046 with 15 equal-mass bins (docs/levels.md:164). The Qwen3-1.7B head at 64 percent depth passes the published Jev number (0.730 vs 0.727), but note the 0.727 was not re-measured, the CI on 0.730 is 0.710 to 0.750 and the gold is a teacher label. VERIFIED/UNVERIFIED mix.
- Shipped heads validated through the user path: typed pooled accuracy 0.728, 0.784, 0.763, 0.794, 0.795 for the five models; also banking20 (0.785 to 0.875), newsgroups (0.585 to 0.755), injection (0.960 to 0.995) (docs/results_exit.md, "Shipped heads, validated live"). Newsgroups heads sit at or below L0 for the 1.7B, 4B and 32B (docs/levels.md:125-127). VERIFIED.
- The paraphrase result is a limit, not a buy: a reworded question drops the 4B head from 0.771 to 0.626-0.679 ("original head as is"), recovers to 0.743-0.757 with recentring on 300 unlabelled states, versus 0.761-0.776 for a full refit. Flip vs the original wording is 0.22 to 0.32 for the stale head (docs/results_exit.md, "The same question reworded"). VERIFIED.

### What the heads cost

- Labels: 100 to 300 per question per model (docs/levels.md:148-150). For a generic request with a new schema, there is nothing to fit, so L2 is not available. UNVERIFIED that any user workflow can supply that many labels per schema. This is the main reason heads do not help MirethSTM1 by default.
- Time to run (H100 NVL, bf16, eager transformers, truncated forward, batched 32 prompts, never-seen states) from docs/results_exit.md:128-:
  - Qwen3-1.7B, 110-token state (212-token prompt): raw 28 blocks 4.2 ms batched / 42.1 ms single; 18 blocks 4.2 ms batched / 28.5 ms single. At 1000 tokens: raw 20.6 / 45.3 ms; 18 blocks 14.5 / 32.2 ms.
  - Qwen3-4B, 110 tokens: raw 7.8 ms batched / 51.8 ms single; 24 blocks 5.2 / 34.8 ms. At 1000 tokens: raw 43.2 / 55.0 ms; 24 blocks 29.9 / 38.9 ms.
  - Qwen3-8B, 110 tokens: raw 11.0 / 49.4 ms; 24 blocks 7.5 / 33.9 ms. At 1000 tokens: raw 61.6 / 67.2 ms; 24 blocks 42.0 / 51.6 ms.
  - Qwen3-32B, 110 tokens: raw 41.8 / 85.0 ms; 52 blocks 34.1 / 70.8 ms.
  - Ratio to one plain forward: 0.67x to 0.70x on the 4B and 8B, 0.84x on the 32B, 1.00x on the 1.7B at 110 tokens ("launch-bound") (docs/results_exit.md:13-19). VERIFIED (docs).
- The single-request floor is about 50 ms on every model up to 8B at 110 tokens: "eager-mode transformers overhead, not model compute" (docs/results_latency.md), roughly 1.4 ms of kernel launch per block (docs/research_log.md entry 6). So these numbers say little about how fast the model could run on an optimized stack, and a 1.7B at 42 ms single is not faster than the founder's reference demo class (68 to 89 ms on a laptop GPU, different hardware). UNVERIFIED comparison.
- The heads' speed edge needs the early-exit forward, which only the local transformers path and a truncated checkpoint on a served engine provide. Stock vLLM embed serves the last layer only; the repo says truncation to about 18 of 28 blocks was both 2 points more accurate (0.850 vs 0.830) and faster (1.21x on one question) on Qwen2.5-7B / banking20 (CHANGELOG 0.1.0, "Measured"). VERIFIED (docs); a second-run reproduction is cited by the authors.
- MoE cost not measured: the 30B-A3B cell is blank because eager-mode transformers does not reflect 3B active parameters (docs/results_exit.md:18; docs/method_v3.md section 2.4). The shipped-heads table shows 234.3 ms per decision at batch 16 for the 30B-A3B vs 63.3 ms for the 32B, which is the eager-MoE artifact. VERIFIED.
- Question-agnostic or cascaded early exit did not beat a fixed block (entries 4, 10, 11, closed in ROADMAP.md). VERIFIED.

### Mismatch with our case

- An L2 head reads the same hidden state as the logit readout and replaces the unembedding step with a fitted matrix. It is therefore a calibrated, supervised probe. It does not address option-text scoring at all (it still classifies into K letter slots), and it needs the option set to be fixed.
- Raw readout at the final layer already costs one prefill, the same order as what we plan. So for speed, MirethSTM1 cannot beat AnyJev raw or L2 on the same GPU, only match raw. Full-label scoring adds the label tokens after the prefill; the cost is tokens beyond raw. This is the honest framing for "lightning fast".

## 7. Order flip and calibration

### 7.1 Order-flip measurement (the headline 0.230 to 0.073)

- Definition: "fraction of items whose argmax changes when the option list is reversed (`choice`) or the phrasing is swapped ... Measured at every level" (docs/levels.md:164-168). Only one alternative order is tried, the exact reverse. VERIFIED.
- Qwen3-8B, BANKING77 20-way (banking20), n=300, `docs/results_bench.md:147-154`: raw acc 0.747, ECE 0.240, flip 0.230, cov@5% 0.077; L0 acc 0.803, Brier 0.374, ECE 0.184, flip 0.073, cov@5% 0.463; L1 acc 0.807, ECE 0.095, flip 0.077, cov@5% 0.520. These match the README table (README.md:111-119). VERIFIED.
- Other cells from the same file are less dramatic. Newsgroups (20-way) raw flip 0.233 on Qwen3-8B; Qwen3-4B newsgroups flip 0.273 raw to 0.197 L0 (docs/results_bench.md:123-129); Qwen3-1.7B newsgroups 0.313 raw to 0.237 L0. Across 20-way tasks L0 cuts flips by about a third to two thirds, not always to 0.07. Qwen3-32B banking20 0.213 to 0.067; Qwen2.5-7B banking20 0.197 to 0.080. VERIFIED.
- Smallest model: SmolLM2-1.7B raw choice flip 0.89 (label mass only 0.41, the model often does not answer in format), down to 0.18 at L0 (docs/results_small_models.md:9). Qwen3-1.7B flip 0.37 raw to 0.21 L0. A small model on a 20-way letter question is badly position biased. VERIFIED.
- L0 costs K prefills (up to 20 for a 20-way choice). Measured on Qwen3-8B transformers, 994-token state, K=20, batch: raw 69.3 ms, L0 full 1232.6 ms (17.8x), L0 shared prefix 328.8 ms (4.7x); single-request: 74.1, 1232.8 (16.6x), 323.4 (4.4x) (docs/results_latency.md). VERIFIED (docs).
- Rotation budget: stopping early with a certificate reads 7.2 of 18 shifts at a 1% disagreement target on Qwen2.5-7B / massive_route; 2.22x decisions per second on vLLM (37.20 vs 16.73), 2.34x on transformers (16.47 vs 7.04); agreement 0.987, accuracy 0.703 vs 0.697 (docs/research_log.md entry 24, `bench/results_layout/2026-09-27/`). The full cycle is worth only +1.4 to +2.2 points over an average single rotation; "insurance and an invariance guarantee, not headroom" (entry 23). VERIFIED (docs).
- Head fit on random option orders reduces reversed-order flip to 0.069 to 0.073 (vs 0.176 to 0.183 for a head fit on one order) but only for up to 8 options; wider lists keep canonical order because random orders cost the 20-way heads 6 to 18 points (docs/method_v3.md section 2.2 step 1; docs/results_exit.md reverse-order table). VERIFIED.
- How position bias looks (entry 23): spread across positions 3.90 to 6.81 log units; on Qwen3-8B / massive_route position 0 carries 911 times the prior weight of position 6; the bias estimated on calibration states correlates 0.998 to 1.000 with held-out. Subtracting a constant position prior does not replace the cycle (agreement stays 0.875 to 0.895). VERIFIED (docs).

What this means for us: our full-label scoring removes letter positions as a scored slot, but the option list still sits in the prompt in some order and the model still attends to it. We cannot claim order invariance without measuring a reversed-listing flip rate the same way. UNVERIFIED that full-label scoring flips less; it is our most testable claim.

### 7.2 Calibration

- ECE is on top-1 confidence with 15 equal-mass bins (docs/levels.md:164). Brier and NLL are also reported; ECE alone can be gamed by base-rate predictors (also noted in file 09). VERIFIED.
- L1 temperature scaling brings ECE from about 0.24 raw to 0.07 to 0.14 on the 20-way tasks across models (docs/results_bench.md; docs/results_small_models.md, "L1 ECE (mean)" 0.065 to 0.131). On the typed set, L1 ECE 0.036 to 0.055 (docs/results_typed.md). VERIFIED.
- The prior in L1 is frozen into the artifact and the artifact refuses a different model or a different option layout (docs/levels.md:96-107). VERIFIED.
- Raw ECE on the typed set is worst for the small models: Qwen3-1.7B 0.484, Qwen3-4B 0.411 (docs/results_typed.md). The Jev row shows 0.144, which is Laya's figure, not measured here. VERIFIED.
- Coverage at 5 percent risk is flagged as "a high-variance estimate at n = 300" (README.md:214). VERIFIED.
- Reproduction: "a second run from a clean checkout reproduced every zero-label number bit for bit" (README.md:190-191). This is a claim by the authors about their own repro; no third party has re-run it (the madewithjev.com guide, quoted in file 09, says no alternative has had a calibration comparison run by anyone but its author). UNVERIFIED independently.
- Honest limits stated by the authors: calibration cannot fix a model that cannot answer (maze edges, Minesweeper); the batch prior hurts when one label dominates (docs/when_l0_helps.md, over 230 model-question points) (README.md:211-212). VERIFIED.

## 8. Layout and cost research that bears on our design

Read from docs/research_log.md entries 20 to 24. These are the authors' measurements on Qwen2.5-7B and Qwen3 with the letter readout, so they carry over to our full-label scorer only as hypotheses.

- Entry 22, packing K rotations into one sequence with a block mask: exact (max |dp| 2.17e-05), but only 0.98x to 1.20x over the shared-prefix path already shipped. 5.9x over no sharing at all. Decisions per second flat at about 9 from 2 concurrent decisions up, on both paths. Packing saves memory (10.2 GiB at 16 concurrent) and nothing else. The bill is the feed-forward work on option tokens: 293 TFLOPS on an H100 NVL, about 35% MFU, attention 2 to 7% of prefill. VERIFIED (docs).
  - This supports our packed single-pass design as sound but not a speed multiplier over a shared-prefix batch. The real lever is token count, not mask cleverness.
- Entry 21, "candidates as a set" (each option line sees only the state and the question, same position ids): the mechanism is exact (candidates truly exchangeable, 3.7e-06 vs 2.24e-03 under shuffle) but accuracy collapses: Qwen2.5-7B / massive_route 0.092 (1 pass, options blinded to each other) vs 0.633 for the causal 18-pass baseline; chance is 0.056. Conclusion by the authors: the model compares candidates among the option tokens, not at the readout position. VERIFIED (docs).
  - Caution for us, not a threat: this blinds the option lines inside the prompt. Our packed mask lets each label's tokens see the full prefix (including all options as text) and only hides label suffixes from each other, which is what sequential scoring would do. These are different masks, but any future "tree" or parallel-candidates idea in our code should be tested the same way. UNVERIFIED for our mask beyond our own exactness test (2e-4 in SPEC 3.4).
- Entry 20, reusing the cached question/option block across requests: the splice is exact, but the decision is transported from the previous request (3.0% agreement with the true answer on K=18, chance 5.6%). Recomputing 15% of the span buys nothing; the decision returns only near 100% (Qwen2.5-7B: 0% recompute agree 0.110, 100% agree 1.000). VERIFIED (docs). Closed by the authors.
- Entry 20, moving the state: three layouts compared at L0 on 10 model-task cells, paired bootstrap:
  - `state-first` (what everyone does): prefix cache keeps only the system prefix.
  - `question-first` `[P][Q][O][S]`: the whole fixed block becomes a permanent prefix; 5x to 10.4x fewer tokens on short states, but accuracy was better on 2 cells, worse on 3, inconclusive on 5, and badly worse on long states (Qwen2.5-7B newsgroups -0.228, Qwen3-1.7B massive_route -0.147).
  - `middle-state` `[P][Q][S][O]`: at or above `state-first` on 9 of 10 cells, but only 1.1x to 1.2x saving.
  - Their conclusion: the layout must be chosen per (model, question) on calibration data. VERIFIED (docs).
  - For us: our prompt is state then questions (SPEC 3.1). Putting the repeated schema before the state would let a prefix cache keep it across requests with a stable schema. Large win on short states if accuracy holds, but their data says it is model- and task-dependent. UNVERIFIED for full-label scoring.
- Raw prefill prompt-length facts: a Jev-style prompt is mostly fixed text; a 7-token MASSIVE utterance sits in a 159-token prompt of which 42 tokens are the cross-request prefix (entry 20). VERIFIED (docs).
- `Qwen3 thinking mode left on` made order flip 1.000 across whole rows and a harness bug inverted Yes/No on `noul` (entry 20, "Mistakes made"): a real trap for us too (we use `enable_thinking=False`, SPEC 3.1, and a Yes/No label is the first-token collision case our design avoids). VERIFIED (docs).

## 9. Platform, install, API

- Windows: no mention in README, docs, pyproject, CI or CONTRIBUTING (grep over all for "windows", "rtx", "4090", "5090", "consumer", "mac", "mlx", "apple" returned only unrelated hits). CI is `ubuntu-latest` on Python 3.10 and 3.12 and runs `ruff` plus `pytest -q` (`.github/workflows/ci.yml`). The test suite runs on a `FakeBackend`, so CI never loads a real model. VERIFIED.
- Hardware in every published latency table: one H100 NVL, bf16. Qwen2.5-7B vLLM run used vLLM 0.7.0. A 2026-09-26 changelog line says a user hit a segfault on Apple Silicon through `device_map`, which a contributor fixed (issue 5, CREDITS.md and CHANGELOG), so Mac works at least for loading. VERIFIED.
- Install: `pip install "anyjev[hf]"` pulls `torch>=2.3` and `transformers>=4.53` (pyproject.toml). `bench` extra needs `datasets` and `scikit-learn`. A transformers 5 break was reported and fixed (CHANGELOG, issue 4). On Windows with Blackwell there is no statement; whether `cu128` wheels work is UNVERIFIED but nothing in the code is OS specific that I found.
- API: Python library only (`Decider.decide`, `decide_batch`, `fit_head`, `observe`, `calibrate`, `export_artifacts`). Not TypeSafe `/v1/systemone` compatible; a Jev-compatible `POST /v1/decisions` server is "Next" in the roadmap and not built. No MCP, no JSONL event stream, no console. VERIFIED (grep for "event", "jsonl", "stream" in `anyjev/`: none).
- Demo: `space/app.py` is a Gradio page that shows raw vs L0 on the same state with options in typed and reversed order, bars that move vs bars that hold still (docstring). A side-by-side "raw vs fixed" idea like our two-column race view, but it is a static comparison of two orders, not a live log. VERIFIED.
- `demo/jev_mode.py` runs the fit, route and adapt story on a CPU `FakeBackend` in under a second (README.md:178-179). VERIFIED (docs).
- Roadmap items that touch our claims, all unticked (ROADMAP.md): span readout for more than 26 options with `sequence_logprobs` (teacher-forced option strings, PMI correction); Jev-compatible HTTP server; SGLang, llama.cpp, MLX and Ollama backends (help wanted); conformal abstention; Llama, Gemma, Mistral, DeepSeek, Phi-4 and gpt-oss heads (help wanted); an HF Space on ZeroGPU; heads on the HF Hub; a technical report. `sequence_logprobs` appears nowhere in code (`grep -rn sequence_logprobs`: no hit). VERIFIED.
- Also on the roadmap: "Real agentic evaluation" and a "chained-decision bench" (a 95% judge applied 20 times is 36% end to end). Both are unticked. VERIFIED.

## 10. Ideas worth reimplementing in MirethSTM1

Founder rule applied: take the fact and the shape, never the expression. None of the below needs a line of AnyJev code, and nothing here is a quote of their code. Ranked by impact on speed and accuracy, then effort. Effort in engineer-days.

1. Reverse-order flip rate as a first-class benchmark metric (impact: accuracy claims, effort 0.5 to 1).
   - Fact: a position-bias number defined as the share of items whose argmax changes when the option list is reversed, reported alongside accuracy, ECE and Brier, for every readout. It is the number the clone READMEs admit to and nobody tabulates (docs/levels.md:164-168).
   - Shape for us: run each benchmark item twice (listing reversed) and report the flip rate for full-label scoring and for a letter-readout fast path in the same table. This tests our own differentiator directly. It is the single most testable claim in file 09.
2. Canonical option listing (impact: determinism, effort 0.5).
   - Fact: sorting the options by their text before building the prompt makes the output a function of the option set, so any re-listing returns identical probabilities (docs/levels.md:66-72; entry 24).
   - Shape for us: optionally sort options by name before rendering, map probabilities back to the caller's order. Cheap; also removes one source of "same question, different answer" bugs in a console.
3. Prompt layout selection per model and question (impact: speed, possibly large, effort 2 to 3).
   - Fact: moving the fixed question block ahead of the state lets a prefix cache keep it forever (5x to 10.4x fewer tokens on short states), but accuracy changed by -0.23 to +0.39 depending on model and task; `middle-state` was the safe layout on 9 of 10 cells but saves little (entry 20).
   - Shape for us: add a layout switch to the prompt builder (state first, schema first, middle) and measure accuracy and ms on our own benchmark with full-label scoring. Only useful for repeated schemas, which is the console demo and agent loops. This is the biggest speed lever that does not need a new model, but it is a hypothesis for our scorer.
4. Measurement protocol (impact: credibility, effort 1).
   - Fact: median over repeats with the spread printed, because on a shared machine one pass can show a configuration both faster and slower than baseline (README.md:89-93); never-seen states per run so a prefix cache cannot flatter the number (docs/results_latency.md, header); every result JSON records batch size, dtype, backend, versions and commit and the readout settings (CHANGELOG 0.2.0); a second run from a clean checkout.
   - Shape for us: the benchmark harness records hardware, torch, transformers, model hash, layout, batch tokens, and seeds in the JSON; report ms as median plus min and max over at least 5 runs; separate "single request" from "batched". Never publish the 68 to 270 ms M4 Max numbers as a target without our own same-protocol run on the 5090 and the Mac.
5. Certified label-free agreement for a cheaper configuration (impact: speed, effort 2 to 4).
   - Fact: a cheaper readout can be accepted if its disagreement with the full-strength answer on unlabelled states is below a target by a Clopper-Pearson 95% upper bound; with zero disagreements in 24 states the bound is only 12%, so a small calibration set refuses to certify (entry 24).
   - Shape for us: certify a small model (for instance Qwen3-0.6B or 1.7B) against the 4B default on unlabelled states the user supplies, and let the console say "this cheaper model agrees with the big one on at least 99% of your traffic at 95% confidence" or refuse. That also answers the founder's "plug in different models" request with a safety check, not a promise.
6. Labelled-head mode (L2-like probe) as an opt-in later (impact: accuracy for recurring schemas, effort 4 to 6).
   - Fact: a closed-form shrunk-LDA or ridge head on the last-position hidden state at about two thirds of the depth, selected by out-of-fold NLL with a temperature on out-of-fold scores, reaches +0.12 to +0.24 accuracy over L1 on typed-decisions with 300 labels per question, 0.740 with 100 on the 8B (entries 5, 6).
   - Shape for us: a "fit" command that takes labelled examples for one schema and stores a small file; at inference one forward stopped at the chosen block. It does not fit the generic no-label request. Treat as a post-v0.1 feature, and benchmark it on our own full-label scorer rather than assuming their gain. Without labels it adds nothing, so it must not appear in the zero-config path.
7. Early-exit forward and truncated checkpoints for speed (impact: speed 0.67x to 0.84x of one forward, effort 3 to 5, only with a head).
   - Fact: stopping the forward at about two thirds of depth gives 0.67x to 0.70x the time on 4B and 8B (docs/results_exit.md:13-19). The logit lens at those depths is at chance (accuracy 0.25 to 0.40), so early exit needs a trained head (docs/results_exit.md, "Accuracy versus block").
   - For us: this does not apply to the unembedding readout, since our scored tokens come from the last layer. Only relevant if idea 6 ships.
8. Launch-bound floor and CUDA graphs (impact: single-request latency, effort 2 to 3).
   - Fact: eager transformers has about 1.4 ms per block of launch overhead and a roughly 50 ms single-request floor up to 8B at short states (docs/results_latency.md; entry 6). That floor swamps the model compute for small models.
   - For us: our SPEC says no torch.compile and no Triton. If the founder's goal is a latency that beats 68 to 89 ms for tiny schemas, the floor is likely the first wall; the batched-vs-single gap here (4.2 vs 42 ms for a 1.7B at 110 tokens) is the size of prize. UNVERIFIED on our stack.
9. Batch label-prior correction at strength 0.75 for `noul`-like binary questions (impact: accuracy on skewed tasks, effort 1 to 2).
   - Fact: dividing out the model's mean label distribution over real inputs, raised to a power below 1, gave +1 to +2 points and large ECE gains over 230 model-question points but hurts when the true label marginal is skewed (docs/levels.md:27-38; docs/when_l0_helps.md).
   - For us: full-label scoring has no letter prior, but a true/false prior still exists. Add as an option, off by default, and measure; never as a silent default.
10. Honest framing items (effort 0.1): label Jev's 0.727 as "published by a third party, not re-run"; say accuracy on typed-decisions is agreement with a teacher; state ECE bin count and definition next to every ECE number.

## 11. Threats to our claims

- Full-label scoring as the headline. AnyJev has no such path today (readout.py refuses multi-token labels, `LabelTokenError`), so our claim stays open. But the planned span readout has a PMI correction and a 77-way target, and AnyJev has an engine, a benchmark harness, a calibration stack and a contributor base (127 forks) to ship it. If it lands, the "only decoder engine" wording becomes false. Do not write "only" or "first"; write "tested on". Check the roadmap again before release.
- Calibration as a selling point. AnyJev already publishes ECE with a stated bin rule, with intervals on accuracy, before and after, for 11 models. Our calibration (single temperature, SPEC section 9) is a subset of their L1. Position it as parity and be careful with numbers: their ECE uses 15 equal-mass bins, our definition must be stated and ideally the same to compare.
- Speed. Raw readout in AnyJev is one prefill, same order of cost as ours, and L2 is 0.67x to 0.84x of one forward. For "lightning fast" we cannot beat a labelled head on the same GPU; we can only beat their L0 (K prefills, 4.4x to 17.8x raw at K=20). If the speed pitch is "faster than 20 rotations", it is true; if it is "faster than any open alternative", it is not, because raw and L2 are faster than full-label scoring on the same model.
- Accuracy. On typed-decisions, raw letter readout on Qwen3-4B is only 0.547 and L0 0.564 (docs/results_typed.md). A full-label scorer on a stock 4B could well beat that, but we have no number; we should run the same 2,000 decisions, since AnyJev's dataset choice is already the public comparison point (file 09).
- Windows and NVIDIA. AnyJev says nothing about Windows; our claim holds only if we ship exact commands and a recorded test. Do not claim they cannot run on Windows; nothing in their code is OS specific that I found. UNVERIFIED either way.
- The console and event log. AnyJev ships a two-column Gradio comparison of two orders but no event stream, no live console, no TypeSafe-compatible server. Our race view and JSONL stream stay a small differentiator. Their Space shows the right instinct (bars that move vs bars that hold still); our two-sided live log is a different, richer thing.
- Model plug-in. AnyJev's `HFBackend` takes any HF causal LM, and the backend protocol is one method. Our "plug in different models" is not unique; the unique part is certified comparison of models (idea 5) and per-model temperatures.
- Community risk: the authors invite help on engines and model rows. An MLX or llama.cpp backend could appear any week and make a Mac story. UNVERIFIED timing.

## 12. Open questions for the lead engineer

- Does full-label scoring actually flip less than a letter readout under reversed listing on the same model and items? Run the same banking20 and typed-decisions items through both paths and report flip, accuracy, ECE, Brier. This is the one experiment that would turn file 09's differentiator into a number, and it lets us answer AnyJev's strongest table with our own.
- Does a schema-first layout hold accuracy for our scorer? If yes it is the cheapest speed win for repeated schemas.
- What is our single-request floor on Windows with a 5090 for a 1.5B to 4B model? Their eager floor of about 50 ms suggests the first thing to measure is launch overhead, not FLOPs.
