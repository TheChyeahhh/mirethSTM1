# 13. Deep read: open-alternative-jev, jev-style, Verdict-open-jev

Date read: 2026-09-30. Repo facts come from `gh api repos/OWNER/REPO` on that date. Code was read in local clones at the commits named below. Tags: VERIFIED (read in code, a result file, or a fetched page), UNVERIFIED (secondary source or not accessible), CONTRADICTED (a claim the project makes, or an earlier note of ours made, that the code or its own data refutes).

No GPU code was run, nothing was installed, no paid API was called. Hugging Face pages were fetched read-only.

Earlier pass: file `09-landscape-and-differentiators.md`. This file corrects three items in it (section 5).

## 0. One-screen summary

| | ikermoel/open-alternative-jev (`so1`) | lawrence3699/jev-style | Heman10x-NGU/Verdict-open-jev |
|---|---|---|---|
| What | In-process Python library, stock decoder LLM, letter readout | Fine-tuned Qwen3.5 0.8B / 2B, local server, MCP, Claude Code guard | Fine-tuned 151M ModernBERT/GLiClass encoder, ONNX, WebGPU demo |
| Stars / forks / last push | 56 / 11 / 2026-09-25 | 9 / 0 / 2026-09-27 | 109 / 14 / 2026-09-28 |
| License | Apache-2.0, full text, SPDX matched | Apache-2.0, full text, SPDX matched; weights Apache-2.0 with data caveats | Apache-2.0 intended, LICENSE text abridged, SPDX `NOASSERTION` (section 4.3) |
| Scoring | One letter token (A to Z) at a read position | Trained yes-minus-no logit at a marker after each option | Trained encoder logits over label slots |
| Plug in other models | Yes, any ChatML-style HF or vLLM model | No (own releases only) | No |
| Accuracy / calibration published | Yes, with scripts and raw results | Yes (model cards, own harness) | Yes (receipts), plus an unflattering 48% external number |
| Windows | Not stated, ubuntu CI only | README says Windows CUDA works; CI is ubuntu and macOS; skill text says "Windows/WSL" | Not stated |
| Threat to our claims | Medium: owns the same benchmark numbers we want, but letter-based | High on the headline "real label text": a trained model also avoids letters; Low on "stock model, plug in any" | Low |

## 1. ikermoel/open-alternative-jev (commit 3689c67, 2026-09-24)

### 1.1 What it is
- Python package `open-alternative-jev`, import name `so1`. In-process library only, no HTTP server and no TypeSafe-compatible endpoint. README: "there is no HTTP server or drop-in API for the official Jev SDK" (README.md line 18-19). VERIFIED.
- About 630 lines of library code (`so1/decider.py` 137, `so1/prompting.py` 124, `so1/backends/hf.py` 79, `so1/backends/vllm.py` 88, `so1/calibration.py` 67, `so1/schema.py` 73). VERIFIED by `wc -l`.
- Open source: code and benchmark scripts and raw result files are in the repo (about 27 MB). No weights (uses stock Qwen). Datasets are rebuilt by script from pinned revisions; RACE is not committed because of its research-use terms (README "Reproduce"). VERIFIED.
- License: Apache-2.0, full text in `LICENSE`, `pyproject.toml` says `Apache-2.0`, GitHub SPDX `Apache-2.0`. VERIFIED.

### 1.2 Models
- Any causal LM with a ChatML template (README "Backends": "Any model with a ChatML template (Qwen, and many fine-tunes) works out of the box"); other templates need a `ChatFormat` with a user-turn start string and a separator string (`so1/prompting.py` lines 33-38). VERIFIED. This is the only one of the three projects that lets a user plug in other models.
- Models measured: Qwen3-0.6B, Qwen3-1.7B, Qwen3.5-2B, Qwen3.5-4B, Qwen3.6-27B (8-bit via bitsandbytes), Laya as a comparison. All stock, zero-shot, no training. VERIFIED from `benchmarks/results/td_*/summary.json`.
- No Qwen2.5-1.5B run exists. The test suite uses Qwen2.5-0.5B-Instruct (`.github/workflows/ci.yml` line 11). So nothing here reproduces the original demo model. VERIFIED.

### 1.3 Scoring mechanism (the important part)
- Prompt per question: a ChatML user turn, "Choose the correct option. Reply with only its letter.", the state, the question, and options as `A. option` lines (`so1/prompting.py` lines 29-61). The answer is read from the next-token logits restricted to the letters A to Z (`label_ids`, lines 77-86, which raises `ValueError` if a letter is not a single token). VERIFIED.
- So: first-token, single-letter scoring. Maximum 26 options (`so1/schema.py` lines 7-8, 24-25). Real option text is never scored. This confirms the 09 finding for this project. VERIFIED.
- Packed mode: all questions of one state go into ONE sequence as successive chat turns, each followed by a fixed placeholder answer `_<|im_end|>` (`so1/prompting.py` lines 106-120, separator at line 36). Plain causal attention: question 2 can see question 1 and the placeholder. There is no tree mask, no per-question isolation. VERIFIED. The authors say so themselves: "Later questions can attend to earlier questions" (prompting.py lines 16-19) and measure it (below).
- One forward pass, logits taken only at read positions via `logits_to_keep` (`so1/backends/hf.py` lines 71-79). VERIFIED.
- Separate mode: one sequence per question. On vLLM it uses prefix caching plus `allowed_token_ids` (`so1/backends/vllm.py` lines 224-228). On HF it just re-reads the state per question. VERIFIED.
- vLLM packed mode reads `prompt_logprobs` top-20, and a letter outside the top 20 gets a floor score (`so1/backends/vllm.py` lines 194-204). It is approximate by construction, counted in `missing_labels`. VERIFIED.
- Batching on HF: pads to the longest sequence in a batch of 8 (`so1/backends/hf.py` lines 57-62). The authors found padding waste, not packing, explained their first speed claim, and wrote that up (README, "The correction that made this README honest"). VERIFIED in README; the corrected analysis is in `benchmarks/docs/RESULTS.md` (not re-derived by me).
- Position-bias control: `permutations=k` asks each question with the options in several orders (original, reversed, cyclic shifts) and averages the probability vectors, mapped back to the original order (`so1/decider.py` lines 13-28 and 85-106). VERIFIED.

### 1.4 Speed (every number, with conditions)
Hardware is an NVIDIA H200 MIG slice. The README names "one H200 MIG slice with 35 GB" for the 27B runs. The metadata of the RACE runs records `NVIDIA H200 NVL MIG 2g.35gb` (`benchmarks/results/race_qwen3-0_6b_32860835/metadata.json`) and the library comparison used 3g.71gb. The typed-decisions runs record no GPU in `summary.json`, so the slice for those is UNVERIFIED.

Method: `time.perf_counter()` around `runner.predict(...)` per case, p50 and mean over 400 cases, after one warm-up case (`benchmarks/scripts/typed_decisions.py` lines 232-243). This is wall-clock and includes Python prompt building, tokenization and host copies. One "case" = one state + 5 typed decisions. No CUDA synchronize is called around it, but `.cpu().tolist()` in the HF backend forces sync (`so1/backends/hf.py` line 67). VERIFIED.

typed-decisions, p50 ms per case (5 decisions), values re-read from `summary.json` and equal to the README table:

| Model | Engine | Mode | p50 ms/case | Source dir |
|---|---|---|---:|---|
| Qwen3-0.6B | HF bf16 | packed | 46 | `td_qwen3-0_6b_packed_32768218` |
| Qwen3-0.6B | HF bf16 | separate | 62 | `td_qwen3-0_6b_separate_32768218` |
| Qwen3-0.6B | vLLM bf16 | packed | 71 | `td_qwen3-0_6b_vllm_packed_32860694` |
| Qwen3-0.6B | vLLM bf16 | separate | 14 | `td_qwen3-0_6b_vllm_separate_32860694` |
| Qwen3-1.7B | HF bf16 | packed / separate | 55 / 95 | `td_qwen3-1_7b_*` |
| Qwen3.5-2B | HF bf16 | packed / separate | 85 / 180 | `td_qwen3_5-2b_*` |
| Qwen3.5-4B | HF bf16 | packed / separate | 105 / 229 | `td_qwen4b_*` |
| Qwen3.6-27B | HF 8-bit | packed / separate | 582 / 1234 | `td_qwen27b_*` |
| Jev 1.13.0 | TypeSafe API | n/a | 710 | not measured by them |

- The 710 ms and the 72.7 % / ECE 0.144 for Jev are on the dataset card of Luni/laya-jev-benchmark (fetched, "Accuracy 0.727, ECE 0.144, latency 710 ms per case"). VERIFIED. The card as fetched gives no KL or Brier for Jev, so the README's "KL 1.44" and "Brier 0.148" for Jev have no source I could see. UNVERIFIED.
- Library comparison, Qwen3.5-4B, RACE-H, 100 passages x 4 questions (`benchmarks/results/lib_race_32698399/summary.json`): HF separate 20.0 q/s, HF packed 49.6 q/s, vLLM separate 71.9 q/s, vLLM packed 38.7 q/s. VERIFIED. Takeaway the authors state and the data supports: with a prefix cache the exact separate mode is fastest on vLLM; packing only wins on engines without prefix reuse.
- Incomparable with the original demo (stock 1.5B, M4 Max, 68 to 89 ms): different GPU class, MIG slice, 5 questions per case, wall-clock with Python overhead, and the 27B is 8-bit with PyTorch fallback kernels for the hybrid layers (the README says so: "Not a speed record"). The closest stock-size rows are Qwen3-1.7B at 55 ms (HF packed) and 95 ms (separate).

### 1.5 Accuracy and calibration
- Benchmark: `LocalLLaMA/typed-decisions`, 400 test cases, 2,000 decisions, gold built from a teacher model. Scoring is a port of the Luni scorer (`typed_decisions.py` lines 118-169); they say it reproduces Laya's published 0.766 and ECE 0.213 exactly, and their Laya rerun gives 0.766 and 0.213 (`td_laya_ft_cpu_mac`) and 0.769 and 0.216 (`td_laya_ft_32768027`). VERIFIED against result files.
- Raw (T = 1) results, all VERIFIED from `summary.json`:

| Model | Acc | ECE | Brier | KL |
|---|---:|---:|---:|---:|
| Qwen3-0.6B packed | 0.291 | 0.534 | 0.672 | 2.25 |
| Qwen3-1.7B packed | 0.459 | 0.509 | 0.682 | 4.25 |
| Qwen3.5-2B packed | 0.472 | 0.124 | 0.269 | 0.50 |
| Qwen3.5-4B packed | 0.593 | 0.118 | 0.164 | 0.38 |
| Qwen3.6-27B 8-bit packed | 0.737 | 0.020 | 0.113 | 0.27 |
| Qwen3.6-27B 8-bit separate | 0.727 | 0.063 | 0.120 | 0.36 |

- Read this carefully. A stock 1.7B (the nearest size to the original 1.5B demo) gets 45.9 % with ECE 0.509 and KL 4.25 on this benchmark. Stock sub-2B models are badly calibrated and weak on typed decisions. Accuracy parity with the original demo's model on this benchmark is not a good target for a letter method.
- Temperature scaling does not uniformly help. For the 27B, T = 2 fitted on 200 train cases against the teacher's soft labels improves Brier (0.113 to 0.074) and KL (0.27 to 0.15) but raises hard-label ECE from 0.020 to 0.135 (`td_qwen27b_packed_cal_32768027`, matches the README). The headline "ECE 0.020" is the raw, uncalibrated value. The authors disclose this. VERIFIED. For us: report Brier and KL beside ECE, and say which target T was fitted to.
- Option order: reversing options moves the 4B's yes/no accuracy by 13.5 points (76.7 % to 63.2 %) and averaging both orders halves its ECE (0.118 to 0.062); 27B ECE 0.020 to 0.0075, accuracy 73.7 to 75.5. Cost 1.7x latency (576 to 969 ms). README only; I read the code path but did not recompute from `oo_*` results. UNVERIFIED numerically, mechanism VERIFIED.
- Interference (packed vs separate): share of answers that change when packed: 36.9 %, 20.0 %, 13.9 %, 7.6 %, 3.9 % from 0.6B to 27B, with a padding-only noise floor of 2.7 % (README). Accuracy falls 7.5 points at 0.6B and 1.8 to 1.9 points at 1.7B and 2B packed vs one-at-a-time on RACE-H. README only, not recomputed. UNVERIFIED numerically. The direction matches the typed-decisions result files (0.6B packed 0.291 vs separate 0.389). VERIFIED for that one row.
- ECE definition: 10 equal-width bins in two places (`so1/calibration.py` lines 96-105 and the benchmark script). The authors also note a base-rate predictor scores ECE 0.088 on this benchmark and cite KL and Brier as the guard against gaming. VERIFIED in README.

### 1.6 Platform
- Pure Python on torch plus transformers (`pyproject.toml`: torch>=2.1, transformers>=4.51, accelerate>=0.30). No flash-attn, no custom kernels. README: "Works on CPU for small models". CI is ubuntu-latest only, Python 3.10 and 3.13, with a weekly run against unpinned upstream versions (`.github/workflows/ci.yml`, `upstream.yml`). Windows: not mentioned anywhere. UNVERIFIED whether it runs on Windows; nothing suggests it would not. 8-bit loading needs bitsandbytes (`[quant]`).
- A hand-built ChatML string plus position arithmetic means transformers template changes are correctness changes; they pin versions and test it (`tests/test_prompting.py`). Good practice.

### 1.7 API
No HTTP, no MCP, no console. Python objects `Choice`, `yes_no`, `scale`, `Decision` (`so1/schema.py`). Yes/no is a two-option choice with letters. No score expected value (a `scale` is just integer-labelled options). A HF Space demo exists (Qwen3.5-4B). `demo/app.py` is a Gradio app. Not TypeSafe wire-compatible.

### 1.8 Ideas worth reimplementing (our own code, from the fact and the shape only)
1. Order-permutation averaging as an option (`permutations=k`, cyclic shifts, reversal first). Effect measured on stock models: halves ECE on the 4B, fixes a 13.5 point order gap. Our full-label scorer has no letters, but the order of options inside the prompt still matters, and we have not measured it. Cost for us: a second prefill (the prompt changes), so about 2x latency; keep it off by default. Effort: small (a day), plus a benchmark row. Impact: accuracy and calibration, not speed.
2. Noise-floor control for any packed-vs-separate claim: run the same prompt right-padded (`A_pad`) to separate kernel noise from real interference. Our design (SPEC 3.4) gives each label only the prefix and its own tokens, so cross-question interference should be zero by construction, which is a testable claim that this project's packed mode cannot make. Effort: small. Impact: credibility of the headline.
3. Report Brier and KL against soft gold next to ECE, state which objective T was fitted to, and show raw and fitted ECE side by side. Effort: tiny.
4. Decision for the race view: the data show the prefix-cached "separate" mode is both exact and fastest on vLLM. Our single-pass packed full-label pass on plain HF is the equivalent of their packed mode. A fair race view should include a "prefill once, score per question" path as the exact baseline.
5. Vendor-neutral model plug-in: a small `ChatFormat`-style hook (user-turn start string and separator) is all they need for non-ChatML models. Ours uses `apply_chat_template` and is more general; the pinned-version test of template behaviour is worth copying as a shape.

### 1.9 Threats to our claims
- Medium. It already owns published numbers on `typed-decisions` for stock Qwen from 0.6B to 27B, measured with the community scorer. We must report on the same benchmark with the same scorer or we cannot be compared.
- It does not threaten full-label scoring (letters only, 26 options), Windows/NVIDIA (nothing stated), or the race view (no console).
- Its honest finding that stock models below about 2B are poor on typed decisions applies to us too. Qwen3-1.7B scoring 45.9 % there means a "match the demo model" run is a speed match, not an accuracy match.

## 2. lawrence3699/jev-style (commit 1d86451, 2026-09-27)

### 2.1 What it is
- A package that ships a local FastAPI server, a client, an MCP server, a Claude Code PreToolUse guard, an eval CLI, a Playground and six agent skills (`jev_style/`, `skills/`, `plugins/`). About 3,000 lines of Python in `jev_style/` (server 122, adapter 141, schema 210, models 288, evaluate 474, guard 693, mcp_server 256). VERIFIED by `wc -l`.
- The models are fine-tuned weights hosted on Hugging Face (Jev-Style-0.8B-Decision-v3 and 2B-v3, each in torch, MLX and GGUF builds). The scoring code lives in a Python file shipped next to the weights on the Hub, not in this GitHub repo. `jev_style/models.py` downloads a pinned revision and imports that file (`import_runtime`, lines 237-248). Anything else needs `trust_remote_code=True` (lines 198-210). VERIFIED. I read the 0.8B runtime directly from the Hub (`jev_style_decision.py`, 603 lines).
- Open source: code Apache-2.0 (LICENSE full text, SPDX `Apache-2.0`, `NOTICE` file). Weights Apache-2.0 (card front matter `license: apache-2.0`), base Qwen3.5-0.8B Apache-2.0. Training data is mixed: the card lists sources with their own terms (some non-commercial research-only) and says an OpenAI model and Claude models generated or labelled part of the data and that "providers' terms of use may restrict" use of models trained on such output (model card, "Outputs of other models"). The training data and training script are not in the repo. So: open code, open weights, closed training recipe, data with caveats. VERIFIED from card text.

### 2.2 Models, plug-in
- Qwen3.5-0.8B (752M parameters, 24 layers) and Qwen3.5-2B, full fine-tuned in bf16 on a pool of 321,756 rows over 19 languages, about 96 minutes on one H100 for the 0.8B (card, via fetched summary and README; UNVERIFIED in detail, the data is not in the repo).
- No plug-in of other models in practice: the readout needs weights trained for it. A `NAME=hf:<repo id>` engine spec exists in `jev-style eval` for comparison only (README "Compare engines"). The server accepts a `model` field and ignores it (`jev_style/schema.py` lines 23-24).

### 2.3 Scoring mechanism
- "Verdict slot" readout (runtime header and card): the prompt lists the options, then `Judge each option:` and then one line per option of the form `<option> ->`. The score of option k is `logit(" yes") - logit(" no")` at the ` ->` token that ends option k's line, computed as `h_k . (w_yes - w_no)` from the final normalised hidden state and the tied embedding rows in float32. Probabilities are `softmax(scores / T)` (runtime lines 1-34, `_result` lines 384-399, and `fastpath.py` line 69 which does the same dot product with `runtime.direction`). VERIFIED.
- All options of a question are read in one forward pass over one sequence with ordinary causal attention; each option is judged knowing all the others listed above it. The option text appears twice in the prompt (once in the list, once before the arrow) (`Renderer.render`, runtime lines 234-262). VERIFIED.
- No letters, no first-token collision, no 26 cap: 255 options accepted by the API (`jev_style/schema.py` line 40). Option lists over the 2,048-token head budget are split into chunks, each chunk repeating the question, and one softmax runs over all options (`plan_option_chunks`, runtime lines 192-232). 77 options is the largest set evaluated in one head (card). VERIFIED / card claim.
- This is a trained model, so it is a different thing from a stock-model method. The mechanism is, however, directly relevant: it reads the real option text without letter indirection, which is the very thing our SPEC 3 claims as the differentiator.
- Calibration: a global T of 0.880 (below 1, so raw scores are slightly flat) and 20 fitted group temperatures keyed by family, question type and option-count bucket, clamped to 0.3 to 5.0 (`lookup_temperature`, runtime lines 285-293; card). With no `category` the global T is used. VERIFIED.
- Budgets: 25,600 tokens in total, 2,048 for question plus options plus readout, over-budget raises `InputBudgetError` and nothing is truncated (runtime lines 24-30; `adapter.py` lines 164-165 return HTTP 422 `input_budget_exceeded`). VERIFIED. This is a good contract.

### 2.4 Batching and speed
- Many questions about one state: the MLX runtime pre-fills whole 2,048-token blocks of the state once into a prompt cache, deep-copies it per question and runs the remainder. `fastpath.py` docstring: identical probabilities (max difference 0.0) and 5.1x / 8.0x faster at 6.7K / 18.9K state tokens, 14 questions, M1 Max. A variant that reads the whole state once changes scores "up to 0.11 in probability and 3 of 14 top answers at 18.9K tokens, so it is not offered" (`jev_style/fastpath.py` lines 1-12 and 28-35). VERIFIED in code comments; the benchmark itself is not in the repo, so the numbers are UNVERIFIED. Shape worth noting: chunked prefill block boundaries change the numerics, so "prefill once" is not automatically score-identical.
- The GGUF runtime shares the state in whole 1,024-token ubatches; a `many_mode="batched"` mode reads the state once and scores all questions together, differing from exact mode by at most 0.002 in their tests (card, "Several questions about one state"). Card claim.
- Published latencies, all Apple M1 Max, warm p50, prefix reuse off:
  - README example: 194.0 ms for a 250-token request with 3 questions, 0.8B on MLX bf16 (README API example, `latency_ms`). The README prose says "about 0.15 to 0.2 s for a short request". VERIFIED as published, not reproduced.
  - Card: GGUF F16, 4K-token state with 10 questions 1,381 ms vs 6,364 ms for their own Laya-architecture comparison engine (FP32 on Apple MPS, one call per question). That comparison engine is their own fine-tune, "not an official Laya checkpoint", and the model is an untrained export (timing does not depend on weights). Card claim, UNVERIFIED (chart data not in this repo).
  - Guard: 443 ms on one `--check` example (README).
- No NVIDIA latency number is published anywhere I read. For a Windows/NVIDIA claim this is a gap, not evidence.
- The 0.8B has 24 layers, 18 of them Gated DeltaNet (card, "Weights"); the plain KV-cache machinery of our SPEC 3.4 does not apply to such models, same note as in `06-models-and-licenses.md` for Qwen3.5.

### 2.5 Accuracy and calibration
All below are from the 0.8B/2B cards and the README, with the project's own harness. UNVERIFIED independently; the harness and models are not in this repo.
- JevBench v1.4.1, 231 public items, self-run: 2B 73.6 %, 0.8B 64.1 %, hosted Jev 86.6 %. The authors say 42 of 82 board systems score higher and the lead over decider-2b (71.0 %) is inside the 95 % interval (README).
- typed-decisions: the card's v3 0.8B teacher agreement is 79.15 % but that is IN-DOMAIN (v3 trained on the train split), so it is not comparable with the zero-shot numbers in `so1` (73.7 % for a stock 27B). The card says so ("In-domain for v3 and Laya typed") and adds a teacher-noise analysis (one teacher draw matches gold 65.9 %). Card text, VERIFIED as written.
- tweet_topic zero-shot: 0.8B 75.5 % vs Jev 79.3 % (Jev number taken from a third-party study, not re-run), ECE 0.027 vs 0.063 (README and card).
- `jev-style eval` reports accuracy, Brier, log loss, ECE (15 bins in the card's study, ECE function in `evaluate.py` lines 132-143), the share of decisions automatable at 1, 5, 10 % error budgets with the threshold (lines 145-163), a temperature refit on half the rows applied to the other half, and paired bootstrap differences between engines (lines 274-305). VERIFIED in code.
- Guard: 77.6 % agreement with 49 hand-labelled tool calls, model alone 61.2 %, 2 of 16 "ask" calls were allowed, no "deny" call allowed (README). Tiny, hand-written, single-author set. UNVERIFIED beyond the README.

### 2.6 Platform and install
- Backends: MLX (Apple), PyTorch (CUDA, MPS, CPU), llama.cpp GGUF via a scorer binary the user builds once (`jev_style/models.py` lines 8-17; README "Backends").
- Windows: README line 82 says `pip install "jev-style[torch]"  # Linux, Windows or an Intel Mac`. But `pyproject.toml` classifiers list only macOS and Linux, CI is `ubuntu-latest` and `macos-latest` only (`.github/workflows/ci.yml` lines 11-14), tests run on a fake engine with no model, and the serve skill text says "macOS, Linux or Windows/WSL" (`skills/jev-style-serve/SKILL.md` line 3). So a Windows install is claimed, not tested; the earlier note 09 called it "documented", which is accurate, but "Windows CUDA" is UNVERIFIED. PyTorch default dtype is float32 (`models.py` `load_release`, `dtype="float32"`), so an RTX 50-series run would be fp32 unless overridden.
- Remote code execution by design: the server imports a Python file from the Hub at the pinned revision (see 2.1). The manifest hash check is opt-in (`verify=False` default, `models.py` line 224).

### 2.6b API and tooling
- TypeSafe-compatible: `POST /v1/systemone` with `state` and `questions` (`noul` P(true), `choice` 1 to 255 options, `score` 2 to 10 levels, expected index plus legend plus probabilities), `GET /v1/models`, `/healthz` (`jev_style/server.py` lines 305-321; `schema.py`; `adapter.py` lines 176-188). VERIFIED. It also returns `confidence = (k * p_max - 1) / (k - 1)` (`adapter.py` lines 105-109), and `usage` and `latency_ms`.
- Differences from our SPEC 2: a score question allows only 2 to 10 levels; `confidence` is a rescaled top probability. It accepts and ignores the `model` field.
- MCP server (stdio, forwards to the running local server): tools `decide`, `noul`, `choice`, `score`, `model_info`. Listed in the official MCP registry per README. Local and keyless once the server runs. Contradicts the "existing MCPs need a paid key" premise, as 09 already found.
- Console: a browser Playground and three demos (agent approval, Snake, Chinese). No live log or race view. No JSONL event stream.
- `evaluate.py compare` can run the same JSONL through several engines (including any other `/v1/systemone` server) and prints a paired-bootstrap difference table. This is a comparison tool for engines, so it overlaps our race view in purpose but not in form.

### 2.7 Ideas worth reimplementing
1. Verdict-slot readout as an OPTIONAL scoring mode in our scorer, tested on stock models first. Mechanism: after the options list, repeat each option followed by a marker, read `logit(yes) - logit(no)` at each marker in one pass, softmax with T. It needs no letters and no per-label suffix copies, so it removes our label-length term entirely, but stock models were not trained for it, so I expect it to be weaker than full-label scoring on Qwen3 stock. It is cheap to test (the prompt shape and a readout at marked positions; no cache tricks) and gives a head-to-head table: full-label sum of log-probs vs slot verdict vs letter. Effort: 1 to 2 days including the benchmark rows. Impact: accuracy and a clean answer to "why full labels".
2. Per-group temperatures: key T by question type (choice, noul, score) and option-count bucket instead of one global T. They fit 20 groups on held-out rows; the README for the 0.8B shows global T 0.880. Effort: small. Impact: calibration; cheap to add in our fit script. Needs enough rows per group to avoid overfit.
3. Automation-at-error-budget metric (largest coverage at 1, 5, 10 % error with the probability threshold). It speaks to users better than ECE and is trivial to compute. Effort: tiny. Impact: how we present accuracy.
4. Hard budget errors instead of silent truncation: a typed 422-style error with the token counts and "nothing was truncated". Effort: tiny. Impact: honesty and correctness.
5. Option chunking for lists above a head budget with one softmax over all chunks. We already score up to 255 options packed; our equivalent is the batch_tokens split in SPEC 3.4. Their idea differs because chunks repeat the question; only worth noting as the way a verdict-slot mode would scale.
6. Paired-bootstrap engine comparison (`compare`): reuse the shape for our race view summary, so that "model A vs model B" always shows a confidence interval. Effort: small.
7. Guard pattern for Claude Code (questions plus code thresholds plus regex hard rules that can only tighten, never loosen; errors map to "ask"): not our v0.1 scope, but it is the best-built example of an agent-facing use and a reference for our later MCP and hook story.

### 2.8 Threats to our claims
- High for the headline "scores the full real label text, not a letter". jev-style also reads the real option text with no letter indirection, with no cap and with tested 77 options. The accurate claim for us is narrower: "among stock, plug-in decoders, full-label log-prob scoring is the only method that reads the real label text; the trained models do it with a verdict readout". Do not say nobody does this. This reinforces the 09 advice to not use "first" or "only" loosely.
- Medium for Windows/NVIDIA: it advertises Windows CUDA, untested.
- Low for the race view and the event stream: nothing like it.
- Low for "plug in other models": it cannot do this.

## 3. Heman10x-NGU/Verdict-open-jev (commit 30f1556, 2026-09-20)

### 3.1 What it is
- A Python library (`core/`, `rlcd/`, `openjev/`) around a 151M-parameter ModernBERT-base plus GLiClass bi-encoder head, post-trained with cross entropy plus Brier loss, plus a browser WebGPU playground (`webgpu-demo/`). `pyproject.toml` names the distribution `rlcd` (not `openjev`), with imports `from rlcd import DecisionEngine`. VERIFIED.
- Weights live on Hugging Face (`heman10x/rlcd-modernbert-151m`, safetensors and ONNX), not in the repo, fetched by `scripts/download_artifacts.py`. The page says weights Apache-2.0, base model `knowledgator/gliclass-modern-base-v2.0` (fetched page, VERIFIED as written; I did not read the base model's own license). Training data: Banking77 (PolyAI) and CLINC150 plus synthetic and "cookbook" data (README). Not read in detail.
- The repo includes data splits and many experiment receipts (`reports/v2/exp_e1..e9_*.json`, 12 receipt files). Roughly 41 MB tracked.

### 3.2 License, read carefully
- GitHub API returns `NOASSERTION` with size 3,027 bytes (`gh api repos/Heman10x-NGU/Verdict-open-jev/license`). VERIFIED.
- The `LICENSE` file (3,092 bytes in the clone; standard Apache-2.0 text is about 11 KB) opens with the Apache 2.0 title and the same section numbering, but is a hand-shortened rewrite: it has no definitions of Contributor, Derivative Works, Contribution or Object form beyond a few terms; the patent grant has no termination clause; the redistribution section has none of the conditions (give recipients a copy, keep notices, NOTICE handling); there is no copyright line naming a holder and no appendix. VERIFIED by reading it in full. Section 2 and 3 grants and sections 7 and 8 are present in short form.
- README and weights page both say "Apache 2.0" (README badge line 8). So: the intent is Apache-2.0, the text is not Apache-2.0 and is not recognised as such. An abridged text can differ in legal effect. For our purposes it changes nothing, because the rule is that we copy no code and no files from any of these projects. If someone proposes depending on or vendoring this repo, treat the license as unclear and ask the author for the verbatim text. The earlier note 09 said "Apache-2.0 per LICENSE file"; the precise statement is the one above.

### 3.3 Scoring mechanism and DAG
- Joint encoding: prompt string is `<<LABEL>>desc1<<LABEL>>desc2...<<SEP>>Question: ...\n\nContext:\n...` (`core/formatting.py` lines 20-65), with special tokens 50368 and 50369. Option descriptions are wrapped as `It is {description}`, and every choice and score question gets an extra `__insufficient_evidence__` candidate for abstention (`formatting.py` lines 81-109). The encoder is bidirectional, so every context token attends every label. The head emits one logit per label slot (`engine_encoder.py` lines 223-248). VERIFIED. Not an autoregressive log-prob; it does not compare with our scorer on mechanism.
- Per question one batch row: each question carries its own copy of the context, padded batch, one `model(**inputs)` call (`engine_encoder.py` lines 209-239). The "single call" is one batched forward, not shared state. VERIFIED.
- Silent truncation: `truncation=True, max_length=512` at the tokenizer call (`engine_encoder.py` lines 215-221), with a warning-free cut of long contexts; and choices beyond 24 options are silently truncated with a log warning (`formatting.py` lines 83-89, `engine_encoder.py` lines 190-197).
- Capacity: 24 options plus abstention, 25 logits. README "Known boundaries" item 7 says "Passing 26 or more candidates raises a typed `CapacityError`". CONTRADICTED: `CapacityError` is defined (`formatting.py` line 29) and imported, but no `raise CapacityError` exists anywhere in the repo; a `Choice` with more than 24 options is rejected by a pydantic `max_length=24` validator in `core/primitives.py` (line 52), and the engine path truncates. Minor, but it shows the README is ahead of the code.
- DAG executor: independent fields run in one stage, dependent fields in a second stage with the parent's selection appended to the context as `[STATE] ...` (`core/dag.py`, README). Not read line by line. Idea only; it is a handy shape for conditional questions.
- Calibration: temperature from `calibrator.json` loaded automatically, with a per-K temperature table, applied by dividing logits (`engine_encoder.py` lines 124-166 and 261-276). VERIFIED. The README's v1.4 note says earlier versions silently ran at T = 1.0 because the calibrator failed to auto-load. VERIFIED as a described fix.

### 3.4 Speed
- README claims "under 35 milliseconds" and "Sub-35ms Latency". The authors' own receipt `reports/v2/exp_e7_latency.json` says: ONNX Runtime 1.30.0, FP32, `macOS arm64` (chip not named), single thread, 20 warm-up and 200 timed runs, one question with 80 tokens at K = 5: p50 35.58 ms, p90 36.09 ms, p99 42.72 ms. At K = 3: 27.58 ms; K = 9: 49.04; K = 17: 92.16; K = 25: 140.10 ms (p50). So "under 35 ms" is CONTRADICTED at K = 5 by its own receipt (35.58), and false for K above 5. VERIFIED. The Hub page says "sub-40ms", consistent with the receipt for K <= 5.
- Method: ONNX session `run` only, with fixed short input (62 to 260 tokens). It excludes tokenization and Python. Latency scales roughly with K because label descriptions are part of the sequence. Not comparable to any of the end-to-end numbers above; the contexts are tiny (under 71 training tokens per README).
- Because a question's whole candidate list is encoded jointly with the context, 25 options cost 140 ms single-thread. The speed advantage vs a decoder disappears for large option lists.

### 3.5 Accuracy and calibration
- Primary held-out set (Banking77-style, 1,000 cases, 5 candidates): accuracy 95.00 %, ECE 1.13 % uncalibrated and 3.35 % after temperature scaling (fitted T = 1.4265), NLL 0.1731 vs 0.1768, 95 % bootstrap CI for accuracy 93.60 to 96.20 (README generated tables; receipts in `reports/v2/`). Note that the fitted T makes ECE worse (1.13 to 3.35) and NLL worse. So their own data show the temperature step did not help on the main split. The earlier note 09 listed "ECE 1.13 % ... fitted T = 1.4265" together without saying this. VERIFIED from the README table.
- Cardinality: accuracy 97 % at K = 3 falls to 72 % at K = 25, ECE 5.9 % to 12.6 % at K = 17 (README table).
- It is a Banking77-specialised model. On TypeSafe's public evals (337 cases) it scores 48.07 % vs 90.80 % for Jev and 88.43 % for a 26B comparison model (`reports/v2/exp_e9_external_cases.json`; choice 25.6 %, score 14.8 %, noul 61.4 %). VERIFIED. The 12-task cookbook sweep passes the gate on only 3 tasks (README matrix).
- The README's typed-decisions table lists a "Verdict Baseline (TF-IDF + LogReg)" row (ECE 0.0207, 66.10 %), not the neural model, next to a Jev accuracy of 68.00 % with ECE 0.1440. The Jev accuracy on the Luni dataset card is 72.7 % (fetched), so the 68.00 % is CONTRADICTED by the source the ECE comes from. UNVERIFIED where 68.00 came from. The neural model itself is not scored on typed-decisions in this README.
- Label-length check: Pearson r = 0.1027 between candidate token length and selection probability (README item 5, script `scripts/exp_label_length.py`). This is an encoder with description prefixes, so it says nothing about summed log-prob bias in our scorer. Do not cite it as evidence for us.
- Option-order: reversing options flips 3.00 % of choices, any permutation 4.50 %, concentrated at low confidence (README E2). Encoder with joint attention, so low order sensitivity is natural.
- Abstention: 97.5 % recall on out-of-scope in the main split, but it collapses to 18 % on hard negatives and 10 % under synonyms (README E5). Honest, and a useful warning about abstention heads.

### 3.6 Platform and API
- Python torch/transformers/gliclass/onnx/onnxruntime/pydantic (`pyproject.toml`); ONNX CPU path preferred when `model.onnx` is found (`engine_encoder.py` lines 87-122). A WebGPU/WASM browser demo (`webgpu-demo/`). Windows not mentioned, no CI directory in the repo (no `.github/` among tracked files). UNVERIFIED on Windows.
- API is a Python object model (`Choice`, `Score`, `Noul`), not TypeSafe wire-compatible; no server, no MCP in the repo. "Inference receipts" are exported as JSON by the playground.

### 3.7 Ideas worth reimplementing
1. Explicit abstention candidate (`insufficient evidence`) as an optional extra label in our `choice` scoring, reporting P(abstain). For us it is just one more candidate text scored in the same pass; no training. Test whether a stock model uses it sensibly; their own data warn it does not generalise across paraphrases. Effort: tiny to add, a benchmark row to judge. Impact: usability, not speed.
2. Latency receipts: warm-up count, timed runs, p50/p90/p99, mean and std, system metadata, token count, per-K table, all in one JSON. Reuse the shape for our latency receipts, including a K-sweep because latency grows with option count. Effort: small. Impact: credible speed claims.
3. Experiment receipts as committed JSON next to the claim (E1 to E9 style): shuffled-context control (accuracy collapses from 95 % to 22.4 % when the context is shuffled, proving the model reads the context), contamination audit, FP16 parity. The shuffled-context and question-only controls are a cheap sanity check we should run on every model in our race. Effort: small.
4. DAG staging for dependent questions (append parent answer to the state, second pass): optional later feature, not v0.1.
5. A browser console that runs with no server (WebGPU/WASM) is a different product direction; we need the race view driven by a live JSONL stream, not WebGPU.

### 3.7b Threats to our claims
- Low. It is a 151M specialised encoder, a different mechanism. It is fast for small K on CPU, but its latency grows with K, it caps at 24 options, and it scores 48 % on the TypeSafe public cases. It does not support plug-in models.
- It is the closest thing to a shipped "calibration with ECE" story among the three, and also the clearest example of why raw ECE alone misleads (fitted T made its ECE worse).

## 4. Cross-project facts for the lead engineer

### 4.1 Where the three projects sit on our SPEC
| Our SPEC element | open-alternative-jev | jev-style | Verdict |
|---|---|---|---|
| Stock open model, any HF causal LM | Yes (ChatML) | No (trained) | No (trained encoder) |
| Real label text scored | No (letters) | Yes (trained verdict slot) | Yes (label slot logits, encoder) |
| Prefill once, per-label isolation (tree mask, restart positions) | No (causal packed, interference measured) | Partly (block-shared prefix per question) | No (state copied per question) |
| TypeSafe wire format | No | Yes, `/v1/systemone` | No |
| Temperature fitted on held-out | Optional scaler | Shipped per group | Shipped, per-K |
| Windows tested | No | Claimed, not tested | No |
| JSONL event stream or race view | No | No (Playground) | No (receipts) |

### 4.2 Numbers that are not comparable (do not put in one table)
- `so1` ms per case: wall clock, H200 MIG slice, 5 decisions, includes Python, p50 of 400.
- jev-style ms: M1 Max, MLX or GGUF F16, warm p50 or single example, with prefix reuse off or a block-sharing variant.
- Verdict ms: ONNX Runtime session time only, single thread on an unnamed Apple chip, one 80-token question.
- The original demo: M4 Max, stock Qwen2.5-1.5B, first-token scoring, 68 to 89 ms small schemas and 270 ms for 28 fields.
- Accuracy: `so1` is zero-shot on typed-decisions; jev-style's typed-decisions number is in-domain; Verdict's numbers are Banking77-style in-domain. Never put 73.7 / 79.15 / 95.0 in one row without these labels.

### 4.3 What this means for the founder's question about matching the original
Nothing in these three projects reproduces the original demo model or its speed class on the same hardware. The cleanest comparable data point is `so1` with stock Qwen3-1.7B at 55 ms per 5-decision case (HF, packed, H200 MIG) and 95 ms separate, but accuracy there is 45.9 % with ECE 0.509 on typed-decisions, so a stock model under 2B is a speed target, not an accuracy target. A Qwen3.5-4B (59.3 %) or larger is where stock models start to be usable on that benchmark. This matches the plan in SPEC 11 (match the 1.5B first for speed, default to 4B for quality).

## 5. Corrections to `09-landscape-and-differentiators.md`
1. Verdict license: 09 says "Apache-2.0 per LICENSE file". Corrected: the LICENSE file is an abridged, rewritten Apache text with no copyright holder, hence `NOASSERTION`; intent is Apache-2.0 (section 3.2).
2. Verdict calibration: 09 lists "ECE 1.13 % ... T = 1.4265" as calibrated. In the README table the 1.13 % is the UNCALIBRATED ECE; after fitting T = 1.4265 it rises to 3.35 % (section 3.5).
3. Verdict latency and capacity claims: its README's "under 35 ms" and "raises CapacityError at 26 candidates" are contradicted by its own receipt and code (sections 3.3, 3.4). Not used by 09, but the earlier sentence "Verdict ... encoder" could be misread as a validated speed claim.
4. jev-style Windows: 09 says "documented", which is right, but add that CI is ubuntu and macOS only, classifiers list only macOS and Linux, and the skill text says Windows/WSL (section 2.6).
5. Full-label scoring as the headline: 09 says "real differentiator, but narrow" and "only hard-to-reproduce trained encoders (Verdict) score label text". jev-style is a decoder that also reads real option text (trained verdict slot). Narrow further: it is the only STOCK-model, plug-in method that does so (section 2.8).

## 6. Ideas ranked by impact on speed and accuracy
| Rank | Idea | From | Effort | Impact |
|---|---|---|---|---|
| 1 | Benchmark rows for three readouts in one harness: full-label sum, verdict slot (yes minus no at a marker after each option), letter. Tells us if the headline holds on stock models | jev-style, `so1` | 1 to 2 days | Accuracy; defends or revises the headline |
| 2 | Order-permutation averaging as an off-by-default option, and an order-flip metric | `so1` | 0.5 day | Accuracy and ECE on small models |
| 3 | Per-group temperature (type x option-count bucket) with raw vs fitted ECE, Brier and KL reported together | jev-style, Verdict, `so1` | 0.5 day | Calibration quality, honest reporting |
| 4 | Noise-floor control (`A_pad`) and zero cross-question interference test for our packed pass | `so1` | 0.5 day | Credibility of "no interference by construction" |
| 5 | Automation-at-error-budget and paired-bootstrap comparison tables | jev-style | 0.5 day | Better race-view summary |
| 6 | Latency receipts with warm-up, p50/p90/p99, token count, K-sweep and hardware metadata | Verdict | 0.5 day | Credible speed claims |
| 7 | Hard budget errors with token counts instead of truncation | jev-style | 2 hours | Correctness |
| 8 | Optional explicit abstention candidate, scored in the same pass | Verdict | 2 hours | Usability |
| 9 | Prefix-numerics test: block-boundary prefill vs single prefill score difference | jev-style `fastpath.py` note | 2 hours | Confirms our 2e-4 acceptance holds under chunked prefill |
| 10 | Shuffled-context and question-only control runs per model in the race | Verdict E1 | 0.5 day | Sanity check that models read the state |

## 7. Plug-in models: what these projects imply
- `so1` is the only one that supports arbitrary models, via a ChatML assumption plus a small format hook. Ours (SPEC 11) is more general. A model picker in the race view should declare tokenizer and chat-template compatibility and surface failures at load time (their letter-is-one-token check fails at run time, not at load).
- Trained-readout projects (jev-style, Verdict, von, Laya in other notes) cannot be swapped for stock models. A multi-engine race view could show them as external HTTP engines through the TypeSafe-compatible endpoint (jev-style's server speaks `/v1/systemone`), which gives the founder a side-by-side of a stock model vs a trained model without us implementing the trained readout.

## 8. Items not verified
- jev-style: all training, accuracy and latency numbers beyond what the runtime file and README state; the 2B runtime; the GGUF scorer code; Windows behaviour.
- `so1`: the numbers in the option-order and interference sections were read from the README, not recomputed from `oo_*` results; the Jev KL and Brier figures; whether it runs on Windows.
- Verdict: the weights, the base model license, the DAG implementation beyond its structure, Windows behaviour, and the 68.00 % Jev figure's origin.
