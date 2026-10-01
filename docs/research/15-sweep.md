# 15. Sweep of the other open Jev-style projects

Date checked: 2026-09-30. Stars, forks, push dates and licenses come from `gh api repos/OWNER/REPO` on that date. Numbers quoted from a project are that project's own unless marked "JevBench" (an independent board, see 15.4).
Status tags: VERIFIED (read in code, data files or API output), UNVERIFIED (README claim or secondary source, not reproduced), CONTRADICTED (a claim in our own notes or in the brief is wrong).
Scope: this file covers everything not assigned to files 01 to 09 and not on the skip list (openjev razorback16, kev, the RLCD repos, simple-jev, AnyJev, von, open-alternative-jev, jev-style, Verdict, cu-Jev, laya-go, jev-x-kit, openjev-sglang, bnsd55/openjev, jev-on-a-laptop).

## 15.0 Method and coverage

- `gh search repos` for: jev, jev-style, systemone, "system one", "parallel constrained decoding", RLCD, "typed decisions", "jev engine", "jev local", "jev benchmark", each sorted by stars. About 330 distinct repos came back. Everything with 20 or more stars that is an engine, a server, a benchmark or a UI was kept; awesome-lists, agent apps that merely call hosted Jev, and the unrelated RLCD hits (Waveshare e-paper boards, a 2023 RLHF repo) were dropped.
- Hugging Face: `api/models?search=` for jev, rlcd, systemone, laya, kev and `api/spaces?search=` for jev, rlcd, systemone, laya, decision (15.8).
- Directory: fetched `madewithjev.com/open-source-jev` (dated 2026-09-28; says 55 open alternatives, every figure self-reported, none re-run there). I did not page through all 749 directory entries (VERIFIED: the page is what I read; UNVERIFIED: the long tail).
- Deep reads (code or data files, not just README): SemIf, jevmlx, JevBench plus the Jobe and Cygnet runs it ranks, the llama.cpp `/v1/decision` pull request and the vLLM `/v1/systemone` pull request, Laya plus laya.cpp. Shorter reads: Nimble, NanoJev, decision-playground, sys1, snap, Rizzo Flow, ollaya, JevK5, Imajev, arbiter. Everything else is README and GitHub API only and says so.
- No GPU code was run, nothing was installed into the project venv, no paid API was called. Clones live in the session scratch area, not in this repo.

## 15.1 Findings that change our claims (read this first)

| # | Our claim (file 09, SPEC, brief) | Verdict | Evidence |
|---|---|---|---|
| 1 | Full-label scoring is not done by any decoder engine | **CONTRADICTED in part.** At least four others score multi-token option text: jevmlx (default `labels` scorer, trie rows), the llama.cpp `/v1/decision` pull request (trie, exact probabilities), `us/jev-local` (mean log-prob per option, one pass per option) and Jobe's `textscore` readout. None of them is calibrated and benchmarked head to head against letters except Jobe, and Jobe found letters slightly better (15.3) | 15.2.2, 15.3, 15.5 |
| 2 | Native Windows plus NVIDIA is a weak competitor field | **CONTRADICTED.** laya.cpp ships Windows x64 CUDA and Vulkan binaries, ollaya has a Windows installer with CUDA, snap ships a Windows binary, Rizzo Flow was tested on Windows 10 with an RTX 5060 Ti (Blackwell) for every number in its README, Laya documents Windows PowerShell install | 15.6 |
| 3 | No project ships a JSONL event stream for a live console | **CONTRADICTED.** jevmlx tails `heartbeat.jsonl`, `predictions.jsonl` and `run.json` into a terminal or web dashboard over server-sent events (`watch`) | 15.2.2 |
| 4 | A race-view console is novel | **Partly contradicted.** Side-by-side "decision versus streamed generation" with live stopwatches exists (decision-playground), a replay of the same idea exists (SemIf), three-panel same-clock replays exist (NanoJev), and head-to-head game arenas exist (two repos). What nobody ships: two engines or two models firing raw request and response JSON side by side, live, with a feed for a second tool | 15.7 |
| 5 | Calibration with published ECE is parity | **Confirmed, and the field is further ahead.** Per-workload temperatures with out-of-fold ECE (SemIf), per (question type, option count) temperatures with a clamp (Laya), calibration repaired on bucket level. A raw, uncalibrated stock Qwen3-4B-Instruct-2507 scores hard-tier ECE 0.452 on JevBench; fitted temperatures bring comparable stock readouts to 0.08 to 0.13 | 15.2.1, 15.4 |
| 6 | Nobody will build `/v1/systemone` into mainstream engines soon | **At risk.** Two open pull requests, one in llama.cpp (opened 2026-09-30) and one in vLLM (opened 2026-09-29), add the endpoint to the engines themselves. vLLM's is single-token labels only, uncalibrated; llama.cpp's has the trie | 15.5 |
| 7 | Matching the original demo speed (68 to 89 ms small, 270 ms for 28 fields) is the bar | **Reachable, not special.** Stock-model p50 latencies published by others: 82 ms (Qwen3-4B-Instruct-2507, JevBench), 66 ms (frozen Gemma-4-12B, RTX A6000, vLLM), about 50 ms (Qwen-class 4B Q8_0, RTX 5060 Ti, llama.cpp). Trained models with CUDA graphs reach 11 to 13 ms on an H100 | 15.4, 15.6 |

Single most useful discovery: **JevBench** (15.4) is an independent, public, rerunnable board with 91 ranked systems and, crucially, a **public 231-item slice we can run offline** and a stock "Raw Qwen3 4B Instruct 2507 direct logits" control row, which is our default model. It gives us a baseline to beat and a way to be ranked.

## 15.2 Deep reads (top five by relevance)

### 15.2.1 SemIf (formerly OpenJev), TheoLeeCJ/SemIf

| Field | Value | Tag |
|---|---|---|
| License | MIT (`LICENSE` first line); code only, weights are the stock Qwen checkpoints | VERIFIED |
| Stars / forks / last push | 4,620 / 322 / 2026-09-23 | VERIFIED (`gh api`) |
| Open source | Yes: code, fixtures, row-level results and raw timings are all committed (`results/raw/`, `benchmarks/data/`) | VERIFIED |
| Models | Stock Qwen3.5-4B (default, pinned revision), Qwen3-0.6B and MiniCPM5-2B in a browser demo, Qwen3.8-27B via an EXL3 bridge. Nothing trained | VERIFIED (`README.md`, `manifests/models.json`) |
| Pluggable models | Yes by pinned revision and backend flag; backends: PyTorch CUDA, MPS, MLX, llama.cpp CPU/GGUF, WebGPU | VERIFIED (README quick start) |
| Scoring | One forward pass; softmax over the logits of fixed **uppercase letter tokens** A to P. Prompt says "Respond with only its uppercase letter" (`src/semif_phase1/core.py:14`); each letter must be exactly one token (`direct.py:15-18`) and the boundary tokenization is asserted (`direct.py:43-44`). 2 to 16 options (`docs/METHOD.md`) | VERIFIED |
| Batching | State prefilled once, then suffix branches either serial or **padded batch with per-row attention mask and position ids** after `reorder_cache` (`shared.py:40-51`, `shared.py:100-135`). On MPS it falls back to a loop with `deepcopy` of the cache | VERIFIED |
| Speed | 21 binary criteria on one state, Qwen3.5-4B BF16, RTX 3090: 1.023 s direct vs 5.332 s for a generated JSON array of 111 tokens (5.21x). 37 states x 21 criteria, roughly 8,000 character states: 2.33 decisions/s fresh, 10.75 serial prefix reuse, 20.03 parallel suffixes. Timed region includes tokenization, transfers, readout, excludes model load | VERIFIED (README, `docs/METHOD.md`, `results/raw/shape777-direct.json`). Incomparable to small-schema latency: the states are long |
| Known numeric drift | BF16 prefix reuse changes 5 to 6 of 777 argmaxes against fresh scoring | VERIFIED (`docs/RESULTS.md`) |
| Accuracy | Authored 144 rows balanced accuracy 0.813 (4B); 0.440 / 0.686 for Qwen3-0.6B / MiniCPM5-2B; WANLI 0.637; TypeSafe public subset 0.845 modal agreement against 0.883 for published Jev on the same 102 rows. 27B EXL3: 0.958 on authored | VERIFIED as published, UNVERIFIED as reproduced |
| Calibration | Per-workload temperature, out-of-fold: authored ECE 0.068 to 0.038 (T 1.23), WANLI 0.208 to 0.069 (T 2.50), Every 0.050 to 0.047 (T 1.71) | VERIFIED (README table, `docs/CALIBRATION.md`) |
| Robustness | Option reversal flips 10 of 36, wrapper rewording 9, irrelevant context 4 | VERIFIED (`docs/RESULTS.md`) |
| JevBench placement | Rank 13 of 91, score 47.7, hard-tier accuracy 0.595, hard ECE 0.121, p50 198 ms raw on an RTX PRO 4500 Blackwell | VERIFIED (JevBench results JSON) |
| Platform | NVIDIA (3090), Apple (MLX, MPS), CPU (llama.cpp), browser. No Windows statement in README, docs or CI that I could find | VERIFIED absent |
| API | Own JSONL input and output, no `/v1/systemone`, no MCP, no server | VERIFIED |
| UI | `demo/index.html` is a **replay** of measured numbers: a `requestAnimationFrame` loop scales real times, types JSON out token by token on one side and drops 21 distributions at once on the other. It is not a live run | VERIFIED (`demo/index.html` tick function) |

Ideas worth reimplementing (fact and shape only):
1. The measured race scenario itself: same model, same state, same questions, decision readout against a streamed generated JSON, aligned at t=0. Our console can do this live instead of as a replay.
2. Publish a perturbation suite (reverse option order, reword the criterion, append irrelevant text) and report flip counts. Cheap, and it is the honest test for position bias.
3. Row-level result files plus SHA256 manifests so every table can be re-derived.
4. Out-of-fold temperature fitting and a stated rule that calibration never changes the argmax.

Threats to us: it is the cleanest stock-model baseline and already beats a naive reading of our plan on documentation. Its letter protocol is capped at 16 options; ours is not, which is a real difference for 28-field and 255-option cases.

### 15.2.2 jevmlx, bnsd55/jevmlx (not the skipped openjev repo)

| Field | Value | Tag |
|---|---|---|
| License | MIT (`LICENSE`) | VERIFIED |
| Stars / forks / last push | 68 / 8 / 2026-09-25 | VERIFIED |
| Models | Stock `mlx-community` Qwen2.5 4-bit (3B `fast`, 7B `quality` default, 1.5B `test`) or any Hub id | VERIFIED (README alias table) |
| Platform | Apple Silicon only ("Requires an Apple Silicon Mac (M1+)") | VERIFIED. Not usable on Windows |
| Scoring, two modes | `labels` (default, "measured better on the 7B") scores the real option text; `slots` scores single-token alias codes. Multi-token options are scored with a **branch-point trie**: `P(choice)` is the product of restricted-softmax factors at each branch, temperature applied once at the end (`ARCHITECTURE.md:106`, README line 46, `jevmlx/trie.py`) | VERIFIED (design); UNVERIFIED (the "better on 7B" claim, I did not find the table) |
| Alias codebook | A per-tokenizer search picks neutral alias codes whose rows tokenize cleanly, so the slot path does not depend on letters (`ARCHITECTURE.md` row for `schema.py`) | VERIFIED |
| Batching | One prefill, KV cache broadcast across rows, chunked by a measured memory budget, Metal allocation failures halve the chunk and retry | VERIFIED (README "How it works") |
| Extra signals per field | `legal_mass` (probability the model puts on the allowed continuations versus the full vocabulary, a leakage signal), `probability_margin`, `expected_index` for ordered enums, top-3 alternatives, abstention by margin threshold | VERIFIED (README, `ARCHITECTURE.md:244`) |
| Bias handling | Optional `prior_correction` subtracts a neutral-context prior from the scores; a near-tie rescore at batch size 1 inside a 5e-2 nat band to stop batch-shape flips | VERIFIED (design); UNVERIFIED (effect size) |
| Timing ledger | Non-overlapping spans: prior, prefill, plan compile, cache broadcast, suffix eval, lm-head gather, total (`ARCHITECTURE.md:202`) | VERIFIED |
| Live view | `jevmlx watch` is a read-only dashboard over a bench output directory: it watches `RUNBOOK.md`, `run.json`, and per-combo `heartbeat.jsonl` and `predictions.jsonl` by mtime, serves HTML with server-sent events (`jevmlx/watch.py:2047`, `_handle_sse`) | VERIFIED |
| API | Own `/decide` HTTP route and Python, not `/v1/systemone` | VERIFIED |
| Speed / accuracy | No headline latency in the README; benchmark harness with parity gate (batch 1 versus batched versus chunked) and a leaderboard script citing TypeSafe numbers | VERIFIED (files present), UNVERIFIED (numbers) |

Ideas worth reimplementing:
1. `legal_mass` per question. For us it is nearly free: we already take `log_softmax` over the vocabulary at each scored position. Report the probability mass the model gave to the option tokens at the first label position. It is an abstention signal and a sanity check on the prompt.
2. A trie or prefix-sharing layout for labels (see 15.5): many option names share leading tokens, so scoring each shared token once saves compute and gives a constrained distribution.
3. The parity gate as a CI test: batch 1 equals batched equals chunked within a tolerance, plus a near-tie rescore for decisions inside the tolerance. Our acceptance test already compares cached and uncached; this adds batch-shape drift.
4. Timing split by phase in the event stream (prefill, cache build, suffix eval, head, total).
5. `heartbeat.jsonl` as the live feed. Same shape as our frozen event log; confirms the design.

Threats: it is the closest match to our headline (real-text options, trie, legal mass, prior correction) on a different platform. It is Mac-only, which is the whole of our remaining gap.

### 15.2.3 JevBench (fstandhartinger/jevbench) and the stock-model rows it ranks

| Field | Value | Tag |
|---|---|---|
| What | Benchmark and public board for "Jev-class" systems; 95 systems listed, 91 ranked in the v1.4.2.2 results file | VERIFIED (`results/v1.4.2.2/jevbench-v1.4.2.2-results.json`) |
| License | MIT for the harness and the 72 original public items; third-party item licenses listed in `THIRD-PARTY.md` | VERIFIED |
| Stars / last push | 190 / 2026-09-29 | VERIFIED |
| Items | 534 frozen v1.2 decisions of which 231 are public (`datasets/public/easy.jsonl`, `original.jsonl`, `hard.jsonl`: 48 easy, 72 standard, 111 hard) plus 308 sealed items kept private | VERIFIED |
| Score | Four axes, equal weight: chance-corrected Intelligence, Calibration (ECE plus fidelity to gold distributions), Speed (`100 - 20 log10(s / 0.1 s)` on the mean of p50 and p95), Cost. Gates multiply the score down when Intelligence or Speed is low; a public-to-sealed accuracy gap above 25 points is penalized | VERIFIED (`README.md`, `docs/METHOD-v1.4.md`) |
| Latency method | Serial requests over HTTP through a `typesafe` adapter to the author's `/v1/systemone` server; self-hosted endpoints get a **x2 plus 0.15 s adjustment** "to approximate production load", an assumption, and raw p50 is also published | VERIFIED (`speed_note` in the results JSON) |
| To be ranked | The system must answer `POST /v1/systemone` (or an adapter); sealed items are run by the evaluator on its own hardware | VERIFIED (method docs, Cygnet recipe) |

Rows that matter for MirethSTM1 (all JevBench, rank of 91, score, hard-tier accuracy, hard ECE, raw p50):

| Row | Rank | Score | Hard acc | Hard ECE | p50 raw | What it is |
|---|---:|---:|---:|---:|---:|---|
| Imajev-4B | 1 | 67.4 | 0.721 public, 0.752 held-out | n/a | 40 ms | trained 4B |
| Jev 1.13.0 | 4 | 63.3 | 0.741 | n/a | 652 ms | hosted API |
| Cygnet, frozen Gemma-4-12B-it | 6 | 61.8 | 0.755 | n/a | 35 ms | **no training**, letters, vLLM, one temperature |
| SemIf (Qwen3.5-4B) | 13 | 47.7 | 0.595 | 0.121 | 198 ms | stock, letters |
| Jobe (Qwen3.5-4B, frozen) | 14 | 46.9 | 0.586 | 0.127 | 135 ms | stock, letters |
| local-jev (Qwen3.5-4B) | 15 | 46.8 | 0.605 | 0.081 | 708 ms | stock, letters, calibrated |
| jqv (Qwen3-32B zero-shot) | 19 | 44.4 | n/a | n/a | 747 ms | stock, letters, T = 3.02 |
| **Raw Qwen3 4B Instruct 2507 direct logits** | **22** | **41.0** | **0.518** | **0.452** | **82 ms** | **our default model, uncalibrated control row** |
| open-alternative-jev (Qwen3.5-4B) | 38 | 33.2 | n/a | n/a | 207 ms | stock |
| jev-local (Qwen3.5-9B, mean log-prob per option) | 39 | 32.5 | 0.591 | 0.146 | 1,045 ms | stock, one pass per option |
| Laya (421M encoder, CPU) | 43 | 30.3 | n/a | n/a | 787 ms | trained encoder |
| Raw Qwen3 1.7B direct logits | 65 | 18.1 | n/a | n/a | 67 ms | stock |
| Raw Qwen3 0.6B direct logits | 83 | 7.1 | n/a | n/a | 68 ms | stock |

(Hard ECE of "n/a" means I did not extract it; accuracy values are from the `tiers` fields. Hardware differs per row: A6000, RTX PRO 4500 or 6000 Blackwell, H100, one CPU row. VERIFIED as published in the results JSON, UNVERIFIED as reproduced.)

What this tells us:
1. **Our default model is already a row**, and it is mediocre until calibrated: Calibration axis 29.1 against 66 to 73 for stock 4B readouts that fit a temperature. Fitting T is worth more than anything else we do to the scoring.
2. **Model size beats readout.** Frozen 12B (Cygnet) reaches 85 of 111 on hard with letters and one temperature; stock 4B reaches 0.52 to 0.60. A Qwen3-1.7B or 0.6B control scores 18.1 and 7.1. If we want accuracy near Jev we need a 9B to 12B class model on the 5090 (32 GB), not a smarter readout.
3. **Training buys more than any readout trick**: the same Qwen3.5-4B base goes from about 0.59 hard (stock) to 0.72 to 0.75 (trained, rows 1 to 3). We are not training; our README framing ("approximation, not an equivalent") is correct and necessary.
4. The Speed axis rewards a p50 below 100 ms; the x2 plus 0.15 s adjustment means our raw 40 ms would be scored as 0.23 s. Publish raw numbers and say which adjustment a board applied.

### 15.2.4 llama.cpp `/v1/decision` pull request, vLLM `/v1/systemone` pull request, decision-playground

| Item | Facts | Tag |
|---|---|---|
| llama.cpp PR 29752 "server : add /v1/decision for parallel constrained decisions" | Open, created 2026-09-30, 1,358 lines added. Enabled by `--decision-seqs N`. Instructions and a generated field catalogue are prefilled once as a cached prefix; each context (1 to 256 per request) forks into one KV branch per field with `llama_memory_seq_cp`; each field's allowed values are tokenized into a **trie** and every divergence node is scored in one batched pass, giving **exact probabilities over the values** ("tree" mode, up to `tree_max` = 128 values per field), else a greedy walk. JSON is assembled in code. Enum and integer fields take 1 to 255 values | VERIFIED (PR body, `tools/parallel-decision/README.md` in the diff) |
| Its speed | RTX 3060 12 GB: Gemma 4 12B, 3 fields, warm cache about 100 ms per decision (50 ms prefill plus 50 ms scoring); Qwen3.5 9B, 12 fields 161 to 90 ms with the hybrid-model padding fix; Qwen3.5 4B scoring 1.5 to 2.2x faster than one branch at a time | VERIFIED as stated in the PR, UNVERIFIED as reproduced |
| Its limits | Hybrid models (Qwen3.5, Nemotron-H) batch sequences only at equal token counts, so branches are right-padded; sliding-window models allocate per sequence. No calibration. PR description says it is heavily AI-assisted | VERIFIED |
| vLLM PR 59299 "Structured decisions endpoint (`/v1/systemone`)" | Open, created 2026-09-29, 1,482 lines added. Autoregressive models only, `choice` type only, up to 64 questions and at most 128 options; labels are single-token codes A to ZZ chosen to fit the template, **shuffled deterministically per question** with an optional `seed` so averaging over seeds cancels letter bias; response carries `label_mass` and `argmax_is_label` diagnostics; "Probabilities are currently uncalibrated". Tested on a DGX Spark (28 tests pass) | VERIFIED (PR body) |
| A related merged vLLM change | PR 57250 "structured generation mode for DiffusionGemma (Jev-like)" merged; the example server needs single-token choices | VERIFIED |
| decision-playground (thecodacus) | No license file (GitHub reports none). Browser-only React app that points at a llama-server; runs one `/v1/decision` pass and a streamed grammar-constrained `/v1/chat/completions` on the same model **side by side with live `requestAnimationFrame` stopwatches that later swap to server-reported time**, shows per-field probabilities and the exact request payloads; also a small game driven by the endpoint | VERIFIED (`README.md`, `src/components/Stopwatch.tsx`, `src/pages/Playground.tsx`) |

Ideas worth reimplementing:
1. **Trie scoring.** For labels with shared prefixes, score each shared token once and read the conditional at each branch point. Two readings are possible and we should keep them distinct: (a) our current sequence log-likelihood softmaxed over labels, (b) constrained decoding probability, a product of restricted softmax factors at each branch. They differ. (b) ignores mass the model puts outside the allowed tokens at each node, so it never penalizes a label for being "unlikely to be said at all"; (a) does. The PR and jevmlx both use (b). Offer (b) as a mode and compare. Effort M, needs a tree attention mask in our packed pass.
2. Per-question deterministic option shuffle with seed averaging (vLLM PR) as an optional bias control. Jobe measured order averaging for Qwen3.5-4B and found it did not pay (15.3), so default off.
3. A paired live stopwatch that starts at the same instant for both engines and swaps to server-reported time at the end. Small and exactly what a race view needs (15.7).

Threats: if the llama.cpp PR merges, any GGUF model on any OS gets a keyless typed-decision endpoint with trie probabilities, which erodes both the full-label and the Windows differentiators. Neither PR calibrates, which keeps calibration ours.

### 15.2.5 Laya (NandhaKishorM/laya) and laya.cpp (lkarlslund/laya.cpp)

| Field | Laya | laya.cpp |
|---|---|---|
| What | Non-autoregressive typed-decision **encoders** (ModernBERT-large 421M English, mmBERT-base 322M multilingual, plus a typed-decisions fine-tune) trained with RLCD, a `Router` that picks a checkpoint per request, HTTP server with `POST /v1/systemone` and `/v1/systemone/batch` (`laya/serve.py:695`, `:784`), MCP, ONNX, TileLang GPU fast path | Native C++ (ggml, CUDA, Vulkan, Core ML) inference for the same three Laya checkpoints, JEV-compatible `POST /v1/systemone` with automatic batching |
| License | Apache-2.0 (`LICENSE`) | MIT (`LICENSE`); models keep their own licenses |
| Stars / forks / push | 29,298 / 2,547 / 2026-09-29 | 111 / 11 / 2026-09-27 |
| Open source | Code and weights public; the training mix is described, the fine-tuned typed-decisions number has "no committed result file behind it yet" (`BENCHMARKS.md:49`) | Code and measurements public |
| Scoring | Encoder reads options packed into a head budget (`head_max_len`, 192 tokens English, 256 multilingual); one forward pass; per-checkpoint temperatures | Same model, ggml graph |
| Option limit | Options share the head budget, not Jev's 255; Banking77 (77 labels) drops to 0.425 against 0.870 for Jev because each option gets 3 to 4 tokens (`README.md:1180`); the HTTP server caps at 100 choice options | inherits |
| Speed | 32.8 ms p50 for one question on a T4, 7.2 ms per question batched; RTX 4070 Ti SUPER fast path 4.0 ms for a short single question (stock 17.2 ms); RTX 5060 Ti about 10 ms to about 1 ms per decision batched (`README.md:674`, `benchmarks/results/fast_english_rtx4070.json`). Directory tweet: 86.5 decisions/s, p50 about 9 ms | RTX PRO 6000 Blackwell (450 W cap): english FP32 batch 1 is 341.7 questions/s CUDA versus 147.9 for matching-precision Python (2.31x); multilingual BF16 batch 4 1,164 q/s (`docs/performance.md`) |
| Accuracy | typed-decisions (2,000) 0.766 for the fine-tuned checkpoint against 0.727 for published Jev; AG News 0.950; the English checkpoint collapses outside English (51-language macro accuracy 0.227); self-reported, Jev numbers "third-party published, never measured here" | answers checked against Python: exact categories and numeric error at most 0.0001 |
| Calibration | Fitted temperature per (question type, option count): mean ECE 0.466 to 0.081 (`laya`) and 0.314 to 0.106 (multilingual), temperatures clamped to [0.5, 5.0] (`README.md:1240-1250`) | inherits |
| JevBench | Rank 43 of 91, score 30.3, p50 787 ms **on a 4-thread CPU** | not ranked |
| Platform | Python on Windows documented (PowerShell commands, `README.md:147-180`); CUDA, MPS, XPU, CPU | **Windows x64 CUDA 12/13 and Vulkan prebuilt executables** built in CI on `windows-2022` (`.github/workflows/binary-build.yml:21,36`) plus Linux and macOS Core ML |
| API | `/v1/systemone`, batch, MCP, LangChain, LlamaIndex, CrewAI | `/v1/systemone` |

Also in this family, all shallow (README and API only): laya-mlx (6,659 stars, Apache-2.0, "13.4 ms median end to end, 7.4 ms multilingual" on an M3 Max), laya-coreml (1,530, about 5 ms), receptron/laya (664, MIT, Node and TypeScript), sys1 (48, Apache-2.0, Rust with candle, Flash Attention on Ampere to Hopper, serves Laya on `/v1/systemone`), laya-vision (69), laya-bangla (40), the Laya GGUF and ONNX ports on Hugging Face.

Ideas worth reimplementing:
1. **Temperatures per (question type, option count) with a clamp.** The fact: ECE depends on option count, so one global T is wrong. Cheap to add to our calibration step and to publish.
2. A **reference-parity rule for any fast path**: laya.cpp accepts a path only if exact categories match and numeric error is under 1e-4 at matching precision. Matches our 2e-4 acceptance test; keep it for every optimization we add.
3. Report **decisions per second at batch sizes 1, 2, 4, 8** on a fixed 250-question corpus, not just single-question latency.
4. `/v1/systemone/batch`: several independent requests in one call, scored in shared passes.

Threats: its encoders set the speed floor (4 to 13 ms) and run natively on Windows with CUDA. We cannot beat that with a 1.5B to 4B decoder, so we should never claim "fastest" and should compare against Laya on accuracy per label count, where its option-budget limit gives us a real edge.

## 15.3 Evidence on our headline: full-label versus letters

The most relevant measurement in the sweep, from Jobe (MantisShrimpdev/jobe, MIT, 2 stars, notable because JevBench ranks it the top native-logit system and it publishes every failed experiment):

- `bench/RESULTS.md:835` "Letter readout vs option-text readout": same Qwen3.5-4B backbone, same 231 JevBench public tasks, same option mapping, both readouts in one process, **length normalization on** for the text readout.
- Accuracy: easy 0.979 vs 0.979, standard 0.972 (letters) vs 0.931 (text), hard 0.613 vs 0.622, all 0.801 vs 0.792. The two agree on 204 of 231; text fixes 11 and breaks 13.
- Text is less decisive (mean top probability 0.768 against 0.838) and slower (710 ms against 561 ms mean).
- The standard-tier loss is all `choice` items in routing and intent where option names have unequal token length: "the length normalisation is doing real work".
- Jobe keeps letters as default and text as fallback above sixteen options. It also measured: option-order averaging does not pay for a 4B model on any tier, a top-two runoff measures position rather than judgment (`bench/RESULTS.md:618`), and two LoRA training runs that improved every training metric lowered every JevBench metric (`bench/RESULTS.md` section "Training, second full run").
- A second independent datapoint: `us/jev-local` (Qwen3.5-9B, mean log-prob per option, one forward pass per option) ranks 39 of 91 with hard accuracy 0.591, no better than letter readouts of 4B models (0.586 to 0.605). VERIFIED (JevBench row, README), different model sizes so only indicative.

Consequences for MirethSTM1 (VERIFIED facts, our decision):
1. Our SPEC section 3.4 step 5 uses **raw summed log-prob with no length normalization**. Both Jobe's result and jevmlx's trie design suggest that is the weakest variant. We should implement mean log-prob and trie (constrained) modes and let a benchmark pick the default. Do not claim full-label is more accurate until our own table says so.
2. A claim we can defend: full-label scoring has no 16-option ceiling (SemIf, Jobe letters), no 128 cap (vLLM PR), no head-token budget (Laya), and needs no letter indirection. Position bias is a letter effect (SemIf flips 10 of 36 under reversal), so we should publish our flip count next to theirs.
3. A claim we cannot make: "no other engine scores option text". Say "scores the real option text, with length-normalized and constrained modes, benchmarked against letters in this repository".

## 15.4 Training baselines and benchmark suites (shorter reads)

- **Nimble** (bespokelabsai/nimble, 1,989 stars, last push 2026-09-24): recipe plus Bespoke-Nimble-9B, Qwen3.5-9B with LoRA on answer tokens, single-token answer codes, up to 255 choices, 8,192-token context. **No LICENSE file in the repository** and GitHub reports no license; the Hugging Face model card tag says Apache-2.0. Reports 90.1% reference match against 66.4% for the stock base on 324 held-out examples and 93.2% for Jev (VERIFIED from README image alt text and text, UNVERIFIED as reproduced). Its `docs/PUBLIC_BENCHMARKS.md` is the best external-label suite found: 13 human-labeled subsets (BoolQ, SQuAD 2.0, PAWS, MultiNLI, Civil Comments, Aegis2, HelpSteer2, SummEval x2, PubMedQA, VitaminC, MASSIVE en and de), 3,880 records, per-subset accuracy with confidence intervals, McNemar p against Jev, ECE, Brier, JSD. Dataset licenses are listed per subset and are mostly CC BY and CC BY-SA, so we would fetch them at benchmark time rather than redistribute. Latency table (their H100): Gemma 3 270M 21.8 ms, Qwen3.5-0.8B 48.6, Qwen3.5-4B 58.0, Qwen3.5-9B 58.1, Qwen3.8-27B 145.3, Jev API 246.7 ms median per example. JevBench ranks Nimble 9B 62nd (score 18.7), so training on a small curated set did not transfer there.
- **NanoJev** (TianyuCodings/NanoJev, 2,452 stars, MIT, last push 2026-09-21): Qwen3-0.6B backbone plus decision heads, a **set-attention head over 2 to 255 candidates**, "complete-question cross entropy" training, 18,760 questions in four game tasks, side-by-side three-panel replays of Jev, NanoJev and untuned Qwen on a shared game clock with action probabilities. Claims 128 of 128 on ViZDoom Basic against 56 of 128 for Jev, 27 of 128 on Predict Position (UNVERIFIED, README only). Serving needs CUDA and a disabled native Triton flag, so Windows is doubtful (UNVERIFIED). Task set is games, not text decisions.
- **decider** (Mapika/decider, 994, Apache-2.0): trained one-pass decision readout on Qwen3.5 (2B, 4B, 35B-A3B); softmax over option letters with per-model temperature from `decider_config.json`; JevBench rank 3 for decider-4b v2 (64.1, raw p50 17 ms), the 2B ranks 41. Handles Windows ARM64 CPU in tests (bf16 about 13x slower on that CPU, so it forces float32) (README lines 71, 190).
- **JevK5** (allebee/jevk5, 126, Apache-2.0): Qwen3.5-4B plus distilled LoRA, **SemIf's letter readout**, **one CUDA graph per padded input length: 13 ms against about 70 ms eager on an H100, same answers** (README table). JevBench rank 5, score 62.0. UNVERIFIED as reproduced.
- **Imajev** (mohit67890/imajev, 170, Apache-2.0): trained 4B, rank 1; its README reports `--fast --merge-lora` taking a text decision from 59 ms to 11 ms on an H100 (`scripts/bench_fast_path.py`).
- **Rizzo Flow** (Rizzo-AI-Academy/rizzo-flow, 772, Apache-2.0): llama.cpp runtime, own LoRA of Spark-X2.5 4B, typed-decisions accuracy 0.574 base to 0.648 fine-tuned, ECE 0.349 to 0.112, "about 50 ms per decision at Q8_0 on an RTX 5060 Ti", every number measured on **Windows 10 plus RTX 5060 Ti**. Probabilities stated as uncalibrated by default.
- **ollaya** (ollaya-dev/ollaya, 1,036, Apache-2.0, Rust): "run open decision models locally, the way Ollama runs LLMs": `pull`, `run`, `list`, `serve`, a desktop app for macOS, Windows and Linux, `/v1/systemone`, installer adds the CUDA runtime on Windows. Model catalog at ollaya.dev (UNVERIFIED). This is the closest thing to the founder's "plug in different models" idea as a product.
- **snap** (emnlmn/snap, 22, MIT, Rust over llama.cpp): letters A to Z, one shared batched `llama_decode`, Windows x86_64 release archive, 50 ms single question and 32 ms per question at four on MiniCPM5-2B Q4_K_M on an M1 Max; 263 ms for 8 questions vs 504 ms for Ollama answering one letter each (own `vs_ollama.py`). It caches the template head and per-question heads across requests.
- **jev-rs** (yijunyu/jev-rs, 16, Apache-2.0, Rust): scoring backend can be llama-server or any OpenAI-compatible endpoint with `logprobs` (`max_tokens: 1`, `top_logprobs: 20`), so hosted engines work; `eval` and `calibrate` commands; MCP. macOS and Linux binaries only.
- **Other benchmarks**: AbdelStark/jev-benchmarks (21 stars, Apache-2.0, "probability-aware evaluation: calibration, selective risk, latency"), Hugging Face spaces `multimodalart/jev-decision-index` (347 likes, static, JSON data files for several index versions) and `mayafree/typed-decision-leaderboard` (43 likes), the shared dataset `LocalLLaMA/typed-decisions` (2,000 decisions, used by Laya, Rizzo Flow, open-alternative-jev, AnyJev).

## 15.5 Complete list of remaining open-source engines (20 or more stars, or a notable claim)

Columns: stars / license / last push / kind / scoring / platform notes / API. "Shallow" means README and `gh api` only; every cell is then UNVERIFIED beyond the metadata, which is VERIFIED.

| Project | Stars | License | Push | Kind and model | Scoring | Platform | API | Depth |
|---|---:|---|---|---|---|---|---|---|
| NandhaKishorM/laya | 29,298 | Apache-2.0 | 09-29 | encoder 421M / 322M, RLCD | head-packed options | Win/Linux/Mac, CUDA | systemone, batch, MCP | deep |
| mizorewww/laya-mlx | 6,659 | Apache-2.0 | 09-22 | Laya on MLX | encoder | Apple only | own, systemone | shallow |
| TheoLeeCJ/SemIf-OpenJev | 4,620 | MIT | 09-23 | stock Qwen3.5-4B etc. | letters A..P | CUDA, MPS, MLX, CPU, WebGPU | none | deep |
| TianyuCodings/NanoJev | 2,452 | MIT | 09-21 | Qwen3-0.6B plus heads, trained | set-attention head, 2..255 candidates | CUDA | own server | README |
| bespokelabsai/nimble | 1,989 | none in repo (weights Apache-2.0) | 09-24 | Qwen3.5-9B LoRA | single-token codes, 255 | CUDA, Mac | hosted API, Python | README + docs |
| mizorewww/laya-coreml | 1,530 | Apache-2.0 | 09-22 | Laya on Core ML | encoder | Apple only | own | shallow |
| vinnylarouge/jevlike | 1,335 | MIT | 09-16 | trained game agents (Doom, chess) | reverse-engineered architecture | unknown | none | shallow |
| feder-cr/jev ("jevos") | 1,139 | MIT | 09-30 | C++ yes/no on a laptop, Windows zip listed | unknown | Win/Linux/Mac | `jev serve` | shallow, repo created 2024 (see 15.9) |
| ollaya-dev/ollaya | 1,036 | Apache-2.0 | 09-30 | Rust runner and registry, many models | per model | Win (CUDA), Linux, Mac | systemone | README |
| Mapika/decider | 994 | Apache-2.0 | 09-30 | Qwen3.5 2B/4B/35B trained | letters plus temperature | CUDA, CPU incl. Win ARM64 | systemone | README |
| Rizzo-AI-Academy/rizzo-flow | 772 | Apache-2.0 | 09-25 | llama.cpp, Spark-X2.5 4B LoRA | typed readout | **Win + RTX 5060 Ti tested**, Vulkan, Mac | Jev-compatible | README |
| receptron/laya | 664 | MIT | 09-21 | Laya from Node/TS | encoder | any Node | library | shallow |
| Liuziyu77/Valen | 579 | Apache-2.0 | 09-30 | multimodal Jev-like, trainable | trained | CUDA | unknown | shallow |
| Yinsongxu/LLM2Jev | 379 | Apache-2.0 | 09-26 | local LLM to decision model | unknown | unknown | unknown | shallow |
| malevrigns/agent-jev | 331 | Apache-2.0 | 09-23 | AgentJev-0.6B | trained head; JevBench-type 57.8% (606 of 1,048) per directory | CUDA | unknown | shallow |
| mohit67890/imajev | 170 | Apache-2.0 | 09-29 | trained 2B/4B/9B incl. images | trained | CUDA, Mac | systemone | README |
| fstandhartinger/jevbench | 190 | MIT | 09-29 | benchmark | n/a | any | adapters | deep |
| HarnessRouter/SystemOneHarness | 188 | Apache-2.0 | 09-21 | agent harness around System One models with step traces | n/a | any | systemone client | shallow |
| kshetrajna12/reflex | 159 | MIT | 09-27 | frozen Qwen3.5 2B/4B/27B | letters, **two option orders averaged**, T = 1 | CUDA | systemone | README |
| allebee/jevk5 | 126 | Apache-2.0 | 09-28 | Qwen3.5-4B/9B LoRA | letters (SemIf readout), CUDA graphs | CUDA | systemone | README |
| lkarlslund/laya.cpp | 111 | MIT | 09-27 | C++ Laya | encoder | **Windows CUDA, Vulkan** | systemone | deep |
| mmastrac/djev | 110 | Apache-2.0 | 09-24 | DiffusionGemma structured reads on vLLM | single-token slots | Linux CUDA | systemone | README |
| zwliJay/jev-forge | 100 | NOASSERTION | 09-23 | train and infer stack | trained | CUDA | unknown | shallow |
| bnsd55/jevmlx | 68 | MIT | 09-25 | stock Qwen2.5 MLX | **trie labels** or slots | Apple only | own `/decide` | deep |
| r-ms/mini-jev | 57 | MIT | 09-18 | frozen Qwen3-4B | option-letter readout, 13,600 of 13,600 questions put a letter on top | any | library | README |
| deepanwadhwa/OpenDecision | 57 | Apache-2.0 | 09-25 | ModernBERT-large zero-shot | encoder | CPU/CUDA | systemone | README; JevBench rank 58 |
| alvarobartt/sys1 | 48 | Apache-2.0 | 09-30 | Rust server for Laya | encoder | CPU, Metal, CUDA | systemone, decide | README |
| intikhab49/open-jev-typed-decision-engine | 44 | Apache-2.0 | 09-21 | 150M encoder | encoder | any | unknown | shallow |
| JoshuaSP/open-jev | 41 | MIT | 09-16 | DiffusionGemma JSON canvas | canvas | CUDA | unknown | JevBench rank 91 |
| shamspias/laya-bangla | 40 | Apache-2.0 | 09-28 | Laya adaptation | encoder | any | unknown | shallow |
| nico-martin/open-jev | 38 | MIT | 09-21 | browser TypeScript library | in-browser | browser | library | shallow |
| caiovicentino/eikos | 39 | MIT | 09-23 | finance 4B and 27B | trained | CUDA | unknown | shallow |
| Shanghua-Gao/RSI-Jev | 38 | MIT | 09-30 | trained by self-improvement loop | trained | CUDA | unknown | shallow |
| 0xBakeer/arbiter | 33 | MIT | 09-24 | serves Laya or your own models, NVIDIA and Apple | encoder | GB10, M2 Max | systemone plus playground page | README |
| hawkymisc/typed-decision-bert | 32 | MIT | 09-22 | BERT-style PoC | encoder | any | systemone | shallow |
| Argos1111/jev_local | 37 | none | 09-23 | local LLM replication | unknown | unknown | unknown | shallow |
| KaLM-Embedding/KaLM-Jev | 37 | none | 09-21 | Nano, Small, Large | unknown | unknown | unknown | shallow |
| iapp-technology/openthai-systemone | 65 | Apache-2.0 | 09-21 | 0.8B Thai plus English, 256-way slot head | trained head | many quantizations | unknown | shallow |
| rkinas/basal | 64 | Apache-2.0 | 09-30 | Polish decision models | letter logprobs | CUDA, Mac, Ollama | unknown | shallow |
| SAGAR-TAMANG/sarvam-jev | 58 | none | 09-18 | Indic LLMs | generation-free | unknown | unknown | shallow |
| emnlmn/snap | 22 | MIT | 09-28 | Rust over llama.cpp, MiniCPM5-2B fine-tune | letters A..Z | **Windows x86_64 release**, Mac, Linux | systemone | README |
| yijunyu/jev-rs | 16 | Apache-2.0 | 09-28 | Rust harness over any decoder | logprob | Mac, Linux | systemone, MCP | README |
| thecodacus/decision-playground | 24 | none | 09-19 | browser UI for llama.cpp `/v1/decision` | n/a | browser | `/v1/decision` | deep |
| virajbhartiya/laya-vs-jev | 109 | Apache-2.0 | 09-21 | T-Rex and Snake arena, Laya MLX vs hosted Jev | n/a | Apple Silicon | n/a | README |
| PromptEngineer48/laya-vs-jev-arena | 29 | MIT | 09-22 | browser arena: Snake race, fight, Tetris, runner; Python proxy | n/a | any (Docker) | n/a | README |
| sutro-sh/jev-align | 301 | Apache-2.0 | 09-20 | alignment-failure detection with RLCD, not an engine | n/a | n/a | n/a | not read |
| Heman10x-NGU/openJev-verdict-2.0 | 293 | NOASSERTION | 09-20 | covered by the Verdict file | | | | skipped |

Plus, below 20 stars but notable: MantisShrimpdev/jobe (2, MIT, stock Qwen3.5-4B, JevBench rank 14, text versus letter measurement), Octalab-Inc/jqv (0, Apache-2.0, stock Qwen3-32B, shared-prefix branches behind a block attention mask plus a Hydragen-style mask-free variant, 53 to 73x faster than one forward per question at 8k tokens by 100 questions on an Apple M5 Max per its README, JevBench rank 19), amithgc/local-jev (13, MIT, default Qwen3.5-4B, Windows only through WSL2), us/jev-local (7, no license file), AbdelStark/jev-benchmarks (21).

## 15.6 Speed numbers by hardware (nothing here is directly comparable)

| Source | Hardware | Model | Workload | Number | Tag |
|---|---|---|---|---|---|
| Original demo (from the brief) | M4 Max | stock Qwen2.5-1.5B-Instruct | small schema / 28 fields | 68 to 89 ms / 270 ms | from the brief |
| JevBench control | RTX A6000 | Qwen3-4B-Instruct-2507 direct logits | standard tier, serial, in process | 82 ms p50 | VERIFIED (results JSON) |
| Cygnet | RTX A6000 / L40S | frozen Gemma-4-12B, vLLM 0.30 | standard tier, serial | 66 / 50 ms p50 | VERIFIED (its README) |
| Nimble table | H100 | Qwen3.5-0.8B / 4B / 9B / 27B base | one question per example | 48.6 / 58.0 / 58.1 / 145.3 ms median | VERIFIED (README) |
| Nimble table | M5 Pro 64 GB | Bespoke-Nimble-9B | 324 examples | 444 ms median | VERIFIED |
| SemIf | RTX 3090 | Qwen3.5-4B BF16 | 21 criteria, one long state | 1.023 s | VERIFIED |
| llama.cpp PR | RTX 3060 12 GB | Gemma 4 12B / Qwen3.5 9B | 3 fields / 12 fields, warm cache | about 100 ms / 90 ms | stated in PR |
| Rizzo Flow | RTX 5060 Ti | 4B Q8_0 | typed decision | about 50 ms, 195 to 201 ms per request in its accuracy table | README |
| snap | M1 Max | MiniCPM5-2B Q4_K_M | 1 / 8 questions, fresh state | 50 ms / 252 ms | README |
| JevK5 | H100 | 4B | short decision | 13.2 ms p50 (CUDA graphs; about 70 ms eager) | README |
| Imajev | H100 | 4B | text decision, `--fast --merge-lora` | 11 ms against 59 ms | README |
| Laya | T4 / RTX 4070 Ti SUPER / RTX PRO 6000 | 421M encoder | one question | 32.8 ms / 4.0 ms / about 2.9 ms (341.7 per second) | README, JSON, `docs/performance.md` |
| Jev hosted | API | n/a | one request | 236 to 276 ms (Nimble and Laya) / 652 ms (JevBench, adjusted differently) | VERIFIED as published |

Things that make numbers incomparable: prompt length (SemIf's 8,000 character states take about 1 s for 21 questions, tiny prompts take tens of ms), whether HTTP and tokenization are inside the timer, serial versus batched, warm versus cold cache, quantization (Q4_K_M versus BF16), thermal cap (laya.cpp's 450 W cap), and JevBench's x2 plus 0.15 s adjustment on self-hosted rows.

## 15.7 Race view, console and event stream: what exists

| Project | What it shows | Live or replay | Raw JSON visible | Tag |
|---|---|---|---|---|
| decision-playground | decision pass and streamed chat side by side, stopwatches, per-field probabilities, request payloads, plus a game | live | yes (payloads) | VERIFIED |
| SemIf `demo/index.html` | 21 distributions appear together while generated JSON types out token by token | replay of measured times | the JSON text | VERIFIED |
| NanoJev | Jev, NanoJev and untuned Qwen on one game clock with action probabilities | recorded replays (an interactive viewer "requires access") | no | README |
| laya-vs-jev, laya-vs-jev-arena | local Laya against hosted Jev in games, live latency charts, JSON export of per-run stats | live | no | README |
| jevmlx `watch` | tails `heartbeat.jsonl`, `predictions.jsonl`, `run.json` with server-sent events to a terminal or web page | live | row data | VERIFIED |
| arbiter `playground/index.html` | one-file page showing answers, probabilities, checkpoint and latency | live | partly | README |
| JevK5 and others | paced replays of recorded GPU output as GIF and video | replay | no | README |

Not found anywhere: two engines or two models each streaming their own request and response JSON into a left and a right log, with the same inputs and a shared clock, and an outbound feed for a second tool. Absence is UNVERIFIED across the 55 plus projects.

Ideas for our console (all reimplementation of shape only):
1. Start both sides on one `performance.now()` origin; swap each stopwatch to its engine-reported time on completion (decision-playground shape).
2. Put the per-phase timing ledger (prefill, cache build, suffix pass, head, total; jevmlx shape) on each event so the race can show where the milliseconds go.
3. Default left versus right pairing "letters readout" against "full label readout" on the same model and state. That is our own head-to-head, it is cheap, and it answers the question 15.3 raises.
4. A paced replay mode from a recorded event file for demos and screenshots, clearly labeled as replay (the SemIf and JevK5 pages do this and label it).

## 15.8 Hugging Face sweep

Models (search results, counts of rows seen, not totals): about 60 for "jev", 25 for "rlcd", 40 for "systemone", 60 for "laya", 9 for "kev". Largest by downloads, all Apache-2.0 unless noted:
- jev: chaoliangUNSW Jev-Style-Qwen3.5-2B-Decision GGUF and MLX (8,293 downloads), alibiserikbay/JevK5 family (8,047), com-kotobalabs/open-jev-deberta-v3-large (2,770, 69 likes), TokenRhythm/NeoHorse-Jev-4B (2,202), akhilaaa3/Jev-Omni (330 likes), mradermacher jevify-gemma4 GGUFs (Gemma license).
- rlcd: heman10x/rlcd-modernbert-151m (23,257 downloads), LFM2.5-350M RLCD variants (license "other"), anthonym21/qwen3-0.6b-rlcd-decision, the three small Qwen-2.5-1B-RLCD forks (covered elsewhere).
- systemone: iapp/OpenThai-SystemOne (9,363) with ten quantizations, pngwn/system-one-qwen3.5-4b-scorer (**CC BY-NC 4.0**, so not usable for us), dwidlee/systemone-lite-0.5b.
- laya: GGUF, ONNX, Core ML, LiteRT, MXFP ports from many authors.
- kev: jaredpalmer/kev-0.6b/4b/9b (covered by file 03).
- Several models carry non-Apache licenses (Gemma, "other", CC BY-NC); license must be read per model before any comparison is distributed.

Spaces: multimodalart/jev-decision-index (347 likes), pngwn/open-jev (40), FINAL-Bench/Tetris-JEV-LAYA-ZTC (40, a game), akhilaaa3/jev-omni (22), benchmarkheaven/JevBench (8, the board front end), convaiinnovations/laya-demo (257), mayafree/typed-decision-leaderboard (43). None looked like a race view; I did not open them all (UNVERIFIED).

## 15.9 Anomalies and unverified items

- Star counts are very high for a two-week-old ecosystem (Laya 29,298; browser-use/jev-ultrafast 21,579, which is an agent, not an engine). I report them as returned by the API and do not treat them as quality evidence.
- feder-cr/jev: GitHub says created 2024-08-15, written in C++, description "an open-source alternative to Jev for yes/no decisions that runs on your laptop", Windows zip listed. The creation date predates the Jev launch; this may be a renamed repo. I read only the README. UNVERIFIED.
- Laya's headline "typed-decisions 0.766" is for a fine-tuned checkpoint with "no committed result file behind it yet" (`BENCHMARKS.md:49`); the committed English T4 suites give lower numbers (AG News 0.947 against 0.950 in the headline table, Emotion 0.573 against 0.595). The Jev comparison numbers are third-party published, not re-run. Treat Laya's accuracy rows as self-reported.
- `us/jev-local`: README says the default scorer is "deterministic and carries no intelligence" and only `JEVLOCAL_SCORER=hf` runs a model; its JevBench row used the HF scorer.
- Nimble has no license file in the repository; do not copy anything from it, and we do not.
- jqv, vinnylarouge/jevlike, SystemOneHarness, laya-vision, Valen, agent-jev, imajev, NanoJev were read at README level only, and the 20-star floor means I did not look at dozens of small repos in the directory.
- Directory page states every figure is the project's own; I did not re-run any benchmark, and I did not verify the JevBench numbers beyond reading its published results file.

## 15.10 Ideas to reimplement, ranked by impact on speed and accuracy

Founder rule applied: facts and shapes only, no code or names copied.

| Rank | Idea (source) | Impact | Effort | Notes |
|---:|---|---|---|---|
| 1 | Fit and publish temperature per (question type, option count) with a clamp (Laya, SemIf) | Large on calibration: raw hard ECE 0.452 (JevBench control) to 0.08 to 0.13 for fitted stock readouts | S | Calibration axis is a quarter of the JevBench score |
| 2 | Make length-normalized mean log-prob and constrained (trie) probability selectable beside raw sum; benchmark all three against a letter baseline on the same model (Jobe, jevmlx, llama.cpp PR) | Decides our headline honestly; Jobe saw -0.9 point overall and -4.2 on routing choices for text versus letters | M | Our SPEC 3.4 step 5 is raw sum, the variant with the most length bias |
| 3 | Expose `/v1/systemone` early and run JevBench's 231 public items through its `typesafe` adapter (JevBench, Cygnet recipe) | Gives a rank-comparable number against 90 systems and a baseline (stock Qwen3-4B score 41.0) | S to M | SPEC marks the HTTP server "cut second"; the board needs it |
| 4 | Prefix-sharing tree for labels in the packed pass, scoring shared tokens once (llama.cpp PR, jevmlx) | Speed for many options and long names; also gives mode (b) from rank 2 | M | Needs a tree attention mask; keep the current flat pack as reference for the 2e-4 test |
| 5 | `legal_mass` per question and `label_mass` diagnostics (jevmlx, vLLM PR) | Accuracy: abstention and prompt sanity at near-zero cost | S | Use the log-softmax already computed |
| 6 | CUDA graphs per padded length, and a merged fast path (JevK5, Imajev: 13 vs 70 ms, 11 vs 59 ms on an H100) | Largest speed lever for 1.5B to 4B models where launch overhead dominates | M to L | UNVERIFIED on Windows with torch 2.11 cu128; we already assume no Triton and no torch.compile, graphs do not need either |
| 7 | Cross-request cache of the fixed system prompt and instruction head (snap, Jobe: 11x per question after priming a 2,000-token document) | Speed on repeated schemas | S to M | Must key the cache on the exact prefix; Jobe has a guard "will not score a question against the wrong document" |
| 8 | Parity gate in CI: batch 1 equals batched equals chunked within tolerance, near-tie rescore (jevmlx, laya.cpp 1e-4 rule) | Prevents silent accuracy drift from speed work | S | Pairs with rank 6 |
| 9 | Perturbation suite: option reversal, rewording, irrelevant context; report flip counts (SemIf, Jobe) | Accuracy honesty, answers position bias | S | We already plan order-flip rate |
| 10 | Phase timing ledger in every event (jevmlx) and a paired stopwatch (decision-playground) | Console clarity | S | Fits the frozen event schema as optional fields; check with the founder before changing a frozen section |
| 11 | Model manifest: per-model chat-template flags, default temperature, license, size, tested hardware (ollaya registry, jevmlx aliases, Laya Router) | Realizes "plug in different models" | S | Our `Engine.load(hub id)` already does the loading |
| 12 | Add a 9B to 12B tier on the 5090 (Cygnet: 85 of 111 hard frozen at 35 ms to 66 ms p50) | Largest accuracy lever without training | S | Check licenses: Qwen3.5-9B Apache-2.0; Gemma carries a Prohibited Use Policy |
| 13 | Option-order seed averaging as an optional flag (vLLM PR, reflex) | Small; Jobe found it does not pay at 4B | S | Default off |
| 14 | Neutral-context prior correction (jevmlx) | Possible bias reduction, effect size unknown | M | Needs an extra cached pass; test before adopting |

## 15.11 What to change in our own documents

1. File 09 section 5 claim 1: replace "no decoder engine scores full label strings" with "no decoder engine we found benchmarks full-label, length-normalized and constrained scoring against letters on the same model; Jobe measured one such comparison and letters won by 0.9 point". Add jevmlx, jev-local, Jobe and the llama.cpp PR to the list of option-text scorers.
2. File 09 section 5 claim 4: Windows native plus NVIDIA has several documented competitors (laya.cpp, ollaya, snap, Rizzo Flow, Laya). Reframe as "tested on Windows 11 with an RTX 5090 (Blackwell), with exact commands" or drop it as a differentiator.
3. File 09 section 5 claim 6: jevmlx ships a JSONL heartbeat tailed into a live dashboard. Keep the event stream as a feature, not a first.
4. SPEC section 3.4 step 5 and section 9: add length-normalized and constrained modes and fit temperatures per option-count bucket.
5. SPEC section 10: the race view is not novel; say so in the console README. Our differentiator is live paired raw logs plus the feed for another tool.
6. SPEC section 6: promote `/v1/systemone` so JevBench can rank us; a JevBench row is the strongest public evidence available.
7. README framing stays: approximation, stock model, temperature-scaled. The JevBench rows show exactly where that places us (stock 4B rank 13 to 22 of 91) and what training or a larger frozen model would add.
