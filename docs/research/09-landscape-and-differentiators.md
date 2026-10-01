# 09. Landscape check of the MirethSTM1 differentiators

Date checked: 2026-09-30. Stars, push dates and licenses come from `gh api repos/OWNER/REPO` on that date.
Status tags: VERIFIED (read in code, docs or API output), UNVERIFIED (secondary source or not accessed), CONTRADICTED (the brief says X, reality is Y).
Scope note: `razorback16/openjev`, `jaredpalmer/kev` and the RLCD/harshatheg repos are covered by other research files. They are only mentioned here.

## Summary

The field moved fast: the Jev launch is about two weeks old and the directory lists 55 open alternatives. Of the six brief differentiators:

| # | Brief claim | Verdict |
|---|---|---|
| 1 | Full-label log-prob scoring (multi-token labels, no first-token collisions) | Still open among decoder-based engines. Every decoder engine I read reads one single-token label (`A`..`Z`, digits, `yes`/`no`). cu-Jev lists teacher-forced multi-token scoring as "planned". Only hard-to-reproduce trained encoders (Verdict) score label text, by a different mechanism. **Real differentiator, but narrow.** |
| 2 | Score fields via bins, expected value plus distribution | **Parity.** AnyJev has `Question.score(bins=N)`, simple-jev and cu-Jev return expected index plus a distribution. Ours can only differ in detail (real-valued bin edges). |
| 3 | Temperature scaling with published ECE and reliability diagrams | **Parity, and partly behind.** `open-alternative-jev` ships `TemperatureScaler`, `expected_calibration_error`, a calibration figure and ECE tables. AnyJev ships L1 temperature scaling and ECE tables. `jev-style` and Verdict ship fitted temperatures. |
| 4 | Native Windows/NVIDIA | **Weak.** `jev-style` documents `pip install "jev-style[torch]"` on Windows with CUDA. Most pure-Python HF projects would run on Windows but none tests it in CI. cu-Jev is Linux-only. Differentiator only if we test and document it better. |
| 5 | Local MCP server for Claude Code ("existing MCPs need a paid API key") | **CONTRADICTED.** At least four local, no-key MCP servers exist (jev-style, laya-go, jev-x-kit, danna-zhou/jev-mcp via local Kev). Only some MCPs (JevMCP, miaopj0325-collab/jev_mcp) need a paid key. |
| 6 | JSONL event stream for a live console | No project found that ships one. Not proof of absence (see section 5). Plausible small differentiator, cheap to build. |

## 1. The four projects named in the brief

| Project | What it is | License | Stars | Last push | Native Windows | Needs paid key | Full-label scoring | Calibration |
|---|---|---|---|---|---|---|---|---|
| [dtunai/cu-Jev](https://github.com/dtunai/cu-Jev) | C/CUDA engine for Qwen3.5 (0.8B to 9B), TypeSafe wire format (`POST /v1/systemone`), no PyTorch at runtime | Apache-2.0 | 4 | 2026-09-23 | No. README: "Requirements: Linux, an NVIDIA GPU with compute capability 8.0+" | No | No. Single-token labels; "Teacher-forced multi-token scoring of the option names themselves is planned" | No. "not calibrated the way Jev's RLCD-trained model claims to be. Temperature / isotonic calibration ... is next." Reports ECE 0.104 (4B) uncalibrated |
| [Heman10x-NGU/Verdict-open-jev](https://github.com/Heman10x-NGU/Verdict-open-jev) | Non-autoregressive 151M ModernBERT/GLiClass encoder, trained with RLCD-style loss, WebGPU playground | Apache-2.0 per LICENSE file and README badge (GitHub API reports `NOASSERTION` because the LICENSE text is trimmed) | 109 | 2026-09-28 | Not stated (browser and Python) | No | Different mechanism: candidate descriptions are prefixed as labels into a bidirectional encoder, no autoregressive log-prob. Its own audit: label token length vs selection probability r = 0.1027 | Yes: post-hoc L-BFGS temperature scaling, fitted T = 1.4265, ECE 1.13% equal-width on its held-out split |
| [NicolasRisso/JevMCP](https://github.com/NicolasRisso/JevMCP) | MCP client that fans questions out to hosted Jev | MIT | 0 | 2026-09-22 | Yes (PowerShell install docs), but only a client | **Yes**: `TYPESAFE_API_KEY` or `OPENROUTER_API_KEY`, else `ConfigError` | n/a (proxy) | n/a (proxy) |
| [miaopj0325-collab/jev_mcp](https://github.com/miaopj0325-collab/jev_mcp) | Node MCP client exposing `jev_noul`, `jev_choice`, `jev_score`, `jev_batch_decisions` | MIT | 1 | 2026-09-20 | Node, untested | **Yes**: `OPENROUTER_API_KEY` (endpoint `https://openrouter.ai/api/alpha/decisions` in `src/index.js`) | n/a | n/a |

Sources (VERIFIED, read in a shallow clone): cu-Jev commit 2699e7c README lines ~205-217, ~276-280, ~320-326; Verdict commit 30f1556 README "Known boundaries" item 5 and the calibration table; JevMCP commit 4531c9d `src/jev_mcp/config.py` lines 12-17 and 60; jev_mcp commit 238674a `src/index.js` lines 8 and 21.

Takeaway: the brief's four examples confirm the claim for JevMCP and jev_mcp (both need a key), but they are thin clients. They say nothing about whether a local no-key MCP exists (section 3).

## 2. Closest competitors found (not in the brief's list)

These are the projects that actually overlap our design: a decoder LLM, one prefill, read logits at the answer position. All are local and keyless.

| Project | What it is | License | Stars | Last push | Windows | Paid key | Full-label | Calibration |
|---|---|---|---|---|---|---|---|---|
| [featherless-ai/simple-jev](https://github.com/featherless-ai/simple-jev) (146d3a6) | HF Transformers + PyTorch server, shared-prefix KV cache, `POST /v1/classifier`, TypeSafe-style choice/score/noul. Needs Python 3.12+ | Apache-2.0 | 575 | 2026-09-30 | No Windows mention; CI is ubuntu only | No | No. README: "answer labels that each extend the rendered prompt by exactly one distinct token" (two-letter labels for more than 26 options) | No. Response carries `"calibration": "not_calibrated"` (`common/response_scoring.py` line ~167) |
| [ikermoel/open-alternative-jev](https://github.com/ikermoel/open-alternative-jev) (3689c67) | Python package `so1`, in-process, HF Transformers and vLLM backends, packed shared state, no HTTP server | Apache-2.0 | 56 | 2026-09-25 | No Windows mention; CI ubuntu only | No | No. README: "Option labels must be single tokens", letters A to Z, raises if not | **Yes.** `TemperatureScaler`, `expected_calibration_error`, `cross_fit_temperature` in `so1/calibration.py`; `benchmarks/figures/calibration.png`; ECE 0.020 (Qwen3.6-27B, typed-decisions) |
| [nokia-applied-research/AnyJev](https://github.com/nokia-applied-research/AnyJev) (10d5db9) | Python package, HF and vLLM backends, levels raw / L0 (rotation debiasing) / L1 (temperature) / L2 (closed-form head) | Apache-2.0 | 988 | 2026-09-28 | No Windows; CI ubuntu, Python 3.10 and 3.12 | No | No. `anyjev/readout.py`: "labels ... single tokens whose logits we read"; refuses multi-token with `LabelTokenError` | **Yes.** L1 temperature scaling; ECE 0.240 raw to 0.095 L1 on typed-decisions (their table); `Question.score(bins=5, scale=...)` for score |
| [ekzhang/openjev-sglang](https://github.com/ekzhang/openjev-sglang) (5f633dc) | TypeSafe-compatible API on Qwen3.6-35B-A3B over SGLang radix cache, runs on Modal B200 | no license file or SPDX in the repo (GitHub API returns none) | 335 | 2026-09-25 | No (Modal, Linux containers) | No (you pay Modal) | No. "prefill plus first-token-readout workload" | No. README: "not calibrated estimates of correctness" |
| [lawrence3699/jev-style](https://github.com/lawrence3699/jev-style) (1d86451) | Fine-tuned Qwen3.5 0.8B/2B decision models, `serve` (MLX, PyTorch CUDA or CPU, llama.cpp), MCP tools `decide/noul/choice/score`, Claude Code guard skill | Apache-2.0 | 9 | 2026-09-27 | **Yes, documented**: `pip install "jev-style[torch]"  # Linux, Windows or an Intel Mac` | No | Not stated (trained model, own protocol) | Yes: "temperatures fitted on held-out data" per release; `jev-style eval` reports automation rate at 1, 5, 10% error budget |
| [wfzyx/von](https://github.com/wfzyx/von) (0acaa06) | 395M ModernBERT encoder, TypeSafe-compatible, CPU (OpenVINO), CUDA, ROCm, MPS | Apache-2.0 | 795 | 2026-09-30 | Not stated; Docker image documented | No | Encoder, different mechanism | Yes; `von calibrate labels.jsonl` refits confidence on your labels |
| [neko233-com/laya-go](https://github.com/neko233-com/laya-go) | Go server, CLI and MCP (`laya`/`laya_decide`) around the Laya encoder family | MIT | 5 | 2026-09-21 | **Yes**: `deploy/install-service.ps1` Windows service, `install-mcp.ps1` | No | Encoder | Not stated |

Also seen, not cloned: [Kadihx/jev-x-kit](https://github.com/Kadihx/jev-x-kit) (MIT, 2 stars; offline MCP and CLI whose default backend is a deterministic heuristic simulator, real engines optional via env var, README claims a calibration check tool) and [danna-zhou/jev-mcp](https://github.com/danna-zhou/jev-mcp) (0 stars; MCP client whose "Option A" points at a local Kev server). Other names from the directory, unverified by me: open-spark-jev (21 stars, Apache-2.0, DGX Spark), NanoJev, KaLM-Jev, mini-Jev, decider-2b, Open-Jev-9B, jev-calibrate (smkrv, 31 stars, MIT, calibrates question wording against your labels).


## 3. Directory check: madewithjev.com

VERIFIED by fetching https://madewithjev.com on 2026-09-30 and reading the embedded page data and the open-source guide page (https://madewithjev.com/open-source-jev, dated 2026-09-28):

- Total entries: **749** (brief said about 330; 330 is only the "GitHub" type). Type counts: Jev Engineering 15, GitHub 330, Skills 12, X posts 433, YouTube 57, Sites 43, Resources 159. Categories overlap, so they do not sum to 749.
- Category "Open source": 146 entries. The open-source guide says **55 projects** train, adapt or replace a System One model, in 5 families.
- The directory is self-described as independent and each figure as "the project's own ... None has been re-run here".
- Local-first engines: yes, several (von under 15 ms local, Laya about 1 GB, kev on a MacBook, Simple Jev, AnyJev, Open Alternative to Jev). The guide's own summary of what open alternatives do not match: "Calibrated confidence is the part nobody has matched in public" and "No alternative has published a calibration comparison run by anyone other than its own author."
- Local MCPs: the GitHub listing shows many MCP entries (jev-mcp, typesafe-mcp, jevwire, maza, laya-go and others), most wrapping hosted Jev. laya-go is the one listed as "A Go server plus agent CLI and MCP for open System 1 models".
- Windows-native engines: the directory text I parsed has no Windows filter. The only Windows-specific evidence found is in repos (jev-style, laya-go, JevMCP client).
- One correct point for us: the guide lists "batching" (shared prefill), "calibration" and "keeping a GPU warm" as the three things TypeSafe absorbs. A project that does the first two well and documents the third is on message.

## 4. Web search, September 2026

WebSearch for local keyless MCP servers for Claude Code returned: danna-zhou/jev-mcp, darwintechlab/claude-jev (MIT, 2 stars, live-only, needs the TypeSafe API), lawrence3699/jev-style, Kadihx/jev-x-kit, amidabuddha/jev-decision-mcp (MIT, 1 star, hosted Jev), rahulrajaram/jev-mcp (MIT, 0 stars, hosted Jev), paulrobello/jev-mcp-server, CodeIA-Academy/jev-mcp, and a community integration page at systemonemodels.org. GitHub search additionally surfaced sutro-sh/jev-align (301 stars, Apache-2.0), y0usaf/pi-jev (150 stars, MIT, hosted Jev as tool-call gate), kraayenjon/awesome-jev (164 stars) and Heman10x-NGU/openJev-verdict-2.0. I did not read the last group.

UNVERIFIED items: the stars and code of the hosted-Jev MCPs beyond what `gh api` returned, and any project created after 2026-09-28 that GitHub search had not indexed.

## 5. Per-claim analysis

### 1. Full-label log-prob scoring
- No decoder-based project read in code scores full label strings. simple-jev, open-alternative-jev, AnyJev, openjev-sglang and cu-Jev all use single-token labels and either raise or fall back when a label is multi-token. AnyJev's fallback is to switch to letters (`readout.py` lines 36-40).
- The common trick is to hide the real label behind `A`, `B`, `C`. That removes first-token collisions but introduces position bias (AnyJev measures an order-flip rate of 0.230 and fixes it by rotating options K times). Full-label scoring on the real label text avoids the letter indirection altogether, which is a genuine technical difference.
- Risk we must own: summed log-prob penalizes long labels. Verdict's audit of label length vs probability (r = 0.1027) shows the question is measurable. We should report length-bias numbers (and offer a length-normalized option) rather than assert there is none.
- Cost: one suffix per candidate label instead of one per question. With up to 255 options this is where the chunked cache-repeat design matters. No project I read benchmarks this, so no public baseline to beat or match.
- Status: differentiator stands (VERIFIED for the repos read; UNVERIFIED for the 55-project long tail).

### 2. Score fields via bins, expected value plus distribution
- AnyJev `Question.score(bins=5, scale=(0,1))` produces equal-width bins or explicit ordered levels, never permuted (`anyjev/question.py` lines 45-53). simple-jev returns "Expected zero-based rubric index, confidence, distribution, and rubric legend". cu-Jev returns `score`, `probabilities`, `confidence`.
- Conclusion: parity. Note TypeSafe's wire format already specifies score as an ordered legend with a probability per level, so any drop-in engine reproduces it.
- Possible edge: real-valued `min`/`max` with bin-center expected value and a full-label option for bin names.

### 3. Calibration with published ECE and reliability diagrams
- Parity already exists: open-alternative-jev (temperature scaler, ECE, calibration figure, cross-fitting), AnyJev (L1, ECE 0.240 to 0.095), Verdict (L-BFGS T = 1.4265), jev-style (shipped temperatures), von (`von calibrate`).
- Everyone reports ECE on `LocalLLaMA/typed-decisions` or their own set, with no third-party replication (the directory says so). A careful reproducible benchmark (fixed seeds, n, bin count, bootstrap CIs, plus Brier and NLL since open-alternative-jev notes ECE alone can be gamed by a base-rate predictor) is worth having, but it is parity on the feature and only a possible edge on rigor.
- Do not claim "first with published ECE".

### 4. Native Windows/NVIDIA
- Evidence of Windows in open engines: jev-style documents a Windows CUDA install; laya-go ships a Windows service installer. None of the decoder engines (simple-jev, AnyJev, open-alternative-jev, cu-Jev) documents or CI-tests Windows; cu-Jev is explicitly Linux-only.
- Since they are pure PyTorch/Transformers they probably run on Windows, but nobody says so. A tested RTX 50-series (Blackwell) Windows setup with `cu128` wheels is a true differentiator only if we ship exact commands and a CI or recorded test. UNVERIFIED that the others fail on Windows.

### 5. Local MCP server
- CONTRADICTED as stated. The brief says existing MCPs need a paid API key. Reality: JevMCP and jev_mcp need one, but jev-style ships local MCP tools with its own models, laya-go ships a local MCP, jev-x-kit runs offline, and danna-zhou/jev-mcp can target a local Kev.
- What could still be different: a local MCP backed by our full-label, calibrated engine, with calibration shown in the tool output. That is a packaging advantage, not a first.
- The brief already makes this cut-first. The finding supports leaving it cut.

### 6. JSONL event stream for a live console
- I found no project that ships a documented JSONL event stream designed to be tailed by a live UI. cu-Jev draws probabilities in a demo game, Verdict exports "inference receipts", jev-style has a Playground, but none publishes a stable line-delimited event schema I found. The absence is UNVERIFIED across 55+ projects.
- It is cheap, unlike the model work, and it makes the tool observable. Worth keeping as a small extra, not as the headline.

## 6. What this means for MirethSTM1

### Revised differentiator list, in priority order
1. **Full-label log-prob scoring over the real label text**, with per-label length-bias reporting and an optional length-normalized mode. The only claim I could not find matched among decoder engines.
2. **Rigorous, reproducible calibration and benchmarks** (temperature scaling is parity; the edge is method: held-out split, bootstrap CIs, ECE plus Brier plus NLL, reliability diagrams, before and after, run on a consumer GPU). Frame it as "we publish how to check it yourself", not as a first.
3. **Verified native Windows + NVIDIA (RTX 50-series, `cu128`) with no WSL**, with exact commands and a test recorded. Only jev-style documents Windows and it is a different product (trained models).
4. **One engine behind one wire-compatible API, CLI and JSONL event stream**, all local, all keyless. Parity on API shape, small edge on the event stream.
5. **Local MCP server** (parity at best, keep cut-first).
6. **Score bins with expected value plus distribution**: parity, list as a feature not a differentiator.

### Concrete implications for the SPEC and scaffold
- Do not write "existing MCPs need a paid API key" anywhere. Rewrite as "MCP tool available for agents, runs locally, no key".
- Do not write "first" or "only" for calibration, score bins, or Windows. Use "tested on".
- Design the scorer so a single-token-letter fast path (simple-jev style) can be benchmarked against full-label scoring in the same codebase. That gives a head-to-head table where full-label either wins (fewer order flips, no letter indirection) or loses on latency. Either is a publishable result.
- Report both accuracy AND order-flip rate (reverse or shuffle options), since AnyJev measures position bias and full-label scoring should reduce it. This is our most testable selling point.
- Score `min`/`max` bins: implement real-valued bin centers and return `value` (expected), `probs` per bin and the bin edges, so the output is a superset of simple-jev and cu-Jev.
- Keep the TypeSafe wire compatibility (choice up to 255, noul, score) as in simple-jev and cu-Jev; every serious competitor does, so it is table stakes, not a feature.
- Benchmark harness: use `LocalLLaMA/typed-decisions` (used by cu-Jev, open-alternative-jev, AnyJev, Laya, Verdict) in addition to the brief's AG News, Banking77, SST-2 and Yelp, so numbers line up with the public table. Dataset access details belong to the benchmark-design research file.
- Event stream: freeze the JSONL schema early and document it with one real sample file. It is the single feature nobody ships and it costs little.
- Model licensing and base-model choice are covered in file 06.

### Suggested honest framing for the README
MirethSTM1 is a free, Apache-2.0 approximation of a Jev-style decision engine, not an equivalent: it uses an unmodified open Qwen3 model rather than a model trained for calibrated decisions, so its probabilities are the base model's own, rescaled by a temperature fitted on held-out labels. What it tries to do well is narrow and checkable: it scores the full text of each answer option instead of a single letter or first token, it reports an expected value and a distribution for score fields, it publishes accuracy, ECE, Brier and reliability diagrams with the scripts that produced them so you can rerun them, and it runs natively on Windows with an NVIDIA GPU without WSL. Several other open projects (simple-jev, AnyJev, open-alternative-jev, jev-style and trained encoders such as Verdict and von) cover overlapping ground, and we link to them; try them and compare on your own labels before trusting any threshold.
