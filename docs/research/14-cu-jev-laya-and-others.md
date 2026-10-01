# 14. cu-Jev, laya-go, jev-x-kit, openjev-sglang, jevmlx, jev-on-a-laptop (deep read)

Status: research note, 2026-09-30. Read in shallow clones, code first, README second. Tags: VERIFIED (read in code or data files), UNVERIFIED (claimed, not reproduced or not read), CONTRADICTED (the evidence disagrees with an earlier claim, ours or the project's). Paths are repo-relative with the commit read. No GPU code was run, nothing was installed, no paid calls were made. Quotes are short on purpose.

Stars, forks and last push come from the GitHub API on 2026-09-30.

## 0. What matters most (read this first)

1. **CONTRADICTED: "no decoder engine scores full label strings" (doc 09, row 1 and the line about single-token labels).** `bnsd55/openjev` (jevmlx) scores real, multi-token option strings by default (`labels` scorer) with a token trie, and reports it beating single-letter "slots" scoring on its own small benchmark. The headline "full-label scoring is a differentiator" must be reworded. What stays true: jevmlx is Apple Silicon only, 4-bit, scores a different quantity (constrained path probability, see 1.5), and does not pack in one no-copy pass.
2. **There is a Windows/NVIDIA competitor we had not read: `rorshopping/parallel-decisions`** (found through jev-on-a-laptop). Torch/CUDA backend, tested on an RTX 2060 SUPER, CUDA Graphs, prefix reuse. 2 stars, no LICENSE file in its root listing (UNVERIFIED beyond the API 404). It weakens "native Windows/NVIDIA" as a differentiator, but only as a tested, documented path, not as a clean engine (it still scores first tokens plus a collision fallback).
3. **We now have the one public apples-to-apples accuracy and calibration number against real hosted Jev on a public dataset**: `ekzhang/openjev-sglang` BoolQ dev (3,270 examples, yes/no as `noul`): stock Qwen3.6-35B-A3B first-token scoring 89.45% vs hosted Jev 91.56%, selected-answer ECE (10 bins) 4.05% vs 2.51%. That is the bar to aim at, and its protocol is worth copying (section 6).
4. **laya-go is not what doc 09 says.** It is a hand-weighted linear rule router with no neural model and no text input. CONTRADICTED: "Go server around the Laya encoder family". Zero threat to our engine claims.
5. **jev-x-kit is an agent-orchestration client**, not an engine. Its default backend is a seeded fake. No threat to claims; some ideas for the Claude Code hook side.
6. **Speed numbers across these projects are not comparable** (section 8). The cleanest GPU number is cu-Jev: Qwen3.5-4B bf16, RTX 5090, 16 questions of 40 tokens, 1,024-token state: 91 ms cold, 34 ms with the state cached. Nobody publishes a 1.5B first-token number on an NVIDIA card that matches the original demo setup.
7. **Not in my assignment but seen while searching**: `NandhaKishorM/laya` (29,298 stars, Apache-2.0, pushed 2026-09-29), a trained ModernBERT/mmBERT encoder family with a Jev-like API, plus `mizorewww/laya-mlx` (6,659), `mizorewww/laya-coreml` (1,530), `receptron/laya` (664, ONNX for Node). These have far more mindshare than every project I read. They are trained encoders, so the "stock decoder, any model plugs in" position still stands, but anyone comparing "open Jev" will meet Laya first. Needs its own file.

## 1. Project cards

### 1.1 dtunai/cu-Jev (cuda-Jev)

| Item | Finding |
| --- | --- |
| What | C/CUDA inference engine for Qwen3.5 (hybrid Gated DeltaNet plus gated GQA), TypeSafe wire format on `POST /v1/systemone`, FastAPI shell in Python. `README.md:1-34` VERIFIED |
| License | Apache-2.0, LICENSE file present (header read). GitHub SPDX Apache-2.0. cJSON (MIT) vendored in `third_party/`. VERIFIED |
| Stars / forks / push | 4 / 0 / 2026-09-23 |
| Open | Code yes. Weights are stock Qwen3.5 checkpoints, not shipped. No training, no data of its own beyond a benchmark replay script. VERIFIED |
| Models | Qwen3.5 0.8B, 2B, 4B (recommended), 9B; 27B and MoE not supported. `README.md:73-80` VERIFIED. Stock, no LoRA, no head. Pluggable only inside that one architecture family (kernels are hand-written for it); no other family. |
| Scoring | One token per question. Options are shown as letters, read as the logits of single-token labels `A..Z`, `AA..`, then lowercase; noul reads `yes`/`no`; score reads digits `0..9`. `python/cujev/prompt.py:52-54,83-110` VERIFIED. Labels are verified single tokens at load (`prompt.py:65-77`). Option names are never scored. Max 255 choices (`systemone.py:112-113`), 10 score levels (`systemone.py:119-120`). |
| Batching | State prefilled once, each question is an isolated short branch that attends to the state KV cache and starts linear-attention layers from a read-only copy of the recurrent state (`README.md:24-27`). Questions never see each other. Branches packed into eval calls under a token budget (`systemone.py:160-170`). Last-state-ids equality gives a cross-request prefix cache (`systemone.py:153-156`, header `x-cujev-prefix-cached` in `server.py:86-92`). |
| Probabilities | `softmax` of the label logits; confidence `1 - H(p)/ln K`; score is the expected index (`systemone.py:52-66,175,188`). |
| Speed | See section 8. Cold 4B 91 ms (5090) / 223 ms (3090); cached-state eval 34 ms / 84 ms; 0.8B 31 ms / 66 ms cold, 10 ms / 21 ms cached (`README.md:176-194`). Method: `benchmarks/benefit.py`, 1,024-token state, 16 questions x 40 tokens, bf16. HF transformers batched baseline 939 ms (4B, 5090). VERIFIED as published, UNVERIFIED as reproduced (no GPU run). |
| Accuracy | `LocalLLaMA/typed-decisions` test, 400 cases x 5 questions, zero-shot, one prompt: 0.8B 46.4%, 2B 48.9%, 4B 63.1% (Brier sum 0.224, ECE 0.104), 9B 64.0% (ECE 0.112). `README.md:205-210`. ECE uses 10 bins (`benchmarks/typed_decisions.py` docstring). Brier is against the gold distribution, not the label. Wall time per case there is `time.perf_counter` around `evaluate` (`typed_decisions.py:59-61`), so it includes tokenization and Python. |
| Calibration | None. "not calibrated ... Temperature / isotonic calibration ... is next" (`README.md:321-324`). |
| Correctness testing | Compares full-vocab logits with `transformers` (argmax and top-5 identical on 0.8B to 9B); fp32 oracle; 59 GPU tests with per-kernel sync (`README.md:157-171`). Strong engineering evidence. |
| Platform | "Linux, an NVIDIA GPU with compute capability 8.0+" (`README.md:55-57`). The author develops under WSL2 (`docs/ARCHITECTURE.md:90`). No native Windows. |
| API / UI | TypeSafe `POST /v1/systemone`, `/v1/models`, `/health`, 422 error shape, optional bearer key. Official `typesafe-sdk` works by changing the base URL. No MCP. A browser game ("Starfighter") draws probabilities live at about 95 ms per decision (`README.md:36-51`). Not a race view. |

Threats to our claims: (a) it is the fastest published engine and its cached-state path (34 ms) is a speed bar we will be compared with, though on a different model and a 5090; (b) the "isolated branch" design is a real alternative to our one shared prompt that lists all questions; (c) it does not threaten full-label scoring (it lists it as "planned", `README.md:325-326`), nor Windows (Linux only).

### 1.2 neko233-com/laya-go

| Item | Finding |
| --- | --- |
| What | A Go HTTP server, CLI and stdio MCP server ("laya", "laya_decide"). The "models" are built-in tables of hand-set weights: `score = bias + sum(weight * feature)`, then `softmax(score / T)`. `internal/engine/engine.go:74-96` and `models.go:36-60` VERIFIED. README: "Built-in models are rule/weight patterns, not neural checkpoints" (`README.md:313`). |
| License | MIT, file present. GitHub SPDX MIT. |
| Stars / forks / push | 5 / 1 / 2026-09-21 |
| Input | Not text. Callers pass a map of 0..1 feature names (`README.md:149-172`, "The server does not parse natural language"). Fixed pattern ids like `tool.edit`, `agent.finish`, `guard.block`. |
| Models / pluggable | Three rule sets (`laya-mini`, `laya-base`, `laya-pro`); a JSON overlay can hot-reload more rule sets. No LLM. |
| Speed | `latency_us` field around the loop; budget claims "1 ms", "2 ms", "5 ms" are configuration constants, not measurements (`models.go:41,69`). UNVERIFIED as measured. |
| Accuracy / calibration | None. "calibrated probabilities" is a design sentence in `docs/architecture.md`, temperature is a constant 0.1 to 0.15. |
| Platform | Windows service installer and Linux/macOS scripts; Go so any OS. Default bind is `0.0.0.0:7710` with no auth (`README.md:271-273`). |
| API | Own `POST /v1/decide`, `/v1/predict`, `/v1/jev/decide`; not TypeSafe `/v1/systemone`. MCP yes. Admin web page. |

CONTRADICTED: doc 09 called it a server "around the Laya encoder family". It never loads a model. Not a competitor; not a source of engine ideas. Only usable lesson: a feature-map-in, pattern-out router is a different product, and it shows how thin an "open Jev" label can be.

### 1.3 Kadihx/jev-x-kit

| Item | Finding |
| --- | --- |
| What | TypeScript MCP server plus CLI plus Claude Code skill: 28 tools (planning, red-team, research, compaction, guardrail, "gatekeeper" thresholds, memory), all built on a `JevBackend` interface that talks to some other engine. `README.md:10-12,110-140` VERIFIED. |
| License | MIT, file present. GitHub SPDX MIT. |
| Stars / forks / push | 2 / 0 / 2026-09-23 |
| Own engine? | No. Backends are: hosted TypeSafe (`/v1/systemone`), a local OpenJev, a local Laya server, or a "deterministic offline simulator" (`src/core/providers/heuristic.ts:1-11`: seeded random, flagged `synthetic`). `README.md:80-92`. |
| Local LLM path | `chat-compatible.ts` sends `/chat/completions` at temperature 0 with `response_format: json_object` and parses the model's own JSON (`chat-compatible.ts:78-95`). For yes/no it asks the model to write a "probability" number and derives confidence as `clamp(abs(p-0.5)*2+0.15, 0.05, 0.99)` (`chat-compatible.ts:258,286`). That is generation plus a made-up confidence, not logprob scoring. VERIFIED. |
| Speed data | Its own report: hosted Jev single calls 258 to 860 ms (first call 860, rest about 260 to 320, network included), heuristic 0.27 ms, Laya on CPU 184 ms per question in one batch of 10 (`artifacts/backend-benchmark-report.md`, `artifacts/laya-benchmark-report.md`, dated 2026-09-22). Useful only as a sanity check on the hosted API's real latency from a client. |
| Calibration | `jev_calibration_check` buckets confidence into three zones and tests option-order flips (`src/modules/jev-calibration.ts:26-117`). Order-flip test is a good idea (section 7). It reports bucket gaps, not ECE. |
| Platform | Node 22.5+, Windows paths in examples. |

No threat. The "BELKI" gatekeeper (execute above 0.85, split at 0.6 to 0.85, escalate below) is a policy layer that would need calibrated probabilities to mean anything; with its backends it does not have them.

### 1.4 ekzhang/openjev-sglang

| Item | Finding |
| --- | --- |
| What | TypeSafe-compatible API server for Qwen3.6-35B-A3B (NVFP4) on SGLang, deployed on Modal B200. `README.md:1-12` VERIFIED. |
| License | **No license, CONFIRMED.** `git ls-files` has no LICENSE or COPYING; `pyproject.toml` has no `license` field; `gh api repos/ekzhang/openjev-sglang/license` returns 404; repo `license` is null. Default copyright: all rights reserved. We read it as ideas only. |
| Stars / forks / push | 335 / 45 / 2026-09-25 |
| Open | Code yes (no license), model weights are NVIDIA's NVFP4 checkpoint, evaluation results and scripts committed. |
| Models | One profile: Qwen3.6-35B-A3B, NVFP4, fp8 KV (`src/openjev/profiles.py:20-33`). Stock. Connect to any SGLang with `--connect`, but the label/prompt code is tuned to Qwen's chat template. |
| Scoring | First token. Prefix warmed with a one-token generation, then one request per question with `max_new_tokens=1` and `token_ids_logprob` for the label ids, renormalised with softmax at `OPENJEV_TEMPERATURE` (default 1.0). Options render as `A:`, `B:`; choice keys are hidden unless the description is null; labels `A..Z` then verified single-token pairs, up to 64 (`README.md:120-150`, `src/openjev/scoring.py:7-40`). N+1 calls per request. |
| Confidence | `1 - H/ln K` with the docstring "TypeSafe's exact statistic is not published" (`scoring.py:18-21`). |
| Speed | The only latency table is a capped-reasoning experiment: zero reasoning median 0.11 s, p95 0.21 s on 128 MMLU-Pro questions, concurrency 16, "includes both internal SGLang requests but excludes external networking, startup, and the client-side concurrency queue" (`evals/results/thinking-2026-09-18/report.md`). Cold start 387.8 s (`evals/results/boolq-2026-09-18/comparison/deployment.json`). No published end-to-end p50/p95 for the API itself (`mmlu-pro-research` report says so, line 68 and 96). |
| Accuracy | BoolQ dev 3,270: OpenJev 89.45% [88.38, 90.59], Brier 0.0812, log loss 0.2926, selected ECE 4.05%; hosted Jev (via OpenRouter, model id `typesafe/jev-1.13-20260917`) 91.56%, Brier 0.0640, log loss 0.2257, selected ECE 2.51%. No recalibration. 95% bootstrap by passage cluster. `evals/results/boolq-2026-09-18/comparison/report.md`. VERIFIED as data. MMLU-Pro 1,000 questions, one token, thinking off: 58.8% vs hosted Jev 82.9%; Qwen3.8-27B via OpenRouter 60.0%. Prompt variants moved it by +/- 2 to 4 points; the report notes selection bias. 1,024-token reasoning raised the sample to 68.8% (n=128), at 7.25 s median. |
| Platform | Linux containers on B200 (Modal). Python client side runs anywhere. Not a local engine. |
| API | `POST /v1/systemone`, 64 questions max, 2 to 64 answers per choice/score, chat-message states, `Server-Timing` and `x-openjev-prefix-tokens` headers, Scalar API docs at `/`. |

Threats: weakest on our claims. It proves a big stock model with first-token scoring reaches within about 2 points of hosted Jev on BoolQ, which is both a target and a warning: the interesting gap on reasoning-heavy sets (MMLU-Pro, 24 points) is not closed by prompts. It says nothing about full-label scoring.

### 1.5 bnsd55/openjev (python package `jevmlx`)

| Item | Finding |
| --- | --- |
| What | Apple Silicon (MLX) engine: schema of boolean, enum and multi-select fields plus a context string; one prefill, KV cache copied per row, one batched pass, JSON assembled from winners. `README.md:5` VERIFIED. 20,932 lines of Python in `jevmlx/`, 564 commits, heavy test and benchmark tooling. |
| License | MIT, file present. GitHub SPDX MIT. Copyright line also names a second author for portions. |
| Stars / forks / push | 68 / 8 / 2026-09-25 |
| Models | Aliases: `quality` Qwen2.5-7B-Instruct-4bit (default), `fast` Qwen2.5-3B-Instruct-4bit, `test` Qwen2.5-1.5B-Instruct-4bit (called "too small for production"); any mlx-lm Hub id works (`README.md:26-34`). Bench folders exist for Qwen2.5-7B, Qwen3-8B, Llama-3.1-8B and Gemma-3-12B, all 4-bit. A model needs an adapter for its LM head (`adapters.py`). Also an OpenAI-compatible mode: one request per field with server top-k logprobs, floor probability and a `truncated` flag (`README.md:95-103`, `openai_slots.py`). Stock models, no training. |
| Scoring: two modes | `labels` (default) scores the real option strings through a token trie; `slots` scores single-token neutral alias codes picked per tokenizer by a codebook search (`README.md:46`, `schema.py:285-345`, `PROMPT_PROTOCOL.md` section 2). |
| What a label score is | **Constrained path probability**, not a sequence likelihood: at every branch point of the trie the child logits are log-softmaxed over the allowed continuations only, and each choice multiplies those factors (`jevmlx/trie.py:6-12,125-183`). Tokens shared by all options, and mass the model puts outside the allowed continuations, are discarded. A strict-prefix option is rejected at compile time (`trie.py:36-39`). Ours (SPEC 3.2 and 3.4) sums the log-probs of every token of ` "name"}` and then normalises over labels, so it keeps closing-token evidence and prefix competition. These are different estimators. No head-to-head of the two exists anywhere I read. |
| Legal mass | Each branch also records `sum(exp(z_allowed)) / sum(exp(z_vocab))`, a leakage signal "did the model want any valid code here", exposed per field (`trie.py:150-166`, `ARCHITECTURE.md` field_telemetry table). |
| Batching | Prefill once; KV cache copied per scoring row with `merge([copy.copy(c) ...])` (`engine.py:433-449`); rows chunked by a measured memory budget with halve-and-retry on Metal allocation failure; `decide_many` batches several contexts. Not a shared no-copy cache. |
| Near-tie rescore | Scalar fields whose top two scores are within 0.05 nats are rescored at batch size 1 (`engine.py:62,856`). The package's own parity gate reports DRIFT between batched and single-call scores: 0.078 nats (Qwen2.5-7B) and 0.750 (Qwen3-8B) in the README leaderboard (`README.md:157-162`). A CHANGELOG entry states the earlier "band" check "could never fail and tested nothing" and was removed. |
| Calibration | `fit_temperature`: golden-section search on log NLL over [0.05, 20], 60 iterations, 10-bin ECE; pooled logistic for multi-select; a typed `CalibrationBundle` that the engine refuses when the prompt version, scoring mode, prior mode or model revision differ (`calibrate.py:71-93,104-125,182-215,224-354`). Same method as our SPEC section 9, minus our "raise at bounds" check (theirs returns the midpoint silently). Optional neutral-context "prior correction" subtracts the score of the same schema under `(no context provided)` (`engine.py:2591`). No published ECE before/after for any model. UNVERIFIED that any shipped checkpoint uses a fitted T. |
| Speed | Per-case median on an M5 Max 128 GB: Qwen2.5-7B-4bit `labels` 244 ms on 24 bundled cases, 586 ms on 44 TypeSafe-public cases; Qwen3-8B 234 / 647 ms; Llama-3.1-8B 248 / 707 ms; Gemma-3-12B 409 / 1,182 ms (`benchmarks/results/*/SUMMARY.md`; the table header has 14 columns but rows have 15, so the latency column assignment is my reading; the leaderboard's "0.6 s" for labels agrees). Naive generate-and-parse is 1.5 s on the same 45 cases. Presets have 19 to 30 fields. |
| Accuracy | 45 public example cases from TypeSafe's eval pages, scored as agreement with the published consensus labels: 7B `labels` 82.1% [68.7, 90.7], `slots` 63.2%, naive 67.7%; Qwen3-8B `labels` 84.6%, `slots` 34.7%, naive 69.3% (`README.md:157-162`). The README states "indicative, not the same test"; hosted Jev is cited at 67.8% on a private full eval, not comparable. A prompt change (v10) dropped `labels` from 82.1 to 71.6 and was reverted (`CHANGELOG.md`, "revert: field-local prompts"), so prompt noise on n=45 is about 10 points. 4-bit weights throughout. |
| Platform | Apple Silicon only, Python 3.12 (`README.md:16`). No Windows, no CUDA. |
| API / UI | `POST /v1/systemone` plus its own `/decide`, `/ready`, queue with 429 backpressure; TypeScript client; no MCP. `jevmlx watch` is a live dashboard for benchmark runs: terminal and a self-contained web page updated over SSE by watching file mtimes, DOM patched in place (`BENCHMARKING.md`, "Watching a run"). It monitors a bench, it does not race models. |

Threats: this is the one project that matches our full-label headline in spirit, with 564 commits of rigor. See section 0 item 1. It does not threaten Windows/NVIDIA or the race view, and it cannot load a non-MLX model.

### 1.6 rorshopping/jev-on-a-laptop (and its release library)

| Item | Finding |
| --- | --- |
| What | A research and benchmark repo, not an engine. It runs the community "parallel constrained decoding" artifact (HF repo `harshatheg/Qwen-2.5-1B-RLCD`) on Apple Silicon, audits it, and benchmarks three models. `README.md:3-17` |
| License | Repo has an MIT LICENSE file (header read) with an extra note about third-party code after a `---` line. GitHub reports NOASSERTION, almost certainly because of that trailing note, not because the license is missing. The engine itself is cloned at install from HF (Apache-2.0 per the project), not vendored. |
| Stars / forks / push | 24 / 1 / 2026-09-17 |
| Key audit finding | The HF "model" has zero weights and no training; it is a harness around stock Qwen2.5-1.5B-Instruct-4bit (`docs/04-the-hf-artifact-what-it-actually-is.md`: `usedStorage: 0`, no safetensors). This is almost certainly the lineage of the demo the founder cited (stock 1.5B, first-token scoring). VERIFIED by that audit; the 68 to 89 ms and 270 ms figures are the founder's, UNVERIFIED here. |
| Scoring of that engine | First token of each option, with a sequential collision fallback of up to 4 extra passes per colliding field and a synthesized confidence `clamp(prod p, 0.75, 0.9999)` (`docs/05-how-parallel-constrained-decoding-works.md` step 7). Hardcoded ChatML. enum and boolean only, up to 255 choices. |
| Speed | M5 MacBook Air 16 GB, 4-bit, 28-field fraud preset: 1.5B 0.41 s (naive JSON 3.3 s), 7B 1.52 s, 8B 2.03 s; 255-choice 4-field 0.15 s; support-triage 0.76 s because of collisions (`README.md:66-72`, `docs/05` lines on 414 ms and 756 ms). On the same engine: 147 ms per 3-field case for 1.5B, 611 ms for 7B (`quality-eval/SUMMARY.md`). |
| Accuracy | 24 synthetic cases x 3 fields: 1.5B primary 58.3% (majority baseline 54.2%), 7B 95.8%, 8B 91.7%. On 343 shared reference pairs from TypeSafe's public eval: hosted Jev 86.6%, local Qwen2.5-7B 73.8% (`README.md:105-120`). Confidence "does not reliably flag errors": 7B was above 0.90 on 13 of 20 wrong fields (`README.md:86`). Labels are rule-built, synthetic, tiny n. |
| Calibration | Raw softmax; its release library `parallel-decisions` ships temperature fitting plus adaptive-bin ECE (`CALIBRATION.md` in that repo, read through the API). No published before/after number found. |
| GPU / Windows | The release library `rorshopping/parallel-decisions` (2 stars, pushed 2026-09-29, no LICENSE file in its root listing, UNVERIFIED beyond that) has a torch backend. On an RTX 2060 SUPER with Qwen2.5-0.5B fp16: CUDA 8.1x faster than CPU; shared-prefix reuse 88.5 to 61.9 ms (8 fields, 405-token prefix); CUDA Graph replay 60.1 to 51.8 ms (8 fields) and 240.9 to 83.8 ms (27 fields); one fresh verification run refused graph capture because CUDA reported zero free bytes (`docs/14-gpu-torch-backend.md`). It also found and fixed a real bug: HF `DynamicCache.batch_repeat_interleave` mutates layers in place, so a shallow cache copy silently corrupts the caller's cache. |
| API / UI | Python library and CLI, Gradio and static demo Spaces. No `/v1/systemone` in the repo I read. |

Threats: the Windows CUDA evidence undermines our "Windows/NVIDIA is open ground" story (section 0 item 2). The audit style, and the "Real vs marketing" table, is the honesty standard we should meet.

## 2. Is anything here plug-in-a-model friendly?

| Project | Other models | How |
| --- | --- | --- |
| cu-Jev | Only Qwen3.5 sizes | Hand-written kernels for one architecture |
| laya-go | n/a | no model |
| jev-x-kit | Any OpenAI-compatible endpoint | generation, not scoring |
| openjev-sglang | Any SGLang-served model in principle | prompt and labels tuned to Qwen, one profile shipped |
| jevmlx | Any mlx-lm model with an LM-head adapter, any OpenAI-compatible logprob server | trade-off documented: top-k truncation gives a floor probability and a `truncated` flag |
| jev-on-a-laptop / parallel-decisions | Any mlx-lm or torch causal LM | hardcoded ChatML in the upstream artifact |

Nobody ships a model picker or multi-model comparison. The "plug in many models" idea is open, and a race view across models is open (jevmlx watches runs; it does not race engines).

## 3. Windows, NVIDIA, install friction

| Project | Windows native | NVIDIA | Install |
| --- | --- | --- | --- |
| cu-Jev | no (Linux, WSL2 by the author) | yes, sm_80+, needs nvcc 12.4+/CMake/uv | build from source |
| laya-go | yes (service) | n/a | Go build or release |
| jev-x-kit | yes (Node) | n/a | npm install |
| openjev-sglang | client yes, server runs on Modal Linux | B200 | `uv sync`, Modal account |
| jevmlx | no | no | `./setup.sh`, Mac only |
| parallel-decisions | yes, tested on one 2060S | yes, torch | pip |

Our torch 2.11 + cu128 + sm_120 tested path on Windows remains a real, if narrow, gap, but the claim must read "tested and documented", never "only".

## 4. TypeSafe API and MCP compatibility

`/v1/systemone` compatible: cu-Jev (byte-for-byte shape, 422 error list, aliases), openjev-sglang (plus chat-message states), jevmlx (maps to its schema). Not compatible: laya-go (own route), jev-x-kit (client of the real API). MCP: laya-go and jev-x-kit only; none of the engine projects.

Shape notes worth matching: cu-Jev and sglang both report confidence as `1 - H/ln K`, so that is table stakes. Score is the expected index in both. sglang accepts a list of chat messages as `state` and renders it with the chat template (`README.md:90-98`); we should accept it too.

## 5. Calibration evidence: who measured what

| Project | Raw ECE | Recalibrated | Note |
| --- | --- | --- | --- |
| cu-Jev | 0.104 (4B), 0.112 (9B), 0.165 (2B), 0.218 (0.8B), 10 bins, 5-question typed-decisions rows | none | zero-shot letters |
| openjev-sglang | 4.05% selected, 4.64% P(yes), BoolQ | none | hosted Jev 2.51% / 3.53% |
| jevmlx | not published | fit exists, no numbers | |
| jev-on-a-laptop | informal: 7B over 0.90 on 13 of 20 wrong | none | n=72 |

Nobody has published a before and after temperature-scaling result on a public set. That stays open for us (doc 09 reached the same conclusion).

## 6. Ideas worth reimplementing (fact and shape only, never code)

Ranked by expected impact on speed and accuracy. Effort: S under a day, M a few days, L a week or more.

| # | Idea | Source fact | Impact | Effort |
| --- | --- | --- | --- | --- |
| 1 | **Adopt the BoolQ protocol as a headline accuracy test**: dev set, passage as state, yes/no as `noul`, raw P(yes), no recalibration, passage-clustered bootstrap CIs, report accuracy, Brier, log loss, ECE (10 bins) next to the published hosted-Jev row (91.56%, ECE 2.51%) and the 35B first-token row (89.45%, 4.05%). Add a second row with a fitted T on a held-out split. | openjev-sglang BoolQ report | High (credibility, a real bar to beat) | S to M |
| 2 | **Ablation: letters vs full label vs trie-path** on the same model and prompts, using the same cases. Their claim (labels beat slots by 19 points on Qwen2.5-7B and 50 points on Qwen3-8B, n=45, wide CIs, prompt-noise of 10 points) is large enough to test, and testing it is how we earn the full-label headline. | jevmlx README leaderboard | High (validates or kills our headline) | M |
| 3 | **Tree-packed candidates**: build a token trie per question and run each trie node once in the packed pass, with a tree attention mask, so 255 options that share ` "` and common stems cost the unique tokens only. Because we sum full-sequence log-probs, each label's score is still exact; only the redundant forward positions disappear. | jevmlx trie construction (`trie.py:20-71`), cu-Jev's 255-candidate tests | High for big option sets (speed) | M |
| 4 | **Two-level prefix cache keyed by token ids**: cache (system plus state), then append the question block per request. Repeated state with new questions skips the state prefill. | cu-Jev cached 34 ms vs cold 91 ms (2.7x); parallel-decisions prefill 54.8 to 29.3 ms | High for repeated state, which is the Claude Code hook pattern | M (needs prompt order: state first, then questions, which SPEC 3.1 already has) |
| 5 | **Legal mass per question** (probability the model puts on the union of allowed first tokens against the full vocab, and optionally the full label set) as an output field and an abstain signal. We already compute the full-vocab `log_softmax` at candidate positions, so the cost is near zero. | jevmlx `legal_mass` | Medium to high for accuracy and trust | S |
| 6 | **bf16 CUDA parity gate** and a near-tie rescore: our 7.6e-5 drift figure is fp32 CPU. jevmlx and cu-Jev both found batch shape and bf16 change scores (jevmlx DRIFT 0.078 and 0.75 nats; cu-Jev notes cuBLAS picks different reductions per batch shape, `docs/ARCHITECTURE.md:76`). Measure ours on GPU in bf16 and decide whether a batch-1 rescore inside a small band is needed. | jevmlx parity gate, cu-Jev notes | High for honest accuracy claims | S to M |
| 7 | **Fixed-shape bucketing plus CUDA Graph replay** of the packed pass for small models. Measured gain on a 0.5B at 27 fields was 2.9x (240.9 to 83.8 ms), at 8 fields 1.16x. Capture can fail for lack of free VRAM, so fall back to eager and say so. | parallel-decisions doc 14, cu-Jev roadmap | High for 0.6B to 1.7B (our fast path) | L |
| 8 | **Calibration bundle with provenance**: store T with model revision, prompt version and scoring mode, and refuse a mismatch. Keep our "raise when T hits a bound" check (jevmlx returns the midpoint silently). | `calibrate.py:224-354` | Medium (prevents silent misuse) | S |
| 9 | **Isolated-question mode** as an A/B: one short branch per question that sees only its own question, versus our shared prompt that lists all questions. Cu-Jev does isolation; our prompt shows every question to every answer, which can bias answers. Test both on typed-decisions. | cu-Jev design (`README.md:24-27`) | Medium (accuracy), unknown sign | M |
| 10 | **Option-order flip test** as a built-in diagnostic: score a set with reversed option order and report the flip rate. Cheap position-bias metric. | jev-x-kit `jev-calibration.ts:35-44,72-78` | Medium | S |
| 11 | **Timing split in headers and the event log**: `Server-Timing` with prompt, prefill, eval; `prefix_cached`; non-overlapping spans (jevmlx `timing.py`). This feeds the race view's prefill-vs-eval bars. | sglang headers, cu-Jev headers, jevmlx ledger | Medium (console) | S |
| 12 | **Neutral-context prior correction** (subtract the score under an empty context) as an optional flag, with a measured effect or not at all. jevmlx ships it but publishes no gain. | `engine.py:2591` | Unknown | S to M |
| 13 | **Chat-message `state`** accepted and rendered by the template. | sglang README | Small, compatibility | S |
| 14 | **Second backend for models we cannot load**: an OpenAI-compatible or vLLM/SGLang logprob backend, with explicit truncation flag. Full-label scoring needs prompt logprobs for chosen tokens, which top-k endpoints often hide, so label scoring would degrade to first-token there. | jevmlx `openai_slots.py`, sglang `token_ids_logprob` | Medium (plug-in models) | L |
| 15 | **Race view that streams the same event rows from several engines** (ours plus any server speaking `/v1/systemone`, such as cu-Jev or sglang). Because three engines already speak that route, the console can race them without engine code. | cu-Jev, openjev-sglang, jevmlx all expose `/v1/systemone` | High (differentiator for the console) | M |

Do not copy: their prompts (nonce fences, codebook search, "classifier" system text), their code, names or notices. The facts above are enough.

## 7. Things worth stealing as tests, not code

- A DynamicCache shallow copy bug (shared layer objects mutated by `batch_repeat_interleave`): we avoid it by not copying, but a test that runs two packed passes over the same prefix references and compares to the first would catch any regression (jev-on-a-laptop doc 14).
- Adversarial sibling test: one question that "screams" a different answer must not change another's probabilities (cu-Jev branch isolation test). Our SPEC 3.3 injection test is the same spirit; add a cross-question-leak test for the packed mask.
- A prompt-sensitivity test: cu-Jev, jevmlx and sglang all found 2 to 10 point swings from small prompt edits. Pin our prompt with golden token ids and change it only with a measured reason.

## 8. Speed numbers: what they are and why they do not line up

| Source | Hardware | Model and precision | Workload | Number | Method |
| --- | --- | --- | --- | --- | --- |
| Original demo (founder) | M4 Max | Qwen2.5-1.5B-Instruct (4-bit in the lineage) | small schemas; 28 fields | 68 to 89 ms; 270 ms | UNVERIFIED here |
| jev-on-a-laptop | M5 Air 16 GB | Qwen2.5-1.5B 4-bit | 28 fields; 3-field cases | 414 ms; 147 ms | wall, warm |
| jevmlx | M5 Max 128 GB | Qwen2.5-7B 4-bit, `labels` | 19 to 30 fields | 244 to 586 ms p50 | per-item end to end |
| cu-Jev | RTX 5090 | Qwen3.5-4B bf16 | 1,024-token state, 16 x 40-token questions | 91 ms cold, 34 ms cached | prefill plus eval kernels, CUDA events |
| cu-Jev | RTX 5090 | Qwen3.5-0.8B bf16 | same | 31 ms cold, 10 ms cached | same |
| cu-Jev | RTX 3090 | Qwen3.5-4B / 0.8B | same | 223 / 66 ms cold | same |
| cu-Jev typed-decisions | 5090 / 3090 | 4B | 5 questions per case, wall | 53 / 129 ms per case | Python wall clock |
| openjev-sglang | B200 | Qwen3.6-35B-A3B NVFP4 | 1 question, MMLU-Pro, no reasoning | 0.11 s median, 0.21 s p95 | server side only, concurrency 16 |
| parallel-decisions | RTX 2060S | Qwen2.5-0.5B fp16 | 8 fields, 405-token prefix; 27 fields | 51.8 to 61.9 ms; 83.8 ms with graphs | warmed medians |
| hosted Jev (jev-x-kit measurement) | network | n/a | single question | 258 to 340 ms typical (860 first) | client wall clock |

Incomparable because: different devices, 4-bit vs bf16 vs NVFP4, different tokenizers and prompt lengths, different field counts, wall vs kernel timing, cold vs cached state, warm-up and capture excluded in some and not in others, and cu-Jev's Qwen3.5 is a hybrid linear-attention model that our engine defers. Nobody has run a 1.5B first-token engine on an NVIDIA card with the same schemas as the original demo. That is the number we are best placed to publish: same model, same schemas, first-token (letters) mode and full-label mode side by side, 4-bit MLX figures quoted only as context.

## 9. Corrections to earlier notes

- doc 09 row 1 and its statement "no decoder-based project read in code scores full label strings": CONTRADICTED by jevmlx `labels` mode. Reword to: "jevmlx scores real option strings through a trie as a constrained path probability on MLX; we sum full-sequence log-probs and normalise per question, in one packed pass, on torch." Run idea 2 before we say it is better.
- doc 09 laya-go row ("Laya encoder"): CONTRADICTED, see 1.2.
- doc 09 Windows row: add `parallel-decisions` (tested CUDA on Windows).
- doc 09 "no JSONL event stream": still true for these six. jevmlx's dashboard reads run files (`heartbeat.jsonl`, `predictions.jsonl`) but that is a bench monitor.

## 10. Threat summary for our claims

| Claim | Status after this read |
| --- | --- |
| Full-label scoring is a headline differentiator | Weakened: jevmlx exists. Still unique on torch and as exact sequence log-likelihood; must be backed by idea 2 |
| Windows/NVIDIA native | Weakened by parallel-decisions (tested, CUDA Graphs); unthreatened by cu-Jev (Linux only) |
| Race-view console | Intact. Nobody races engines; three engines already speak `/v1/systemone`, so racing them is cheap |
| Speed matches the original demo | Not yet testable; no one publishes the comparable number |
| Accuracy | BoolQ vs hosted Jev is the public bar (91.56%); cu-Jev's typed-decisions 4B 63.1% is the low letter-scoring bar; trained Laya sits at 76.6% on typed-decisions per cu-Jev's catalogue (their citation, UNVERIFIED by me) |
| Calibration | Open field: nobody has a published before and after temperature-scaling result |
