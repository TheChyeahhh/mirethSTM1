# 01. razorback16/openjev

Research date: 2026-09-30. Repo: https://github.com/razorback16/openjev, read at commit `dcd20947b5ddad5be4a8f5aed6aa6dd245653823` (2026-09-29, 54 commits). All file:line references below are to that commit. "VERIFIED" = read in code or API output by the researcher.

## Headline findings

1. CONTRADICTED (brief): openjev is not a Qwen/first-token scorer. Its default model is **DiffusionGemma 26B-A4B (NVFP4)** run through a patched vLLM "diffusion read" (`openjev/engine.py:1-12`, `openjev/config.py:37,129`). It never does autoregressive prefill plus per-label suffix scoring. It seeds a 64-token answer "canvas", runs one read-only denoise step, and takes per-slot logprobs. Our prefill-once/suffix-scoring design shares the goal, not the mechanism.
2. CONTRADICTED (brief): the brief says `score` is "min/max via bins". On the real wire API, `score.criteria` is an **ordered list of 1 to 10 level descriptions**, and the answer is the expected 0-indexed level (`api.py:56-60`, `engine.py:134-142,449-451`). There is no min/max field.
3. VERIFIED: stars 549, last push 2026-09-29T16:23:32Z, license Apache-2.0 (`gh api`, date checked 2026-09-30). The brief said "~531, unverified": close.
4. VERIFIED: it claims wire compatibility with TypeSafe and mostly delivers it (section g).
5. VERIFIED: no ECE, reliability diagram or calibration code for the main model. "Calibrated" in the README is a marketing word for a raw label softmax. The only calibration temperatures it touches belong to third-party encoder backends (JevK5 T=1.532, Verdict per-option-count temperatures).
6. VERIFIED: no Windows support is stated. The README has zero mention of Windows; targets are Docker/Linux+NVIDIA (vLLM) and macOS arm64 (MLX).

## (a) Exact HTTP schema

Source: `openjev/api.py`. Docstring (lines 1-7) says shapes "follow TypeSafe's published OpenAPI 0.2.0".

### Endpoints

| Path | What | Source |
|---|---|---|
| `POST /v1/systemone` | the decision endpoint | `api.py:244` |
| `GET /v1/models` | `{"models": [{name, description, release_date}]}` | `api.py:240-242` |
| `GET /health` | `{"status":"ok"}` (not under `/v1`, no auth) | `api.py:236-238` |
| `POST /v1/chat/completions` | OpenAI-style generation (out of scope for us) | `api.py:277-278` |

### Request (`SystemOneRequest`, `api.py:71-79`)

| Field | Type | Notes |
|---|---|---|
| `state` | str, dict or list | required. Non-string is `json.dumps`-ed into the prompt (`engine.py:373`) |
| `model` | str | required. Must be in the served set or 400 |
| `questions` | dict[str, Question] | required, min 1, max `OPENJEV_MAX_QUESTIONS` (default 256) (`config.py:55`) |
| `images` | list (data URI or `{content_type, base64}`) | optional extension, `api.py:75` |
| `steps` (1-8), `samples` (1-32), `think` (0-4096), `sequential` | optional extensions | `api.py:76-79`; the TypeSafe SDKs never send them |

### Question types (`api.py:39-63`, discriminated union on `type`)

| type | Definition fields | Constraints |
|---|---|---|
| `noul` | `instructions` (str/dict/list/None), `criteria: {true, false}` optional, each a description | none |
| `choice` | `instructions`, `criteria: {name: description}` (required) | 1 option is answered without a model read (`engine.py:126-129`); max options = number of single-token letter labels found for the tokenizer, capped at 255 (`engine.py:30-107,130-131`) |
| `score` | `instructions`, `criteria: [level0, level1, ...]`, `min_length=1` | 1 to 10 levels (`engine.py:136-137`); 1 level answered directly (`engine.py:138-141`) |

### Response (`api.py:274-275`, `engine.py:441-451`)

```
{ "model": "openjev-0.1",
  "answers": { "<qid>": <answer> },
  "usage": { "input_tokens": int, "output_tokens": int } }
```

| type | Answer fields | Source |
|---|---|---|
| noul | `{"type":"noul","noul": P(yes)}` (single float, no probabilities map, no confidence) | `engine.py:443-444` |
| choice | `{"type":"choice","choice": name,"probabilities":{name:p},"confidence":float}` | `engine.py:445-448` |
| score | `{"type":"score","score": sum(i*p_i),"legend":{"0":level0,...},"probabilities":{"0":p,...},"confidence":float}` | `engine.py:449-451` |

`confidence = 1 - H(p)/ln(K)`, clamped to [0,1] (`engine.py:434-438`). `output_tokens` is 0 unless `think` is set (`api.py:273`). Answers are returned in request order (`engine.py:393`).

### Headers

`x-typesafe-request-id` and `x-request-id` (`req_<32 hex>`) on every response, plus `server-timing: model;dur=..,server;dur=..,total;dur=..` (`api.py:214-234`).

### Errors (`api.py:116-117,149-154,194-212,326-339`)

| Case | Status | Body |
|---|---|---|
| Unknown model | 400 | `{"detail":{"error_type":"api_usage_error","message":"Unknown model: X"}}` |
| Unknown question `type` | 400 | same shape, message "Invalid request." |
| Wrong field shape | 422 | `{"detail":[{type,loc,msg,input,...}]}` (FastAPI list, input trimmed) |
| Semantic problem (no options, too many options/levels, too many questions) | 400 | `{"detail":"<plain string>"}` |
| Missing key (when `OPENJEV_API_KEY` set) | 403 | `detail.error_type=authentication_error` |
| Wrong key | 401 | `detail.error_type=authentication_error` |
| Body too large | 413 | `api_usage_error` |
| Overloaded | 529 | `overloaded_error` with `retry-after: 1` |
| Backend down | 503 | `api_error` with `retry-after: 2` |

Auth is `Authorization: Bearer <key>`; auth is optional (off if no env key).

## (b) How it scores

### Labels are single tokens by construction, not full labels

- For `choice`, the option names are **never shown to the model as labels**. Each option is mapped to a one-token letter (`A`..`Z`, `a`..`z`, then `AA`..`ZZ` codes that happen to be one token), chosen so the letter stays exactly one token after the prefix `"q1: "` (`engine.py:94-107`). The model is shown lines like `  A: outage (service down)` (`engine.py:153-168`). The answer is mapped back by position (`engine.py:446-448`).
- `noul` labels are `yes`/`no` (`engine.py:119-121`). `score` labels are the digit strings `"0".."9"` (`engine.py:142-143`), hence the 10-level cap.
- For each question, `resolve_template` **asserts that every label changes exactly one token at the same position** and raises `SchemaError("labels do not share one template slot")` otherwise (`engine.py:174-208`, check at line 199).
- So first-token scoring is exact here because it is the only token. The collision "Sports" vs "Sci-Tech" cannot occur: both names become letters such as `A`/`B`. The cost is that the model must read a lettered legend and answer with a letter, a known source of position/letter bias. The repo test `test_labels_are_single_tokens` (`tests/test_api.py:66`) covers the single-token invariant, not semantic collisions.
- Consequence for us: multi-token full-label log-prob scoring is **not** something openjev does. Our differentiator 1 stands and is a real difference.

### Probability extraction

- vLLM is asked for the exact logprobs of every label id at every slot via `logprob_token_ids` plus `top_logprobs: 20` (`engine.py:297-300`, `MAX_LABEL_IDS=512` at line 34). Labels missing from the top-k get `floor = min(top) - 5.0` (`engine.py:425`). Then a plain softmax over the label logprobs at temperature 1 (`engine.py:421-431`; comment: "Read-only logprobs are at temperature 1, so the label softmax uses them directly").
- Reads are averaged when `samples` is set. By default there is one read, plus up to 3 automatic re-reads if the top-20 entropy exceeds 0.1 (`engine.py:344-355`, `config.py:64-65`). Noise draws are seeded from a hash of the request, so answers are reproducible (`api.py:262-263`).

### Score questions

- Not bins and not numeric tokens. A score is a choice over up to 10 digit labels; output is the expected level `sum(i*p_i)` plus the full distribution (`engine.py:449-451`). This is the closest existing match to our differentiator 2 ("expected value plus distribution"), but the levels are user-written legend strings, not a numeric min/max range.

### Many questions

Questions are split into groups that fit the 64-token canvas (`config.py:50`, `engine.py:210-226`); groups are read in parallel (`engine.py:380-384`). Above 10 questions the answer format switches from `lines` to `indexed` (`engine.py:151`). `sequential` mode feeds earlier chosen labels into the prompt for later groups (`engine.py:396-418`).

## (c) Calibration and published numbers

| Item | Finding | Status |
|---|---|---|
| Temperature scaling for the main DiffusionGemma path | None. T=1 softmax (`engine.py:421-424`) | VERIFIED |
| ECE / Brier / reliability code | None in the repo. A search for "ece" and "calibrat" hits only the README and the encoder backends | VERIFIED |
| Published accuracy | Only for borrowed backends: JevK5 scored 86.6% on JevBench's 231 public items, identical to JevK5's own run; CLM FP8 vs bf16 accuracy 89.7% to 88.8% on 581 SQuAD questions (README, sections JevK5 and CLM) | read in README, not reproduced |
| Published latency (DiffusionGemma, RTX PRO 6000) | 1 question p50 27 ms, 3 questions p50 31 ms (one request at a time); 94 ms p50 for 3-question cache-busted requests at concurrency 1 (README "Measured on an RTX PRO 6000") | read in README, not reproduced; different hardware and model from ours |
| Third-party temperatures | JevK5 uses T=1.532 from its checkpoint config; Verdict uses per-option-count temperatures (`config.py:84-87`, README) | VERIFIED |

Implication: our "published ECE + reliability diagrams" differentiator (3) is unoccupied by openjev. The closest calibration code in this family is in the Mac port jevmlx (section h).

## (d) Backends, prefix caching, Windows

| Topic | Finding | Source |
|---|---|---|
| vLLM | Main backend. Needs a **pinned vLLM commit `1b3b88ec...`** with diffusion support (vLLM PR #57250) and a `sed` patch raising `MAX_LOGPROB_TOKEN_IDS` 128 to 512. Serve flags include `--enable-prefix-caching --attention-backend TRITON_ATTN` | README "Without Docker"; `engine.py:3-4` |
| MLX | In-process on Apple silicon, DiffusionGemma 26B 4-bit. Own LRU prefill cache of 12 entries / 16384 tokens | `mlx_backend.py:28-49,116-173`; `config.py:48-49` |
| Encoder backends | `laya`, `verdict`, `clm`, `jevk5` selected via `OPENJEV_BACKEND`; routed or served separately | `config.py:137-158`, `api.py:159-175` |
| Prefill-once | Via vLLM automatic prefix caching (server-side) and the MLX prefill cache. The client does not manage KV tensors. Re-reads share the prompt prefix | `mlx_backend.py:28`, README |
| Caveat | Qwen3.5 linear-attention: vLLM caches in 528-token blocks, so states shorter than that share no prefill | README, JevK5 section |
| Windows | Not mentioned anywhere. GPU path is Linux Docker with the patched vLLM | search of README and `docker/` found nothing |
| Hardware | Reported on RTX PRO 6000 (Blackwell workstation) and RTX 3090; model weights about 18 GB, too big for a 12 GB card | README |

Implication: we cannot reuse any of its inference code for a Transformers+PyTorch Windows stack. The DiffusionGemma path needs a forked vLLM that has no Windows build.

## (e) License and reuse

LICENSE file read: full Apache License 2.0 text with the unfilled appendix boilerplate (`Copyright [yyyy] [name of copyright owner]`); `pyproject.toml` and GitHub both say Apache-2.0. The README states (lines 538-542) that `openjev/encoders.py` adapts Verdict's prompt format and calibration (Verdict is Apache-2.0). `engine.py` is "adapted from vLLM's examples/features/diffusion_reads/structured_server.py (PR #57250, Apache-2.0)" (`engine.py:3-4`). README line 18: "independent project, not affiliated with or endorsed by TypeSafe AI". The pyproject `Homepage` is codiv.ai, a hosted service by the author.

| Reuse | Verdict |
|---|---|
| The wire schema (field names, answer shapes, error shapes) | Safe. Schema shapes are facts about TypeSafe's public API; we may implement the same. Cite openjev as the reference that checked them against the live API |
| Test ideas (single-option forced answer, single-level score, choice needs an option, deeply nested body, body cap, unknown model shape) from `tests/test_api.py:361-397` | Safe as ideas; write our own tests. If any test code is copied, keep the Apache-2.0 notice and add a NOTICE entry |
| The `confidence = 1 - H/ln K` formula (`engine.py:434-438`) | Standard normalized entropy; reimplement freely. Credit is courteous, not required. Cross-check (reviewer, 2026-09-30): it does NOT reproduce TypeSafe's live value. For the live choice answer in doc 04 (probabilities 0.68/0.28/0.04) it gives 0.32, while TypeSafe returned 0.52, which matches the margin form `(K*p_max-1)/(K-1)` that Kev uses (doc 03). So "same wire shapes" holds for field names, not for `confidence` values |
| Error/validation patterns (trim echoed input, cap body, 413/529/503 with retry-after) | Ideas only. Reimplement |
| Verbatim code (engine, api) | Allowed by Apache-2.0 with LICENSE/NOTICE retention and changed-file notices, but avoid: it is DiffusionGemma/vLLM-diffusion specific, and the founder's rule is reimplement, never copy |
| Letter-label mapping for choices | Avoid as our core method (it is what differentiator 1 replaces). Could be an optional "letter mode" baseline arm in the benchmark |
| Docker/vLLM patches (`docker/patches/vision_prefix_lm.py`), image support | Avoid, out of scope |
| Branding: "OpenJev", "codiv.ai", `openjev-*` model names | Avoid. We use `mirethstm` |

## (f) Stars and activity (2026-09-30)

- razorback16/openjev: 549 stars, pushed 2026-09-29T16:23:32Z, Apache-2.0 (`gh api repos/razorback16/openjev`). The brief's "~531" was close. VERIFIED.
- Very young (first model release date 2026-09-18 per `config.py:128`) and active (54 commits, PRs merged 2026-09-29, including external contributors).

## (g) Claim of wire compatibility with TypeSafe

VERIFIED from code and README:

| Aspect | openjev | Source |
|---|---|---|
| Endpoint path | Same: `POST /v1/systemone` | `api.py:244` |
| Base URL override | TypeSafe SDK works by setting `TYPESAFE_BASE_URL` | `api.py:4-5`, README lines 37-39 |
| Field names | Same as the TypeSafe contract (`state`, `model`, `questions`, `type`, `instructions`, `criteria`; answers `noul`, `choice`, `probabilities`, `confidence`, `score`, `legend`) | `api.py`, `engine.py:441-451` |
| Schema source | "TypeSafe's published OpenAPI 0.2.0" | `api.py:3` |
| Verification | README: "Errors use the same shapes as Jev, checked against the live API"; `tests/test_api.py:102` `test_quickstart_decodes_with_typesafe_sdk` runs the `typesafe-sdk` client (>=0.7) against it | README; tests |
| Model names | Its own: `openjev-latest`, `openjev-0.1`. Accepts `jev-latest` and `jev-preview` as aliases so SDK defaults work; pinned names like `jev-1.13.0` return 400 "Unknown model" | `config.py:120-125,161-166`; README "Differences from Jev" |
| Headers | Copies `x-typesafe-request-id` | `api.py:212,221,233` |

Stated differences: chunks of about 12 questions per read; 10-level score cap; choice max 255 (the brief's "up to 255" is correct).

Caveat: we did not call TypeSafe's live API here, so "matches TypeSafe" is UNVERIFIED on our side. It is openjev's claim. Confirming it is the job of the TypeSafe-API research file.

## (h) Mac ports, short comparison

| | razorback16/openjev | bnsd55/openjev (repo content is "jevmlx") | rorshopping/jev-on-a-laptop |
|---|---|---|---|
| Stars / last push (2026-09-30) | 549 / 2026-09-29 | 68 / 2026-09-25 (last commit 2026-09-24) | 24 / 2026-09-17 |
| License (file read) | Apache-2.0 | MIT (copyright Ben Shaharizad and jevmlx contributors, portions rorshopping) | MIT (file read; GitHub API shows NOASSERTION because it cannot match the file) |
| Model | DiffusionGemma 26B, plus encoder models | Qwen2.5-7B-4bit default, smaller alias; any MLX model | Stock Qwen2.5-1.5B to 8B on MLX, using harshatheg's engine (cloned, not vendored) |
| Wire API | TypeSafe-compatible `/v1/systemone` | Its own `POST /decide` with `enum`/`boolean`/`multi` schema; NOT TypeSafe-compatible | None (research repo: scripts, benchmarks, a demo Space) |
| Scoring | single-token letter labels, diffusion read | Trie scoring over full option strings with legal-mass tracking; `lint.py` flags option collisions | Raw logit softmax; README admits fields sharing a first token hit a slow fallback |
| Calibration | none (T=1) | `calibrate.py`: `fit_temperature` by NLL, ECE before/after, pooled logistic for multi-select | README: "raw softmax, a proxy, not trained calibration" |
| Windows | no | no (Apple Silicon only) | points Windows/NVIDIA users to the author's `parallel-decisions` library |

The jevmlx facts come from its ARCHITECTURE.md and README; its code was not read in depth (UNVERIFIED at code level). Worth a later look: `jevmlx/calibrate.py`, `trie.py`, `lint.py` as an MIT-licensed reference for full-label trie scoring and temperature fitting. Reimplement, do not copy.

## What this means for MirethSTM1

For the SPEC:

1. **Keep a drop-in path `POST /v1/systemone`** in addition to our own `POST /v1/decide`. Same request (`state`, `model`, `questions`) and same answer shapes (`noul`: float; `choice`: `choice`, `probabilities`, `confidence`; `score`: `score`, `legend`, `probabilities`, `confidence`; `usage`). This gives TypeSafe SDK compatibility at little cost. Accept `jev-latest` and `jev-preview` as model aliases, and return our own model name in `model`.
2. **Fix the brief's score definition.** On the wire, `score.criteria` is an ordered list of level descriptions, the value is `sum(i*p_i)`, and openjev caps at 10 levels. Support the list form for compatibility. The brief's "min/max via bins" can be an optional extension (for example `{"min":1,"max":5}` expanding to integer bins labelled by value) but must not replace the list form. With full-label scoring there is no single-digit limit, so a 10-level cap is needed only in compat mode. Report the expected value plus `probabilities`.
3. **Adopt the error contract.** 400 `api_usage_error` for unknown model/type, 422 FastAPI-style list for shape errors, 400 plain-string detail for semantic errors, 401/403 `authentication_error`, 413, 503/529 with `retry-after`. Add `x-request-id` and `server-timing`.
4. **Differentiator 1 is real.** openjev avoids first-token collisions by replacing labels with single-token letters and asserting single-slot templates. It does not score multi-token labels. Add a benchmark arm that mimics the letter-label method on the same Qwen model to show the difference honestly, and keep the "Sports" vs "Sci-Tech" collision test.
5. **Differentiator 3 is open ground.** openjev publishes no ECE. Our temperature scaling, ECE and reliability diagrams would be the first in this family on a local NVIDIA stack (jevmlx has tooling on Mac only).
6. **Adopt as ideas:** forced answers for 1-option choices and 1-level scores (no model read, probability 1.0, confidence 1.0); a normalized confidence (but use the margin form from doc 03, not openjev's entropy form, if matching TypeSafe numbers matters); a cap on questions per request; splitting big schemas into groups if the batch would not fit memory.
7. **Skip the auto re-read policy.** openjev re-reads when entropy is high because its diffusion reads are noisy. Our scoring is deterministic, so it has no analogue.
8. **Do not chase its performance numbers.** 27-31 ms was measured on an RTX PRO 6000 with a 26B-A4B diffusion model and a custom vLLM. Our Qwen3-4B on a 12 GB RTX 5070 with HF Transformers will differ. Report our own p50/p95.
9. **Attribution.** Add a README credit line such as "Wire format and test ideas informed by razorback16/openjev (Apache-2.0)". If any test file is adapted verbatim, add its NOTICE. Do not use the names "OpenJev" or "codiv".
10. **For the scaffold:** `decide()` should return per question: `type`, label probabilities dict, `confidence` (normalized entropy), and for score the expected value plus legend. JSONL event fields `probs{}` and `p` map onto `probabilities` and the top probability (or the `noul` value).

## Open questions

- Does TypeSafe's live API still use exactly these shapes (especially `noul` being a bare float)? Needs the TypeSafe-API research file; openjev's claim is that it checked against the live API.
- Should `/v1/systemone` be labelled "compat mode" in our docs to keep the trademark line (Jev-style in the README only)? The path name and `jev-latest` alias are functional compatibility, not branding. Founder decision.

## Sources

- https://github.com/razorback16/openjev (commit dcd2094; files `openjev/api.py`, `openjev/engine.py`, `openjev/config.py`, `openjev/mlx_backend.py`, `tests/test_api.py`, `README.md`, `LICENSE`, `pyproject.toml`)
- https://github.com/bnsd55/openjev (ARCHITECTURE.md, README.md, LICENSE)
- https://github.com/rorshopping/jev-on-a-laptop (README.md, LICENSE)
- GitHub API `gh api repos/OWNER/REPO` for stars, push date, license (2026-09-30)
- https://typesafe.ai/blog/introducing-system-one-models-and-jev (linked by the openjev README; not independently read in this file)
