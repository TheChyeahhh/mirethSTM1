# 00. Research index and review

Date checked: 2026-09-30. Reviewer pass over docs 01 to 09. Status tags: VERIFIED (re-read from the primary source by the reviewer), UNVERIFIED, CONTRADICTED (brief says X, reality is Y).

## Documents

| Doc | Summary |
|---|---|
| [01-openjev.md](01-openjev.md) | razorback16/openjev (549 stars, Apache-2.0) implements the TypeSafe wire shape but with a 26B diffusion model and single-token letter labels, not Qwen prefill scoring.<br>No calibration or ECE code; no Windows. Useful as a schema and error-contract reference only. |
| [02-rlcd-and-ports.md](02-rlcd-and-ports.md) | The "RLCD" repos are inference code on stock Qwen2.5-1.5B: first-token scoring, plain softmax, no training, no calibration. Only the shreyansh26 port has full-label scoring, and only for colliding fields.<br>Gives the suffix format, what to avoid (fabricated probabilities, hard-coded flags) and a Qwen3 collision-test correction. |
| [03-kev.md](03-kev.md) | jaredpalmer/kev (about 8k stars) is a trained LoRA plus pointer-head model, a different mechanism, but its metric code (ECE, NLL, Brier, temperature fit, bootstrap) is a good protocol reference.<br>The "ECE 0.065" is a raw pre-calibration number from older cards. |
| [04-typesafe-jev-api.md](04-typesafe-jev-api.md) | Exact request and response fields for `POST /v1/systemone`, from live docs and two real calls. Score is an ordered level array, noul is a bare float, confidence formula undisclosed.<br>Includes a drop-in compatibility checklist. |
| [05-environment.md](05-environment.md) | torch 2.11+cu128 has sm_120 and cp311 Windows wheels, but it is frozen (cu128 ends at 2.11; current torch is 2.14.1 on cu130). The transformers 5.18 cache recipe for prefill once, chunked suffix scoring is verified equal to uncached scoring (about 1e-5). |
| [06-models-and-licenses.md](06-models-and-licenses.md) | All four Qwen3 models are Apache-2.0; Qwen2.5-3B is the non-commercial Qwen Research License. Architecture table, chat-template thinking handling, and a VRAM table showing 64 candidates cannot be repeated at once on 12 GB, so chunking is mandatory. |
| [07-benchmark-design.md](07-benchmark-design.md) | Working HF dataset IDs, splits, metric definitions (15-bin ECE, summed Brier), baseline and latency protocol, time budget, Yelp bin metrics.<br>`PolyAI/banking77` no longer loads; use `mteb/banking77`. |
| [08-console-integration.md](08-console-integration.md) | Tarnlight (the live console, formerly jevscope) does not tail arbitrary JSONL; it ingests TypeSafe-shaped envelopes from a drop-box folder.<br>Recommends the frozen per-field JSONL plus an optional drop-box sink, and defines `id` as a per-call id. |
| [09-landscape-and-differentiators.md](09-landscape-and-differentiators.md) | Only differentiator 1 (full-label scoring) holds up among decoder engines. Score bins and calibration are parity; the no-key local MCP claim is contradicted; Windows is weak unless tested. Suggests an honest README framing. |

## Verified facts the spec relies on

Each row was re-checked by the reviewer from the primary source, not from the docs.

| Fact | Evidence | Status |
|---|---|---|
| Endpoint is `POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer`, JSON body | Live docs https://docs.typesafe.ai/api.md; the working client used on the target machine posts to `${BASE_URL}/v1/systemone` with `{model, state, questions}` | VERIFIED |
| Request: `model` (use `jev-latest`), `state` (string, object or array), `questions` map. Question fields: `type`, `instructions`, `criteria` | api.md | VERIFIED |
| noul `criteria` is optional `{true, false}`; choice `criteria` is `{option: description or null}`, max 255 options; score `criteria` is an ordered array, "at least two levels; the API accepts up to 10" | api.md | VERIFIED |
| Response: `model`, `answers`, `usage{input_tokens, output_tokens}`. noul answer `{type, noul}` (no confidence); choice `{type, choice, probabilities, confidence}`; score `{type, score, legend{"i":text}, probabilities{"i":p}, confidence}` with string index keys and `score` the probability-weighted level index | api.md; the client's parsing code reads exactly `a.noul`, `a.choice/confidence/probabilities`, `a.score/confidence` | VERIFIED |
| Errors documented: 401, 422, 429, 529. Error body shape is not documented | api.md | VERIFIED |
| `jev-latest` and `jev-preview` both resolve to `jev-1.13.0`; limits 64k context, 32k for state plus longest question | https://docs.typesafe.ai/models.md | VERIFIED |
| Qwen3-0.6B, 1.7B, 4B and 4B-Instruct-2507 are all Apache-2.0 (HF API license field and LICENSE file opening lines read). Revisions: 0.6B `c1899de2`, 1.7B `70d244cc`, 4B `1cfa9a72`, 4B-Instruct-2507 `cdbee75f` | https://huggingface.co/api/models/<id>, https://huggingface.co/<id>/raw/main/LICENSE | VERIFIED |
| cu128 index has `torch-2.11.0+cu128-cp311-cp311-win_amd64.whl` (and 2.10.0); no 2.12 or later on cu128 | https://download.pytorch.org/whl/cu128/torch/ listing | VERIFIED |
| The installed wheel (torch 2.11.0+cu128, CUDA 12.8) reports `get_arch_list()` = sm_75, sm_80, sm_86, sm_90, sm_100, sm_120 | ran in the project venv with no GPU visible | VERIFIED (compiled in; a real kernel launch on the RTX 5070 is still untested) |
| Current torch is 2.14.1 on PyPI, and cu130 has cp311 Windows wheels for 2.14.0 and 2.14.1 | https://pypi.org/pypi/torch/json, cu130 index listing | VERIFIED |
| transformers 5.18.0 cache API as used in doc 05: `DynamicCache(config=...)`, `DynamicCache(ddp_cache_data=...)`, `cache.layers[i].keys/values`, `batch_repeat_interleave`, `batch_select_indices`, `crop`, forward grows layers by `torch.cat` | installed source `cache_utils.py` (lines 114, 146, 175, 200, 206, 1764, 1809, 1823 match the doc) | VERIFIED |
| Cached, right-padded, chunked suffix scoring equals uncached full-sequence scoring on Qwen3-0.6B, CPU, fp32: max abs log-prob diff 1.3e-05 on the reviewer's own prompt (doc 05 measured 9.8e-06) | re-ran a variant of the doc 05 snippet with transformers 5.18.0 | VERIFIED |
| On the Qwen3 tokenizer, ` Sports` is 1 token; ` Sci-Tech` and ` Sci/Tech` both start with token ` Sci`. So Sports vs Sci-Tech is not a first-token collision; Sci-Tech vs Sci/Tech is | ran the Qwen3-0.6B tokenizer | VERIFIED |
| Dataset IDs that have data files on the Hub: `fancyzhx/ag_news`, `mteb/banking77` (mit), `stanfordnlp/sst2` (has a validation parquet), `Yelp/yelp_review_full` (license `other`). `PolyAI/banking77` holds only a loading script (`banking77.py`), which datasets 5 refuses | HF dataset API file listings | VERIFIED (listings; the loading tests are doc 07's) |
| Stars on 2026-09-30: openjev 549, kev 8060 (doc 03 said 8059, moved by one), simple-jev 575; Harsha's HF repo 577 likes, shreyansh26 port 4 likes | `gh api`, HF API | VERIFIED |
| Tarnlight ingests records from `~/.tarnlight/inbox/<source>-<YYYY-MM-DDTHH>.jsonl`, appended one JSON object per line, not created by the writer | `src/tarnlight/inbox.py` docstring and constants in the public repo | VERIFIED |

## Contradictions found and how they were resolved

| # | Contradiction | Resolution |
|---|---|---|
| 1 | `confidence` formula. Doc 01: openjev uses `1 - H/ln K`. Doc 03: Kev uses the margin `(p_max - 1/K)/(1 - 1/K)`. Doc 04: live TypeSafe fits `(3*p_max-1)/2` at K=3. Doc 08: use the top probability | Computed on the live sample (0.68/0.28/0.04): entropy form gives 0.32, margin form 0.52, TypeSafe returned 0.52. The margin form matches for choice; entropy form does not. The live docs example (0.88/0.12/0 gives 0.81) also fits the margin form within rounding. Score confidence formula is still unknown (Kev's differs). Resolution for the SPEC: compat route `confidence` uses the margin form, and says so; the JSONL `p` is always the top-label probability after T; the Tarnlight sink labels which one it wrote (edit to doc 08, edit to doc 01) |
| 2 | Score level limits. Doc 01 (openjev) 1 to 10; doc 03 (Kev) 1 to 255; doc 04 TypeSafe docs 2 to 10, live call accepted 1 | Re-read live docs: "at least two ... up to 10". SPEC: accept 1 to 10 on the compat route (lenient like TypeSafe), and document that the native route may allow more (edit to doc 04) |
| 3 | ECE bin count. Doc 07 headline 15 bins (Guo et al.). Doc 03 recommends 10 bins to match Kev | Not a factual conflict, a choice. Resolution: headline 15 bins, and always also output the 10-bin value and the bin table. Never mix them in one comparison |
| 4 | "Kev ECE 0.065". Doc 07 could not find it; doc 03 found it | Doc 03 is right: raw pre-calibration ECE on older checkpoints. Doc 07 updated to point at doc 03 |
| 5 | Banking77 source. Brief and doc 03 list `legacy-datasets/banking77`; doc 07 only tested `PolyAI/banking77` (broken) and `mteb/banking77` | `legacy-datasets/banking77` also has parquet files (cc-by-4.0). Added to doc 07 as an alternative. Default stays `mteb/banking77` (labels already flat, loads today) |
| 6 | Doc 02 flagged Qwen3 tokenizer splits as UNVERIFIED | Verified, see above; doc 02 updated |
| 7 | Doc 05 says mem-efficient SDPA "should be what runs" with a mask, while the PR text it cites says mem-efficient is not eligible when GQA is requested natively | Not a contradiction in fact: transformers passes `enable_gqa` only with no mask, so with a mask K/V are repeated and mem-efficient is eligible. Still unproven at runtime (see Still unverified) |
| 8 | Doc 09 says no project ships a JSONL event stream; doc 08 shows Tarnlight consumes drop-box JSONL, and TypeSafe-shaped records are what openjev-style tools emit | Not contradictory: Tarnlight's format is an envelope, not a documented engine event log. Our frozen per-field JSONL remains the differentiator, with the optional drop-box sink for interop |

No license conflicts between docs: doc 05 and doc 06 agree on all four Qwen3 licenses, and both agree with the reviewer's re-check.

## Still unverified

- Any real CUDA forward on the RTX 5070 with sm_120 kernels, and which SDPA backend runs with a mask on Windows (mem-efficient, cuDNN, math). Needs the GPU session.
- bf16 and fp16 cached-vs-uncached log-prob difference on GPU (expected about 1e-2, from dtype).
- Exact driver floor for the cu130 torch wheel (doc 05: 580+ is from NVIDIA minor-version compatibility notes, not torch-specific).
- Torch 2.14.1 cu130 arch list includes sm_120 (not installed here).
- Whether the official `typesafe-sdk` accepts a base-URL override and works unchanged against a compat server (claimed by openjev and Kev, run by neither of us).
- TypeSafe's error body shape, max questions per call, exact score confidence formula, and its latency claims.
- Reproduction of any third-party latency or accuracy number (openjev, RLCD repos, Kev, simple-jev).
- Qwen3.5 hybrid cache behaviour (`batch_repeat_interleave` on linear-state layers) and Windows support for its fast kernels.
- The 55-project long tail on the madewithjev directory; absence claims about JSONL event streams and Windows testing are UNVERIFIED for it.
- All per-document time estimates in doc 07 section 5 (estimates, to be replaced with measurements).
- Windows long-path setting and whether Developer Mode is on, on the target machine.
- X/Twitter posts from the brief were not fetched by any doc, so claims about what the origin post said are unchecked.

## Brief claims now known wrong

| Brief says | Reality | Doc |
|---|---|---|
| `score`: min/max, handled via bins (as wire API) | TypeSafe wire `score` is an ordered array of 2 to 10 level descriptions; answer is the expected 0-based level index with string-keyed `legend` and `probabilities`. Min/max must be an optional native extension that expands to bins, not the wire format | 01, 03, 04 |
| razorback16/openjev is a close relative ("vLLM/NVIDIA/Linux + MLX") | Default model is DiffusionGemma 26B-A4B on a patched vLLM; single-token letter labels; needs about 18 GB of weights; no calibration | 01 |
| Kev "reported ECE about 0.065" | 0.065 is the raw, pre-temperature number on two older checkpoints; after scaling 0.031 (prototype), current Kev-4B 0.013 in-distribution | 03 |
| jevscope "tails a JSONL event log" | The console (Tarnlight) ingests TypeSafe-shaped envelopes from a drop-box folder, UDP or an HTTP proxy; it does not tail arbitrary JSONL | 08 |
| Existing MCPs need a paid API key | At least four local keyless MCP servers exist (jev-style, laya-go, jev-x-kit, a Kev-backed one). Only some clients need a key | 09 |
| Directory has about 330 Jev projects | 749 entries total; 330 is only the GitHub type; open-source guide counts 55 alternatives | 09 |
| Blackwell support reference is pytorch issue 164342 | The issue is an open request thread and not evidence either way; the cu128 wheel itself contains sm_120 | 05 |
| Torch cu128 is the current stable path | cu128 is frozen at torch 2.11.0; current is 2.14.1 on cu130. cu128 still works for the one-week ship | 05 |
| Test collision example "Sports" vs "Sci-Tech" | Not a first-token collision on Qwen2.5 or Qwen3 (` Sports` vs ` Sci`). Use "Sci-Tech" vs "Sci/Tech", which share ` Sci`; keep "Sports" vs "Sci-Tech" as the length-bias test | 02 |
| Harsha's repo is "enum/bool only; first-token only" | Correct. But note there is no LICENSE file in it (license is in the README and card only) | 02 |
| Qwen2.5-3B is non-commercial | Correct (Qwen Research License) | 06 |
| shreyansh26 port is a CUDA port | True, but it selects the CUDA wheel only on Linux x86_64 as shipped, and its tree mode needs FlexAttention and Triton | 02 |
| Driver 570+ | Correct for CUDA 12.8 GA (570.65); 572.61+ is safer for later 12.8.x | 05 |
| Differentiators 2, 3, 4, 5 are differentiators | Bins and calibration are parity with existing projects, Windows is weak unless tested, local MCP is parity. Only full-label scoring (1) is clearly open, and the rigor of the benchmark is the edge for 3 | 09 |
| "Tarnlight" vs "jevscope" | The code and repo call it Tarnlight; "jevscope" appears nowhere in it | 08 |

## Brief claims no doc addressed

- Driver 570+ on the Ryzen/RTX box beyond a version read: doc 05 read driver 591.86 from `nvidia-smi`, which was not re-read by the reviewer.
- "Commit and push to a branch" and commit rules: out of research scope.
- vLLM prefix caching docs and the SGLang Score API PR 38965 from the reference links: no doc read them. Not needed for the scaffold.
- Claims in the Twitter/X links (origin post, critiques): unfetchable, so no doc checked what they assert.
- The AlphaSignal, Turing Post RLCD explainer and DataCamp pages: doc 04 lists them as secondary; none was read in depth.

## Edits the reviewer made to other docs

1. 01-openjev.md: added the confidence cross-check (entropy form does not match TypeSafe's live value) and adjusted the "adopt as ideas" bullet.
2. 02-rlcd-and-ports.md: replaced the Qwen3 tokenizer UNVERIFIED note with the verified result; updated open question 3.
3. 04-typesafe-jev-api.md: added the live-docs re-read of the score-level limit and the openjev and Kev differences.
4. 07-benchmark-design.md: added `legacy-datasets/banking77`; replaced the "UNVERIFIED Kev ECE" note with the doc 03 finding.
5. 08-console-integration.md: clarified that the top-probability confidence applies only to the Tarnlight sink.
