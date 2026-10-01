# 10. featherless-ai/simple-jev (deep read)

Date checked: 2026-09-30. Read at commit 146d3a6 (shallow clone, one commit of history, so no history to mine). Paths below are repo-relative to simple-jev.
Tags: VERIFIED (read in code or file), UNVERIFIED (not checked or secondary), CONTRADICTED (a claim we held is wrong).

## 1. What it is

- A Hugging Face Transformers + PyTorch server that turns any compatible open causal LM into a TypeSafe-style classifier endpoint. It reads next-token logits at an assistant prefill and builds the JSON itself; the model never generates. VERIFIED (`README.md` lines 9, 331).
- Single-file server `hf-server/hf_server.py` (1337 lines), a plain-Python shared contract in `common/` (prompt builder, request schema, response scoring), a trainer (`RFDT/`), an eval harness (`eval/`), a website and demos. VERIFIED (`README.md` lines 364-372).
- Run by Featherless, a hosted-model company. It has a public keyless demo API (2k context, 2 requests per second). The README says the demo serves `featherless-ai/gemma-4-26B-A4B-classifier`. UNVERIFIED that the demo is up (not called, per rules). (`README.md` lines 15-23).
- Honest self-description: "It does not reproduce TypeSafe's model architecture or training, or establish equivalent accuracy, calibration, or speed." VERIFIED (`README.md` line 316). Same stance as our SPEC section 1.

## 2. Facts

| Item | Value | Tag |
| --- | --- | --- |
| License | Apache-2.0, full text in `LICENSE` (11561 bytes); GitHub SPDX `Apache-2.0` | VERIFIED |
| Stars / forks | 575 / 66 | VERIFIED (gh api, 2026-09-30) |
| Last push | 2026-09-30T02:11:49Z (PR merge 13, "html-redirect-prefix") | VERIFIED |
| Created | 2026-09-18, about 12 days old, 3 open issues | VERIFIED |
| Open source scope | Code: yes. Weights: not bundled; stock HF models, plus Featherless-hosted `*-classifier` ids whose weights and training are not in the repo. Eval raw runs and large datasets: "not bundled" (`README.md` line 58) | VERIFIED |
| Python / deps | Python 3.12+, torch>=2.6, transformers>=5.16.1,<6, accelerate, fastapi, uvicorn, numpy, Pillow, torchvision (`hf-server/pyproject.toml` lines 9-11) | VERIFIED |
| CI | Only an eval-tooling workflow on ubuntu-latest, Python 3.10 and 3.12 (`.github/workflows/test-evaluations.yml`); the server tests are not in CI | VERIFIED |

## 3. Models

- Any HF causal LM with a chat template, "compatible cache operations" and single-token answer labels (`README.md` line 389). Loaded with `AutoModelForCausalLM` or `AutoModelForImageTextToText` (`hf-server/hf_server.py` lines 1214-1246). Plug-in models: yes, by `--model` (hub id or local dir). VERIFIED.
- Prompt format is auto-chosen by matching the backbone config fingerprint (hidden size, layers, heads, experts, vocab), not the model name; unknown sizes fall back to `baseline` with a warning (`README.md` lines 151-169, `hf-server/hf_prompt_policies.py` KNOWN_PROFILES). VERIFIED. Registered sizes are only 4B, 12B, 26B, 27B and 35B. Nothing at or below 2B is registered, so a 1.5B model runs untuned `baseline`.
- Models named in docs: Qwen3.5-0.8B (CPU example), Qwen3.5-4B, Qwen3.8-27B, Qwen3.6-35B-A3B, Gemma 4 12B and 26B-A4B, plus a Laya encoder backend. Qwen2.5-1.5B is not mentioned anywhere (grep). VERIFIED. (Model names are as the repo states them; I did not check they exist on the Hub.)
- Stock vs trained: stock models with prompt policies only ("Prompt policies, not newly trained models", `website/evaluations.html`). Optional RFDT fine-tune: LoRA or full weights, loss is soft cross-entropy over the allowed answer-token logits, no classifier head (`RFDT/README.md` line 5, `RFDT/train.py` lines 138, 322). VERIFIED. Nothing in the repo reports an RFDT-trained model result.
- Pluggable backends: `--backend laya` swaps in a trained encoder (`convaiinnovations/laya`) behind the same endpoint, with no KV cache and no vocabulary labels (`hf-server/hf_server.py` line 687). VERIFIED. So the project already shows "plug in different models" at two levels: decoder via HF, encoder via a backend class.

## 4. Scoring mechanism

- First-token scoring of single-token labels. Choice: letters `A`-`Z` then `a`-`x` (up to 50), then adapter-validated two-letter uppercase labels up to 255. Score: digits 0-9 (up to 10 levels), letters above (up to 50). Noul: digits 1-9 as a rating, mapped to a 0.01-0.99 value (`common/PROMPT_STRUCTURE_V1.md` lines 225-231, 252-254, 273; `common/response_scoring.py` lines 257-282). VERIFIED.
- Prefill is the unfinished assistant text `{"answer": "` (letters) or `{"answer": ` (digits), appended after the chat template with thinking disabled (`common/PROMPT_STRUCTURE_V1.md` lines 345-354; `hf-server/hf_server.py` lines 308-317 in the baseline path). VERIFIED.
- Explicit rejection of multi-token scoring: "never truncate options or silently substitute multi-token scoring" (`common/PROMPT_STRUCTURE_V1.md` lines 229-231), and the compiler raises if a label is not single-token stable at the real answer boundary by re-encoding `text + label` and comparing ids (`hf-server/hf_server.py` lines 357-368). VERIFIED. This is a stronger boundary check than a plain `encode(label)`. It does not threaten full-label scoring; it confirms nobody here does it.
- Softmax over the allowed labels only. No temperature. Confidence = max probability. `calibrated: False` on every answer; response metadata says `"calibration": "not_calibrated"` (`common/response_scoring.py` lines 167, 291). VERIFIED.
- Score returns the expected zero-based level index plus variance (lines 311-326). Choice also returns margin and exact-tie list (lines 294-309). Diagnostics stay internal unless `ENABLE_OPEN_JEV_ADVANCED_METRICS=1` and `options.raw_logits` (`README.md` line 310). VERIFIED.
- Number of options: choice 2-255, score 2-50, questions 1-256 (`common/PROMPT_STRUCTURE_V1.md` lines 83-95). Matches the TypeSafe limits we already target.

### How it batches (the speed design)

All in `hf-server/hf_server.py` `HFBackend._score` (line 477):
1. One prompt (one "branch") per question, each holding the whole prompt through the answer prefix. Compute the longest common token prefix of all branches (`common_prefix`, line 111; used at line 492), minus one so every row keeps a suffix token. VERIFIED.
2. One unchunked prefill of the prefix to a cache (lines 497-513).
3. Suffix rows (the question text and answer prefix) are batched with right padding, longest first, capped by `--max-batch-size` and `--max-batch-tokens` (lines 521-535). Per batch it does `copy.deepcopy(cache)` then `reorder_cache(zeros)` to broadcast the single prefix row to the batch (lines 536-542). VERIFIED.
4. `logits_to_keep` is used when the model supports it, so only the needed positions get logits (lines 460, 567). VERIFIED.
5. Only the allowed token ids are gathered to CPU float32 (the `selected[row, branch.output_ids]` line, 585). A single `threading.Lock` serializes all model work; requests also pass a semaphore (lines 456, 870). Cancellation is cooperative between batches. VERIFIED.
- The cache lives for one request only: "Cache reuse currently lasts only for a single request" (`README.md` line 325). No cross-request prefix cache, no vLLM, no tree mask. VERIFIED.
- Comparison with our SPEC section 3.4: they copy and broadcast the prefix cache once per padded batch of questions, and score one label position per question. We pack all labels of all questions in one pass with a 4D tree mask and no cache copy, but must score every label's tokens. The costs are different: they pay padding plus a cache deep copy per batch; we pay extra suffix tokens per label.
- Prompt policies change how much can be shared: the repeat and example policies put question or context text before the shared part, so "breaks mixed-question context sharing" and the auto-selector excludes them unless the shared variant is chosen (`hf-server/hf_prompt_policies.py`, AUTO_TUNE_POLICIES line 16 and the warning in `resolve_prompt_policy`). VERIFIED. This is an explicit accuracy-vs-speed trade they manage in code.

## 5. Speed

- Published latency numbers: none for the engine. VERIFIED by grep of all `.md` and `.html` for ms, latency, throughput, tokens per second. The README states token-count arithmetic only ("These token counts illustrate avoided repeated input processing, not a measured latency ratio", `README.md` line 360). Other docs repeat "no throughput guarantee" (`hf-server/README.md` line 181, `hf-server/VISION.md` lines 190, 227).
- Only anecdotes, all through a network demo API: emotion demo "about 1.5-2 seconds per frame on Gemma" (`website/cool-demo/emotion/README.md` line 14), pictionary "Qwen answers about twice as fast" as Gemma (`website/cool-demo/pictionary/README.md` line 12). Incomparable to our local ms: includes network, image input, rate-limit pauses of 300-550 ms.
- Diagnostics exist: `backend_seconds`, `queue_seconds`, `total_seconds`, prefix tokens, forwards, padded tokens in the metrics (`hf-server/API_REFERENCE.md` lines 299-301; `hf_server.py` metrics dict at the end of `_score`). VERIFIED. These are exposed only behind the advanced-metrics flag.
- Takeaway: there is no published simple-jev speed number to beat or match. Our 68-270 ms comparison target is the original Twitter demo, not this project.

## 6. Accuracy and calibration

- Accuracy: measured, against hosted Jev, on public data. Data and method in `website/assets/evaluations/results.json` and `website/evaluations.html`. Source of the Jev side: "Published Jev per-task outcomes" from a third-party JevBench repo (MIT). VERIFIED.
- JevBench public, 231 questions (correct counts): Qwen3.8-27B 214, Gemma4-26B-A4B 207, Qwen3.6-35B-A3B 204, Jev 1.13 200 (86.6%), Gemma4-12B 199, Qwen3.5-4B 178 (77.1%). On the "hard" tier (111 questions): 27B 95, Gemma MoE 88, Qwen MoE 85, Jev 81, Gemma12B 81, Qwen3.5-4B 62. VERIFIED (read from the JSON).
- Broader "decision" mean over 26 items (21,364 rows): Qwen3.8-27B 0.899, Gemma MoE 0.888, Qwen MoE 0.886, Jev 0.871, Gemma12B 0.834, Qwen3.5-4B 0.787. VERIFIED. A 4B stock model with a tuned prompt is about 8 points below hosted Jev on decisions and 9 on JevBench. Models of 1.5B or smaller are not evaluated.
- Caveats the authors state: prompt selection used a 477-case development set that overlaps the evaluation, so it is "not a wholly held-out test"; no significance, latency or throughput ranking is implied (`website/evaluations.html`, methodology block). The Jev numbers in `eval/benchmarks/jev-1.13/2026-09-20/` are runs of the hosted Jev endpoint through their harness, not of simple-jev. VERIFIED.
- Calibration: none. No temperature, no ECE, no Brier, no reliability data anywhere. "These distributions, and the Noul value, are not calibrated probabilities of correctness" (`README.md` line 273). Accuracy only. VERIFIED. The Laya backend does bring its own temperature scaling from the SDK (`hf-server/hf_server.py` class docstring at line 687).
- Position bias / order flip: not measured. Letters are fixed in source order. UNVERIFIED that anything permutes options. (grep of prompt builder showed no rotation.)

## 7. Platform and install

- Linux-first in practice. Windows is never mentioned in the docs, code or CI (grep for windows, win32, powershell, .ps1 found nothing outside vendored demos). The CPU example is `--device cpu --dtype float32` on Qwen3.5-0.8B. CUDA or ROCm: "install the appropriate PyTorch build for your hardware" (`README.md` line 102). VERIFIED.
- Pure PyTorch and Transformers, so it plausibly runs on Windows with a CUDA torch wheel. UNVERIFIED (not run). It needs transformers 5.16 or newer and Python 3.12 or newer, which is close to our own pin (transformers 5.18, Python 3.11 in SPEC section 11); a Python 3.11 env would not install it as written.
- Tests: pytest suite on tiny locally-initialized models, no downloads, not wired into CI. The README says they do not measure accuracy (`README.md` lines 378-387). UNVERIFIED that they pass on Windows.
- Single-process, serial model lock. No GPU warm-pool story beyond keeping the server up.

## 8. API

- TypeSafe-shaped, not a clone. `POST /v1/classifier` with `/v1/systemone` as a hidden alias (`hf-server/hf_server.py` lines 1075-1076); `GET /v1/models`; `/health`; `/docs` (`README.md` lines 108-114). VERIFIED.
- Request: `model`, exactly one of `state` or `messages`, `questions` with `type` choice, score or noul, `instructions`, `criteria`. Same field names as TypeSafe. Response: `model`, `answers`, `usage`, with `choice`/`score`/`noul`, `confidence`, `probabilities`, `legend` (`README.md` lines 242-265). VERIFIED. Differences to check against our file 04: confidence is max probability and noul is a single float (not a probability object); extra fields `calibrated`, `margin`, `ties` only in advanced mode. Unknown top-level fields ignored; unknown fields inside questions rejected (`README.md` line 308).
- Over-limit requests get 422, never truncation (`README.md` line 144).
- Image inputs for vision models (Qwen-VL, Gemma 3/4, LLaVA family).
- MCP: none in this repo. UI: a website with playground and demos (Doom, 2048, drive, pictionary, emotion), hosted. No race view and no live event stream. VERIFIED by file listing.
- Eval client: `eval/run.py` runs any TypeSafe-compatible endpoint with resumable HTTP calls, raw-response audit and matched-metric comparisons (`README.md` lines 45-59). This could drive MirethSTM1's HTTP server unchanged. UNVERIFIED (not run).

## 9. Prompting ideas they found (data, not code)

From `common/PROMPT_STRUCTURE_V1.md` and `hf-server/hf_prompt_policies.py`:
- The question is stated twice in the suffix, with a "think slowly" line between, plus a list of all questions in the system message before the context. So a causal model has seen the question before the state. VERIFIED (`PROMPT_STRUCTURE_V1.md` lines 163-221).
- `repeat_state` and `strict_mix_repeat2`: repeat the context or input a second time, labelled as the same text, so tokens of the second copy can attend to the whole first copy. Chosen by dev-set search for the 4B, 12B, 26B and 35B models. VERIFIED that these are shipped defaults; the gain size per policy is in their summary files not bundled, so UNVERIFIED how much each one adds.
- `examples_binary`: worked examples in the system prompt, and Noul as a binary no/yes choice instead of nine ratings (`restore_binary_noul`, `hf_prompt_policies.py` line 182). Chosen for the 27B model.
- Noul as nine rating bins, mean mapped to 0.01-0.99, rather than P(yes) of two tokens (`response_scoring.py` lines 257-282). The eval page says Noul is scored as P(yes) in the public benchmark and a plain probability, so it varies by policy.

## 10. Ideas to reimplement in MirethSTM1

Founder rule: take the fact and the shape, never the expression. Ranked by impact on speed and accuracy.

1. Single-letter fast path as a mode inside our engine (impact: speed, high; effort: small to medium). Fact: simple-jev scores one token per label, so each question needs one suffix position, not a multi-token candidate. We already plan a letters mode for the head-to-head table (file 09). Keep it as `scoring="letters"` next to the default full-label mode, and report both. Their lack of any latency data means our table will be the first to compare the two designs.
2. Prefix-boundary check at the real answer position (impact: correctness, high; effort: small). Shape: compare `encode(text + label)` against `encode(text)` plus one id, and raise if the label tokenizes across the boundary. Our SPEC 3.3 test asserts joined equality for the full-label case; extend the same test to the letters mode and to every model the picker can load.
3. Question repeated before answering, and the whole-question list placed before the state (impact: accuracy, medium; effort: small). Cheap prompt variants to benchmark against our section 3.1 prompt on the same data. Do not copy their wording; write our own. Note the second copy of the question costs prefix-shared tokens only if placed before the per-question part, so test the speed cost.
4. Context repeat policy for small models (impact: accuracy for weak models; effort: small). Measure on Qwen2.5-1.5B and Qwen3-4B: repeating the state doubles prefill, which hurts the "match 68 to 270 ms" goal, so make it opt-in and show the latency price next to the accuracy gain.
5. Backbone fingerprint to choose a prompt policy (impact: usability when users plug in models; effort: medium). Shape: pick settings from config fields (hidden size, layers, heads, vocab), never from the repo name, and fall back to a safe default with a loud warning. Useful for the model picker: store per-model best settings (prompt variant, temperature) keyed by that fingerprint.
6. Metrics block on every response (impact: console, medium; effort: small). Shape: report prefix tokens, forwards, padded and computed token counts and backend seconds per request. We already have the event log; add the same counters so the race view can show why a run was fast.
7. Backend interface so encoders can plug in (impact: "plug in different models", high; effort: medium). Shape: one `Backend` boundary with `score(compiled) -> label scores`, and a separate encoder backend (they do this for Laya). Our file 09 lists von and Verdict as encoders; a thin adapter to a local encoder would let the console race decoder against encoder, which matches the founder's wish.
8. Training on answer-token logits (impact: accuracy, high for tiny models; effort: large). Shape: soft cross-entropy over the allowed label scores only, LoRA, same prompt as inference. Out of scope for v0.1 per SPEC section 12, but it is the path from "stock Qwen" to something closer to a trained Jev. Note: with full-label scoring the loss would sum label-token log-probs, which is what we already compute.
9. Eval harness against any endpoint (impact: credibility; effort: small). Run `eval/run.py` from this repo against our HTTP server as a third-party check, and publish the result next to ours. Uses their client as a tool, not as a copied file.
10. Request-limit table: 255 choices, 50 score levels, 256 questions, 422 on overflow, no silent truncation. Same as TypeSafe; keep.

## 11. Threats to our claims

- Full-label scoring as headline: NOT threatened. This project explicitly forbids multi-token scoring and checks stability at the boundary. But it removes any "nobody tests the boundary" angle: they do it well. Our edge remains scoring the real label text; we must measure order-flip and length bias to justify it.
- Speed: no threat from published numbers (none exist). A plausible threat is practical: their path is one forward per batch of questions with `logits_to_keep`, which will often beat a multi-token packed pass on raw ms when labels are short. Our letters mode must be as fast as theirs or we should say it is not.
- Accuracy: they publish real numbers on 231 public questions and a 21k-row decision set, with hosted Jev as the reference, and a stock 4B model scores 178/231. That is the bar a reader will compare us against. We have nothing equivalent until file 07's benchmarks run. We should run the same JevBench public set (the source is public and MIT licensed according to the page) so a reader can compare directly.
- Calibration: parity or better for us. They publish none and say so. Our temperature scaling plus ECE is a real difference over this project, but it is not a first overall (file 09).
- Windows and NVIDIA: not threatened, no Windows claim here, but also unproven that it fails. Our edge only exists if we document and test it. Their Python 3.12 floor may itself block users on 3.11.
- Race-view console: no race view, no event stream, no console. Their demos are separate game-style web pages calling the hosted API. Not threatened. A tap into their demo UI is not available.
- Plug in any model: already shown here (any HF causal LM, plus a Laya backend). Not a differentiator; the console picker is.
- Distribution: 575 stars, a company behind it and a public keyless demo. They will be the default reference people try first. Link to them and compare on the same dataset.

## 12. Open items

- UNVERIFIED: runs on Windows, passes its own tests there.
- UNVERIFIED: public demo availability and latency (no calls made).
- UNVERIFIED: per-policy accuracy gain (summary files are not in the repo).
- Not read in depth: `hf-server/hf_vision.py`, `eval/` adapters, the web demos, `RFDT` data prep.
