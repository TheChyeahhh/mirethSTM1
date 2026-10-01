# 12. Von (wfzyx/von)

Date checked: 2026-09-30. Read in a fresh clone at commit 0acaa06. Tags: VERIFIED (read in code or API output), UNVERIFIED (README or model card claim not reproduced), CONTRADICTED.

## 1. What it is

An encoder, not a decoder. A 395M ModernBERT-large backbone plus a small MLP head scores K option descriptions in one bidirectional pass. Not an LLM, no token generation. TypeSafe `/v1/systemone` compatible server, Python and TypeScript SDKs, CLI.

| Item | Value | Tag |
|---|---|---|
| License | Apache-2.0, full text in `LICENSE.md`; GitHub SPDX `Apache-2.0` | VERIFIED |
| Stars / forks | 795 / 59 | VERIFIED (gh api) |
| Created / last push | 2026-09-18 / 2026-09-30 | VERIFIED |
| Open source | Code yes. Weights on HF `wfzyx/von` (Apache-2.0 per card). Training builders in `training/`, eval in `benchmarks/`. The ~290k training corpus is assembled from public datasets (Banking77, ANLI, WANLI, dair-ai/emotion) plus synthetic sets; the exact corpus file is not in the repo | VERIFIED (code), UNVERIFIED (weights, not downloaded) |

## 2. Models, training, pluggability

- One model only: ModernBERT-large fine-tuned (listwise softmax cross-entropy plus Brier) with an option-marker head (`src/von/models/option_marker.py`, `OptionMarkerScorer`, hidden to hidden/2 to 1). VERIFIED. Card: 1.5 GB fp32; README install note says ~3 GB download.
- Trained, not stock. Not a LoRA and not a probe on a frozen model; the backbone is fine-tuned. Weights are tied to the packing format and an `independent_options` flag in `marker_calibration.json`.
- Plug in other models: NO in practice. `VON_MODEL_ID` env var is documented as a Hub repo or local checkpoint, but it must be an `OptionMarkerModel` checkpoint (encoder plus `option_marker.pt` head). The backend hardcodes `VON_MODEL_ID = "von-1.3.0"` as the label returned in responses (`option_marker_backend.py:70`). Swapping in a decoder LLM is not supported. VERIFIED.

## 3. Scoring mechanism

- Packing: `[question state] [SEP] [MASK] opt0 [MASK] opt1 ...`; the hidden state at each `[MASK]` goes through the MLP to a scalar logit, softmax over K (`option_marker.py`, `pack_sequence`, `forward`). VERIFIED.
- Single pass for all K options of one question. No option count cap found in code (limited by the 8192 token window). VERIFIED.
- Order invariance: option tokens attend only to the premise and their own span, and every option's position ids restart at the premise length (`build_independent_option_masks`, `build_option_invariant_position_ids`). Same shape as our packed mask in SPEC 3.4, applied inside an encoder. Card claim: 1.1 flipped 49.5 percent of answers under reordering, 1.2 flips 0. VERIFIED mechanism, UNVERIFIED number.
- Noul = two-option choice over the caller's criteria or default descriptions ("Yes, condition holds true." / "No, condition is false."); Score = choice over level descriptions with the expectation taken (`evaluate_noul`, `evaluate_score`). VERIFIED.
- Multiple questions: CONTRADICTED vs README. README says "Several questions over one state cost one forward pass". `evaluate` loops over questions, and each of `evaluate_choice/noul/score` packs and runs its own forward (`option_marker_backend.py`: evaluate loop near line 850, forward calls near 647, 706, 801). `usage.input_tokens` sums every pass, so the state is re-encoded per question. An encoder with bidirectional option-to-premise attention cannot share a prefix cache the way a decoder can.
- Labels are descriptions, not label strings. The option name is used only when the description is null (`desc.strip() if desc else opt.strip()`). Label text is read, but through a learned head, not as log-prob of the label's tokens.
- Digit splitting: spaces out every digit ("2026" to "2 0 2 6") because ModernBERT BPE merges digit runs inconsistently (`split_digits`). VERIFIED.
- Chain-of-options (`src/von/chains/`, 8 TOML chains in `chains/library/`): regex finds dates and amounts, a fixed operator library does the arithmetic, Von picks among spans. Up to 16 extra encoder passes, hard items only. VERIFIED code; accuracy gain UNVERIFIED (their own McNemar p = 0.065, which their gate labels UNRESOLVABLE).

## 4. Speed

| Number | Hardware | Method | Tag |
|---|---|---|---|
| p50 0.096 s, p95 0.110 s | AWS c7i.xlarge, 4 vCPU Xeon 8488C, OpenVINO CPU | Serial HTTP loopback, one question per request, model warmed, 72 JevBench standard items | VERIFIED (results/speed_remeasure.md, raw json present) |
| p50 0.023 s, p95 0.024 s | AWS g5.xlarge, A10G, CUDA | same | VERIFIED |
| Hard tier p50 0.339 s CPU / 0.039 s GPU; with chains 4.25 s CPU / 0.45 s GPU, p95 75 s CPU | same | 111 hard items, mean 1103 tokens | VERIFIED |
| ~18 ms per decision (Doom), 32.8 ms p50 on A10G (Decision Index) | A10G | model card | UNVERIFIED |

Incomparability notes:
- Standard items average 68 tokens (`results/speed/tokens_public.json`). Larger contexts cost more in an encoder, and every question is its own pass.
- Timings are loopback HTTP round trips, one question per request. A 28-field request would be 28 passes; they never measured it.
- No Apple M4 Max number, so not directly comparable to the original Qwen2.5-1.5B demo (68 to 89 ms small, 270 ms for 28 fields).
- The server serialises on a lock (`threading.Lock`, `option_marker_backend.py:370, 531`) and runs inference via `asyncio.to_thread` (`server.py` near line 106). One request at a time.
- The model card gives a board-measured median of 0.92 s versus 0.096 s self-measured. JevBench applies an assumed `raw x2 + 0.15 s` adjustment for self-hosted endpoints. Treat 0.096 s as best case.

## 5. Accuracy and calibration

- JevBench v1.4 (their README, Von 1.2): Intelligence 34.5, Calibration 75.7, composite 27.5, hard 0.373, sealed 0.279, sealed ECE 0.107. Jev 1.13 (closed) composite 63.3. hopper (Qwen3.5-4B + LoRA, 4B) composite 59.4, Intelligence 48.0 versus Von 34.5, at 0.41 s versus 0.34 s on their board. UNVERIFIED (not rerun), but it is the authors' own table.
- Noul problem: the v1.5 board scored Von 1.2 Noul as abstentions (sealed competence -92 to -100). The fix is the `band` rule, `p' = 0.8 + 0.1*(p-0.5)`, default on (`--noul-decision band`). This distorts the returned probability to fit a benchmark's abstention rule, so a default Von noul value is not a calibrated probability. VERIFIED (README, card).
- Calibration: input-conditioned temperature `T = bias + w_H*H_norm + w_len*log10(state_tokens)/4 + w_K*(K/8)`, clamped to [0.3, 12], in `marker_calibration.json`, fitted on the 231 public JevBench items. Card: "in-sample ... does not carry to other distributions". `von calibrate labels.jsonl` refits on user labels, picks scalar vs feature map by k-fold CV, reports NLL and ECE (`calibrate.py`: `fit_map`, `cross_validate`, `ece`). VERIFIED code. Their 38-label probe: ECE 0.21 to 0.11 (UNVERIFIED).
- Statistics: every accuracy claim goes through paired exact McNemar plus minimum detectable effect (`benchmarks/stat_gate.py`), with an UNRESOLVABLE label. VERIFIED file exists.
- Out of sample: Decision Index v0.2.1 13.74 (raw 34.11); jabr v2 72.0 percent macro. UNVERIFIED.
- Authors state: English only; `judge` without criteria is weakest; weak on multi-clause policy and multi-hop temporal or numeric items.

## 6. Platform and install

- Python 3.12 or newer, torch, transformers 5, FastAPI (`pyproject.toml`). Extra: `intel` (OpenVINO). VERIFIED.
- Devices: auto, cuda, mps, openvino:gpu, openvino:cpu, cpu (`device.py`, README). README says ROCm (UNVERIFIED); `device.py:94` mentions DirectML in a comment.
- Windows: not mentioned in the README; the Dockerfile is Linux amd64; CI is `ubuntu-latest` only (`.github/workflows/test.yml:13`). Pure Python, so it probably runs, but they do not test it. UNVERIFIED.
- Container: `ghcr.io/wfzyx/von:cpu` (OpenVINO, linux/amd64). No published CUDA image.
- Weights come from the Hub on first use.

## 7. API

- TypeSafe `/v1/systemone` compatible, with `VON_API_KEY` bearer auth and extras `truncation`, `X-Von-Truncated`, `Warning` header. VERIFIED (`server.py`).
- `usage.output_tokens = len(answers)`, not 0 (`evaluate`). Minor mismatch to TypeSafe.
- Confidence uses `(n*p_max - 1)/(n - 1)`, the same formula as ours (`_margin_confidence`, `option_marker_backend.py:36-48`).
- No MCP, UI, console or event stream (grep for mcp over `src`, README and `pyproject.toml` returned nothing). Only a Doom demo GIF (`benchmarks/doom_eval.py`). VERIFIED.
- Extras: `von calibrate`, chains, TypeScript SDK (`js/`), `--on-overflow refuse`.

## 8. Ideas to reimplement (shape only, no code)

Ranked by impact.

1. Input-conditioned temperature, `T = f(entropy, log length, K)`, fitted by k-fold CV with scalar-versus-map selection (high impact on calibration). Our SPEC has one scalar T per model. Effort: small to medium.
2. A `calibrate` command that refits on the user's own labels with the model frozen, reporting NLL and ECE before and after (high user value). Effort: medium.
3. Option-order invariance test and a published "flip rate under reordering" metric. We already use per-sequence position reset and a block mask (SPEC 3.4), so this is a test, not a feature. Effort: small.
4. Paired McNemar plus minimum detectable effect gate for every accuracy claim, with an UNRESOLVABLE label (credibility). Effort: small.
5. Middle truncation (60 percent head, 40 percent tail) with a `truncation` field and header, plus a `refuse` mode returning 422 (`option_marker_backend.py` near 455-472). Effort: small.
6. Speed protocol with tiers, token counts, raw and adjusted numbers both published. Effort: small.
7. Chain-of-options for date and amount arithmetic. Their gain is statistically unresolved and the hard tail costs 4 s to 75 s on CPU. Effort: large. Not now.
8. Digit splitting: encoder tokeniser fix; Qwen3 already tokenises digits singly, so likely no-op for us. Skip.

## 9. Threats to our claims

- Speed: Von is 96 ms on a 4-vCPU CPU and 23 ms on an A10G for small items. A Qwen3-4B prefill will not beat that on small schemas, so we must not claim to be faster than Von there. Where we may win is many questions: Von runs one encoder pass per question and re-encodes the state each time (28 fields means 28 passes, about 2.7 s at 96 ms by simple multiplication, not measured by them); we prefill once and score all labels in one packed pass. This follows from the code and needs our own measurement.
- Accuracy: their own board puts a LoRA-trained Qwen3.5-4B well above Von (48.0 versus 34.5 Intelligence). A stock Qwen3 may sit below such a trained model. Do not claim parity with trained models.
- Full-label scoring headline: Von does not score label tokens at all (learned head over descriptions), so it does not contradict the claim. It does undercut "no first-token collisions" as unique, since an encoder has no first-token problem.
- Windows/NVIDIA: Von has no Windows CI either, and does not claim it. Not a threat.
- Race-view console and MCP: Von has neither. Still open.
- Plug-in models: Von cannot take other models. Our plug-in story is a real differentiator against Von.
- Wire compatibility: Von also claims byte-compatible `/v1/systemone`. Not unique.
- Benchmark overlap: Von's calibration is fitted on JevBench public items, so its numbers measure that distribution. Our published numbers should use a held-out set we also publish.
