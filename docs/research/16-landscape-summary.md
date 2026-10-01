# 16. Landscape summary: what the open projects do and what we should do about it

Synthesis of notes 09 to 15. Tags: VERIFIED (read in code or a result file), UNVERIFIED (self-published or not rerun), CONTRADICTED (an earlier note or a README was wrong). Nothing here was re-run; no GPU code, no paid calls.

## Spot checks by the synthesizer

1. openjev-sglang BoolQ numbers: its result report lists accuracy 89.45% [88.38, 90.59] for the open engine against 91.56% [90.56, 92.50] for hosted Jev (evals/results/boolq-2026-09-18/comparison/report.md, line 7). VERIFIED.
2. cu-Jev speed: README table lines 180 to 181 give 91 ms (prefill plus eval) and 34 ms (state already cached) for the 4B row on the RTX 5090. VERIFIED as published; Python wall time, not reproduced.
3. Two structural claims. (a) simple-jev rejects any label that is not single-token stable by re-encoding text plus label at the real boundary (hf_server.py, about lines 357 to 368), so it cannot do full-label scoring. (b) von evaluates questions in a loop, one encoder pass per question (option_marker_backend.py, about lines 838 to 860), which CONTRADICTS its README line about one pass for several questions. Both VERIFIED.

## 1. Comparison table

Speed numbers are NOT comparable with each other (hardware, quantization, workload and timing method all differ). Read section 2 before quoting any.

| Project | License | Stars | Models | Scoring | Speed claim (hardware) | Accuracy claim | Windows | /v1/systemone | UI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original demo (lineage via the jev-on-a-laptop audit) | MIT (audit repo) | 24 | Stock Qwen2.5-1.5B-Instruct 4-bit | First token per option, collision fallback | 68 to 89 ms small, 270 ms at 28 fields (M4 Max, UNVERIFIED); 0.41 s at 28 fields on an M5 Air | None; 58.3% on 24 synthetic cases | No (Mac) | No | No |
| SemIf (was OpenJev) | MIT | 4620 | Stock Qwen3.5-4B, others by flag | Letters A to P, 2 to 16 options, shared prefix, per-workload temperature | 21 criteria 1.02 s vs 5.33 s generated JSON (3090, BF16); JevBench raw p50 198 ms | JevBench rank 13, hard acc 0.595, hard ECE 0.121 | Not stated | No | Replay, JSONL CLI |
| simple-jev (featherless) | Apache-2.0 | 575 | Any HF causal LM, plus Laya backend | Single-token labels up to 255, shared prefix | None published | JevBench public 231: Qwen3.8-27B 214, 4B 178, hosted Jev 200 | Not stated | Yes (/v1/classifier, alias) | No |
| AnyJev (Nokia) | Apache-2.0 | 990 | Qwen3 1.7B to 32B; 11 models for L0/L1 | Single-token, 26 options max; rotations, temperature, trained head | 4.2 ms batched, 42 ms single (1.7B, H100, eager) | Flip 0.230 to 0.073 (8B banking20); ECE 0.240 to 0.095 | Not stated | No (library) | Gradio Space |
| von | Apache-2.0 | 795 | One ModernBERT-large 395M, trained | Encoder, option markers, one pass per question | p50 96 ms (4-vCPU CPU), 23 ms (A10G), small items | Own board: 34.5 intelligence vs 48.0 for a trained 4B | Untested | Yes | No |
| open-alternative-jev (so1) | Apache-2.0 | 56 | Any chat LM, stock | Single letters, packed or separate, permutation averaging | 4B 105 ms per 5-question case (H200 slice); vLLM separate fastest | Stock 1.7B 45.9% ECE 0.509; 4B 59.3%; 27B 73.7% (typed-decisions) | Not stated | No | No |
| jev-style | Apache-2.0 code | 9 | Trained Qwen3.5 0.8B/2B, fixed | Verdict slot: yes minus no logit after each option text | 194 ms example, 0.8B (M1 Max) | JevBench 2B 73.6, 0.8B 64.1 (own harness) | Claimed, untested | Yes, plus MCP | Playground |
| Verdict-open-jev | NOASSERTION (rewritten Apache text) | 109 | 151M encoder, Banking77 specialist | Encoder label slots, max 24 | p50 35.6 ms at K=5, 140 ms at K=25 (arm64 ONNX CPU) | 95% own set; 48% on TypeSafe public | Untested | No | Browser demo |
| cu-Jev | Apache-2.0 | 4 | Qwen3.5 0.8B to 9B | Single-token, isolated branches, prefix cache | 91 ms cold, 34 ms cached (4B bf16, RTX 5090) | 4B 63.1%, ECE 0.104 | No (Linux, sm_80+) | No | Game demo |
| laya-go | MIT | 5 | No model (hand-weighted rules) | Linear rules | Config budgets, not measured | None | Yes | No | No |
| jev-x-kit | MIT | 2 | No engine | Asks an LLM to write a probability | Hosted calls 258 to 860 ms | None | Yes | Client | MCP plugin |
| openjev-sglang | None (no license file) | 335 | Qwen3.6-35B-A3B NVFP4 | First token, one request per question | Median 0.11 s server side (B200) | BoolQ 89.45% vs hosted 91.56%; ECE 4.05% vs 2.51% | Server Linux only | Own | No |
| jevmlx (bnsd55/openjev) | MIT | 68 | Stock Qwen2.5 4-bit, any mlx-lm | Real multi-token labels via trie, or alias slots | 7B 244 ms (24 cases), 586 ms (44 cases), M5 Max | Labels 82.1% vs slots 63.2% (n=45) | No (Apple) | No (/decide) | SSE dashboard |
| JevBench | MIT | 190 | Benchmark, 91 ranked systems | Four-axis score | n/a | Raw Qwen3-4B direct: score 41.0, hard ECE 0.452 | Python, any OS | Needs it | Board |
| llama.cpp PR 29752 / vLLM PR 59299 | MIT / Apache-2.0 | n/a | Any GGUF / any decoder | Trie exact probabilities / letters | About 100 ms at 3 fields, 12B (RTX 3060) | None | llama.cpp yes | Own / yes | Playground |
| Laya and laya.cpp | Apache-2.0 / MIT | 29298 | Trained encoders 322M to 421M | Packed options, per-bucket temperatures | 4 ms single (4070 Ti SUPER), 342 q/s (Blackwell) | typed-decisions 0.766 self-reported, no result file | Yes (CUDA, Vulkan builds) | Yes, plus MCP | No |

Star counts look inflated in places; treat as a hype signal only.

## 2. Targets to beat

Our setting: RTX 5070 (12 GB), Windows, torch 2.11 cu128, plain SDPA, Qwen2.5-1.5B-Instruct first, Qwen3-4B-Instruct-2507 default.

| Setting | Best published number | Source | Comparable to a 5070 run? |
| --- | --- | --- | --- |
| Original demo, 1.5B, small schema | 68 to 89 ms | Demo, M4 Max (UNVERIFIED) | Low: Mac, 4-bit MLX, no accuracy. A 5070 should beat it if our pass is efficient. |
| Original demo, 28 fields | 270 ms (M4 Max); 0.41 s on an M5 Air | Demo, jev-on-a-laptop | Low, same caveats. We score more tokens per field with full labels. |
| Windows RTX, tiny model | 0.5B fp16: 240.9 to 83.8 ms at 27 fields with CUDA Graphs; 88.5 to 61.9 ms at 8 fields with prefix reuse | parallel-decisions, RTX 2060 SUPER | Medium. Same OS and vendor, older card, smaller model. Closest like-for-like. |
| 4B, NVIDIA consumer | 91 ms cold, 34 ms cached state | cu-Jev, RTX 5090 | Medium. Qwen3.5 hybrid, custom kernels, Linux. A 5070 has far less bandwidth, so expect worse. |
| 4B, raw first-token | 82 ms p50 (A6000), 198 ms (RTX PRO 4500) | JevBench rows | Medium. Serial HTTP, includes server overhead. |
| Single-request floor | About 50 ms from eager launch overhead, 1.7B to 8B | AnyJev (H100) | Directly relevant: small models are launch bound, so CUDA graphs matter more than FLOPs. |
| Speed ceiling (trained encoders) | 4 ms single, about 1 ms batched | Laya | Different regime (encoder, small head budget). We cannot claim fastest. |
| Accuracy, hosted Jev | BoolQ 91.56%, ECE 2.51%; JevBench hard 0.741 | openjev-sglang, JevBench | Reproducible: public sets we can run. |
| Accuracy, stock 4B | JevBench public 178/231 (simple-jev); hard acc 0.518 raw, 0.595 with temperature (SemIf); typed-decisions 59.3 to 63.1% (so1, cu-Jev) | Various | High: same model class, public items. We must appear in this table. |
| Accuracy, stock 1.5B to 1.7B | 45.9% typed-decisions, ECE 0.509 (so1 1.7B); 58.3% on 24 synthetic cases (demo audit) | so1, jev-on-a-laptop | Matching the demo model is a speed target only; accuracy is poor everywhere. |
| Calibration | Hard ECE 0.08 to 0.13 for stock readouts with fitted temperature; raw 0.452 | JevBench, SemIf | High. |
| Reverse-order flip | 0.230 raw to 0.073 (L0), 8B banking20; SmolLM2-1.7B raw 0.89 | AnyJev | Publish ours next to it. |

Rule for every number we publish: record GPU, model, field count, batch, warmup, and whether timing includes tokenization and HTTP. State raw p50 and any adjustment.

## 3. Ideas worth reimplementing (ranked)

Idea and shape only; we write our own code.

| # | Idea | Source | Why | Effort |
| --- | --- | --- | --- | --- |
| 1 | Readout ablation on one model and one case set: our raw summed full-label, length-normalized full-label, letters, trie path, verdict slot | jevmlx, Jobe, jev-style, simple-jev | Decides our default scorer. Jobe measured letters 0.801 vs normalized text 0.792 on stock 4B; our raw sum (SPEC 3.4) is the weakest variant. | M |
| 2 | Run JevBench public 231 and the BoolQ protocol (dev set, noul, raw P(yes), clustered bootstrap) against our server | JevBench, openjev-sglang | Rank-comparable numbers against 90 systems and hosted Jev. Needs /v1/systemone earlier than SPEC plans. | S to M |
| 3 | Temperatures per (question type, option count) with a clamp, fitted out of fold; show raw and fitted ECE, Brier | Laya, SemIf, von, jev-style | Calibration is a quarter of the JevBench score; raw ECE 0.452 vs 0.08 to 0.13 fitted | S |
| 4 | CUDA graph replay with fixed-shape buckets and eager fallback | parallel-decisions, AnyJev | Small models are launch bound (about 50 ms floor); 2.9x at 27 fields on 0.5B. Unverified on our stack | L |
| 5 | Letters fast path as a selectable mode, with the real answer-boundary stability check | simple-jev | Its one forward with logits_to_keep may beat our packed pass on short labels; match it or say so | S to M |
| 6 | Label prefix tree in the packed pass (one position per unique trie node, tree mask) | jevmlx, llama.cpp PR | Big win for 255 options; keeps exact log-likelihood | M |
| 7 | Cross-request two-level prefix cache (state cached, questions appended) | cu-Jev, parallel-decisions | 2.7x in their numbers; prefill 54.8 to 29.3 ms | M |
| 8 | Reverse-listing flip rate; option-order averaging (off by default) | AnyJev, so1, SemIf | Tests our core claim; averaging halves 4B ECE per so1 README (UNVERIFIED) | S |
| 9 | legal_mass per question and an abstain signal | jevmlx, vLLM PR | Nearly free from the log-softmax we compute | S |
| 10 | Per-request timing block (prefill ms, eval ms, prefix tokens, forwards) as headers and event fields | simple-jev, cu-Jev, openjev-sglang | Feeds the race view and Tarnlight | S |
| 11 | Parity gate in CI: packed vs unpacked, cached vs uncached, repeated-prefix cache integrity | jevmlx, parallel-decisions, jev-style | Protects the zero-interference claim; a DynamicCache in-place mutation bug was hit elsewhere | S |
| 12 | Measurement protocol: medians with spread, never-seen states, JSON receipts with hardware and versions | AnyJev, Verdict, von | Credibility | S |
| 13 | Prompt layout switch (schema-first so the cache keeps the fixed block) | AnyJev | Possible 5 to 10x fewer tokens on short states; accuracy varied -0.23 to +0.39 | M |
| 14 | Certified label-free agreement for a cheaper model vs the 4B default | AnyJev | Makes plug-in models safe to offer | M |
| 15 | Backend interface so an encoder or other engine races a decoder in the console | simple-jev | Plug-in models and race view | M |
| 16 | Paired bootstrap or exact McNemar table for the race view | von, jev-style | Honest accuracy claims | S |
| 17 | Hard budget errors with token counts, no silent truncation | jev-style, von | Safety | S |
| Later | Opt-in labelled hidden-state head; LoRA on label scores | AnyJev, simple-jev | Large gain (+0.12 to +0.24 reported) but per question and per model; not zero-label | L |

## 4. What this means for our claims

Survive:
- Raw summed log-prob over the real label text on stock models, in one packed pass with no prefix copies, is a defensible method. Honest wording: "stock, plug-in models; the real label text is scored."
- A live side-by-side of two engines' raw JSON logs with a feed to Tarnlight: nobody ships that. SemIf has a replay, decision-playground a live decision vs chat race, jevmlx an SSE dashboard of its own runs.
- Calibration is a fair strength only if we publish raw and fitted ECE with the bin rule; others (AnyJev, SemIf, Laya) publish too, so we are at parity, not first.

Must be reworded:
- "No decoder engine scores full labels" is CONTRADICTED: jevmlx (labels mode, trie), llama.cpp PR 29752 (open, 2026-09-30), jev-style (verdict slot reads real option text), us/jev-local and Jobe all score option text.
- Full-label as an accuracy headline is unsupported: Jobe measured letters 0.801 vs length-normalized text 0.792 on stock 4B; jevmlx saw labels beat slots by 19 and 50 points on 7B and Qwen3-8B but with about 10 points of prompt noise at n=45. Run our own ablation (idea 1) before claiming either way.
- "Windows and NVIDIA is a gap" is wrong: laya.cpp ships Windows CUDA and Vulkan builds, parallel-decisions is tested on an RTX 2060 SUPER, and one project measured on Windows 10 with an RTX 5060 Ti. Claim "tested and documented on Windows with an RTX 5070", not "only".
- The JSONL event stream is not a first (jevmlx tails heartbeat and prediction files).
- Do not claim fastest: Laya, trained 4B models with CUDA graphs and von on small schemas are faster. Our speed claim is relative to the demo and to stock decoder engines on the same card.
- Speed versus the demo is a target, not a result, until measured on the 5070 with the protocol in section 2.

Cite:
- Hosted Jev BoolQ 91.56% vs best open stock run 89.45% (openjev-sglang, VERIFIED).
- JevBench control: raw stock Qwen3-4B direct logits score 41.0, hard ECE 0.452 (UNVERIFIED, from its results JSON).
- AnyJev flip 0.230 to 0.073 (VERIFIED on one cell; other cells are smaller).
- von runs one pass per question (VERIFIED): our shared prefix should win on many-field schemas; measure it.
- Verdict README says "under 35 ms" while its own receipt shows 35.58 ms at K=5 and 140 ms at K=25 (VERIFIED).
- Licensing: openjev-sglang has no license file (ideas only); Verdict-open-jev is NOASSERTION; depend on neither.

Threat to watch: llama.cpp PR 29752 and vLLM PR 59299 are open. If merged, a keyless typed-decision endpoint lands in mainstream engines. Both are uncalibrated, so calibration and the console remain ours.

## 5. Plug-in models

| Project | What can be plugged in |
| --- | --- |
| simple-jev | Any HF causal LM with a chat template, plus a Laya encoder backend; per-backbone settings |
| AnyJev | Any HF causal LM via a one-method backend for raw, L0, L1; heads do not transfer between models |
| open-alternative-jev | Any ChatML-style HF or vLLM model through a chat-format hook |
| jevmlx | Any mlx-lm model, 4-bit; also an OpenAI-compatible top-k logprob backend |
| SemIf | Pinned revisions and backend flags (CUDA, MPS, MLX, llama.cpp, WebGPU) |
| llama.cpp PR / vLLM PR | Any GGUF / any decoder |
| von, jev-style, Verdict, Laya | Fixed trained checkpoints; cannot swap |
| cu-Jev | Qwen3.5 only |

Implications for our model picker:
- Plug-in is not a differentiator against the decoder projects, but it is against every trained-checkpoint project.
- Keep the picker generic (any HF causal LM with a chat template) and gate each entry with checks: the real-boundary token check, the parity gate and a chat-template pin. Show the license next to each model (Qwen2.5-3B is non-commercial and stays out).
- Tested list: Qwen2.5-1.5B-Instruct first (matches the demo), Qwen3-4B-Instruct-2507 default, Qwen3-1.7B fast, Qwen3-0.6B tests. Consider a 9B to 12B tier for accuracy: JevBench suggests model size beats readout (a frozen Gemma-4-12B with letters and one temperature ranks 6), but 12 GB of VRAM makes 12B tight on a 5070. Qwen3.5 stays deferred (hybrid cache).
- Store per-model settings (temperature table, layout, scorer mode) keyed by a backbone fingerprint, and fail loudly on mismatch.
- A certified-agreement check (idea 14) is the safe way to let a user swap in a smaller model.
- Racing two models, or a decoder against an encoder, needs the backend interface (idea 15).

## Caveats

Self-published numbers throughout; nothing re-run. Column assignment in jevmlx's SUMMARY is the research agent's reading. Laya's 0.766 has no committed result file. Star counts may be inflated.
