# GPU notes

Facts about the machine behind every number in [benchmark.md](benchmark.md), how the attention kernels behave on it, the speed of each approved model next to the original demo's published numbers, and what bf16 costs. Measured 2026-10-01.

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5070, 12 GB (Blackwell, compute capability 12.0, `sm_120`) |
| Driver | 591.86 (nvidia-smi, recorded in the run's meta.json) |
| CPU | AMD Ryzen 9 5900X (`AMD64 Family 25 Model 33`) |
| OS | Windows 11 (Python's platform string reads `Windows-10`) |
| Python | 3.11.15 |
| PyTorch | 2.11.0+cu128 (CUDA 12.8 runtime, from the cu128 wheel index) |
| Compiled CUDA archs | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` (`torch._C._cuda_getArchFlags()`) |
| transformers | 5.18.0 |
| Attention | `attn_implementation="sdpa"`; no flash-attn package, no Triton, no `torch.compile` |
| Dtype | bf16 for every benchmark number unless a table says otherwise |

The cu128 index stops at torch 2.11; CUDA 12.8 needs driver 572.61 or newer on Windows. Clocks: nvidia-smi read 2655 to 2722 MHz at the start of each model's latency run and 2917 to 2925 MHz at the end, so the first setting after a model loads can run before the clock settles (Qwen3-1.7B's 1-field p50 of 48.5 ms sits above its 5-field 44.7 ms; treat differences of a few ms between the small field counts as noise).

## Which attention kernel runs

- This Windows build of torch has no flash SDPA kernel: forcing it raises "No available kernel".
- transformers 5.18 asks SDPA for native grouped-query attention (`enable_gqa=True`) only when a forward has no attention mask (`integrations/sdpa_attention.py`, `use_gqa_in_sdpa`). With a mask it repeats the key and value heads to the full head count first.
- MirethSTM1 always passes an explicit 4D boolean mask (the token-tree mask, SPEC 3.4), so its keys and values are repeated and the masked call can use the memory-efficient kernel. Measured on Qwen3-4B-Instruct-2507 with a 3,960-token prompt: the engine's pass needed 319 MB over the weights and took 1.0 s. The memory fits the memory-efficient kernel; the kernel was not probed by name.
- A plain causal forward with no mask asks for native GQA, which on this build only the math kernel serves (the memory-efficient kernel refuses native GQA, flash is absent). Normal generation (`baseline.generate`) and the first-token arm's prompt pass take this path. The math kernel's memory grows with the square of the prompt length. Same 3,960-token prompt on Qwen3-4B-Instruct-2507: 5.3 GB over the weights (13.0 GB peak on a 12 GB card, so it spills into shared system memory) and 84 to 96 s. On a 2,000-token forward: math 1,588 MiB extra, cuDNN (forced with `torch.nn.attention.sdpa_kernel`) 438 MiB.
- What this means for the numbers: normal generation and the first-token arm pay the math kernel in their prompt pass, so their times include kernel overhead MirethSTM1 does not pay; the speedups are not a pure method comparison. Accuracy is not affected. The first-token arm is also slower here for a second reason: our reimplementation runs a prompt pass and then one more forward per question. Its latency is not a claim about the original demo's speed.
- Not done (a possible fix, outside the engine): force the cuDNN kernel, or pass an explicit mask, in `baseline.generate` and the first-token arm.

## GPU test suite per model

`MIRETHSTM_TEST_MODEL=<id> MIRETHSTM_TEST_DEVICE=cuda`, the full suite (252 tests). Skips: the POSIX file-mode test always skips on Windows; models without Qwen3's hybrid thinking template (Qwen2.5, Qwen3-4B-Instruct-2507) skip the empty-think-block test; SmolLM3-3B skips three tests that need Qwen's own tokenizer.

| Model | bf16 | fp32 | Largest bf16 drift from the uncached reference (band 2.5) |
| --- | --- | --- | --- |
| Qwen2.5-1.5B-Instruct | 250 passed, 2 skipped | 250 passed, 2 skipped | up to 0.6 |
| Qwen3-4B-Instruct-2507 | 250 passed, 2 skipped | not run: 16.1 GB of fp32 weights do not fit; not yet checked on the CPU either | 1.86 (router) |
| Qwen3-1.7B | 251 passed, 1 skipped | 251 passed, 1 skipped | 2.00 (router), the closest to the band |
| Qwen3-0.6B | 251 passed, 1 skipped | 251 passed, 1 skipped | 1.30 (tree) |
| SmolLM3-3B | 247 passed, 4 skipped, 1 failed | not run on the GPU (12.3 GB); the fp32 exactness tests pass on the CPU (max 2.3e-5 against 2e-4) | 1.06 (router) |
| Qwen2.5-0.5B-Instruct (rejected) | 249 passed, 2 skipped, 1 failed | 250 passed, 2 skipped | 3.95 (tree), over the band |

SmolLM3-3B's one failure is `test_prefix_ends_with_empty_think_block`, which expects Qwen3's exact whitespace after `</think>` (two newlines); SmolLM3's own template writes one, and the engine reproduces that template exactly. Qwen2.5-0.5B-Instruct fails the bf16 band because its plain bf16 forward (the test's reference) is 2.64 off the fp32 result while the engine's path is 1.32 off, in the other direction; its fp32 run is exact. Phi-4-mini-instruct and Granite 3.3 2B Instruct are refused by `Engine.load`, so their suites cannot run.

## Speed per approved model

p50 wall time in ms of one `Engine.decide` call (bench.latency): warm, batch 1, bf16, RTX 5070, after 10 warmup runs; 30 timed runs per field count (10 for SmolLM3-3B and Qwen3-0.6B) and 20 per scenario. Field counts are an AG News topic choice plus rule-checked yes/no questions; the four scenarios are the console's own texts.

| Setting | Qwen2.5-1.5B-Instruct | Qwen3-1.7B | Qwen3-4B-Instruct-2507 | Qwen3-0.6B | SmolLM3-3B |
| --- | --- | --- | --- | --- | --- |
| 1 field | 37.2 | 48.5 | 58.7 | 45.5 | 48.4 |
| 5 fields | 36.5 | 44.7 | 72.0 | 45.7 | 52.5 |
| 10 fields | 37.1 | 44.0 | 83.5 | 44.0 | 64.9 |
| 20 fields | 50.6 | 52.6 | 121 | 44.1 | 83.8 |
| Support triage, 28 fields | 98.1 | 110 | 274 | 64.6 | 184 |
| Security review, 28 fields | 99.3 | 113 | 279 | 66.4 | 186 |
| Incident triage, 20 fields with scores | 83.3 | 95.3 | 223 | 54.8 | 154 |
| Router, 4 fields, one with 255 options | 220 | 258 | 637 | 163 | 409 |
| Same model, normal generation, 28-field support | 9704 | 11827 | 17163 | 15014 | 12887 |
| Peak memory of a MirethSTM1 run, MiB | 3545 | 3883 | 8331 | 1733 | 6487 |

### Next to the original demo

The original demo (Harsha Gundala's Qwen-2.5-1B-RLCD) publishes these numbers in its README, measured on a Mac with an M4 Max on 4-bit MLX weights of Qwen2.5-1.5B-Instruct: 4 fields 75 ms against 420 ms for generation (5.6x), 28 fields 270 ms against 1,900 ms (7.0x), 255 choices 89 ms against 500 ms. The demo posts quote 68 to 89 ms for small schemas.

| Schema | Original demo (M4 Max, 4-bit, as published) | MirethSTM1, Qwen2.5-1.5B-Instruct (RTX 5070, bf16) | Our setting |
| --- | --- | --- | --- |
| Small schema | 68 to 89 ms (4 fields: 75 ms) | 36.5 to 37.2 ms | 1, 5 and 10 fields |
| 28 fields | 270 ms | 98.1 and 99.3 ms | support triage and security review |
| 255 options | 89 ms | 220 ms | the router: 4 fields, one with 255 options, 4,388 input tokens |

This is not like for like, in either direction:

- Hardware and weights: a Mac M4 Max with 4-bit weights against an NVIDIA RTX 5070 with bf16 weights.
- Method: the demo scores only the first token of each label (for colliding labels its MLX path generates a few tokens and fills in the rest of the distribution); MirethSTM1 scores every token of every label. On the 255-option router that is 4,388 input tokens in one pass, which is why our router number is above the demo's.
- Prompts: the demo's scenarios and prompts are its own; ours are our own texts, so field counts match but token counts do not.
- Evidence: the demo's numbers are round and ship without raw logs; ours are p50 over the runs above, with every timed run saved in bench/out.

## Precision: bf16, fp16 and fp32

Qwen2.5-1.5B-Instruct, the MirethSTM1 arm, the same 500 evaluation rows per dataset in each dtype; full table in [benchmark.md](benchmark.md#precision-bf16-fp16-and-fp32-qwen25-15b-instruct).

| Dataset | Accuracy bf16 / fp16 / fp32 | ECE-15 at own pooled T | bf16 answers equal to fp32 | Median ms per row bf16 / fp16 / fp32 |
| --- | --- | --- | --- | --- |
| AG News | 0.826 / 0.824 / 0.824 | 0.082 / 0.081 / 0.080 | 0.994 | 36.8 / 36.7 / 57.2 |
| Banking77 | 0.550 / 0.542 / 0.542 | 0.132 / 0.124 / 0.115 | 0.966 | 80.9 / 72.6 / 240 |
| SST-2 yes/no | 0.884 / 0.886 / 0.886 | 0.089 / 0.090 / 0.090 | 0.994 | 38.5 / 36.5 / 51.2 |
| SST-2 choice | 0.920 / 0.922 / 0.922 | 0.020 / 0.027 / 0.029 | 0.998 | 37.6 / 36.7 / 50.9 |
| Yelp | 0.374 / 0.374 / 0.372 | 0.198 / 0.199 / 0.195 | 0.978 | 38.2 / 37.1 / 78.2 |

Pooled T: bf16 2.285, fp16 2.289, fp32 2.291.

- bf16 costs nothing measurable: accuracy moves by at most 0.008 and ECE-15 by at most 0.017 against fp32, well inside the bootstrap intervals.
- fp16 is as fast as bf16 and numerically closer to fp32: mean absolute score difference 0.008 to 0.029 against 0.063 to 0.193 for bf16.
- fp32 is 1.3x to 3.0x slower per row, worst on Banking77's 77 options.
- bf16 logits sit on a coarse grid, which matters for the first-token arm only: on labels with distinct first tokens it produced exact ties (Qwen2.5-1.5B-Instruct: 38 AG News rows, 151 Yelp rows). The full-label arm had no exact top ties in any dtype.
- fp32 for the larger models does not fit the 12 GB card (Qwen3-4B-Instruct-2507 16.1 GB, SmolLM3-3B 12.3 GB), so bf16 is the only dtype measured for them.
