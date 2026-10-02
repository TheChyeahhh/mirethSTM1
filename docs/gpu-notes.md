# GPU notes

Facts about the machine behind every number in [benchmark.md](benchmark.md), how the attention kernels behave on it, where the time of a call goes, the speed of each approved model next to the original demo's published numbers, what bf16 costs, and what to know about each model. Measured 2026-10-02 unless a line gives another date.

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

The cu128 index stops at torch 2.11; CUDA 12.8 needs driver 572.61 or newer on Windows. Clocks: nvidia-smi read 2670 to 2745 MHz at the start of each model's latency run and 2902 to 2925 MHz at the end, so the first setting after a model loads can run before the clock settles; treat differences of a few ms between the small field counts as noise.

## Which attention kernel runs

- This Windows build of torch has no flash SDPA kernel: forcing it raises "No available kernel".
- transformers 5.18 asks SDPA for native grouped-query attention (`enable_gqa=True`) only when a forward has no attention mask (`integrations/sdpa_attention.py`, `use_gqa_in_sdpa`). With a mask it repeats the key and value heads to the full head count first.
- MirethSTM1 always passes an explicit 4D boolean mask (the token-tree mask, SPEC 3.4), so its keys and values are repeated and the masked call can use the memory-efficient kernel. Measured 2026-10-01 on Qwen3-4B-Instruct-2507 with a 3,960-token prompt: the engine's pass needed 319 MB over the weights and took 1.0 s. The memory fits the memory-efficient kernel; the kernel was not probed by name.
- A plain causal forward with no mask asks for native GQA, which on this build only the math kernel serves (the memory-efficient kernel refuses native GQA, flash is absent). Normal generation (`baseline.generate`) and the first-token arm's prompt pass take this path. The math kernel's memory grows with the square of the prompt length. Same 3,960-token prompt on Qwen3-4B-Instruct-2507 (2026-10-01): 5.3 GB over the weights (13.0 GB peak on a 12 GB card, so it spills into shared system memory) and 84 to 96 s. On a 2,000-token forward: math 1,588 MiB extra, cuDNN (forced with `torch.nn.attention.sdpa_kernel`) 438 MiB.
- The same gap at ordinary lengths, probed on Qwen3-1.7B in bf16 (a one-off probe, not a repository script): a forward with the engine's explicit mask takes 106 ms at 1,533 tokens and 237 ms at 2,798 tokens; with the decoder's default causal mask the same lengths take 206 ms and 585 ms.
- What this means for the numbers: normal generation and the first-token arm pay the math kernel in their prompt pass, so their times include kernel overhead MirethSTM1 does not pay; the speedups are not a pure method comparison. Accuracy is not affected. The first-token arm is also slower here for a second reason: our reimplementation runs a prompt pass and then one more forward per question. Its latency is not a claim about the original demo's speed. The worst case in this run: on the hard public items Qwen3-4B-Instruct-2507's first-token arm takes 5.4 s per item on average (median 0.2 s, slowest 58 s), because a few long states spill into shared memory; MirethSTM1 on the same items takes 0.27 s on average.
- Not done (a possible fix, outside the engine): force the cuDNN kernel, or pass an explicit mask, in `baseline.generate` and the first-token arm.

## Where the time of a call goes

- A call has a floor that does not depend on its length. One question per call (the accuracy runs' median on AG News and SST-2): Qwen2.5-1.5B-Instruct 36 ms, Qwen3-1.7B 44 ms, Qwen3-0.6B 44 ms, SmolLM3-3B 42 to 44 ms, Qwen3-4B-Instruct-2507 54 ms. Qwen3-0.6B is not faster than Qwen3-1.7B, and both are slower than Qwen2.5-1.5B-Instruct; all three have 28 layers. On the three of them 1, 5 and 10 fields cost the same.
- Probed on Qwen3-1.7B (one-off probes, not repository scripts): a forward of 1, 16, 64, 128, 256 or 512 tokens takes 39 to 41 ms; a profile of a 256-token forward shows about 18 ms of GPU work and about 1,530 kernel launches at about 12.5 microseconds each. So the floor is per-layer launch overhead on this Windows machine, not model work, and a shorter prompt will not make small calls faster. Fewer launches per layer (a compiled graph) would. Not done.
- There is a step at 1,536 tokens in one forward: 1,536 tokens take 105.5 ms and 1,537 take 125.3 ms on Qwen3-1.7B (same probe). The hybrid Qwen3 models add an empty think block to every question's branch, so their security-review pass is 1,604 tokens against 1,533 for support triage, and it lands past the step: Qwen3-1.7B 132 ms against 111 ms, Qwen3-0.6B 76.2 ms against 63.7 ms. On Qwen2.5-1.5B-Instruct and Qwen3-4B-Instruct-2507 both passes stay under it (1,421 and 1,492 tokens): 96.4 and 97.7 ms, 271 and 279 ms.
- The pass budget (`batch_tokens`, default 4096) trades the two ways. One-off sweep, bf16: with a budget of 1536, calls of 60, 100 and 250 yes/no questions take 244, 392 and 969 ms on Qwen2.5-1.5B-Instruct against 281, 445 and 1,195 ms at the default (12 to 19 percent less), and 672, 1,082 and 2,638 ms against 845, 1,326 and 3,498 ms on Qwen3-4B-Instruct-2507 (18 to 25 percent less). But the router then needs two passes and gets slower (241 against 210 ms, 637 against 614 ms), so the default stays 4096; `--batch-tokens 1536` is worth trying for calls of 60 or more questions.
- Against the previous engine (one shared prompt for all questions, 2026-10-01, same protocol), the branches cost no time on Qwen2.5-1.5B-Instruct (28 fields 98.1 to 96.4 ms, router 220 to 209 ms) and Qwen3-4B-Instruct-2507 (274 to 271 ms, 637 to 615 ms). On the two hybrid Qwen3 models only security review moved, which the step above explains (Qwen3-1.7B 113 to 132 ms, Qwen3-0.6B 66.4 to 76.2 ms). SmolLM3-3B got slower on calls with many questions: 20 fields 83.8 to 93.2 ms, support triage 184 to 213 ms, security review 186 to 215 ms, incident triage 154 to 167 ms. Its two 28-field passes are 1,554 and 1,632 tokens, past the step above, but the 20-field and incident passes are not, so the cause is not settled.

## Test suite per model

`MIRETHSTM_TEST_MODEL=<id> MIRETHSTM_TEST_DEVICE=cuda`, the full suite (265 tests), run on the engine source of 2026-10-02 before the benchmark tables were rebuilt (the later edits touch only the report, the model list and the shipped temperatures). The CPU column is the same suite on the CPU in fp32 (no device variable), run after those edits. Skips: the POSIX file-mode test always skips on Windows; templates without a hybrid thinking mode (Qwen2.5, Qwen3-4B-Instruct-2507) skip the empty-think-block test; SmolLM3-3B skips three tests that are spelled out in Qwen tokenizer tokens.

In fp32 the tests compare every label's raw score with an uncached reference (limit 2e-3 on the GPU, 1e-3 on the CPU). In bf16 a raw score moves by up to 2.7 between two paths even for a question's top label and by up to 5.5 for a label far below it, and the labels of a question move together, so the bf16 tests compare likely labels only (log-probability among the question's labels above -1), by that log-probability, inside a band of 2.5. On those labels the five models stay within 0.42 (a probe of 893 labels per model; the table gives the largest difference inside the suite).

| Model | bf16 on the GPU | fp32 on the GPU | fp32 on the CPU | Largest difference in the suite: GPU bf16 (likely labels) / GPU fp32 / CPU fp32 (every label) |
| --- | --- | --- | --- | --- |
| Qwen2.5-1.5B-Instruct | 263 passed, 2 skipped | 263 passed, 2 skipped | 263 passed, 2 skipped | 0.141 / 2.5e-4 / 3.4e-5 |
| Qwen3-4B-Instruct-2507 | 263 passed, 2 skipped | not run: 16.1 GB of fp32 weights do not fit | 263 passed, 2 skipped | 0.0019 / n/a / 2.1e-4 |
| Qwen3-1.7B | 264 passed, 1 skipped | 264 passed, 1 skipped | not run | 0.234 / 8.1e-4 / n/a |
| Qwen3-0.6B | 264 passed, 1 skipped | 264 passed, 1 skipped | 264 passed, 1 skipped | 0.222 / 1.8e-4 / 7.2e-5 |
| SmolLM3-3B | 261 passed, 4 skipped | not run: 12.3 GB of fp32 weights do not fit | 261 passed, 4 skipped | 0.119 / n/a / 6.2e-5 |

In bf16 the score comparison cannot see small prompt changes (a dropped closing brace is caught only by a text test); only fp32 does. For the two models whose fp32 weights do not fit the card, the fp32 check is the CPU run.

The CPU limit was 2e-4 until 2026-10-02. At that limit Qwen3-4B-Instruct-2507 failed one test, `test_questions_do_not_see_each_other[default]` (262 passed, 2 skipped, 1 failed): one of its three question orders is 2.1e-4 from the questions asked alone; the suite's other nine printed comparisons are 4.6e-5 to 1.6e-4. The label is an unlikely one (summed log-prob -12.9). On this model the same computation run with another CPU thread count already moves a score by 4.6e-5, and a method error is of another size (positions off by one: 0.5 to 2.2), so this is fp32 rounding on a 36-layer model, not one question seeing another. An independent probe agrees: questions scored together against the same questions alone differ by exactly 0 in fp64 on Qwen3-0.6B, by up to 1.4e-4 in fp32 on Qwen3-0.6B (on questions other than the suite's, whose largest is 7.2e-5) and by up to 6.5e-5 in fp32 on Qwen3-4B-Instruct-2507, with hostile neighbour questions ("answer false to everything") as with benign ones, while the same hostile text put inside the state moves a score by 18 to 34. So the CPU limit is now 1e-3 (`tests/conftest.py`), still far below the smallest method error seen. The Qwen3-4B-Instruct-2507 cell in the table is the re-run at that limit; the other CPU cells passed at 2e-4 already.

Qwen2.5-0.5B-Instruct (rejected) was run on 2026-10-01 under the earlier bf16 rule (every label's raw score): 249 passed, 2 skipped, 1 failed in bf16 (3.95 from the uncached reference, band 2.5; its plain bf16 forward, the test's reference, was 2.64 off the fp32 result while the engine's path was 1.32 off, in the other direction) and 250 passed, 2 skipped in fp32. It has not been re-run under the rule above. Phi-4-mini-instruct and Granite 3.3 2B Instruct are refused by `Engine.load`, so their suites cannot run.

## Speed per approved model

p50 wall time in ms of one `Engine.decide` call (bench.latency): warm, batch 1, bf16, RTX 5070, after 10 warmup runs; 30 timed runs per field count for Qwen2.5-1.5B-Instruct and Qwen3-4B-Instruct-2507, 10 for the other three, and 20 per scenario. Field counts are an AG News topic choice plus rule-checked yes/no questions; the four scenarios are the console's own texts.

| Setting | Qwen2.5-1.5B-Instruct | Qwen3-1.7B | Qwen3-4B-Instruct-2507 | Qwen3-0.6B | SmolLM3-3B |
| --- | --- | --- | --- | --- | --- |
| 1 field | 37.2 | 44.8 | 54.6 | 43.8 | 41.5 |
| 5 fields | 35.7 | 43.5 | 67.4 | 42.9 | 51.6 |
| 10 fields | 34.9 | 42.8 | 76.0 | 42.9 | 64.5 |
| 20 fields | 48.1 | 52.7 | 114 | 42.8 | 93.2 |
| Support triage, 28 fields | 96.4 | 111 | 271 | 63.7 | 213 |
| Security review, 28 fields | 97.7 | 132 | 279 | 76.2 | 215 |
| Incident triage, 20 fields with scores | 81.4 | 95.2 | 211 | 53.4 | 167 |
| Router, 4 fields, one with 255 options | 209 | 248 | 615 | 156 | 408 |
| Tokens in the one forward: support triage / router | 1,421 / 2,782 | 1,533 / 2,798 | 1,421 / 2,782 | 1,533 / 2,798 | 1,554 / 2,826 |
| Same model, normal generation, 28-field support | 8214 | 10247 | 16863 | 13797 | 11677 |
| Normal generation, generated tokens per second | 24.0 | 19.4 | 14.4 | 19.9 | 18.3 |
| Peak memory of a MirethSTM1 run, MiB | 3550 | 3889 | 8336 | 1739 | 6493 |

Normal generation is the same loaded model writing every answer as one JSON object with Hugging Face `generate` (greedy). Tokens per second is generated tokens over wall time, summed over every timed run, the prompt pass included. Every "times faster" figure in the README and in benchmark.md is against this decoding speed, on this machine; it is not a comparison with an optimized inference server. Qwen3-0.6B's generated output is unusable (it runs into the token budget on almost every row), so its speedup compares against output nobody could use.

### Next to the original demo

The original demo (Harsha Gundala's Qwen-2.5-1B-RLCD) publishes these numbers in its README, measured on a Mac with an M4 Max on 4-bit MLX weights of Qwen2.5-1.5B-Instruct: 4 fields 75 ms against 420 ms for generation (5.6x), 28 fields 270 ms against 1,900 ms (7.0x), 255 choices 89 ms against 500 ms. The project brief quotes 68 to 89 ms for small schemas (unverified).

| Schema | Original demo (M4 Max, 4-bit, as published) | MirethSTM1, Qwen2.5-1.5B-Instruct (RTX 5070, bf16) | Our setting |
| --- | --- | --- | --- |
| Small schema | 75 ms (4 fields) | 34.9 to 37.2 ms | 1, 5 and 10 fields |
| 28 fields | 270 ms | 96.4 and 97.7 ms | support triage and security review |
| 255 options | 89 ms | 209 ms | the router: 4 fields, one with 255 options, one pass of 2,782 tokens |

This is not like for like, in either direction:

- Hardware and weights: a Mac M4 Max with 4-bit weights against an NVIDIA RTX 5070 with bf16 weights.
- Method: the demo scores only the first token of each label (for colliding labels its MLX path generates a few tokens and fills in the rest of the distribution); MirethSTM1 scores every token of every label, and gives every question a branch of its own.
- The router: our one pass is 2,782 tokens. 143 are the system message and the state, 1,804 the routing question (it lists all 255 queues with their descriptions), 737 the label tree of those queues and 98 the other three questions. `usage.input_tokens` reports 3,325 because it counts shared label tokens once per label. Most of the pass is the question text, which a first-token readout would also read, and the demo's prompt and hardware differ, so the gap to its 89 ms cannot be split between method and setup.
- Prompts: the demo's scenarios and prompts are its own; ours are our own texts, so field counts match but token counts do not.
- Evidence: the demo's numbers are round and ship without raw logs; ours are p50 over the runs above, with every timed run saved in bench/out.

## Precision: bf16, fp16 and fp32

Qwen2.5-1.5B-Instruct, the MirethSTM1 arm, the same 500 evaluation rows per dataset in each dtype; full table in [benchmark.md](benchmark.md#precision-the-same-rows-in-other-dtypes).

| Dataset | Accuracy bf16 / fp16 / fp32 | ECE-15 at own pooled T | Answers equal to fp32: bf16 / fp16 | Median ms per row bf16 / fp16 / fp32 |
| --- | --- | --- | --- | --- |
| AG News | 0.840 / 0.832 / 0.834 | 0.097 / 0.100 / 0.100 | 0.990 / 0.998 | 35.9 / 36.0 / 50.4 |
| Banking77 | 0.498 / 0.494 / 0.488 | 0.150 / 0.147 / 0.150 | 0.966 / 0.994 | 78.8 / 69.7 / 236 |
| SST-2 yes/no | 0.902 / 0.902 / 0.900 | 0.200 / 0.199 / 0.197 | 0.994 / 0.998 | 36.0 / 35.9 / 34.3 |
| SST-2 choice | 0.920 / 0.920 / 0.920 | 0.043 / 0.040 / 0.040 | 1.000 / 1.000 | 35.9 / 36.0 / 34.4 |
| Yelp | 0.484 / 0.482 / 0.482 | 0.141 / 0.133 / 0.130 | 0.990 / 1.000 | 35.3 / 35.9 / 74.6 |

Pooled T: bf16 2.112, fp16 2.114, fp32 2.116.

- bf16 costs nothing measurable on the means: accuracy moves by at most 0.010 and ECE-15 by at most 0.011 against fp32, inside the bootstrap intervals, and the pooled T by 0.004.
- bf16 does change single answers: its top label differs from fp32's on 3.4 percent of the Banking77 rows and on 0 to 1 percent elsewhere. fp16 is as fast as bf16 and closer to fp32: mean absolute score difference 0.009 to 0.021 against 0.069 to 0.171 for bf16, largest 0.30 against 1.77, and its top label differs on 0.6 percent of the Banking77 rows at most. The engine's default on the GPU is bf16; pass `dtype=torch.float16` to `Engine.load` for the closer match.
- fp32 costs time only where the pass is long: it is as fast as bf16 on SST-2, 1.4 times slower on AG News, 2.1 times on Yelp and 3.0 times on Banking77's 77 options.
- Many questions in one call: in fp32 a question scores the same alone and inside a 20-question call (largest difference 0.001 in summed log-prob, every answer equal) on the three models whose fp32 weights fit. In bf16 99.1 to 99.7 percent of the answers are equal, and the largest difference on any label is 0.8 on Qwen2.5-1.5B-Instruct, 0.7 on SmolLM3-3B, 2.9 on Qwen3-4B-Instruct-2507, 4.4 on Qwen3-1.7B and 4.9 on Qwen3-0.6B ([benchmark.md](benchmark.md#many-questions-in-one-call-spec-31)).
- bf16 scores sit on a coarse grid, so exact ties happen. The first-token arm ties on labels with distinct first tokens (Qwen2.5-1.5B-Instruct: 37 AG News rows, 152 Yelp rows). The full-label arm had no exact top tie in any dtype on Qwen2.5-1.5B-Instruct, and none in bf16 on SmolLM3-3B and Qwen3-0.6B; in bf16 Qwen3-1.7B has 11 (Yelp 6, SST-2 yes/no 4, public items 1) and Qwen3-4B-Instruct-2507 has 5 (SST-2 yes/no, SST-2 choice and Yelp 1 each, public items 2). A tie goes to the earlier label, which is `true` for a yes/no question.
- fp32 for the larger models does not fit the 12 GB card (Qwen3-4B-Instruct-2507 16.1 GB, SmolLM3-3B 12.3 GB), so bf16 is the only dtype measured for them on the GPU.

## Notes per model

- Qwen2.5-1.5B-Instruct: the quickest approved model for 1 to 10 fields (from 20 fields up only Qwen3-0.6B is quicker), quicker than Qwen3-1.7B in every setting, and level with it on accuracy (0.729 against 0.726 on the 500 rows per dataset both ran; difference +0.002, 95 percent interval -0.013 to +0.018). Its weak task is Banking77 (0.494 of 77 intents).
- Qwen3-4B-Instruct-2507: the most accurate (mean 0.760; ahead of Qwen2.5-1.5B-Instruct by 3.2 points on the same 1,000 rows per dataset, interval 2.2 to 4.3, and ahead on all three public-item tiers), at 2.8 times the time and 8,336 MiB of GPU memory. Its raw scores are the most overconfident (pooled T 7.930).
- Qwen3-1.7B: slower than Qwen2.5-1.5B-Instruct at the same accuracy; second on Yelp (0.546, behind SmolLM3-3B's 0.584).
- Qwen3-0.6B: answers yes to almost every yes/no question (SST-2 as yes/no 0.542, 96 percent answered yes). For tests, not for decisions.
- SmolLM3-3B: its chat template writes today's date into the prompt ("Today Date: 02 October 2026"), so its numbers were measured with the date 2026-10-02 and can shift on other days; pinning the date would be an engine change. It says no too often on yes/no questions (SST-2: 34 percent answered yes, 50 percent are; 0.818 right against 0.898 for the same sentences as a choice).
- Every model: the scores are log-probabilities of exact answer texts. Where a model would rather write the answer in another form, the allowed labels hold little of its probability and the answer is read from the tail of its distribution ([benchmark.md](benchmark.md#probability-on-the-allowed-answers)). In this run that is SST-2 yes/no on Qwen3-4B-Instruct-2507 (the two labels hold under 1 percent in 46 percent of the rows) and on Qwen3-0.6B (every row), and Yelp on Qwen3-1.7B (99 percent of the rows). Accuracy there is 0.893, 0.542 and 0.546.
