# MirethSTM1 roadmap to v0.1 (ship Sunday 2026-10-04)

Promise: the engine and a live console ship free and public by Sunday. Spec: `SPEC.md`. Research: `docs/research/`.

Order of work (founder, 2026-09-30): speed first. Match the original demo on its own model (Qwen2.5-1.5B-Instruct) before trying other models; any model plugs in.

## Must ship (never cut)

- SDK: `Engine.load(model).decide(context, schema)` with full-label scoring, any Hugging Face chat model
- CLI: `mirethstm decide`
- Race-view console (`mirethstm console`): MirethSTM1 and normal generation side by side, like the original demo; feeds Tarnlight when it is installed, works without it
- Calibration: `fit_temperature`, ECE, shipped default temperatures
- Benchmark table with ECE before and after calibration, and a speed table against the original demo's numbers
- README with honest framing, model license table, credits

## Cut order (first to go)

1. MCP server
2. FastAPI server
3. Letter-label comparison arm in the benchmark

## Deferred (after v0.1)

Tree batching (one shared copy of each question's suffix); vLLM, llama.cpp and MLX backends; fine-tuning; Qwen3.5 hybrid models; numeric min/max score range; move to torch 2.14 on cu130.

## Wednesday 9/30: research, spec, scaffold

- [x] Research: 9 topic docs plus an index, cross-checked (`docs/research/`)
- [x] `SPEC.md`, this roadmap
- [x] CPU scaffold: engine with full-label scoring, CLI `decide`, calibration, frozen event log, benchmark stub, tests on Qwen3-0.6B
- [x] `HANDOFF.md`
- [x] Packed single-pass scorer (all labels of all questions in one pass; 3.7x to 4.9x faster than day 1 on CPU)
- [x] Model plug-in (any chat model; full model test suite passes on Qwen2.5-1.5B-Instruct on CPU)
- [x] Race-view console and the normal-generation baseline
- [x] Tarnlight tap
- [x] Deep dive on the open-source engines (`docs/research/10` to `16`)

## Thursday 10/1: GPU bring-up, match the original demo

1. Windows GPU check from `HANDOFF.md`: `sm_120` in the arch list, a real CUDA forward. Verify: the check prints OK.
2. Full test suite on the GPU in bf16; record the cached vs uncached max difference. Verify: tests green, number written in `docs/research/05-environment.md`.
3. Match first: `mirethstm console` on Qwen2.5-1.5B-Instruct, all four scenarios. Record our milliseconds and the normal-generation milliseconds next to the original demo's claims (68 to 89 ms for small schemas, 270 ms for 28 fields, 5.6x to 7x, on an M4 Max). Verify: table in `docs/gpu-notes.md`.
   - If the 255-option router is still slower than normal generation on the GPU (it is on CPU: 0.7x), build trie batching: score each shared label prefix once (most router labels share their first tokens), the idea jevmlx and the open llama.cpp pull request use. Verify: same scores within 2e-4, router faster than generation.
4. Then Qwen3-1.7B and Qwen3-4B-Instruct-2507 in the model picker; peak VRAM at 2k context for `batch_tokens` 1024, 2048, 4096; which SDPA backend runs with the 4D mask. Verify: numbers in `docs/gpu-notes.md`, default `batch_tokens` set.
5. Latency table: 1, 5, 10, 20 fields, p50 and p95 over 50 runs, per model. Verify: table in `docs/gpu-notes.md`.

## Friday 10/2: benchmark harness

1. Loaders: `fancyzhx/ag_news`, `mteb/banking77`, `stanfordnlp/sst2` (validation), `Yelp/yelp_review_full` (5 levels). Fixed seeds, fixed sample lists. Verify: loader test prints counts and label names.
2. Baseline arm: `mirethstm.baseline.generate` (same model, greedy JSON), invalid or hallucinated answers counted wrong. Verify: validity rate printed.
3. Original-method arm: first-token scoring on the same model, reimplemented from its description, so speed and accuracy compare head to head (the original demo published no accuracy; an independent audit put its model at 58 percent on 24 synthetic cases, barely above the 54 percent majority answer). Verify: both arms in one table.
4. Comparable accuracy: run the public JevBench items (231) and the BoolQ protocol other engines publish, so our numbers sit next to theirs (`docs/research/16-landscape-summary.md`). Needs `POST /v1/systemone`; a small route on the console's server is enough (founder decision).
5. Metrics: accuracy, macro-F1, NLL, Brier (sum over classes), ECE 15-bin (and 10-bin), bootstrap 95% CI; per-sample JSONL with raw label scores so tables rebuild without the GPU. Verify: unit tests, including accuracy and F1 unchanged under T.
6. Dry run on Qwen2.5-1.5B-Instruct and Qwen3-1.7B, n = 200 per dataset. Verify: a full table renders.

## Saturday 10/3: runs, calibration, README

1. Full runs on Qwen2.5-1.5B-Instruct, Qwen3-1.7B and Qwen3-4B-Instruct-2507: our scorer vs the baseline and the original-method arm, plus the latency table.
2. Fit T on calibration splits; ECE before and after on held-out data; reliability diagrams (PNG). Fill `DEFAULT_TEMPERATURES`.
3. README benchmark and speed tables, diagrams, a console screenshot.
4. **HARD CUT LINE: Saturday 18:00 local.** Anything not green by then is cut in the order above. The must-ship list is never cut; if a must-ship item is red at 18:00, Sunday morning goes to it.
5. After the cut line, only if green: FastAPI server (compat `POST /v1/systemone`), then the MCP server.

## Sunday 10/4: release

1. Fresh clone on Windows, run the README install commands exactly, full test suite.
2. README final pass: honest framing, license table, credits, trademark note, no em dashes.
3. Privacy scan of the whole tree (no personal paths, names or private project notes).
4. Founder review, then flip the repo public, tag `v0.1.0`, GitHub release.
5. Launch announcement.
