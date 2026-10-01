# MirethSTM1 roadmap to v0.1 (ship Sunday 2026-10-04)

Promise: the engine and a live console ship free and public by Sunday. Spec: `SPEC.md`. Research: `docs/research/`.

## Must ship (never cut)

- SDK: `Engine.load(model).decide(context, schema)` with full-label scoring
- CLI: `mirethstm decide`
- Calibration: `fit_temperature`, ECE, shipped default temperatures
- Benchmark table with ECE before and after calibration
- README with honest framing, model license table, credits
- Minimal console that tails the event log and plots confidence (`mirethstm console`)

## Cut order (first to go)

1. MCP server
2. FastAPI server
3. Tarnlight tap
4. Letter-label comparison arm in the benchmark

## Deferred (after v0.1)

Packed-mask or tree batching; vLLM, llama.cpp and MLX backends; fine-tuning; Qwen3.5 hybrid models; numeric min/max score range; move to torch 2.14 on cu130.

## Wednesday 9/30: research, spec, scaffold (this session)

- [x] Research: 9 topic docs plus an index, cross-checked (`docs/research/`)
- [x] `SPEC.md`, this roadmap
- [x] CPU scaffold: engine with cached full-label scoring, CLI `decide`, calibration, frozen event log, benchmark stub, tests on Qwen3-0.6B
- [x] `HANDOFF.md`

## Thursday 10/1: GPU bring-up, engine complete

1. Windows GPU check from `HANDOFF.md`: `sm_120` in the arch list, a real CUDA forward. Verify: the check script prints OK.
2. Full test suite on the GPU in bf16; record the cached vs uncached max difference. Verify: tests green, number written in `docs/research/05-environment.md`.
3. Load Qwen3-4B-Instruct-2507; measure peak VRAM at 2k context for chunk sizes 4, 8, 16; set the default. Check which SDPA backend runs with a mask. Verify: numbers in a new `docs/gpu-notes.md`.
4. Latency smoke: 1, 5, 10, 20 fields, p50 and p95 over 50 runs. Verify: table in `docs/gpu-notes.md`.
5. `mirethstm console` (stdlib page, per-field lines, gauge, feed). Verify: run `decide --log` in a loop and watch it update.

## Friday 10/2: benchmark harness

1. Loaders: `fancyzhx/ag_news`, `mteb/banking77`, `stanfordnlp/sst2` (validation), `Yelp/yelp_review_full` (5 levels). Fixed seeds, fixed sample lists. Verify: loader test prints counts and label names.
2. Baseline: same model, greedy `generate()` of compact JSON, schema check, invalid output counted wrong. Verify: validity rate printed.
3. Metrics: accuracy, macro-F1, NLL, Brier (sum over classes), ECE 15-bin (and 10-bin), bootstrap 95% CI; per-sample JSONL with raw label scores so tables rebuild without the GPU. Verify: unit tests, including accuracy and F1 unchanged under T.
4. Dry run on Qwen3-1.7B, n = 200 per dataset. Verify: a full table renders.
5. Tarnlight tap (small, cut third). Verify: a decision appears in Tarnlight.

## Saturday 10/3: runs, calibration, README

1. Full runs on Qwen3-4B-Instruct-2507 and Qwen3-1.7B: our scorer vs the generate baseline, plus the latency table.
2. Fit T on calibration splits; ECE before and after on held-out data; reliability diagrams (PNG). Fill `DEFAULT_TEMPERATURES`.
3. README benchmark table and diagrams.
4. **HARD CUT LINE: Saturday 18:00 local.** Anything not green by then is cut in the order above. The must-ship list is never cut; if a must-ship item is red at 18:00, Sunday morning goes to it.
5. After the cut line, only if green: FastAPI server (compat `POST /v1/systemone`), then the MCP server.

## Sunday 10/4: release

1. Fresh clone on Windows, run the README install commands exactly, full test suite.
2. README final pass: honest framing, license table, credits, trademark note, no em dashes.
3. Privacy scan of the whole tree (no personal paths, names or private project notes).
4. Founder review, then flip the repo public, tag `v0.1.0`, GitHub release.
5. Launch announcement.
