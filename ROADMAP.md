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

## Done (Wednesday 9/30 to Friday 10/2)

- [x] Research: 17 notes, cross-checked (`docs/research/`), including the open-source engines and the open-weight trained decision models
- [x] `SPEC.md`, kept in step with the code
- [x] Engine: whole-answer scoring in one forward pass, token trees, every question in its own private branch (asked alone or among 20, same answer)
- [x] Any Hugging Face chat model through a loader check; five approved models with measured numbers; three refused with reasons
- [x] CLI `decide`, race-view console with a model picker, local TypeSafe-compatible API (works with the official SDK), Tarnlight tap, frozen event log
- [x] GPU bring-up on the RTX 5070; full test suite green on the GPU for every approved model
- [x] Benchmark kit and campaign: accuracy, calibration, latency, public JevBench items, many-questions test; report and reliability diagrams in `docs/`
- [x] Shipped temperatures per model
- [x] README with the benchmark, speed and model tables and the honest framing
- [x] Every published number recomputed by an independent check
- [x] A recorded race on the GPU

Measured and dropped along the way (kept in the docs as lessons): a questions-first prompt with a cache (accuracy fell from 0.84 to 0.48 on AG News), a shared prompt for all questions (yes/no answers collapsed inside a 20-question call), a long answer rule after every question (twice the time).

Not built: the MCP server (first on the cut list) and a FastAPI server (replaced by the API on the console's own server).

## After v0.1

1. A second backend so trained decision models (fastino/GLiNER2.5-Decide first) can be picked and raced in the console.
2. Accuracy: score both answer spellings for yes/no (bare and quoted), a temperature per question type, better handling of ordered scales.
3. Speed: one question with many options (the option list in the prompt is the cost), the cuDNN attention kernel (measured 10 to 19 percent on the 4B), a smaller pass budget for calls with 60 or more questions.
4. Models: fix the Phi-4-mini loader refusal and its position bug; re-run Qwen2.5-0.5B.
5. The MCP server.
6. Move to torch 2.14 on cu130.

## Sunday 10/4: release

1. Fresh clone on Windows, run the README install commands exactly, full test suite.
2. README final pass: honest framing, license table, credits, trademark note, no em dashes.
3. Privacy scan of the whole tree (no personal paths, names or private project notes).
4. Founder review, then flip the repo public, tag `v0.1.0`, GitHub release.
5. Launch announcement.
