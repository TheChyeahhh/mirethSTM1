# Handoff: v0.1.0, released 2026-10-02

Branch `day1-scaffold`. The contract is `SPEC.md`, the plan is `ROADMAP.md`, the numbers are `docs/benchmark.md` and `docs/gpu-notes.md`, the research is `docs/research/` (start with `00-index.md`, `16-landscape-summary.md` and `17-open-weight-decision-models.md`).

## What exists

- **Engine** (`mirethstm.Engine`): one forward pass reads the state and scores every allowed answer of every question as a whole label. Every question sits in its own private branch, so it is answered exactly as if it were asked alone. TypeSafe request and response shapes (noul, choice, score). Any Hugging Face chat model that passes the loader's check.
- **CLI**: `mirethstm decide` and `mirethstm console`.
- **Race console** (`mirethstm console`, http://127.0.0.1:8766): MirethSTM1 next to the same model writing JSON, live, with a model picker over the approved list.
- **Local API on the same server**: `POST /v1/systemone` (TypeSafe drop-in, works with the official Python SDK), `POST /v1/decide`, `GET /v1/models`. Local only, no key, no outside calls.
- **Tarnlight tap**: every decision also lands in Tarnlight when it is installed; nothing is written otherwise.
- **Frozen event log** (`ts, id, field, label, p, probs, T, latency_ms, model`).
- **Calibration**: `fit_temperature`, `ece`, one shipped temperature per approved model.
- **Benchmark kit** (`bench/`): four datasets as five tasks, three arms (ours, first token, normal generation), latency, the public JevBench items, a many-questions test, a report generator with reliability diagrams.
- **Approved models** (`mirethstm/models.py`), each with measured speed and accuracy.

## Numbers (RTX 5070, bf16, 2026-10-02)

| Model | Role | 28 questions | 255 options | Mean accuracy | Calibration error at shipped T |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-1.5B-Instruct | Fast, console default | 96 ms | 209 ms | 0.728 | 0.109 |
| Qwen3-4B-Instruct-2507 | SDK and CLI default | 271 ms | 615 ms | 0.760 | 0.117 |
| Qwen3-1.7B | Alternative | 111 ms | 248 ms | 0.726 | 0.097 |
| SmolLM3-3B | Alternative | 213 ms | 408 ms | 0.688 | 0.061 |
| Qwen3-0.6B | CPU tests | 64 ms | 156 ms | 0.575 | 0.137 |

- 1 to 10 questions take 35 to 37 ms on the fast model.
- The original demo published 75 ms (4 fields), 270 ms (28 fields) and 89 ms (255 options) on a Mac M4 Max with 4-bit weights of the same base model. Not like for like.
- Normal generation on the same GPU takes 8 to 10 seconds for 28 questions (about 24 tokens a second through Hugging Face `generate`), so "times faster" depends on that baseline.
- Many questions in one call: 0.842 alone and 0.841 to 0.842 inside a 20-question call on the fast model (the earlier shared prompt: 0.500 to 0.809). Identical answers in fp32.
- Whole answer against first token on Banking77 (77 options): 0.494 against 0.313 on the fast model, 0.670 against 0.380 on the 4B.

## Verified, and how

| What | How we know |
| --- | --- |
| 264 tests pass on CPU | `pytest -q` on Qwen3-0.6B fp32: 264 passed, 1 skipped (POSIX file modes) |
| Full suite passes on the GPU for all five approved models | per-model runs in bf16, and in fp32 where the weights fit (`docs/gpu-notes.md`) |
| The engine equals asking each question alone without any cache | fp32 difference at most 2.2e-4 on CPU, 1.3e-3 on the GPU; exactly 0 in fp64 |
| The tests catch real bugs | more than 100 deliberate bugs injected across all rounds; every one is caught now; source restored each time |
| Every published number | recomputed from the raw per-row results by an independent check with its own code, twice |
| The official TypeSafe Python SDK works against the local API | run against `/v1/systemone` with a dummy key; nothing left the machine |
| GPU support | torch 2.11.0+cu128, arch list includes `sm_120`, driver 591.86 |

Not verified:

- Speed and accuracy on any other GPU, on Linux or on a Mac.
- GPU speed of the trained decision models (Fastino, Mapika); they were only run on the CPU.
- Accuracy against hosted Jev itself. Nothing here was compared with it directly.
- Qwen2.5-0.5B under the current test rule (it failed under the earlier one and was not re-run).

## Known limits

1. One question with 255 options is slower than the original demo's figure (209 ms against 89 ms): the prompt lists all 255 options and every option is read in full.
2. Ordered scales (Yelp stars) are the weakest area: 0.44 to 0.58 on the four larger models, 0.28 on the smallest. A trained model (GLiNER2.5-Decide) gets 0.56 on the same rows.
3. One temperature per model fits yes/no questions poorly on two models. A temperature per question type is the next calibration step.
4. With the short prompt, models often want another answer format (a quoted "yes" instead of true); the allowed answers then share a small part of the probability. Accuracy holds, but scoring both spellings is a likely gain (measured +1 point on SST-2 for the 4B).
5. The short prompt costs the 4B model 1.6 points on one yes/no test and the fast model 5 points on Banking77; it gains 10 points on Yelp and is twice as fast as repeating the rule per question.
6. Trained decision models (fastino/GLiNER2.5-Decide, Mapika/decider) do not plug into this engine. They are compared in the README; a second backend is planned.
7. Phi-4-mini and Granite are refused by the loader (reasons in `SPEC.md` 11.1).
8. SmolLM3's chat template writes today's date into the prompt, so its numbers can shift from day to day.

## Decisions taken with the founder

- Console is a race view like the original demo; it feeds Tarnlight and works without it.
- Score questions are TypeSafe's level list. No numeric min/max in v0.1.
- torch 2.11 on cu128 until after the release.
- Honest README: no "first", "only" or "fastest"; other projects named.
- Repo went public on 2026-10-02 after the founder's review (two days before the promised Sunday).
- Match the original demo's model first, then offer a list of approved models.
- The local TypeSafe-compatible API costs nothing and is tied to no account.
- The founder waits to post until the final numbers and the video exist (they do now).

## Settled 2026-10-02

1. Console default stays the fast model (Qwen2.5-1.5B-Instruct).
2. After the release: build the second backend so GLiNER2.5-Decide can be picked and raced in the console.
3. After the release: build the MCP server.

v0.1.0 is released. Next is the after-release list in `ROADMAP.md`.

## Run it

Environment on this desktop: `.venv` in the repo (Python 3.11, torch 2.11.0+cu128, transformers 5.18.0). The five approved models are in the local Hugging Face cache.

```powershell
.\.venv\Scripts\mirethstm.exe console
```

Open http://127.0.0.1:8766, pick a scenario and a model, press Run comparison.

```powershell
cmd /c ".venv\Scripts\mirethstm.exe decide --schema s.json < ctx.txt"
.\.venv\Scripts\python.exe -m pytest -q
$env:MIRETHSTM_TEST_DEVICE = "cuda"; .\.venv\Scripts\python.exe -m pytest -q -m model
.\.venv\Scripts\python.exe -m bench.report --run v2 --md docs/benchmark.md --figures docs/benchmark
```

Fresh machine: see the README install section (venv, torch from the cu128 index, `pip install -e ".[dev,bench]"`).

## Release checklist (done 2026-10-02)

1. Founder reads the README, `docs/benchmark.md` and the console once.
2. Fresh clone on Windows, README install commands exactly, full test suite.
3. Privacy scan of the whole tree (no personal paths, names or private notes).
4. Merge `day1-scaffold` into `main`, flip the repo public, tag `v0.1.0`, GitHub release.
5. Announcement, using only the claims in the README.
