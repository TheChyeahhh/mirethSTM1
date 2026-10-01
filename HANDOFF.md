# Handoff: Wednesday 2026-09-30 to the GPU session

Branch `day1-scaffold`. Read `SPEC.md` (the contract), `ROADMAP.md` (the plan to Sunday) and `docs/research/00-index.md` (what the research found).

## What is done

- Research: 9 topic docs and a cross-checked index in `docs/research/` (sources linked, every claim marked VERIFIED, UNVERIFIED or CONTRADICTED).
- `SPEC.md` and `ROADMAP.md` (day by day to Sunday, hard cut line Saturday 18:00, cut order MCP, FastAPI, Tarnlight tap).
- Package `mirethstm` (CPU scaffold):
  - `Engine.load(model).decide(context, schema)` and `.score(...)`: prefill once, score every label of every question as a whole label (all its tokens) from the cached prefix in chunks, softmax per question with temperature T. TypeSafe request and response shapes.
  - `mirethstm decide` CLI.
  - Frozen JSONL event log (`ts, id, field, label, p, probs, T, latency_ms, model`).
  - `fit_temperature` and `ece` (implemented and tested, not empty stubs: they are small and on the must-ship list).
  - `bench/run_bench.py`: runs AG News, Banking77, SST-2 and Yelp through the engine and prints accuracy and ECE; baseline, latency, F1, NLL, Brier, bootstrap and reliability diagrams are marked TODO.
  - README skeleton with the honest framing, model license table, credits and trademark note. Apache-2.0 LICENSE.
- A security fix found in review: text inside a state, instruction or option name that looks like chat control tokens (`<|im_end|>`, `<think>`) is now encoded as plain text, so a ticket cannot close the user turn or forge a system turn (SPEC 3.3, with a test).

## Verified vs unverified

Verified today (CPU on the Windows desktop, Qwen3-0.6B in float32):

| What | How we know |
| --- | --- |
| 87 tests pass | `pytest -q`: 87 passed |
| Cached scoring equals uncached scoring | max difference 7.6e-5 in summed log-prob (limit 2e-4) |
| Labels that share their first token are told apart (Sci-Tech vs Sci-Fi) | test, p above 0.999 for the right label |
| The tests catch real bugs | 32 deliberate bugs injected one at a time; all are caught now (5 slipped through at first and got new tests); source restored byte for byte |
| torch 2.11.0+cu128 is built for the RTX 5070 | `get_arch_list()` includes `sm_120` (no GPU kernel run yet) |
| GPU and driver | nvidia-smi: RTX 5070 12 GB, driver 591.86 |
| TypeSafe wire format | live docs plus 2 real API calls (raw responses in `docs/research/04-typesafe-jev-api.md`) |
| Qwen3 0.6B, 1.7B, 4B, 4B-Instruct-2507 are Apache-2.0 | Hugging Face API and LICENSE files |
| Benchmark dataset IDs load | datasets 5.0.1; `PolyAI/banking77` no longer loads, `mteb/banking77` does |
| CLI and benchmark run end to end | CPU smoke runs |

Not verified yet:

- Any CUDA kernel on the RTX 5070; the bf16 cached vs uncached gap (test tolerance 0.25 is a guess); which SDPA backend runs with a mask on Windows.
- VRAM, the best chunk size and latency for Qwen3-4B-Instruct-2507; every benchmark number.
- TypeSafe's confidence formula (undisclosed; ours is documented as ours), its error body, and whether the official SDK can point at our server.
- Whether full-label scoring beats single-letter labels on accuracy (needs the benchmark).
- Yes-bias: on Qwen3-0.6B, SST-2 as a yes/no question answered "true" 38 of 40 times. Check on the 4B model.

## Brief claims the research found wrong

1. Score on the wire is an ordered list of 2 to 10 level descriptions, not min/max. The levels are the bins; the answer is the expected level plus the distribution.
2. openjev is not a Qwen scorer: a 26B diffusion model with single-letter labels and no calibration.
3. Kev's "ECE 0.065" is a raw pre-calibration number from older checkpoints (after temperature: 0.013 to 0.042).
4. The console (Tarnlight, formerly jevscope) reads a drop box of TypeSafe-shaped records, not an arbitrary JSONL file.
5. "Existing MCP servers need a paid key" is false: several run locally with no key.
6. Sports vs Sci-Tech is not a first-token collision on Qwen3; Sci-Tech vs Sci-Fi is. Both are tested.
7. torch 2.11 is not the current release (2.14.1 is, on CUDA 13.0); the cu128 index stops at 2.11. Fine for Sunday.
8. Only full-label scoring stands out; score bins, calibration and a local MCP server exist elsewhere. Windows counts only with a recorded test run.
9. The directory lists 749 projects (330 of them on GitHub), not about 330.
10. This session ran on the Windows desktop itself, not a cloud container. It stayed CPU-only as the brief asked.

## Defaults picked (change any)

- Repo cloned from the existing private GitHub repo into a folder next to the other projects; branch `day1-scaffold`.
- Prompt: questions are shown as `q1`, `q2`, ... (caller ids never reach the model) and each is answered as `{"qk": value}`; the closing `}` ends every label.
- Confidence for choice and score: `(K * p_max - 1) / (K - 1)`, our own formula.
- Event `ts` is an RFC 3339 UTC string; `id` is one per call.
- SST-2 is a yes/no question in the benchmark.
- Package version 0.0.1, author "Mireth AI".

## Open questions for the founder

1. Console: ship `mirethstm console` in this repo (it is on the must-ship list) and add the Tarnlight tap as a small extra? Jev: both 0.49, Tarnlight tap only 0.40, new console only 0.11; Jev gives the tap alone 0.13 of meeting the promise. Recommendation: both, console first.
2. Score questions: level list only for v0.1, numeric min/max later? Jev 1.00. Recommendation: yes.
3. torch: stay on 2.11 cu128 through Sunday, move to cu130 after? Jev 1.00. Recommendation: yes.
4. The README drops "first" and "only" claims and links similar projects by name. Does that fit what was already announced?
5. Make the repo public on Sunday after your review? It is private now.
6. If the yes-bias holds on the 4B model, switch SST-2 to a positive/negative choice in the benchmark and note the risk in the README?

## Windows setup

On this desktop the environment already exists (`.venv` in the repo: Python 3.11, torch 2.11.0+cu128, transformers 5.18.0, Qwen3-0.6B cached). Skip to "GPU check".

Fresh machine, PowerShell, from the folder that should hold the repo (uv commands verified on this desktop):

```powershell
git clone https://github.com/TheChyeahhh/mirethSTM1.git
cd mirethSTM1
git checkout day1-scaffold
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -e ".[dev,bench]"
```

Without uv: `py -3.11 -m venv .venv`, then `.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128`, then `.venv\Scripts\python.exe -m pip install -e ".[dev,bench]"`.

### GPU check (sm_120 plus a real kernel)

```powershell
.\.venv\Scripts\python.exe -c "import torch; a = torch.cuda.get_arch_list(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), a); assert 'sm_120' in a; x = torch.randn(2048, 2048, device='cuda'); print('cuda matmul ok', float((x @ x).abs().mean()))"
```

Expect `2.11.0+cu128 12.8 NVIDIA GeForce RTX 5070 [... 'sm_120']` and `cuda matmul ok`.

### First GPU test

The model tests rerun on the GPU in bfloat16:

```powershell
$env:MIRETHSTM_TEST_DEVICE = "cuda"; $env:HF_HUB_OFFLINE = "1"
.\.venv\Scripts\python.exe -m pytest -m model -s -q
```

Write down the printed `max abs diff cached vs uncached` (bf16) in `docs/research/05-environment.md`, then tighten the bf16 tolerance in `tests/test_engine.py`.

### Then the default model (about 8 GB download)

```powershell
Remove-Item Env:HF_HUB_OFFLINE
.\.venv\Scripts\python.exe -c "from mirethstm import Engine; import torch; e = Engine.load('Qwen/Qwen3-4B-Instruct-2507'); q = {'team': {'type': 'choice', 'instructions': 'Which team should handle this ticket?', 'criteria': {'billing': None, 'bug': None, 'other': None}}}; r = e.decide('I was charged twice this month.', q); print(r['answers'], round(r['latency_ms']), 'ms', 'peak GB', round(torch.cuda.max_memory_allocated() / 1e9, 2))"
```

The second run of `decide` in the same process gives the real latency (the first includes warmup). Next steps are Thursday in `ROADMAP.md`.
