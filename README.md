# MirethSTM1

A free, open-source decision engine by Mireth AI: ask typed questions about a text and get a probability for every allowed answer, scored by a local model instead of generated.

## What it is, and what it is not

MirethSTM1 is a free, Apache-2.0, Jev-style approximation of TypeSafe's Jev, not an equivalent. Jev is a model trained for calibrated decisions. MirethSTM1 is an inference method on stock open models, so its probabilities are the base model's own, rescaled by a temperature fitted on held-out labels.

What it does: you plug in a Hugging Face chat model; it reads the real text of every allowed answer (not a single letter or first token) for all questions in one packed pass; it returns an expected value and a distribution for score questions; it ships a console that races it against the same model writing the JSON itself; and it publishes accuracy, calibration and speed numbers with the scripts that produced them, measured on Windows with an NVIDIA RTX 5070.

What it is not: the only open engine, the fastest, or the first to read label text. [Laya](https://github.com/NandhaKishorM/laya) trains small encoders that answer in milliseconds, [jevmlx](https://github.com/bnsd55/jevmlx) also scores label text, and [simple-jev](https://github.com/featherless-ai/simple-jev), [AnyJev](https://github.com/nokia-applied-research/AnyJev), [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev), [open-alternative-jev](https://github.com/ikermoel/open-alternative-jev), [jev-style](https://github.com/lawrence3699/jev-style), [Verdict](https://github.com/Heman10x-NGU/Verdict-open-jev) and [von](https://github.com/wfzyx/von) cover overlapping ground. Try them and compare on your own labels before trusting any threshold.

## Status

Pre-release; v0.1 ships 2026-10-04. The engine, the `decide` command, the console and the calibration code pass their tests on CPU (Qwen3-0.6B and Qwen2.5-1.5B-Instruct). The benchmark below ran on the GPU on 2026-10-01; the GPU test results per model are in [docs/gpu-notes.md](docs/gpu-notes.md). The contract is [SPEC.md](SPEC.md).

## How it works

1. One forward pass reads the text and all questions and, in the same pass, scores every allowed answer of every question as a whole label (the summed log-probs of all its tokens).
2. Answers that share leading tokens share them in a token tree, so each shared token is computed once.
3. Turn each question's label scores into probabilities with a softmax.
4. Divide the scores by a temperature T before the softmax; T is fitted on held-out labels. Each approved model ships its fitted T (table below); any other model uses 1.0.

Nothing is generated, so `output_tokens` is always 0.

## Install (Windows, NVIDIA)

You need Python 3.10 or newer (tested on 3.11) and an NVIDIA driver recent enough for CUDA 12.8 (572.61 or newer). From the repository folder, in cmd.exe or PowerShell:

```
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

Check that PyTorch sees the GPU. This should print `True`; on an RTX 50-series card the list must include `sm_120`:

```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"
```

Model weights are not bundled. They download from Hugging Face on first use (about 8 GB for the default model). Without a GPU, pass `device="cpu"` (or `--device cpu`); it runs in float32 and is slow for the default model.

Tests: `pip install -e ".[dev]"`, then `pytest`. The model tests use Qwen3-0.6B from the local Hugging Face cache and are skipped when it is missing.

## Usage

Questions use TypeSafe's format, so code written for its API maps over directly:

| Type | `criteria` | Answer |
| --- | --- | --- |
| `noul` | optional `{"true": ..., "false": ...}` | `noul`: P(true) |
| `choice` | option name to description (or `null`), 1 to 255 options | `choice`, `confidence`, `probabilities` |
| `score` | ordered list of level descriptions, 1 to 10 levels | `score` (expected level index), `confidence`, `legend`, `probabilities` |

The text (TypeSafe's `state`) is a string or any JSON object or array.

### Python

```python
from mirethstm import Engine

engine = Engine.load("Qwen/Qwen3-4B-Instruct-2507")

questions = {
    "wants_refund": {"type": "noul", "instructions": "Is the customer asking for money back?",
                     "criteria": {"true": "Asks for a refund", "false": "Does not"}},
    "category": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": "Charges, invoices, refunds", "bug": "Crashes", "other": None}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["No time pressure", "Can wait days", "Needs attention today", "Critical outage"]},
}

text = "I was charged twice for my subscription this month. Please refund one of the charges."
result = engine.decide(text, questions)
print(result["answers"]["category"]["choice"])  # billing

raw = engine.score(text, questions)  # {question: {label: summed log-prob}}, before temperature
```

`Engine.load` also takes `device`, `dtype`, `batch_tokens`, `temperature`, `event_log` and `tarnlight` (see SPEC.md section 4). Any Hugging Face chat model id works in place of the default; models the engine would score wrongly (sliding-window attention, logits changed after the output head) are refused with a clear error.

### Command line

Put the questions map in `s.json` and the text in `ctx.txt`. In cmd.exe or Git Bash:

```
mirethstm decide --schema s.json < ctx.txt
```

PowerShell has no `<`, and piping the file re-encodes it (Windows PowerShell turns accented and other non-ASCII characters into `?` or garbage), so let cmd.exe do the redirect:

```
cmd /c "mirethstm decide --schema s.json < ctx.txt"
```

Options: `--model`, `--temperature`, `--device`, `--batch-tokens`, `--state-json` (read stdin as JSON), `--log events.jsonl` (append one JSON line per question) and `--no-tarnlight`. A bad questions map exits with code 2.

The state is stdin exactly as read, so a final newline in `ctx.txt` is part of it and moves the numbers slightly. Output of the example above with `--model Qwen/Qwen3-0.6B --device cpu --temperature 1` and a `ctx.txt` that ends in one newline (numbers shortened here). This small model at T = 1 is overconfident, which is what the fitted temperature is for: without `--temperature`, its shipped T of 3.83 spreads these probabilities out.

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "answers": {
    "wants_refund": {"type": "noul", "noul": 0.9997},
    "category": {"type": "choice", "choice": "billing", "confidence": 0.999996,
                 "probabilities": {"billing": 0.999997, "bug": 5.2e-08, "other": 2.5e-06}},
    "urgency": {"type": "score", "score": 2.2533, "confidence": 0.6494,
                "legend": {"0": "No time pressure", "1": "Can wait days", "2": "Needs attention today", "3": "Critical outage"},
                "probabilities": {"0": 0.0008, "1": 0.0036, "2": 0.7370, "3": 0.2585}}
  },
  "usage": {"input_tokens": 299, "output_tokens": 0},
  "id": "7bbf7a8b6a254975a963136be7a19d1f",
  "latency_ms": 1592.3
}
```

## Console

```
mirethstm console
```

Opens a local page at http://127.0.0.1:8766 that races MirethSTM1 against the same model writing the JSON itself, side by side, like the demo that started this project. Pick a scenario and a model, edit the text if you like, and press Run comparison. The left card shows every answer with its probability at once. The right card shows the model's own JSON as it is generated, with answers outside the allowed set marked as hallucinated. A pill on top shows how many times faster MirethSTM1 was (or slower, when it was). The default model is Qwen/Qwen2.5-1.5B-Instruct, the original demo's model; switch models from the page. Options: `--model`, `--device`, `--host`, `--port`, `--no-tarnlight`.

## Tarnlight

If [Tarnlight](https://github.com/TheChyeahhh/tarnlight) is installed, every decision (from the SDK, the CLI or the console) also shows up there live. MirethSTM1 never needs it: nothing is written unless Tarnlight's drop folder already exists, and a failed write never fails a decision. Turn it off with `tarnlight=False` or `--no-tarnlight`.

## Models and licenses

The console's model picker and `GET /v1/models` offer the approved models below. A model is approved when its license allows free commercial use, it fits a 12 GB card, `Engine.load` accepts it, the model test suite passes on the GPU, and its speed, accuracy and calibration are measured (SPEC 11.1). Any other Hugging Face id still loads, unvetted.

| Model | Role | License | 28 fields | 255-option router | Accuracy | ECE at T = 1 | ECE at shipped T | Shipped T | Peak GPU memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | console default (the original demo's model) | Apache-2.0 | 98.1 ms | 220 ms | 0.713 | 0.164 | 0.094 | 2.285 | 3545 MiB |
| [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | SDK and CLI default | Apache-2.0 | 274 ms | 637 ms | 0.752 | 0.238 | 0.119 | 8.036 | 8331 MiB |
| [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | fast mode | Apache-2.0 | 110 ms | 258 ms | 0.696 | 0.268 | 0.091 | 6.750 | 3883 MiB |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | CPU tests | Apache-2.0 | 64.6 ms | 163 ms | 0.581 | 0.291 | 0.127 | 3.830 | 1733 MiB |
| [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | alternative | Apache-2.0 | 184 ms | 409 ms | 0.672 | 0.195 | 0.076 | 2.677 | 6487 MiB |

Measured on one RTX 5070 in bf16 on 2026-10-01. 28 fields and router: p50 of a warm `decide` call on the console's support-triage scenario and its 255-option router. Accuracy: MirethSTM1's mean over the five benchmark datasets below. ECE: the mean 15-bin ECE over the same datasets. Shipped T: the temperature `Engine.load` applies by default. Peak GPU memory: the largest of any MirethSTM1 run in the latency benchmark. SmolLM3-3B fails one GPU test that expects Qwen3's exact template whitespace; the engine reproduces SmolLM3's own template ([docs/gpu-notes.md](docs/gpu-notes.md)).

Evaluated and not approved:

- [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct): in bf16 its scores drift past the GPU test band (fp32 passes), so it was not benchmarked.
- [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) (MIT): `Engine.load` refuses it, because its config declares sliding-window attention.
- [ibm-granite/granite-3.3-2b-instruct](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct): `Engine.load` refuses it, because its forward rescales the logits after the output head.

Weights are not bundled; each downloads under its own license. Qwen2.5-3B is under the Qwen Research License (non-commercial only), so it is not used.

## Benchmark

Full tables with confidence intervals, per-dataset temperatures, reliability diagrams and every arm: [docs/benchmark.md](docs/benchmark.md). The GPU setup, the attention kernels and a bf16, fp16 and fp32 comparison: [docs/gpu-notes.md](docs/gpu-notes.md). Everything below is our own run on one RTX 5070 in bf16, 2026-10-01.

Five tasks, each asked as one TypeSafe question: AG News (`fancyzhx/ag_news`, choice of 4 topics), Banking77 (`mteb/banking77`, choice of 77 intents), SST-2 (`stanfordnlp/sst2` validation, once as yes/no and once as a two-option choice) and Yelp (`Yelp/yelp_review_full`, score with 5 levels). Each model answers them three ways, with the same question text:

- MirethSTM1: every token of every allowed answer, scored in one pass.
- First token: only the first token of each answer's name, the original demo's method (our reimplementation).
- Normal generation: the model writes the JSON itself; an invalid, made-up or missing answer counts as wrong.

### Accuracy and calibration

Mean over the five tasks. ECE is top-label with 15 equal-width bins (Guo et al. 2017), at T = 1 and at the shipped T, one scalar per model fitted on separate calibration rows.

| Model | MirethSTM1 | First token | Normal generation (valid answers) | Banking77: MirethSTM1 / first token | ECE at T = 1 | ECE at shipped T |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-1.5B-Instruct | 0.713 | 0.670 | 0.555 (0.850) | 0.545 / 0.323 | 0.164 | 0.094 |
| Qwen3-4B-Instruct-2507 | 0.752 | 0.692 | 0.611 (0.849) | 0.670 / 0.380 | 0.238 | 0.119 |
| Qwen3-1.7B | 0.696 | 0.660 | 0.515 (0.758) | 0.505 / 0.327 | 0.268 | 0.091 |
| SmolLM3-3B | 0.672 | 0.657 | 0.572 (0.846) | 0.266 / 0.168 | 0.195 | 0.076 |
| Qwen3-0.6B | 0.581 | 0.540 | 0.008 (0.008) | 0.478 / 0.286 | 0.291 | 0.127 |

- Rows: 1,000 evaluation rows per task (SST-2: all 872) for the first three models and 500 for SmolLM3-3B and Qwen3-0.6B; normal generation ran 300 rows per task (100 for the last two). All five models share 500 rows per task, and the ranking holds on them.
- Reading the whole answer matters when answers share their first token. On Banking77 about 52 to 57 of the 77 intents share a first token with another intent, so the first-token method falls back to a tie rule. On the other four tasks the two methods are within about one point of each other, in either direction.
- Normal generation is checked strictly. Qwen3 models often write `"true"` as a string, which counts as invalid (Qwen3-4B-Instruct-2507 gets 0.250 on SST-2 yes/no for that reason), and Qwen3-0.6B rarely finishes its JSON within the token budget.
- At T = 1 every model is overconfident. One fitted T per model removes most of it, not all: at the shipped T, Yelp keeps an ECE of 0.179 on Qwen2.5-1.5B-Instruct and 0.317 on Qwen3-4B-Instruct-2507, and on SST-2 yes/no the shipped T moves Qwen2.5-1.5B-Instruct from 0.044 to 0.090.
- Harder questions: on the 231 public items of JevBench (fetched from its repository and scored by us, not an official JevBench score), the hard tier stays close to its chance level of 0.336 for every model except Qwen3-4B-Instruct-2507, which reaches 0.477.

### Speed

p50 wall time of one `decide` call, warm, batch 1, bf16, RTX 5070. Normal generation is the same model writing the same answers as JSON.

| Schema | Original demo, as published (M4 Max, 4-bit) | Qwen2.5-1.5B-Instruct | Same model, normal generation | Qwen3-4B-Instruct-2507 | Qwen3-1.7B | SmolLM3-3B | Qwen3-0.6B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Small: 4 fields (demo), 5 fields (ours) | 75 ms | 36.5 ms | 1470 ms | 72.0 ms | 44.7 ms | 52.5 ms | 45.7 ms |
| 28 fields | 270 ms | 98.1 ms | 9704 ms | 274 ms | 110 ms | 184 ms | 64.6 ms |
| 255 options | 89 ms | 220 ms | 1649 ms | 637 ms | 258 ms | 409 ms | 163 ms |

The demo's numbers are not like for like. They come from its README, measured on a Mac with an M4 Max on 4-bit MLX weights of Qwen2.5-1.5B-Instruct, on its own prompts, scoring only the first token of each label, with no raw logs. Ours run on an NVIDIA RTX 5070 with bf16 weights, on our own scenario texts, and score every token of every label. That is why our 255-option router is slower than the demo's figure: it reads all 255 full labels, 4,388 input tokens in one pass. Normal generation's prompt pass runs on a slower attention kernel than MirethSTM1 on this Windows build ([docs/gpu-notes.md](docs/gpu-notes.md)), so the speedups are not a pure method comparison.

To reproduce (`pip install -e ".[bench]"`; every script skips files that already exist, so an interrupted run resumes):

```
python -m bench.run_bench --model Qwen/Qwen2.5-1.5B-Instruct --arms mireth,first_token --n 1000 --n-cal 500 --out full
python -m bench.run_bench --model Qwen/Qwen2.5-1.5B-Instruct --arms baseline --n 300 --out full
python -m bench.latency --model Qwen/Qwen2.5-1.5B-Instruct --runs 30 --out full
python -m bench.public_items --model Qwen/Qwen2.5-1.5B-Instruct --out full
python -m bench.report --run full --md docs/benchmark.md --figures docs/benchmark
```

Design: [docs/research/07-benchmark-design.md](docs/research/07-benchmark-design.md).

## Credits

These projects shaped the design. They are credited as inspiration; no code was copied from them.

- [Harsha Gundala's Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD): prefill once, then score every field in parallel from the shared cache.
- [shreyansh26's Transformers port](https://huggingface.co/shreyansh26/Qwen-2.5-1B-RLCD): scoring colliding labels by their full text, in cached chunks.
- [razorback16/openjev](https://github.com/razorback16/openjev): the TypeSafe-compatible wire format.
- [jaredpalmer/kev](https://github.com/jaredpalmer/kev): the calibration metric protocol (held-out fits, ECE, NLL, Brier, bootstrap intervals).
- [Guo et al. 2017, On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599): temperature scaling and ECE.

## Trademark

Jev is a trademark of TypeSafe AI. MirethSTM1 is not affiliated with or endorsed by TypeSafe AI.

## License

Apache-2.0. See [LICENSE](LICENSE).
