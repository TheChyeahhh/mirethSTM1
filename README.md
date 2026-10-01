# MirethSTM1

A free, open-source decision engine by Mireth AI: ask typed questions about a text and get a probability for every allowed answer, scored by a local model instead of generated.

## What it is, and what it is not

MirethSTM1 is a free, Apache-2.0, Jev-style approximation of TypeSafe's Jev, not an equivalent. Jev is a model trained for calibrated decisions. MirethSTM1 is an inference method on stock open models, so its probabilities are the base model's own, rescaled by a temperature fitted on held-out labels.

What it does: you plug in a Hugging Face chat model; it reads the real text of every allowed answer (not a single letter or first token) for all questions in one packed pass; it returns an expected value and a distribution for score questions; it ships a console that races it against the same model writing the JSON itself; and it will publish accuracy, calibration and speed numbers with the scripts that produced them, tested on Windows with an NVIDIA RTX 5070.

What it is not: the only open engine, the fastest, or the first to read label text. [Laya](https://github.com/NandhaKishorM/laya) trains small encoders that answer in milliseconds, [jevmlx](https://github.com/bnsd55/jevmlx) also scores label text, and [simple-jev](https://github.com/featherless-ai/simple-jev), [AnyJev](https://github.com/nokia-applied-research/AnyJev), [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev), [open-alternative-jev](https://github.com/ikermoel/open-alternative-jev), [jev-style](https://github.com/lawrence3699/jev-style), [Verdict](https://github.com/Heman10x-NGU/Verdict-open-jev) and [von](https://github.com/wfzyx/von) cover overlapping ground. Try them and compare on your own labels before trusting any threshold.

## Status

Pre-release. The engine, the `decide` command, the console and the calibration code pass their tests on CPU (Qwen3-0.6B and Qwen2.5-1.5B-Instruct). GPU runs and benchmark numbers land before v0.1, which ships 2026-10-04. The contract is [SPEC.md](SPEC.md).

## How it works

1. Prefill the text and all questions once into a KV cache.
2. Score every allowed answer of every question as a whole label (the summed log-probs of all its tokens), all in one packed pass on top of the cached prefix.
3. Turn each question's label scores into probabilities with a softmax.
4. Divide the scores by a temperature T before the softmax; T is fitted on held-out labels (1.0 until the benchmark sets it).

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

The state is stdin exactly as read, so a final newline in `ctx.txt` is part of it and moves the numbers slightly. Output of the example above with `--model Qwen/Qwen3-0.6B --device cpu` and a `ctx.txt` that ends in one newline (numbers shortened here). This small model at T = 1 is overconfident, which is what the fitted temperature is for:

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

| Model | Role | License |
| --- | --- | --- |
| [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | console default (the original demo's model) | Apache-2.0 |
| [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | SDK and CLI default | Apache-2.0 |
| [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | fast mode | Apache-2.0 |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | CPU tests | Apache-2.0 |

Weights are not bundled; each downloads under its own license. Qwen2.5-3B is under the Qwen Research License (non-commercial only), so it is not used.

## Benchmark

Numbers land before release.

| Dataset | Question | n | Accuracy | Macro-F1 | ECE at T = 1 | Fitted T | ECE after |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AG News (`fancyzhx/ag_news`) | choice, 4 topics | | | | | | |
| Banking77 (`mteb/banking77`) | choice, 77 intents | | | | | | |
| SST-2 (`stanfordnlp/sst2`, validation) | yes/no | | | | | | |
| Yelp (`Yelp/yelp_review_full`) | score, 5 levels | | | | | | |

ECE is top-label with 15 equal-width bins (Guo et al. 2017); T is fitted on a separate calibration split. A latency table against the same model generating JSON comes with it. To try the script now: `pip install -e ".[bench]"`, then `python bench/run_bench.py --dataset ag_news --n 200 --fit`. Design: [docs/research/07-benchmark-design.md](docs/research/07-benchmark-design.md).

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
