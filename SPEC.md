# MirethSTM1 specification (v0.1)

Status: draft, 2026-09-30. Sections marked FROZEN change only with the founder's sign-off.
Research behind every decision here: `docs/research/` (start with `00-index.md`).

## 1. What it is

MirethSTM1 answers typed questions about a context with a probability for every allowed answer. It does not generate text. It prefills the context and all questions once into a KV cache, then scores every allowed answer of every question as a whole label (all of the label's tokens), reusing that cache.

It is a free, Jev-style approximation of TypeSafe's Jev, not an equivalent. Jev is a trained model; MirethSTM1 is an inference method on a stock open model (Qwen3). Say this plainly wherever the project is described.

## 2. Wire format (TypeSafe compatible)

Matches the live TypeSafe API as captured on 2026-09-30 (`docs/research/04-typesafe-jev-api.md`).

### 2.1 Request

```json
{
  "model": "any string",
  "state": "a string, or any JSON object or array of text",
  "questions": {
    "wants_refund": {"type": "noul", "instructions": "Is the customer asking for money back?",
                     "criteria": {"true": "Asks for a refund", "false": "Does not"}},
    "category": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": "Charges, invoices, refunds", "bug": "Crashes", "other": null}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["No time pressure", "Can wait days", "Needs attention today", "Critical outage"]}
  }
}
```

| Question type | Fields | Rules |
| --- | --- | --- |
| `noul` | `instructions`; optional `criteria: {"true": desc, "false": desc}` (either key may be missing) | Answer is P(true) |
| `choice` | `instructions`; `criteria`: object, option name to description (description may be `null`) | 1 to 255 options. Option names are the labels, used verbatim |
| `score` | `instructions`; `criteria`: ordered array of level descriptions | 1 to 10 levels (TypeSafe docs say 2 to 10; the live API accepts 1). Levels are addressed by 0-based index |

- `instructions` and descriptions may be strings, objects or arrays. Non-strings are rendered as compact JSON.
- Question ids are for code only. They never reach the model (the prompt uses `q1`, `q2`, ... in request order).
- `model` is accepted and ignored for routing: the engine answers with the model it has loaded.
- Validation failures raise `mirethstm.SchemaError` (HTTP 422): state not a string, object or array; unknown type; missing or wrong-typed fields; noul `criteria` keys other than `"true"`/`"false"`; empty option name; 0 or more than 255 options; 0 or more than 10 levels; a `null` score level; empty `questions`. A `null` noul description counts as absent.

### 2.2 Response

```json
{
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "answers": {
    "wants_refund": {"type": "noul", "noul": 0.58},
    "category": {"type": "choice", "choice": "billing", "confidence": 0.52,
                 "probabilities": {"billing": 0.68, "bug": 0.04, "other": 0.28}},
    "urgency": {"type": "score", "score": 1.97, "confidence": 0.93,
                "legend": {"0": "No time pressure", "1": "Can wait days", "2": "Needs attention today", "3": "Critical outage"},
                "probabilities": {"0": 0.0, "1": 0.04, "2": 0.95, "3": 0.01}}
  },
  "usage": {"input_tokens": 470, "output_tokens": 0}
}
```

- `noul`: P(true) after temperature. No `confidence` (TypeSafe has none).
- `choice`: `choice` is the most likely option (ties go to the earlier option in the request); `probabilities` has every option, in request order, summing to 1.
- `score`: `score` = sum over levels of index times probability (range 0 to n-1); `legend` and `probabilities` are keyed by the index as a string. Non-string level descriptions appear in `legend` as compact JSON.
- `confidence` (choice and score) is OUR formula, not TypeSafe's (theirs is undisclosed): `(K * p_max - 1) / (K - 1)` for K >= 2 options or levels, and `1.0` when K = 1. It matches TypeSafe's one published three-option example; it is documented as ours.
- `usage.input_tokens` = prefill tokens plus every token fed for each label (its question's suffix plus its candidate). `usage.output_tokens` = 0 (nothing is generated).
- Precision: the Python API and `POST /v1/decide` return full floats. The compatibility route `POST /v1/systemone` rounds every number to 2 decimals, as TypeSafe does.

## 3. Scoring method (normative)

### 3.1 Prompt

Rendered with the model's chat template, `add_generation_prompt=True`, `enable_thinking=False` (Qwen3 hybrid models then end the prefix with an empty think block; 2507 Instruct ignores the flag).

System message, exactly:

```
You read a state and answer questions about it. You answer one question per reply, as a JSON object that holds only that question's key.
```

User message (questions first, so the part that repeats across calls can be cached, see 3.6):

```
Questions:
<question blocks, separated by one blank line>

State:
<state>

Answer one question per reply as a JSON object with only that question's key, for example {"q1": true}. A yes/no question takes true or false. A choice question takes one option name as a JSON string, exactly as written. A score question takes one level number.
```

- `<state>`: a string verbatim; an object or array as `json.dumps(state, ensure_ascii=False, indent=2)`.
- Question blocks, where `k` is the 1-based position of the question in the request:

```
qk (yes/no): <instructions>
  true: <criteria.true>          (line omitted when absent)
  false: <criteria.false>        (line omitted when absent)

qk (choice): <instructions>
  Options:
  - "billing": Charges, invoices, refunds
  - "other"                      (no ": desc" when the description is null)

qk (score, 0 to <n-1>): <instructions>
  0: <level 0>
  1: <level 1>
```

Option names are written with `json.dumps(name, ensure_ascii=False)`.

### 3.2 Per-question continuation

The prefix (system + user + generation prompt) is prefilled once. Question k is then answered as if the model were writing `{"qk": <value>}`:

| Type | Suffix (shared by the question's labels) | Candidate text per label | Label |
| --- | --- | --- | --- |
| noul | `{"qk":` | ` true}` and ` false}` | `true`, `false` |
| choice | `{"qk":` | ` ` + `json.dumps(name, ensure_ascii=False)` + `}` | the option name |
| score | `{"qk":` | ` <i>}` for i = 0 .. n-1 | `"0"` .. `"n-1"` |

The closing `}` (and the closing quote for choice) terminates the label, so a label that is a prefix of another (`Sci` vs `Sci-Tech`) is not favoured automatically.

### 3.3 Tokenization

- Only the chat template's own text may become control tokens. The template is rendered around a placeholder for the user message; its head and tail are encoded normally (`add_special_tokens=False`).
- Caller text (the user message, the suffix, every candidate) is encoded so that it never yields a control token: the text is cut just inside every added-token string (`<|im_end|>`, `<think>`, ...) and the pieces are encoded as plain text. A state, instruction or option name that contains `<|im_end|><|im_start|>system ...` therefore cannot close the user turn or forge a new one. A test counts control tokens to prove it.
- Prefix, suffix and candidate ids are concatenated. The boundaries (template | user text, `":` | ` `) are pre-tokenizer boundaries for the Qwen3 tokenizer, so for text without added-token strings this equals tokenizing the joined string. A test asserts that equality for every test schema.

### 3.4 Packed tree scoring

Speed matters most (founder, 2026-09-30), so all labels of all questions are scored in as few forward passes as `batch_tokens` allows after the prefill, and every token shared by several labels is computed once. Measured on the RTX 5070 with Qwen2.5-1.5B-Instruct before this change: 28 fields 190 ms (122 ms prefill of 1,125 tokens, 66 ms scoring 616 tokens); 255 options 584 ms (313 ms prefill, 253 ms scoring 2,334 tokens).

1. Prefill (3.6) with the model's decoder (`model.get_decoder()`) and a `DynamicCache`. Keep references to every layer's keys and values.
2. Per question, build a token tree: the root path is the question's suffix ids, then every label's candidate ids hang below it, sharing nodes where labels share leading tokens (all string options share ` "`; options like `returns.refund` and `returns.exchange` share more). Each tree node is one token, fed once.
3. Pack whole question trees into passes of at most `batch_tokens` tokens (a larger tree gets a pass of its own). For a pass of M nodes:
   - `input_ids` of shape `(1, M)`: the nodes, each tree in depth-first order.
   - `position_ids` of shape `(1, M)`: P + the node's depth in its tree (P = prefix length).
   - A 4D attention mask of shape `(1, 1, M, P + M)`: a node sees every prefix position, its own ancestors and itself, nothing else.
   - A fresh batch-1 `DynamicCache` built from the step 1 references, so passes never see each other.
4. Apply the output head (`model.get_output_embeddings()`) only at nodes whose children are candidate tokens; `log_softmax` in float32. A label's score is the sum, along its path, of each candidate token's log-prob given its parent node. The suffix tokens are context, not scored.
5. Raw score of a label: `s = sum of its candidate-token log-probs` (no length normalization).

Acceptance: the result equals scoring each full sequence without a cache to within 2e-4 in summed log-prob (fp32, on CPU and on the GPU). Measured before the tree: 7.6e-5 with a 280-token prefix on CPU, 4.6e-5 on the RTX 5070 in fp32. In bf16 the uncached path alone is 0.3 to 0.4 off its own fp32 result, so bf16 is only checked for staying within that noise.

### 3.5 Probabilities

Per question: `p = softmax(s / T)` over that question's labels, T > 0 (default from section 9, else 1.0). A question with one label (1-option choice, 1-level score) gets p = 1 without running the model for it.

### 3.6 Question-part cache

The prompt splits into a part that only depends on the questions (template head, system message, `Questions:` and every block, up to and including `State:` and its line break) and a part that depends on the state (the state, the closing rule, the template tail). The engine keeps the computed keys and values of the question part for recent question sets (keyed by its token ids; bounded so it cannot crowd the model out of a 12 GB card), so a repeated question set only prefills the state part. The split point must be a pre-tokenizer boundary for typical states, so cached and uncached calls feed the same ids; a test checks that the scores are the same with and without a cache hit. The benchmark measures accuracy with this question-first order against the earlier state-first order before release.

## 4. Python API

```python
from mirethstm import Engine, SchemaError

engine = Engine.load(
    "Qwen/Qwen3-4B-Instruct-2507",
    device=None,        # "cuda" if available, else "cpu"
    dtype=None,         # bfloat16 on cuda, float32 on cpu
    batch_tokens=2048,  # candidate tokens per scoring pass (section 3.4); bounds memory
    temperature=None,   # None: the shipped default for this model (section 9), else 1.0
    event_log=None,     # path: append the frozen JSONL events (section 8)
    tarnlight=True,     # feed Tarnlight when its drop folder exists (section 10.3)
)
result = engine.decide(context, schema)   # context: str | dict | list (TypeSafe state); schema: TypeSafe questions map
raw = engine.score(context, schema)       # {question_id: {label: s}} raw summed log-probs, before temperature
```

`decide` returns the section 2.2 body plus two extra keys: `id` (32 hex chars, one per call) and `latency_ms` (float, wall time of the call).

## 5. CLI

```
mirethstm decide --schema s.json [--model ID] [--temperature T] [--device D] [--batch-tokens N] [--log events.jsonl] [--state-json] [--no-tarnlight] < ctx.txt
mirethstm console [--model ID] [--device D] [--host 127.0.0.1] [--port 8766] [--no-tarnlight]
```

- stdin is the state, read as UTF-8 exactly as given (a final newline is part of the state). By default it is a plain string; with `--state-json` it is parsed as JSON.
- `s.json` holds a TypeSafe `questions` map.
- Prints the `decide` result as JSON on stdout (non-ASCII escaped, so any Windows console can print it).
- Exit code 2, message on stderr, before any model loads: `SchemaError`, unreadable or invalid JSON input, non-UTF-8 stdin, `--batch-tokens` below 1, `--temperature` not above 0.
- In Windows PowerShell 5.1, pipe-free: `cmd /c "mirethstm decide --schema s.json < ctx.txt"` (a PowerShell pipe re-encodes the text and corrupts non-ASCII characters).

## 6. HTTP API (on the console's server)

The console's standard-library server (section 10) also answers the API, so there is one local server and no FastAPI dependency. It runs on the user's own machine, bound to 127.0.0.1 by default; it never contacts TypeSafe or any other service, needs no key and costs nothing (founder question 2026-09-30: nothing is linked to any account).

| Route | Behaviour |
| --- | --- |
| `POST /v1/systemone` | TypeSafe drop-in: section 2 shapes, numbers rounded to 2 decimals, `id` in header `x-request-id`, latency in `server-timing`. The request's `model` is accepted and ignored; the answer names the loaded model |
| `POST /v1/decide` | Same request; full precision; body includes `id` and `latency_ms` |
| `GET /v1/models` | The loaded model and the approved list (section 11) |

Requests share the one model: an API request waits for the current run (up to 60 s, then 529). Errors: `{"error": {"type": "...", "message": "..."}}` (our own shape; TypeSafe's is undocumented): 422 validation, 413 body too large, 529 busy or loading. Any Authorization header is accepted and ignored, so existing TypeSafe clients work unchanged when pointed at `http://127.0.0.1:8766`.

## 7. MCP server (cut first)

`mirethstm mcp`: stdio MCP server with one tool, `decide(context, schema)`, returning the `decide` result. Local, no API key. Do not claim that other MCP servers need a paid key: several local ones exist (`docs/research/09-landscape-and-differentiators.md`).

## 8. Event log (FROZEN)

One JSON object per line, one line per question per `decide` call, UTF-8, `\n`-terminated, appended. All lines of one call are written with a single `write` so a tailing reader never sees half a call. Keys, exactly these and in this order:

| Key | Type | Meaning |
| --- | --- | --- |
| `ts` | string | When the call finished: UTC, RFC 3339 with milliseconds and `Z`, e.g. `2026-10-02T14:03:22.123Z` |
| `id` | string | Call id, 32 lowercase hex chars, shared by every line of the call; `(id, field)` is unique |
| `field` | string | The question id as the caller named it |
| `label` | string | Most likely label: `"true"`/`"false"` (noul), the option name (choice), the level index as a string (score) |
| `p` | number | Probability of `label` after temperature (the top probability) |
| `probs` | object | Every label to its probability after temperature, in request order |
| `T` | number | Temperature used |
| `latency_ms` | number | Wall time of the whole call (same on every line of the call) |
| `model` | string | Model id as loaded |

Example:

```json
{"ts": "2026-10-02T14:03:22.123Z", "id": "3f2a...", "field": "category", "label": "billing", "p": 0.68, "probs": {"billing": 0.68, "bug": 0.04, "other": 0.28}, "T": 1.0, "latency_ms": 212.4, "model": "Qwen/Qwen3-4B-Instruct-2507"}
```

Any new key needs a new schema version and the founder's sign-off.

## 9. Calibration

- `mirethstm.calibration.fit_temperature(logits, labels) -> float`: one scalar T minimizing mean NLL of `softmax(s / T)` over a calibration set. `logits` is a list of 1-D arrays (questions may have different label counts), `labels` the index of the true label. Bounded golden-section search on log T over [0.05, 20]; raises `ValueError` if the optimum sits at a bound.
- `mirethstm.calibration.ece(confidences, correct, n_bins=15) -> float`: top-label expected calibration error, equal-width bins over [0, 1] (Guo et al. 2017). The benchmark reports 15-bin ECE as the headline and 10-bin ECE for comparison with Kev.
- Shipped defaults: `mirethstm.calibration.DEFAULT_TEMPERATURES = {model_id: T}`, filled from the benchmark's calibration splits before release (empty until then, so T = 1.0).
- Accuracy and macro-F1 do not change with T (argmax is invariant); a test enforces it.

## 10. Console (founder ruling 2026-09-30: a race view like the original demo)

### 10.1 Race view

`mirethstm console [--model ID] [--device D] [--host 127.0.0.1] [--port 8766] [--no-tarnlight]`. Python standard library server (`http.server`, threaded) plus one page whose HTML, CSS and JS ship in the package. No CDN, no build step, works offline. The live feed of past decisions is Tarnlight's job (10.3), not this page's.

Look and feel follows the original demo by Harsha Gundala (parallel vs normal inference, see `docs/research/02-rlcd-and-ports.md`); our own HTML, CSS, JS and scenario text, nothing copied (founder rule: take the shape, never the expression).

- Light page. Top bar: scenario picker, model picker, a "Run comparison" button (shows "Running..." while busy).
- Under it, after a run: a green pill, `5.6x faster · 296 ms vs 1670 ms`.
- Two cards side by side (stacked on a narrow screen):
  - Left, "MirethSTM1 (<model>)": a green milliseconds badge; the answers as monospace JSON with blue keys, one line per field, all at once: noul and choice `{"value": ..., "prob": ...}`, score `{"value": <top level>, "prob": ..., "score": <expected level>}`.
  - Right, "Normal generation (<model>)": a grey milliseconds badge that ticks live; the model's own JSON appearing token by token as it is generated; when done, a red badge "N fields hallucinated" (and "invalid JSON" if it does not parse).
- The state text of the chosen scenario is shown in an editable box, so a run can use any text.
- Order: the MirethSTM1 run first, then the normal generation; each badge is that run's own wall time. One run at a time (one GPU).
- Model picker: Qwen/Qwen2.5-1.5B-Instruct (default: the original demo's model, match it first), Qwen/Qwen3-1.7B, Qwen/Qwen3-4B-Instruct-2507, Qwen/Qwen3-0.6B, plus `--model` if it is another id. Switching frees the old model before loading the new one; the page shows "Loading <model>...".
- Built-in scenarios, our own neutral text: support ticket triage (28 fields), code change security review (28 fields), incident triage with score questions (about 20 fields), a 255-option request router (4 fields, one with 255 options).
- Every MirethSTM1 run goes through `Engine.decide`, so it feeds Tarnlight (10.3) and the event log like any other call.

### 10.2 Normal-generation baseline

`mirethstm.baseline.generate(engine, context, schema, on_text=None) -> dict`, shared by the console and the benchmark:

- Same model and question blocks (SPEC 3.1), with its own system prompt and closing rule asking for ONE JSON object holding every key `q1`..`qn` in order (the scorer's "one question per reply" wording would contradict that and handicap the baseline). Rendered with `prefix_ids(tokenizer, state, questions, rules=BASELINE_RULES, system=BASELINE_SYSTEM)`, so caller text is encoded safely (3.3).
- The right card shows the caller's field ids as keys, with the model's own key (`q1`..`qn`) in a faint gutter, so the two cards read across line by line.
- Greedy decoding, stop at end of turn, `max_new_tokens` bounded from the questions (enough for the longest label of each question plus JSON syntax).
- `on_text(chunk)` is called as text is generated (streaming).
- Returns `text` (the raw output), `answers` (question id to the parsed value or `None`), `valid_json` (bool), `hallucinated` (ids whose value is not an allowed answer: noul not a bool, choice not an exact option name, score not an integer level in range), `missing` (ids absent from the output), `latency_ms`, `output_tokens`.

### 10.3 Tarnlight tap

Feeds Tarnlight (the founder's public live console, `docs/research/08-console-integration.md`) without depending on it: on by default (`tarnlight=True`, `--no-tarnlight` turns it off), and it does nothing unless `~/.tarnlight/inbox/` already exists (Tarnlight creates it; we never do).

- After each `decide`, append one line to `~/.tarnlight/inbox/mirethstm-<UTC YYYY-MM-DDTHH>.jsonl`: `{"v": 1, "ts": <epoch seconds>, "source": "mirethstm", "request_id": <id>, "request": {"model", "state", "questions"}, "response": <the /v1/systemone body: model, answers, usage, numbers rounded to 2 decimals>, "latency_ms": ..., "status": 200, "cost_est_micro": 0}`.
- One append write per call (Tarnlight has no dedup), UTF-8, `\n`. Skips when the disk has under 1 GB free. Never raises: a failed write must not fail the decision.

## 11. Models and environment

Any Hugging Face causal LM with a chat template plugs in: `Engine.load("<hub id or local path>")`, `--model`, or the console's model picker. The engine uses only `get_decoder()`, `get_output_embeddings()` and the tokenizer's chat template. Tested families: Qwen2.5 and Qwen3 (full model test suite passes on Qwen3-0.6B and Qwen2.5-1.5B-Instruct). `Engine.load` refuses, with a clear error, a model it would score wrongly: one with sliding-window or other non-full attention layers, or one whose forward changes the logits after the output head (softcapping, scaling).

### 11.1 Approved models (founder, 2026-09-30: "switch models from a list of approved models")

`mirethstm/models.py` holds the approved list: model id, display name, license, size, role and the measured numbers (speed on the RTX 5070, benchmark accuracy and ECE, fitted temperature). The console's model picker and `GET /v1/models` show only approved models with those numbers; `Engine.load` and `--model` still take any id (unvetted, at the user's own risk). A model is approved only when all of these hold:

1. Its license allows free commercial use (Apache-2.0 or MIT); weights are public, not gated.
2. It fits the 12 GB card with room for the cache (bf16 weights well under 10 GB).
3. `Engine.load` accepts it (full attention, no logit post-processing) and the full model test suite passes on the GPU.
4. Its speed and its accuracy and calibration on the benchmark are measured and written into the list.

Candidates to evaluate: Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B-Instruct-2507, microsoft/Phi-4-mini-instruct (MIT), HuggingFaceTB/SmolLM3-3B, ibm-granite/granite-3.3-2b-instruct. Excluded up front: Qwen2.5-3B (non-commercial), gated or custom-license models (Llama, Gemma), sliding-window models.

### 11.2 Current roles

| Role | Model | License | Notes |
| --- | --- | --- | --- |
| Match first | Qwen/Qwen2.5-1.5B-Instruct | Apache-2.0 | The original demo's model; console default. Match its speed, then compare |
| Default SDK/CLI | Qwen/Qwen3-4B-Instruct-2507 | Apache-2.0 | Non-thinking only |
| Fast | Qwen/Qwen3-1.7B | Apache-2.0 | Hybrid thinking; `enable_thinking=False` |
| CPU tests | Qwen/Qwen3-0.6B | Apache-2.0 | fp32 on CPU (bf16 on this CPU is about 500x slower) |
| Not used | Qwen2.5-3B | Qwen Research License (non-commercial) | |
| Deferred | Qwen3.5 small models | Apache-2.0 | Hybrid linear attention; the plain KV cache does not apply |

- Tested stack: Python 3.11, torch 2.11.0+cu128 (arch list includes `sm_120`), transformers 5.18.0, `attn_implementation="sdpa"`. No flash-attn, no Triton, no torch.compile.
- The cu128 index stops at torch 2.11. torch 2.14.x on cu130 is the path after launch.

## 12. Out of scope for v0.1

vLLM, llama.cpp and MLX backends; fine-tuning; Qwen3.5 hybrid models; a numeric min/max score range (the level list covers Yelp 1 to 5); length-normalized scoring (a benchmark option later, not the default); multi-GPU.
