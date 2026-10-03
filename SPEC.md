# MirethSTM1 specification (v0.1)

Status: draft, 2026-09-30. Sections marked FROZEN change only with the founder's sign-off.
Research behind every decision here: `docs/research/` (start with `00-index.md`).

## 1. What it is

MirethSTM1 answers typed questions about a context with a probability for every allowed answer. It does not generate text. One forward pass reads the context and the questions and scores every allowed answer of every question as a whole label (all of the label's tokens), as token trees in the same pass.

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

- `instructions` and descriptions may be strings, objects or arrays. Non-strings are rendered as compact JSON. `instructions` is optional for every type, as in TypeSafe's own SDK (`Noul()`, `Choice(criteria=...)`); without it the question block shows only its criteria.
- Question ids are for code only. They never reach the model (every question is `q1` in its own branch, section 3.1).
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
- `usage.input_tokens` = the shared part's tokens, plus every token of each scored question's branch, template tail and suffix (once per question), plus every label's candidate tokens. A question with one label adds nothing. `usage.output_tokens` = 0 (nothing is generated).
- Precision: the Python API and `POST /v1/decide` return full floats. The compatibility route `POST /v1/systemone` rounds every number to 2 decimals, as TypeSafe does.

## 3. Scoring method (normative)

### 3.1 Prompt: every question answered as if it were alone

Measured 2026-10-01 (Qwen2.5-1.5B-Instruct, 200 AG News articles, four yes/no questions with known answers plus the topic choice): one question per call 86 percent right; the same questions inside one 20-question call 81 percent when placed first, 58 percent when placed last and 50 percent when spread out (later yes/no questions were answered yes for every article). So the questions of a call never share a prompt. The text is read once; every question then gets a private branch that holds its own question and nothing of the others. This is also how TypeSafe describes Jev: questions of one request cannot see one another.

Rendered with the model's chat template, `add_generation_prompt=True`, `enable_thinking=False` (Qwen3 hybrid models then put an empty think block before the answer; 2507 Instruct ignores the flag).

System message, exactly:

```
You read a state and answer questions about it. You answer one question per reply, as a JSON object that holds only that question's key. A yes/no question takes true or false.
```

Call-level instructions (optional, added 2026-10-03): a caller may pass one text that every question of the call reads, such as a brief for the task. It follows the system message's own text after a blank line, inside the system message, and is encoded as plain text like the state (section 3.3), so it cannot end a turn or open a new one. Without instructions (absent, null or an empty string) the prompt is byte for byte the prompt above. The instructions belong to the shared part, so every question reads them and each question still has a private branch.

```
You read a state and answer questions about it. ... A yes/no question takes true or false.

<instructions>
```

Shared part, read once (the template's head with that system message, and the start of the user message):

```
State:
<state>

```

Private branch of each question (the rest of the user message, then the template's tail, then the answer):

```
Question:
<question block>
```

- The only answer rule is the yes/no sentence in the system message; a branch holds only its question. Measured (bf16, one question per call, three row samples of about 900 rows per dataset on four datasets plus the yes/no probe, three models): against repeating a full rule after every question it is within noise on Qwen3-4B-Instruct-2507 (-0.003, 95 percent interval -0.007 to +0.003) and better on the two small models (Qwen2.5-1.5B-Instruct 0.712 against 0.702, Qwen3-1.7B 0.724 against 0.697), and it needs about half the tokens for 28 questions. Known cost, accepted for speed: Qwen3-4B-Instruct-2507 is 1.6 points lower on the yes/no probe than with a rule and example right before each answer, and Qwen2.5-1.5B-Instruct loses 5 points on Banking77 while gaining 10 on Yelp (`docs/benchmark.md`). Eight wordings were measured in all.
- `<state>`: a string verbatim; an object or array as `json.dumps(state, ensure_ascii=False, indent=2)`.
- Question block; every branch uses the key `q1`, since each question is alone in its branch (caller ids never reach the model):

```
q1 (yes/no): <instructions>
  true: <criteria.true>          (line omitted when absent)
  false: <criteria.false>        (line omitted when absent)

q1 (choice): <instructions>
  Options:
  - "billing": Charges, invoices, refunds
  - "other"                      (no ": desc" when the description is null)

q1 (score, 0 to <n-1>): <instructions>
  0: <level 0>
  1: <level 1>
```

Option names are written with `json.dumps(name, ensure_ascii=False)`.

Independence (tested): the answer to a question does not depend on which other questions are in the call or on their order. `decide(state, {a, b, c})` gives the same scores as `decide(state, {a})`, `decide(state, {b})` and `decide(state, {c})`, within the 3.4 tolerance. Measured (AG News, 200 articles, the five checked questions, bf16): Qwen2.5-1.5B-Instruct 0.842 alone and 0.841 to 0.842 inside a 20-question call (the earlier shared prompt: 0.864 alone, 0.500 to 0.809 inside); in fp32 the answers are identical on every model that fits the card.

### 3.2 Per-question continuation

After its branch text, a question is answered as if the model were writing `{"q1": <value>}`:

| Type | Suffix (shared by the question's labels) | Candidate text per label | Label |
| --- | --- | --- | --- |
| noul | `{"q1":` | ` true}` and ` false}` | `true`, `false` |
| choice | `{"q1":` | ` ` + `json.dumps(name, ensure_ascii=False)` + `}` | the option name |
| score | `{"q1":` | ` <i>}` for i = 0 .. n-1 | `"0"` .. `"n-1"` |

The closing `}` (and the closing quote for choice) terminates the label, so a label that is a prefix of another (`Sci` vs `Sci-Tech`) is not favoured automatically.

### 3.3 Tokenization

- Only the chat template's own text may become control tokens. The template is rendered around a placeholder for the user message; its head and tail are encoded normally (`add_special_tokens=False`).
- Caller text (the user message, the suffix, every candidate) is encoded so that it never yields a control token: the text is cut just inside every added-token string (`<|im_end|>`, `<think>`, ...) and the pieces are encoded as plain text. A state, instruction or option name that contains `<|im_end|><|im_start|>system ...` therefore cannot close the user turn or forge a new one. A test counts control tokens to prove it.
- Shared part, branch, suffix and candidate ids are concatenated. The boundaries (template | user text, shared part | branch: a blank line then `Question`, and `":` | ` `) are pre-tokenizer boundaries for the Qwen tokenizers, so for text without added-token strings this equals tokenizing the joined string. A test asserts that equality for every test schema.

### 3.4 Packed tree scoring

Speed matters most (founder, 2026-09-30), so the prompt and the labels of all questions go through as few forward passes as `batch_tokens` allows (one, for every built-in scenario), and every token shared by several labels is computed once. Measured on the RTX 5070 with Qwen2.5-1.5B-Instruct before this change: 28 fields 190 ms (122 ms prefill of 1,125 tokens, 66 ms scoring 616 tokens); 255 options 584 ms (313 ms prefill, 253 ms scoring 2,334 tokens).

1. One forward pass reads the prompt and scores the first trees together (each forward has a fixed cost of about 36 ms on the RTX 5070 with a 1.5B model, so passes are the thing to save): the shared part's P tokens (3.1) come first in the pass, causal among themselves, then the tree nodes (step 3). Only when the trees do not fit in `batch_tokens` do later passes run, on top of the prompt's keys and values from the first pass (with the model's decoder, `model.get_decoder()`, and a `DynamicCache`).
2. Per question, build a token tree: the root path is the question's branch (3.1: its question text, the template tail) and suffix ids, then every label's candidate ids hang below it, sharing nodes where labels share leading tokens (all string options share ` "`; options like `returns.refund` and `returns.exchange` share more). Each tree node is one token, fed once.
3. Pack whole question trees into passes of at most `batch_tokens` tree nodes (a larger tree gets a pass of its own). The leading root tokens that every tree of a pass shares (`Question`, `:` and so on) are fed once, as common ancestors of all trees in that pass; it is exact, and it keeps 28 questions at about 1,450 tokens in one forward. In every pass the M nodes are laid out in depth-first order of each tree, a node's position is P + its depth (P = prompt length), and a node sees every prompt position, its own ancestors and itself, nothing else. The first pass also carries the P prompt tokens in front (positions 0 to P-1, causal); later passes start from a fresh batch-1 `DynamicCache` holding only the prompt's keys and values, so passes never see each other.
4. Apply the output head (`model.get_output_embeddings()`) only at nodes whose children are candidate tokens; `log_softmax` in float32. A label's score is the sum, along its path, of each candidate token's log-prob given its parent node. The suffix tokens are context, not scored.
5. Raw score of a label: `s = sum of its candidate-token log-probs` (no length normalization).

Acceptance: the result equals scoring each full sequence without a cache to within 1e-3 in summed log-prob in fp32 on the CPU (measured up to 2.2e-4 on Qwen3-4B-Instruct-2507; exactly 0 in fp64 on Qwen3-0.6B) and 2e-3 in fp32 on the GPU (on the 2,058-token router prompt every GPU fp32 path, the plain uncached reference included, is 1.2e-3 to 1.3e-3 off an fp64 result on Qwen3-1.7B). In bf16 raw scores of unlikely labels move by several units, so bf16 tests compare likely labels by their share among the question's labels. Measured speed (RTX 5070, bf16, warm, p50): Qwen2.5-1.5B-Instruct 28 questions 96 ms, 255 options 209 ms, 1 to 10 questions 35 to 37 ms; Qwen3-4B-Instruct-2507 28 questions 271 ms, 255 options 615 ms.

### 3.5 Probabilities

Per question: `p = softmax(s / T)` over that question's labels, T > 0 (default from section 9, else 1.0). A question with one label (1-option choice, 1-level score) gets p = 1 without running the model for it.

### 3.6 No question-part cache

A cache of the question part needs the questions before the state, which costs too much accuracy (3.1). It was built, measured (a repeated 28-field question set: 77 ms instead of 158 ms) and removed on 2026-10-01 in favour of the one-pass design in 3.4.

## 4. Python API

```python
from mirethstm import Engine, SchemaError

engine = Engine.load(
    "Qwen/Qwen3-4B-Instruct-2507",
    device=None,        # "cuda" if available, else "cpu"
    dtype=None,         # bfloat16 on cuda, float32 on cpu
    batch_tokens=4096,  # tree nodes per scoring pass (section 3.4); bounds memory. A question's tree holds its own text, so 4096 keeps every built-in scenario in one pass
    temperature=None,   # None: the shipped default for this model (section 9), else 1.0
    event_log=None,     # path: append the frozen JSONL events (section 8)
    tarnlight=True,     # feed Tarnlight when its drop folder exists (section 10.3)
)
result = engine.decide(context, schema)   # context: str | dict | list (TypeSafe state); schema: TypeSafe questions map
raw = engine.score(context, schema)       # {question_id: {label: s}} raw summed log-probs, before temperature
result = engine.decide(context, schema, instructions="A brief every question reads.")  # optional, section 3.1
```

`decide` returns the section 2.2 body plus two extra keys: `id` (32 hex chars, one per call) and `latency_ms` (float, wall time of the call). `instructions` (both methods) is `None` or a string without NUL characters; anything else raises `SchemaError` before the model is used. The event log and the Tarnlight drop keep their formats and do not carry the instructions.

`mirethstm.engine.softmax(scores, temperature)` and `mirethstm.engine.answer(question, probabilities)` are public, so callers that apply their own calibration to `score` can build the same answers as `decide`.

## 5. CLI

```
mirethstm decide --schema s.json [--instructions brief.md] [--model ID] [--temperature T] [--device D] [--batch-tokens N] [--log events.jsonl] [--state-json] [--no-tarnlight] < ctx.txt
mirethstm console [--model ID] [--device D] [--host 127.0.0.1] [--port 8766] [--no-tarnlight]
```

- stdin is the state, read as UTF-8 exactly as given (a final newline is part of the state). By default it is a plain string; with `--state-json` it is parsed as JSON.
- `s.json` holds a TypeSafe `questions` map.
- `--instructions brief.md`: a UTF-8 text file passed as the call-level instructions (section 3.1).
- Prints the `decide` result as JSON on stdout (non-ASCII escaped, so any Windows console can print it).
- Exit code 2, message on stderr, before any model loads: `SchemaError`, unreadable or invalid JSON input, non-UTF-8 stdin, `--batch-tokens` below 1, `--temperature` not above 0.
- In Windows PowerShell 5.1, pipe-free: `cmd /c "mirethstm decide --schema s.json < ctx.txt"` (a PowerShell pipe re-encodes the text and corrupts non-ASCII characters).

## 6. HTTP API (on the console's server)

The console's standard-library server (section 10) also answers the API, so there is one local server and no FastAPI dependency. It runs on the user's own machine, bound to 127.0.0.1 by default; it never contacts TypeSafe or any other service, needs no key and costs nothing (founder question 2026-09-30: nothing is linked to any account).

| Route | Behaviour |
| --- | --- |
| `POST /v1/systemone` | TypeSafe drop-in: section 2 shapes, numbers rounded to 2 decimals, `id` in headers `x-request-id` and `x-typesafe-request-id` (the official SDK reads the second), latency in `server-timing`. The request's `model` is accepted and ignored; the answer names the loaded model |
| `POST /v1/decide` | Same request, plus an optional `instructions` string (section 3.1); full precision; body includes `id` and `latency_ms`. `/v1/systemone` keeps TypeSafe's request and ignores `instructions` |
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
- Shipped defaults: `mirethstm.calibration.DEFAULT_TEMPERATURES = {model_id: T}`, one pooled T per approved model, fitted on the benchmark's calibration splits (2026-10-02: 2.112, 7.930, 7.007, 2.702 and 4.639). Models not in the table use T = 1.0. Known limit: one T per model fits yes/no questions poorly on two models; a T per question type is the next step.
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
- Model picker: the approved list of `mirethstm/models.py` (section 11.1), each with its role and measured speed and accuracy; the default is Qwen/Qwen2.5-1.5B-Instruct (the fast model, and the original demo's model), plus `--model` if it is another id. Switching frees the old model before loading the new one; the page shows "Loading <model>...".
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

Any Hugging Face causal LM with a chat template plugs in: `Engine.load("<hub id or local path>")`, `--model`, or the console's model picker. The engine uses only `get_decoder()`, `get_output_embeddings()` and the tokenizer's chat template. Tested: the five approved models of 11.1 (Qwen2.5, Qwen3 and SmolLM3 families); the full test suite passes on each on the GPU. `Engine.load` refuses, with a clear error, a model it would score wrongly: one with sliding-window or other non-full attention layers, or one whose forward changes the logits after the output head (softcapping, scaling).

### 11.1 Approved models (founder, 2026-09-30: "switch models from a list of approved models")

`mirethstm/models.py` holds the approved list: model id, display name, license, size, role and the measured numbers (speed on the RTX 5070, benchmark accuracy and ECE, fitted temperature). The console's model picker and `GET /v1/models` show only approved models with those numbers; `Engine.load` and `--model` still take any id (unvetted, at the user's own risk). A model is approved only when all of these hold:

1. Its license allows free commercial use (Apache-2.0 or MIT); weights are public, not gated.
2. It fits the 12 GB card with room for the cache (bf16 weights well under 10 GB).
3. `Engine.load` accepts it (full attention, no logit post-processing) and the full model test suite passes on the GPU.
4. Its speed and its accuracy and calibration on the benchmark are measured and written into the list.

Evaluated 2026-10-01 and 2026-10-02. Approved: Qwen2.5-1.5B-Instruct, Qwen3-4B-Instruct-2507, Qwen3-1.7B, HuggingFaceTB/SmolLM3-3B, Qwen3-0.6B. Not approved: Qwen2.5-0.5B-Instruct (failed the bf16 GPU suite under the earlier test rule, not re-run), microsoft/Phi-4-mini-instruct (the loader refuses its sliding-window config, and lifting that exposes a position bug near 4,096 tokens), ibm-granite/granite-3.3-2b-instruct (its forward rescales the logits after the output head). Excluded up front: Qwen2.5-3B (non-commercial), gated or custom-license models (Llama, Gemma), sliding-window models. Trained decision models (fastino/GLiNER2.5-Decide, Mapika/decider) do not fit this engine; they need a backend of their own (`docs/research/17-open-weight-decision-models.md`).

### 11.2 Current roles

| Role | Model | License | Notes |
| --- | --- | --- | --- |
| Fast, console default | Qwen/Qwen2.5-1.5B-Instruct | Apache-2.0 | The original demo's model. 96 ms for 28 questions, mean accuracy 0.728 |
| Default SDK/CLI | Qwen/Qwen3-4B-Instruct-2507 | Apache-2.0 | Most accurate (0.760), 271 ms for 28 questions. Non-thinking only |
| Alternative | Qwen/Qwen3-1.7B, HuggingFaceTB/SmolLM3-3B | Apache-2.0 | Hybrid thinking templates; `enable_thinking=False`. SmolLM3's template writes today's date into the prompt |
| CPU tests | Qwen/Qwen3-0.6B | Apache-2.0 | fp32 on CPU (bf16 on this CPU is about 500x slower) |
| Not used | Qwen2.5-3B | Qwen Research License (non-commercial) | |
| Deferred | Qwen3.5 small models | Apache-2.0 | Hybrid linear attention; the plain KV cache does not apply |

- Tested stack: Python 3.11, torch 2.11.0+cu128 (arch list includes `sm_120`), transformers 5.18.0, `attn_implementation="sdpa"`. No flash-attn, no Triton, no torch.compile.
- The cu128 index stops at torch 2.11. torch 2.14.x on cu130 is the path after launch.

## 12. Out of scope for v0.1

vLLM, llama.cpp and MLX backends; fine-tuning; Qwen3.5 hybrid models; a numeric min/max score range (the level list covers Yelp 1 to 5); length-normalized scoring (a benchmark option later, not the default); multi-GPU.
