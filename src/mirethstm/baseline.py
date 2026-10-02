"""Normal generation: the same model writes every answer as one JSON object (SPEC 10.2).

The console races it against `Engine.decide`, and the benchmark uses it as the baseline.
"""

import json
import math
import re
import time

import torch
from transformers.generation.streamers import BaseStreamer

from . import schema as sch
from .engine import encode, prefix_ids

# The baseline's own system message and closing rule: every question in one prompt, all answers in one reply.
BASELINE_SYSTEM = (
    "You read a state and answer questions about it. You answer every question in one reply, "
    "as a single JSON object."
)
BASELINE_RULES = (
    "Answer every question in this one reply, as a single JSON object that holds every "
    'question\'s key in order: {"q1": ..., "q2": ..., ...}. A yes/no question takes true or '
    "false. A choice question takes one option name as a JSON string, exactly as written. "
    "A score question takes one level number. Reply with the JSON object only."
)

_FENCE = re.compile(r"```[A-Za-z]*\s*(.*?)\s*```", re.S)


class _Streamer(BaseStreamer):
    """Passes each newly decoded piece of the output to `on_text`."""

    def __init__(self, tokenizer, on_text):
        self.tokenizer = tokenizer
        self.on_text = on_text
        self.ids = []
        self.sent = 0
        self.prompt_seen = False

    def put(self, value):
        if not self.prompt_seen:  # generate() hands over the prompt first
            self.prompt_seen = True
            return
        self.ids += value.reshape(-1).tolist()
        self._flush(final=False)

    def end(self):
        self._flush(final=True)

    def _flush(self, final):
        text = self.tokenizer.decode(self.ids, skip_special_tokens=True)
        if not final and text.endswith("\N{REPLACEMENT CHARACTER}"):
            return  # half of a multi-byte character; wait for the next token
        if len(text) > self.sent:
            self.on_text(text[self.sent:])
            self.sent = len(text)


def max_new_tokens(tokenizer, questions):
    """Room for every key with its longest allowed answer, plus JSON syntax and line breaks."""
    total = 8  # braces and a possible markdown fence
    for k, q in enumerate(questions.values(), 1):
        answers = [c[1:-1] for c in sch.candidates(q)]  # ' "billing"}' -> '"billing"'
        total += max(len(encode(tokenizer, f'"q{k}": {a}, ')) for a in answers) + 4
    return total


def _reject(constant):
    raise ValueError(f"{constant} is not valid JSON")


def _finite(number):
    """A JSON number as a float; one too large for a float (1e999) counts as invalid JSON."""
    value = float(number)
    if not math.isfinite(value):
        raise ValueError(f"{number} does not fit a float")
    return value


def _allowed(q, value):
    if q["type"] == "noul":
        return isinstance(value, bool)
    if q["type"] == "choice":
        return isinstance(value, str) and value in q["criteria"]
    return type(value) is int and 0 <= value < len(q["criteria"])  # bool is not a level


def check(text, questions):
    """Parse generated text and judge each answer.

    The text must be one JSON object, optionally inside a markdown code fence. Its keys
    q1..qn map back to the caller's question ids in request order. Returns `answers`
    (id to the parsed value, or None when absent), `valid_json`, `hallucinated` (ids whose
    value is not an allowed answer) and `missing` (ids absent from the output).
    """
    body = text.strip()
    fenced = _FENCE.fullmatch(body)
    if fenced:
        body = fenced.group(1)
    try:
        output = json.loads(body, parse_constant=_reject, parse_float=_finite)
    except ValueError:
        output = None
    valid_json = isinstance(output, dict)
    if not valid_json:
        output = {}
    answers, hallucinated, missing = {}, [], []
    for k, (qid, q) in enumerate(questions.items(), 1):
        key = f"q{k}"
        if key not in output:
            answers[qid] = None
            missing.append(qid)
            continue
        answers[qid] = output[key]
        if not _allowed(q, output[key]):
            hallucinated.append(qid)
    return {"answers": answers, "valid_json": valid_json, "hallucinated": hallucinated, "missing": missing}


def generate(engine, context, schema, on_text=None):
    """Greedy generation of all answers as one JSON object, timed like `Engine.decide`.

    Same model and question blocks as the scorer (SPEC 3.1), with BASELINE_SYSTEM and
    BASELINE_RULES asking for one object. `on_text(chunk)` receives the output as it is generated.
    """
    start = time.perf_counter()
    sch.validate(context, schema)
    tokenizer, model = engine.tokenizer, engine.model
    ids = torch.tensor([prefix_ids(tokenizer, context, schema, rules=BASELINE_RULES, system=BASELINE_SYSTEM)],
                       device=model.device)
    streamer = _Streamer(tokenizer, on_text or (lambda chunk: None))
    with torch.inference_mode():
        out = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.0,  # pure greedy; some chat models ship a penalty in their config
            max_new_tokens=max_new_tokens(tokenizer, schema),
            streamer=streamer,
        )
    new_ids = out[0, ids.shape[1]:].tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    result = {"text": text, **check(text, schema)}
    result["latency_ms"] = (time.perf_counter() - start) * 1000
    result["output_tokens"] = len(new_ids)
    return result
