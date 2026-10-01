"""The decision engine: cached full-label scoring on a local causal LM (SPEC 2.2, 3, 4)."""

import functools
import math
import re
import time
import uuid

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.cache_utils import DynamicLayer

from . import schema as sch
from . import tarnlight
from .calibration import DEFAULT_TEMPERATURES
from .events import write_events


# Stands in for the user message while the chat template is rendered; never reaches the model.
_SLOT = "\x00mirethstm-user\x00"


def encode(tokenizer, text):
    """Ids for caller text (state, questions, labels), never a control token.

    The text is cut just inside every added-token string ("<|im_end|>", "<think>", ...) so
    the pieces encode as plain text and cannot end a turn or open a new one.
    """
    added = re.compile("|".join(map(re.escape, sorted(tokenizer.get_added_vocab(), key=len, reverse=True))))
    pieces, cut = [], 0
    while match := added.search(text, cut):
        pieces.append(text[cut:match.start() + 1])
        cut = match.start() + 1
    pieces.append(text[cut:])
    return [i for piece in pieces for i in tokenizer.encode(piece, add_special_tokens=False)]


def prefix_ids(tokenizer, state, questions, rules=sch.ANSWER_RULES, system=sch.SYSTEM_PROMPT):
    """The chat-templated prompt (system + user + generation prompt) that is prefilled once.

    Only the template's own text yields control tokens; the user message is encoded as plain text.
    `system` and `rules` (the closing of the user message) differ for the baseline, which asks
    for one JSON object holding every answer.
    """
    messages = [{"role": "system", "content": system}, {"role": "user", "content": _SLOT}]
    head, tail = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    ).split(_SLOT)
    return (tokenizer.encode(head, add_special_tokens=False)
            + encode(tokenizer, sch.render_user(state, questions, rules))
            + tokenizer.encode(tail, add_special_tokens=False))


def _softmax(scores, temperature):
    top = max(scores)
    weights = [math.exp((s - top) / temperature) for s in scores]
    total = sum(weights)
    return [w / total for w in weights]


def _check_settings(batch_tokens, temperature):
    if batch_tokens < 1:
        raise ValueError("batch_tokens must be at least 1")
    if not temperature > 0:
        raise ValueError("temperature must be greater than 0")


def check_supported(lm, model_id):
    """Raise ValueError for a model this engine would score wrongly (SPEC 11).

    The per-pass caches hold full-attention layers only, and the engine applies the output head
    itself, so a forward that changes the logits after the head (softcapping, scaling) is skipped.
    """
    if any(type(layer) is not DynamicLayer for layer in DynamicCache(config=lm.config).layers):
        raise ValueError(f"{model_id} is not supported: it has sliding-window or other non-full attention layers")
    ids = torch.tensor([[0, 1, 2, 3]], device=lm.device)
    with torch.inference_mode():
        full = lm(input_ids=ids).logits.float()
        ours = lm.get_output_embeddings()(lm.get_decoder()(input_ids=ids).last_hidden_state).float()
    if not torch.allclose(full, ours, rtol=1e-3, atol=1e-3):
        raise ValueError(f"{model_id} is not supported: its forward changes the logits after the output head")


def _passes(seqs, batch_tokens):
    """Consecutive (suffix_ids, candidate_ids) pairs grouped into passes of at most batch_tokens tokens.

    A pair longer than the budget gets a pass of its own.
    """
    batch, size = [], 0
    for s, c in seqs:
        if batch and size + len(s) + len(c) > batch_tokens:
            yield batch
            batch, size = [], 0
        batch.append((s, c))
        size += len(s) + len(c)
    if batch:
        yield batch


def _rounded(value):
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, dict):
        return {key: _rounded(v) for key, v in value.items()}
    return value


def systemone_body(result):
    """The POST /v1/systemone body for a decide result: no id or latency, every number rounded to 2 decimals."""
    return _rounded({key: result[key] for key in ("model", "answers", "usage")})


def _answer(q, probs):
    """One SPEC 2.2 answer from a question and its {label: probability}."""
    if q["type"] == "noul":
        return {"type": "noul", "noul": probs["true"]}
    top = max(probs, key=probs.get)  # ties go to the earlier label
    k = len(probs)
    confidence = 1.0 if k == 1 else (k * probs[top] - 1) / (k - 1)
    if q["type"] == "choice":
        return {"type": "choice", "choice": top, "confidence": confidence, "probabilities": probs}
    return {
        "type": "score",
        "score": sum(i * p for i, p in enumerate(probs.values())),
        "confidence": confidence,
        "legend": {str(i): sch.text(level) for i, level in enumerate(q["criteria"])},
        "probabilities": probs,
    }


class Engine:
    """Scores every allowed answer of every question against one prefilled KV cache."""

    def __init__(self, model, tokenizer, model_id, batch_tokens=2048, temperature=1.0, event_log=None,
                 tarnlight=True):
        _check_settings(batch_tokens, temperature)
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.batch_tokens = batch_tokens
        self.temperature = temperature
        self.event_log = event_log
        self.tarnlight = tarnlight
        # Suffixes and labels repeat from call to call (usually only the state changes): encode each once.
        self._label_ids = functools.lru_cache(maxsize=1 << 16)(lambda text: tuple(encode(tokenizer, text)))

    @classmethod
    def load(cls, model, device=None, dtype=None, batch_tokens=2048, temperature=None, event_log=None,
             tarnlight=True):
        """Load a Hugging Face causal LM and its tokenizer."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        if temperature is None:
            temperature = DEFAULT_TEMPERATURES.get(model, 1.0)
        _check_settings(batch_tokens, temperature)  # before the weights load, not after
        tokenizer = AutoTokenizer.from_pretrained(model)
        lm = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, attn_implementation="sdpa").to(device).eval()
        check_supported(lm, model)
        return cls(lm, tokenizer, model, batch_tokens, temperature, event_log, tarnlight)

    def score(self, context, schema):
        """Raw summed label log-probs before temperature: {question_id: {label: s}}.

        A question with one label is not run through the model; its label gets 0.0 (log 1).
        """
        return self._run(context, schema)[0]

    def decide(self, context, schema):
        """The SPEC 2.2 response body plus `id` and `latency_ms`."""
        start = time.perf_counter()
        scores, input_tokens = self._run(context, schema)
        answers, probs_by_field = {}, {}
        for qid, q in schema.items():
            probs = dict(zip(scores[qid], _softmax(list(scores[qid].values()), self.temperature)))
            answers[qid] = _answer(q, probs)
            probs_by_field[qid] = probs
        result = {
            "model": self.model_id,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            "id": uuid.uuid4().hex,
            "latency_ms": (time.perf_counter() - start) * 1000,
        }
        if self.event_log:
            write_events(self.event_log, result["id"], probs_by_field, self.temperature,
                         result["latency_ms"], self.model_id)
        if self.tarnlight:
            tarnlight.drop({"model": self.model_id, "state": context, "questions": schema},
                           systemone_body(result), result["id"], result["latency_ms"])
        return result

    def _run(self, context, schema):
        """Validate, then score every label; returns (scores, input token count)."""
        sch.validate(context, schema)
        scores = {qid: {} for qid in schema}
        pairs, seqs = [], []
        for k, (qid, q) in enumerate(schema.items(), 1):
            labels = sch.labels(q)
            if len(labels) == 1:
                scores[qid][labels[0]] = 0.0
                continue
            suffix_ids = self._label_ids(sch.suffix(k))
            for label, cand in zip(labels, sch.candidates(q)):
                pairs.append((qid, label))
                seqs.append((suffix_ids, self._label_ids(cand)))
        if not seqs:
            return scores, 0
        prefix = prefix_ids(self.tokenizer, context, schema)
        for (qid, label), s in zip(pairs, self._score_ids(prefix, seqs)):
            scores[qid][label] = s
        input_tokens = len(prefix) + sum(len(s) + len(c) for s, c in seqs)
        return scores, input_tokens

    @torch.inference_mode()
    def _score_ids(self, prefix_ids, seqs):
        """Sum of candidate-token log-probs for each (suffix_ids, candidate_ids): one prefill, then
        every sequence packed into as few passes as batch_tokens allows (SPEC 3.4)."""
        decoder, device = self.model.get_decoder(), self.model.device
        cache = DynamicCache(config=self.model.config)
        decoder(input_ids=torch.tensor([prefix_ids], device=device), past_key_values=cache, use_cache=True)
        # References, not copies: a later forward rebinds each layer's tensors and never writes into these.
        prefix_kv = [(layer.keys, layer.values) for layer in cache.layers]
        out = []
        for batch in _passes(seqs, self.batch_tokens):
            out += self._score_pass(decoder, prefix_kv, len(prefix_ids), batch)
        return out

    def _score_pass(self, decoder, prefix_kv, plen, batch):
        """One forward over the sequences of `batch` laid end to end after the prefix.

        Each sequence restarts at position plen and sees the prefix and its own earlier tokens only.
        """
        device = self.model.device
        ids, positions, owner, rows, targets, counts = [], [], [], [], [], []
        for i, (s, c) in enumerate(batch):
            start = len(ids)
            ids += s + c
            positions += range(plen, plen + len(s) + len(c))
            owner += [i] * (len(s) + len(c))
            # Position j predicts token j + 1, so candidate token t is read at start + len(s) - 1 + t.
            rows += range(start + len(s) - 1, start + len(s) - 1 + len(c))
            targets += c
            counts.append(len(c))
        owner = torch.tensor(owner, device=device)
        order = torch.arange(len(ids), device=device)
        own = (owner[:, None] == owner[None, :]) & (order[None, :] <= order[:, None])
        # Bool 4D mask, True = attend: transformers hands a 4D mask to SDPA unchanged.
        mask = torch.cat([own.new_ones((len(ids), plen)), own], dim=1)[None, None]
        hidden = decoder(input_ids=torch.tensor([ids], device=device),
                         position_ids=torch.tensor([positions], device=device),
                         attention_mask=mask, past_key_values=DynamicCache(ddp_cache_data=prefix_kv),
                         use_cache=True).last_hidden_state[0]
        head = self.model.get_output_embeddings()
        logp = head(hidden[torch.tensor(rows, device=device)]).float().log_softmax(-1)
        token_lp = logp[torch.arange(len(targets), device=device), torch.tensor(targets, device=device)]
        return torch.stack([part.sum() for part in token_lp.split(counts)]).tolist()
