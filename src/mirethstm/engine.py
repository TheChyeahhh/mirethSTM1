"""The decision engine: cached full-label scoring on a local causal LM (SPEC 2.2, 3, 4)."""

import math
import re
import time
import uuid

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from . import schema as sch
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


def prefix_ids(tokenizer, state, questions, rules=sch.ANSWER_RULES):
    """The chat-templated prompt (system + user + generation prompt) that is prefilled once.

    Only the template's own text yields control tokens; the user message is encoded as plain text.
    `rules` closes the user message (the baseline asks for one JSON object instead).
    """
    messages = [{"role": "system", "content": sch.SYSTEM_PROMPT}, {"role": "user", "content": _SLOT}]
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


def _check_settings(chunk_size, temperature):
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if not temperature > 0:
        raise ValueError("temperature must be greater than 0")


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

    def __init__(self, model, tokenizer, model_id, chunk_size=8, temperature=1.0, event_log=None):
        _check_settings(chunk_size, temperature)
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.chunk_size = chunk_size
        self.temperature = temperature
        self.event_log = event_log

    @classmethod
    def load(cls, model, device=None, dtype=None, chunk_size=8, temperature=None, event_log=None):
        """Load a Hugging Face causal LM and its tokenizer."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        if temperature is None:
            temperature = DEFAULT_TEMPERATURES.get(model, 1.0)
        _check_settings(chunk_size, temperature)  # before the weights load, not after
        tokenizer = AutoTokenizer.from_pretrained(model)
        lm = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, attn_implementation="sdpa")
        return cls(lm.to(device).eval(), tokenizer, model, chunk_size, temperature, event_log)

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
            suffix_ids = encode(self.tokenizer, sch.suffix(k))
            for label, cand in zip(labels, sch.candidates(q)):
                pairs.append((qid, label))
                seqs.append((suffix_ids, encode(self.tokenizer, cand)))
        if not seqs:
            return scores, 0
        prefix = prefix_ids(self.tokenizer, context, schema)
        for (qid, label), s in zip(pairs, self._score_ids(prefix, seqs)):
            scores[qid][label] = s
        input_tokens = len(prefix) + sum(len(s) + len(c) for s, c in seqs)
        return scores, input_tokens

    @torch.inference_mode()
    def _score_ids(self, prefix_ids, seqs):
        """Sum of candidate-token log-probs for each (suffix_ids, candidate_ids), prefix prefilled once."""
        model, device = self.model, self.model.device
        cache = DynamicCache(config=model.config)
        model.model(input_ids=torch.tensor([prefix_ids], device=device), past_key_values=cache, use_cache=True)
        # References, not copies: later forwards rebind layer tensors and never write into these.
        base = [(layer.keys, layer.values) for layer in cache.layers]
        plen = len(prefix_ids)
        out = []
        for start in range(0, len(seqs), self.chunk_size):
            chunk = seqs[start:start + self.chunk_size]
            n = len(chunk)
            width = max(len(s) + len(c) for s, c in chunk)
            ids = torch.zeros((n, width), dtype=torch.long)  # pad id is irrelevant: masked, never scored
            mask = torch.zeros((n, plen + width), dtype=torch.long)
            mask[:, :plen] = 1
            rows, cols, targets, counts = [], [], [], []
            for i, (s, c) in enumerate(chunk):
                ids[i, :len(s) + len(c)] = torch.tensor(s + c)
                mask[i, plen:plen + len(s) + len(c)] = 1
                # Position j predicts token j + 1, so candidate token t is read at len(s) - 1 + t.
                rows += [i] * len(c)
                cols += range(len(s) - 1, len(s) - 1 + len(c))
                targets += c
                counts.append(len(c))
            kv = DynamicCache(ddp_cache_data=[
                (k.repeat_interleave(n, 0), v.repeat_interleave(n, 0)) for k, v in base])
            hidden = model.model(input_ids=ids.to(device), attention_mask=mask.to(device),
                                 past_key_values=kv, use_cache=True).last_hidden_state
            picked = hidden[torch.tensor(rows, device=device), torch.tensor(cols, device=device)]
            logp = model.lm_head(picked).float().log_softmax(-1)
            token_lp = logp[torch.arange(len(targets), device=device), torch.tensor(targets, device=device)]
            out += [part.sum().item() for part in token_lp.split(counts)]
        return out
