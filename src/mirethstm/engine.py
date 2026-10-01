"""The decision engine: full-label tree scoring on a local causal LM, one forward reading the prompt
and the trees together (SPEC 2.2, 3, 4)."""

import functools
import math
import re
import time
import uuid
from typing import NamedTuple

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
    """The chat-templated prompt (system + user + generation prompt) that is read once.

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


class _Tree(NamedTuple):
    """One question's token tree, nodes in depth-first order (SPEC 3.4)."""

    tokens: list  # token id of each node
    depths: list  # depth of each node; the first suffix token is depth 0
    ends: list    # one past the last node of each node's subtree
    paths: list   # per label: its (parent node, candidate token) pairs, whose log-probs sum to its score


def _tree(suffix_ids, candidate_ids):
    """The suffix as a chain, then every candidate below its last node, sharing the nodes of
    leading tokens that candidates have in common."""
    trie = {}
    for cand in candidate_ids:
        node = trie
        for t in cand:
            node = node.setdefault(t, {})
    n = len(suffix_ids)
    tokens, depths, parents = list(suffix_ids), list(range(n)), list(range(-1, n - 1))
    index, stack = {}, [(n - 1, iter(trie.items()))]
    while stack:  # depth-first without recursion: a label can be hundreds of tokens long
        parent, children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            continue
        token, below = child
        index[parent, token] = len(tokens)
        stack.append((len(tokens), iter(below.items())))
        tokens.append(token)
        depths.append(depths[parent] + 1)
        parents.append(parent)
    ends = list(range(1, len(tokens) + 1))
    for i in range(len(tokens) - 1, 0, -1):  # a subtree ends where its last descendant does
        ends[parents[i]] = max(ends[parents[i]], ends[i])
    paths = []
    for cand in candidate_ids:
        parent, path = n - 1, []
        for t in cand:
            path.append((parent, t))
            parent = index[parent, t]
        paths.append(path)
    return _Tree(tokens, depths, ends, paths)


def _passes(trees, batch_tokens):
    """Consecutive trees grouped into passes of at most batch_tokens nodes.

    A tree larger than the budget gets a pass of its own.
    """
    batch, size = [], 0
    for tree in trees:
        if batch and size + len(tree.tokens) > batch_tokens:
            yield batch
            batch, size = [], 0
        batch.append(tree)
        size += len(tree.tokens)
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
    """Scores every allowed answer of every question; the prompt is read once, in the first forward."""

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
        if device is None:  # is_available() can be True with no visible device (CUDA_VISIBLE_DEVICES="")
            device = "cuda" if torch.cuda.device_count() else "cpu"
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
        """Validate, then score every label; returns (scores, input token count).

        The count follows SPEC 2.2: the prompt, plus each label's suffix and candidate as if fed on
        their own (tree nodes that labels share are fed once, but counted for every label).
        """
        sch.validate(context, schema)
        scores = {qid: {} for qid in schema}
        owners, trees, label_tokens = [], [], 0
        for k, (qid, q) in enumerate(schema.items(), 1):
            labels = sch.labels(q)
            if len(labels) == 1:
                scores[qid][labels[0]] = 0.0
                continue
            suffix_ids = self._label_ids(sch.suffix(k))
            cands = [self._label_ids(cand) for cand in sch.candidates(q)]
            owners.append((qid, labels))
            trees.append(_tree(suffix_ids, cands))
            label_tokens += sum(len(suffix_ids) + len(c) for c in cands)
        if not trees:
            return scores, 0
        prompt = prefix_ids(self.tokenizer, context, schema)
        for (qid, labels), label_scores in zip(owners, self._score_trees(prompt, trees)):
            scores[qid] = dict(zip(labels, label_scores))
        return scores, len(prompt) + label_tokens

    @torch.inference_mode()
    def _score_trees(self, prompt, trees):
        """Label scores of every tree, in order (SPEC 3.4): the trees packed into as few passes as
        batch_tokens allows, the first pass reading the prompt in the same forward. Later passes,
        if any, each start from a fresh cache holding only the prompt's keys and values."""
        decoder, plen = self.model.get_decoder(), len(prompt)
        first, *rest = _passes(trees, self.batch_tokens)
        cache = DynamicCache(config=self.model.config) if rest else None
        out = self._score_pass(decoder, prompt, plen, first, cache)
        if rest:
            # Views of the prompt's keys and values; each later forward copies them into its own
            # cache (torch.cat), so no pass ever sees another pass's nodes.
            prompt_kv = [(layer.keys[:, :, :plen], layer.values[:, :, :plen]) for layer in cache.layers]
        for batch in rest:
            out += self._score_pass(decoder, [], plen, batch, DynamicCache(ddp_cache_data=prompt_kv))
        return out

    def _score_pass(self, decoder, prompt, plen, batch, cache):
        """One forward over `prompt` (the whole prompt in the first pass, else empty: its keys and
        values are then in `cache`) followed by the trees of `batch`; their label scores.

        Prompt tokens sit at positions 0 .. plen - 1, causal among themselves. A tree node sits at
        plen + its depth and sees the prompt, its ancestors and itself. The output head runs only
        at nodes whose children are candidate tokens.
        """
        device, f = self.model.device, len(prompt)
        ids, positions, ends, heads, rows, targets = list(prompt), list(range(f)), [], {}, [], []
        for tree in batch:
            base = len(ends)  # nodes laid out so far
            ids += tree.tokens
            positions += [plen + d for d in tree.depths]
            ends += [base + e for e in tree.ends]
            for path in tree.paths:
                for parent, token in path:
                    rows.append(heads.setdefault(f + base + parent, len(heads)))
                    targets.append(token)
        order = torch.arange(len(ends), device=device)
        end = torch.tensor(ends, device=device)
        # Depth-first order: node j's subtree is j .. end[j] - 1, so node i sees j when j <= i < end[j].
        own = (order[None, :] <= order[:, None]) & (order[:, None] < end[None, :])
        # Bool 4D mask, True = attend. transformers hands a 4D mask to SDPA unchanged, so it spans every
        # key of the forward: the plen prompt keys (from the cache or from this forward), then the nodes.
        mask = torch.cat([own.new_ones((len(ends), plen)), own], dim=1)
        if f:  # the prompt's own rows: causal, blind to the nodes
            mask = torch.cat([torch.cat([own.new_ones((f, f)).tril(), own.new_zeros((f, len(ends)))], dim=1), mask])
        hidden = decoder(input_ids=torch.tensor([ids], device=device),
                         position_ids=torch.tensor([positions], device=device),
                         attention_mask=mask[None, None], past_key_values=cache,
                         use_cache=cache is not None).last_hidden_state[0]
        head = self.model.get_output_embeddings()
        logp = head(hidden[torch.tensor(list(heads), device=device)]).float().log_softmax(-1)
        token_lp = logp[torch.tensor(rows, device=device), torch.tensor(targets, device=device)].tolist()
        out, i = [], 0
        for tree in batch:
            out.append([])
            for path in tree.paths:
                out[-1].append(sum(token_lp[i:i + len(path)]))
                i += len(path)
        return out
