"""The decision engine: full-label tree scoring on a local causal LM, one forward reading the state
and every question's own branch and labels together (SPEC 2.2, 3, 4)."""

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


# Stand in for the user message and the caller's instructions while the chat template is rendered;
# never reach the model.
_SLOT = "\x00mirethstm-user\x00"
_ISLOT = "\x00mirethstm-instructions\x00"


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


def template_ids(tokenizer, system=sch.SYSTEM_PROMPT, instructions=None):
    """(head, tail): the chat template's ids before and after the user message.

    The head holds the system message, the tail the generation prompt (and an empty think block
    on hybrid models). Only this, the template's own text, yields control tokens. The caller's
    instructions (SPEC 3.1) follow the system message's own text after a blank line, encoded as
    plain text like the state.
    """
    content = f"{system}\n\n{_ISLOT}" if instructions else system
    messages = [{"role": "system", "content": content}, {"role": "user", "content": _SLOT}]
    head, tail = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    ).split(_SLOT)
    tail_ids = tokenizer.encode(tail, add_special_tokens=False)
    if not instructions:
        return tokenizer.encode(head, add_special_tokens=False), tail_ids
    before, after = head.split(_ISLOT)
    return (tokenizer.encode(before, add_special_tokens=False) + encode(tokenizer, instructions)
            + tokenizer.encode(after, add_special_tokens=False)), tail_ids


def shared_ids(tokenizer, state, instructions=None):
    """The part of the prompt every question shares, read once: the template's head, then the state (SPEC 3.1)."""
    return template_ids(tokenizer, instructions=instructions)[0] + encode(tokenizer, sch.shared_text(state))


def branch_ids(tokenizer, q):
    """One question's private branch: its own text, then the template's tail (SPEC 3.1).

    `shared_ids` + `branch_ids` is the whole prompt of that question asked alone.
    """
    return encode(tokenizer, sch.branch_text(q)) + template_ids(tokenizer)[1]


def prefix_ids(tokenizer, state, questions, rules, system):
    """A chat prompt with every question in one user message, closed by `rules`: the
    normal-generation baseline's prompt (SPEC 10.2). The user message is encoded as plain text."""
    head, tail = template_ids(tokenizer, system)
    return head + encode(tokenizer, sch.render_user(state, questions, rules)) + tail


def softmax(scores, temperature):
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
    depths: list  # depth of each node; the first root token is depth 0
    ends: list    # one past the last node of each node's subtree
    paths: list   # per label: its (parent node, candidate token) pairs, whose log-probs sum to its score


def _tree(root_ids, candidate_ids):
    """The root path (a question's branch and suffix) as a chain, then every candidate below its
    last node, sharing the nodes of leading tokens that candidates have in common."""
    trie = {}
    for cand in candidate_ids:
        node = trie
        for t in cand:
            node = node.setdefault(t, {})
    n = len(root_ids)
    tokens, depths, parents = list(root_ids), list(range(n)), list(range(-1, n - 1))
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


def answer(q, probs):
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
    """Scores every allowed answer of every question. The state is read once, in the first forward;
    every question is answered in a branch of its own, as if it were the only question (SPEC 3.1)."""

    def __init__(self, model, tokenizer, model_id, batch_tokens=4096, temperature=1.0, event_log=None,
                 tarnlight=True):
        _check_settings(batch_tokens, temperature)
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.batch_tokens = batch_tokens
        self.temperature = temperature
        self.event_log = event_log
        self.tarnlight = tarnlight
        # Questions and labels repeat from call to call (usually only the state changes): encode each once.
        self._label_ids = functools.lru_cache(maxsize=1 << 16)(lambda text: tuple(encode(tokenizer, text)))
        # So do instructions (usually one profile's): render the template once per distinct text.
        self._template = functools.lru_cache(maxsize=64)(
            lambda instructions: tuple(map(tuple, template_ids(tokenizer, instructions=instructions))))

    @classmethod
    def load(cls, model, device=None, dtype=None, batch_tokens=4096, temperature=None, event_log=None,
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

    def score(self, context, schema, instructions=None):
        """Raw summed label log-probs before temperature: {question_id: {label: s}}.

        A question with one label is not run through the model; its label gets 0.0 (log 1).
        `instructions`: optional text that every question of the call reads (SPEC 3.1).
        """
        return self._run(context, schema, instructions)[0]

    def decide(self, context, schema, instructions=None):
        """The SPEC 2.2 response body plus `id` and `latency_ms`; `instructions` as in `score`."""
        start = time.perf_counter()
        scores, input_tokens = self._run(context, schema, instructions)
        answers, probs_by_field = {}, {}
        for qid, q in schema.items():
            probs = dict(zip(scores[qid], softmax(list(scores[qid].values()), self.temperature)))
            answers[qid] = answer(q, probs)
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

    def _run(self, context, schema, instructions=None):
        """Validate, then score every label; returns (scores, input token count).

        The count follows SPEC 2.2: the shared part (instructions included), plus every token of
        each scored question's branch (its question text, the template's tail and the suffix), plus
        every label's candidate tokens (tree nodes that labels share are fed once, but counted for
        every label).
        """
        sch.validate(context, schema)
        sch.check_instructions(instructions)
        scores = {qid: dict.fromkeys(sch.labels(q), 0.0) for qid, q in schema.items()}
        asked = [qid for qid in schema if len(scores[qid]) > 1]
        if not asked:
            return scores, 0
        head, tail = self._template(instructions or None)
        trees, tokens = [], 0
        for qid in asked:
            # The branch is the root path of the question's tree, so it is read in the same forward.
            root = [*self._label_ids(sch.branch_text(schema[qid])), *tail, *self._label_ids(sch.SUFFIX)]
            cands = [self._label_ids(cand) for cand in sch.candidates(schema[qid])]
            trees.append(_tree(root, cands))
            tokens += len(root) + sum(map(len, cands))
        shared = [*head, *encode(self.tokenizer, sch.shared_text(context))]
        for qid, label_scores in zip(asked, self._score_trees(shared, trees)):
            scores[qid] = dict(zip(scores[qid], label_scores))
        return scores, len(shared) + tokens

    @torch.inference_mode()
    def _score_trees(self, prompt, trees):
        """Label scores of every tree, in order (SPEC 3.4): the trees packed into as few passes as
        batch_tokens allows, the first pass reading `prompt` (the shared part) in the same forward.
        Later passes, if any, each start from a fresh cache holding only the prompt's keys and values."""
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
        plen + its depth and sees the prompt, its ancestors and itself, so no question sees
        another. The output head runs only at nodes whose children are candidate tokens.
        """
        device, f = self.model.device, len(prompt)
        # The root tokens every tree of the pass starts with ("Question:", the key) are fed once, as
        # ancestors of all the trees: they see the prompt and each other, exactly as in each tree alone.
        # Every tree keeps its last root node, the one that predicts its candidates.
        lead = min(tree.paths[0][0][0] for tree in batch)
        lead = next((i for i in range(lead) if len({tree.tokens[i] for tree in batch}) > 1), lead)
        nodes = lead + sum(len(tree.tokens) - lead for tree in batch)
        ids, positions = prompt + batch[0].tokens[:lead], [*range(f), *range(plen, plen + lead)]
        ends, heads, rows, targets = [nodes] * lead, {}, [], []
        for tree in batch:
            base = len(ends) - lead  # where the tree's node 0 would sit; its own nodes start at `lead`
            ids += tree.tokens[lead:]
            positions += [plen + d for d in tree.depths[lead:]]
            ends += [base + e for e in tree.ends[lead:]]
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
