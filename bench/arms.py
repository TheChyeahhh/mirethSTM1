"""The benchmark arms. Each takes (engine, state, questions) and returns one result per question.

- mireth: `Engine.score`, full-label scoring (SPEC 3).
- baseline: `mirethstm.baseline.generate`, the same model writing JSON; an answer outside the
  allowed set, or a missing one, counts wrong; no confidence.
- first_token: the original demo's readout, reimplemented from its description
  (docs/research/02 sections 3.3 to 3.5): our prompt (SPEC 3.1), then per question one suffix
  that ends where the label starts, and only the first token of each label is scored; the
  probabilities are a softmax over the options. Options whose first tokens are the same token
  get the same score (a tie); the tie goes to the earlier option in the request, as in the
  original's argmax, and the report counts the ties and the predictions they decide.
- questions_first: full-label scoring of the prompt with the question before the state (the
  order that was measured and dropped), so the benchmark can compare the two orders. One
  forward per label after one prefill per question: slow, a reference, not a speed claim.

first_token and questions_first ask every question alone, with the engine's own prompt for one
question (SPEC 3.1: the shared part, then that question's branch), so the arms compare like for like.
"""

import json
import re

import torch
from transformers import DynamicCache

from mirethstm import baseline
from mirethstm import schema as sch
from mirethstm.engine import branch_ids, encode, shared_ids, template_ids


def mireth(engine, state, questions):
    return {qid: {"scores": s} for qid, s in engine.score(state, questions).items()}


def _label(q, value):
    """A generated value as an engine label: true/false, the option name, or the level as a string."""
    if q["type"] == "noul":
        return "true" if value else "false"
    return str(value)


_VALUE = r'\s*:\s*(true|false|-?\d+(?![\d.eE])|"(?:[^"\\]|\\.)*")'


def salvage(text, k, q):
    """The first allowed value written for key "qk" anywhere in the text, as an engine label, or None.

    A secondary number only (docs/research/07 section 3): it reads answers out of output that
    is not valid JSON, for example cut off after the model invented extra keys.
    """
    for match in re.finditer(f'"q{k}"' + _VALUE, text):
        value = json.loads(match.group(1))
        if baseline.check(json.dumps({"q1": value}), {"x": q})["hallucinated"] == []:  # alone, q is key q1
            return _label(q, value)
    return None


def generate(engine, state, questions):
    out = baseline.generate(engine, state, questions)
    cap_hit = out["output_tokens"] >= baseline.max_new_tokens(engine.tokenizer, questions)
    results = {}
    for k, (qid, q) in enumerate(questions.items(), 1):
        bad = qid in out["hallucinated"] or qid in out["missing"]
        results[qid] = {"answer": out["answers"][qid], "pred": None if bad else _label(q, out["answers"][qid]),
                        "salvaged": salvage(out["text"], k, q), "valid_json": out["valid_json"],
                        "hallucinated": qid in out["hallucinated"], "missing": qid in out["missing"],
                        "text": out["text"], "output_tokens": out["output_tokens"], "cap_hit": cap_hit}
    return results


def first_token_plan(tokenizer, questions):
    """{qid: (suffix ids, first-token id per label)} for the first_token arm.

    The suffix is the engine's own (SPEC 3.2) plus every token that all of the question's
    candidates share: the JSON syntax before the label (' "' before an option name, the
    space before a level number). Like the original's shared-prefix hoisting, label tokens
    that every option shares move into the suffix too, so each label keeps a first token.
    A question with one label gets no plan (it scores 0.0, as in the engine).
    """
    plan = {}
    for qid, q in questions.items():
        cands = [encode(tokenizer, c) for c in sch.candidates(q)]
        if len(cands) == 1:
            continue
        shared, shortest = 0, min(map(len, cands))
        while shared < shortest - 1 and len({c[shared] for c in cands}) == 1:
            shared += 1
        plan[qid] = (encode(tokenizer, sch.SUFFIX) + cands[0][:shared], [c[shared] for c in cands])
    return plan


def tie_groups(labels, first_ids):
    """Groups (in request order) of two or more labels that share their scored first token."""
    groups = {}
    for label, token in zip(labels, first_ids):
        groups.setdefault(token, []).append(label)
    return [g for g in groups.values() if len(g) > 1]


def _prefill(model, ids):
    cache = DynamicCache(config=model.config)
    model.get_decoder()(input_ids=torch.tensor([ids], device=model.device), past_key_values=cache, use_cache=True)
    return cache


@torch.inference_mode()
def first_token(engine, state, questions):
    sch.validate(state, questions)
    tokenizer, model = engine.tokenizer, engine.model
    plan = first_token_plan(tokenizer, questions)
    results = {qid: {"scores": {sch.labels(q)[0]: 0.0}} for qid, q in questions.items() if qid not in plan}
    if plan:
        cache = _prefill(model, shared_ids(tokenizer, state))
        for qid, (suffix, first) in plan.items():
            ids = branch_ids(tokenizer, questions[qid]) + suffix  # the question alone, as in the engine
            logits = model(input_ids=torch.tensor([ids], device=model.device), past_key_values=cache,
                           use_cache=True, logits_to_keep=1).logits[0, -1]
            logp = logits.float().log_softmax(-1)[torch.tensor(first, device=model.device)].tolist()
            results[qid] = {"scores": dict(zip(sch.labels(questions[qid]), logp))}
            cache.crop(-len(ids))  # back to the shared part for the next question
    return {qid: results[qid] for qid in questions}


def questions_first_ids(tokenizer, state, q):
    """The engine's prompt for one question (SPEC 3.1) with the question moved before the state."""
    head, tail = template_ids(tokenizer)
    user = f"{sch.branch_text(q)}\n\n{sch.shared_text(state)[:-2]}"  # without the blank line that ends the state
    return head + encode(tokenizer, user) + tail


@torch.inference_mode()
def questions_first(engine, state, questions):
    sch.validate(state, questions)
    tokenizer, model = engine.tokenizer, engine.model
    suffix = encode(tokenizer, sch.SUFFIX)
    results = {}
    for qid, q in questions.items():
        labels = sch.labels(q)
        if len(labels) == 1:
            results[qid] = {"scores": {labels[0]: 0.0}}
            continue
        cache = _prefill(model, questions_first_ids(tokenizer, state, q))
        scores = {}
        for label, cand in zip(labels, sch.candidates(q)):
            c = encode(tokenizer, cand)
            logits = model(input_ids=torch.tensor([suffix + c], device=model.device), past_key_values=cache,
                           use_cache=True).logits[0]
            # Position j predicts token j + 1: candidate token t is read at len(suffix) - 1 + t.
            logp = logits[len(suffix) - 1:-1].float().log_softmax(-1)
            picked = logp.gather(1, torch.tensor(c, device=model.device)[:, None])
            scores[label] = float(picked.sum())
            cache.crop(-(len(suffix) + len(c)))  # back to the prefix
        results[qid] = {"scores": scores}
    return results


ARMS = {"mireth": mireth, "baseline": generate, "first_token": first_token, "questions_first": questions_first}
SCORING = ("mireth", "first_token", "questions_first")  # arms that give a distribution
