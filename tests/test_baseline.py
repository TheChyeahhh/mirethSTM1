"""Normal-generation baseline (SPEC 10.2). Generation is scripted here; no model runs."""

import json
from types import SimpleNamespace

import pytest
import torch

from mirethstm import SchemaError, baseline
from mirethstm.engine import encode, prefix_ids
from mirethstm.scenarios import SCENARIOS

QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"},
    "team": {"type": "choice", "instructions": "Which team should handle it?",
             "criteria": {"billing": "Charges and refunds", "Sci-Tech": None, "Crème brûlée 🍮": None}},
    "urgency": {"type": "score", "instructions": "How urgent is it?",
                "criteria": ["No rush", "This week", "Today", "Right now"]},
}
STATE = "I was billed twice this month. Please refund the extra charge today. Café crème."


# --- parsing and judging ----------------------------------------------------------------


def test_check_maps_keys_back_to_question_ids():
    result = baseline.check('{"q1": true, "q2": "Sci-Tech", "q3": 3}', QUESTIONS)
    assert result == {"answers": {"refund": True, "team": "Sci-Tech", "urgency": 3},
                      "valid_json": True, "hallucinated": [], "missing": []}


@pytest.mark.parametrize("qid, key, value", [
    ("refund", "q1", "true"), ("refund", "q1", 1), ("refund", "q1", None), ("refund", "q1", "yes"),
    ("team", "q2", "billing "), ("team", "q2", "Billing"), ("team", "q2", "sci-tech"), ("team", "q2", 0),
    ("team", "q2", ["billing"]),
    ("urgency", "q3", 4), ("urgency", "q3", -1), ("urgency", "q3", 2.0), ("urgency", "q3", "2"),
    ("urgency", "q3", True),
])
def test_values_outside_the_allowed_answers_are_hallucinated(qid, key, value):
    output = {"q1": False, "q2": "billing", "q3": 0, key: value}
    result = baseline.check(json.dumps(output), QUESTIONS)
    assert result["valid_json"] is True
    assert result["hallucinated"] == [qid]
    assert result["missing"] == []
    assert result["answers"][qid] == value


def test_missing_keys():
    result = baseline.check('{"q1": false, "refund": true, "q9": 1}', QUESTIONS)
    assert result["answers"] == {"refund": False, "team": None, "urgency": None}
    assert result["missing"] == ["team", "urgency"]
    assert result["hallucinated"] == []
    assert result["valid_json"] is True


@pytest.mark.parametrize("text", [
    '{"q1": true, "q2": ',            # cut off
    '{"q1": true} trailing words',
    '[true, "billing", 1]',            # JSON, but not an object
    '"q1"',
    '{"q1": true, "q2": "billing", "q3": NaN}',
    '{"q1": true, "q2": "billing", "q3": 1e999}',     # too large for a float: the page could not parse it
    '{"q1": true, "q2": "billing", "q3": -1e999}',
    "",
])
def test_invalid_json(text):
    result = baseline.check(text, QUESTIONS)
    assert result["valid_json"] is False
    assert result["answers"] == {"refund": None, "team": None, "urgency": None}
    assert result["missing"] == ["refund", "team", "urgency"]
    assert result["hallucinated"] == []


def test_markdown_fence_and_whitespace_are_accepted():
    text = '\n```json\n{\n  "q1": true,\n  "q2": "Crème brûlée 🍮",\n  "q3": 0\n}\n```\n'
    result = baseline.check(text, QUESTIONS)
    assert result["valid_json"] is True
    assert result["answers"] == {"refund": True, "team": "Crème brûlée 🍮", "urgency": 0}


# --- generation, with a scripted model ----------------------------------------------------


class ScriptedModel:
    """Stands in for model.generate(): feeds the streamer the ids of `reply` one at a time."""

    device = torch.device("cpu")

    def __init__(self, tokenizer, reply):
        self.reply_ids = tokenizer.encode(reply, add_special_tokens=False)
        self.reply_ids.append(tokenizer.eos_token_id)
        self.calls = []

    def generate(self, input_ids, streamer, max_new_tokens, **settings):
        self.calls.append({"prompt": input_ids[0].tolist(), "max_new_tokens": max_new_tokens, **settings})
        new = self.reply_ids[:max_new_tokens]
        streamer.put(input_ids)
        for token in new:
            streamer.put(torch.tensor([token]))
        streamer.end()
        return torch.cat([input_ids, torch.tensor([new])], dim=1)


def scripted(tokenizer, reply):
    return SimpleNamespace(tokenizer=tokenizer, model=ScriptedModel(tokenizer, reply))


def test_generate_streams_and_judges(tokenizer):
    reply = '{"q1": true, "q2": "Crème brûlée 🍮", "q3": 7}'
    engine = scripted(tokenizer, reply)
    chunks = []
    result = baseline.generate(engine, STATE, QUESTIONS, on_text=chunks.append)

    assert list(result) == ["text", "answers", "valid_json", "hallucinated", "missing", "latency_ms", "output_tokens"]
    assert result["text"] == reply
    assert "".join(chunks) == reply
    assert len(chunks) > 5 and not any("\N{REPLACEMENT CHARACTER}" in c for c in chunks)  # half characters are held back
    assert result["answers"] == {"refund": True, "team": "Crème brûlée 🍮", "urgency": 7}
    assert result["hallucinated"] == ["urgency"]
    assert result["missing"] == []
    assert result["output_tokens"] == len(engine.model.reply_ids)
    assert result["latency_ms"] > 0

    call = engine.model.calls[0]
    assert call["prompt"] == prefix_ids(tokenizer, STATE, QUESTIONS, rules=baseline.BASELINE_RULES,
                                        system=baseline.BASELINE_SYSTEM)
    assert call["do_sample"] is False
    assert call["max_new_tokens"] == baseline.max_new_tokens(tokenizer, QUESTIONS)


def test_generate_without_a_callback(tokenizer):
    result = baseline.generate(scripted(tokenizer, '{"q1": false}'), STATE, QUESTIONS)
    assert result["missing"] == ["team", "urgency"]


def test_prompt_asks_for_one_object(tokenizer):
    text = tokenizer.decode(prefix_ids(tokenizer, STATE, QUESTIONS, rules=baseline.BASELINE_RULES,
                                       system=baseline.BASELINE_SYSTEM))
    assert baseline.BASELINE_SYSTEM in text and baseline.BASELINE_RULES in text
    # Nothing of the scorer's one-question-per-reply instructions is left to contradict them.
    assert "one question per reply" not in text and "only that question's key" not in text


def test_bad_schema_fails_before_generating(tokenizer):
    engine = scripted(tokenizer, "{}")
    with pytest.raises(SchemaError):
        baseline.generate(engine, STATE, {"q": {"type": "bool", "instructions": "x"}})
    assert engine.model.calls == []


@pytest.mark.parametrize("questions", [QUESTIONS] + [s["questions"] for s in SCENARIOS],
                         ids=["small"] + [s["id"] for s in SCENARIOS])
def test_token_budget_fits_the_longest_pretty_printed_answer(tokenizer, questions):
    longest = {}
    for k, q in enumerate(questions.values(), 1):
        answers = [False] if q["type"] == "noul" else (
            list(q["criteria"]) if q["type"] == "choice" else [len(q["criteria"]) - 1])
        longest[f"q{k}"] = max(answers, key=lambda a: len(encode(tokenizer, json.dumps(a, ensure_ascii=False))))
    reply = json.dumps(longest, ensure_ascii=False, indent=2)
    assert len(encode(tokenizer, reply)) + 1 <= baseline.max_new_tokens(tokenizer, questions)  # + end of turn
