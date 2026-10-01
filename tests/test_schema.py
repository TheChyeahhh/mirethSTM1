"""Schema validation, prompt text and continuations (no model needed)."""

import pytest

from mirethstm import SchemaError
from mirethstm.schema import candidates, labels, render_user, suffix, validate, SYSTEM_PROMPT

NOUL = {"type": "noul", "instructions": "Yes?"}
CHOICE = {"type": "choice", "instructions": "Which?", "criteria": {"a": "first", "b": None}}
SCORE = {"type": "score", "instructions": "How much?", "criteria": ["low", "high"]}


def many(n):
    return {f"option {i}": None for i in range(n)}


@pytest.mark.parametrize(
    "state, questions, message",
    [
        (5, {"q": NOUL}, "state must be"),
        (None, {"q": NOUL}, "state must be"),
        ("s", {}, "non-empty"),
        ("s", [NOUL], "non-empty"),
        ("s", {"q": "noul"}, "must be an object"),
        ("s", {"q": {"instructions": "x"}}, "unknown type"),
        ("s", {"q": {"type": "bool", "instructions": "x"}}, "unknown type"),
        ("s", {"q": {"type": "noul", "instructions": 5}}, "instructions"),
        ("s", {"q": {"type": "choice", "instructions": True, "criteria": {"a": None}}}, "instructions"),
        ("s", {"q": {**NOUL, "criteria": "yes"}}, "noul criteria"),
        ("s", {"q": {**NOUL, "criteria": {"yes": "x"}}}, "noul criteria"),
        ("s", {"q": {**NOUL, "criteria": {"true": 1}}}, "descriptions"),
        ("s", {"q": {"type": "choice", "instructions": "x"}}, "choice criteria"),
        ("s", {"q": {**CHOICE, "criteria": ["a", "b"]}}, "choice criteria"),
        ("s", {"q": {**CHOICE, "criteria": {}}}, "1 to 255 options, got 0"),
        ("s", {"q": {**CHOICE, "criteria": many(256)}}, "1 to 255 options, got 256"),
        ("s", {"q": {**CHOICE, "criteria": {"": "empty"}}}, "non-empty strings"),
        ("s", {"q": {**CHOICE, "criteria": {"a": 3}}}, "description"),
        ("s", {"q": {"type": "score", "instructions": "x"}}, "score criteria"),
        ("s", {"q": {**SCORE, "criteria": {"0": "low"}}}, "score criteria"),
        ("s", {"q": {**SCORE, "criteria": []}}, "1 to 10 levels, got 0"),
        ("s", {"q": {**SCORE, "criteria": [str(i) for i in range(11)]}}, "1 to 10 levels, got 11"),
        ("s", {"q": {**SCORE, "criteria": ["low", None]}}, "level descriptions"),
        ("s", {"ok": NOUL, "bad": {"type": "noul", "instructions": 1.5}}, "question 'bad'"),
    ],
)
def test_validation_errors(state, questions, message):
    with pytest.raises(SchemaError, match=message):
        validate(state, questions)


@pytest.mark.parametrize(
    "state, questions",
    [
        ("s", {"q": NOUL}),
        ("s", {"q": {**NOUL, "criteria": None}}),
        ("s", {"q": {**NOUL, "criteria": {"false": "No"}}}),
        ({"a": 1}, {"q": CHOICE}),
        (["x", "y"], {"q": SCORE}),
        ("s", {"q": {**CHOICE, "criteria": {"only": None}}}),
        ("s", {"q": {**CHOICE, "criteria": many(255)}}),
        ("s", {"q": {**SCORE, "criteria": ["one level"]}}),
        ("s", {"q": {**SCORE, "criteria": [{"what": str(i)} for i in range(10)]}}),
        ("s", {"q": {"type": "choice", "instructions": {"task": "pick"}, "criteria": {"a": ["x", "y"]}}}),
        # Instructions are optional for every type (TypeSafe's SDK sends Noul() and Choice(criteria=...)).
        ("s", {"q": {"type": "noul"}}),
        ("s", {"q": {"type": "noul", "instructions": None, "criteria": {"true": "Yes"}}}),
        ("s", {"q": {"type": "choice", "criteria": {"a": None, "b": "second"}}}),
        ("s", {"q": {"type": "score", "criteria": ["low", "high"]}}),
    ],
)
def test_valid_schemas(state, questions):
    validate(state, questions)


def test_schema_error_is_value_error():
    assert issubclass(SchemaError, ValueError)


GOLDEN_QUESTIONS = {
    "wants_refund": {"type": "noul", "instructions": "Is the customer asking for money back?",
                     "criteria": {"true": "Asks for a refund"}},
    "category": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": "Charges, invoices, refunds", 'say "hi"': None}},
    "urgency": {"type": "score", "instructions": {"ask": "How urgent is this?"},
                "criteria": ["No time pressure", {"what": "Can wait days"}, "Critical outage"]},
}

GOLDEN_USER = """State:
{
  "ticket": "Charged twice, café card"
}

Questions:
q1 (yes/no): Is the customer asking for money back?
  true: Asks for a refund

q2 (choice): Which team should handle this?
  Options:
  - "billing": Charges, invoices, refunds
  - "say \\"hi\\""

q3 (score, 0 to 2): {"ask":"How urgent is this?"}
  0: No time pressure
  1: {"what":"Can wait days"}
  2: Critical outage

Answer one question per reply as a JSON object with only that question's key, for example {"q1": true}. A yes/no question takes true or false. A choice question takes one option name as a JSON string, exactly as written. A score question takes one level number."""


def test_golden_prompt():
    state = {"ticket": "Charged twice, café card"}
    assert render_user(state, GOLDEN_QUESTIONS) == GOLDEN_USER
    assert SYSTEM_PROMPT == (
        "You read a state and answer questions about it. You answer one question per reply, "
        "as a JSON object that holds only that question's key."
    )


def test_string_state_verbatim_and_both_noul_criteria():
    questions = {"x": {"type": "noul", "instructions": "Rain?", "criteria": {"false": "Dry", "true": "Wet"}}}
    text = render_user("  line one\nline two", questions)
    assert text.startswith("State:\n  line one\nline two\n\nQuestions:\nq1 (yes/no): Rain?\n  true: Wet\n  false: Dry\n\n")


def test_noul_without_criteria_has_one_line():
    text = render_user("s", {"x": NOUL})
    assert "\nq1 (yes/no): Yes?\n\nAnswer one question" in text


def test_blocks_without_instructions_show_only_their_criteria():
    questions = {"a": {"type": "noul"}, "b": {"type": "choice", "instructions": None, "criteria": {"x": None}},
                 "c": {"type": "score", "criteria": ["low", "high"]}}
    assert render_user("s", questions, rules="Own rules.") == (
        "State:\ns\n\nQuestions:\n"
        "q1 (yes/no):\n\n"
        'q2 (choice):\n  Options:\n  - "x"\n\n'
        "q3 (score, 0 to 1):\n  0: low\n  1: high\n\n"
        "Own rules.")


def test_continuations():
    assert suffix(1) == '{"q1":'
    assert suffix(12) == '{"q12":'
    assert labels(NOUL) == ["true", "false"]
    assert candidates(NOUL) == [" true}", " false}"]
    choice = {"type": "choice", "instructions": "x", "criteria": {"Sci-Tech": None, 'a "q"': None, "café": None}}
    assert labels(choice) == ["Sci-Tech", 'a "q"', "café"]
    assert candidates(choice) == [' "Sci-Tech"}', ' "a \\"q\\""}', ' "café"}']
    assert labels(SCORE) == ["0", "1"]
    assert candidates(SCORE) == [" 0}", " 1}"]
