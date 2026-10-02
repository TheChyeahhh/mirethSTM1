"""Question validation, prompt text and per-question continuations (SPEC 2.1, 3.1, 3.2)."""

import json

TYPES = ("noul", "choice", "score")
MAX_OPTIONS = 255
MAX_LEVELS = 10

# All of the answer rule there is, read once (SPEC 3.1). Measured 2026-10-01 on three models: the
# full rule after every question cost 57 tokens a question, lowered accuracy on the two small
# models and was within noise on Qwen3-4B-Instruct-2507; an example answer and the choice and
# score sentences here cost that model 1.6 points on Banking77 (77 options); without the yes/no
# sentence Qwen3-1.7B lost 5 points on SST-2.
SYSTEM_PROMPT = (
    "You read a state and answer questions about it. You answer one question per reply, "
    "as a JSON object that holds only that question's key. A yes/no question takes true or false."
)


class SchemaError(ValueError):
    """The state or the questions do not follow the TypeSafe request format."""


def _is_text(value):
    return isinstance(value, (str, dict, list))


def validate(state, questions):
    """Raise SchemaError unless `state` and `questions` follow SPEC 2.1."""
    if not isinstance(state, (str, dict, list)):
        raise SchemaError("state must be a string, an object or an array")
    if not isinstance(questions, dict) or not questions:
        raise SchemaError("questions must be a non-empty object")
    for qid, q in questions.items():
        where = f"question {qid!r}"
        if not isinstance(qid, str):
            raise SchemaError(f"{where}: id must be a string")
        if not isinstance(q, dict):
            raise SchemaError(f"{where}: must be an object")
        kind = q.get("type")
        if kind not in TYPES:
            raise SchemaError(f"{where}: unknown type {kind!r} (expected noul, choice or score)")
        if q.get("instructions") is not None and not _is_text(q["instructions"]):  # optional; null is absent
            raise SchemaError(f"{where}: instructions must be a string, an object or an array")
        criteria = q.get("criteria")
        if kind == "noul":
            if criteria is None:
                pass  # criteria are optional
            elif not isinstance(criteria, dict) or not set(criteria) <= {"true", "false"}:
                raise SchemaError(f'{where}: noul criteria must be an object with keys "true" and/or "false"')
            elif not all(d is None or _is_text(d) for d in criteria.values()):
                raise SchemaError(f"{where}: descriptions must be strings, objects, arrays or null")
        elif kind == "choice":
            if not isinstance(criteria, dict):
                raise SchemaError(f"{where}: choice criteria must be an object of option name to description")
            if not 1 <= len(criteria) <= MAX_OPTIONS:
                raise SchemaError(f"{where}: choice needs 1 to {MAX_OPTIONS} options, got {len(criteria)}")
            for name, desc in criteria.items():
                if not isinstance(name, str) or not name:
                    raise SchemaError(f"{where}: option names must be non-empty strings")
                if desc is not None and not _is_text(desc):
                    raise SchemaError(f"{where}: option {name!r}: description must be a string, an object, an array or null")
        else:
            if not isinstance(criteria, list):
                raise SchemaError(f"{where}: score criteria must be an array of level descriptions")
            if not 1 <= len(criteria) <= MAX_LEVELS:
                raise SchemaError(f"{where}: score needs 1 to {MAX_LEVELS} levels, got {len(criteria)}")
            if not all(_is_text(level) for level in criteria):
                raise SchemaError(f"{where}: level descriptions must be strings, objects or arrays")


def text(value):
    """A string as is; anything else as compact JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _block(k, q):
    kind, criteria = q["type"], q.get("criteria")
    ask = "" if q.get("instructions") is None else " " + text(q["instructions"])  # without it: "q1 (yes/no):"
    if kind == "noul":
        lines = [f"q{k} (yes/no):{ask}"]
        for key in ("true", "false"):
            if (criteria or {}).get(key) is not None:
                lines.append(f"  {key}: {text(criteria[key])}")
    elif kind == "choice":
        lines = [f"q{k} (choice):{ask}", "  Options:"]
        for name, desc in criteria.items():
            option = json.dumps(name, ensure_ascii=False)
            lines.append(f"  - {option}" if desc is None else f"  - {option}: {text(desc)}")
    else:
        lines = [f"q{k} (score, 0 to {len(criteria) - 1}):{ask}"]
        lines += [f"  {i}: {text(level)}" for i, level in enumerate(criteria)]
    return "\n".join(lines)


def state_text(state):
    """A string state verbatim; an object or an array as indented JSON."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=2)


def shared_text(state):
    """The start of the user message, which every question of a call shares (SPEC 3.1)."""
    return f"State:\n{state_text(state)}\n\n"


def branch_text(q):
    """The rest of the user message in one question's private branch: the question, alone under
    the key q1 (SPEC 3.1)."""
    return f"Question:\n{_block(1, q)}"


def render_user(state, questions, rules):
    """One user message holding every question, numbered q1, q2, ... in request order.

    This is the normal-generation baseline's prompt (SPEC 10.2); the scorer never puts two
    questions in one prompt (SPEC 3.1).
    """
    blocks = "\n\n".join(_block(k, q) for k, q in enumerate(questions.values(), 1))
    return f"State:\n{state_text(state)}\n\nQuestions:\n{blocks}\n\n{rules}"


def labels(q):
    """Answer labels in request order: true/false, option names, or level indices as strings."""
    if q["type"] == "noul":
        return ["true", "false"]
    if q["type"] == "choice":
        return list(q["criteria"])
    return [str(i) for i in range(len(q["criteria"]))]


# Text shared by all labels of a question: the model starts writing {"q1": ... (SPEC 3.2)
SUFFIX = '{"q1":'


def candidates(q):
    """Continuation text per label, aligned with labels(q). The closing brace ends the label."""
    if q["type"] == "noul":
        return [" true}", " false}"]
    if q["type"] == "choice":
        return [" " + json.dumps(name, ensure_ascii=False) + "}" for name in q["criteria"]]
    return [f" {i}}}" for i in range(len(q["criteria"]))]
