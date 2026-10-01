"""The FROZEN event log (SPEC 8)."""

import json
import re

import pytest

from mirethstm import Engine
from mirethstm.events import write_events

KEYS = ["ts", "id", "field", "label", "p", "probs", "T", "latency_ms", "model"]
TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
CALL_ID = "0123456789abcdef0123456789abcdef"


def read_lines(path):
    data = path.read_bytes()
    assert b"\r" not in data and data.endswith(b"\n")
    return [json.loads(line) for line in data.decode("utf-8").splitlines()]


def test_lines_keys_and_append(tmp_path):
    path = tmp_path / "events.jsonl"
    probs = {
        "category": {"billing": 0.25, "bug": 0.7, "café": 0.05},
        "urgent": {"true": 0.5, "false": 0.5},
    }
    write_events(path, CALL_ID, probs, 1.5, 12.5, "some/model")
    write_events(path, "f" * 32, {"category": probs["category"]}, 1.5, 3.0, "some/model")
    lines = read_lines(path)
    assert len(lines) == 3
    for line in lines:
        assert list(line) == KEYS
        assert TS.match(line["ts"])
        assert line["p"] == max(line["probs"].values())
        assert line["T"] == 1.5 and line["model"] == "some/model"
    first, second, third = lines
    assert first["id"] == second["id"] == CALL_ID and third["id"] == "f" * 32
    assert first["ts"] == second["ts"]
    assert (first["field"], first["label"], first["p"]) == ("category", "bug", 0.7)
    assert list(first["probs"]) == ["billing", "bug", "café"]
    assert (second["field"], second["label"]) == ("urgent", "true")  # tie goes to the earlier label
    assert first["latency_ms"] == second["latency_ms"] == 12.5
    assert "café" in path.read_text(encoding="utf-8")  # UTF-8, not escaped


def test_one_write_per_call(tmp_path, monkeypatch):
    writes = []

    class Recorder:
        def __init__(self, path, mode):
            assert mode == "ab"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def write(self, data):
            writes.append(data)

    monkeypatch.setattr("mirethstm.events.open", Recorder, raising=False)
    probs = {f"f{i}": {"a": 0.9, "b": 0.1} for i in range(5)}
    write_events(tmp_path / "x.jsonl", CALL_ID, probs, 1.0, 1.0, "m")
    assert len(writes) == 1 and writes[0].count(b"\n") == 5


@pytest.mark.model
def test_decide_writes_events(engine, tmp_path):
    path = tmp_path / "events.jsonl"
    logged = Engine(engine.model, engine.tokenizer, engine.model_id, chunk_size=3,
                    temperature=1.3, event_log=str(path))
    schema = {
        "is_sport": {"type": "noul", "instructions": "Is this text about sport?"},
        "topic": {"type": "choice", "instructions": "Topic?", "criteria": {"Sports": None, "Business": None, "World": None}},
        "only": {"type": "choice", "instructions": "Pick one", "criteria": {"single": None}},
        "level": {"type": "score", "instructions": "How exciting?", "criteria": ["Dull", "Fine", "Thrilling"]},
    }
    result = logged.decide("The striker scored twice in the cup final.", schema)
    lines = read_lines(path)
    assert [line["field"] for line in lines] == list(schema)
    for line in lines:
        assert list(line) == KEYS
        assert TS.match(line["ts"])
        assert line["id"] == result["id"]
        assert line["p"] == max(line["probs"].values())
        assert line["T"] == 1.3
        assert line["latency_ms"] == result["latency_ms"]
        assert line["model"] == engine.model_id
    by_field = {line["field"]: line for line in lines}
    answers = result["answers"]
    assert by_field["is_sport"]["probs"]["true"] == answers["is_sport"]["noul"]
    assert list(by_field["is_sport"]["probs"]) == ["true", "false"]
    assert by_field["topic"]["label"] == answers["topic"]["choice"]
    assert by_field["topic"]["probs"] == answers["topic"]["probabilities"]
    only = by_field["only"]
    assert (only["label"], only["p"], only["probs"]) == ("single", 1.0, {"single": 1.0})
    assert by_field["level"]["probs"] == answers["level"]["probabilities"]
    assert by_field["level"]["label"] == max(answers["level"]["probabilities"], key=answers["level"]["probabilities"].get)
