"""The Tarnlight tap (SPEC 10.3). No model: the scorer is stubbed. conftest.py points the home folder
at the test's own tmp_path, so the real ~/.tarnlight is never touched."""

import json
import math
import re
import stat
import sys
import time
from types import SimpleNamespace

import pytest

from mirethstm import Engine
from mirethstm import tarnlight
from mirethstm.engine import systemone_body

STATE = {"ticket": "Charged twice for my café order, please refund 12.50 €"}
SCHEMA = {
    "refund": {"type": "noul", "instructions": "Is the customer asking for money back?"},
    "team": {"type": "choice", "instructions": "Which team?",
             "criteria": {"billing": "Charges and refunds", "bug": None, "café": None}},
    "urgency": {"type": "score", "instructions": "How urgent?",
                "criteria": ["Can wait", "Today", {"level": "Outage"}]},
}
SCORES = {"refund": {"true": -0.31, "false": -1.42},
          "team": {"billing": -0.12, "bug": -3.7, "café": -2.05},
          "urgency": {"0": -2.2, "1": -0.4, "2": -1.3}}
NAME = re.compile(r"^mirethstm-\d{4}-\d{2}-\d{2}T\d{2}\.jsonl$")

# Tarnlight's own checks at the door, restated: the envelope keys it keeps, which of them must be
# text and which numbers, and the longest line its drop box reads.
ENVELOPE = {"v", "ts", "seq", "source", "session_id", "tool_use_id", "project", "label", "sdk", "request_id",
            "latency_ms", "status", "retry_count", "cost_est_micro", "request", "response", "error"}
TEXT_FIELDS = ("source", "session_id", "tool_use_id", "project", "label", "sdk", "request_id")
NUMBER_FIELDS = ("latency_ms", "status", "retry_count", "cost_est_micro")
MAX_LINE = 16 << 20


def finite(x):
    return not isinstance(x, bool) and (isinstance(x, int) or isinstance(x, float) and math.isfinite(x))


def door(line):
    """The record in one drop-box line, asserting that Tarnlight would store it and chart every answer."""
    assert len(line) <= MAX_LINE

    def refuse(name):
        raise AssertionError(f"{name} is not JSON")

    rec = json.loads(line.decode("utf-8"), parse_constant=refuse)
    assert isinstance(rec, dict) and set(rec) <= ENVELOPE
    assert type(rec["v"]) is int and rec["v"] == 1
    assert finite(rec["ts"])
    assert all(rec.get(k) is None or isinstance(rec[k], str) for k in TEXT_FIELDS)
    assert all(rec.get(k) is None or finite(rec[k]) for k in NUMBER_FIELDS)
    for answer in rec["response"]["answers"].values():  # what its indexer reads per answer type
        need = {"noul": ["noul"], "choice": ["choice", "probabilities", "confidence"],
                "score": ["score", "probabilities", "confidence"]}[answer["type"]]
        assert all(key in answer for key in need)
        assert 0.0 <= answer.get("confidence", 0.0) <= 1.0
    return rec


def numbers(value):
    if isinstance(value, dict):
        return [n for v in value.values() for n in numbers(v)]
    return [value] if isinstance(value, (int, float)) and not isinstance(value, bool) else []


def stub_engine(**settings):
    engine = Engine(None, None, "stub/model", **settings)
    engine._run = lambda context, schema: (SCORES, 470)  # stands in for the model
    return engine


def inbox(home):
    return home / ".tarnlight" / "inbox"


def test_no_inbox_no_write_and_no_folder(home):
    result = stub_engine().decide(STATE, SCHEMA)
    assert result["answers"]["team"]["choice"] == "billing"
    assert not (home / ".tarnlight").exists()


def test_one_valid_line_per_decide(home):
    inbox(home).mkdir(parents=True)
    engine = stub_engine()
    start = time.time()
    results = [engine.decide(STATE, SCHEMA), engine.decide(STATE, SCHEMA)]
    end = time.time()
    records = []
    for path in sorted(inbox(home).iterdir()):  # two files only if the UTC hour turned between the calls
        assert NAME.match(path.name)
        data = path.read_bytes()
        assert data.endswith(b"\n") and b"\r" not in data
        for line in data.split(b"\n")[:-1]:
            rec = door(line)
            assert time.strftime("%Y-%m-%dT%H", time.gmtime(rec["ts"])) == path.name[-19:-6]
            records.append(rec)
    assert len(records) == 2
    for rec, result in zip(sorted(records, key=lambda r: r["ts"]), results):
        assert list(rec) == ["v", "ts", "source", "request_id", "request", "response", "latency_ms",
                             "status", "cost_est_micro"]
        assert (rec["v"], rec["source"], rec["status"], rec["cost_est_micro"]) == (1, "mirethstm", 200, 0)
        assert start <= rec["ts"] <= end
        assert rec["request_id"] == result["id"] and rec["latency_ms"] == result["latency_ms"]
        assert rec["request"] == {"model": "stub/model", "state": STATE, "questions": SCHEMA}
        assert rec["response"] == systemone_body(result)
        assert list(rec["response"]) == ["model", "answers", "usage"]
        assert all(round(n, 2) == n for n in numbers(rec["response"]))
    assert records[0]["request_id"] != records[1]["request_id"]


def test_low_disk_skips(home, monkeypatch):
    inbox(home).mkdir(parents=True)
    asked = []

    def usage(free):
        def disk_usage(path):
            asked.append(path)
            return SimpleNamespace(total=free * 2, used=free, free=free)
        return disk_usage

    monkeypatch.setattr("shutil.disk_usage", usage((1 << 30) - 1))
    stub_engine().decide(STATE, SCHEMA)
    assert list(inbox(home).iterdir()) == []
    assert asked == [inbox(home)]  # the disk that holds the drop box
    monkeypatch.setattr("shutil.disk_usage", usage(1 << 30))
    stub_engine().decide(STATE, SCHEMA)
    assert len(list(inbox(home).iterdir())) == 1


@pytest.mark.parametrize("error", [OSError("disk full"), RuntimeError("anything else")])
def test_failed_write_never_fails_the_decision(home, monkeypatch, error):
    inbox(home).mkdir(parents=True)

    def broken(path, data):
        raise error

    monkeypatch.setattr(tarnlight, "append_line", broken)
    result = stub_engine().decide(STATE, SCHEMA)
    assert list(result["answers"]) == list(SCHEMA) and result["usage"]["input_tokens"] == 470


def test_off_writes_nothing(home):
    inbox(home).mkdir(parents=True)
    stub_engine(tarnlight=False).decide(STATE, SCHEMA)
    assert list(inbox(home).iterdir()) == []


def test_line_longer_than_tarnlight_reads_is_skipped(home, monkeypatch):
    inbox(home).mkdir(parents=True)
    monkeypatch.setattr(tarnlight, "MAX_LINE_BYTES", 200)
    stub_engine().decide(STATE, SCHEMA)
    assert list(inbox(home).iterdir()) == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_new_file_is_private(home):
    inbox(home).mkdir(parents=True)
    stub_engine().decide(STATE, SCHEMA)
    (path,) = inbox(home).iterdir()
    assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0  # only the owner may read the caller's state


def test_appends_to_an_existing_file(home):
    inbox(home).mkdir(parents=True)
    path = inbox(home) / "mirethstm-2026-10-02T14.jsonl"
    path.write_bytes(b'{"v":1,"ts":1}\n')
    tarnlight.append_line(path, b'{"v":1,"ts":2}\n')
    tarnlight.append_line(path, b'{"v":1,"ts":3}\n')
    assert path.read_bytes() == b'{"v":1,"ts":1}\n{"v":1,"ts":2}\n{"v":1,"ts":3}\n'
