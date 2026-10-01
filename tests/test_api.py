"""The local HTTP API on the console's server (SPEC 6) and the approved models (SPEC 11.1).

A real server on an ephemeral port with a stub engine, unless marked `model`.
"""

import dataclasses
import http.client
import json
import re
import threading
import time

import pytest

from mirethstm import baseline, console, models
from mirethstm.engine import systemone_body
from mirethstm.models import approved, entry

QUESTIONS = {  # the SPEC 2.1 example
    "wants_refund": {"type": "noul", "instructions": "Is the customer asking for money back?",
                     "criteria": {"true": "Asks for a refund", "false": "Does not"}},
    "category": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": "Charges, invoices, refunds", "bug": "Crashes", "other": None}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["No time pressure", "Can wait days", "Needs attention today", "Critical outage"]},
}
STATE = {"ticket": {"subject": "Charged twice for my subscription",
                    "body": "I was billed two times this month and the app also crashes when I open settings."}}
REQUEST = {"model": "jev-latest", "state": STATE, "questions": QUESTIONS}
RESULT = {  # full precision, as Engine.decide returns it
    "model": "stub/a",
    "answers": {
        "wants_refund": {"type": "noul", "noul": 0.583333},
        "category": {"type": "choice", "choice": "billing", "confidence": 0.521234,
                     "probabilities": {"billing": 0.680822, "bug": 0.041111, "other": 0.278067}},
        "urgency": {"type": "score", "score": 1.968, "confidence": 0.933333,
                    "legend": {"0": "No time pressure", "1": "Can wait days", "2": "Needs attention today",
                               "3": "Critical outage"},
                    "probabilities": {"0": 0.0021, "1": 0.0412, "2": 0.9455, "3": 0.0112}},
    },
    "usage": {"input_tokens": 470, "output_tokens": 0},
    "id": "3f2a" * 8,
    "latency_ms": 12.3456,
}
# What the official TypeSafe Python SDK (typesafe-sdk 0.7.2) sends with every call.
SDK_HEADERS = {"Authorization": "Bearer not-a-real-key", "Accept": "application/json",
               "User-Agent": "typesafe-sdk/0.7.2", "X-TypeSafe-SDK": "typesafe-sdk/0.7.2",
               "X-TypeSafe-Runtime": "python/3.11.9 (win32; AMD64)"}


class StubEngine:
    def __init__(self, model_id, result=RESULT, fail=None):
        self.model_id = model_id
        self.result = result
        self.fail = fail
        self.calls = []

    def decide(self, context, schema):
        self.calls.append((context, schema))
        if self.fail:
            raise self.fail
        return {**self.result, "model": self.model_id}


class HeldRace:
    """Stands in for baseline.generate and holds the page's run open until `release` is set."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, engine, context, schema, on_text=None):
        self.started.set()
        self.release.wait(10)
        return {"text": "{}", **baseline.check("{}", schema), "latency_ms": 50.0, "output_tokens": 1}


def start(engine, generate=None):
    server = console.ConsoleServer(("127.0.0.1", 0), lambda model: engine, [engine.model_id],
                                   generate=generate or HeldRace())
    server.engine = engine
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    return server


@pytest.fixture
def server():
    server = start(StubEngine("stub/a"))
    yield server
    server.shutdown()
    server.server_close()


def call(server, method, path, body=None, headers=None):
    """(status, response headers, body): the body parsed when it is JSON, else the raw bytes."""
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=30)
    headers = dict(headers or {})
    if body is not None and not isinstance(body, bytes):
        body = json.dumps(body).encode()
    if body is not None:
        headers.setdefault("Content-Type", "application/json")
    conn.request(method, path, body=body, headers=headers)
    response = conn.getresponse()
    raw = response.read()
    is_json = response.headers.get_content_type() == "application/json"
    return response.status, response.headers, json.loads(raw) if is_json else raw


def floats(value):
    if isinstance(value, float):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from floats(v)


def assert_error(got, status, kind):
    assert got[0] == status
    assert got[1].get_content_type() == "application/json"
    assert set(got[2]) == {"error"} and got[2]["error"]["type"] == kind
    assert isinstance(got[2]["error"]["message"], str) and got[2]["error"]["message"]


# --- routes --------------------------------------------------------------------------------


def test_systemone_is_typesafe_shaped_and_rounded(server):
    status, headers, body = call(server, "POST", "/v1/systemone", REQUEST, SDK_HEADERS)
    assert status == 200 and headers.get_content_type() == "application/json"
    assert body == systemone_body(RESULT)  # model, answers, usage; no id or latency in the body
    assert body["answers"]["category"]["probabilities"] == {"billing": 0.68, "bug": 0.04, "other": 0.28}
    assert all(round(x, 2) == x for x in floats(body))
    assert headers["x-request-id"] == headers["x-typesafe-request-id"] == RESULT["id"]  # the SDK reads the second
    assert headers["server-timing"] == "decide;dur=12.3"
    # The request's model is ignored; the loaded model answers and the call reaches the engine unchanged.
    assert body["model"] == "stub/a"
    assert server.engine.calls == [(STATE, QUESTIONS)]


def test_decide_keeps_full_precision(server):
    status, headers, body = call(server, "POST", "/v1/decide", REQUEST)
    assert (status, body) == (200, RESULT)
    assert headers["x-request-id"] == RESULT["id"] and headers["server-timing"] == "decide;dur=12.3"


@pytest.mark.parametrize("extra", [{}, {"model": "mirethstm"}, {"model": 7}, {"model": None}])
def test_model_field_is_accepted_and_ignored(server, extra):
    status, _, body = call(server, "POST", "/v1/systemone", {"state": "x", "questions": QUESTIONS, **extra})
    assert status == 200 and body["model"] == "stub/a"


def test_questions_without_instructions_are_accepted(server):
    # What TypeSafe's SDK sends for Noul(), Noul(criteria=...), Choice(criteria=...) and Score(criteria=...).
    questions = {"bare": {"type": "noul"}, "described": {"type": "noul", "criteria": {"true": "Yes", "false": "No"}},
                 "team": {"type": "choice", "criteria": {"billing": None, "bug": "Crashes"}},
                 "urgency": {"type": "score", "criteria": ["Low", "High"]}}
    status, _, _ = call(server, "POST", "/v1/systemone", {"state": "x", "questions": questions}, SDK_HEADERS)
    assert status == 200 and server.engine.calls == [("x", questions)]


@pytest.mark.parametrize("headers", [
    SDK_HEADERS,
    {"Content-Type": "application/json; charset=utf-8"},
    {"Authorization": "Bearer sk-anything", "Host": "localhost:8766"},
    {"Authorization": "Basic Zm9vOmJhcg=="},
])
def test_client_headers_pass_the_guards(server, headers):
    assert call(server, "POST", "/v1/systemone", REQUEST, headers)[0] == 200


def test_models_lists_the_approved_models(server):
    status, _, body = call(server, "GET", "/v1/models", headers=SDK_HEADERS)
    assert status == 200
    assert body == {"models": [entry(m.id) for m in approved()], "current": "stub/a"}
    for listed in body["models"]:  # what TypeSafe's SDK requires of each model, as strict strings
        assert all(isinstance(listed[key], str) for key in ("name", "description", "release_date"))
        assert re.fullmatch(r"\d{4}-\d\d-\d\d", listed["release_date"])


# --- errors --------------------------------------------------------------------------------


@pytest.mark.parametrize("body", [
    b"not json", b"[1]", b'"text"', {}, {"questions": QUESTIONS}, {"state": "x"},
    {"state": 42, "questions": QUESTIONS},
    {"state": "x", "questions": {}},
    {"state": "x", "questions": {"q": {"type": "bool", "instructions": "x"}}},
    {"state": "x", "questions": {"q": {"type": "choice", "instructions": "x",
                                       "criteria": {f"o{i}": None for i in range(256)}}}},
    {"state": "x", "questions": {"q": {"type": "score", "instructions": "x", "criteria": []}}},
])
@pytest.mark.parametrize("path", console.API_ROUTES)
def test_invalid_requests_are_422(server, path, body):
    assert_error(call(server, "POST", path, body), 422, "invalid_request")
    assert server.engine.calls == [] and not server.lock.locked()


def test_too_large_is_413(server):
    too_long = {"Content-Type": "application/json", "Content-Length": str(console.MAX_BODY + 1)}
    assert_error(call(server, "POST", "/v1/systemone", headers=too_long), 413, "too_large")
    assert server.engine.calls == []


def test_the_page_guards_still_apply(server):
    form = {"Content-Type": "application/x-www-form-urlencoded"}
    assert call(server, "POST", "/v1/systemone", b"state=x", form)[0] == 415  # no cross-site form posts
    rebound = {"Host": "attacker.example:8766", **SDK_HEADERS}
    assert call(server, "POST", "/v1/systemone", REQUEST, rebound)[0] == 403  # no DNS rebinding
    assert call(server, "GET", "/v1/models", headers=rebound)[0] == 403
    assert call(server, "POST", "/v1/other", REQUEST)[0] == 404
    assert server.engine.calls == []


def test_api_waits_for_the_running_race(server):
    """An API call during a page run is not refused: it waits, then runs."""
    race = server.generate
    page = threading.Thread(target=call, args=(server, "POST", "/api/run",
                                               {"scenario": "support", "state": "first"}))
    page.start()
    assert race.started.wait(10)
    got = {}
    api = threading.Thread(target=lambda: got.update(r=call(server, "POST", "/v1/systemone", REQUEST)))
    api.start()
    api.join(0.5)
    assert api.is_alive()  # still waiting for the race
    race.release.set()
    api.join(10)
    page.join(10)
    assert got["r"][0] == 200
    assert [context for context, _ in server.engine.calls] == ["first", STATE]


def test_busy_past_the_wait_is_529(server):
    server.api_wait = 0.2
    with server.lock:  # a run or a model load that does not finish in time
        began = time.monotonic()
        got = call(server, "POST", "/v1/systemone", REQUEST)
        waited = time.monotonic() - began
    assert_error(got, 529, "busy")
    assert waited >= 0.2 and server.engine.calls == []


def test_no_model_is_529(server):
    server.engine = None
    assert_error(call(server, "POST", "/v1/systemone", REQUEST), 529, "no_model")
    assert call(server, "GET", "/v1/models")[2]["current"] is None


def test_engine_failure_is_500_and_frees_the_lock():
    server = start(StubEngine("stub/a", fail=RuntimeError("out of memory")))
    try:
        got = call(server, "POST", "/v1/decide", REQUEST)
        assert_error(got, 500, "internal_error")
        assert "out of memory" in got[2]["error"]["message"] and not server.lock.locked()
    finally:
        server.shutdown()
        server.server_close()


def test_non_finite_numbers_never_reach_the_client():
    broken = {**RESULT, "answers": {"wants_refund": {"type": "noul", "noul": float("nan")}}}
    server = start(StubEngine("stub/a", result=broken))
    try:
        assert_error(call(server, "POST", "/v1/decide", REQUEST), 500, "internal_error")
    finally:
        server.shutdown()
        server.server_close()


# --- approved models (SPEC 11.1) -------------------------------------------------------------


def test_candidate_list_follows_the_spec():
    assert {m.id for m in models.MODELS} == {
        "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B-Instruct-2507", "microsoft/Phi-4-mini-instruct", "HuggingFaceTB/SmolLM3-3B",
        "ibm-granite/granite-3.3-2b-instruct"}
    assert len({m.id for m in models.MODELS}) == len(models.MODELS)
    for m in models.MODELS:
        assert m.license in ("Apache-2.0", "MIT")
        assert m.params * 2 < 10  # bf16 weights in GB, well under 10
        assert re.fullmatch(r"\d{4}-\d\d-\d\d", m.release_date)
        assert chr(0x2014) not in m.name + m.role + models.note(m)


def test_approved_models_wait_for_their_numbers():
    assert {m.id: m.status for m in approved()} == dict.fromkeys(
        ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-0.6B"],
        models.PENDING)
    measured = ("ms_28_fields", "ms_router", "accuracy", "ece", "temperature", "measured_on")
    assert all(getattr(m, key) is None for m in models.MODELS for key in measured)
    assert all(m.status == models.CANDIDATE for m in models.MODELS if m not in approved())
    assert models.get("Qwen/Qwen2.5-3B-Instruct") is None  # non-commercial license: never listed


def test_note_shows_numbers_once_measured():
    m = models.get("Qwen/Qwen2.5-1.5B-Instruct")
    assert models.note(m) == "Match first, not approved yet"
    done = dataclasses.replace(m, status=models.APPROVED, ms_28_fields=189.6, accuracy=0.912, ece=0.03,
                               temperature=1.4, ms_router=401.0, measured_on="2026-10-03")
    assert models.note(done) == "Match first, 190 ms for 28 fields, 91% accuracy"
    assert models.note(models.get("HuggingFaceTB/SmolLM3-3B")) == "Not evaluated yet"


# --- real model --------------------------------------------------------------------------------


@pytest.mark.model
def test_real_api_calls(engine):
    server = start(engine)
    try:
        status, headers, body = call(server, "POST", "/v1/systemone", REQUEST, SDK_HEADERS)
        decided = call(server, "POST", "/v1/decide", REQUEST)
    finally:
        server.shutdown()
        server.server_close()
    assert status == 200 and decided[0] == 200
    assert re.fullmatch(r"[0-9a-f]{32}", headers["x-request-id"])
    assert headers["x-typesafe-request-id"] == headers["x-request-id"]
    assert re.fullmatch(r"decide;dur=\d+\.\d", headers["server-timing"])
    assert set(body) == {"model", "answers", "usage"} and body["model"] == engine.model_id
    answers = body["answers"]
    assert list(answers) == list(QUESTIONS)
    assert set(answers["wants_refund"]) == {"type", "noul"} and 0 <= answers["wants_refund"]["noul"] <= 1
    assert list(answers["category"]["probabilities"]) == ["billing", "bug", "other"]
    assert answers["urgency"]["legend"] == {str(i): level for i, level in enumerate(QUESTIONS["urgency"]["criteria"])}
    for a in (answers["category"], answers["urgency"]):
        assert sum(a["probabilities"].values()) == pytest.approx(1, abs=0.03)
    assert all(round(x, 2) == x for x in floats(body))
    assert isinstance(body["usage"]["input_tokens"], int) and body["usage"]["output_tokens"] == 0
    full = decided[2]
    assert re.fullmatch(r"[0-9a-f]{32}", full["id"]) and full["latency_ms"] > 0
    rounded = systemone_body(full)
    assert rounded["answers"]["category"]["choice"] == answers["category"]["choice"]
    assert all(abs(a - b) <= 0.0100001 for a, b in zip(floats(rounded), floats(body)))
