"""Console race view (SPEC 10.1): a real server on an ephemeral port, stub engine unless marked `model`."""

import http.client
import json
import socket
import threading
import time

import pytest

from mirethstm import baseline, console
from mirethstm.scenarios import SCENARIOS
from mirethstm.schema import validate

SMALL = {
    "id": "small",
    "title": "Small (3 fields)",
    "state": "The home team won the football match 3-1 after their striker scored twice.",
    "questions": {
        "sport": {"type": "noul", "instructions": "Is this text about sport?"},
        "section": {"type": "choice", "instructions": "Which section fits this text?",
                    "criteria": {"Sports": None, "Business": None, "Sci-Tech": None}},
        "tone": {"type": "score", "instructions": "How positive is the text?",
                 "criteria": ["Negative", "Neutral", "Positive"]},
    },
}
DECIDED = {
    "sport": {"type": "noul", "noul": 0.9},
    "section": {"type": "choice", "choice": "Sports", "confidence": 0.7,
                "probabilities": {"Sports": 0.8, "Business": 0.1, "Sci-Tech": 0.1}},
    "tone": {"type": "score", "score": 1.5, "confidence": 0.4,
             "legend": {"0": "Negative", "1": "Neutral", "2": "Positive"},
             "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}},
}
CHUNKS = ['{"q1"', ": true,", ' "q2": "Sports", "q3": 5}']


class StubEngine:
    def __init__(self, model_id, fail=False):
        self.model_id = model_id
        self.fail = fail
        self.calls = []

    def decide(self, context, schema):
        self.calls.append((context, schema))
        if self.fail:
            raise RuntimeError("out of memory")
        return {"model": self.model_id, "answers": DECIDED, "usage": {"input_tokens": 9, "output_tokens": 0},
                "id": "0" * 32, "latency_ms": 10.0}


class StubGenerate:
    """Stands in for baseline.generate; `release` can hold a run open."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def __call__(self, engine, context, schema, on_text=None):
        self.started.set()
        self.release.wait(10)
        for chunk in CHUNKS:
            on_text(chunk)
        return {"text": "".join(CHUNKS), **baseline.check("".join(CHUNKS), schema),
                "latency_ms": 50.0, "output_tokens": 12}


def start(load, models, generate, scenarios=(SMALL,)):
    server = console.ConsoleServer(("127.0.0.1", 0), load, models, scenarios=scenarios, generate=generate)
    server.engine = load(models[0])
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    return server


@pytest.fixture
def server():
    def load(model):
        if model == "broken/model":
            raise OSError("not in the cache")
        return StubEngine(model)

    server = start(load, ["stub/a", "stub/b", "broken/model"], StubGenerate())
    yield server
    server.shutdown()
    server.server_close()


def request(server, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=30)
    headers = dict(headers or {})
    if body is not None and not isinstance(body, bytes):
        body = json.dumps(body).encode()
    if body is not None:
        headers.setdefault("Content-Type", "application/json")
    conn.request(method, path, body=body, headers=headers)
    response = conn.getresponse()
    return response.status, response.getheader("Content-Type"), response.read()


def events(raw):
    out = []
    for frame in raw.decode("utf-8").split("\n\n"):
        if frame:
            fields = dict(line.split(": ", 1) for line in frame.split("\n"))
            out.append((fields["event"], json.loads(fields["data"])))
    return out


# --- page and static files --------------------------------------------------------------


def test_page_is_served(server):
    status, kind, body = request(server, "GET", "/")
    assert status == 200 and kind.startswith("text/html")
    assert b"Run comparison" in body and b'src="app.js"' in body and b'href="style.css"' in body
    assert request(server, "GET", "/app.js")[1].startswith("text/javascript")
    assert request(server, "GET", "/style.css")[1].startswith("text/css")


@pytest.mark.parametrize("path", [
    "/../console.py", "/..%2fconsole.py", "/%2e%2e/console.py", "/web/../console.py", "/..\\console.py",
    "//console.py", "/C:/Windows/win.ini", "/index.html/../../console.py", "/missing.js",
])
def test_static_paths_cannot_escape(server, path):
    status, _, body = request(server, "GET", path)
    assert status == 404
    assert b"import" not in body


def test_other_host_names_are_refused(server):
    status, _, _ = request(server, "GET", "/api/models", headers={"Host": "attacker.example:8766"})
    assert status == 403
    assert request(server, "GET", "/api/models", headers={"Host": "localhost:8766"})[0] == 200


# --- scenarios and models ----------------------------------------------------------------


def test_builtin_scenarios_are_valid():
    sizes = {s["id"]: len(s["questions"]) for s in SCENARIOS}
    assert sizes == {"support": 28, "security_review": 28, "incident": 20, "router": 4}
    for s in SCENARIOS:
        validate(s["state"], s["questions"])
        assert s["title"].endswith(f"({len(s['questions'])} fields)")
    incident = SCENARIOS[2]["questions"].values()
    assert sum(q["type"] == "score" for q in incident) >= 5
    route = SCENARIOS[3]["questions"]["route"]["criteria"]
    assert len(route) == 255 and len(set(route)) == 255


def test_scenarios_endpoint(server):
    status, kind, body = request(server, "GET", "/api/scenarios")
    assert status == 200 and kind.startswith("application/json")
    assert json.loads(body) == [SMALL]


def test_models_and_switch(server):
    assert json.loads(request(server, "GET", "/api/models")[2]) == {
        "models": ["stub/a", "stub/b", "broken/model"], "current": "stub/a"}
    status, _, body = request(server, "POST", "/api/model", {"model": "stub/b"})
    assert (status, json.loads(body)) == (200, {"current": "stub/b"})
    assert server.engine.model_id == "stub/b"

    status, _, body = request(server, "POST", "/api/model", {"model": "some/other"})
    assert status == 400 and server.engine.model_id == "stub/b"

    status, _, body = request(server, "POST", "/api/model", {"model": "broken/model"})
    error = json.loads(body)["error"]
    assert status == 500 and "not in the cache" in error["message"]
    assert error["current"] == "stub/b" and server.engine.model_id == "stub/b"  # the previous model is back


def test_default_model_list():
    assert console.MODELS[0] == console.DEFAULT_MODEL == "Qwen/Qwen2.5-1.5B-Instruct"
    assert set(console.MODELS) == {"Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen3-1.7B",
                                   "Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-0.6B"}


# --- runs ----------------------------------------------------------------------------------


def test_display_lines():
    assert console.display({"answers": DECIDED}) == {
        "sport": {"value": True, "prob": 0.9},
        "section": {"value": "Sports", "prob": 0.8},
        "tone": {"value": 2, "prob": 0.6, "score": 1.5},
    }
    low = console.display({"answers": {"x": {"type": "noul", "noul": 0.25}}})
    assert low == {"x": {"value": False, "prob": 0.75}}


def test_run_streams_events_in_order(server):
    status, kind, raw = request(server, "POST", "/api/run", {"scenario": "small", "state": "Edited text"})
    assert status == 200 and kind.startswith("text/event-stream")
    got = events(raw)
    assert [name for name, _ in got] == ["decide", "text", "text", "text", "baseline", "summary"]

    decided = got[0][1]
    assert decided["latency_ms"] == 10.0 and decided["answers"] == DECIDED
    assert decided["display"] == console.display(decided)
    assert [data["text"] for name, data in got if name == "text"] == CHUNKS
    generated = got[4][1]
    assert generated["text"] == "".join(CHUNKS)
    assert generated["hallucinated"] == ["tone"] and generated["missing"] == []
    assert got[5][1] == {"decide_ms": 10.0, "baseline_ms": 50.0, "speedup": 5.0}
    assert server.engine.calls == [("Edited text", SMALL["questions"])]


def test_bad_run_requests(server):
    assert request(server, "POST", "/api/run", {"scenario": "nope", "state": "x"})[0] == 400
    assert request(server, "POST", "/api/run", {"scenario": "small", "state": 42})[0] == 422
    assert request(server, "POST", "/api/run", b"not json")[0] == 400
    assert request(server, "POST", "/api/run", b"[1]")[0] == 400
    form = {"Content-Type": "application/x-www-form-urlencoded"}
    assert request(server, "POST", "/api/run", b"scenario=small", headers=form)[0] == 415
    too_long = {"Content-Type": "application/json", "Content-Length": str(console.MAX_BODY + 1)}
    assert request(server, "POST", "/api/run", headers=too_long)[0] == 413  # refused before reading
    assert request(server, "POST", "/api/nothing", {})[0] == 404
    assert server.engine.calls == []
    assert not server.lock.locked()


def test_concurrent_run_is_refused(server):
    server.generate.release.clear()
    first = {}
    thread = threading.Thread(target=lambda: first.update(
        result=request(server, "POST", "/api/run", {"scenario": "small", "state": "one"})))
    thread.start()
    assert server.generate.started.wait(10)

    status, _, body = request(server, "POST", "/api/run", {"scenario": "small", "state": "two"})
    assert status == 409 and json.loads(body)["error"]["type"] == "busy"
    assert request(server, "POST", "/api/model", {"model": "stub/b"})[0] == 409

    server.generate.release.set()
    thread.join(10)
    assert [name for name, _ in events(first["result"][2])][-1] == "summary"
    assert [context for context, _ in server.engine.calls] == ["one"]
    assert request(server, "POST", "/api/run", {"scenario": "small", "state": "three"})[0] == 200


def test_failed_run_sends_an_error_event_and_frees_the_lock():
    server = start(lambda model: StubEngine(model, fail=True), ["stub/a"], StubGenerate())
    try:
        status, _, raw = request(server, "POST", "/api/run", {"scenario": "small", "state": "x"})
        assert status == 200
        assert events(raw) == [("error", {"message": "RuntimeError: out of memory"})]
        assert not server.lock.locked()
    finally:
        server.shutdown()
        server.server_close()


class Streaming:
    """Stands in for baseline.generate: streams `chunks` small pieces, counting those that got out."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.sent = 0

    def __call__(self, engine, context, schema, on_text=None):
        self.sent = 0
        for _ in range(self.chunks):
            time.sleep(0.02)
            on_text('"x", ')
            self.sent += 1
        return {"text": "{}", **baseline.check("{}", schema), "latency_ms": 50.0, "output_tokens": 1}


def test_page_closed_mid_run_frees_the_lock():
    generate = Streaming(chunks=250)
    server = start(StubEngine, ["stub/a"], generate)
    try:
        body = json.dumps({"scenario": "small", "state": "one"}).encode()
        with socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=30) as sock:
            sock.sendall(b"POST /api/run HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                         + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
            got = b""
            while b"event: text" not in got:
                got += sock.recv(65536)
        deadline = time.monotonic() + 10
        while server.lock.locked() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert generate.sent < generate.chunks  # the server noticed the page was gone and stopped
        assert not server.lock.locked()
        generate.chunks = 1
        status, _, raw = request(server, "POST", "/api/run", {"scenario": "small", "state": "two"})
        assert status == 200 and events(raw)[-1][0] == "summary"
    finally:
        server.shutdown()
        server.server_close()


def test_non_finite_numbers_never_reach_the_page():
    def generate(engine, context, schema, on_text=None):
        return {"text": "{}", **baseline.check("{}", schema), "latency_ms": float("inf"), "output_tokens": 0}

    def refuse(name):
        raise AssertionError(f"{name} reached the page")

    server = start(StubEngine, ["stub/a"], generate)
    try:
        status, _, raw = request(server, "POST", "/api/run", {"scenario": "small", "state": "x"})
    finally:
        server.shutdown()
        server.server_close()
    frames = [f.split("\n") for f in raw.decode("utf-8").split("\n\n") if f]
    names = [lines[0].removeprefix("event: ") for lines in frames]
    assert names == ["decide", "error"]
    for lines in frames:
        json.loads(lines[1].removeprefix("data: "), parse_constant=refuse)  # what the page's JSON.parse accepts


@pytest.mark.model
def test_real_race(engine):
    server = start(lambda model: engine, [engine.model_id], baseline.generate)
    try:
        status, _, raw = request(server, "POST", "/api/run", {"scenario": "small", "state": SMALL["state"]})
    finally:
        server.shutdown()
        server.server_close()
    assert status == 200
    got = events(raw)
    names = [name for name, _ in got]
    assert names[0] == "decide" and names[-2:] == ["baseline", "summary"]
    assert set(names[1:-2]) == {"text"}
    decided, generated, summary = got[0][1], got[-2][1], got[-1][1]
    assert decided["model"] == engine.model_id and set(decided["answers"]) == set(SMALL["questions"])
    assert decided["display"]["section"]["value"] == "Sports"
    assert "".join(data["text"] for name, data in got if name == "text") == generated["text"]
    assert set(generated["answers"]) == set(SMALL["questions"])
    assert 0 < generated["output_tokens"] <= baseline.max_new_tokens(engine.tokenizer, SMALL["questions"])
    assert summary["speedup"] == pytest.approx(generated["latency_ms"] / decided["latency_ms"])
