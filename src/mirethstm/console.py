"""Race view and local API: MirethSTM1 next to normal generation on the same model (SPEC 6, 10.1).

Standard library server plus one page (web/). Everything is local; nothing is fetched
from the internet except model weights on first load. The API needs no key and never
contacts TypeSafe or any other service.
"""

import gc
import ipaddress
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from . import baseline
from .engine import Engine, systemone_body
from .models import approved, entry
from .scenarios import SCENARIOS
from .schema import SchemaError, validate

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # the original demo's model: match it first
WEB = Path(__file__).with_name("web")
CONTENT_TYPES = {".html": "text/html", ".css": "text/css", ".js": "text/javascript"}
MAX_BODY = 1 << 20
API_WAIT = 60.0  # seconds an API request waits for the current run before answering 529
API_ROUTES = ("/v1/systemone", "/v1/decide")


def display(result):
    """The left card's line per field: the top answer and its probability (score adds the expected level)."""
    lines = {}
    for qid, a in result["answers"].items():
        if a["type"] == "noul":
            value = a["noul"] >= 0.5
            lines[qid] = {"value": value, "prob": a["noul"] if value else 1 - a["noul"]}
        elif a["type"] == "choice":
            lines[qid] = {"value": a["choice"], "prob": a["probabilities"][a["choice"]]}
        else:
            top = max(a["probabilities"], key=a["probabilities"].get)
            lines[qid] = {"value": int(top), "prob": a["probabilities"][top], "score": a["score"]}
    return lines


class ConsoleServer(ThreadingHTTPServer):
    """Holds the loaded engine. `lock` keeps runs, API calls and model switches from overlapping.

    `models` are the ids the page's picker offers.
    """

    daemon_threads = True

    def __init__(self, address, load, models, scenarios=SCENARIOS, generate=baseline.generate, api_wait=API_WAIT):
        super().__init__(address, Handler)
        self.load = load
        self.models = list(models)
        self.scenarios = {s["id"]: s for s in scenarios}
        self.generate = generate
        self.api_wait = api_wait
        self.engine = None
        self.lock = threading.Lock()
        self.pages = {p.name for p in WEB.iterdir() if p.is_file()}

    def switch(self, model):
        """Free the current model, then load `model`. Call with the lock held."""
        previous = self.engine.model_id if self.engine else None
        self.engine = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            self.engine = self.load(model)
        except Exception:
            if previous:
                self.engine = self.load(previous)  # keep the console usable
            raise


class Handler(BaseHTTPRequestHandler):
    server_version = "mirethstm"

    def do_GET(self):
        if not self._host_ok():
            return
        path = self.path.split("?", 1)[0]
        engine = self.server.engine
        current = engine.model_id if engine else None
        if path == "/api/scenarios":
            return self._json(200, list(self.server.scenarios.values()))
        if path == "/api/models":
            return self._json(200, {"models": [entry(m) for m in self.server.models], "current": current})
        if path == "/v1/models":
            return self._json(200, {"models": [entry(m.id) for m in approved()], "current": current})
        name = "index.html" if path == "/" else path[1:]
        if name not in self.server.pages:  # a fixed set of file names, so no path can escape web/
            return self._error(404, "not_found", "no such page")
        body = (WEB / name).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPES.get(Path(name).suffix, "application/octet-stream") + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if not self._host_ok():
            return
        path = self.path.split("?", 1)[0]
        body = self._read_json(invalid=422 if path in API_ROUTES else 400)
        if body is None:
            return
        if path == "/api/model":
            return self._switch(body)
        if path == "/api/run":
            return self._run(body)
        if path in API_ROUTES:
            return self._api(body, rounded=path == "/v1/systemone")
        self._error(404, "not_found", "no such endpoint")

    def _api(self, body, rounded):
        """POST /v1/systemone (rounded, TypeSafe's shape) and /v1/decide (full precision, id and latency).

        The request's `model` and any Authorization header are ignored: the loaded model answers.
        """
        state, questions = body.get("state"), body.get("questions")
        try:
            validate(state, questions)
        except SchemaError as e:
            return self._error(422, "invalid_request", str(e))
        if not self.server.lock.acquire(timeout=self.server.api_wait):
            return self._error(529, "busy", f"still busy after {self.server.api_wait:g} s with a run or a model load")
        try:
            engine = self.server.engine
            result = engine.decide(state, questions) if engine else None
        except Exception as e:
            result = e
        finally:
            self.server.lock.release()
        if result is None:
            return self._error(529, "no_model", "no model is loaded")
        if isinstance(result, Exception):
            return self._error(500, "internal_error", f"{type(result).__name__}: {result}")
        # The TypeSafe SDK reads the id from x-typesafe-request-id (its result.request_id).
        headers = {"x-request-id": result["id"], "x-typesafe-request-id": result["id"],
                   "server-timing": f"decide;dur={result['latency_ms']:.1f}"}
        try:
            self._json(200, systemone_body(result) if rounded else result, headers)
        except ValueError:  # NaN or infinity: nothing was sent yet
            self._error(500, "internal_error", "the result holds a number JSON cannot carry")

    def _switch(self, body):
        model = body.get("model")
        if model not in self.server.models:
            return self._error(400, "bad_request", f"unknown model {model!r}")
        if not self.server.lock.acquire(blocking=False):
            return self._error(409, "busy", "a run or a model load is in progress")
        try:
            self.server.switch(model)
        except Exception as e:
            engine = self.server.engine
            return self._error(500, "load_failed", f"could not load {model}: {e}",
                               current=engine.model_id if engine else None)
        finally:
            self.server.lock.release()
        self._json(200, {"current": model})

    def _run(self, body):
        scenario = self.server.scenarios.get(body.get("scenario"))
        if scenario is None:
            return self._error(400, "bad_request", "unknown scenario")
        state, questions = body.get("state", scenario["state"]), scenario["questions"]
        try:
            validate(state, questions)
        except SchemaError as e:
            return self._error(422, "invalid_request", str(e))
        if not self.server.lock.acquire(blocking=False):
            return self._error(409, "busy", "a run or a model load is in progress")
        try:
            engine = self.server.engine
            if engine is None:
                return self._error(503, "no_model", "no model is loaded")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self._race(engine, state, questions)
        except OSError:
            pass  # the page went away mid-run
        finally:
            self.server.lock.release()

    def _race(self, engine, state, questions):
        """Events, in order: decide, text (repeated), baseline, summary; or error."""
        try:
            decided = engine.decide(state, questions)
            self._event("decide", {**decided, "display": display(decided)})
            generated = self.server.generate(engine, state, questions,
                                             on_text=lambda chunk: self._event("text", {"text": chunk}))
            self._event("baseline", generated)
            self._event("summary", {"decide_ms": decided["latency_ms"], "baseline_ms": generated["latency_ms"],
                                    "speedup": generated["latency_ms"] / decided["latency_ms"]})
        except OSError:
            raise
        except Exception as e:
            self._event("error", {"message": f"{type(e).__name__}: {e}"})

    def _event(self, name, data):
        # allow_nan=False: the page's JSON.parse would reject NaN or Infinity and drop the whole run.
        self.wfile.write(f"event: {name}\ndata: {json.dumps(data, allow_nan=False)}\n\n".encode("utf-8"))

    def _host_ok(self):
        """While bound to loopback, answer only loopback Host names (blocks DNS rebinding)."""
        if not ipaddress.ip_address(self.server.server_address[0]).is_loopback:
            return True
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        if host in ("127.0.0.1", "localhost"):
            return True
        self._error(403, "forbidden", "unexpected Host header")
        return False

    def _read_json(self, invalid=400):
        """The request body as a JSON object, or None after sending an error (`invalid` when it is not one)."""
        length = self.headers.get("Content-Length", "0")
        if not length.isdigit():
            return self._error(400, "bad_request", "invalid Content-Length")
        if int(length) > MAX_BODY:
            return self._error(413, "too_large", "request body is too large")
        raw = self.rfile.read(int(length))
        if self.headers.get_content_type() != "application/json":  # cross-site forms cannot send this
            return self._error(415, "bad_request", "send application/json")
        try:
            body = json.loads(raw.decode("utf-8"))
        except ValueError:
            body = None
        if not isinstance(body, dict):
            return self._error(invalid, "bad_request" if invalid == 400 else "invalid_request",
                               "the body must be a JSON object")
        return body

    def _json(self, status, payload, headers=None):
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, kind, message, **extra):
        self._json(status, {"error": {"type": kind, "message": message, **extra}})


def picker(model):
    """The ids the page's picker offers: the approved models, plus `model` when it is another id."""
    ids = [m.id for m in approved()]
    return ids if model in ids else ids + [model]


def serve(model=DEFAULT_MODEL, device=None, host="127.0.0.1", port=8766, tarnlight=True):
    """Bind, load `model`, then serve until Ctrl+C."""
    server = ConsoleServer((host, port), lambda m: Engine.load(m, device=device, tarnlight=tarnlight), picker(model))
    try:
        print(f"Loading {model}...", flush=True)
        server.engine = server.load(model)
        url = f"http://{host}:{server.server_address[1]}"
        print(f"MirethSTM1 console: {url}/  (Ctrl+C stops it)", flush=True)
        print(f"TypeSafe-compatible API, no key needed: POST {url}/v1/systemone", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
