"""mirethstm decide and mirethstm console (SPEC 5)."""

import io
import json

import pytest

from mirethstm import cli
from mirethstm.calibration import DEFAULT_TEMPERATURES

QUESTIONS = {"sport": {"type": "noul", "instructions": "Is this about sport?"}}
RESULT = {"model": "stub", "answers": {"sport": {"type": "noul", "noul": 0.9}},
          "usage": {"input_tokens": 10, "output_tokens": 0}, "id": "0" * 32, "latency_ms": 1.0}


class StubEngine:
    loads = []
    decisions = []

    @classmethod
    def load(cls, model, **settings):
        cls.loads.append((model, settings))
        return cls()

    def decide(self, context, schema):
        self.decisions.append((context, schema))
        return RESULT


@pytest.fixture
def stub(monkeypatch):
    StubEngine.loads, StubEngine.decisions = [], []
    monkeypatch.setattr(cli, "Engine", StubEngine)
    return StubEngine


def feed_stdin(monkeypatch, text):
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(text.encode("utf-8")), encoding="utf-8"))


def write_schema(tmp_path, questions=QUESTIONS):
    path = tmp_path / "s.json"
    path.write_text(json.dumps(questions), encoding="utf-8")
    return str(path)


def test_defaults(stub, monkeypatch, tmp_path, capsys):
    feed_stdin(monkeypatch, "The striker scored twice. Café crème.\n")
    assert cli.main(["decide", "--schema", write_schema(tmp_path)]) == 0
    assert stub.loads == [("Qwen/Qwen3-4B-Instruct-2507",
                           {"device": None, "batch_tokens": 4096, "temperature": None, "event_log": None,
                            "tarnlight": True})]
    assert stub.decisions == [("The striker scored twice. Café crème.\n", QUESTIONS)]
    assert json.loads(capsys.readouterr().out) == RESULT


def test_all_options_and_state_json(stub, monkeypatch, tmp_path, capsys):
    feed_stdin(monkeypatch, '{"ticket": {"body": "refund please"}, "tags": ["a", "b"]}')
    log = str(tmp_path / "events.jsonl")
    argv = ["decide", "--schema", write_schema(tmp_path), "--model", "some/model", "--temperature", "1.5",
            "--device", "cpu", "--batch-tokens", "3", "--log", log, "--state-json", "--no-tarnlight"]
    assert cli.main(argv) == 0
    assert stub.loads == [("some/model", {"device": "cpu", "batch_tokens": 3, "temperature": 1.5,
                                          "event_log": log, "tarnlight": False})]
    assert stub.decisions == [({"ticket": {"body": "refund please"}, "tags": ["a", "b"]}, QUESTIONS)]
    assert json.loads(capsys.readouterr().out) == RESULT


@pytest.mark.parametrize(
    "questions, stdin, extra, message",
    [
        ({}, "text", [], "non-empty"),
        ({"q": {"type": "bool", "instructions": "x"}}, "text", [], "unknown type"),
        ({"q": {"type": "score", "instructions": "x", "criteria": []}}, "text", [], "1 to 10 levels"),
        (QUESTIONS, "42", ["--state-json"], "state must be"),
        (QUESTIONS, "{not json", ["--state-json"], "Expecting"),
    ],
)
def test_bad_input_exits_2_without_loading(stub, monkeypatch, tmp_path, capsys, questions, stdin, extra, message):
    feed_stdin(monkeypatch, stdin)
    assert cli.main(["decide", "--schema", write_schema(tmp_path, questions), *extra]) == 2
    out, err = capsys.readouterr()
    assert out == "" and message in err
    assert stub.loads == []


def test_unreadable_input_exits_2_without_loading(stub, monkeypatch, tmp_path, capsys):
    feed_stdin(monkeypatch, "text")
    assert cli.main(["decide", "--schema", str(tmp_path / "missing.json")]) == 2
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(b"\xff not utf-8")))
    assert cli.main(["decide", "--schema", write_schema(tmp_path)]) == 2
    out, err = capsys.readouterr()
    assert out == "" and err.count("mirethstm: ") == 2
    assert stub.loads == []


@pytest.mark.parametrize("argv", [
    [], ["decide"], ["serve"],
    ["decide", "--schema", "s.json", "--batch-tokens", "x"],
    ["decide", "--schema", "s.json", "--batch-tokens", "0"],
    ["decide", "--schema", "s.json", "--chunk-size", "8"],
    ["console", "--port", "x"], ["console", "--schema", "s.json"],
    ["decide", "--schema", "s.json", "--temperature", "0"],
])
def test_argument_errors(stub, served, argv):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    assert exit_info.value.code == 2
    assert stub.loads == [] and served == []


@pytest.fixture
def served(monkeypatch):
    calls = []
    monkeypatch.setattr(cli.console, "serve", lambda **settings: calls.append(settings))
    return calls


def test_console_defaults(stub, served):
    assert cli.main(["console"]) == 0
    assert served == [{"model": "Qwen/Qwen2.5-1.5B-Instruct", "device": None, "host": "127.0.0.1",
                       "port": 8766, "tarnlight": True}]
    assert stub.loads == []


def test_console_options(served):
    argv = ["console", "--model", "some/model", "--device", "cpu", "--host", "0.0.0.0", "--port", "9000",
            "--no-tarnlight"]
    assert cli.main(argv) == 0
    assert served == [{"model": "some/model", "device": "cpu", "host": "0.0.0.0", "port": 9000, "tarnlight": False}]


@pytest.mark.model
def test_end_to_end(engine, monkeypatch, tmp_path, capsys):
    # `engine` makes this skip when the model is not cached and names the test model and
    # device; the CLI loads its own copy.
    questions = {
        "sport": {"type": "noul", "instructions": "Is this text about sport?"},
        "topic": {"type": "choice", "instructions": "Which section fits this text?",
                  "criteria": {"Sports": None, "Business": None, "Sci-Tech": None}},
    }
    feed_stdin(monkeypatch, "The home team won the football match 3-1 after their striker scored twice.")
    log = tmp_path / "events.jsonl"
    argv = ["decide", "--schema", write_schema(tmp_path, questions), "--model", engine.model_id,
            "--device", str(engine.model.device), "--batch-tokens", "16", "--log", str(log), "--no-tarnlight"]
    assert cli.main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["model"] == engine.model_id
    assert result["answers"]["sport"]["noul"] > 0.5
    assert result["answers"]["topic"]["choice"] == "Sports"
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [line["field"] for line in lines] == ["sport", "topic"]
    shipped = DEFAULT_TEMPERATURES.get(engine.model_id, 1.0)  # no --temperature: the model's shipped T
    assert all(line["id"] == result["id"] and line["T"] == shipped for line in lines)
