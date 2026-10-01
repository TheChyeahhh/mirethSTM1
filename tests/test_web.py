"""The race view's page script (SPEC 10.1). The Node tests run app.js against a stand-in DOM
(web_harness.cjs) and are skipped when Node.js is not installed."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from mirethstm.console import WEB
from mirethstm.models import approved, entry

NODE = shutil.which("node")
HARNESS = Path(__file__).with_name("web_harness.cjs")
needs_node = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")

SC = {
    "id": "s",
    "title": "S (3 fields)",
    "state": "I was billed twice.",
    "questions": {
        "refund": {"type": "noul", "instructions": "Refund?"},
        "team": {"type": "choice", "instructions": "Team?", "criteria": {"billing": None, "bug": None}},
        "urgency": {"type": "score", "instructions": "Urgency?", "criteria": ["Low", "High"]},
    },
}


def code(path):
    """Source without comments, so a comment can neither satisfy nor fail a check."""
    text = re.sub(r"/\*.*?\*/|<!--.*?-->", "", path.read_text(encoding="utf-8"), flags=re.S)
    return re.sub(r"(?m)(^|\s)//.*$", "", text)


def test_page_never_parses_text_as_html():
    for name in ("app.js", "index.html"):
        source = code(WEB / name)
        for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "DOMParser",
                     "createContextualFragment", "eval(", "Function("):
            assert sink not in source, f"{name} uses {sink}"


def page(snippet):
    """Load app.js in Node, run `snippet` as an async function body, and return its JSON result."""
    done = subprocess.run([NODE, str(HARNESS), str(WEB / "app.js")], input=snippet, capture_output=True,
                          text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def keyed(rows):
    """(gutter, key, row class) per row."""
    out = []
    for r in rows:
        gutter = [t for t, c, _ in r["pieces"] if c == "gutter"]
        key = [t for t, c, _ in r["pieces"] if c == "k"]
        out.append((gutter[0] if gutter else None, key[0] if key else None, r["cls"]))
    return out


@needs_node
def test_generated_keys_show_their_fields():
    text = '{"q1": true, "q2": "billing", "q3": 9, "q4": 1, "q2'
    rows = page(f"renderGenerated({json.dumps(SC)}, {json.dumps(text)}, new Set(['urgency']));"
                "return dump('gen-out');")
    assert keyed(rows) == [
        ("", None, "row"),
        ("q1", '"refund"', "row"),
        ("q2", '"team"', "row"),
        ("q3", '"urgency"', "row bad"),
        ("", None, "row"),  # q4 is not a question: shown as the model wrote it
        ("", None, "row"),  # a key still being written
    ]
    team = rows[2]["pieces"]
    assert ["billing" in t for t, c, _ in team if c == "v"] == [True]
    assert [title for _, c, title in team if c == "k"] == ['"q2" in the model\'s output']
    assert "".join(t for t, _, _ in rows[4]["pieces"]) == '  "q4": 1,'
    assert "".join(t for t, _, _ in rows[5]["pieces"]) == '  "q2'


@needs_node
def test_text_stays_text():
    text = '{"q1": "<b>bold?</b>"}'
    rows = page(f"renderGenerated({json.dumps(SC)}, {json.dumps(text)}); return dump('gen-out');")
    assert ["<b>bold?</b>" in t for t, c, _ in rows[1]["pieces"] if c == "v"] == [True]


@needs_node
def test_picker_shows_each_model_with_its_note():
    listed = {"models": [entry(m.id) for m in approved()] + [entry("some/other-model")], "current": "Qwen/Qwen3-0.6B"}
    got = page(f"""
        const answers = {{ "/api/scenarios": [{json.dumps(SC)}], "/api/models": {json.dumps(listed)} }};
        globalThis.fetch = async (path) => ({{ ok: true, json: async () => answers[path] }});
        await init();
        return {{ options: document.getElementById("model").children.map((o) => [o.value, o.textContent, o.title]),
                  value: ui.model.value, title: text("ours-title") }};
    """)
    assert got["options"][0] == ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5 1.5B Instruct (Match first, not approved yet)",
                                 "Qwen/Qwen2.5-1.5B-Instruct"]
    assert [o[0] for o in got["options"]] == [m.id for m in approved()] + ["some/other-model"]
    assert got["options"][-1][1] == "some/other-model (Unvetted: not on the approved list)"
    assert got["value"] == "Qwen/Qwen3-0.6B" and got["title"] == "MirethSTM1 (Qwen3-0.6B)"


def run_snippet(fetch_body):
    """Start a run with `fetch` answering as `fetch_body` says, wait for it, and read the cards."""
    return page(f"""
        scenarios = [{json.dumps(SC)}];
        ui.scenario.value = "s";
        currentModel = "org/Model-1B-Instruct";
        globalThis.fetch = async () => {{ {fetch_body} }};
        await run();
        return {{ oursMs: text("ours-ms"), oursCls: cls("ours-ms"), oursOut: text("ours-out"),
                  genMs: text("gen-ms"), genCls: cls("gen-ms"), genOut: text("gen-out"),
                  notice: text("notice"), label: text("run-label") }};
    """)


@needs_node
def test_run_refused_marks_both_cards():
    got = run_snippet('return { ok: false, status: 409, json: async () => ({ error: { message: "busy" } }) };')
    assert got["notice"] == "Run failed: busy"
    assert (got["oursMs"], "failed" in got["oursCls"]) == ("failed", True)
    assert got["oursOut"] == "No result: the run failed."
    assert got["genOut"] == "Did not run." and "failed" not in got["genCls"]
    assert got["label"] == "Run comparison"


@needs_node
def test_generation_failure_marks_only_its_card():
    decided = {"latency_ms": 12.0, "display": {"refund": {"value": True, "prob": 0.9},
                                               "team": {"value": "billing", "prob": 0.8},
                                               "urgency": {"value": 1, "prob": 0.7, "score": 0.7}}}
    frames = f"event: decide\ndata: {json.dumps(decided)}\n\nevent: error\ndata: {json.dumps({'message': 'boom'})}\n\n"
    got = run_snippet(f"""
        const bytes = new TextEncoder().encode({json.dumps(frames)});
        let sent = false;
        const read = async () => (sent ? {{ done: true }} : (sent = true, {{ value: bytes, done: false }}));
        return {{ ok: true, body: {{ getReader: () => ({{ read }}) }} }};
    """)
    assert got["notice"] == "Run failed: boom"
    assert got["oursMs"] == "12.0 ms" and "failed" not in got["oursCls"]
    assert '"team": { "value": "billing", "prob": 0.8000 }' in got["oursOut"]
    assert (got["genMs"], "failed" in got["genCls"]) == ("failed", True)
    assert got["genOut"] == "No result: the run failed."
