"use strict";

// State text and model output are untrusted: they only ever reach the page as text nodes.

const $ = (id) => document.getElementById(id);
const ui = {
  scenario: $("scenario"), model: $("model"), run: $("run"), runLabel: $("run-label"), state: $("state"),
  noticeRow: $("notice-row"), notice: $("notice"),
  oursTitle: $("ours-title"), oursMs: $("ours-ms"), oursOut: $("ours-out"),
  genTitle: $("gen-title"), genMs: $("gen-ms"), genOut: $("gen-out"), genFlags: $("gen-flags"),
};

let scenarios = [];
let currentModel = null;
let busy = false;

const scenario = () => scenarios.find((s) => s.id === ui.scenario.value);
const ms = (value) => `${value.toFixed(1)} ms`;

function span(text, cls, title) {
  const node = document.createElement("span");
  if (cls) node.className = cls;
  if (title) node.title = title;
  node.textContent = text;
  return node;
}

// pieces: [text] for plain text, [text, class, title] for a styled span.
function row(pieces, cls) {
  const div = document.createElement("div");
  div.className = cls ? `row ${cls}` : "row";
  for (const [text, kind, title] of pieces) div.append(kind ? span(text, kind, title) : text);
  return div;
}

function hint(box, text) {
  const div = document.createElement("div");
  div.className = "hint";
  div.textContent = text;
  box.replaceChildren(div);
}

function showNotice(kind, pieces) {
  ui.notice.className = `notice ${kind}`;
  ui.notice.replaceChildren(...pieces.map(([text, cls]) => span(text, cls)));
  ui.noticeRow.hidden = false;
}

function hideNotice() {
  ui.noticeRow.hidden = true;
}

// Short model name for the card titles ("Qwen/Qwen2.5-1.5B-Instruct" -> "Qwen2.5-1.5B"); hover shows the id.
function setTitles() {
  const name = currentModel ? currentModel.split("/").pop().replace("-Instruct", "") : "no model";
  ui.oursTitle.textContent = `MirethSTM1 (${name})`;
  ui.genTitle.textContent = `Normal generation (${name})`;
  ui.oursTitle.title = ui.genTitle.title = currentModel || "";
}

function setBusy(on, label) {
  busy = on;
  ui.run.disabled = on || !currentModel;
  ui.run.classList.toggle("busy", on);
  ui.scenario.disabled = on;
  ui.model.disabled = on;
  ui.runLabel.textContent = on ? label : "Run comparison";
}

function resetCards() {
  hideNotice();
  ui.oursMs.classList.remove("failed");
  ui.genMs.classList.remove("failed");
  ui.oursMs.textContent = ui.genMs.textContent = ms(0);
  ui.genFlags.replaceChildren();
  hint(ui.oursOut, "Every field scored at once. Press \"Run comparison\".");
  hint(ui.genOut, "The same model writes the JSON token by token.");
}

// Counts up on `node` until the returned function is called.
function ticker(node) {
  const start = performance.now();
  let on = true;
  const tick = () => {
    if (!on) return;
    node.textContent = ms(performance.now() - start);
    requestAnimationFrame(tick);
  };
  tick();
  return () => { on = false; };
}

// Left card: one line per field, all at once.
function renderOurs(sc, lines) {
  const ids = Object.keys(sc.questions);
  const rows = [row([["{"]])];
  ids.forEach((id, i) => {
    const d = lines[id];
    const pieces = [["  "], [JSON.stringify(id), "k"], [': { "value": '], [JSON.stringify(d.value), "v"],
      [`, "prob": ${d.prob.toFixed(4)}`]];
    if ("score" in d) pieces.push([`, "score": ${d.score.toFixed(2)}`]);
    pieces.push([i < ids.length - 1 ? " }," : " }"]);
    rows.push(row(pieces));
  });
  rows.push(row([["}"]]));
  ui.oursOut.replaceChildren(...rows);
}

// Lays generated JSON out one top-level key per line, whatever spacing the model chose.
// Characters are never changed, only whitespace between tokens; text outside the outer
// braces is shown as written.
function reflow(text) {
  let out = "";
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (const ch of text) {
    if (inString) {
      out += ch;
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
    } else if (depth === 0) {
      out += ch === "{" ? "{\n  " : ch;
      if (ch === "{") depth = 1;
    } else if (ch === '"') {
      inString = true;
      out += ch;
    } else if (ch === "{" || ch === "[") {
      depth += 1;
      out += ch;
    } else if (ch === "}" || ch === "]") {
      depth -= 1;
      out += depth === 0 ? `\n${ch}` : ch;
    } else if (ch === ",") {
      out += depth === 1 ? ",\n  " : ", ";
    } else if (ch === ":") {
      out += ": ";
    } else if (!/\s/.test(ch)) {
      out += ch;
    }
  }
  return out;
}

const KEY_LINE = /^(\s*)("(?:[^"\\]|\\.)*"?)(.*)$/;
const VALUE = /^(\s*:\s*)(.*?)(,?)$/;

// Right card: the model's own text. The model only knows the prompt's keys q1..qn, so a
// finished key is shown as the field it stands for, with the model's key in the gutter.
// Values that are not an allowed answer turn red once generation ends.
function renderGenerated(sc, text, bad) {
  const ids = Object.keys(sc.questions);
  const rows = reflow(text).split("\n").map((line) => {
    const m = KEY_LINE.exec(line);
    const q = m && /^"q(\d+)"$/.exec(m[2]);
    const id = q ? ids[Number(q[1]) - 1] : undefined;
    if (!id) return row([["", "gutter"], [line]]);
    const pieces = [[`q${q[1]}`, "gutter"], [m[1]], [JSON.stringify(id), "k", `${m[2]} in the model's output`]];
    const v = VALUE.exec(m[3]);
    if (v && v[2]) pieces.push([v[1]], [v[2], "v"], [v[3]]);
    else pieces.push([m[3]]);
    return row(pieces, bad && bad.has(id) ? "bad" : "");
  });
  ui.genOut.replaceChildren(...rows);
}

// A card whose part of the run did not finish: no time on its badge, no stale hint.
function failed(badge, out) {
  badge.textContent = "failed";
  badge.classList.add("failed");
  if (!out.querySelector(".row")) hint(out, "No result: the run failed.");
}

function showFlags(result) {
  const flags = [];
  const n = result.hallucinated.length;
  if (n) flags.push(`${n} field${n === 1 ? "" : "s"} hallucinated`);
  if (!result.valid_json) flags.push("invalid JSON");
  else if (result.missing.length) flags.push(`${result.missing.length} missing`);
  ui.genFlags.replaceChildren(...flags.map((text) => span(text, "flag")));
}

function showSpeed(summary) {
  const faster = summary.speedup >= 1;  // say "slower" plainly when generation wins
  const ratio = faster ? summary.speedup : 1 / summary.speedup;
  showNotice(faster ? "speed" : "speed slow", [
    [`${ratio.toFixed(1)}x ${faster ? "FASTER" : "SLOWER"}`, "big"],
    ["\xb7", "dot"],
    [`${Math.round(summary.decide_ms)} ms vs ${Math.round(summary.baseline_ms)} ms`, "times"],
  ]);
}

// Server-sent events from a fetch response, as [name, data] pairs.
async function* events(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      let name = "message";
      let data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event: ")) name = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      yield [name, JSON.parse(data)];
    }
  }
}

async function problem(response) {
  try {
    return (await response.json()).error.message;
  } catch {
    return `HTTP ${response.status}`;
  }
}

function post(path, body) {
  return fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

async function run() {
  const sc = scenario();
  if (!sc || busy) return;
  setBusy(true, "Running...");
  resetCards();
  hint(ui.oursOut, "Scoring every field...");
  hint(ui.genOut, "Starts when MirethSTM1 is done.");
  const stopOurs = ticker(ui.oursMs);
  let stopGen = () => {};
  let text = "";
  let stage = "ours";  // the part still running: "ours", then "gen", then "done"
  try {
    const response = await post("/api/run", { scenario: sc.id, state: ui.state.value });
    if (!response.ok) throw new Error(await problem(response));
    for await (const [name, data] of events(response)) {
      if (name === "decide") {
        stopOurs();
        ui.oursMs.textContent = ms(data.latency_ms);
        renderOurs(sc, data.display);
        ui.genOut.replaceChildren();
        stopGen = ticker(ui.genMs);
        stage = "gen";
      } else if (name === "text") {
        text += data.text;
        renderGenerated(sc, text);
        ui.genOut.scrollTop = ui.genOut.scrollHeight;
      } else if (name === "baseline") {
        stopGen();
        ui.genMs.textContent = ms(data.latency_ms);
        renderGenerated(sc, data.text, new Set(data.hallucinated));
        showFlags(data);
        ui.oursOut.scrollTop = ui.genOut.scrollTop = 0;
        stage = "done";
      } else if (name === "summary") {
        showSpeed(data);
      } else if (name === "error") {
        throw new Error(data.message);
      }
    }
  } catch (err) {
    stopOurs();
    stopGen();
    showNotice("error", [[`Run failed: ${err.message}`]]);
    if (stage === "ours") {
      failed(ui.oursMs, ui.oursOut);
      hint(ui.genOut, "Did not run.");
    } else if (stage === "gen") {
      failed(ui.genMs, ui.genOut);
    }
  } finally {
    stopOurs();
    stopGen();
    setBusy(false);
  }
}

async function switchModel() {
  const wanted = ui.model.value;
  setBusy(true, "Loading...");
  resetCards();
  showNotice("", [[`Loading ${wanted}...`]]);
  try {
    const response = await post("/api/model", { model: wanted });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      if (body.error && "current" in body.error) currentModel = body.error.current;
      throw new Error((body.error && body.error.message) || `HTTP ${response.status}`);
    }
    currentModel = body.current;
    hideNotice();
  } catch (err) {
    showNotice("error", [[err.message]]);
  } finally {
    if (currentModel) ui.model.value = currentModel;
    setTitles();
    setBusy(false);
  }
}

function fill(select, entries) {
  select.replaceChildren(...entries.map(([value, label]) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    return option;
  }));
}

// Side by side, keep both cards at the same scroll position so line k faces line k.
function syncScroll(a, b) {
  const sideBySide = window.matchMedia("(min-width: 761px)");
  let echo = false;
  const follow = (from, to) => () => {
    if (echo || !sideBySide.matches) return;
    echo = true;
    to.scrollTop = from.scrollTop;
    requestAnimationFrame(() => { echo = false; });
  };
  a.addEventListener("scroll", follow(a, b));
  b.addEventListener("scroll", follow(b, a));
}

async function getJSON(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(await problem(response));
  return response.json();
}

async function init() {
  syncScroll(ui.oursOut, ui.genOut);
  ui.run.addEventListener("click", run);
  ui.model.addEventListener("change", switchModel);
  ui.scenario.addEventListener("change", () => {
    ui.state.value = scenario().state;
    resetCards();
  });
  resetCards();
  try {
    const [list, models] = await Promise.all([getJSON("/api/scenarios"), getJSON("/api/models")]);
    scenarios = list;
    fill(ui.scenario, scenarios.map((s) => [s.id, s.title]));
    fill(ui.model, models.models.map((m) => [m, m]));
    currentModel = models.current;
    if (currentModel) ui.model.value = currentModel;
    ui.state.value = scenarios.length ? scenarios[0].state : "";
  } catch (err) {
    showNotice("error", [[`Could not reach the console server: ${err.message}`]]);
  }
  setTitles();
  setBusy(false);
}

init();
