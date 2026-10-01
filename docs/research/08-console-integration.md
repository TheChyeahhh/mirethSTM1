# 08. Console integration (Tarnlight vs the frozen JSONL event schema)

Date checked: 2026-09-30. Scope: can the live decision console (the brief's "jevscope", now public as Tarnlight) show MirethSTM1 decisions, and in what shape should MirethSTM1 log them.

Status legend: VERIFIED (read in code or docs by the researcher), UNVERIFIED, CONTRADICTED (brief differs from reality).

## 1. What Tarnlight is

| Fact | Status | Source |
|---|---|---|
| Public repo, Apache-2.0, 0 stars, last push 2026-09-27, version 0.1.0, read at commit 8408a73 | VERIFIED | `gh api repos/TheChyeahhh/tarnlight`; https://github.com/TheChyeahhh/tarnlight (pyproject.toml) |
| Desktop app (PySide6 + pyqtgraph + SQLite + zstd), Windows-only tested | VERIFIED | DESIGN.md "Known limits"; https://github.com/TheChyeahhh/tarnlight/blob/main/DESIGN.md |
| It records and grades decisions from TypeSafe's `/v1/systemone` API: confidence chart, gauge, counters, feed, inspector, bands, review queue, export | VERIFIED | DESIGN.md "The window" |
| It does NOT tail an arbitrary JSONL log. It imports its own envelope records from a drop box folder, UDP port 7337, or an optional HTTP proxy on 7338 | CONTRADICTED (brief: "tails a JSONL event log") | `src/tarnlight/inbox.py`, `src/tarnlight/ingest.py` |
| Calibration view is "not built yet" (reliability diagram, ECE, Brier, Platt) | VERIFIED | DESIGN.md "Not built yet" |
| Three ways in: drop box (main), proxy (one app at a time), UDP and `tarnlight replay FILE` | VERIFIED | DESIGN.md "How calls get in" |

The name "jevscope" appears nowhere in the code. The product is "Tarnlight" and describes itself as unofficial and not affiliated with TypeSafe (DESIGN.md, opening lines).

## 2. (a) The exact drop-box record format

### Location and naming (VERIFIED: `src/tarnlight/inbox.py` docstring and lines 33-40; `taps/python/tarnlight_tap.py` lines 209-216)

- Folder: `<home>/.tarnlight/inbox/`. The writer must NOT create it. A writer only writes if the folder already exists; the console (or `tarnlight install`) creates it. Removing it turns every writer off.
- File: `<source>-<YYYY-MM-DDTHH>.jsonl`, the timestamp being the current UTC hour, for example `mirethstm-2026-10-02T14.jsonl`. The reference tap sanitises `source` with `re.sub(r"[^A-Za-z0-9._-]+", "_", source)[:64]`.
- The console reads the hour from `name[-19:-6]` to decide when a file is finished. A name without that hour is never deleted.
- Only append. One JSON object per line, UTF-8, terminated by `\n`. A line is read only after its newline exists. A leading BOM is ignored; NUL bytes separate records.
- Multi-writer safety: the reference tap opens append-only (Win32 `FILE_APPEND_DATA`, POSIX `O_APPEND`) and writes the whole line in one call. The tap also skips writing when the disk has under 1 GB free (tap convention, not enforced by the console).
- Limit: a line over 16 MB (`MAX_LINE`) goes to a rejects file.

### Envelope (VERIFIED: `src/tarnlight/storage.py` line 39 `ENVELOPE`; `src/tarnlight/ingest.py` `validate()`, lines 69-78)

Only these keys are kept; any other top-level key is dropped.

| Key | Required by the door check | Type | Notes |
|---|---|---|---|
| `v` | YES, integer `1` | int | anything else is rejected ("v is not 1") |
| `ts` | YES | finite number | seconds since epoch |
| `source` | no | string or null | names the sender chip; "unknown" when missing |
| `project`, `label`, `session_id`, `tool_use_id`, `sdk`, `request_id` | no | string or null | must be text, not numbers |
| `latency_ms`, `status`, `retry_count`, `cost_est_micro` | no | finite number or null | `cost_est_micro` is filled from `response.usage.input_tokens` when it is missing (`inbox.py` `_take_one`) |
| `request`, `response`, `error` | no | any JSON | stored verbatim |
| `seq` | ignored | | assigned on arrival |

The door checks the envelope only. NaN, Infinity and numbers too large for a float are rejected. Rejects go to a rejects file with a reason, size and hash, never the content.

### The TypeSafe request/response inside `request` and `response` (VERIFIED: DESIGN.md "Wire format facts"; `storage.py` `summarize()`, lines 139-171)

```
request : {"state": <any JSON>, "model": "<string>", "questions": {"<name>": {...}}}
response: {"model": "jev-1.13.0",
           "answers": {
             "<name>": {"type":"noul",   "noul": 0.83},
             "<name>": {"type":"choice", "choice":"<opt>", "probabilities":{"<opt>":0.61}, "confidence":0.48},
             "<name>": {"type":"score",  "score":1.12, "legend":{"0":"<level>"}, "probabilities":{"0":0.2}, "confidence":0.35}},
           "usage": {"input_tokens": 399, "output_tokens": 0}}
```

What the indexer needs per answer (`summarize` wraps each answer in try/except, so a bad answer is skipped, never fatal):

- `noul`: `a["noul"]` is p(yes). Confidence is derived as `max(p, 1-p)`. No `confidence` field needed.
- `choice`: `a["choice"]`, `a["probabilities"]` (option to float) and `a["confidence"]`. If one is missing the answer is not charted.
- `score`: `a["score"]`, `a["probabilities"]` (keys "0".."N-1") and `a["confidence"]`. `legend` is optional for indexing but used for display text.
- Confidence must be 0..100 percent to be charted (`src/tarnlight/model.py` line 146).
- A failed call is `response: null`, `error: <body or {"jev": "<note>"}>`, with `status` set.

### UDP (VERIFIED: `src/tarnlight/ingest.py` lines 26-31 and 81-90; DESIGN.md "UDP and replay")

- There is no UDP "nudge" for the drop box. The console polls the folder every 0.05 s (`POLL_S`, `inbox.py` line 33). UDP is a separate, independent channel: one JSON record per datagram to `127.0.0.1:7337`, same envelope and checks, records over 60,000 bytes split into `{"id","part","of","data"}` chunks (base64 of 44,880 byte slices). It only works while the console runs, so the drop box is the right channel for a producer that must not depend on the console.

### Delivery, dedup, retention (VERIFIED: `inbox.py` docstring; DESIGN.md "The drop box")

- At least once. Read offsets are saved in `inbox/.positions.json` (keyed by file name plus file id) only after every earlier line is stored. A hard kill can store the last 0.1 s twice; the kill test with 4,800 calls saw "0 to 448 were stored twice".
- NO dedup: "Duplicates are not removed" (DESIGN.md). `request_id` is stored but never used as a key. A producer must write each decision exactly once.
- A finished file (hour ended over 300 s ago and fully stored) is deleted by the console. Half a line at the end of a finished hour is rejected.
- On start the console imports everything that arrived while it was closed and marks it with a blue line and a "Caught up: N calls" notice.

## 3. (b) Could Tarnlight show MirethSTM1 decisions with zero changes?

Answer: YES for display, with caveats. Nothing validates or displays the `model` string (VERIFIED: a search for `"model"` in `src/tarnlight` hits only `demo.py` lines 88 and 94, which write `jev-latest` and `jev-1.13.0` into synthetic demo records). It is stored verbatim and never used.

Conditions for zero changes:

1. Write TypeSafe-shaped records to `~/.tarnlight/inbox/mirethstm-<UTC hour>.jsonl` with `v:1`, `ts`, `source:"mirethstm"`, and `request`/`response` as above.
2. Map our outputs to the wire types: yes/no to `noul`, up to 255 options to `choice` with a full `probabilities` dict plus `confidence`, score to `score` with bins keyed "0".."N-1" and the expected value as `score`. This is already the brief's wire-compatibility decision, so it costs nothing extra.
3. Define `confidence` ourselves for choice and score. Tarnlight charts the API's own value, and TypeSafe's formula is not published (DESIGN.md "Wire format facts"). For the Tarnlight sink only, we should set it to the top probability after temperature scaling (note the compat HTTP route may use the TypeSafe-matching margin form from docs 03 and 04; the two must not be mixed in one record, so the sink should say which it used) and document that in the SPEC, so nobody compares it to real Jev numbers without a caveat. Tarnlight's bands (escalate below 40, review below 70) are per question name and user-set, so they work unchanged.
4. What is lost or off:
   - `T`, per-field `latency_ms`, `model` and extra fields are not shown. Extra top-level envelope keys are dropped. Extra keys inside `response.answers.<q>` are stored verbatim in the log but not shown, and the in-memory ring keeps only `type, noul, confidence, probabilities, legend, choice, score` (`ingest.py` `ANSWER_FIELDS`, `_slim`).
   - Cost: the console estimates $0.042 per million input tokens (`replay.py` line 5). For a local engine that is meaningless. Set `cost_est_micro: 0` explicitly (the console fills it only when it is null), or accept a wrong cost counter.
   - The counters compare with TypeSafe's 1,200 requests per minute limit, and "replay this one" and "copy as curl" target TypeSafe's API. Cosmetic or irrelevant for a local engine.
   - For us the chosen label always equals the argmax, so the gauge's "picked X, Y is likelier" text simply never fires.

Minimal changes that would make it first class (proposals only; we do not modify Tarnlight): non-TypeSafe-aware cost of 0, a hidden replay button for other sources, and display of `T`. None is required.

Practical constraints: Windows-only tested, PySide6 based (bundled licence texts include LGPL-3.0 and GPL-3.0 in `packaging/licenses`), run from source with `pip install .`. The console must be opened once (or `tarnlight install` run) to create the drop box folder.

## 4. (c) Frozen per-field schema vs Tarnlight's record

Brief's frozen schema: `ts, id, field, label, p, probs{}, T, latency_ms, model` (one event per field).

| Frozen field | Tarnlight equivalent | Gap |
|---|---|---|
| `ts` | envelope `ts` (epoch seconds, per call) | Match. One `ts` per call, not per field |
| `id` | envelope `request_id` (text, per call, stored, never a key) | No per-field id. Per (call, question) Tarnlight uses `seq`, assigned on arrival |
| `field` | answer key in `response.answers` (question name) | Same meaning, but many fields share one call record |
| `label` | `choice` (choice), `score` (score), derived yes/no (noul, `noul` >= 0.5) | No single `label` key; it depends on the type |
| `p` | noul: `max(p,1-p)` (raw `noul` kept); choice/score: `confidence` | For choice and score the chart uses `confidence`, NOT the probability of the chosen label. We must emit `confidence = p` |
| `probs{}` | `probabilities` (choice, score); absent for noul | Score keys are index strings plus a `legend`, not bin labels |
| `T` | none | No temperature concept; it would live only inside the verbatim `response` |
| `latency_ms` | envelope `latency_ms` (per call) | Per call, not per field |
| `model` | `request.model` and `response.model` (verbatim, never shown) | Display gap only |

Tarnlight has these and the frozen schema does not: `v` (REQUIRED), `source`, `project`, call `label`, `session_id`, `tool_use_id`, `status`, `retry_count`, `cost_est_micro`, `request.state`, `request.questions`, `usage` tokens, `error`. The frozen schema also has no call grouping, while Tarnlight's unit is a call with many answers, so flat per-field lines must be regrouped by a call id.

## 5. (d) Recommendation

Do BOTH, with one source of truth.

1. Keep the frozen per-field JSONL (`ts, id, field, label, p, probs, T, latency_ms, model`) as MirethSTM1's own documented, console-agnostic log. It is the simplest contract for notebooks, grep and other tools, it carries `T` and the full-label outcome that Tarnlight drops, and the brief already freezes it.
2. Add an OPTIONAL Tarnlight sink, about 40 lines, inside the Engine (off by default, flag or env var). It builds one envelope per `decide()` call from the same in-memory result (not by re-reading the JSONL) and appends it to `mirethstm-<UTC hour>.jsonl` in the drop box, only if the folder exists, following the tap rules (never raise, skip under 1 GB free, single append write). Tarnlight has no dedup, so it must be fed from the one place where an event is emitted.
3. Make one SPEC decision now: define `id` in the frozen schema as a per-call id shared by all fields of a `decide()` call, unique on `(id, field)`. Any consumer can then regroup fields into calls. The frozen list does not say this today.

Why not only Tarnlight: Windows-only tested, heavy (Qt), needs a pre-existing folder and cannot carry `T`. Why not only our own log: a finished live console (chart, gauge, feed, grading, review queue) comes free, and the brief's console promise is met without building one.

Legal and naming: Tarnlight is Apache-2.0 and unofficial. Our sink should be reimplemented from the documented format (short, original code), name Tarnlight as a compatible console, and keep "Jev" out of product names.

### Smallest separate console, if the founder still wants one

Zero dependencies, about 120 lines, no Qt:

- `mirethstm watch LOG.jsonl` starts a stdlib `http.server` on localhost serving one static HTML file and an `/events?from=<byte offset>` endpoint that returns new complete lines (tail by byte offset, whole lines only, the same idea as Tarnlight's read positions).
- The page polls every 250 ms, keeps a ring of the newest 2,000 events per `field`, and draws on a `<canvas>`: a confidence line per field chip (`p`), a dashed escalate threshold, a gauge for the latest event, and a feed of the last 50 rows `(ts, field, label, p, latency_ms)`.
- Out of scope for the minimal version: grading, persisted bands, calibration view, export, multiple sessions. ECE and reliability diagrams belong to the calibration and benchmark module.
- Even smaller: a terminal view with `rich` (Live table with a sparkline per field), about 60 lines, at the cost of stock-chart style graphs.

## 6. What this means for MirethSTM1

SPEC:
- Freeze the per-field JSONL as written, and add: `id` is a per-call id, `(id, field)` is unique; one line per field; `ts` is epoch seconds (float); `p` is the probability of `label` after temperature; `probs` sums to 1; `T` is the temperature used; `model` is the model id; `latency_ms` is per call (say so).
- Add an "Interop: Tarnlight" section: a `decide()` result maps to the TypeSafe wire shape (noul, choice, score) and to a drop box envelope `{v:1, ts, source:"mirethstm", request:{state, model, questions}, response:{model, answers, usage}, latency_ms, status:200, cost_est_micro:0}`.
- State our `confidence` definition for choice and score (top probability after T) and that it is not comparable to TypeSafe's unpublished confidence.
- Write-once rule: one record per decision, retries never duplicated, because the console has no dedup.

Scaffold:
- `mirethstm/events.py`: frozen-schema writer (append JSONL, flush per call).
- `mirethstm/sinks/tarnlight.py`: optional sink, no dependencies, folder-exists check, hourly UTC file name with a sanitised source, one append write per line, never raises, off by default.
- Tests: build a record and assert it passes Tarnlight's door rules (`v == 1`, finite `ts`, text fields are text, number fields finite) and the indexer needs (choice has `choice`, `probabilities`, `confidence`; noul has `noul`; score has `score`, `probabilities`, `confidence`). Mirror the rules in the test; do not import Tarnlight.
- The minimal console above is optional and is cut with MCP and FastAPI if time is short. Tarnlight already covers the "live console" promise.

Open items for the founder:
- Does the public promise mean "works with Tarnlight" or "a console ships in this repo"? The recommendation meets the first at almost no cost and the second with about 120 lines.
- Optional: a small change to Tarnlight (same owner) for non-TypeSafe sources: cost 0, hide replay, show `T`.
