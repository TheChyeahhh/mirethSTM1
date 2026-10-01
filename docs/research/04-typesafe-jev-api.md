# 04. The current TypeSafe Jev API (drop-in target)

Date checked: 2026-09-30. Status legend: VERIFIED (read in live docs or seen in a real API response), UNVERIFIED (secondary source or not reachable), CONTRADICTED (docs or brief say X, reality is Y).

Sources:
- Live docs (primary): https://docs.typesafe.ai/api.md, /primitives/{noul,choice,score}.md, /confidence.md, /models.md, /concepts/state.md, /sdk/python.md, /sdk/javascript.md, index at https://docs.typesafe.ai/llms.txt
- A working Node client (hand-written, plain `fetch`) that talks to the endpoint; used only to confirm request shape.
- Two live API calls made 2026-09-30 (raw output below).
- Secondary: https://typesafe.ai/blog/introducing-system-one-models-and-jev, https://www.turingpost.com/p/what-is-jev-rlcd, https://www.datacamp.com/blog/system-one-models-jev

## 1. Endpoint, auth, model

| Item | Value | Status |
| --- | --- | --- |
| Endpoint | `POST https://api.typesafe.ai/v1/systemone` | VERIFIED (docs /api.md; live call returned 200) |
| Auth | `Authorization: Bearer <API_KEY>` | VERIFIED |
| Content type | `Content-Type: application/json` | VERIFIED |
| Model list | `GET /v1/models` exists (per /models.md); not called by us | UNVERIFIED (docs only) |
| Model names | `jev-latest` and `jev-preview` are aliases; both resolve to `jev-1.13.0` | VERIFIED (docs /models.md); live response echoed `"model":"jev-1.13.0"` when `jev-latest` was sent |
| Request id | Response header `x-typesafe-request-id: req_<32 hex>` | VERIFIED (live call) |
| SDKs | Python `typesafe-sdk` (sync `TypeSafeClient`, async `AsyncTypeSafeClient`), JS `@typesafe-ai/sdk` (Node 20+), key from env `TYPESAFE_API_KEY` | VERIFIED (docs) |
| Access | Launch reporting says early access behind a waitlist | UNVERIFIED (DataCamp) |

Brief check: the endpoint in the brief (`POST https://api.typesafe.ai/v1/systemone`) is correct. VERIFIED.

## 2. Request body

Top level, all three required (VERIFIED, /api.md):

| Field | Type | Notes |
| --- | --- | --- |
| `model` | string | Docs say use `"jev-latest"` |
| `state` | string, object, or array | Text only. Object preferred so parts have names. Strings inside are the only leaf type (no image/audio) |
| `questions` | map of question id to Question | Keys are chosen by the caller; ids are for code and are NOT sent to the model (docs, skill guidance) |

Question objects:

| Type | Fields | Status |
| --- | --- | --- |
| `noul` | `type:"noul"`, `instructions` (string, object or array), `criteria` optional: `{ "true": <desc>, "false": <desc> }` | VERIFIED (/api.md, /primitives/noul.md says criteria optional) |
| `choice` | `type:"choice"`, `instructions`, `criteria`: map `option name -> description`; description may be string, object, array or `null` | VERIFIED (/api.md; live call used `null` for an option and it worked) |
| `score` | `type:"score"`, `instructions`, `criteria`: ordered ARRAY of level descriptions (strings, or objects with `what` and `examples`) | VERIFIED |

Details:
- Choice max options: 255 (docs /api.md, /primitives/choice.md; blog says "up to 255"). VERIFIED in docs, not stress-tested by us.
- Score levels: docs say 2 to 10 ordered levels. CONTRADICTED in the lenient direction: a live call with a single level returned HTTP 200, not 422 (see raw output 2). Do not rely on that; our server should accept 2 to 10 and may also accept 1.
- Re-read by the reviewer on 2026-09-30 (https://docs.typesafe.ai/api.md): "A Score should have at least two levels; the API accepts up to 10." Other implementations differ: openjev 1 to 10 levels (doc 01), Kev 1 to 255 (doc 03). The limit is therefore TypeSafe 2 to 10 (docs), 1 accepted in practice (doc 04 live call).
- Score has NO min/max/steps/anchor fields. It is an array of level descriptions; levels are addressed by 0-based index. This differs from the brief's "score: min/max, handled via bins" wording: the brief's min/max is our own design, TypeSafe's wire format is a level array. CONTRADICTED (brief vs reality).
- Descriptions may be structured objects. Docs suggest `what`, `not_for`, `examples` for choice options and `what`, `examples` for score levels (docs). These are free-form hints, not validated fields as far as we can tell. UNVERIFIED that any key other than free JSON is interpreted specially.
- Instructions can reference state with backticked paths such as `` `ticket.messages[0].text` `` (skill doc). UNVERIFIED in our own calls.

## 3. Response body

Top level (VERIFIED, /api.md and live):

| Field | JSON type | Notes |
| --- | --- | --- |
| `model` | string | Resolved model name, e.g. `"jev-1.13.0"` |
| `answers` | object | One entry per question id |
| `usage.input_tokens` | integer | Live: 470 for the test request |
| `usage.output_tokens` | integer | Live: 72. Output tokens are free (see pricing) |

There is no per-request `id` in the body and no `latency` field. The request id is only in the `x-typesafe-request-id` header. VERIFIED (live).

Per-type answers (every answer carries a `type` discriminator string):

| Type | Fields (JSON type) | Observed |
| --- | --- | --- |
| noul | `type` string, `noul` number 0..1. NO confidence, NO probabilities field | `{"type":"noul","noul":0.58}` |
| choice | `type`, `choice` string (argmax option), `confidence` number 0..1, `probabilities` object option->number (sums to about 1, includes every option) | see raw output |
| score | `type`, `score` number, `confidence` number 0..1, `legend` object level-index-string -> description string, `probabilities` object level-index-string -> number | see raw output |

Facts from the live capture:
- Numbers are rounded to 2 decimals (probabilities, scores, confidences). Matters for us: emit 2 decimals to look identical, or more precision as a superset (clients should not care).
- Score `score` is the probability-weighted mean of 0-based level indices, range 0 to (n-1). Check: `0*0.00 + 1*0.04 + 2*0.95 + 3*0.01 = 1.97`, matches the returned 1.97. VERIFIED (live arithmetic; docs say "probability-weighted mean of the level numbers").
- Score `legend` and `probabilities` keys are STRINGS of the integer index (`"0"`, `"1"`, ...), not the level text. VERIFIED.
- Choice `probabilities` key order in the response was not the request order (live: other, bug, billing). Key order is not meaningful.
- Choice `confidence` matches `(3*p_max - 1)/2` for 3 options: `(3*0.68-1)/2 = 0.52`, returned 0.52. The docs present that as an approximation "for three options" (docs /confidence.md). The general formula is not published; Turing Post also says the exact calculation is undisclosed. A normalized form consistent with this is `(K*p_max - 1)/(K - 1)` (p_max of 1 gives 1, uniform gives 0), but K-generalization is UNVERIFIED. For the score example: p_max 0.95 over 4 levels, returned 0.95; `(4*0.95-1)/3 = 0.933`, so that generalization does NOT match (rounded). The score confidence formula is therefore unknown. One data point only: 1 level gave confidence 1.0.
- Noul gives no confidence: "Noul answers don't carry one" (docs /confidence.md). Our server must not invent a `confidence` for noul if we want strict drop-in; adding extra fields is a superset and safe for tolerant clients, but strict typed SDK clients may choke (UNVERIFIED either way).

## 4. Errors

| Code | Meaning (docs /api.md) | Status |
| --- | --- | --- |
| 401 | Missing or invalid API key | VERIFIED in docs, not reproduced |
| 422 | Request validation failed | VERIFIED in docs, not reproduced |
| 429 | Rate limit, retry with exponential backoff | VERIFIED in docs |
| 529 | Overloaded, retry with exponential backoff | VERIFIED in docs |

The JSON error BODY shape is not documented on any page we could read (API, Python SDK and JS SDK pages all omit it). UNVERIFIED. A working client reads `await res.json()` on failure and treats it as opaque, and the request id header is present on responses. Recommendation: our server returns `{"error": {"type": "...", "message": "..."}}` style JSON with the same status codes, and tell users the body shape is our own. The single error-path probe we made (one-level score) did not error, so we have no captured error body.

Attempted-but-not-done: we did not spend a third call to force a 422 (limit of 2 calls).

## 5. Limits and pricing

| Item | Value | Status |
| --- | --- | --- |
| Context per request | 64k tokens total; 32k for state plus the longest question | VERIFIED (docs /models.md) |
| Max questions per call | Not stated anywhere we read. Docs encourage many questions per request, run in parallel | UNVERIFIED |
| Max choice options | 255 | VERIFIED (docs) |
| Score levels | 2 to 10 (docs); 1 accepted in practice | see section 2 |
| Rate limits | 100K tokens/s and 40 requests/s, "can change without notice" | VERIFIED (docs /models.md) |
| Pricing | $0.042 per million input tokens ($42 per billion); output tokens free | VERIFIED (docs /models.md; blog and DataCamp agree) |
| Latency | 70 to 500 ms end-to-end (vendor claim) | UNVERIFIED (vendor marketing, not measured here) |
| Language | English primary; other languages accepted at lower accuracy; text only | VERIFIED (docs /concepts/state.md) |

## 6. Raw captured output (2026-09-30)

Request 1 (neutral made-up support ticket; one noul, one choice with a `null` description, one 4-level score):

```json
{
  "model": "jev-latest",
  "state": {"ticket": {"subject": "Charged twice for my subscription",
            "body": "I was billed two times this month and the app also crashes when I open settings. Please fix ASAP."}},
  "questions": {
    "wants_refund": {"type": "noul", "instructions": "Is the customer asking for money back?",
      "criteria": {"true": "Explicitly or implicitly requests a refund or reversal of a charge",
                   "false": "Does not ask for money back"}},
    "category": {"type": "choice", "instructions": "Which team should handle this ticket?",
      "criteria": {"billing": "Charges, invoices, refunds", "bug": "Software defects or crashes", "other": null}},
    "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
      "criteria": ["No time pressure", "Some urgency, can wait days", "Needs attention today", "Critical outage"]}
  }
}
```

Response 1: HTTP 200, `content-type: application/json`, header `x-typesafe-request-id: req_01a0f4780c7470aa982b7f8511895991`:

```json
{"model":"jev-1.13.0","answers":{"wants_refund":{"type":"noul","noul":0.58},"category":{"type":"choice","choice":"billing","confidence":0.52,"probabilities":{"other":0.28,"bug":0.04,"billing":0.68}},"urgency":{"type":"score","score":1.97,"confidence":0.95,"legend":{"0":"No time pressure","1":"Some urgency, can wait days","2":"Needs attention today","3":"Critical outage"},"probabilities":{"0":0.0,"1":0.04,"2":0.95,"3":0.01}}},"usage":{"input_tokens":470,"output_tokens":72}}
```

Request 2 (validation probe: state `"x"`, one score question with a single level `["only one"]`). Response 2: HTTP 200 (not 422), request id `req_01a0f478221473fe9b160c845612262f`:

```json
{"model":"jev-1.13.0","answers":{"q":{"type":"score","score":0.0,"confidence":1.0,"legend":{"0":"only one"},"probabilities":{"0":1.0}}},"usage":{"input_tokens":281,"output_tokens":17}}
```

Notes on the capture: the ticket says "billed twice" and "Please fix ASAP" and Jev gave noul 0.58 for a refund request, so the model is fairly uncertain on implicit asks; not an API fact, just an honest observation.

## 7. What this means for MirethSTM1

Wire format we must match exactly (POST `/v1/systemone`; also serve our own `POST /v1/decide` as in the brief, which can be a thin alias over the same handler):

1. Accept `model`, `state` (string, object or array), `questions` (map). Treat `model` as accepted-and-echoed but mapped: accept `jev-latest`, `jev-preview` and our own names (for example `mirethstm-qwen3-4b`). Return our real model name in `model`, never `jev-*`.
2. Score is an ordered ARRAY of level descriptions, 0-based indices. The brief's "score: min/max via bins" is an internal idea. To be drop-in: bins ARE the levels; the returned `score` is the expected index; `legend` and `probabilities` keys are index strings. If we also want min/max numeric ranges (for example Yelp 1 to 5), offer it as an optional extension field (for example `range: [min, max]`) that TypeSafe clients never send.
3. Choice `criteria` is a map with optional `null` descriptions. Option names must be used verbatim as labels and as `probabilities` keys. Up to 255 options. Note: full-label log-prob scoring (our differentiator 1) means each option's label is scored as a whole, so the option name itself is the score target, and descriptions go into the prompt context.
4. Noul returns only `{type, noul}`. Do not add confidence by default.
5. Choice/score `confidence`: Jev's formula is undisclosed. Our best documented match is `(3*p_max-1)/2` for K=3; the score example did not fit the K-generalization. SPEC should define our own confidence explicitly (recommend `(K*p_max - 1)/(K - 1)` or normalized-entropy based) and state it differs from TypeSafe's. Do not claim equivalence.
6. Top-level `usage.input_tokens` and `usage.output_tokens` must be integers. Output tokens are 0 for us (no generation; or report the count of scored suffix tokens, to be decided). Put `latency_ms` and per-request `id` in the `x-mireth-request-id` style headers or an optional extra field; TypeSafe's body has neither.
7. Rounding: TypeSafe rounds to 2 decimals. Our JSON should keep full precision internally (needed for ECE and the JSONL events) and may round to 2 decimals in the compatible `/v1/systemone` body and keep full precision in `/v1/decide`.
8. Errors: use 401, 422, 429, 529 semantics from the docs. Body shape is not public, so pick `{"error":{"type","message"}}` and document it. Support Bearer auth but make the key optional for localhost (a local engine should not need one).
9. Limits to mirror: 255 choice options, 2 to 10 score levels (accept 1 too), 64k total context. Reject over-limit with 422.
10. Validation strictness: TypeSafe does accept odd input (1-level score). Being lenient is fine for drop-in; do not be stricter than the docs.
11. Order independence: answers are keyed by question id, so our per-field scoring loop can return them in any order. JSON objects only.
12. SDK compatibility goal: the Python SDK takes a base URL override (hand-written clients did via an environment variable on the working Node client), so a simple compatibility test is to point the official `typesafe-sdk` at our server. Not verified that the SDK exposes a `base_url` option; check when the test is written.

## 8. Drop-in compatibility checklist

| Field | TypeSafe behavior | What we must return |
| --- | --- | --- |
| Path | `POST /v1/systemone` | Same path, plus `/v1/decide` alias |
| Auth header | `Authorization: Bearer <key>` | Accept it; do not require it on localhost |
| Req `model` | string, required, `jev-latest` | Accept any string; echo our real model name in response |
| Req `state` | string or object or array, text only | Accept all three; serialize objects as JSON text into the prompt |
| Req `questions` | map id to question, ids not sent to model | Same; keep ids out of the prompt |
| noul question | `criteria.true`/`criteria.false` optional | Accept and inject into prompt; both optional |
| choice question | `criteria` map, `null` allowed, max 255 | Same; answer labels are the option keys exactly |
| score question | `criteria` ordered array, 2 to 10 levels | Same; 0-based level indices |
| Resp `model` | `"jev-1.13.0"` | Our model id string |
| Resp `answers` | object keyed by question id | Same keys |
| noul answer | `{type:"noul", noul: number}` | Same; no `confidence` |
| choice answer | `{type, choice, confidence, probabilities{option:number}}` | Same; `probabilities` includes every option, sums to 1 |
| score answer | `{type, score, confidence, legend{"i":text}, probabilities{"i":number}}` | Same; `score` = sum(i * p_i); string index keys |
| `confidence` | Undisclosed formula, in [0,1] | In [0,1], our own documented formula |
| Number rounding | 2 decimals | 2 decimals on compat route; full precision elsewhere |
| `usage` | `{input_tokens:int, output_tokens:int}` | Same two ints |
| Request id | header `x-typesafe-request-id` | Send an equivalent header; no body field |
| 401 | missing or invalid key | Only when a key is configured |
| 422 | validation failure | Same, on bad types, >255 options, >10 levels |
| 429, 529 | rate limit, overloaded | 429 if we throttle; 529 while the model is loading or busy |
| Error body | shape undocumented | Our own `{"error":{...}}`, documented |
| Content type | `application/json` | Same |
| Limits | 64k context, 32k state plus longest question; 40 req/s | Mirror the context limits; no hard rate limit needed locally |
| Pricing | $0.042 per M input tokens, output free | Not applicable (free, local) |
