# 02. Harsha Gundala's "RLCD" repo and its ports

Research date: 2026-09-30. Status tags: VERIFIED (read in code, docs or API output by the researcher), UNVERIFIED (secondary source, or not checked), CONTRADICTED (the brief or the repo's own docs say X, reality is Y).

All repos were cloned with `GIT_LFS_SKIP_SMUDGE=1` (no weights) and read in full. No code was executed (no GPU or Mac here); behavior claims come from reading the code.

## 1. Summary

1. There is **no training code and no trained weights** in any of the three main repos. All load stock `Qwen/Qwen2.5-1.5B-Instruct` (MLX path: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`). "RLCD" appears only as aliases (`run_rlcd_generation`, `compile_rlcd_metadata`, `/api/run-rlcd`). The acronym is never expanded and nothing implements reinforcement learning. VERIFIED.
2. The original scores the **first token only**, for enum and boolean fields only. The brief's "enum/bool only; first-token only" is correct. VERIFIED.
3. "Calibrated" means a plain softmax over raw logits at T=1.0 (`temperature` is a request parameter). Nothing is fit, and no accuracy, F1, ECE or Brier number exists in the original repos. shreyansh26's model card says so explicitly. VERIFIED.
4. The Transformers/PyTorch/CUDA port (shreyansh26) is the most useful reference: it adds **full-label scoring** for colliding candidates and a FlexAttention **tree mode**. VERIFIED.
5. The "Fast" variant (epsilon3) is an Apple MPS tree-attention speed change. It stays first-token only and only reports collisions. VERIFIED.
6. The brief says shreyansh26 is a "CUDA" port: true, but its `uv.lock` uses the cu128 index only under the marker `sys_platform == 'linux' and platform_machine == 'x86_64'`, so it does not select the CUDA wheel on native Windows as shipped. VERIFIED (`pyproject.toml`).

## 2. Repo facts

HF API `https://huggingface.co/api/models/<id>`, GitHub API `gh api repos/<o>/<r>`. Date checked 2026-09-30.

| Repo | Likes/stars | Downloads | License (card/API) | Last change | Notes |
|---|---:|---:|---|---|---|
| harshatheg/Qwen-2.5-1B-RLCD | 577 | 0 | apache-2.0 | 2026-09-16 | 3 commits, HEAD `2af8684`; library_name mlx |
| shreyansh26/Qwen-2.5-1B-RLCD | 4 | 0 | apache-2.0 | 2026-09-16 | duplicate of harsha, plus one commit `217e8f8` "Migrate to CUDA and add tree attention masking" |
| epsilon3/Qwen-2.5-1B-RLCD-Fast | 0 | 0 | apache-2.0 | 2026-09-16 | derived from harsha `2af8684` per its NOTICE; HEAD `300e12b` |
| keysleo43/Qwen-2.5-1B-RLCD | 0 | 0 | apache-2.0 | 2026-09-26 | mirror |
| botp/Qwen-2.5-1B-RLCD | 2 | 0 | apache-2.0 | 2026-09-18 | mirror |
| Space drinkmoonshine/parallel-constrained-decoding | 34 | n/a | apache-2.0 (README frontmatter) | 2026-09-16 | Gradio, ZeroGPU `zero-a10g`, RUNNING, SHA `2cb1107` |
| github.com/vicksiyi/qwen-2.5-1b-rlcd | 3 stars | n/a | none (no LICENSE file, API returns no SPDX) | pushed 2026-09-19 | derivative, not a pure mirror |
| github.com/rnorth/rlcd-play | 1 star | n/a | none | pushed 2026-09-17 | copy plus small UI change; GitHub API `fork: false` (the brief calls it a fork) |

Downloads are 0 because the repos hold code, not weights. The only social signal is harsha's 577 likes.

### Licenses

- Code and cards: Apache-2.0 by frontmatter and the harsha README's last line. **No LICENSE file exists** in harsha, shreyansh26 or epsilon3 (`git ls-files` has no match). The harsha README also has a placeholder clone URL (`github.com/your-org/parallel-constrained-decoding`). epsilon3 ships a `NOTICE` crediting harsha at `2af8684`. VERIFIED.
- Base weights: `Qwen/Qwen2.5-1.5B-Instruct` is `apache-2.0` and `mlx-community/Qwen2.5-1.5B-Instruct-4bit` is `apache-2.0` (HF API). For contrast `Qwen/Qwen2.5-3B-Instruct` is `other` / `qwen-research`, matching the brief's warning. VERIFIED. We use Qwen3, so this is context only.

## 3. Harsha original (HEAD `2af8684`)

Key files: `core/schema.py`, `core/engine_torch.py` (PyTorch, CUDA or CPU), `core/engine_mlx.py` (Mac only), `core/prompt_builder.py`, `server/app.py` (FastAPI), `app.py` (Gradio Space), `presets/*.json` (4 scenarios, up to 28 fields; one field with 255 choices).

### 3.1 Input schema (VERIFIED, `schema.py`, `server/app.py`)

```json
{"context": "<free text>",
 "schema": {"fraud_risk": {"type": "enum", "description": "...", "choices": ["LOW","ELEVATED","SUSPICIOUS","CRITICAL"]},
            "block_account": {"type": "boolean", "description": "..."}},
 "temperature": 1.0}
```

- Types: `boolean`, `enum` (aliases `choice`, `selection`). Anything else raises ValueError. No numeric or score type.
- Max 255 choices per enum (`schema.py` line 21). No uniqueness check.
- Routes: `POST /api/run-parallel` (alias `/api/run-rlcd`), `/api/run-naive`, `/api/stream-naive` (SSE), `/api/compare`, `GET /api/presets`.

### 3.2 Output schema (VERIFIED, `engine_torch.py` lines 158 to 193)

```json
{"mode": "parallel_constrained_calibrated", "elapsed_ms": 75.1, "prefill_ms": 0, "suffix_eval_ms": 0,
 "total_tokens_generated": 0, "sequential_forward_passes": 1, "is_valid_json": true, "schema_match": true,
 "parsed_json": {"fraud_risk": {"value": "CRITICAL", "prob": 0.9942}},
 "field_telemetry": {"fraud_risk": {"value": "CRITICAL", "type": "enum", "confidence": 0.9942,
    "cardinality": 4, "top_choices": [{"choice": "CRITICAL", "probability": 0.9942}]}},
 "has_calibrated_probabilities": true, "num_fields": 3, "device": "cuda"}
```

Only the top 5 choices come back, rounded to 4 decimals, so the full distribution is not exposed. `is_valid_json`, `schema_match` and `has_calibrated_probabilities` are hard-coded `True`, and `sequential_forward_passes` is hard-coded 1 (there are two passes: prefill and the suffix batch). The README example shows a boolean as the string `"true"`, while the torch path returns a Python bool. VERIFIED.

### 3.3 Parallel mechanism (VERIFIED, `engine_torch.py` lines 93 to 135)

1. One prompt: `<|im_start|>system\nClassify JSON attributes:\n{schema_str}<|im_end|>\n<|im_start|>user\n{context}<|im_end|>\n<|im_start|>assistant\n{\n`. `schema_str` is only `  "name": <first line of description>` per field. **The allowed choices are not in the parallel prompt** (the naive baseline prompt does list them), so the model never sees the label set.
2. Prefill once: `model(base_toks, use_cache=True)`.
3. Per field, tokenize a short JSON-shaped suffix (3.4); right-pad all suffixes to the longest with the pad id.
4. Broadcast: `copy.deepcopy(cache)` then `batch_repeat_interleave(M)` (a full physical copy of the prefix KV per field).
5. One forward over `[M, max_suffix_len]` with `attention_mask = [ones(prefix_len) | suffix_mask]`. No explicit `position_ids`: correctness relies on right padding so real tokens keep positions `prefix_len + i`.
6. Read `logits[i, suffix_len_i - 1]`, gather candidate token ids, divide by T, softmax.

### 3.4 Suffix format (what our scorer must write) (VERIFIED, `schema.py` lines 147 to 168)

- Boolean: suffix `  "<name>": ` (two spaces, quoted key, colon, **trailing space**). Candidates: first token of `true` and of `false`.
- Enum: `prefix = os.path.commonprefix(choices)`; suffix `  "<name>": "<prefix>` (opening quote, shared prefix absorbed). Candidates: first token of `choice[len(prefix):]`, or the `"` token when the remainder is empty.

Tokenizer check I ran (Qwen2.5 `tokenizer.json`, `tokenizers` library):

| Text | Tokens |
|---|---|
| `  "flag": ` (harsha boolean suffix) | `Ġ` `Ġ"` `flag` `":` `Ġ` (lone space token 220 last) |
| `  "flag": true` (natural) | `Ġ` `Ġ"` `flag` `":` `Ġtrue` (token 830) |
| `SEV_0_CRITICAL` / `SEV_1_MAJOR` | `SE V _ 0 _CRITICAL` / `SE V _ 1 _MAJOR` |
| `Sports` / `Sci-Tech` / `Sci/Tech` | `Sports` / `Sci -T ech` / `Sci /T ech` |
| `Business"` / `World"` | `Business "` / `World "` (closing quote is its own token here) |

Findings: harsha's boolean suffix ends in a lone space token and then scores bare `true`/`false`, which the tokenizer would never emit in that position (it emits ` true`). This is off-distribution but workable. Separately, `commonprefix` hoisting works at the character level, so it can cut a BPE token in half (`Sci-Tech` vs `Sci/Tech` hoist `Sci`, and the remainders `-Tech` / `/Tech` are then tokenized standalone, not as they would appear in context). Reasoning (not run): because the hoisted prefix removes every shared leading character, the remainders of two distinct labels start with different characters, so their first tokens cannot be equal. The only way to hit `has_collisions` is an empty remainder (mapped to the `"` token) meeting a remainder that begins with `"`. The collision branch is therefore close to dead code in this scheme; the real weakness is scoring one standalone-tokenized token of the remainder instead of the whole label. Do not hoist in v1.

### 3.5 First-token-only and collisions (VERIFIED, code read)

- Candidates are `c_toks[0]` (`schema.py` line 162). `has_collisions` is computed at line 168.
- **Torch path: `has_collisions` is read into a variable and never used.** If two labels did share a first token they would get identical logits, `argmax` would return the first, and softmax would split mass between them. (See 3.4: with char-level hoisting this is rarely reachable.)
- **MLX path (`engine_mlx.py` lines 362 to 405):** for a colliding field it greedily decodes up to 4 tokens, matches the string to a choice (`startswith` in either direction, then a digit-index fallback, then `choices[0]`), then **fabricates the distribution**: `w_prob = clamp(product of token probs, 0.75, 0.9999)` and the remaining mass is spread uniformly over the other choices. This is independently described in vicksiyi's `README.run_parallel_generation.zh-CN.md` (lines 143 and 154): even non-colliding multi-token labels are scored only on the first token of the remainder.

### 3.6 Weights, training, calibration

- Weights: `MODEL_ID` env default `Qwen/Qwen2.5-1.5B-Instruct` (`engine_torch.py` line 22); `mlx-community/Qwen2.5-1.5B-Instruct-4bit` in `engine_mlx.py` line 22. bf16 or fp16 on CUDA, fp32 on CPU.
- Training: no training script, dataset, LoRA, reward model or weight file is in the tree, and grepping for train/finetune/LoRA/reinforce finds nothing relevant. The model card says "This repository provides an inference implementation". Verdict: **"RLCD" here labels inference code on an unmodified instruct model.** VERIFIED. (Do not assume this represents how TypeSafe's Jev works; that belongs to another research file.)
- Calibration: none. `has_calibrated_probabilities: True` and "Calibrated Field Confidence" mean normalization. `core/benchmark.py` measures latency and syntax only; the README mentions "accuracy comparison" in the UI but no accuracy dataset or metric exists. VERIFIED.

### 3.7 Claimed benchmarks

README: M4 Max, 4 fields 420 ms vs 75 ms (5.6x), 28 fields 1,900 ms vs 270 ms (7.0x), 255 choices 500 ms vs 89 ms. The numbers are round, no raw log is included, and "100% schema validity" is hard-coded. UNVERIFIED. The naive baseline prompt asks for pretty-printed, 2-space-indented JSON, which lengthens the baseline's output and inflates the ratio.

## 4. shreyansh26 port (HEAD `217e8f8`)

Files: `core/engine.py`, `core/schema.py`, `tests/test_engine.py` (17 KB), `pyproject.toml` (torch>=2.7, transformers>=5.0,<6), `uv.lock`, FastAPI server, same web UI.

### 4.1 I/O schema

Same request as harsha, with validation added: at least one field, unique non-empty choices, max 255; `max_tokens` 1 to 4096 (naive only). New `POST /api/run-tree`; `/api/compare` adds a `tree` result. Responses keep `parsed_json: {field: {value, prob}}` and `field_telemetry` (top 5); tree adds `tree_nodes`, `prefix_tokens`, `scoring: "mean_token_log_probability"`. Booleans are real JSON booleans. VERIFIED (README, code).

### 4.2 Batched mode (`run_parallel_generation`, `engine.py` lines 204 to 264)

1. Same prompt as harsha (choices still absent), `add_special_tokens=False`.
2. Prefill with `logits_to_keep=1`.
3. `cache.batch_repeat_interleave(len(schema))` in place.
4. One padded batch forward with `attention_mask = [prefix ones | suffix_attention_mask]`; right padding, no explicit position ids.
5. Non-colliding field: softmax over `logits[[first tokens]]` (first token only).
6. **Colliding field: full-label scoring** (`_candidate_scores`, lines 170 to 200). `branch = deepcopy(cache)`, `batch_select_indices([i])`, crop right padding. Score = `log_softmax(first_logits)[first_token]` + sum of continuation log-probs (`log_probs[row, arange, tokens[1:]].sum()`), **including the closing delimiter**. Candidates are processed 16 at a time (`deepcopy`, `batch_repeat_interleave(batch)`, right-padded forward with mask) to bound memory. This matches the algorithm in the brief's Phase 3. Its tests (`test_cached_scores_match_full_forward`, `test_batched_padding_and_collision`) compare against independent full forwards on a tiny random Qwen2 model (read, not run).
7. Asymmetry: fields without a first-token collision are still scored on the first token only, so a field's scoring method depends on whether its labels collide.

### 4.3 Suffix format (`schema.py` lines 146 to 160)

- Key via `json.dumps(name)`.
- Boolean: suffix `  "<name>": `, continuations `true,` and `false,` (the comma is scored).
- Enum: labels escaped by `json.dumps(c, ensure_ascii=False)[1:-1]`; suffix `  "<name>": "<commonprefix>`; continuation `c[len(prefix):] + '"'`.
- Pad id `tokenizer.pad_token_id` else 0.

### 4.4 Tree mode (`run_parallel_generation_tree`, lines 267 to 365)

- `compile_tree_metadata` builds a token trie over all field suffixes and all candidate continuations with DFS intervals for ancestor tests.
- One forward on batch size 1 (no KV repetition): FlexAttention `create_block_mask` with predicate "key in prefix, or key is an ancestor of the query"; `position_ids = depth + prefix_length`; `logits_to_keep` limited to needed parent rows. Backend switched to `flex_attention` for the call and restored in `finally`.
- Score = sum of edge log-probs **divided by candidate length** (mean token log-prob, line 342). Mean normalization flattens distributions; a test targets the shorter-candidate bias. Summed log-prob is the proper sequence likelihood.
- Requires full attention (raises on `sliding_attention`), `torch.compile(fullgraph=True)`, and FlexAttention (Triton). Whether that works on native Windows is UNVERIFIED; do not plan on it.

### 4.5 Weights, calibration

- Weights: stock `Qwen/Qwen2.5-1.5B-Instruct` ("unmodified ... does not contain a new trained checkpoint", its model card). VERIFIED.
- Calibration: none; "Reported probabilities are normalized model scores, not empirically calibrated confidence." VERIFIED.
- Authorship: two commits by Shreyansh Singh on 2026-09-16.

## 5. epsilon3 "Fast" variant (HEAD `300e12b`)

- Derived from harsha `2af8684`; `NOTICE` and `UPSTREAM_*` docs preserved. Its README: "inference code, not new or fine-tuned weights. Despite the repository name, the actual model is Qwen/Qwen2.5-1.5B-Instruct". VERIFIED.
- Mechanism (`tree_decode.py`, `core/engine_tree.py`): prefill via `model.model(ids)`, then one packed pass with a dense additive 4D mask (each suffix token sees the prefix and its own branch ancestors) and explicit `position_ids = prefix_len + depth`. The prefix KV is not repeated. Only endpoint hidden states go through `lm_head`. Requires `transformers==4.57.6` (hand-builds `DynamicCache(ddp_cache_data=...)`), SDPA or eager. VERIFIED.
- Scoring: **first token only** (`logits[i, meta['cands_per_field'][i]]`), `has_calibrated_probabilities=False`, collisions only listed in `candidate_collision_fields`. VERIFIED.
- Benchmarks: M4 Pro, MPS fp16, 8 randomized paired trials, batch vs tree decode 1.41x to 2.37x at 4 to 28 fields (1.07x to 1.41x including prefill). Raw JSON is in the repo and it discloses a macOS-version inconsistency. The claim is batch vs tree on the same model, not accuracy. UNVERIFIED by me.

## 6. Live demo Space

- HF API: `sdk: gradio`, stage RUNNING, hardware `zero-a10g`, 34 likes, SHA `2cb1107`, created 2026-09-16. VERIFIED.
- Code: the Space's `core/engine_torch.py` is **byte-identical to harsha's HEAD** (`cmp`). `app.py` (Gradio) runs `run_parallel_generation_torch` (first-token only, collision flag unused) against `run_naive_generation_torch` (`generate`, max 512 new tokens), loading `Qwen/Qwen2.5-1.5B-Instruct` in bf16 under `spaces.GPU(duration=60)`. The Dockerfile is a leftover (unused for a Gradio SDK Space). VERIFIED.
- CONTRADICTED (docs vs runtime): the Space README still says "for Apple Silicon using MLX" and quotes the M4 Max 5.6x/7.0x figures, but the Space executes the torch/CUDA path on an A10G.
- The public demo therefore shows the weakest scoring variant.

## 7. Mirrors and derivatives

| Repo | Verdict |
|---|---|
| keysleo43/Qwen-2.5-1B-RLCD | **Mirror**: one commit "Duplicate from harshatheg/Qwen-2.5-1B-RLCD" (2026-09-26); tree identical to harsha HEAD (`diff -rq` empty). VERIFIED |
| botp/Qwen-2.5-1B-RLCD | **Mirror**: same pattern (2026-09-18); identical tree. VERIFIED |
| vicksiyi/qwen-2.5-1b-rlcd | Derivative (not a mirror): adds `decision_demo.py`, docs in Chinese, a game host, a modified `engine_mlx.py`; no LICENSE file. Its run_parallel_generation write-up is a good independent read of the code. VERIFIED |
| rnorth/rlcd-play | Copy of harsha's tree plus an editable-context UI; not a GitHub fork. Low value. VERIFIED |

## 8. What this means for MirethSTM1

### Reuse (ideas, with attribution)

1. **Mechanism skeleton** (harshatheg, Apache-2.0): prefill once, batched per-field scoring against the shared cache with an attention mask, softmax over allowed answers, typed assembly. README credits: "inspired by harshatheg/Qwen-2.5-1B-RLCD; full-label chunked scoring also appears in shreyansh26/Qwen-2.5-1B-RLCD".
2. **Full-label chunked scoring** (shreyansh26 `_candidate_scores`): log-softmax at the decision position, add the sum of continuation log-probs, chunks of 16 to bound memory, crop right padding before branching. This is our core scorer, reimplemented by us (clean code, attribution in README and a source comment), applied to every field, not only collisions.
3. **Score the closing delimiter** so labels that are prefixes of other labels stay distinguishable.
4. **`json.dumps` escaping** for keys and labels.
5. **Validation rules**: unique non-empty choices, at least one field, max 255, non-negative temperature.
6. **Test style**: a tiny randomly initialized Qwen2-family model, no download, compared against independent full forwards (alongside the brief's Qwen3-0.6B CPU tests).
7. **Deferred to v2:** dense-mask tree pass with branch-local `position_ids` and endpoint-only LM-head projection (epsilon3); SDPA-compatible, no FlexAttention.

### Avoid

1. First-token-only scoring, and computing `has_collisions` without using it.
2. Fabricated probabilities (MLX clamp 0.75 to 0.9999 with uniform remainder). Never return a number we did not compute.
3. Calling raw softmax "calibrated". Publish `p_raw` vs calibrated `p`; claim calibration only after `fit_temperature` and a held-out ECE.
4. Hard-coded `is_valid_json`, `schema_match`, `sequential_forward_passes`.
5. Mean-token-logprob as the default (shreyansh26 tree mode). Use the sum; make length normalization an explicit benchmarked option.
6. Leaving the allowed labels out of the prompt (all three parallel paths do). Include labels in the prefill (cap or shorten for 255 choices) and benchmark both variants.
7. Mixed scoring methods across fields.
8. A pretty-printed-JSON baseline; our baseline is compact JSON via `generate()`, and we report both if useful.
9. FlexAttention/Triton dependency on native Windows.
10. Copying version pins or private cache constructors. The cache methods used (`batch_repeat_interleave`, `batch_select_indices`, `crop`, `get_seq_length`) must be checked against the transformers version we pin.
11. Returning only the top 5 choices; we return the full distribution.

### Per-field prompt and suffix format for our scorer

Prefill via the tokenizer's chat template (not hand-written special tokens, so Qwen3 works):

```
system: <decision instruction + per field: name, description, allowed labels>
user:   <context>
assistant (prefilled): {
```

Each field is then a fixed JSON prefix plus the scored continuation. All answers of a field share the same prefix so scores are comparable:

| Type | Fixed prefix | Scored continuation per answer | Note |
|---|---|---|---|
| `choice` (up to 255) | `  "<json.dumps(name)>": "` | `<json-escaped label>"` | Sum of label-token log-probs plus closing quote; no commonprefix hoisting in v1. |
| `noul` (yes/no) | `  "<name>":` | ` true,` and ` false,` | Ending at `":` and scoring the leading-space token matches natural tokenization (harsha's lone space token 220 is off-distribution). |
| `score` (min/max via bins) | `  "<name>": "` | bin labels (e.g. `1` to `5`, or ranges), scored like `choice` | Expected value and distribution from bin probabilities; no equivalent in any of these repos. |

Test cases: `Sci-Tech` vs `Sci/Tech` (full-label tokenization shares the first token `Sci` on Qwen2.5, a true first-token collision when labels are scored whole) and `Sports` (1 token) vs `Sci-Tech` (3 tokens) for length bias. The brief's example `Sports` vs `Sci-Tech` is not a first-token collision by itself on Qwen2.5 (first tokens `Sports` and `Sci` differ), so the collision test needs a pair like `Sci-Tech` / `Sci/Tech`. Qwen3 tokenizer check (reviewer, 2026-09-30, Qwen3-0.6B tokenizer, VERIFIED by running): ` Sports` is 1 token, ` Sci-Tech` splits into ` Sci` `-T` `ech`, ` Sci/Tech` into ` Sci` `/T` `ech`. So the same holds on Qwen3: Sports vs Sci-Tech is not a first-token collision, Sci-Tech vs Sci/Tech is (shared first token ` Sci`). Note AG News' own label is `Sci/Tech`.

### Spec implications

- Responses include the **full** distribution per field (harsha returns 5), keep `p_raw` and post-T `p` distinct, and carry `T`, `latency_ms`, `model` as in the frozen JSONL schema.
- Framing: these repos are stock-model inference; "RLCD training" is not present in any of them. README says "Jev-style" and credits the inspiration honestly.
- Differentiators 1 to 3 in the brief are real gaps: full-label scoring (only shreyansh26's collision path has it), bins (none), calibration (none).
- README license table: Apache-2.0 for the three reference repos and for Qwen2.5-1.5B(-Instruct, -4bit); Qwen2.5-3B is `qwen-research` (non-commercial). Qwen3 cards are covered by another research file.

## 9. Open questions

1. What the origin post claimed about training: not checked (X pages cannot be fetched). UNVERIFIED.
2. Whether the claimed latencies reproduce: not measured.
3. Qwen3 tokenizer splits for the test labels: checked by the reviewer, see section 8.
4. Behavior of `DynamicCache.batch_repeat_interleave` and `crop` in our pinned transformers version: to be tested in the scaffold.

## 10. Sources

- https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD (HEAD 2af8684): `core/schema.py` lines 10-67, 134-189; `core/engine_torch.py` lines 70-193; `core/engine_mlx.py` lines 290-420; `core/prompt_builder.py`; `README.md`; `MODEL_CARD.md`
- https://huggingface.co/shreyansh26/Qwen-2.5-1B-RLCD (HEAD 217e8f8): `core/engine.py` lines 170-365; `core/schema.py` lines 140-260; `MODEL_CARD.md`; `README.md`; `pyproject.toml`; `tests/test_engine.py`
- https://huggingface.co/epsilon3/Qwen-2.5-1B-RLCD-Fast (HEAD 300e12b): `tree_decode.py`, `core/engine_tree.py`, `NOTICE`, `README.md`, `M4_RESULTS.md`
- https://huggingface.co/spaces/drinkmoonshine/parallel-constrained-decoding (SHA 2cb1107) and https://huggingface.co/api/spaces/drinkmoonshine/parallel-constrained-decoding
- https://huggingface.co/keysleo43/Qwen-2.5-1B-RLCD, https://huggingface.co/botp/Qwen-2.5-1B-RLCD
- https://github.com/vicksiyi/qwen-2.5-1b-rlcd (HEAD d7e7ede), https://github.com/rnorth/rlcd-play (HEAD ded7724)
- https://huggingface.co/api/models/Qwen/Qwen2.5-1.5B-Instruct, https://huggingface.co/api/models/Qwen/Qwen2.5-3B-Instruct, https://huggingface.co/api/models/mlx-community/Qwen2.5-1.5B-Instruct-4bit
