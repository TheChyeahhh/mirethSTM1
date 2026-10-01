# 06. Model choices and licenses

Date checked: 2026-09-30. Status tags: VERIFIED (read in API output, config or code), UNVERIFIED (secondary source or not accessible), CONTRADICTED (brief differs from reality).

All Hugging Face numbers come from `https://huggingface.co/api/models/<id>`, `.../raw/main/config.json`, `.../raw/main/tokenizer_config.json` (chat template) and `.../api/models/<id>/tree/main` (file sizes). Transformers facts come from `huggingface/transformers` main (PyPI release 5.18.0 at time of check).

## 1. License confirmation

| Model | License field (HF API) | LICENSE file | Status | Link |
| --- | --- | --- | --- | --- |
| Qwen/Qwen3-0.6B | apache-2.0 | Apache 2.0 | VERIFIED (field) | https://huggingface.co/Qwen/Qwen3-0.6B |
| Qwen/Qwen3-1.7B | apache-2.0 | Apache 2.0 | VERIFIED (field) | https://huggingface.co/Qwen/Qwen3-1.7B |
| Qwen/Qwen3-4B | apache-2.0 | Apache 2.0 | VERIFIED (field) | https://huggingface.co/Qwen/Qwen3-4B |
| Qwen/Qwen3-4B-Instruct-2507 | apache-2.0 | Apache License 2.0 (file read) | VERIFIED | https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 |
| Qwen/Qwen3.5-0.8B, 2B, 4B | apache-2.0 | not opened | VERIFIED (field) | https://huggingface.co/Qwen/Qwen3.5-4B |
| Qwen/Qwen2.5-3B-Instruct | other, `qwen-research` | "Qwen RESEARCH LICENSE AGREEMENT", release 2024-09-19 | VERIFIED | https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE |

Qwen2.5-3B claim: VERIFIED. The LICENSE grants rights "FOR NON-COMMERCIAL PURPOSES ONLY", defines Non-Commercial as "for research or evaluation purposes only", and says commercial users "shall request a license" from Alibaba Cloud (sections 1.i, 2.a, 2.b). The brief's cited page https://qwen.ai/blog?id=qwen2.5-llm is a JavaScript app and returned no readable text, so the blog itself is UNVERIFIED; the model LICENSE file is the primary source and is enough.

For the README license table: Qwen3 and Qwen3.5 weights are Apache-2.0, so MirethSTM1 (Apache-2.0) can name them as defaults. Users who load Qwen2.5-3B take on the Research License themselves; do not ship it as a default.

## 2. Qwen3 dense models (config.json read directly)

| Property | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B | Qwen3-4B-Instruct-2507 |
| --- | --- | --- | --- | --- |
| Total params (safetensors) | 751,632,384 | 2,031,739,904 | 4,022,468,096 | 4,022,468,096 |
| Card says | 0.6B | 1.7B | 4.0B | 4.0B |
| Layers | 28 | 28 | 36 | 36 |
| Hidden size | 1024 | 2048 | 2560 | 2560 |
| Q heads / KV heads (GQA) | 16 / 8 | 16 / 8 | 32 / 8 | 32 / 8 |
| head_dim | 128 | 128 | 128 | 128 |
| Vocab size | 151,936 | 151,936 | 151,936 | 151,936 |
| tie_word_embeddings | True | True | True | True |
| Context (config max_position_embeddings) | 40,960 | 40,960 | 40,960 | 262,144 |
| Context (card text) | 32,768 | 32,768 | 32,768 native, 131,072 with YaRN | 262,144 native |
| rope_theta | 1,000,000 | 1,000,000 | 1,000,000 | 5,000,000 |
| Architecture class | Qwen3ForCausalLM | same | same | same |
| sliding_window / layer_types | None | None | None | None |
| dtype | bfloat16 | bfloat16 | bfloat16 | bfloat16 |
| bf16 safetensors on disk | 1.50 GB | 4.06 GB | 8.04 GB | 8.04 GB |
| Thinking behavior | hybrid | hybrid | hybrid | non-thinking only |

Sources: `https://huggingface.co/<id>/raw/main/config.json`, `.../api/models/<id>/tree/main`, card READMEs. Status: VERIFIED.

The "1.7B" and "0.6B" names exclude embeddings; total with embeddings is 2.03B and 0.75B. Budget memory from the totals.

### Plain dense transformer with standard attention? VERIFIED yes

All four are `Qwen3ForCausalLM`: full causal attention on every layer, GQA (8 KV heads), no sliding window (`use_sliding_window` False), no `layer_types`, no linear or state-space layers. A standard `DynamicCache` of per-layer K/V tensors is the whole state. Prefill once then reuse works as the brief assumes. `attn_implementation="sdpa"` is supported.

### Thinking vs non-thinking and the chat template

- Qwen3-0.6B, 1.7B, 4B (hybrid): the template contains `enable_thinking`. With `add_generation_prompt=True` and `enable_thinking=False` the prompt ends `<|im_start|>assistant\n<think>\n\n</think>\n\n`. With the flag unset it ends at `<|im_start|>assistant\n` and the model opens its own `<think>`. VERIFIED from the tokenizer_config chat_template tail.
- Qwen3-4B-Instruct-2507: the template contains no `think` string (2,630 chars vs 4,168 for the hybrids), and the card says: "This model supports only non-thinking mode and does not generate `<think></think>` blocks in its output. Meanwhile, specifying `enable_thinking=False` is no longer required." VERIFIED. Prompt ends `<|im_start|>assistant\n`.

Consequence: for hybrid models always pass `enable_thinking=False`, otherwise labels are scored as the start of a thinking trace and the probabilities are garbage. For 2507 the kwarg is harmless, so the engine can pass it unconditionally.

### Smaller 2507 sibling? VERIFIED no

`https://huggingface.co/api/models?author=Qwen&search=2507` returns only Qwen3-4B-Instruct-2507 (+FP8), Qwen3-4B-Thinking-2507 (+FP8), and the 30B-A3B and 235B-A22B Instruct/Thinking 2507 models. There is no 0.6B or 1.7B 2507. The fast and CPU-test models are therefore the older hybrid-generation checkpoints, which are weaker instruction followers. Benchmark differences between modes come from both size and generation; say so in the write-up.

## 3. Challenge check: Qwen3.5 small models

Newer small Qwen exists (cu-Jev uses Qwen3.5). Evidence:

| Model | License | Params (safetensors) | bf16 on disk | Layers | Hidden | Q/KV heads | head_dim | Vocab | Likes | Last modified |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen3.5-0.8B | apache-2.0 | 0.873B | 1.75 GB | 24 | 1024 | 8 / 2 | 256 | 248,320 | 738 | 2026-03-02 |
| Qwen/Qwen3.5-2B | apache-2.0 | 2.274B | 4.55 GB | 24 | 2048 | 8 / 2 | 256 | 248,320 | 421 | 2026-03-02 |
| Qwen/Qwen3.5-4B | apache-2.0 | 4.660B | 9.32 GB | 32 | 2560 | 16 / 4 | 256 | 248,320 | 985 | 2026-03-02 |

Also on the Qwen org: Qwen3.5-9B, 27B, 35B-A3B, 122B-A10B, 397B-A17B. Qwen3.6 exists only as 27B and 35B-A3B (HF search "Qwen3.6"). Nothing named Qwen4 was found. Status: VERIFIED via HF API.

All three: `Qwen3_5ForConditionalGeneration`, `model_type qwen3_5`, pipeline `image-text-to-text` (natively multimodal, so file size includes a vision tower), tied embeddings, context 262,144.

### Architecture: hybrid linear attention. VERIFIED.

- `layer_types` repeats `linear_attention` x3 then `full_attention` (`full_attention_interval` 4). Transformers docs: "3:1 hybrid attention stack, three Gated DeltaNet (linear attention) layers for every one Gated Attention (full attention) layer" and the text backbone "reuses Qwen3-Next's linear-attention decoder" (https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3_5.md).
- Linear layers hold a causal conv state (kernel 4) and a fixed-size recurrent state, not a growing K/V. Only 6 of 24 layers (0.8B, 2B) or 8 of 32 layers (4B) have a K/V cache.
- Transformers main has `src/transformers/models/qwen3_5/` and a hybrid cache layer `LinearAttentionAndFullAttentionLayer` in `cache_utils.py` (around line 1120) with `conv_states` and `recurrent_states`. The doc badges list SDPA.
- Default thinking differs inside the family. 0.8B and 2B templates emit an empty think block unless `enable_thinking` is true; the 4B template emits `<think>\n` (thinking) unless `enable_thinking=False`. VERIFIED from each repo's `chat_template.jinja`. Always pass the flag explicitly.

### Consequence for prefill once, then score

1. Reuse is valid in principle: the prefilled state is K/V for full layers plus conv and recurrent state for linear layers.
2. The suffix forward pass mutates the recurrent state in place. A K/V cache is append-only and can be cropped back; a recurrent state cannot. So each candidate chunk needs a deep copy of the whole prefilled state.
3. `Cache.batch_repeat_interleave` forwards to every layer (cache_utils.py, around line 1641). I did not confirm the linear-state layers implement it correctly: UNVERIFIED. Needs a CPU test.
4. Padding: a recurrent state would consume pad tokens. We read logits at real positions only, so this may be harmless, but padded-batch equivalence must be tested.
5. Kernels: Qwen3-Next style models normally want the `flash-linear-attention` and `causal-conv1d` packages for speed; otherwise a slow torch path is used. Whether these work on native Windows is UNVERIFIED (I did not run it), and it conflicts with the brief's "native Windows, sdpa, no extra kernels" plan.
6. Upside: K/V per token is 4.5x to 12x smaller than Qwen3 (section 5).

Verdict: attractive for memory and recency, but a riskier first implementation for a one-week ship on native Windows. Keep the brief's Qwen3 default. Keep the cache layer behind a small interface (`prefill`, `expand`, `score`). List Qwen3.5 as an experimental backend after a CPU test on Qwen3.5-0.8B passes.

## 4. Recommendation

| Role | Model | Evidence | Caveat |
| --- | --- | --- | --- |
| Default | Qwen3-4B-Instruct-2507 | Apache-2.0; plain dense attention; non-thinking only so no template flag needed; 262k context; 8.04 GB bf16 fits 12 GB | About 3 GiB left for activations and KV, so chunk candidates (section 5) |
| Fast | Qwen3-1.7B (`enable_thinking=False`) | Apache-2.0; 2.03B params, 4.06 GB; same tokenizer and vocab as the default | Older hybrid generation, no 2507 version; expect lower accuracy |
| CPU test | Qwen3-0.6B (`enable_thinking=False`) | Apache-2.0; 1.50 GB; same architecture family, vocab and template branch as 1.7B | Quality too low for accuracy claims; use for correctness tests only |
| Experimental | Qwen3.5-2B or 4B | Apache-2.0; much smaller K/V | Hybrid linear attention, see section 3 |
| Not a default | Qwen2.5-3B | Research License, non-commercial | Confirmed in section 1 |

Qwen3-4B (hybrid) should not be the default: same size as 2507 but needs the flag and follows instructions less well.

## 5. VRAM on a 12 GB card (bf16, 2k-token prompt, 64 candidates)

Formula (full-attention layers only, K and V, bf16): bytes per token = 2 x full_layers x kv_heads x head_dim x 2. Inputs are VERIFIED from the config tables; the arithmetic is mine.

| Model | KV bytes/token | KV for 2,048 tokens, 1 copy | x64 fully materialized | Weights (bf16) |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | 114,688 (112 KiB) | 224 MiB | 14.0 GiB | 1.40 GiB |
| Qwen3-1.7B | 114,688 (112 KiB) | 224 MiB | 14.0 GiB | 3.78 GiB |
| Qwen3-4B and 4B-Instruct-2507 | 147,456 (144 KiB) | 288 MiB | 18.0 GiB | 7.49 GiB |
| Qwen3.5-4B (8 full layers) | 32,768 (32 KiB) | 64 MiB | 4.0 GiB plus linear state per copy (not computed, UNVERIFIED) | 8.68 GiB (includes vision tower) |
| Qwen3.5-2B (6 full layers) | 12,288 (12 KiB) | 24 MiB | 1.5 GiB plus linear state | 4.24 GiB |

Findings:

- CONTRADICTS a naive reading of the brief: repeating the cache across all 64 candidates at once does NOT fit for the default. 64 copies of a 2k prefix on Qwen3-4B is 18 GiB of KV alone, plus 7.5 GiB of weights, on a 12 GB card. Even Qwen3-1.7B (14 GiB KV) does not fit as one batch.
- `DynamicLayer.batch_repeat_interleave` physically copies (`repeat_interleave(repeats, dim=0)`, cache_utils.py, around line 200), so there is no sharing. Chunking is mandatory; Phase 3 of the brief already says "in chunks".
- Budget for the default: 12 GB minus about 1 GB for display and CUDA context minus 7.49 GiB weights leaves roughly 3 GiB. Prefix KV is 0.28 GiB. A chunk of 8 copies is 2.25 GiB of KV, a chunk of 4 is 1.1 GiB. Plan for chunk size 4 to 8 at 2k context on the 4B default, 16 to 24 on 1.7B, scaling inversely with prompt length. Compute chunk size from `torch.cuda.mem_get_info` and context length rather than hardcoding.
- Do not materialize full logits: candidates x suffix_len x vocab x 2 bytes. At 64 x 10 x 151,936 that is about 195 MB in bf16, double in fp32. Gather hidden states at label positions and apply the lm_head to those only.
- Tied embeddings on every model mean the lm_head adds no weight memory.
- Better than copying (packed suffixes with a custom mask, or shared prefix views) is deferred per the brief.

## 6. What this means for MirethSTM1

SPEC:

1. Model table in README and SPEC: default `Qwen/Qwen3-4B-Instruct-2507`, fast `Qwen/Qwen3-1.7B`, test `Qwen/Qwen3-0.6B`, all Apache-2.0 (VERIFIED). Add a warning row: Qwen2.5-3B is under the Qwen Research License (non-commercial) and is unsupported by default.
2. Chat template handling is a first-class item: call `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` always; the 2507 template ignores the kwarg. Test that the 0.6B and 1.7B prompts end with `<think>\n\n</think>\n\n` and the 2507 prompt ends with `assistant\n`.
3. Record model id and revision SHA in the JSONL `model` field so ECE numbers are reproducible. Revisions seen today: 0.6B `c1899de2`, 1.7B `70d244cc`, 4B `1cfa9a72`, 4B-Instruct-2507 `cdbee75f`.
4. Scorer: per-chunk copy of a prefilled `DynamicCache`; chunk size derived from free VRAM and prompt length; expose `max_candidates_per_chunk`. State in the SPEC that a 2k prompt with 64 candidates needs chunking on 12 GB.
5. Read vocab size and tokenizer from the model; Qwen3.5 uses 248,320 vs 151,936.
6. Keep a `prefill/expand/score` interface so Qwen3.5 can be added later. Document that hybrid models need full state copies (recurrent state cannot be cropped) and may need kernels that are doubtful on native Windows.
7. Label scoring must start right after the generation prompt, which for hybrid models includes the empty think block.

Scaffold:

- CPU tests on Qwen3-0.6B (float32 on CPU, roughly 3 GB RAM): first-token collision ("Sports" vs "Sci-Tech"), chunked vs unchunked equivalence, cached vs full-forward log-prob equivalence within tolerance, template tail check.
- One small optional experiment on Qwen3.5-0.8B: cache deep-copy isolation, `batch_repeat_interleave` on the hybrid cache, padded batch equivalence. If any fails, mark Qwen3.5 unsupported for v0.1.
- First GPU test on Windows: load the 4B-Instruct-2507 in bf16, run the collision test, log `torch.cuda.max_memory_allocated()` at chunk sizes 4, 8, 16, and record where it runs out of memory.

Open items (UNVERIFIED): `batch_repeat_interleave` on hybrid linear-state layers; Qwen3.5 speed without fla and causal-conv1d kernels on Windows; the text of the qwen.ai Qwen2.5 blog (page unreadable, LICENSE file used instead).
