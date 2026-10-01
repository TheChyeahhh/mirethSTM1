# 05. Environment facts: Windows + NVIDIA, and the Transformers KV-cache API

Date checked: 2026-09-30. Status tags: VERIFIED (read in code/API output or run), UNVERIFIED (secondary source, or could not access), CONTRADICTED (brief says X, reality is Y).

Test stack: Python 3.11, torch 2.11.0+cu128, transformers 5.18.0 (installed wheel source read directly), huggingface_hub 1.33.0, CPU-only runs on Qwen/Qwen3-0.6B.

## 1. Headline findings

| # | Finding | Status |
|---|---|---|
| 1 | torch 2.11 is NOT the current stable. Latest is 2.14.1 (tagged 2026-09-30). | CONTRADICTED (brief and venv assume 2.11 is current) |
| 2 | The cu128 index stops at torch 2.11.0. CUDA 12.8 was deprecated and dropped from the binary matrix starting with torch 2.12. | VERIFIED |
| 3 | cu128 wheels for 2.11 include sm_120 (arch list below), and cp311/cp312 Windows wheels exist. The existing venv is valid for an RTX 5070. | VERIFIED |
| 4 | Better choice for a fresh install: cu130 (torch 2.14.1). The cu129 index has nothing newer than torch 2.9.0. | VERIFIED (index listing) |
| 5 | The KV-cache reuse trick works in transformers 5.18 with plain DynamicCache. Measured max abs difference vs uncached scoring: 9.8e-06 (fp32, CPU). | VERIFIED (ran it) |
| 6 | Forward calls mutate the passed cache object (layers rebind keys/values via torch.cat). Tensors are never written in place, so a snapshot of tensor references is a safe, cheap "copy". | VERIFIED (source + run) |
| 7 | On Windows, PyTorch SDPA has no flash backend. Qwen3 is GQA and we pass a mask, so the mem-efficient backend should be what runs. cuDNN SDPA on Windows is unknown. | Mostly UNVERIFIED (section 3) |
| 8 | Triton on native Windows exists only as a community fork (triton-windows), whose repo was archived 2026-02-18. torch.compile is not on our critical path. | VERIFIED (repo page) |
| 9 | bf16 on this Ryzen 5900X CPU is about 500x slower than fp32 on a 1024x1024 matmul. Run CPU tests in fp32. | VERIFIED (measured) |
| 10 | Qwen3-0.6B, 1.7B, 4B and 4B-Instruct-2507 are all tagged apache-2.0 on the Hub API. | VERIFIED |

## 2. (a) torch / CUDA wheels

### 2.1 Current stable torch

GitHub releases API (`gh api repos/pytorch/pytorch/releases`), newest first:

| Tag | Published |
|---|---|
| v2.14.1 | 2026-09-30 |
| v2.14.0 | 2026-09-02 |
| v2.13.0 | 2026-07-08 |
| v2.12.1 | 2026-06-18 |
| v2.12.0 | 2026-05-13 |
| v2.11.0 | 2026-03-23 |

PyPI `torch` latest = 2.14.1 (https://pypi.org/pypi/torch/json). VERIFIED. 2.14.1 is hours old at the check date, so a prudent pin is 2.14.0, or simply "the same torch on both sides of a benchmark".

### 2.2 What each CUDA index hosts (Windows cp311, listing of `https://download.pytorch.org/whl/<cuXXX>/torch/`)

| Index | Newest cp311 win_amd64 wheel | Notes |
|---|---|---|
| cu126 | 2.14.1 | still built (older drivers) |
| cu128 | 2.11.0 (then 2.10.0, 2.9.1 ...) | cp310 to cp314 all present for 2.11.0, including cp311 and cp312. No 2.12 or later |
| cu129 | 2.9.0 | dead end |
| cu130 | 2.14.1 (2.14.0, 2.13.0 ...) | current default CUDA |
| cu131 | empty | not a torch index |
| cu132 | 2.14.1 | "experimental" CUDA 13.2 |

All VERIFIED by direct listing. Example names: `torch-2.11.0+cu128-cp311-cp311-win_amd64.whl`, `torch-2.11.0+cu128-cp312-cp312-win_amd64.whl`.

Release-matrix sources:
- CUDA 12.8 deprecation for 2.12: https://dev-discuss.pytorch.org/t/introducing-cuda-13-2-and-deprecating-cuda-12-8-release-2-12/3337 (VERIFIED quote: "CUDA 12.8 is being deprecated and removed from CI/CD pipelines and binary build matrices", scheduled for the week of 2026-04-06).
- 2.14 release blog https://pytorch.org/blog/pytorch-2-14-release-blog/ and https://github.com/pytorch/pytorch/blob/main/RELEASE.md. From search summaries: 2.14 supports CUDA 12.6, 13.0 and 13.2 on Windows, 13.0 is the default wheel, 13.4 is excluded from Windows. UNVERIFIED (secondary summary, but consistent with the index listing above).

### 2.3 sm_120 (Blackwell, RTX 50xx) in cu128 builds

Run in the venv, CPU only, no GPU touched:

```
torch 2.11.0+cu128, torch.version.cuda = 12.8
torch.cuda.get_arch_list() = ['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']
```

VERIFIED: sm_120 is compiled in. The target GPU reports `NVIDIA GeForce RTX 5070`, driver 591.86 (nvidia-smi, 2026-09-30).

Issue https://github.com/pytorch/pytorch/issues/164342 ("Official support for sm_120 (RTX 50-series / Blackwell) in stable PyTorch builds"): VERIFIED it exists, opened 2025-10-01, still OPEN on 2026-09-30. It is a stale request thread with mixed comments (some link third-party builds: treat as untrusted, do not use). It is NOT evidence that cu128 stable lacks sm_120: the arch list above shows it has it. The brief's use of this issue as the Blackwell reference is slightly misleading: the answer is in the wheel, not in the issue.

Newer builds: the PyTorch support matrix says CUDA 13.0 and 13.2 builds cover Blackwell (10.0, 12.0 + PTX) on Linux x86 and Windows (dev-discuss post above plus issue https://github.com/pytorch/pytorch/issues/178665). UNVERIFIED for the exact 2.14.1 cu130 arch list (we do not install into the venv). First check after any install: `torch.cuda.get_arch_list()` must contain `sm_120`.

### 2.4 Minimum NVIDIA driver (Windows)

CUDA Toolkit release notes, https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html (VERIFIED):

| CUDA toolkit | Min Windows driver |
|---|---|
| 12.8 GA | >= 570.65 |
| 12.8 Update 1 | >= 572.61 |
| 12.9 GA | >= 576.02 |
| 12.9 Update 1 | >= 576.57 |

The brief's "Driver 570+" is right for 12.8 GA; 572.61+ is the safer floor since PyTorch builds against a later 12.8.x. For CUDA 13.x, the same page says existing 13.x applications run on drivers >= 580 via minor-version compatibility. The exact floor for the torch cu130 wheel was not separately confirmed: UNVERIFIED, but 591.86 is above it.

### 2.5 Would cu129 or cu130 be better for an RTX 5070?

- cu129: no. The index ends at torch 2.9.0.
- cu130: yes for a supported current torch. Same Blackwell coverage, still receives fixes, is the official default CUDA. Cost: a different torch than the validated venv, so re-run the test suite.
- cu128 / 2.11.0: works now, frozen (no more patch releases). Fine for a one-week ship. Recommendation: keep the existing venv for Sunday, document cu130 as the forward path, pin one index in the README.

## 3. (c) SDPA on Windows

| Backend | Windows CUDA status | Status |
|---|---|---|
| math | always available (materializes the full attention matrix) | VERIFIED (by design) |
| mem-efficient | available | UNVERIFIED at runtime here (no GPU); implied by the PR text below |
| flash (native SDPA flash) | NOT built on Windows: flag hard-disabled for MSVC | VERIFIED as stated by a PR description; not runtime-tested |
| cuDNN fused attention | unknown on Windows | UNVERIFIED |

Source: https://github.com/pytorch/pytorch/pull/186343 ("Enable FA build for SDPA on Windows+CUDA"), state closed and not merged (checked with `gh api`, merged=false). Description: "Today, the flag is hard-disabled on MSVC, so every Windows CUDA build ships without the flash SDPA backend and silently falls back to slower, memory-heavy backends." It also says that for GQA models the memory-efficient backend is not eligible (when GQA is requested natively) so SDPA falls to math. This is a PR author's claim, not maintainer docs. The "Torch was not compiled with flash attention" warning is also widely reported on Windows.

`torch.backends.cuda.flash_sdp_enabled()` returned True in the venv, but that only reports the user toggle, not build availability. Do not use it to conclude flash works.

What transformers 5.18 does (`integrations/sdpa_attention.py`, `use_gqa_in_sdpa` lines 29-39, use at lines 98-102): on CUDA it passes `enable_gqa=True` only when `attention_mask is None` and head_dim <= 256; otherwise it `repeat_kv`s K/V to the full head count. VERIFIED. Our cached-continuation path always passes a mask (right padding), so:
- K/V are expanded by the GQA factor before SDPA (Qwen3-0.6B: 16 query heads, 8 KV heads, head_dim 128, 28 layers, read from its config.json). A small cost for short continuations.
- With a mask and no native GQA flag, the mem-efficient backend is eligible, so we should not land on the math path. Needs a GPU check (section 7).

Mask and cache caveats (VERIFIED in `masking_utils.py` and `cache_utils.py`):
- A 2D `attention_mask` must cover past + new tokens: shape `(batch, past_len + new_len)`. Kv length is derived as `cache.get_seq_length() + query_len` (`DynamicLayer.get_mask_sizes`, cache_utils.py lines 150-154).
- A 4D mask `(batch, 1, q_len, kv_len)` is returned as-is (masking_utils.py line 816). If we later pass 4D (tree or packed masks, deferred in the brief) it must already be the final mask and include the cache prefix.
- With a mask present, `is_causal` is off in SDPA (sdpa_attention.py line 124), so causality comes from the mask. Right padding is safe because pad tokens sit at the end and their outputs are discarded.

## 4. (b) transformers 5.18 KV-cache API (installed source)

Line numbers refer to `transformers/cache_utils.py` of 5.18.0.

### 4.1 Layout

- `DynamicCache(Cache)` (line 1764) holds `cache.layers`, a list of `DynamicLayer` (line 114), one per decoder layer (28 for Qwen3-0.6B, confirmed by running).
- Each layer stores `.keys` and `.values` shaped `[batch, num_kv_heads, seq_len, head_dim]`. Observed after prefill of a 21-token prompt: `torch.Size([1, 8, 21, 128])`.
- Construct with `DynamicCache(config=model.config)` (correct sliding/hybrid layers) or from tensors with `DynamicCache(ddp_cache_data=[(k, v), ...])` (lines 1807-1849).
- Iterating yields `(keys, values, sliding_window_tensor_or_None)` per layer (line 1851). Old 2-tuple unpacking breaks: use `cache.layers[i].keys`.

### 4.2 Methods (all VERIFIED present)

| Need | API | Where |
|---|---|---|
| Repeat batch N times | `cache.batch_repeat_interleave(N)` (in place) | Cache line 1641, layer line 200 |
| Select batch rows | `cache.batch_select_indices(idx)` | lines 1646, 206 |
| Drop last k tokens | `cache.crop(-k)` (negative removes k; positive absolute form is deprecated and warns) | lines 1631, 175-198 |
| Sequence length | `cache.get_seq_length()` | line 1514 |
| Copy | No `copy()` method on Cache; `copy.deepcopy` would copy all layer tensors | method list of the file |

### 4.3 Does forward mutate the cache? Yes, by rebinding

`DynamicLayer.update` (lines 129-148): `self.keys = torch.cat([self.keys, key_states], dim=-2)`. A forward with a cache object grows that object's layers, but old tensors are never written in place.

Run output: prefill gives cache length 21; a second forward of a 1-token continuation on the same object gives 22; `cache.crop(-1)` returns it to 21.

Consequences:
1. Two forwards on one cache object do not give independent results. Either crop back, or use a new cache object per chunk.
2. Cheapest copy: after prefill, keep the list of `(layer.keys, layer.values)` references (no data copy). Per chunk build `DynamicCache(ddp_cache_data=[(k.repeat_interleave(N,0), v.repeat_interleave(N,0)) ...])`. `repeat_interleave` allocates new tensors, so the base stays batch-1 (verified: still `[1, 8, 21, 128]` after all chunks). Memory per chunk is N x prefix KV, so chunking bounds it.
3. `batch_repeat_interleave` on the live cache also works but destroys the batch-1 prefill, forcing `batch_select_indices` or a rebuild when the next chunk has a different N. The snapshot approach avoids that.

### 4.4 Passing inputs for a right-padded batch continuing from a cache

- `input_ids`: `(N, L)` continuation tokens, right-padded to the chunk maximum. Pad id is irrelevant.
- `attention_mask`: `(N, P + L)` = ones for the prefix, then 1/0 for real/pad suffix tokens. It must include the prefix.
- `past_key_values`: the repeated cache.
- `position_ids` and `cache_position`: not needed. The model derives `cache_position = arange(past_len, past_len + L)` from the cache, so real tokens get positions P..P+len-1, correct for right padding. (Left padding would need explicit position_ids; we avoid it.)
- Hidden states without full-vocab logits: call `model.model(...)` and read `.last_hidden_state` of shape `(N, L, hidden)`, then apply `model.lm_head` only on the positions needed. `logits_to_keep` also exists on `Qwen3ForCausalLM.forward` (modeling_qwen3.py line 457, slice at line 489): an int keeps the last k positions, a tensor keeps given sequence indices shared by all rows. With ragged lengths the manual `model.model` + `lm_head` route is cleaner. Verified `logits_to_keep=1` returns shape `(1, 1, 151936)`.
- The first continuation token is predicted by the LAST prompt position, so the prefill must also keep `lm_head(last_hidden_state[:, -1])`. Position j of a continuation predicts token j+1, so only positions 0..len-2 are needed from the batch pass.
- Qwen3 vocab is 151,936; skipping full logits saves real memory (a 4k prompt would otherwise produce 4096 x 151936 floats).

### 4.5 Verified snippet and result

Setup: 21-token prompt, 3 multi-token labels of 1, 3 and 3 tokens, chunk size 2 (so chunks of N=2 then N=1, exercising right padding and a ragged last chunk), compared with scoring each full sequence from scratch without a cache. fp32, CPU.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

name = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(
    name, dtype=torch.float32, attn_implementation="sdpa").eval()

@torch.inference_mode()
def cached_scores(prompt_ids, cands, chunk=2):
    P = prompt_ids.shape[1]
    cache = DynamicCache(config=model.config)
    out = model.model(input_ids=prompt_ids, past_key_values=cache, use_cache=True)
    first_lp = model.lm_head(out.last_hidden_state[:, -1]).float().log_softmax(-1)
    base = [(l.keys, l.values) for l in cache.layers]   # references, not copies
    res = []
    for s in range(0, len(cands), chunk):
        cs = cands[s:s + chunk]; N = len(cs); L = max(map(len, cs))
        ids = torch.full((N, L), tok.pad_token_id)
        m = torch.zeros(N, L, dtype=torch.long)
        for i, c in enumerate(cs):                       # RIGHT padded
            ids[i, :len(c)] = torch.tensor(c); m[i, :len(c)] = 1
        kv = DynamicCache(ddp_cache_data=[
            (k.repeat_interleave(N, 0), v.repeat_interleave(N, 0)) for k, v in base])
        am = torch.cat([torch.ones(N, P, dtype=torch.long), m], 1)
        h = model.model(input_ids=ids, attention_mask=am,
                        past_key_values=kv, use_cache=True).last_hidden_state
        for i, c in enumerate(cs):
            score = first_lp[0, c[0]].item()
            if len(c) > 1:
                lg = model.lm_head(h[i, :len(c) - 1]).float().log_softmax(-1)
                score += sum(lg[j, c[j + 1]].item() for j in range(len(c) - 1))
            res.append(score)
    return res
```

Measured (CPU, fp32, Qwen3-0.6B, transformers 5.18.0, torch 2.11.0):

```
cont lens [1, 3, 3]
' Sports'                 cached=-11.024235 ref=-11.024236 diff=9.54e-07
' Science and Technology' cached=-8.462020  ref=-8.462010  diff=9.78e-06
' World politics news'    cached=-27.381251 ref=-27.381253 diff=1.91e-06
max abs diff 9.775161743164062e-06
```

VERIFIED: max abs difference 9.8e-06 in summed log-prob, which is float32 round-off. The cached, right-padded, chunked path equals full-sequence scoring, including a 1-token label batched next to 3-token labels.

Not tested: bf16/fp16 on GPU (expect differences near 1e-2 in log-prob from the dtype, not from the method), and long prefixes that might switch SDPA kernels.

Tokenization note: labels were tokenized separately with `add_special_tokens=False` and appended as token ids. That equals tokenizing the joined string only when the boundary is stable (labels starting with a space or newline token). The scorer must tokenize prompt and label separately and score exactly those ids; the SPEC should fix an answer-prefix format so every label starts at a clean token boundary.

## 5. (d) Windows gotchas

| Topic | Fact | Status |
|---|---|---|
| HF cache symlinks | huggingface_hub probes symlink support in the cache dir (`are_symlinks_supported`, huggingface_hub/file_download.py line 92) and falls back to file copies with a warning when unsupported. Windows needs Developer Mode or admin for symlinks. Consequence: possible duplicate disk use per revision. Silence with `HF_HUB_DISABLE_SYMLINKS_WARNING=1`; enable Developer Mode for real symlinks. | VERIFIED (source); duplication is inference |
| Long paths | Snapshot paths (`models--Qwen--Qwen3-4B-Instruct-2507/snapshots/<40-char hash>/...`) can approach MAX_PATH (260) when the cache root is deep. Keep `HF_HOME` short (for example `D:\hf`) or enable the `LongPathsEnabled` registry setting. The registry state was not read on the target box. | UNVERIFIED |
| torch.compile / Triton | Official Triton ships no Windows wheels. Community fork `triton-windows` (PyPI latest 3.8.0.post29, Python 3.10 to 3.14). Its README (https://github.com/woct0rdho/triton-windows) lists support for PyTorch 2.4 to 2.10 (Triton 3.1 to 3.6), needs CUDA >= 12.8 for sm_120, and says the repo was archived read-only on 2026-02-18 with development moving to an official-org Windows repo. No stated coverage of torch 2.11 to 2.14. Decision: no torch.compile and no Triton dependency; our method needs neither. | VERIFIED (README, PyPI) |
| bf16 on CPU | Measured in the venv: 1024x1024 matmul, 20 iterations averaged: fp32 3.3 ms, bf16 1749 ms. Zen 3 (Ryzen 9 5900X) has no native bf16 math, so torch takes a slow path. One short measurement, but the gap is far beyond noise. CPU tests must load the model in fp32; use bf16 or fp16 on GPU. | VERIFIED (measured) |
| dtype arg | In transformers 5.x the loader argument is `dtype=` (used without any error). | VERIFIED |
| Offline test runs | Set `HF_HUB_OFFLINE=1` when the model is cached, to avoid network stalls. | VERIFIED (used) |
| Progress bars | "Loading weights" bars go to stderr and pollute captured output. Set `HF_HUB_DISABLE_PROGRESS_BARS=1` in tests. | Observed |

## 6. Model licenses (Hugging Face API, `cardData.license`)

| Model | License tag | Likes / downloads (2026-09-30) |
|---|---|---|
| Qwen/Qwen3-0.6B | apache-2.0 | 1708 / 29.6M |
| Qwen/Qwen3-1.7B | apache-2.0 | 559 / 3.2M |
| Qwen/Qwen3-4B | apache-2.0 | 717 / 5.4M |
| Qwen/Qwen3-4B-Instruct-2507 | apache-2.0 | 984 / 3.9M |

VERIFIED via `https://huggingface.co/api/models/<id>`. The README table should link each model card, and state we do not bundle weights.

## 7. What this means for MirethSTM1

For the SPEC:
1. Pin the torch story: "tested on torch 2.11.0+cu128 (sm_120 confirmed in the arch list); forward path is cu130 (torch 2.14.x). cu129 is a dead index; cu128 receives no torch after 2.11." Correct the brief's "current stable" assumption.
2. Driver floor: 572.61+ for cu128, 580+ for cu130 (the latter UNVERIFIED for torch specifically). Target box has 591.86.
3. Scorer contract, frozen by the verified snippet: prefill once via `model.model`; keep `lm_head` of the last position as the first-token distribution; snapshot `layer.keys/values`; per chunk build a new `DynamicCache(ddp_cache_data=...)` with `repeat_interleave(N)`; right pad; full-length 2D mask (prefix + suffix); never pass `position_ids`; run `lm_head` only on the `len-1` gathered positions. Acceptance test: max abs diff vs uncached scoring under 1e-4 in fp32 (measured 9.8e-06).
4. Do not rely on tuple iteration of caches or legacy tuple caches; use `cache.layers[i].keys/values`. Add a transformers version guard, since the 5.x cache API changed (`crop` with positive values is deprecated).
5. Fix the answer-prefix format in the prompt so every label begins at a stable token boundary, and tokenize prompt and label separately.
6. No torch.compile, no Triton, no flash-attn. Attention stays `sdpa`.
7. Default CPU test dtype fp32; GPU dtype bf16.

For the scaffold:
1. `scorer.py`: implement section 4.5 with `chunk_size` as a parameter (bounds memory at N x prefix KV), plus `device` and `dtype` options.
2. Tests: `test_cached_equals_uncached` (Qwen3-0.6B, CPU, fp32, labels of length 1/3/3, tolerance 1e-4) plus the first-token-collision test.
3. Test fixtures set `HF_HUB_OFFLINE=1` when the model is cached and skip (not fail) if it is absent; also set `HF_HUB_DISABLE_PROGRESS_BARS=1`.
4. A `mirethstm doctor` command printing torch version, `torch.version.cuda`, `get_arch_list()` (assert `sm_120` or the device's capability is present), driver version, transformers version, and whether the HF cache dir supports symlinks.
5. README environment section: cu128-or-cu130 choice, short `HF_HOME`, `HF_HUB_DISABLE_SYMLINKS_WARNING`, Developer Mode note.

Open items for the GPU session:
- Confirm `torch.cuda.get_arch_list()` and a real CUDA forward on the RTX 5070 (sm_120 kernels actually launch).
- Re-run the cached-vs-uncached equality test on GPU in bf16 and fp16 and record the max abs log-prob difference.
- Microbenchmark which SDPA backend runs with a mask on Windows (`torch.nn.attention.sdpa_kernel`, try each backend and catch errors) to settle the cuDNN and mem-efficient questions marked UNVERIFIED.
- Decide whether to move the venv to cu130 after the Sunday ship.
