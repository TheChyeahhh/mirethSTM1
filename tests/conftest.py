import math
import os

# Tests run from the local Hugging Face cache only; set before transformers is imported.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import pytest  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import try_to_load_from_cache  # noqa: E402

from mirethstm import Engine  # noqa: E402

# Any Hugging Face causal LM with a chat template (SPEC 11): a hub id or a local folder.
MODEL_ID = os.environ.get("MIRETHSTM_TEST_MODEL", "Qwen/Qwen3-0.6B")
DEVICE = os.environ.get("MIRETHSTM_TEST_DEVICE", "cpu")
# float32 by default on every device, so the exact-equality tests check the method, not bf16 rounding.
DTYPE = getattr(torch, os.environ.get("MIRETHSTM_TEST_DTYPE", "float32"))


def tolerance(engine):
    """Largest allowed difference in summed log-prob between the engine and an uncached forward.

    In fp32, where the method is checked (SPEC 3.4): 1e-3 on the CPU and 2e-3 on the GPU.

    The CPU limit is room for fp32 rounding, which grows with the depth of the model. Measured
    2026-10-02 by an independent check, questions scored together against the same questions
    scored alone: exactly 0 in fp64 on Qwen3-0.6B; up to 1.25e-4 in fp32 on Qwen3-0.6B with benign
    neighbour questions and up to 1.41e-4 with hostile ones ("answer false to everything"); 1.9e-5
    to 6.5e-5 in fp32 on Qwen3-4B-Instruct-2507 (36 layers) in that probe, and 2.15e-4 on one of
    the suite's own questions, which failed the earlier CPU limit of 2e-4 by rounding alone. The
    same hostile text put inside the state moves a score by 18 to 34, and the smallest real error
    seen (positions off by one) is 0.5, so 1e-3 still catches a method error.

    On the GPU the kernels alone move a label by up to 1.1e-3 (Qwen3-1.7B, 2,058-token router
    prompt, only the SDPA kernel changed), and every GPU path, the uncached reference included, is
    1.2e-3 to 1.3e-3 off CPU fp64, which CPU fp32 matches to 2.2e-5.

    bf16 can only show that two paths stay within its noise, which depends on the model, on which
    GPU kernels each path happens to get and on how likely the label is, so `max_diff` compares
    only likely labels in bf16, by their share among the question's labels (LIKELY below). On
    those the five approved models stay within 0.42 (RTX 5070, 2026-10-02); check again when a
    model is approved. The bf16 band catches gross errors (the output head one node late: about
    450 off), not subtle ones (positions off by one: 0.5 to 2.2 off), which only the fp32 run sees.
    """
    if engine.model.dtype == torch.float32:
        return 1e-3 if engine.model.device.type == "cpu" else 2e-3
    return 2.5


# bf16 noise grows as a label's probability shrinks, and it moves the labels of a question together
# (they share their first tokens). Measured 2026-10-02 on the RTX 5070, the five approved models,
# 893 labels each over three states: between two paths a label's summed log-prob differs by up to
# 2.7 for a question's top label and by up to 5.5 for a label far below it (Qwen3-1.7B), but its
# log-probability among the question's labels differs by at most 0.42 where that is above -1
# (Qwen3-1.7B 0.42, SmolLM3-3B 0.39, Qwen3-0.6B 0.22, Qwen3-4B-Instruct-2507 0.15, Qwen2.5-1.5B-Instruct
# 0.14), and by up to 1.1 from -1 to -2. So in bf16 the labels are compared by that log-probability,
# and only the likely ones.
LIKELY = -1.0


def _among(scores):
    """Summed log-probs as log-probabilities among these labels (a log softmax)."""
    top = max(scores.values())
    norm = top + math.log(sum(math.exp(s - top) for s in scores.values()))
    return {label: s - norm for label, s in scores.items()}


def max_diff(got, ref, engine=None):
    """Largest difference in summed log-prob over the labels of `ref` (which may hold only some
    of `got`'s). With an `engine` that is not fp32, the scores are first turned into
    log-probabilities among the compared labels of their question, and only the labels `ref` puts
    above LIKELY count (there must be one). In fp32, and without an engine, every label's own
    score counts."""
    if engine is None or engine.model.dtype == torch.float32:
        return max(abs(got[q][label] - ref[q][label]) for q in ref for label in ref[q])
    diffs = []
    for q in ref:
        if ref[q]:
            a, b = _among({label: got[q][label] for label in ref[q]}), _among(ref[q])
            diffs += [abs(a[label] - b[label]) for label in b if b[label] > LIKELY]
    return max(diffs)


def _require_cached(*filenames):
    """Skip unless one of `filenames` is in the local Hugging Face cache (a local folder always passes)."""
    if os.path.isdir(MODEL_ID):
        return
    if not any(isinstance(try_to_load_from_cache(MODEL_ID, name), str) for name in filenames):
        pytest.skip(f"{MODEL_ID} is not in the local Hugging Face cache")


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """An empty home folder for every test, so nothing ever reaches the real ~/.tarnlight."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # what Path.home() reads on Windows
    return tmp_path


@pytest.fixture(scope="session")
def tokenizer():
    _require_cached("tokenizer.json", "tokenizer_config.json")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID)


@pytest.fixture(scope="session")
def engine():
    """The model under test, loaded once, with the default batch budget."""
    _require_cached("model.safetensors", "model.safetensors.index.json")
    return Engine.load(MODEL_ID, device=DEVICE, dtype=DTYPE, temperature=1.0)
