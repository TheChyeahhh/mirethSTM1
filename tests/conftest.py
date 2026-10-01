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

    SPEC 3.4: in fp32, where the method is checked, 2e-4 on the CPU and 2e-3 on the GPU. On the
    GPU the kernels alone move a label by up to 1.1e-3 (Qwen3-1.7B, 2,058-token router prompt,
    only the SDPA kernel changed), and every GPU path, the uncached reference included, is 1.2e-3
    to 1.3e-3 off CPU fp64, which CPU fp32 matches to 2.2e-5. bf16 can only show that two paths
    stay within its noise, which depends on the model and on which GPU kernels each path happens
    to get (RTX 5070): Qwen3-0.6B paths differ by up to 1.7 (each path alone up to 1.8 off its
    fp32 result), Qwen2.5-1.5B-Instruct by up to 0.6. Closest to the band on 2026-10-01: Qwen3-1.7B
    router 2.0, Qwen3-4B-Instruct-2507 1.25 (tree) and 1.86 (router), Qwen3-0.6B tree 1.30; check
    again when a model is approved. The bf16 band catches gross errors (the output head one node
    late: about 450 off), not subtle ones (positions off by one: 0.5 to 2.2 off), which only the
    fp32 run sees.
    """
    if engine.model.dtype == torch.float32:
        return 2e-4 if engine.model.device.type == "cpu" else 2e-3
    return 2.5


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
