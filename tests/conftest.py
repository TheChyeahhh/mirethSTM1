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
DTYPE = torch.float32 if DEVICE == "cpu" else torch.bfloat16


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
