import os

# Tests run from the local Hugging Face cache only; set before transformers is imported.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import pytest  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import try_to_load_from_cache  # noqa: E402

from mirethstm import Engine  # noqa: E402

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = os.environ.get("MIRETHSTM_TEST_DEVICE", "cpu")
DTYPE = torch.float32 if DEVICE == "cpu" else torch.bfloat16


def _require_cached(filename):
    if not isinstance(try_to_load_from_cache(MODEL_ID, filename), str):
        pytest.skip(f"{MODEL_ID} is not in the local Hugging Face cache")


@pytest.fixture(scope="session")
def tokenizer():
    _require_cached("tokenizer.json")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID)


@pytest.fixture(scope="session")
def engine():
    """Qwen3-0.6B, loaded once. chunk_size=2 so most calls have ragged chunks."""
    _require_cached("model.safetensors")
    return Engine.load(MODEL_ID, device=DEVICE, dtype=DTYPE, chunk_size=2, temperature=1.0)
