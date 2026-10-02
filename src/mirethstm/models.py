"""The approved models and the models evaluated but not approved (SPEC 11.1).

The console's model picker and GET /v1/models show only approved models. `Engine.load` and
`--model` still take any Hugging Face id (unvetted, at the user's own risk).

A model is approved when its license allows free commercial use (Apache-2.0 or MIT, public and
not gated), it fits a 12 GB card with room for the cache, `Engine.load` accepts it and the full
model test suite passes on the GPU, and its speed, accuracy and calibration are measured and
written here. A rejected model failed one of these and carries a one-line reason. PENDING marks
a model the console offers by name that is not approved yet (none at the moment).

The measured numbers come from the benchmark run in docs/benchmark.md (bench/out/v2,
2026-10-02, one RTX 5070, bf16); its Summary section holds every one of them. Facts below
come from the Hugging Face API (`https://huggingface.co/api/models/<id>`), checked
2026-09-30: `license` from its license tag,
`params` from its safetensors total, `release_date` from its `createdAt` (the day the weights
repo was created, a few days before most public announcements).
"""

from dataclasses import asdict, dataclass

APPROVED = "approved"
PENDING = "pending"
CANDIDATE = "candidate"
REJECTED = "rejected"


@dataclass(frozen=True)
class Model:
    id: str  # Hugging Face hub id
    name: str  # display name
    license: str
    params: float  # billions, every weight counted
    release_date: str  # YYYY-MM-DD
    role: str = ""  # SPEC 11.2
    status: str = CANDIDATE
    # Filled from the benchmark (docs/benchmark.md); None until measured.
    ms_28_fields: float | None = None  # p50 warm wall time of the 28-field support scenario (bench.latency)
    ms_router: float | None = None  # same, the 255-option router scenario
    accuracy: float | None = None  # mean MirethSTM1 accuracy over the five benchmark datasets, 0 to 1
    ece: float | None = None  # mean 15-bin ECE over the five datasets at `temperature`
    temperature: float | None = None  # pooled T fitted on the calibration splits (the shipped default)
    measured_on: str | None = None  # GPU, dtype and date of the measurement
    reason: str = ""  # why a rejected model is not approved, one line


MEASURED = "RTX 5070, bf16, 2026-10-02"

# Approved models first, in the picker's order (the console default leads), then the rejected ones.
# Fast: Qwen2.5-1.5B-Instruct is quicker than Qwen3-1.7B in every latency setting (28 fields 96 against
# 111 ms, 1 to 10 fields 35 to 37 against 43 to 45 ms) at the same accuracy on the 500 rows per dataset
# both ran (0.729 against 0.726; difference +0.002, 95% interval -0.013 to +0.018).
MODELS = (
    Model("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5 1.5B Instruct", "Apache-2.0", 1.54, "2024-09-17",
          "Fast", APPROVED, 96.4, 209.1, 0.728, 0.109, 2.112, MEASURED),
    Model("Qwen/Qwen3-1.7B", "Qwen3 1.7B", "Apache-2.0", 2.03, "2025-04-27", "Alternative", APPROVED,
          110.6, 248.4, 0.726, 0.097, 7.007, MEASURED),
    Model("Qwen/Qwen3-4B-Instruct-2507", "Qwen3 4B Instruct 2507", "Apache-2.0", 4.02, "2025-08-05",
          "Default SDK/CLI", APPROVED, 271.0, 614.9, 0.760, 0.117, 7.930, MEASURED),
    Model("Qwen/Qwen3-0.6B", "Qwen3 0.6B", "Apache-2.0", 0.75, "2025-04-27", "CPU tests", APPROVED,
          63.7, 155.7, 0.575, 0.137, 4.639, MEASURED),
    Model("HuggingFaceTB/SmolLM3-3B", "SmolLM3 3B", "Apache-2.0", 3.08, "2025-07-08", "Alternative", APPROVED,
          213.4, 408.4, 0.688, 0.061, 2.702, MEASURED),
    Model("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5 0.5B Instruct", "Apache-2.0", 0.49, "2024-09-16",
          status=REJECTED, reason="failed the bf16 GPU suite on 2026-10-01: scores 3.95 from the uncached "
                                  "reference, band 2.5 (fp32 passed); not benchmarked, not re-run since"),
    Model("microsoft/Phi-4-mini-instruct", "Phi-4 mini instruct", "MIT", 3.84, "2025-02-19",
          status=REJECTED, reason="Engine.load refuses it (its config sets a sliding window); not benchmarked"),
    Model("ibm-granite/granite-3.3-2b-instruct", "Granite 3.3 2B Instruct", "Apache-2.0", 2.53, "2025-04-09",
          status=REJECTED, reason="Engine.load refuses it (its forward divides the logits by 8 after the output head)"),
)


def get(model_id):
    """The listed Model with this id, or None."""
    return next((m for m in MODELS if m.id == model_id), None)


def approved():
    """Models the picker and GET /v1/models offer, in order: the approved ones and the pending ones."""
    return [m for m in MODELS if m.status in (APPROVED, PENDING)]


def note(m):
    """One short line for the picker: the role, then the speed and accuracy once measured."""
    if m.status == REJECTED:
        return f"Not approved: {m.reason}"
    parts = [m.role or "Not evaluated yet"]
    if m.ms_28_fields is not None:
        parts.append(f"{m.ms_28_fields:.0f} ms for 28 fields")
    if m.accuracy is not None:
        parts.append(f"{m.accuracy:.0%} accuracy")
    if m.status == PENDING:
        parts.append("not approved yet")
    return ", ".join(parts)


def entry(model_id):
    """The GET /v1/models form of a model: TypeSafe's `name`, `description` and `release_date`
    (so its clients can list models), then our own fields. An unlisted id is marked unvetted."""
    m = get(model_id)
    if m is None:
        return {"name": model_id, "title": model_id, "description": "Unvetted: not on the approved list",
                "release_date": "", "status": "unvetted"}
    fields = asdict(m)
    del fields["id"], fields["name"]
    return {"name": m.id, "title": m.name, "description": note(m), **fields}
