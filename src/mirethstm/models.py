"""The approved models and the candidates still to evaluate (SPEC 11.1).

The console's model picker and GET /v1/models show only approved models. `Engine.load` and
`--model` still take any Hugging Face id (unvetted, at the user's own risk).

A model is approved when its license allows free commercial use (Apache-2.0 or MIT, public and
not gated), it fits a 12 GB card with room for the cache, `Engine.load` accepts it and the full
model test suite passes on the GPU, and its speed, accuracy and calibration are measured and
written here. PENDING marks a model the console offers by name (SPEC 10.1) that is not approved
yet: the license, size and load checks hold, but the GPU model suite does not pass yet and the
numbers are not measured.

Facts below come from the Hugging Face API (`https://huggingface.co/api/models/<id>`), checked
2026-09-30: `license` from its license tag, `params` from its safetensors total, `release_date`
from its `createdAt` (the day the weights repo was created, a few days before most public
announcements).
"""

from dataclasses import asdict, dataclass

APPROVED = "approved"
PENDING = "pending"
CANDIDATE = "candidate"


@dataclass(frozen=True)
class Model:
    id: str  # Hugging Face hub id
    name: str  # display name
    license: str
    params: float  # billions, every weight counted
    release_date: str  # YYYY-MM-DD
    role: str = ""  # SPEC 11.2
    status: str = CANDIDATE
    # Filled by the benchmark (ROADMAP Thursday and Saturday); None until measured.
    ms_28_fields: float | None = None  # warm wall time of the 28-field support scenario, RTX 5070, bf16
    ms_router: float | None = None  # same, the 255-option router scenario
    accuracy: float | None = None  # benchmark accuracy, 0 to 1
    ece: float | None = None  # 15-bin ECE after calibration
    temperature: float | None = None  # fitted temperature
    measured_on: str | None = None  # YYYY-MM-DD of the measurement


# Approved models first, in the picker's order (the console default leads), then the candidates.
MODELS = (
    Model("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5 1.5B Instruct", "Apache-2.0", 1.54, "2024-09-17",
          "Match first", PENDING),
    Model("Qwen/Qwen3-1.7B", "Qwen3 1.7B", "Apache-2.0", 2.03, "2025-04-27", "Fast", PENDING),
    Model("Qwen/Qwen3-4B-Instruct-2507", "Qwen3 4B Instruct 2507", "Apache-2.0", 4.02, "2025-08-05",
          "Default SDK/CLI", PENDING),
    Model("Qwen/Qwen3-0.6B", "Qwen3 0.6B", "Apache-2.0", 0.75, "2025-04-27", "CPU tests", PENDING),
    Model("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5 0.5B Instruct", "Apache-2.0", 0.49, "2024-09-16"),
    Model("microsoft/Phi-4-mini-instruct", "Phi-4 mini instruct", "MIT", 3.84, "2025-02-19"),
    Model("HuggingFaceTB/SmolLM3-3B", "SmolLM3 3B", "Apache-2.0", 3.08, "2025-07-08"),
    Model("ibm-granite/granite-3.3-2b-instruct", "Granite 3.3 2B Instruct", "Apache-2.0", 2.53, "2025-04-09"),
)


def get(model_id):
    """The listed Model with this id, or None."""
    return next((m for m in MODELS if m.id == model_id), None)


def approved():
    """Models the picker and GET /v1/models offer, in order: the approved ones and the pending ones."""
    return [m for m in MODELS if m.status in (APPROVED, PENDING)]


def note(m):
    """One short line for the picker: the role, then the speed and accuracy once measured."""
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
