"""Shared pieces of the benchmark scripts: model options, timing, output files, hardware facts."""

import gc
import json
import platform
import subprocess
import time
from pathlib import Path

import torch
import transformers

OUT = Path(__file__).resolve().parent / "out"
DTYPES = ("bfloat16", "float16", "float32")


def add_model_args(parser):
    parser.add_argument("--model", action="append", default=None,
                        help="model id, repeatable (default: Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--dtype", choices=DTYPES, default=None, help="default: bfloat16 on cuda, float32 on cpu")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--batch-tokens", type=int, default=4096, help="engine scoring budget (SPEC 3.4)")
    parser.add_argument("--out", default=time.strftime("run-%Y%m%d-%H%M%S"),
                        help="run name: results go to bench/out/<name>/ (default: a timestamp)")


def models(args):
    return args.model or ["Qwen/Qwen2.5-1.5B-Instruct"]


def load_engine(model, args):
    """The engine as the benchmark uses it: raw scores at T = 1, nothing fed to Tarnlight or an event log."""
    from mirethstm import Engine

    dtype = getattr(torch, args.dtype) if args.dtype else None
    return Engine.load(model, device=args.device, dtype=dtype, batch_tokens=args.batch_tokens,
                       temperature=1.0, tarnlight=False)


def release():
    """Return a dropped model's memory before the next model loads (del the engine first)."""
    gc.collect()
    if torch.cuda.device_count():
        torch.cuda.empty_cache()


def on_cuda(engine):
    return engine.model.device.type == "cuda"


def timed(fn, cuda):
    """(result, wall ms) with torch.cuda.synchronize() right before the clock starts and before it stops."""
    if cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    if cuda:
        torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000


def slug(model):
    """A folder name for a model id or a local path."""
    return model.replace("\\", "/").rstrip("/").replace("/", "--").replace(":", "")


def write_jsonl(path, records):
    """Write all records, then rename into place, so a file that exists is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with part.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    part.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _nvidia_smi():
    query = "name,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return out.strip().splitlines()


def hardware():
    """Versions, device and GPU state, printed in every report (docs/research/07 section 4).

    No host name, user name or path: the run folder may be shared.
    """
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "os": platform.platform(terse=True),
        "cpu": platform.processor() or platform.machine(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.device_count() else None,
        "nvidia_smi": _nvidia_smi(),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return info
