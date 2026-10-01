"""The FROZEN JSONL event log (SPEC 8). Keys and their order must not change."""

import json
from datetime import datetime, timezone


def write_events(path, call_id, probs_by_field, temperature, latency_ms, model):
    """Append one line per question of one decide call, all in a single write.

    `probs_by_field` maps each question id to {label: probability} in request order.
    """
    ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    lines = []
    for field, probs in probs_by_field.items():
        label = max(probs, key=probs.get)  # ties go to the earlier label
        event = {
            "ts": ts,
            "id": call_id,
            "field": field,
            "label": label,
            "p": probs[label],
            "probs": probs,
            "T": temperature,
            "latency_ms": latency_ms,
            "model": model,
        }
        lines.append(json.dumps(event, ensure_ascii=False) + "\n")
    # Binary append: no newline translation on Windows, and one write for the whole call.
    with open(path, "ab") as f:
        f.write("".join(lines).encode("utf-8"))
