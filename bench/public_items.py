"""JevBench public items through MirethSTM1, scored like the other benchmark datasets.

JevBench (github.com/fstandhartinger/jevbench, MIT) is an independent public board for
Jev-class systems. Its 231 public items (48 easy, 72 standard, 111 hard; every item's own
provenance says MIT) are fetched at run time from a pinned commit and checked against the
SHA-256 hashes JevBench publishes in its datasets/manifest.json. Nothing of JevBench is
copied into this repository; its harness is not used.

    python -m bench.public_items --model Qwen/Qwen2.5-1.5B-Instruct --out jb    # engine in process
    python -m bench.public_items --url http://127.0.0.1:8766 --out jb           # POST /v1/systemone

Item to request, as JevBench's own TypeSafe adapter does: the item's state as given (a
string or an object), its question as the one question "decision" (criteria left out when
null). Gold label: "yes"/"no" become true/false for a yes/no item, a score item's integer
level becomes its index string, a choice keeps its option name.

Writes bench/out/<run>/<model>/jevbench_<tier>.eval.<arm>.jsonl. Over HTTP the arm is named
after the route ("systemone": numbers rounded to 2 decimals; "decide": full precision) and
the record holds the returned probabilities ("probs") instead of raw scores.
"""

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import arms
from .common import OUT, add_model_args, hardware, load_engine, models, release, slug, write_jsonl
from .run_bench import DEFAULT_ARMS, Task, append_meta, missing, run_files, split_list, summary

COMMIT = "bb05a335bc809e61b20c0f745d25499a82b326fc"
URL = "https://raw.githubusercontent.com/fstandhartinger/jevbench/{commit}/datasets/public/{file}"
# tier: (file, sha256 of the file with LF line ends, as in JevBench's datasets/manifest.json)
TIERS = {
    "easy": ("easy.jsonl", "231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b"),
    "standard": ("original.jsonl", "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180"),
    "hard": ("hard.jsonl", "89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb"),
}
QID = "decision"


def fetch(tier, items_dir=None):
    """The tier's items, from GitHub or from a local folder holding the same files."""
    name, digest = TIERS[tier]
    if items_dir:
        raw = (Path(items_dir) / name).read_bytes()
    else:
        with urllib.request.urlopen(URL.format(commit=COMMIT, file=name), timeout=60) as resp:
            raw = resp.read()
    raw = raw.replace(b"\r\n", b"\n")  # a Windows checkout may have converted line ends
    if hashlib.sha256(raw).hexdigest() != digest:
        raise SystemExit(f"jevbench {name}: SHA-256 differs from the pinned file at {COMMIT[:7]}")
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def to_task(position, item):
    """A benchmark Task for one JevBench item."""
    q = item["question"]
    question = {"type": q["type"], "instructions": q["instructions"]}
    if q.get("criteria") is not None:
        question["criteria"] = q["criteria"]
    gold = {"yes": "true", "no": "false"}[item["expected"]] if q["type"] == "noul" else str(item["expected"])
    return Task(position, item["state"], {QID: question}, gold,
                {"id": item["id"], "family": item["family"], "n_labels": len(item["labels"])})


def _post(url, body):
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=300) as resp:
            out = json.load(resp)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{url}: HTTP {e.code}: {e.read()[:300]!r}") from e
    return out, (time.perf_counter() - start) * 1000


def over_http(base_url, route, tier, tasks, run_dir):
    """Score tasks through a running server's /v1/systemone or /v1/decide."""
    url = base_url.rstrip("/") + f"/v1/{route}"
    records = []
    for t in tasks:
        out, ms = _post(url, {"model": "any", "state": t.state, "questions": t.questions})
        a = out["answers"][QID]
        probs = {"true": a["noul"], "false": 1 - a["noul"]} if a["type"] == "noul" else a["probabilities"]
        records.append({"model": out["model"], "dataset": f"jevbench_{tier}", "split": "eval", "index": t.index,
                        **t.extra, "gold": t.gold, "arm": route, "probs": probs, "latency_ms": ms})
    path = run_dir / slug(records[0]["model"]) / f"jevbench_{tier}.eval.{route}.jsonl"
    write_jsonl(path, records)
    print(f"wrote  {path.relative_to(run_dir)}: {summary(records)}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(parser)
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                        help=f"in-process arms, comma list of {', '.join(arms.ARMS)} (default: %(default)s)")
    parser.add_argument("--tiers", default=",".join(TIERS), help="comma list (default: %(default)s)")
    parser.add_argument("--limit", type=int, default=None, help="first N items per tier (smoke runs)")
    parser.add_argument("--items-dir", default=None, help="read the item files from this folder instead of GitHub")
    parser.add_argument("--url", default=None, help="score through this server instead of in process, "
                                                    "e.g. http://127.0.0.1:8766 (the console's server)")
    parser.add_argument("--route", choices=("systemone", "decide"), default="systemone",
                        help="with --url: the route (default: %(default)s)")
    args = parser.parse_args(argv)
    tiers = split_list(args.tiers, TIERS, "tier")
    chosen = split_list(args.arms, arms.ARMS, "arm")
    run_dir = OUT / args.out

    tasks = {tier: [to_task(i, item) for i, item in enumerate(fetch(tier, args.items_dir))][: args.limit]
             for tier in tiers}
    append_meta(run_dir, "jevbench", {"args": vars(args), "commit": COMMIT, "hardware": hardware(),
                                      "items": {tier: len(t) for tier, t in tasks.items()}})
    if args.url:
        for tier in tiers:
            over_http(args.url, args.route, tier, tasks[tier], run_dir)
        return
    jobs = [(f"jevbench_{tier}.eval", arm, tasks[tier], {"dataset": f"jevbench_{tier}", "split": "eval"})
            for tier in tiers for arm in chosen]
    for model in models(args):
        todo = missing(model, jobs, run_dir)
        if not todo:
            print(f"model  {model}: every file exists", flush=True)
            continue
        print(f"model  {model}: loading", flush=True)
        engine = load_engine(model, args)
        run_files(engine, model, todo, run_dir)
        del engine
        release()


if __name__ == "__main__":
    main()
