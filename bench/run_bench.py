"""Accuracy and calibration runs: every dataset through every arm, one JSON line per sample.

    python -m bench.run_bench --model Qwen/Qwen2.5-1.5B-Instruct --n 200 --n-cal 200 --out dry-run

Writes bench/out/<run>/<model>/<dataset>.<split>.<arm>.jsonl, split "eval" (scored) or
"cal" (the disjoint calibration split, scoring arms only, for fitting T), plus the sample
lists and hardware in bench/out/<run>/meta.json. Each line holds the dataset, row index,
gold label, arm, the raw label scores (or the generated answer) and latency_ms, so every
table rebuilds without the model: python -m bench.report --run <run>.

A file that exists is complete (written whole, then renamed), so a rerun with the same
--out skips it and an interrupted run resumes where it stopped.
"""

import argparse
from typing import NamedTuple

import numpy as np

from . import arms, data
from .common import OUT, add_model_args, hardware, load_engine, models, on_cuda, read_json, release, slug, \
    timed, write_json, write_jsonl

DEFAULT_ARMS = ("mireth", "baseline", "first_token")


class Task(NamedTuple):
    index: int
    state: object
    questions: dict  # one question
    gold: str  # engine label
    extra: dict = {}  # more keys for the record (JevBench: item id, family)


def dataset_tasks(question, labels, rows):
    return [Task(s.index, s.text, {data.QID: question}, labels[s.gold]) for s in rows]


def run_tasks(engine, arm, tasks, base):
    """Run one arm over tasks (one untimed warmup call first); returns the records."""
    fn = arms.ARMS[arm]
    cuda = on_cuda(engine)
    fn(engine, tasks[0].state, tasks[0].questions)
    records = []
    for t in tasks:
        result, ms = timed(lambda: fn(engine, t.state, t.questions), cuda)
        (r,) = result.values()
        records.append({**base, "index": t.index, **t.extra, "gold": t.gold, "arm": arm, **r, "latency_ms": ms})
    return records


def summary(records):
    """One progress line: accuracy (and validity for generation) and mean latency."""
    if "scores" in records[0]:
        hits = [max(r["scores"], key=r["scores"].get) == r["gold"] for r in records]
        extra = ""
    elif "probs" in records[0]:
        hits = [max(r["probs"], key=r["probs"].get) == r["gold"] for r in records]
        extra = ""
    else:
        hits = [r["pred"] == r["gold"] for r in records]
        extra = f", valid {np.mean([r['pred'] is not None for r in records]):.3f}"
    return f"n {len(records)}, accuracy {np.mean(hits):.3f}{extra}, mean {np.mean([r['latency_ms'] for r in records]):.0f} ms"


def run_files(engine, model, jobs, run_dir):
    """jobs: (file stem, arm, tasks, base record keys). Skips files that already exist."""
    for stem, arm, tasks, base in jobs:
        path = run_dir / slug(model) / f"{stem}.{arm}.jsonl"
        if path.exists():
            print(f"skip   {path.relative_to(run_dir)} (exists)", flush=True)
            continue
        records = run_tasks(engine, arm, tasks, {"model": model, **base})
        write_jsonl(path, records)
        print(f"wrote  {path.relative_to(run_dir)}: {summary(records)}", flush=True)


def missing(model, jobs, run_dir):
    return [job for job in jobs if not (run_dir / slug(model) / f"{job[0]}.{job[1]}.jsonl").exists()]


def append_meta(run_dir, key, entry):
    path = run_dir / "meta.json"
    meta = read_json(path) if path.exists() else {}
    meta.setdefault(key, []).append(entry)
    write_json(path, meta)


def split_list(text, allowed, what):
    items = [x for x in text.split(",") if x]
    unknown = sorted(set(items) - set(allowed))
    if unknown or not items:
        raise SystemExit(f"unknown {what}: {', '.join(unknown) or '(none given)'}; choose from {', '.join(allowed)}")
    return items


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(parser)
    parser.add_argument("--datasets", default=",".join(data.DATASETS), help="comma list (default: all)")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                        help=f"comma list of {', '.join(arms.ARMS)} (default: %(default)s)")
    parser.add_argument("--n", type=int, default=200, help="evaluation samples per dataset (default: %(default)s)")
    parser.add_argument("--n-cal", type=int, default=200,
                        help="calibration samples per dataset, scoring arms only (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=data.SEED)
    args = parser.parse_args(argv)
    names = split_list(args.datasets, data.DATASETS, "dataset")
    chosen = split_list(args.arms, arms.ARMS, "arm")
    run_dir = OUT / args.out

    # Datasets first, so a missing download fails before any model loads.
    jobs, lists = [], {}
    for name in names:
        question, labels = data.question(name)
        n_cal = args.n_cal if any(a in arms.SCORING for a in chosen) else 0
        ev, cal = data.samples(name, args.n, n_cal, args.seed)
        lists[name] = {"hf_id": data.DATASETS[name].hf_id, "question": question, "labels": labels,
                       "eval": [s.index for s in ev], "cal": [s.index for s in cal]}
        for split, rows in (("cal", cal), ("eval", ev)):
            for arm in chosen:
                if rows and (split == "eval" or arm in arms.SCORING):
                    jobs.append((f"{name}.{split}", arm, dataset_tasks(question, labels, rows),
                                 {"dataset": name, "split": split}))
    append_meta(run_dir, "run_bench", {"args": vars(args), "models": models(args), "hardware": hardware(),
                                       "datasets": lists})

    for model in models(args):
        todo = missing(model, jobs, run_dir)
        if not todo:
            print(f"model  {model}: every file exists", flush=True)
            continue
        print(f"model  {model}: loading", flush=True)
        engine = load_engine(model, args)
        append_meta(run_dir, "loaded", {"script": "run_bench", "model": model, "dtype": str(engine.model.dtype),
                                        "device": str(engine.model.device)})
        run_files(engine, model, todo, run_dir)
        del engine
        release()
    print(f"done   tables: python -m bench.report --run {args.out}")


if __name__ == "__main__":
    main()
