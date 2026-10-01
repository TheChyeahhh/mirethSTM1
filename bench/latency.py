"""Latency protocol (docs/research/07 section 4): MirethSTM1 against normal generation, same model.

    python -m bench.latency --model Qwen/Qwen2.5-1.5B-Instruct --runs 50 --out speed

Batch 1, one request at a time, model loaded. Per field count (1, 5, 10, 20) the questions
are an AG News topic choice plus deterministic yes/no questions whose answers follow from
rules, asked about AG News test articles; then the four console scenarios. Per setting:
--warmup untimed runs, then --runs timed runs, the arms interleaved run by run so drift
hits both, every timed run on a different article (scenarios repeat their own text). Each
timed span is one `Engine.decide` or `baseline.generate` call, with torch.cuda.synchronize()
at both ends on CUDA.

Writes bench/out/<run>/<model>/latency.<setting>.jsonl (one line per timed run) and the
hardware state at start and end of each model into bench/out/<run>/meta.json.
"""

import argparse
import re

import torch

from mirethstm import baseline
from mirethstm.scenarios import SCENARIOS

from . import data
from .common import OUT, add_model_args, hardware, load_engine, models, on_cuda, release, slug, timed, write_jsonl
from .run_bench import append_meta, split_list

ARMS = ("mireth", "baseline")
FIELD_COUNTS = (1, 5, 10, 20)


def _has(pattern):
    return lambda t: re.search(pattern, t) is not None


# Yes/no questions with a rule for the true answer, so timed runs still produce checkable answers.
SYNTHETIC = [
    ("Does the text contain a digit?", _has(r"\d")),
    ("Does the text contain a question mark?", _has(r"\?")),
    ("Is the text longer than 200 characters?", lambda t: len(t) > 200),
    ("Does the text contain a double quotation mark?", _has('"')),
    ("Does the text start with an uppercase letter?", lambda t: t[:1].isupper()),
    ('Does the text contain the word "the"?', _has(r"(?i)\bthe\b")),
    ("Does the text contain a comma?", _has(",")),
    ("Does the text contain an exclamation mark?", _has("!")),
    ("Does the text contain a dollar sign?", _has(r"\$")),
    ("Does the text end with a period?", lambda t: t.rstrip().endswith(".")),
    ("Does the text have more than 40 words?", lambda t: len(t.split()) > 40),
    ("Does the text contain a hyphen?", _has("-")),
    ("Does the text contain a parenthesis?", _has(r"[()]")),
    ("Does the text contain a percent sign?", _has("%")),
    ('Does the text contain the word "and"?', _has(r"(?i)\band\b")),
    ("Does the text mention a year from 1900 to 2099?", _has(r"\b(19|20)\d\d\b")),
    ("Does the text contain a word of two or more capital letters only?", _has(r"\b[A-Z]{2,}\b")),
    ("Does the text contain a colon?", _has(":")),
    ("Does the text contain a semicolon?", _has(";")),
    ("Does the text contain an apostrophe?", _has("'")),
    ('Does the text contain the word "said"?', _has(r"(?i)\bsaid\b")),
    ("Does the text contain more than three sentences?", lambda t: len(re.findall(r"[.!?](\s|$)", t)) > 3),
    ("Does the text contain an ampersand?", _has("&")),
    ('Does the text contain the letter "z"?', _has("(?i)z")),
    ("Does the text contain a web address?", _has(r"https?://|www\.")),
]


def field_schema(n_fields, topic_question):
    """The questions for n fields: the topic, then the first n - 1 rule-based yes/no questions."""
    if not 1 <= n_fields <= len(SYNTHETIC) + 1:
        raise ValueError(f"field counts run from 1 to {len(SYNTHETIC) + 1}")
    questions = {"topic": topic_question}
    for i, (instructions, _) in enumerate(SYNTHETIC[: n_fields - 1], 1):
        questions[f"check_{i:02d}"] = {"type": "noul", "instructions": instructions}
    return questions


def truth(questions, text, topic):
    """The checkable answer of every field: the gold topic, or the rule's verdict."""
    rules = {f"check_{i:02d}": rule for i, (_, rule) in enumerate(SYNTHETIC, 1)}
    return {qid: topic if qid == "topic" else rules[qid](text) for qid in questions}


def _mireth(engine, text, questions):
    out = engine.decide(text, questions)
    answers = {qid: a["noul"] >= 0.5 if a["type"] == "noul" else a.get("choice") for qid, a in out["answers"].items()}
    return answers, {"input_tokens": out["usage"]["input_tokens"]}


def _generate(engine, text, questions):
    out = baseline.generate(engine, text, questions)
    bad = set(out["hallucinated"]) | set(out["missing"])
    answers = {qid: None if qid in bad else v for qid, v in out["answers"].items()}
    return answers, {"output_tokens": out["output_tokens"], "valid_json": out["valid_json"],
                     "hallucinated": len(out["hallucinated"]), "missing": len(out["missing"])}


CALLS = {"mireth": _mireth, "baseline": _generate}


def measure(engine, arms, cases, warmup, runs, base):
    """cases: (text, questions, truth or None) in order; the first `warmup` are untimed."""
    cuda = on_cuda(engine)
    records = []
    for i, (text, questions, expected) in enumerate(cases[: warmup + runs]):
        for arm in arms:
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            (answers, extra), ms = timed(lambda: CALLS[arm](engine, text, questions), cuda)
            if i < warmup:
                continue
            record = {**base, "arm": arm, "run": i - warmup, "fields": len(questions), "latency_ms": ms, **extra,
                      "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if cuda else None}
            if expected is not None:
                record["correct"] = sum(answers[q] == v for q, v in expected.items())
            records.append(record)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(parser)
    parser.add_argument("--arms", default=",".join(ARMS), help="comma list (default: %(default)s)")
    parser.add_argument("--fields", default=",".join(map(str, FIELD_COUNTS)), help="default: %(default)s")
    parser.add_argument("--runs", type=int, default=50, help="timed runs per setting (default: %(default)s)")
    parser.add_argument("--warmup", type=int, default=10, help="untimed runs per setting (default: %(default)s)")
    parser.add_argument("--scenarios", default=",".join(s["id"] for s in SCENARIOS),
                        help="console scenarios, comma list or 'none' (default: all four)")
    parser.add_argument("--scenario-runs", type=int, default=20, help="timed runs per scenario (default: %(default)s)")
    args = parser.parse_args(argv)
    chosen = split_list(args.arms, ARMS, "arm")
    counts = [int(x) for x in args.fields.split(",") if x]
    by_id = {s["id"]: s for s in SCENARIOS}
    scenarios = [] if args.scenarios == "none" else split_list(args.scenarios, by_id, "scenario")
    run_dir = OUT / args.out

    topic_question, names = data.question("ag_news")
    articles, _ = data.samples("ag_news", args.warmup + args.runs, 0)
    settings = []
    for n in counts:
        questions = field_schema(n, topic_question)
        cases = [(a.text, questions, truth(questions, a.text, names[a.gold])) for a in articles]
        settings.append((f"f{n:02d}", cases, args.runs, {"setting": f"{n} field" + ("s" if n != 1 else ""), "scenario": None}))
    for sid in scenarios:
        s = by_id[sid]
        cases = [(s["state"], s["questions"], None)] * (args.warmup + args.scenario_runs)
        settings.append((sid, cases, args.scenario_runs, {"setting": s["title"], "scenario": sid}))

    for model in models(args):
        todo = [s for s in settings if not (run_dir / slug(model) / f"latency.{s[0]}.jsonl").exists()]
        if not todo:
            print(f"model  {model}: every file exists", flush=True)
            continue
        print(f"model  {model}: loading", flush=True)
        engine = load_engine(model, args)
        start = hardware()
        for name, cases, runs, base in todo:
            records = measure(engine, chosen, cases, args.warmup, runs, {"model": model, **base})
            path = run_dir / slug(model) / f"latency.{name}.jsonl"
            write_jsonl(path, records)
            line = ", ".join(f"{arm} median {sorted(r['latency_ms'] for r in records if r['arm'] == arm)[runs // 2]:.0f} ms"
                             for arm in chosen)
            print(f"wrote  {path.relative_to(run_dir)}: {line}", flush=True)
        append_meta(run_dir, "latency", {"args": vars(args), "model": model, "dtype": str(engine.model.dtype),
                                         "device": str(engine.model.device), "hardware_start": start,
                                         "hardware_end": hardware()})
        del engine
        release()


if __name__ == "__main__":
    main()
