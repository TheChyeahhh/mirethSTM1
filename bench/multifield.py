"""Do answers hold up when a question shares its call with 19 others? (SPEC 3.1)

    python -m bench.multifield --model Qwen/Qwen2.5-1.5B-Instruct --n 200 --out multifield

AG News test articles. Five checked questions with known answers (the topic choice and four
yes/no questions that follow from the gold topic) are asked alone, one call per question, and
inside one 20-question call next to 15 filler yes/no questions, at three placements: first, last
and spread out. SPEC 3.1 gives every question a branch of its own, so all four modes should
score the same; a question that could see its neighbours would not (measured before the
branches, Qwen2.5-1.5B-Instruct: 0.864 alone, 0.809 first, 0.575 last, 0.500 spread).

Writes bench/out/<run>/<model>/multifield.<mode>.jsonl, one line per article with the raw label
scores of the five checked questions (a file that exists is skipped), and prints the table that
`python -m bench.report --run <run>` also shows.
"""

import argparse

from . import data
from .common import OUT, add_model_args, hardware, load_engine, models, on_cuda, read_jsonl, release, slug, timed, \
    write_jsonl
from .run_bench import append_meta

TOPICS = ("World", "Sports", "Business", "Sci/Tech")  # AG News label order
CHECKED = {
    "topic": {"type": "choice", "instructions": "Which topic is this news article about?",
              "criteria": dict.fromkeys(TOPICS)},
    "is_sports": {"type": "noul", "instructions": "Is this article about sports?"},
    "is_business": {"type": "noul", "instructions": "Is this article about business or the economy?"},
    "is_scitech": {"type": "noul", "instructions": "Is this article about science or technology?"},
    "is_world": {"type": "noul", "instructions": "Is this article about world news or international politics?"},
}
IS_TOPIC = {"is_sports": "Sports", "is_business": "Business", "is_scitech": "Sci/Tech", "is_world": "World"}
FILLER = (
    "Does the article name a company?", "Does the article quote a person?", "Is the tone of the article negative?",
    "Does the article mention money?", "Does the article involve a government?", "Is a specific country named?",
    "Does the article describe a competition or a game?", "Does the article mention a product launch?",
    "Does the article give a statistic?", "Does the article describe a conflict?",
    "Is the article about a court case?", "Does the article mention the internet?",
    "Does the article involve health or medicine?", "Does the article mention a date?",
    "Does the article mention an election?",
)
_checked = list(CHECKED.items())
_filler = [(f"fill_{i:02d}", {"type": "noul", "instructions": text}) for i, text in enumerate(FILLER, 1)]
# mode: (title, the questions of each call)
MODES = {
    "alone": ("Alone, one call per question", [{qid: q} for qid, q in _checked]),
    "first": ("One call of 20, checked questions first (1 to 5)", [dict(_checked + _filler)]),
    "last": ("One call of 20, checked questions last (16 to 20)", [dict(_filler + _checked)]),
    "spread": ("One call of 20, checked questions spread (4, 8, 12, 16, 20)",
               [dict(x for i in range(5) for x in _filler[3 * i:3 * i + 3] + [_checked[i]])]),
}


def truth(topic):
    """The label of every checked question for an article of this gold topic."""
    return {"topic": topic, **{qid: str(topic == name).lower() for qid, name in IS_TOPIC.items()}}


def run_mode(engine, mode, rows, base):
    """One record per article: the raw scores of the checked questions, over the mode's calls."""
    calls = MODES[mode][1]
    cuda = on_cuda(engine)
    for questions in calls:  # untimed warmup
        engine.score(rows[0].text, questions)
    records = []
    for row in rows:
        scores, total = {}, 0.0
        for questions in calls:
            out, ms = timed(lambda: engine.score(row.text, questions), cuda)
            scores.update({qid: out[qid] for qid in CHECKED if qid in out})
            total += ms
        records.append({**base, "mode": mode, "index": row.index, "truth": truth(TOPICS[row.gold]),
                        "scores": {qid: scores[qid] for qid in CHECKED}, "latency_ms": total})
    return records


def _top(scores):
    return max(scores, key=scores.get)  # ties go to the earlier label, as in the engine


def table(by_mode):
    """(header, rows) from {mode: records}: accuracy per checked question, with the share of
    articles answered yes in brackets (the true share is about a quarter), the mean accuracy, the
    share of answers equal to the alone mode's, and the largest score difference from it."""
    header = ["Mode", *CHECKED, "Mean", "Same answer as alone", "Largest score difference from alone"]
    alone = {r["index"]: r for r in by_mode.get("alone", [])}
    out = []
    for mode in (m for m in MODES if m in by_mode):
        records = by_mode[mode]
        cells, hits = [MODES[mode][0]], 0
        for qid in CHECKED:
            right = sum(_top(r["scores"][qid]) == r["truth"][qid] for r in records)
            hits += right
            cell = f"{right / len(records):.3f}"
            if qid != "topic":
                cell += f" [{sum(_top(r['scores'][qid]) == 'true' for r in records) / len(records):.2f}]"
            cells.append(cell)
        cells.append(f"{hits / (len(records) * len(CHECKED)):.3f}")
        pairs = [(r["scores"][qid], alone[r["index"]]["scores"][qid])
                 for r in records if r["index"] in alone for qid in CHECKED]
        if mode == "alone" or not pairs:
            cells += ["n/a", "n/a"]
        else:
            cells.append(f"{sum(_top(a) == _top(b) for a, b in pairs) / len(pairs):.3f}")
            cells.append(f"{max(abs(a[label] - b[label]) for a, b in pairs for label in a):.3g}")
        out.append(cells)
    return header, out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(parser)
    parser.add_argument("--n", type=int, default=200, help="AG News test articles (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=data.SEED)
    args = parser.parse_args(argv)
    run_dir = OUT / args.out
    rows, _ = data.samples("ag_news", args.n, 0, args.seed)
    for model in models(args):
        folder = run_dir / slug(model)
        todo = [mode for mode in MODES if not (folder / f"multifield.{mode}.jsonl").exists()]
        if todo:
            print(f"model  {model}: loading", flush=True)
            engine = load_engine(model, args)
            append_meta(run_dir, "multifield", {"args": vars(args), "model": model, "dtype": str(engine.model.dtype),
                                                "device": str(engine.model.device), "hardware": hardware()})
            for mode in todo:
                write_jsonl(folder / f"multifield.{mode}.jsonl", run_mode(engine, mode, rows, {"model": model}))
                print(f"wrote  {slug(model)}/multifield.{mode}.jsonl", flush=True)
            del engine
            release()
        header, body = table({mode: read_jsonl(folder / f"multifield.{mode}.jsonl") for mode in MODES})
        print(f"\n{model}, AG News, {len(rows)} articles. Accuracy per question [share answered yes]")
        widths = [max(len(cells[i]) for cells in [header, *body]) for i in range(len(header))]
        for cells in [header, *body]:
            print("  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(cells, widths))))
    print(f"\ndone   tables: python -m bench.report --run {args.out}")


if __name__ == "__main__":
    main()
