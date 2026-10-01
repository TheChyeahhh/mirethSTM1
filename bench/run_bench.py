"""Benchmark stub: score a labelled dataset with the engine, report accuracy and ECE.

    python bench/run_bench.py --dataset ag_news --n 200 --fit

Writes one JSON line per sample (dataset, index, gold, raw label scores) under
bench/out/, so tables can be rebuilt without rerunning the model.
Design: docs/research/07-benchmark-design.md.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

from mirethstm import Engine
from mirethstm.calibration import ece, fit_temperature

OUT_DIR = Path(__file__).resolve().parent / "out"
SEED = 0
QID = "answer"

# Evaluation split, text field, label names in dataset index order (None: derived
# at runtime) and the TypeSafe question type each dataset becomes.
DATASETS = {
    "ag_news": {
        "hf_id": "fancyzhx/ag_news",
        "split": "test",
        "text": "text",
        "label_names": ["World", "Sports", "Business", "Sci/Tech"],
        "type": "choice",
        "instructions": "Which topic is this news article about?",
    },
    "banking77": {
        "hf_id": "mteb/banking77",
        "split": "test",
        "text": "text",
        "label_names": None,  # no ClassLabel; see banking77_names()
        "type": "choice",
        "instructions": "Which intent does this banking customer message express?",
    },
    "sst2": {
        "hf_id": "stanfordnlp/sst2",
        "split": "validation",  # test labels are hidden (all -1)
        "text": "sentence",
        "label_names": ["negative", "positive"],
        "type": "noul",
        "instructions": "Is the sentiment of this text positive?",
    },
    "yelp": {
        "hf_id": "Yelp/yelp_review_full",
        "split": "test",
        "text": "text",
        "label_names": ["1 star", "2 star", "3 stars", "4 stars", "5 stars"],
        "type": "score",
        "instructions": "How many stars does this review give?",
    },
}


def banking77_names(hf_id):
    """Intent names in label order, read from the train split's label_text column."""
    train = load_dataset(hf_id, split="train")
    pairs = sorted(set(zip(train["label"], train["label_text"])))
    if [label for label, _ in pairs] != list(range(len(pairs))):
        raise ValueError(f"{hf_id}: label ids and label_text do not map one to one")
    # Two raw names are irregular: "Refund_not_showing_up" and "reverted_card_payment?".
    return [name.lower().rstrip("?") for _, name in pairs]


def build_question(spec, names):
    """The TypeSafe question for a dataset, and the engine label for each gold index."""
    kind, instructions = spec["type"], spec["instructions"]
    if kind == "choice":
        return {"type": kind, "instructions": instructions, "criteria": dict.fromkeys(names)}, list(names)
    if kind == "noul":
        # Binary datasets: gold label 1 is the "true" answer.
        criteria = {"true": names[1], "false": names[0]}
        return {"type": kind, "instructions": instructions, "criteria": criteria}, ["false", "true"]
    return {"type": kind, "instructions": instructions, "criteria": list(names)}, [str(i) for i in range(len(names))]


def softmax(z, t=1.0):
    s = z / t
    e = np.exp(s - s.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def report(logits, gold, fit):
    """Print n, accuracy and ECE at T = 1; with fit, the fitted T and ECE after."""
    p = softmax(logits)
    correct = p.argmax(axis=1) == gold  # argmax, so accuracy does not depend on T

    def ece_line(p):
        conf = p.max(axis=1)
        return f"{ece(conf, correct):.4f} (15 bins), {ece(conf, correct, n_bins=10):.4f} (10 bins, as Kev)"

    print(f"n            {len(gold)}")
    print(f"accuracy     {correct.mean():.4f}")
    print(f"ECE at T=1   {ece_line(p)}")
    if not fit:
        return
    try:
        t = fit_temperature(logits, gold)
    except ValueError as err:
        print(f"fit failed   {err}")
        return
    print(f"fitted T     {t:.4f}")
    print(f"ECE after    {ece_line(softmax(logits, t))}")
    print(f"note         T was fitted and evaluated on the same {len(gold)} samples, so ECE after is optimistic")


# TODO functions, planned in docs/research/07-benchmark-design.md.

def generate_baseline(engine, text, schema):
    raise NotImplementedError("TODO: same model writes the JSON answer with greedy generate(); "
                              "schema-checked, invalid output counts as wrong")


def latency(engine, field_counts=(1, 5, 10, 20)):
    raise NotImplementedError("TODO: p50/p95 over 200 timed runs after 10 warmups per field count, "
                              "batch 1, CUDA synchronized at both ends")


def macro_f1(pred, gold, n_labels):
    raise NotImplementedError("TODO: mean of per-class F1, invalid baseline answers as an extra class")


def nll(probs, gold):
    raise NotImplementedError("TODO: mean of -log p(gold), p clipped at 1e-12")


def brier(probs, gold):
    raise NotImplementedError("TODO: mean over rows of the squared error summed over all classes")


def bootstrap_ci(values, statistic, n_resamples=1000):
    raise NotImplementedError("TODO: 95% interval of a metric over 1000 resamples of the evaluation rows")


def reliability_diagram(confidences, correct, path):
    raise NotImplementedError("TODO: 15-bin accuracy against confidence, before and after T, saved as PNG")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="model id (default: %(default)s)")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--n", type=int, default=200, help="samples, drawn at random with a fixed seed")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--out", default=None, help="JSONL path (default: bench/out/<dataset>-<model>.jsonl)")
    parser.add_argument("--fit", action="store_true", help="also fit T on these samples and report ECE after")
    args = parser.parse_args(argv)

    spec = DATASETS[args.dataset]
    names = spec["label_names"] or banking77_names(spec["hf_id"])
    question, labels = build_question(spec, names)
    schema = {QID: question}
    data = load_dataset(spec["hf_id"], split=spec["split"])
    rows = np.random.default_rng(SEED).permutation(len(data))[: args.n]

    engine = Engine.load(args.model, device=args.device)
    out = Path(args.out) if args.out else OUT_DIR / f"{args.dataset}-{Path(args.model).name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    logits, gold = [], []
    with out.open("w", encoding="utf-8") as f:
        for i in rows:
            row = data[int(i)]
            scores = engine.score(row[spec["text"]], schema)[QID]
            record = {"dataset": args.dataset, "index": int(i), "gold": labels[row["label"]], "scores": scores}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logits.append([scores[label] for label in labels])
            gold.append(row["label"])
    print(f"wrote        {out}")
    report(np.array(logits), np.array(gold), args.fit)


if __name__ == "__main__":
    main()
