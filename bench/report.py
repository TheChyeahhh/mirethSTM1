"""Benchmark tables and reliability diagrams from saved runs; never loads a model.

    python -m bench.report --run dry-run [--md docs/benchmark.md --figures docs/benchmark]

Reads every bench/out/<run>/<model>/*.jsonl and writes one markdown report (the layout of
docs/benchmark.md) plus reliability diagrams (PNG) and their bin tables (JSON) in
bench/out/<run>/report/. Temperature is fitted on the calibration split, per model and arm:
once pooled over the datasets and once per dataset; every number after T is measured on the
evaluation split only. Runs named <run>-fp16 and <run>-fp32, when present, hold the same rows
in another dtype and feed the Precision section. With --figures only the MirethSTM1 arm's
diagrams are written there (PNG, no bin tables), so a report kept in docs/ stays small.
"""

import argparse
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from mirethstm.calibration import ece, fit_temperature
from mirethstm.scenarios import SCENARIOS

from . import data, metrics, multifield
from .common import OUT, read_json, read_jsonl, slug, write_json

ARM_NAMES = {
    "mireth": "MirethSTM1 (full label)",
    "first_token": "First token (original demo's method)",
    "baseline": "Normal generation",
    "questions_first": "MirethSTM1, questions before state",
    "systemone": "MirethSTM1 via /v1/systemone",
    "decide": "MirethSTM1 via /v1/decide",
}
ARM_ORDER = list(ARM_NAMES)
BOARD_TIERS = ("easy", "standard", "hard")
SCENARIO_ORDER = [s["id"] for s in SCENARIOS]
DTYPE_NAMES = {"torch.bfloat16": "bf16", "torch.float16": "fp16", "torch.float32": "fp32"}
PRECISION_RUNS = ("fp16", "fp32")  # run folders <run>-fp16 and <run>-fp32; fp32 is the reference


# --- loading -----------------------------------------------------------------------------


def load(run_dir):
    """({(model, dataset, split, arm): records}, {model: latency records},
    {model: {mode: bench.multifield records}}, meta)."""
    groups, latency, many = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list))
    for path in sorted(run_dir.glob("*/*.jsonl")):
        for r in read_jsonl(path):
            if path.name.startswith("latency."):
                latency[r["model"]].append(r)
            elif path.name.startswith("multifield."):
                many[r["model"]][r["mode"]].append(r)
            else:
                groups[(r["model"], r["dataset"], r["split"], r["arm"])].append(r)
    meta_path = run_dir / "meta.json"
    return groups, latency, many, read_json(meta_path) if meta_path.exists() else {}


def has_dist(records):
    return "scores" in records[0] or "probs" in records[0]


def vectors(records, t=1.0):
    """Probability vectors and gold indices. Raw scores go through softmax(s / t); returned
    probabilities (HTTP arms) are used as given, rescaled to sum to 1 after rounding."""
    probs, gold = [], []
    for r in records:
        if "scores" in r:
            labels, p = list(r["scores"]), metrics.softmax(list(r["scores"].values()), t)
        else:
            labels, p = list(r["probs"]), np.array(list(r["probs"].values()), dtype=float)
            p = p / p.sum()
        probs.append(p)
        gold.append(labels.index(r["gold"]))
    return probs, np.array(gold)


def fit(records):
    """T fitted by NLL on raw scores, or None (no raw scores, or the optimum sits at a bound)."""
    records = [r for r in records if "scores" in r]
    if not records:
        return None
    try:
        return fit_temperature([np.array(list(r["scores"].values())) for r in records],
                               [list(r["scores"]).index(r["gold"]) for r in records])
    except ValueError:
        return None


def fit3(records):
    """`fit` rounded to three decimals: the pooled T as it ships and as the tables print it."""
    t = fit(records)
    return None if t is None else round(t, 3)


def temperatures(groups):
    """{(model, arm): {"pooled": T, dataset: T}} from the calibration splits. The pooled T is
    rounded to three decimals (the value that ships), so every number at it matches the Summary."""
    cal = defaultdict(dict)
    for (model, ds, split, arm), records in groups.items():
        if split == "cal":
            cal[(model, arm)][ds] = records
    out = {}
    for key, by_ds in cal.items():
        out[key] = {"pooled": fit3([r for records in by_ds.values() for r in records])}
        out[key].update({ds: fit(records) for ds, records in by_ds.items()})
    return out


def generated_ids(records):
    """Gold and predicted ints for generated answers; an invalid answer is -1."""
    names = sorted({r["gold"] for r in records} | {r["pred"] for r in records if r["pred"] is not None})
    index = {name: i for i, name in enumerate(names)}
    gold = np.array([index[r["gold"]] for r in records])
    pred = np.array([-1 if r["pred"] is None else index[r["pred"]] for r in records])
    return pred, gold


def top(record):
    """A record's answer: the top label of a scoring arm (ties go to the earlier label), else the generated one."""
    dist = record.get("scores") or record.get("probs")
    return max(dist, key=dist.get) if dist else record["pred"]


def accuracy(records):
    return float(np.mean([top(r) == r["gold"] for r in records]))


def ece15(records, t=1.0):
    r = metrics.rows(*vectors(records, t))
    return ece(r["conf"], r["correct"])


def dtype_of(meta, model, default=""):
    """The dtype a model was loaded in (bf16, fp16, fp32), from the run's meta.json."""
    names = [DTYPE_NAMES.get(e["dtype"], e["dtype"]) for key in ("loaded", "latency", "multifield")
             for e in meta.get(key, []) if e.get("model") == model]
    return names[-1] if names else default


def shown(dataset):
    """A dataset key as the tables print it (the public-item tiers by their tier name)."""
    return f"public items, {dataset.split('_', 1)[1]}" if dataset.startswith("jevbench_") else dataset


# --- formatting ----------------------------------------------------------------------------


def f3(value):
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.3f}"


def ci(stat):
    value, lo, hi = stat
    return f3(value) if lo is None else f"{value:.3f} [{lo:.3f}, {hi:.3f}]"


def ms(value):
    return "n/a" if value is None else (f"{value:.0f}" if value >= 100 else f"{value:.1f}")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out + [""]


def ordered(keys, order):
    return sorted(keys, key=lambda k: (order.index(k) if k in order else len(order), k))


# --- figures -------------------------------------------------------------------------------

INK, INK_2, SURFACE, GRID, BAR = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3de", "#2a78d6"


def diagram(path, title, panels, bins_json=True):
    """Reliability diagram, one column per (subtitle, confidences, correct): accuracy per
    15-bin confidence bin against the diagonal, with the bin counts below. The bin table goes
    next to it as JSON unless `bins_json` is false."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(panels), figsize=(4.2 * len(panels), 5.2), squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1]}, facecolor=SURFACE)
    tables = {}
    for col, (subtitle, conf, correct) in enumerate(panels):
        bins = metrics.reliability_bins(conf, correct, 15)
        tables[subtitle] = bins
        top, bottom = axes[0][col], axes[1][col]
        full = [b for b in bins if b["n"]]
        width = 1 / 15
        centers, accs = [b["lo"] + width / 2 for b in full], [b["acc"] for b in full]
        top.bar(centers, accs, width=width * 0.86, color=BAR, linewidth=0)
        top.plot(centers, accs, "o", color=BAR, markersize=4)  # keeps a bin with accuracy 0 visible
        top.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
        top.set_title(f"{subtitle}\nECE (15 bins) {ece(conf, correct):.3f}", color=INK, fontsize=10)
        top.set_ylabel("Accuracy in bin", color=INK_2, fontsize=9)
        bottom.bar([b["lo"] + width / 2 for b in bins], [b["n"] for b in bins], width=width * 0.86,
                   color=INK_2, linewidth=0)
        bottom.set_xlabel("Confidence (top probability)", color=INK_2, fontsize=9)
        bottom.set_ylabel("Rows", color=INK_2, fontsize=9)
        for ax in (top, bottom):
            ax.set_facecolor(SURFACE)
            ax.set_xlim(0, 1)
            ax.tick_params(colors=INK_2, labelsize=8)
            ax.grid(axis="y", color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
        top.set_ylim(0, 1)
    fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    if bins_json:
        write_json(path.with_suffix(".json"), tables)


# --- sections ------------------------------------------------------------------------------


def accuracy_section(groups, models, datasets, resamples):
    lines = ["## Accuracy on the evaluation split", "",
             "Raw scores (T = 1). Accuracy and macro-F1 do not depend on T (the argmax does not move), so "
             "they are given once. Normal generation has no probabilities: an invalid, hallucinated or missing "
             "answer counts wrong, and NLL, Brier and ECE do not apply. Salvaged accuracy (secondary): the first allowed "
             "value written for the question's key anywhere in the output, even when the JSON is invalid or cut off. "
             "Brackets: bootstrap 95% CI "
             f"({resamples} resamples of the evaluation rows). ECE is biased upward on small samples, "
             "and more so on resamples (they repeat rows), so its interval can sit above the point estimate. "
             "NLL clips the gold label's probability at 1e-12 (27.6 per row), which lowers the T = 1 values "
             "of the Qwen3 models.", ""]
    for ds in datasets:
        any_records = [r for (m, d, s, a), r in groups.items() if d == ds and s == "eval"]
        if not any_records:
            continue
        spec = data.DATASETS.get(ds)
        k = next((len(rs[0].get("scores") or rs[0]["probs"]) for rs in any_records if has_dist(rs)), "?")
        source = f"{spec.hf_id} {spec.eval_split}" if spec else ds
        lines += [f"### {ds} ({source}, {k} labels, n = {max(map(len, any_records))})", ""]
        rows = []
        for model in models:
            for arm in ordered({a for (m, d, s, a) in groups if m == model and d == ds and s == "eval"}, ARM_ORDER):
                records = groups[(model, ds, "eval", arm)]
                lat = ms(float(np.median([r["latency_ms"] for r in records])))
                if has_dist(records):
                    probs, gold = vectors(records)
                    m = metrics.scored(probs, gold, resamples)
                    rows.append([model, ARM_NAMES.get(arm, arm), len(records), ci(m["accuracy"]), ci(m["macro_f1"]),
                                 ci(m["nll"]), ci(m["brier"]), ci(m["ece15"]), f3(m["ece10"][0]), "n/a", "n/a", "n/a",
                                 lat])
                else:
                    pred, gold = generated_ids(records)
                    m = metrics.generated(pred, gold, resamples)
                    rows.append([model, ARM_NAMES.get(arm, arm), len(records), ci(m["accuracy"]), ci(m["macro_f1"]),
                                 "n/a", "n/a", "n/a", "n/a", f3(float(np.mean(pred >= 0))), salvaged(records),
                                 share(records, "cap_hit"), lat])
        lines += table(["Model", "Arm", "n", "Accuracy", "Macro-F1", "NLL", "Brier", "ECE (15 bins)", "ECE (10 bins)",
                        "Valid answers", "Salvaged accuracy", "Hit token cap", "Median ms"], rows)
    return lines


def share(records, key):
    """Share of records whose `key` is true; n/a for records written before the key existed."""
    return f3(float(np.mean([bool(r[key]) for r in records]))) if key in records[0] else "n/a"


def salvaged(records):
    """Accuracy when an allowed answer is read out of invalid output too (a secondary number)."""
    if "salvaged" not in records[0]:
        return "n/a"
    return f3(float(np.mean([r["salvaged"] == r["gold"] for r in records])))


def calibration_section(groups, temps, models, datasets, resamples, fig_dir, md_dir, public=False):
    """`public`: the figures leave bench/out, so only the MirethSTM1 arm gets one, without its bin table."""
    lines = ["## Calibration", "",
             "T is one scalar fitted by NLL on the calibration split (train rows, none of whose texts occur in "
             "the evaluation split): per dataset, and pooled over all datasets per model and arm. ECE and NLL are "
             "measured on the evaluation split only. n/a: the fit did not converge inside [0.05, 20] (too few "
             "calibration rows) or the split was not run. ECE brackets "
             "can sit above the point estimate (see Accuracy). Diagrams: reliability before (T = 1) and after "
             "the pooled T, the one that ships"
             + (", for the MirethSTM1 arm only (a report built without --figures keeps every arm's diagram and "
                "bin table in bench/out/<run>/report/)." if public else "; the bin tables sit next to them as JSON."),
             ""]
    rows = []
    for model in models:
        for arm in ARM_ORDER:
            for ds in datasets:
                records = groups.get((model, ds, "eval", arm))
                t = temps.get((model, arm), {})
                if not records or "scores" not in records[0]:
                    continue
                by_temp = {}
                cells = []
                for label, temp in (("T = 1", 1.0), (f"T = {f3(t.get(ds))} (dataset)", t.get(ds)),
                                    (f"T = {f3(t.get('pooled'))} (pooled)", t.get("pooled"))):
                    if temp is None:
                        cells += ["n/a", "n/a"]
                        continue
                    probs, gold = vectors(records, temp)
                    m = metrics.scored(probs, gold, resamples)
                    cells += [ci(m["ece15"]), f3(m["nll"][0])]
                    r = metrics.rows(probs, gold)
                    by_temp[label] = (r["conf"], r["correct"])
                link = "not published"
                if arm == "mireth" or not public:
                    figure = fig_dir / f"{slug(model)}.{ds}.{arm}.png"
                    # After: the pooled T (the one that ships), else the dataset's own T, else no second panel.
                    after = [label for label in by_temp if label != "T = 1"][-1:]
                    panels = [("Before: T = 1", *by_temp["T = 1"])] + [(f"After: {x}", *by_temp[x]) for x in after]
                    diagram(figure, f"{model}, {ds}, {ARM_NAMES.get(arm, arm)}", panels, bins_json=not public)
                    link = f"[png]({os.path.relpath(figure, md_dir).replace(os.sep, '/')})"
                rows.append([model, ARM_NAMES.get(arm, arm), ds, f3(t.get(ds)), f3(t.get("pooled")), *cells, link])
    lines += table(["Model", "Arm", "Dataset", "T dataset", "T pooled", "ECE-15 at T = 1", "NLL at T = 1",
                    "ECE-15 at T dataset", "NLL at T dataset", "ECE-15 at T pooled", "NLL at T pooled",
                    "Diagram"], rows)
    return lines


def yelp_section(groups, temps, models):
    rows = []
    for model in models:
        for arm in ARM_ORDER:
            records = groups.get((model, "yelp", "eval", arm))
            if not records:
                continue
            if has_dist(records):
                t = temps.get((model, arm), {}).get("pooled")
                probs, gold = vectors(records)
                arg = metrics.ordinal([int(np.argmax(p)) for p in probs], gold)
                ev = metrics.ordinal([metrics.expected_level(p) for p in probs], gold)
                within = np.mean(np.abs(np.array([int(np.argmax(p)) for p in probs]) - gold) <= 1)
                ev_t = metrics.ordinal([metrics.expected_level(p) for p in vectors(records, t)[0]], gold) if t else None
                rows.append([model, ARM_NAMES.get(arm, arm), len(records), f3(arg["mae"]), f3(arg["rmse"]),
                             f3(float(within)), f3(ev["mae"]), f3(ev["rmse"]),
                             f3(ev_t["mae"] if ev_t else None), f3(ev_t["rmse"] if ev_t else None)])
            else:
                valid = [r for r in records if r["pred"] is not None]
                if not valid:
                    continue
                lv, gv = [int(r["pred"]) for r in valid], [int(r["gold"]) for r in valid]
                arg = metrics.ordinal(lv, gv)
                within = np.mean(np.abs(np.array(lv) - np.array(gv)) <= 1)
                rows.append([model, ARM_NAMES.get(arm, arm), f"{len(valid)} valid of {len(records)}", f3(arg["mae"]),
                             f3(arg["rmse"]), f3(float(within)), "n/a", "n/a", "n/a", "n/a"])
    if not rows:
        return []
    return ["## Yelp: score levels", "",
            "Levels 0 to 4 (one to five stars). Argmax is the most likely level; the expected level is "
            "sum of k p_k. Normal generation is scored on its valid answers only.", ""] + table(
        ["Model", "Arm", "n", "MAE argmax", "RMSE argmax", "Within one level", "MAE expected", "RMSE expected",
         "MAE expected at T pooled", "RMSE expected at T pooled"], rows)


def bias_section(groups, models):
    rows = []
    for model in models:
        for ds, form in (("sst2", "yes/no"), ("sst2_choice", "choice")):
            positive = data.question(ds)[1][1]
            for arm in ARM_ORDER:
                records = groups.get((model, ds, "eval", arm))
                if not records:
                    continue
                if has_dist(records):
                    preds = [max(r.get("scores") or r["probs"], key=(r.get("scores") or r["probs"]).get)
                             for r in records]
                else:
                    preds = [r["pred"] for r in records]
                rows.append([model, ARM_NAMES.get(arm, arm), form, len(records),
                             f3(float(np.mean([p == r["gold"] for p, r in zip(preds, records)]))),
                             f3(float(np.mean([p == positive for p in preds]))),
                             f3(float(np.mean([r["gold"] == positive for r in records])))])
    if not rows:
        return []
    return ["## SST-2: yes-bias", "",
            "The same sentences as a yes/no question (\"Is the sentiment positive?\") and as a two-option "
            "choice. A predicted-positive rate well above the gold rate in the yes/no form only is a yes-bias.",
            ""] + table(["Model", "Arm", "Form", "n", "Accuracy", "Predicted positive", "Gold positive"], rows)


def mass_section(groups, models, datasets):
    rows = []
    for model in models:
        cells = []
        for ds in datasets:
            records = groups.get((model, ds, "eval", "mireth"))
            if not records or "scores" not in records[0]:
                cells.append("n/a")
                continue
            mass = np.array([np.exp(list(r["scores"].values())).sum() for r in records])
            cells.append(f"{np.median(mass):.3f} [{np.mean(mass < 0.01):.3f}]")
        if set(cells) != {"n/a"}:
            rows.append([model, *cells])
    if not rows:
        return []
    return ["## Probability on the allowed answers", "",
            "A label's score is the log-probability of its exact answer text, so exp(score) summed over a "
            "question's labels is the share of the model's probability that falls on an allowed answer at all. "
            "Where that share is small the model would rather write something else at that place (a quoted "
            "\"true\", a quoted number, other text), and the answer is read from the tail of its distribution: "
            "the ranking can still be right, but the scores sit far below 0, where bf16 rounding is coarser and "
            "exact ties become possible. MirethSTM1 arm, evaluation split: the median share over the rows and, "
            "in brackets, the share of rows where the labels hold under 1%.", ""] + table(
        ["Model", *datasets], rows)


def tie_section(groups, models, datasets):
    rows = []
    for model in models:
        for ds in datasets:
            records = groups.get((model, ds, "eval", "first_token"))
            if not records:
                continue
            tied_rows, tied_labels, decided, decided_right = 0, 0, 0, 0
            for r in records:
                values = list(r["scores"].values())
                counts = defaultdict(int)
                for v in values:
                    counts[v] += 1
                shared = sum(c for c in counts.values() if c > 1)
                tied_rows += shared > 0
                tied_labels += shared
                if counts[max(values)] > 1:
                    decided += 1
                    decided_right += max(r["scores"], key=r["scores"].get) == r["gold"]
            rows.append([model, shown(ds), len(records), tied_rows, f"{tied_labels / len(records):.1f}", decided,
                         f3(decided_right / decided if decided else None)])
    if not rows:
        return []
    return ["## First-token ties", "",
            "The first-token arm scores only each label's first token, so options whose first tokens are the "
            "same token get the same score. A tie goes to the earlier option in the request, as the argmax of the "
            "original demo's PyTorch path does (its Mac build generates a few tokens when first tokens collide; "
            "this arm does not), and the tied options split the probability. Labels in ties: mean per row of "
            "options whose score equals another option's, which counts shared first tokens and also chance ties "
            "on the bf16 grid.", ""] + table(
        ["Model", "Dataset", "n", "Rows with a tie", "Labels in ties", "Predictions decided by the tie rule",
         "Accuracy of those"], rows)


def order_section(groups, models, datasets, resamples):
    rows = []
    for model in models:
        for ds in datasets:
            a, b = groups.get((model, ds, "eval", "mireth")), groups.get((model, ds, "eval", "questions_first"))
            if not a or not b:
                continue
            by_index = {r["index"]: r for r in b}
            pairs = [(r, by_index[r["index"]]) for r in a if r["index"] in by_index]
            ma = metrics.scored(*vectors([p for p, _ in pairs]), resamples)
            mb = metrics.scored(*vectors([q for _, q in pairs]), resamples)
            agree = np.mean([max(p["scores"], key=p["scores"].get) == max(q["scores"], key=q["scores"].get)
                             for p, q in pairs])
            rows.append([model, ds, len(pairs), ci(ma["accuracy"]), ci(mb["accuracy"]), f3(ma["ece15"][0]),
                         f3(mb["ece15"][0]), f3(float(agree))])
    if not rows:
        return []
    return ["## Question order (SPEC 3.1)", "",
            "The state before the questions (the shipped prompt) against the questions before the state (the "
            "order that allowed a question-part cache, dropped for accuracy), same rows, raw scores.", ""] + table(
        ["Model", "Dataset", "n", "Accuracy, state first", "Accuracy, questions first", "ECE-15, state first",
         "ECE-15, questions first", "Same answer"], rows)


def multifield_section(many, meta, models, exact=None):
    """`exact`: the same run's records in fp32 ({model: {mode: records}}), shown under each model's table."""
    lines = []
    for model in models:
        for name, by_mode in ((dtype_of(meta, model), many.get(model)), ("fp32", (exact or {}).get(model))):
            if by_mode:
                header, rows = multifield.table(by_mode)
                n = max(len(records) for records in by_mode.values())
                lines += [f"### {model}{', ' + name if name else ''} (AG News test, n = {n})", ""] + table(header, rows)
    if not lines:
        return []
    return ["## Many questions in one call (SPEC 3.1)", "",
            "Five checked questions (the topic, and four yes/no questions that follow from the gold topic) asked "
            "alone, one call each, and inside one 20-question call next to 15 filler yes/no questions, at three "
            "placements. Every question has a branch of its own, so a question scores the same wherever it sits; "
            "what is left is float rounding (the largest score difference, in summed log-prob, over every label). "
            "Brackets: share of articles answered yes (the true share is about a quarter). A model whose fp32 "
            "weights fit the card has the same run in fp32 under its table. The earlier engine put all questions "
            "into one shared prompt; on Qwen/Qwen2.5-1.5B-Instruct the same test then gave 0.864 alone, 0.809 "
            "first, 0.575 last and 0.500 spread.", ""] + lines


def board_section(groups, temps, models, resamples):
    rows = []
    for model in models:
        for arm in ARM_ORDER:
            for tier in BOARD_TIERS:
                records = groups.get((model, f"jevbench_{tier}", "eval", arm))
                if not records:
                    continue
                if has_dist(records):
                    probs, gold = vectors(records)
                    m = metrics.scored(probs, gold, resamples)
                    acc, extra = m["accuracy"], [f3(m["ece10"][0]), f3(m["ece15"][0]), f3(m["nll"][0]),
                                                 f3(m["brier"][0])]
                    t = temps.get((model, arm), {}).get("pooled") if "scores" in records[0] else None
                    if t:
                        r = metrics.rows(*vectors(records, t))
                        extra.insert(1, f"{ece(r['conf'], r['correct'], n_bins=10):.3f} (T = {t:.3f})")
                    else:
                        extra.insert(1, "n/a")
                else:
                    pred, gold = generated_ids(records)
                    acc = metrics.generated(pred, gold, resamples)["accuracy"]
                    extra = ["n/a"] * 5
                chance = float(np.mean([1 / r["n_labels"] for r in records]))
                corrected = max(0.0, (acc[0] - chance) / (1 - chance)) * 100
                rows.append([model, ARM_NAMES.get(arm, arm), tier, len(records), ci(acc), f3(chance),
                             f"{corrected:.1f}", *extra])
    if not rows:
        return []
    return ["## JevBench public items", "",
            "The 231 public items of JevBench (github.com/fstandhartinger/jevbench, MIT), fetched at run time, "
            "scored here by us. Not a JevBench score: its board adds sealed items, a speed and a cost axis and "
            "runs entrants on its own hardware. Chance: mean of 1/options over the items run; chance-corrected "
            "accuracy is 100 (accuracy - chance) / (1 - chance), floored at 0, as JevBench's Intelligence axis per "
            "tier. ECE-10 matches JevBench's binning. The pooled T comes from our own calibration splits, never "
            "from JevBench items.", ""] + table(
        ["Model", "Arm", "Tier", "n", "Accuracy", "Chance", "Chance-corrected", "ECE-10 at T = 1",
         "ECE-10 at T pooled", "ECE-15 at T = 1", "NLL", "Brier"], rows)


def latency_section(latency, meta, models, resamples):
    lines = []
    entries = {e["model"]: e for e in meta.get("latency", [])}
    for model in models:
        records = latency.get(model)
        if not records:
            continue
        e = entries.get(model, {})
        hw = e.get("hardware_start", {})
        lines += [f"### {model}", "",
                  f"Device {e.get('device', '?')}, {e.get('dtype', '?')}, {hw.get('gpu') or hw.get('cpu')}, "
                  f"torch {hw.get('torch')}, CUDA {hw.get('cuda')}, transformers {hw.get('transformers')}.",
                  "nvidia-smi (name, driver, temperature C, core clock, memory clock, power) at start: "
                  f"{'; '.join(hw.get('nvidia_smi') or ['n/a'])}; at end: "
                  f"{'; '.join(e.get('hardware_end', {}).get('nvidia_smi') or ['n/a'])}.",
                  ""]
        by_setting = defaultdict(lambda: defaultdict(list))
        rank = {}  # field counts in order, then the scenarios in the console's order
        for r in records:
            rank[r["setting"]] = (0, r["fields"]) if r["scenario"] is None else (1, SCENARIO_ORDER.index(r["scenario"]))
            by_setting[r["setting"]][r["arm"]].append(r)
        rows = []
        for setting in sorted(rank, key=rank.get):
            arms = by_setting[setting]
            m, b = arms.get("mireth", []), arms.get("baseline", [])
            cells = [setting, (m or b)[0]["fields"], len(m or b)]
            p50 = {}
            for arm in ("mireth", "baseline"):
                rs = arms.get(arm)
                if not rs:
                    cells += ["n/a", "n/a"]
                    continue
                x = np.array([r["latency_ms"] for r in rs])
                p50[arm] = float(np.percentile(x, 50))
                for q in (50, 95):
                    value = float(np.percentile(x, q))
                    lo, hi = (metrics.bootstrap_ci(lambda i: np.percentile(x[i], q), len(x), resamples)
                              if resamples else (None, None))
                    cells.append(ms(value) + (f" [{ms(lo)}, {ms(hi)}]" if lo is not None else ""))
            cells.append(f"{p50['baseline'] / p50['mireth']:.1f}x" if len(p50) == 2 else "n/a")
            cells.append(f"{np.mean([r['input_tokens'] for r in m]):.0f}" if m else "n/a")
            cells.append(f"{np.mean([r['output_tokens'] for r in b]):.0f}" if b else "n/a")
            right = []
            for rs in (m, b):
                checked = [r for r in rs if "correct" in r]
                right.append(f"{sum(r['correct'] for r in checked) / sum(r['fields'] for r in checked):.2f}"
                             if checked else "n/a")
            cells.append(" / ".join(right))
            cells.append(f"{np.mean([r['hallucinated'] + r['missing'] for r in b]):.1f}" if b else "n/a")
            peaks = [[r["peak_mem_mb"] for r in rs if r.get("peak_mem_mb") is not None] for rs in (m, b)]
            cells.append(" / ".join(f"{max(x):.0f}" if x else "n/a" for x in peaks))
            rows.append(cells)
        lines += table(["Setting", "Fields", "Runs", "MirethSTM1 p50 ms", "MirethSTM1 p95 ms",
                        "Normal generation p50 ms", "Normal generation p95 ms", "Speedup at p50",
                        "MirethSTM1 input tokens", "Generated tokens", "Checked answers right (MirethSTM1 / generation)",
                        "Generation: bad fields per run", "Peak memory MiB (MirethSTM1 / generation)"], rows)
    if not lines:
        return []
    return ["## Latency", "",
            "Batch 1, one request at a time, model loaded, after the warmup runs; each timed span is one "
            "`Engine.decide` or `baseline.generate` call with torch.cuda.synchronize() at both ends on CUDA. "
            "Normal generation is the same loaded model writing every answer as one JSON object with Hugging Face "
            "`generate` (greedy), so each speedup is against that library's decoding speed on this machine (the "
            "rate is in the Summary), not against an optimized inference server. "
            "p50 and p95 with linear interpolation; brackets: bootstrap 95% CI. With few runs p95 rests on the "
            "slowest one or two runs. Field-count settings use a "
            "different AG News article per run (topic choice plus rule-checked yes/no questions); scenarios "
            "repeat their own text. Checked answers: share of fields whose answer matches the gold topic or the "
            "rule. Bad fields: hallucinated or missing answers per generation run. Peak memory: the largest "
            "`torch.cuda.max_memory_allocated` of a timed run of that arm.", ""] + lines


def _counts(by_dataset):
    """'1000 (sst2: 872)' from {dataset: rows}: the largest count, then every dataset that has fewer."""
    most = max(by_dataset.values())
    fewer = ", ".join(f"{ds}: {n}" for ds, n in by_dataset.items() if n != most)
    return f"{most} ({fewer})" if fewer else str(most)


def header(run_dir, groups, latency, meta, models, datasets):
    """The title, then what every model actually ran, counted from the records themselves."""
    lines = [f"# MirethSTM1 benchmark: {run_dir.name}", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by `python -m bench.report` from "
             f"bench/out/{run_dir.name}/ (per-sample JSONL). Models: {', '.join(models) or 'none'}.", ""]
    entries = meta.get("run_bench", [])
    if entries:
        hw = entries[-1]["hardware"]
        lines += [f"Accuracy runs: seed {', '.join(str(x) for x in sorted({e['args']['seed'] for e in entries}))}; "
                  f"{hw.get('gpu') or hw.get('cpu')}, torch {hw.get('torch')}, transformers {hw.get('transformers')}. "
                  f"Samples are stratified by class, and a smaller sample is a subset of a larger one (same seed). "
                  f"Texts longer than {data.MAX_CHARS} characters are cut at a word boundary (only Yelp).", ""]
    for model in models:
        days = sorted({e[key]["time_utc"][:10] for name, key in (("run_bench", "hardware"), ("latency", "hardware_start"))
                       for e in meta.get(name, []) if model in e.get("models", [e.get("model")])})
        parts = [dtype_of(meta, model, "dtype not recorded") + (f", {' and '.join(days)}" if days else "")]
        rows = {split: {ds: len(groups[(model, ds, split, "mireth")]) for ds in datasets
                        if (model, ds, split, "mireth") in groups} for split in ("eval", "cal")}
        if rows["eval"]:
            parts.append(f"scoring arms on {_counts(rows['eval'])} evaluation rows per dataset"
                         + (f" and {_counts(rows['cal'])} calibration rows" if rows["cal"] else ""))
        generated = {ds: len(groups[(model, ds, "eval", "baseline")]) for ds in datasets
                     if (model, ds, "eval", "baseline") in groups}
        if generated:
            arms_run = [e["args"]["arms"].split(",") for e in entries if model in e["models"]]
            copied = arms_run and not any("baseline" in arms for arms in arms_run)
            note = " (files copied in from an earlier run: meta.json records no run of that arm)" if copied else ""
            parts.append(f"normal generation on {_counts(generated)} evaluation rows{note}")
        timed = defaultdict(set)
        for r in latency.get(model, []):
            if r["arm"] == "mireth":
                timed[r["scenario"] is not None].add(r["run"])
        if timed:
            parts.append("latency over " + " and ".join(
                f"{len(timed[k])} timed runs per {what}" for k, what in ((False, "field count"), (True, "scenario"))
                if k in timed))
        lines.append(f"- {model}: {'; '.join(parts)}.")
    return lines + [""]


def summary_section(groups, latency, temps, models, datasets):
    """One row per model: means over the datasets, the rows every model ran, p50 latency."""
    means, shared_rows, speed, slow = [], [], [], []
    have = [m for m in models if datasets and all((m, ds, "eval", "mireth") in groups for ds in datasets)]
    common = {ds: set.intersection(*({r["index"] for r in groups[(m, ds, "eval", "mireth")]} for m in have))
              for ds in datasets} if have else {}
    for model in have:
        arm = {a: [groups.get((model, ds, "eval", a)) for ds in datasets] for a in ("mireth", "first_token", "baseline")}
        t = {a: temps.get((model, a), {}).get("pooled") for a in arm}

        def mean(a, f):
            return f3(float(np.mean([f(rs) for rs in arm[a]]))) if all(arm[a]) else "n/a"

        def at_t(a):  # at T rounded to three decimals, the value that ships
            return mean(a, lambda rs: ece15(rs, round(t[a], 3))) if t[a] else "n/a"

        cells = [model, f"{max(map(len, arm['mireth']))} / "
                 + (str(max(map(len, arm["baseline"]))) if all(arm["baseline"]) else "n/a"),
                 mean("mireth", accuracy), mean("first_token", accuracy)]
        if all(arm["baseline"]):
            valid = float(np.mean([np.mean([r["pred"] is not None for r in rs]) for rs in arm["baseline"]]))
            same = [[r for r in rs if r["index"] in {b["index"] for b in bs}]
                    for rs, bs in zip(arm["mireth"], arm["baseline"])]
            cells += [f"{mean('baseline', accuracy)} ({f3(valid)})", f3(float(np.mean([accuracy(rs) for rs in same])))]
        else:
            cells += ["n/a", "n/a"]
        cells += [mean("mireth", ece15), at_t("mireth"), f3(t["mireth"]),
                  f"{mean('first_token', ece15)} / {at_t('first_token')}"]
        means.append(cells)
        if len(have) > 1:
            pairs = [(accuracy([r for r in rs if r["index"] in common[ds]]),
                      accuracy([r for r in fs if r["index"] in common[ds]]) if fs else None)
                     for ds, rs, fs in zip(datasets, arm["mireth"], arm["first_token"])]
            pairs.append((float(np.mean([a for a, _ in pairs])),
                          float(np.mean([b for _, b in pairs])) if all(b is not None for _, b in pairs) else None))
            shared_rows.append([model] + [f"{f3(a)} / {f3(b)}" for a, b in pairs])
    lines = ["## Summary across models", "",
             "Means over the datasets (" + ", ".join(datasets) + "), each weighted equally, evaluation split. "
             "Rows: the largest per-dataset row count of the scoring arms and of normal generation; models run on "
             "different row counts are compared on their shared rows in the second table. MirethSTM1 on the "
             "generation rows scores MirethSTM1 on exactly the rows normal generation ran. Normal generation counts "
             "an invalid, hallucinated or missing answer as wrong and is strict about types (a quoted \"true\" is "
             "not a boolean). ECE-15 is the mean of the per-dataset values; at the pooled T it is computed at T "
             "rounded to three decimals, the value that ships in `mirethstm.calibration.DEFAULT_TEMPERATURES`.", ""]
    lines += table(["Model", "Rows: scoring / generation", "MirethSTM1 accuracy", "First-token accuracy",
                    "Normal generation accuracy (valid answers)", "MirethSTM1 on the generation rows",
                    "MirethSTM1 ECE-15 at T = 1", "MirethSTM1 ECE-15 at pooled T", "Pooled T",
                    "First token ECE-15 at T = 1 / at its pooled T"], means)
    if shared_rows:
        lines += ["Accuracy on the evaluation rows that every model ran ("
                  + ", ".join(f"{ds}: {len(common[ds])}" for ds in datasets) + "), MirethSTM1 / first token:", ""]
        lines += table(["Model", *datasets, "Mean"], shared_rows)
    rank, names = {}, {}
    for model in models:
        for r in latency.get(model, []):
            key = r["scenario"] or f"f{r['fields']:02d}"
            rank[key] = (0, r["fields"]) if r["scenario"] is None else (1, SCENARIO_ORDER.index(r["scenario"]))
            names[key] = r["setting"]
    settings = sorted(rank, key=rank.get)
    for model in models:
        by = defaultdict(lambda: defaultdict(list))
        for r in latency.get(model, []):
            by[r["arm"]][r["scenario"] or f"f{r['fields']:02d}"].append(r)
        for arm, out in (("mireth", speed), ("baseline", slow)):
            if not by[arm]:
                continue
            cells = [model] + [ms(float(np.percentile([r["latency_ms"] for r in by[arm][k]], 50))) if by[arm][k]
                               else "n/a" for k in settings]
            runs = [r for rs in by[arm].values() for r in rs]
            if arm == "mireth":
                peaks = [r["peak_mem_mb"] for r in runs if r.get("peak_mem_mb") is not None]
                cells.append(f"{max(peaks):.0f}" if peaks else "n/a")
            else:
                cells.append(f"{sum(r['output_tokens'] for r in runs) / sum(r['latency_ms'] for r in runs) * 1000:.1f}")
            out.append(cells)
    if speed:
        lines += ["MirethSTM1 p50 latency in ms (warm, batch 1; full tables in Latency below). Peak memory: the "
                  "largest `torch.cuda.max_memory_allocated` of any MirethSTM1 run of the model.", ""]
        lines += table(["Model", *(names[k] for k in settings), "Peak memory MiB"], speed)
    if slow:
        lines += ["Normal generation p50 latency in ms on the same settings: the same loaded model writing the "
                  "answers as one JSON object with Hugging Face `generate` (greedy). Tokens per second: generated "
                  "tokens over wall time, summed over every timed run (the prompt pass included). Every speedup in "
                  "this report is against this decoding speed.", ""]
        lines += table(["Model", *(names[k] for k in settings), "Generated tokens per second"], slow)
    return lines if means or speed or slow else []


def precision_section(run_dir, groups, meta, models, datasets, others):
    """The MirethSTM1 arm of `groups` next to the same rows in other dtypes; `others`: {dtype: groups}."""
    lines = []
    for model in models:
        runs = {dtype_of(meta, model, "bf16"): groups}
        runs.update({name: g for name, g in others.items() if any(k[0] == model and k[3] == "mireth" for k in g)})
        if "fp32" not in runs or len(runs) < 2:
            continue
        names = list(runs)
        rest = [name for name in names if name != "fp32"]

        def rows(name, ds, split):
            return {r["index"]: r for r in runs[name].get((model, ds, split, "mireth"), [])}

        used = [ds for ds in datasets if all(rows(name, ds, "eval") for name in names)]
        both = {(ds, split): set.intersection(*(set(rows(name, ds, split)) for name in names))
                for ds in used for split in ("eval", "cal")}

        def on(name, ds, split):
            return [r for i, r in rows(name, ds, split).items() if i in both[(ds, split)]]

        t = {name: fit3([r for ds in used for r in on(name, ds, "cal")]) for name in names}
        body = []
        for ds in used:
            ev = {name: on(name, ds, "eval") for name in names}
            ref = {r["index"]: r for r in ev["fp32"]}
            gaps = {name: [abs(v - ref[r["index"]]["scores"][label])
                           for r in ev[name] for label, v in r["scores"].items()] for name in rest}
            same = {name: float(np.mean([top(r) == top(ref[r["index"]]) for r in ev[name]])) for name in rest}
            body.append([ds, len(ref), " / ".join(f3(accuracy(ev[name])) for name in names),
                         " / ".join(f3(ece15(ev[name])) for name in names),
                         " / ".join(f3(ece15(ev[name], t[name])) if t[name] else "n/a" for name in names),
                         " / ".join(f3(same[name]) for name in rest),
                         " / ".join(f"{np.mean(gaps[name]):.3f} ({max(gaps[name]):.2f})" for name in rest),
                         " / ".join(ms(float(np.median([r["latency_ms"] for r in ev[name]]))) for name in names)])
        cal = _counts({ds: len(both[(ds, "cal")]) for ds in used})
        lines += [f"### {model}", "",
                  f"Pooled T, each dtype fitted on the same {cal} calibration rows per dataset: "
                  + ", ".join(f"{name} {f3(t[name])}" for name in names) + ".", ""]
        lines += table(["Dataset", "n", "Accuracy " + " / ".join(names), "ECE-15 at T = 1", "ECE-15 at own pooled T",
                        "Same answer as fp32: " + " / ".join(rest),
                        "Mean (max) abs score difference from fp32: " + " / ".join(rest),
                        "Median ms " + " / ".join(names)], body)
    if not lines:
        return []
    return ["## Precision: the same rows in other dtypes", "",
            f"The MirethSTM1 arm on the evaluation rows that this run and bench/out/{run_dir.name}-<dtype>/ both "
            "hold. Agreement and score differences are against fp32 on the same rows, over every label's summed "
            "log-prob. Median ms is the per-row time of the accuracy run (one question per call).", ""] + lines


def build(run_dir, md_path, resamples=metrics.N_RESAMPLES, fig_dir=None):
    groups, latency, many, meta = load(run_dir)
    others = {name: load(run_dir.with_name(f"{run_dir.name}-{name}")) for name in PRECISION_RUNS
              if run_dir.with_name(f"{run_dir.name}-{name}").is_dir()}
    models = []
    for entry in meta.get("run_bench", []):
        models += [m for m in entry.get("models", []) if m not in models]
    models += sorted({k[0] for k in groups} - set(models))
    models += sorted((set(latency) | set(many)) - set(models))
    present = {k[1] for k in groups}
    datasets = [d for d in data.DATASETS if d in present]
    temps = temperatures(groups)
    md_dir = md_path.parent
    lines = header(run_dir, groups, latency, meta, models, datasets)
    lines += summary_section(groups, latency, temps, models, datasets)
    lines += accuracy_section(groups, models, datasets, resamples)
    lines += calibration_section(groups, temps, models, datasets, resamples, fig_dir or run_dir / "report", md_dir,
                                 public=fig_dir is not None)
    lines += yelp_section(groups, temps, models)
    lines += bias_section(groups, models)
    lines += mass_section(groups, models, datasets)
    lines += tie_section(groups, models, datasets + [f"jevbench_{t}" for t in BOARD_TIERS])
    lines += order_section(groups, models, datasets, resamples)
    lines += multifield_section(many, meta, models, others["fp32"][2] if "fp32" in others else None)
    lines += board_section(groups, temps, models, resamples)
    lines += latency_section(latency, meta, models, resamples)
    lines += precision_section(run_dir, groups, meta, models, datasets, {name: o[0] for name, o in others.items()})
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run name under bench/out/")
    parser.add_argument("--md", default=None, help="markdown path (default: bench/out/<run>/benchmark.md)")
    parser.add_argument("--resamples", type=int, default=metrics.N_RESAMPLES, help="bootstrap resamples, 0: no CIs")
    parser.add_argument("--figures", default=None,
                        help="folder next to --md for a report that leaves bench/out/: only the MirethSTM1 arm's "
                             "diagrams go there, without bin tables (default: every arm's diagram and bin table "
                             "in bench/out/<run>/report)")
    args = parser.parse_args(argv)
    run_dir = OUT / args.run
    if not run_dir.is_dir():
        raise SystemExit(f"no run folder bench/out/{args.run}")
    md = build(run_dir, Path(args.md) if args.md else run_dir / "benchmark.md", args.resamples,
               Path(args.figures) if args.figures else None)
    print(f"wrote  {md}")


if __name__ == "__main__":
    main()
