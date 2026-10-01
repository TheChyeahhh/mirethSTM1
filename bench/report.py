"""Benchmark tables and reliability diagrams from saved runs; never loads a model.

    python -m bench.report --run dry-run [--md docs/benchmark.md --figures docs/benchmark]

Reads every bench/out/<run>/<model>/*.jsonl and writes one markdown report (the layout of
docs/benchmark.md) plus reliability diagrams (PNG) and their bin tables (JSON) in
bench/out/<run>/report/. Temperature is fitted on the calibration split, per model and arm:
once pooled over the datasets and once per dataset; every number after T is measured on the
evaluation split only.
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

from . import data, metrics
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


# --- loading -----------------------------------------------------------------------------


def load(run_dir):
    """({(model, dataset, split, arm): records}, {model: latency records}, meta)."""
    groups, latency = defaultdict(list), defaultdict(list)
    for path in sorted(run_dir.glob("*/*.jsonl")):
        for r in read_jsonl(path):
            if path.name.startswith("latency."):
                latency[r["model"]].append(r)
            else:
                groups[(r["model"], r["dataset"], r["split"], r["arm"])].append(r)
    meta_path = run_dir / "meta.json"
    return groups, latency, read_json(meta_path) if meta_path.exists() else {}


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


def temperatures(groups):
    """{(model, arm): {"pooled": T, dataset: T}} from the calibration splits."""
    cal = defaultdict(dict)
    for (model, ds, split, arm), records in groups.items():
        if split == "cal":
            cal[(model, arm)][ds] = records
    out = {}
    for key, by_ds in cal.items():
        out[key] = {"pooled": fit([r for records in by_ds.values() for r in records])}
        out[key].update({ds: fit(records) for ds, records in by_ds.items()})
    return out


def generated_ids(records):
    """Gold and predicted ints for generated answers; an invalid answer is -1."""
    names = sorted({r["gold"] for r in records} | {r["pred"] for r in records if r["pred"] is not None})
    index = {name: i for i, name in enumerate(names)}
    gold = np.array([index[r["gold"]] for r in records])
    pred = np.array([-1 if r["pred"] is None else index[r["pred"]] for r in records])
    return pred, gold


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


def diagram(path, title, panels):
    """Reliability diagram, one column per (subtitle, confidences, correct): accuracy per
    15-bin confidence bin against the diagonal, with the bin counts below."""
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
             "and more so on resamples (they repeat rows), so its interval can sit above the point estimate.", ""]
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


def calibration_section(groups, temps, models, datasets, resamples, fig_dir, md_dir):
    lines = ["## Calibration", "",
             "T is one scalar fitted by NLL on the calibration split (train rows, none of whose texts occur in "
             "the evaluation split): per dataset, and pooled over all datasets per model and arm. ECE and NLL are "
             "measured on the evaluation split only. n/a: the fit did not converge inside [0.05, 20] (too few "
             "calibration rows) or the split was not run. ECE brackets "
             "can sit above the point estimate (see Accuracy).", ""]
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
                figure = fig_dir / f"{slug(model)}.{ds}.{arm}.png"
                # After: the dataset's own T, else the pooled T, else no second panel.
                panels = [(f"Before: {label}" if label == "T = 1" else f"After: {label}", *conf_correct)
                          for label, conf_correct in by_temp.items()][:2]
                diagram(figure, f"{model}, {ds}, {ARM_NAMES.get(arm, arm)}", panels)
                link = os.path.relpath(figure, md_dir).replace(os.sep, "/")
                rows.append([model, ARM_NAMES.get(arm, arm), ds, f3(t.get(ds)), f3(t.get("pooled")), *cells,
                             f"[png]({link})"])
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
                t = temps.get((model, arm), {}).get("yelp")
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
         "MAE expected at T", "RMSE expected at T"], rows)


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
            rows.append([model, ds, len(records), tied_rows, f"{tied_labels / len(records):.1f}", decided,
                         f3(decided_right / decided if decided else None)])
    if not rows:
        return []
    return ["## First-token ties", "",
            "The first-token arm scores only each label's first token, so options whose first tokens are the "
            "same token get the same score. A tie goes to the earlier option in the request (as the original "
            "demo's argmax does), and the tied options split the probability. Labels in ties: mean per row of "
            "options sharing their first token with another option.", ""] + table(
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
                        extra.insert(1, f"{ece(r['conf'], r['correct'], n_bins=10):.3f} (T = {t:.2f})")
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
                  f"nvidia-smi at start: {hw.get('nvidia_smi')}; at end: {e.get('hardware_end', {}).get('nvidia_smi')}.",
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
            peaks = [r["peak_mem_mb"] for rs in (m, b) for r in rs if r.get("peak_mem_mb") is not None]
            cells.append(f"{max(peaks):.0f}" if peaks else "n/a")
            rows.append(cells)
        lines += table(["Setting", "Fields", "Runs", "MirethSTM1 p50 ms", "MirethSTM1 p95 ms",
                        "Normal generation p50 ms", "Normal generation p95 ms", "Speedup at p50",
                        "MirethSTM1 input tokens", "Generated tokens", "Checked answers right (MirethSTM1 / generation)",
                        "Generation: bad fields per run", "Peak memory MB"], rows)
    if not lines:
        return []
    return ["## Latency", "",
            "Batch 1, one request at a time, model loaded, after the warmup runs; each timed span is one "
            "`Engine.decide` or `baseline.generate` call with torch.cuda.synchronize() at both ends on CUDA. "
            "p50 and p95 with linear interpolation; brackets: bootstrap 95% CI. With few runs p95 rests on the "
            "slowest one or two runs. Field-count settings use a "
            "different AG News article per run (topic choice plus rule-checked yes/no questions); scenarios "
            "repeat their own text. Checked answers: share of fields whose answer matches the gold topic or the "
            "rule. Bad fields: hallucinated or missing answers per generation run.", ""] + lines


def build(run_dir, md_path, resamples=metrics.N_RESAMPLES, fig_dir=None):
    groups, latency, meta = load(run_dir)
    models = []
    for entry in meta.get("run_bench", []):
        models += [m for m in entry.get("models", []) if m not in models]
    models += sorted({k[0] for k in groups} - set(models)) + sorted(set(latency) - set(models) - {k[0] for k in groups})
    present = {k[1] for k in groups}
    datasets = [d for d in data.DATASETS if d in present]
    temps = temperatures(groups)
    md_dir = md_path.parent
    lines = [f"# MirethSTM1 benchmark: {run_dir.name}", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by `python -m bench.report` from "
             f"bench/out/{run_dir.name}/ (per-sample JSONL). Models: {', '.join(models) or 'none'}.", ""]
    for entry in meta.get("run_bench", [])[-1:]:
        hw, args = entry["hardware"], entry["args"]
        lines += [f"Accuracy runs: n = {args['n']} evaluation and {args['n_cal']} calibration rows per dataset "
                  f"(fewer where the split is smaller), seed {args['seed']}, dtype {args['dtype'] or 'default'}, "
                  f"device {args['device'] or 'default'}; {hw.get('gpu') or hw.get('cpu')}, torch {hw.get('torch')}, "
                  f"transformers {hw.get('transformers')}. Samples are stratified by class; texts longer than "
                  f"{data.MAX_CHARS} characters are cut at a word boundary (only Yelp).", ""]
    lines += accuracy_section(groups, models, datasets, resamples)
    lines += calibration_section(groups, temps, models, datasets, resamples, fig_dir or run_dir / "report", md_dir)
    lines += yelp_section(groups, temps, models)
    lines += bias_section(groups, models)
    lines += tie_section(groups, models, datasets + [f"jevbench_{t}" for t in BOARD_TIERS])
    lines += order_section(groups, models, datasets, resamples)
    lines += board_section(groups, temps, models, resamples)
    lines += latency_section(latency, meta, models, resamples)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run name under bench/out/")
    parser.add_argument("--md", default=None, help="markdown path (default: bench/out/<run>/benchmark.md)")
    parser.add_argument("--resamples", type=int, default=metrics.N_RESAMPLES, help="bootstrap resamples, 0: no CIs")
    parser.add_argument("--figures", default=None,
                        help="folder for the diagrams and bin tables (default: bench/out/<run>/report); "
                             "put it next to --md when the report leaves bench/out/")
    args = parser.parse_args(argv)
    run_dir = OUT / args.run
    if not run_dir.is_dir():
        raise SystemExit(f"no run folder bench/out/{args.run}")
    md = build(run_dir, Path(args.md) if args.md else run_dir / "benchmark.md", args.resamples,
               Path(args.figures) if args.figures else None)
    print(f"wrote  {md}")


if __name__ == "__main__":
    main()
