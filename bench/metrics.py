"""Benchmark metrics (docs/research/07 section 2). Pure numpy; rows may have different label counts.

Notation: per row a probability vector p (sums to 1), the gold index y, the prediction
argmax p (ties go to the earlier label) and the confidence max p.
"""

import numpy as np

from mirethstm.calibration import ece

N_RESAMPLES = 1000


def softmax(scores, t=1.0):
    """softmax(s / t) of one score vector."""
    s = np.asarray(scores, dtype=float) / t
    e = np.exp(s - s.max())
    return e / e.sum()


def rows(probs, gold):
    """Per-row arrays: pred, conf, correct, nll (p clipped at 1e-12), brier (sum over classes)."""
    pred = np.array([int(np.argmax(p)) for p in probs])
    gold = np.asarray(gold, dtype=int)
    p_gold = np.array([p[y] for p, y in zip(probs, gold)])
    return {
        "pred": pred,
        "conf": np.array([float(np.max(p)) for p in probs]),
        "correct": pred == gold,
        "nll": -np.log(np.clip(p_gold, 1e-12, None)),
        # sum_k (p_k - [k = y])^2 = sum_k p_k^2 - 2 p_y + 1
        "brier": np.array([float(np.dot(p, p)) for p in probs]) - 2 * p_gold + 1,
    }


def macro_f1(pred, gold):
    """Mean F1 over every class that occurs in gold or among the valid predictions.

    pred -1 marks an invalid answer: wrong for its row, and a prediction of no class, so it
    lowers the recall of the gold class only. F1 = 2 TP / (support + predicted).
    """
    pred, gold = np.asarray(pred, dtype=int), np.asarray(gold, dtype=int)
    k = int(max(gold.max(), pred.max())) + 1
    full = np.bincount(gold * (k + 1) + pred + 1, minlength=k * (k + 1)).reshape(k, k + 1)  # column 0: invalid
    cm = full[:, 1:]
    support, predicted = full.sum(axis=1), cm.sum(axis=0)
    present = (support + predicted) > 0
    return float(np.mean(2 * np.diag(cm)[present] / (support + predicted)[present]))


def bootstrap_ci(stat, n, n_resamples=N_RESAMPLES, seed=0):
    """95% percentile interval of stat(row indices) over resamples of the n rows with replacement."""
    rng = np.random.default_rng(seed)
    values = [stat(rng.integers(0, n, n)) for _ in range(n_resamples)]
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def reliability_bins(conf, correct, n_bins=15):
    """The bins calibration.ece sums over: ((m-1)/M, m/M], 0 joining the first bin."""
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(conf, edges[1:-1], right=True)
    out = []
    for b in range(n_bins):
        m = idx == b
        out.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()),
                    "conf": float(conf[m].mean()) if m.any() else None,
                    "acc": float(correct[m].mean()) if m.any() else None})
    return out


def expected_level(p):
    """sum_k k p_k over levels 0..K-1."""
    return float(np.dot(np.arange(len(p)), p))


def _with_ci(stats, n, n_resamples):
    whole = np.arange(n)
    return {name: (float(f(whole)), *(bootstrap_ci(f, n, n_resamples) if n_resamples else (None, None)))
            for name, f in stats.items()}


def scored(probs, gold, n_resamples=N_RESAMPLES):
    """Every metric of a scoring arm with its bootstrap 95% CI: {name: (value, lo, hi)}; no CI when n_resamples is 0."""
    r = rows(probs, gold)
    gold = np.asarray(gold, dtype=int)
    n = len(gold)
    stats = {
        "accuracy": lambda i: r["correct"][i].mean(),
        "macro_f1": lambda i: macro_f1(r["pred"][i], gold[i]),
        "nll": lambda i: r["nll"][i].mean(),
        "brier": lambda i: r["brier"][i].mean(),
        "ece15": lambda i: ece(r["conf"][i], r["correct"][i], n_bins=15),
        "ece10": lambda i: ece(r["conf"][i], r["correct"][i], n_bins=10),
    }
    return _with_ci(stats, n, n_resamples)


def generated(pred, gold, n_resamples=N_RESAMPLES):
    """Accuracy and macro-F1 of generated answers (pred -1 = invalid, counted wrong), with CIs."""
    pred, gold = np.asarray(pred, dtype=int), np.asarray(gold, dtype=int)
    n = len(gold)
    stats = {"accuracy": lambda i: (pred[i] == gold[i]).mean(), "macro_f1": lambda i: macro_f1(pred[i], gold[i])}
    return _with_ci(stats, n, n_resamples)


def ordinal(levels, gold):
    """MAE and RMSE of a level prediction (argmax or expected value) against the gold level."""
    err = np.asarray(levels, dtype=float) - np.asarray(gold, dtype=float)
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean()))}
