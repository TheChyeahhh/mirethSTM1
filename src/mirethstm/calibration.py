"""Temperature scaling and expected calibration error (SPEC section 9)."""

import math

import numpy as np

# Filled from the benchmark's calibration splits before release; empty means T = 1.0.
DEFAULT_TEMPERATURES: dict[str, float] = {}

T_MIN = 0.05
T_MAX = 20.0
_TOL = 1e-6  # on log T
_INV_PHI = (math.sqrt(5) - 1) / 2


def _mean_nll(z, y, log_t):
    """Mean negative log-likelihood of softmax(z / T) at the true labels."""
    s = z / math.exp(log_t)
    m = s.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(s - m).sum(axis=1))
    return float(np.mean(lse - s[np.arange(len(y)), y]))


def fit_temperature(logits, labels) -> float:
    """Return the scalar T > 0 that minimizes mean NLL of softmax(logits / T).

    `logits` is a sequence of 1-D score vectors (lengths may differ), `labels`
    the index of the true label in each. Golden-section search on log T over
    [0.05, 20]; raises ValueError when the optimum sits at either bound.
    """
    if len(logits) == 0 or len(logits) != len(labels):
        raise ValueError("logits and labels must be non-empty and the same length")
    # Pad ragged rows with -inf: exp(-inf) = 0, so padding never takes probability.
    width = max(len(row) for row in logits)
    z = np.full((len(logits), width), -np.inf)
    for i, row in enumerate(logits):
        z[i, : len(row)] = row
    y = np.asarray(labels, dtype=int)

    lo, hi = math.log(T_MIN), math.log(T_MAX)
    a, b = lo, hi
    c, d = b - _INV_PHI * (b - a), a + _INV_PHI * (b - a)
    fc, fd = _mean_nll(z, y, c), _mean_nll(z, y, d)
    while b - a > _TOL:
        # Ties keep the lower half: NLL is only exactly flat when it has
        # underflowed to 0 as T shrinks (separable data), or when it does not
        # depend on T at all. Both cases should end at a bound.
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - _INV_PHI * (b - a)
            fc = _mean_nll(z, y, c)
        else:
            a, c, fc = c, d, fd
            d = a + _INV_PHI * (b - a)
            fd = _mean_nll(z, y, d)
    log_t = (a + b) / 2
    if log_t - lo <= _TOL or hi - log_t <= _TOL:
        raise ValueError(
            f"temperature optimum is at the search bound (T = {math.exp(log_t):.4g}, "
            f"range [{T_MIN}, {T_MAX}]); the calibration set cannot fix T"
        )
    return math.exp(log_t)


def ece(confidences, correct, n_bins=15) -> float:
    """Top-label expected calibration error with equal-width bins over [0, 1].

    Bins are ((m-1)/M, m/M] as in Guo et al. 2017, with 0 joining the first bin.
    """
    conf = np.asarray(confidences, dtype=float)
    hit = np.asarray(correct, dtype=float)
    if conf.ndim != 1 or conf.shape != hit.shape or conf.size == 0:
        raise ValueError("confidences and correct must be non-empty 1-D arrays of equal length")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(conf, edges[1:-1], right=True)
    # (n_b / N) * |acc_b - conf_b| equals |sum of hits - sum of confidences| / N per bin.
    hits = np.bincount(idx, weights=hit, minlength=n_bins)
    confs = np.bincount(idx, weights=conf, minlength=n_bins)
    return float(np.abs(hits - confs).sum() / conf.size)
