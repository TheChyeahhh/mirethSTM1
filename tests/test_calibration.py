"""Calibration tests on synthetic data (no model)."""

import math

import numpy as np
import pytest

from mirethstm.calibration import DEFAULT_TEMPERATURES, ece, fit_temperature

T_TRUE = 2.5


def softmax(z, t=1.0):
    s = np.asarray(z, dtype=float) / t
    e = np.exp(s - s.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def sample(rng, n, k, t=T_TRUE):
    """Logits z ~ N(0, 4) and labels drawn from softmax(z / t)."""
    z = rng.normal(0.0, 4.0, size=(n, k))
    y = (softmax(z, t).cumsum(axis=1) > rng.random((n, 1))).argmax(axis=1)
    return z, y


def macro_f1(pred, gold, k):
    scores = []
    for c in range(k):
        tp = np.sum((pred == c) & (gold == c))
        fp = np.sum((pred == c) & (gold != c))
        fn = np.sum((pred != c) & (gold == c))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(scores))


def test_default_temperatures_are_fitted_values():
    # The pooled T per approved model; each model's own entry in mirethstm.models must match (test_api).
    assert DEFAULT_TEMPERATURES
    assert all(isinstance(t, float) and 1.0 < t < 20.0 for t in DEFAULT_TEMPERATURES.values())


def test_fit_temperature_recovers_known_t():
    z, y = sample(np.random.default_rng(0), 20000, 4)
    # At N = 20000 the fitted T has a relative standard deviation of about 1%
    # (measured over 100 seeds), so 5% is about five standard deviations.
    assert fit_temperature(list(z), y) == pytest.approx(T_TRUE, rel=0.05)


def test_fit_temperature_ragged_label_counts():
    rng = np.random.default_rng(1)
    groups = [sample(rng, 5000, k) for k in (2, 3, 5, 77)]
    rows = [(row, label) for z, y in groups for row, label in zip(z, y)]
    order = rng.permutation(len(rows))
    logits = [rows[i][0] for i in order]
    labels = [rows[i][1] for i in order]
    t = fit_temperature(logits, labels)
    assert t == pytest.approx(T_TRUE, rel=0.05)

    # Independent check without padding: NLL summed group by group on a
    # 121-point log grid over [0.25, 4]; the optimum is within one grid step.
    log_grid = np.linspace(math.log(0.25), math.log(4.0), 121)

    def nll(log_t):
        total = 0.0
        for z, y in groups:
            s = z / math.exp(log_t)
            m = s.max(axis=1)
            lse = m + np.log(np.exp(s - m[:, None]).sum(axis=1))
            total += np.sum(lse - s[np.arange(len(y)), y])
        return total / len(rows)

    best = log_grid[np.argmin([nll(u) for u in log_grid])]
    assert abs(math.log(t) - best) <= log_grid[1] - log_grid[0]


@pytest.mark.parametrize("t_true", [0.055, 0.1, 0.15, 1.0, 15.0])
def test_fit_temperature_exact_optimum(t_true):
    # Scores [a, 0] with the first label right 3 times in 4: NLL is minimal where
    # sigmoid(a / T) = 3/4, so at exactly T = a / ln 3. The search on log T ends
    # within 5e-7 relative anywhere in the range; a search on T itself only
    # promises 5e-7 absolute, which is 1e-5 relative near T = 0.05.
    a = t_true * math.log(3)
    assert fit_temperature([np.array([a, 0.0])] * 4, [0, 0, 0, 1]) == pytest.approx(t_true, rel=1e-6)


def test_fit_temperature_raises_at_lower_bound():
    # The top label is always right, so sharper is always better: T wants to go to 0.
    logits = [np.array([5.0, 0.0, -1.0])] * 50
    with pytest.raises(ValueError):
        fit_temperature(logits, [0] * 50)


def test_fit_temperature_raises_at_upper_bound():
    # The same scores with both labels equally often: T wants to go to infinity.
    logits = [np.array([3.0, 0.0])] * 2
    with pytest.raises(ValueError):
        fit_temperature(logits, [0, 1])


def test_ece_hand_computed():
    # 15 bins: 0.95 -> bin 14 (2 rows, acc 0.5), 0.62 -> bin 9 (2 rows, acc 1),
    # 0.3 -> bin 4 (1 row, acc 0).
    # ECE = 2/5 * 0.45 + 2/5 * 0.38 + 1/5 * 0.3 = 0.18 + 0.152 + 0.06 = 0.392
    conf = [0.95, 0.95, 0.62, 0.62, 0.3]
    correct = [1, 0, 1, 1, 0]
    assert ece(conf, correct) == pytest.approx(0.392)
    # Confidence 1.0 lands in the last bin.
    assert ece([1.0], [True]) == pytest.approx(0.0)
    assert ece([1.0], [False]) == pytest.approx(1.0)
    # ... together with 0.95, not in a 16th bin or in bin 0: |1 - 1.95| / 2.
    assert ece([1.0, 0.95], [0, 1]) == pytest.approx(0.475)
    # 1.0 and 0.02 sit in opposite end bins: (1 + 0.98) / 2.
    assert ece([1.0, 0.02], [0, 1]) == pytest.approx(0.99)
    # Confidence 0.0 joins the first bin with 0.05: |1 - 0.05| / 2.
    assert ece([0.0, 0.05], [0, 1]) == pytest.approx(0.475)


def test_ece_near_zero_when_calibrated():
    rng = np.random.default_rng(2)
    conf = rng.uniform(0.5, 1.0, size=200000)
    # About 26000 rows per occupied bin: per-bin accuracy noise is about 0.003,
    # so a calibrated ECE sits near 0.002; 0.01 leaves room.
    assert ece(conf, rng.random(conf.size) < conf) < 0.01
    # Ten points overconfident everywhere shows up as ECE of about 0.1.
    assert ece(conf, rng.random(conf.size) < conf - 0.1) == pytest.approx(0.1, abs=0.01)


def test_ece_respects_n_bins():
    conf, correct = [0.2, 0.8], [0, 1]
    # One bin: mean confidence 0.5, accuracy 0.5.
    assert ece(conf, correct, n_bins=1) == pytest.approx(0.0)
    # Two bins: each row alone, gap 0.2 in both.
    assert ece(conf, correct, n_bins=2) == pytest.approx(0.2)
    assert ece(conf, correct) == ece(conf, correct, n_bins=15)


def test_accuracy_and_macro_f1_unchanged_by_temperature():
    rng = np.random.default_rng(3)
    k = 4
    z = rng.normal(0.0, 3.0, size=(500, k))
    gold = rng.integers(0, k, size=500)
    ref = softmax(z).argmax(axis=1)
    for t in (0.05, 0.3, 2.5, 20.0):
        pred = softmax(z, t).argmax(axis=1)
        assert np.mean(pred == gold) == np.mean(ref == gold)
        assert macro_f1(pred, gold, k) == macro_f1(ref, gold, k)
