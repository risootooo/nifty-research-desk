"""Smoke tests for the vendored regime (Hamilton HMM) and GARCH engines
on synthetic data with known ground truth."""

import numpy as np

from nifty_desk.engines import garch as garch_engine
from nifty_desk.engines import regimes as regime_engine


def test_hmm_recovers_two_regimes():
    rng = np.random.default_rng(42)
    calm = rng.normal(0.0004, 0.006, 600)
    storm = rng.normal(-0.001, 0.028, 150)
    returns = np.concatenate([calm, storm, calm[:300]])
    params = regime_engine.fit(returns)
    # States are volatility-ordered: 0 = low, 1 = high.
    assert params["sigma"][0] < params["sigma"][1]
    assert params["sigma"][0] == np.float64(params["sigma"][0])  # finite
    assert 0.003 < params["sigma"][0] < 0.012
    assert 0.015 < params["sigma"][1] < 0.045
    sm = regime_engine.smoothed_probs(returns, params)
    assert sm.shape == (len(returns), 2)
    assert np.all(sm >= 0.0) and np.all(sm <= 1.0)
    assert np.allclose(sm.sum(axis=1), 1.0)
    # The storm block must be classified high-vol on average.
    assert sm[600:750, 1].mean() > 0.5
    filt = regime_engine.filtered_probs(returns, params)
    assert np.all((filt >= 0.0) & (filt <= 1.0))
    durations = regime_engine.expected_durations(params)
    assert np.all(durations > 1.0)


def test_garch_fit_and_forecast_on_synthetic():
    rng = np.random.default_rng(7)
    n = 800
    omega, alpha, beta = 2e-6, 0.08, 0.90
    h = np.empty(n)
    r = np.empty(n)
    h[0] = omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = rng.normal(0.0, np.sqrt(h[t]))
        if t + 1 < n:
            h[t + 1] = omega + alpha * r[t] ** 2 + beta * h[t]
    fit = garch_engine.garch_fit(r)
    assert fit.success
    assert 0.5 < fit.persistence < 1.0
    fc = garch_engine.forecast(fit.params, "garch", fit.last_var,
                               fit.last_ret, horizon=5)
    assert fc.shape == (5,)
    assert np.all(fc > 0)
    # Mean-reverting forecast: 5-day path stays within a sane band of the
    # last filtered variance (no explosion, no collapse to zero).
    assert np.all(fc < 5 * fit.last_var) and np.all(fc > 0.2 * fit.last_var)
