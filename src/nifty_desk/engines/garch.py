"""GARCH-family volatility models, implemented from scratch.

Models
------
* GARCH(1,1)  -- Bollerslev (1986):  h_t = w + a * r_{t-1}^2 + b * h_{t-1}
* GJR-GARCH   -- Glosten-Jagannathan-Runkle (1993): adds a leverage term
                 gamma * r_{t-1}^2 * 1{r_{t-1} < 0}
* EGARCH(1,1) -- Nelson (1991): log-variance dynamics,
                 log h_t = w + b * log h_{t-1}
                           + a * (|z_{t-1}| - E|z|) + g * z_{t-1}

Estimation is Gaussian maximum likelihood via ``scipy.optimize`` (the
``arch`` package is deliberately *not* used).  All filters are plain
recursions written out in this file.

Conventions
-----------
* ``*_filter`` functions return the conditional variance series ``h``
  aligned with the input returns: ``h[t]`` is the variance of ``r[t]``
  given information up to ``t-1``.
* Returns are used exactly as supplied.  For numerical stability the
  case study fits the models to percentage returns (x100) and rescales
  the forecast variances by 1e-4 before scoring.
* EGARCH multi-step forecasts exponentiate the mean-reverting
  log-variance path.  That ignores the Jensen (log-normal) correction,
  so EGARCH variance forecasts are median-style forecasts; this is a
  documented approximation, not an oversight.

Limitations: Gaussian innovations (fat tails are only captured through
the variance dynamics), single-regime parameters, and daily-frequency
data (intraday realised measures are far better volatility proxies).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import optimize, stats

__all__ = [
    "GarchFit",
    "garch_filter",
    "gjr_filter",
    "egarch_filter",
    "garch_fit",
    "gjr_fit",
    "egarch_fit",
    "forecast",
    "realized_vol",
    "qlike",
    "mse_variance",
    "diebold_mariano",
]

_E_ABS_Z = float(np.sqrt(2.0 / np.pi))  # E|z| for z ~ N(0, 1)


# ----------------------------------------------------------------------
# Filters
# ----------------------------------------------------------------------
def garch_filter(
    returns: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    h0: Optional[float] = None,
) -> np.ndarray:
    """Conditional variances of a GARCH(1,1).

    h[0] = ``h0`` (default: unconditional variance if alpha+beta < 1,
    else the sample variance) and for t >= 1
    ``h[t] = omega + alpha * r[t-1]**2 + beta * h[t-1]``.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    if h0 is None:
        if alpha + beta < 1.0:
            h0 = omega / (1.0 - alpha - beta)
        else:
            h0 = float(np.var(r))
    h = np.empty(n)
    if n == 0:
        return h
    h[0] = h0
    for t in range(1, n):
        h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
    return h


def gjr_filter(
    returns: np.ndarray,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    h0: Optional[float] = None,
) -> np.ndarray:
    """Conditional variances of a GJR-GARCH(1,1).

    ``h[t] = omega + (alpha + gamma * 1{r[t-1] < 0}) * r[t-1]**2
             + beta * h[t-1]``.
    Persistence (symmetric innovations) is alpha + beta + gamma / 2.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    persistence = alpha + beta + gamma / 2.0
    if h0 is None:
        if persistence < 1.0:
            h0 = omega / (1.0 - persistence)
        else:
            h0 = float(np.var(r))
    h = np.empty(n)
    if n == 0:
        return h
    h[0] = h0
    for t in range(1, n):
        shock = alpha + (gamma if r[t - 1] < 0.0 else 0.0)
        h[t] = omega + shock * r[t - 1] ** 2 + beta * h[t - 1]
    return h


def egarch_filter(
    returns: np.ndarray,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    h0: Optional[float] = None,
) -> np.ndarray:
    """Conditional variances of an EGARCH(1,1) (returned in levels).

    ``log h[t] = omega + beta * log h[t-1]
                 + alpha * (|z[t-1]| - sqrt(2/pi)) + gamma * z[t-1]``
    with ``z = r / sqrt(h)``.  ``h0`` defaults to the sample variance.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    if h0 is None:
        h0 = float(np.var(r))
    h = np.empty(n)
    if n == 0:
        return h
    h[0] = h0
    log_h_prev = np.log(h0)
    for t in range(1, n):
        z_prev = r[t - 1] / np.sqrt(np.exp(log_h_prev))
        log_h = (
            omega
            + beta * log_h_prev
            + alpha * (abs(z_prev) - _E_ABS_Z)
            + gamma * z_prev
        )
        h[t] = np.exp(log_h)
        log_h_prev = log_h
    return h


# ----------------------------------------------------------------------
# Estimation (Gaussian MLE)
# ----------------------------------------------------------------------
@dataclass
class GarchFit:
    """Result of a GARCH-family maximum-likelihood fit."""

    model: str  # "garch" | "gjr" | "egarch"
    params: dict[str, float]
    loglik: float
    n_obs: int
    last_var: float  # fitted conditional variance at the final observation
    last_ret: float  # final observed return
    persistence: float
    unconditional_var: Optional[float]  # None when persistence >= 1
    success: bool = True
    extra: dict = field(default_factory=dict)


def _neg_loglik_garch(theta: np.ndarray, r: np.ndarray) -> float:
    omega, alpha, beta = theta
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
        return 1e10
    h = garch_filter(r, omega, alpha, beta, h0=float(np.var(r)))
    if np.any(h <= 0) or not np.all(np.isfinite(h)):
        return 1e10
    return float(0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + r**2 / h))


def _neg_loglik_gjr(theta: np.ndarray, r: np.ndarray) -> float:
    omega, alpha, gamma, beta = theta
    persistence = alpha + beta + gamma / 2.0
    if omega <= 0 or alpha < 0 or gamma < 0 or beta < 0 or persistence >= 0.9999:
        return 1e10
    h = gjr_filter(r, omega, alpha, gamma, beta, h0=float(np.var(r)))
    if np.any(h <= 0) or not np.all(np.isfinite(h)):
        return 1e10
    return float(0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + r**2 / h))


def _neg_loglik_egarch(theta: np.ndarray, r: np.ndarray) -> float:
    omega, alpha, gamma, beta = theta
    if abs(beta) >= 0.9999:
        return 1e10
    h = egarch_filter(r, omega, alpha, gamma, beta, h0=float(np.var(r)))
    if np.any(h <= 0) or not np.all(np.isfinite(h)):
        return 1e10
    return float(0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + r**2 / h))


def garch_fit(returns: np.ndarray) -> GarchFit:
    """Fit GARCH(1,1) by Gaussian MLE."""
    r = np.asarray(returns, dtype=float)
    v = float(np.var(r))
    x0 = np.array([0.05 * v, 0.08, 0.88])
    res = optimize.minimize(
        _neg_loglik_garch,
        x0,
        args=(r,),
        method="L-BFGS-B",
        bounds=[(1e-12, None), (0.0, 1.0), (0.0, 1.0)],
    )
    omega, alpha, beta = (float(x) for x in res.x)
    persistence = alpha + beta
    h = garch_filter(r, omega, alpha, beta)
    return GarchFit(
        model="garch",
        params={"omega": omega, "alpha": alpha, "beta": beta},
        loglik=-float(res.fun),
        n_obs=r.size,
        last_var=float(h[-1]),
        last_ret=float(r[-1]),
        persistence=float(persistence),
        unconditional_var=float(omega / (1 - persistence))
        if persistence < 1
        else None,
        success=bool(res.success),
    )


def gjr_fit(returns: np.ndarray) -> GarchFit:
    """Fit GJR-GARCH(1,1) by Gaussian MLE."""
    r = np.asarray(returns, dtype=float)
    v = float(np.var(r))
    x0 = np.array([0.05 * v, 0.04, 0.08, 0.88])
    res = optimize.minimize(
        _neg_loglik_gjr,
        x0,
        args=(r,),
        method="L-BFGS-B",
        bounds=[(1e-12, None), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    )
    omega, alpha, gamma, beta = (float(x) for x in res.x)
    persistence = alpha + beta + gamma / 2.0
    h = gjr_filter(r, omega, alpha, gamma, beta)
    return GarchFit(
        model="gjr",
        params={"omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta},
        loglik=-float(res.fun),
        n_obs=r.size,
        last_var=float(h[-1]),
        last_ret=float(r[-1]),
        persistence=float(persistence),
        unconditional_var=float(omega / (1 - persistence))
        if persistence < 1
        else None,
        success=bool(res.success),
    )


def egarch_fit(returns: np.ndarray) -> GarchFit:
    """Fit EGARCH(1,1) by Gaussian MLE."""
    r = np.asarray(returns, dtype=float)
    v = float(np.var(r))
    # log-variance intercept consistent with level ~ v at beta = 0.95
    x0 = np.array([(1 - 0.95) * np.log(v), 0.12, -0.06, 0.95])
    res = optimize.minimize(
        _neg_loglik_egarch,
        x0,
        args=(r,),
        method="L-BFGS-B",
        bounds=[(None, None), (0.0, 1.0), (-1.0, 1.0), (-0.999, 0.999)],
    )
    omega, alpha, gamma, beta = (float(x) for x in res.x)
    h = egarch_filter(r, omega, alpha, gamma, beta)
    return GarchFit(
        model="egarch",
        params={"omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta},
        loglik=-float(res.fun),
        n_obs=r.size,
        last_var=float(h[-1]),
        last_ret=float(r[-1]),
        persistence=float(beta),
        unconditional_var=float(np.exp(omega / (1 - beta)))
        if abs(beta) < 1
        else None,
        success=bool(res.success),
    )


# ----------------------------------------------------------------------
# Forecasting
# ----------------------------------------------------------------------
def forecast(
    params: dict[str, float],
    model: str,
    last_var: float,
    last_ret: float,
    horizon: int = 1,
) -> np.ndarray:
    """Variance forecasts for horizons 1..``horizon``.

    ``last_var`` is the filtered variance of the most recent observation
    and ``last_ret`` that observation's return.  GARCH/GJR forecasts
    mean-revert to the unconditional variance at the persistence rate;
    EGARCH forecasts mean-revert in log-variance (see module docstring
    for the Jensen-approximation caveat).
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    out = np.empty(horizon)

    if model == "garch":
        omega, alpha, beta = params["omega"], params["alpha"], params["beta"]
        persistence = alpha + beta
        h1 = omega + alpha * last_ret**2 + beta * last_var
        out[0] = h1
        if horizon > 1:
            if persistence < 1:
                hbar = omega / (1 - persistence)
                steps = np.arange(1, horizon)
                out[1:] = hbar + persistence**steps * (h1 - hbar)
            else:
                out[1:] = h1
        return out

    if model == "gjr":
        omega, alpha = params["omega"], params["alpha"]
        gamma, beta = params["gamma"], params["beta"]
        shock = alpha + (gamma if last_ret < 0 else 0.0)
        h1 = omega + shock * last_ret**2 + beta * last_var
        out[0] = h1
        if horizon > 1:
            persistence = alpha + beta + gamma / 2.0
            if persistence < 1:
                hbar = omega / (1 - persistence)
                steps = np.arange(1, horizon)
                out[1:] = hbar + persistence**steps * (h1 - hbar)
            else:
                out[1:] = h1
        return out

    if model == "egarch":
        omega, alpha = params["omega"], params["alpha"]
        gamma, beta = params["gamma"], params["beta"]
        z_last = last_ret / np.sqrt(last_var)
        log_h1 = (
            omega
            + beta * np.log(last_var)
            + alpha * (abs(z_last) - _E_ABS_Z)
            + gamma * z_last
        )
        out[0] = np.exp(log_h1)
        if horizon > 1:
            log_hbar = omega / (1 - beta) if abs(beta) < 1 else log_h1
            steps = np.arange(1, horizon)
            out[1:] = np.exp(log_hbar + beta**steps * (log_h1 - log_hbar))
        return out

    raise ValueError(f"unknown model: {model!r}")


# ----------------------------------------------------------------------
# Realised volatility and forecast scoring
# ----------------------------------------------------------------------
def realized_vol(
    data,
    window: int = 21,
    is_prices: bool = False,
    trading_days: int = 252,
):
    """Rolling realised volatility, annualised, as a fraction (0.15 = 15%).

    ``data`` may be a price series (``is_prices=True``; log returns are
    computed) or a return series (fractions, not percent).  Returns a
    pandas Series if given one, else a numpy array, with ``window``
    observations required before the first value (NaN / dropped head
    accordingly -- with an ndarray the head entries are NaN).
    """
    import pandas as pd

    is_series = isinstance(data, pd.Series)
    s = data if is_series else pd.Series(np.asarray(data, dtype=float))
    rets = np.log(s / s.shift(1)) if is_prices else s
    rv = rets.rolling(window).std(ddof=1) * np.sqrt(trading_days)
    return rv if is_series else rv.to_numpy()


def qlike(proxy_var: np.ndarray, forecast_var: np.ndarray) -> np.ndarray:
    """QLIKE loss (Patton 2011), up to constants: log(h) + proxy / h."""
    proxy_var = np.asarray(proxy_var, dtype=float)
    forecast_var = np.asarray(forecast_var, dtype=float)
    return np.log(forecast_var) + proxy_var / forecast_var


def mse_variance(proxy_var: np.ndarray, forecast_var: np.ndarray) -> np.ndarray:
    """Squared error of a variance forecast against a variance proxy."""
    return (np.asarray(proxy_var, float) - np.asarray(forecast_var, float)) ** 2


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    hac_lag: Optional[int] = None,
) -> tuple[float, float]:
    """Diebold-Mariano test on two equal-length loss series (h = 1).

    Returns (DM statistic, two-sided p-value).  The statistic is
    mean(loss_a - loss_b) / sqrt(Newey-West LRV / n); a negative value
    means model A has the lower loss.  The long-run variance uses a
    Bartlett kernel with lag ``floor(4 * (n/100)^(2/9))`` unless
    ``hac_lag`` is given.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = d.size
    if n < 10:
        raise ValueError("need at least 10 loss observations")
    dbar = float(np.mean(d))
    if hac_lag is None:
        hac_lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    dc = d - dbar
    lrv = float(np.dot(dc, dc) / n)
    for lag in range(1, hac_lag + 1):
        w = 1.0 - lag / (hac_lag + 1.0)
        cov = float(np.dot(dc[lag:], dc[:-lag]) / n)
        lrv += 2.0 * w * cov
    if lrv <= 0:
        return 0.0, 1.0
    dm = dbar / np.sqrt(lrv / n)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(dm)))
    return float(dm), float(p)
