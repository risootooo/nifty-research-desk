"""Analytic Greeks for Black-Scholes-Merton, including second-order Greeks.

Units (stated once, used everywhere):
  delta  per 1.00 move in the underlying
  gamma  per 1.00 move in the underlying
  theta  per *calendar day* (the desk convention)
  vega   per 1 percentage point of volatility (sigma 0.20 -> 0.21)
  rho    per 1 percentage point of interest rate
  vanna  d(delta)/d(sigma), per 1.00 of sigma
  charm  delta decay per calendar day of time passing
  vomma  d(vega)/d(sigma), per 1.00 of sigma (a.k.a. volga)

Every analytic Greek ships with a validation harness: ``finite_diff_greek``
bumps the relevant input and re-prices with Black-Scholes, and the test
suite asserts analytic vs bumped agreement for *all* of them, including
vanna, charm and vomma (the three most implementations get wrong).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .pricing import bs_price, d1_d2, _check_kind

__all__ = [
    "delta", "gamma", "theta", "vega", "rho",
    "vanna", "charm", "vomma", "all_greeks", "finite_diff_greek",
]


def delta(S, K, T, r, q, sigma, kind):
    kind = _check_kind(kind)
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    out = np.exp(-q * T) * norm.cdf(d1) if kind == "CE" \
        else -np.exp(-q * T) * norm.cdf(-d1)
    return float(out) if np.ndim(out) == 0 else out


def gamma(S, K, T, r, q, sigma, kind=None):
    """Gamma is identical for calls and puts."""
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    out = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return float(out) if np.ndim(out) == 0 else out


def theta(S, K, T, r, q, sigma, kind):
    """Theta per calendar day (negative for long premium, typically)."""
    kind = _check_kind(kind)
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    common = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    if kind == "CE":
        per_year = (common - r * K * np.exp(-r * T) * norm.cdf(d2)
                    + q * S * np.exp(-q * T) * norm.cdf(d1))
    else:
        per_year = (common + r * K * np.exp(-r * T) * norm.cdf(-d2)
                    - q * S * np.exp(-q * T) * norm.cdf(-d1))
    out = per_year / 365.0
    return float(out) if np.ndim(out) == 0 else out


def vega(S, K, T, r, q, sigma, kind=None):
    """Vega per 1 percentage point of volatility."""
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    out = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100.0
    return float(out) if np.ndim(out) == 0 else out


def rho(S, K, T, r, q, sigma, kind):
    """Rho per 1 percentage point of interest rate."""
    kind = _check_kind(kind)
    _, d2 = d1_d2(S, K, T, r, q, sigma)
    if kind == "CE":
        out = K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
    else:
        out = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0
    return float(out) if np.ndim(out) == 0 else out


def vanna(S, K, T, r, q, sigma, kind=None):
    """Vanna = d(delta)/d(sigma) = -exp(-qT) * phi(d1) * d2 / sigma."""
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    out = -np.exp(-q * T) * norm.pdf(d1) * d2 / sigma
    return float(out) if np.ndim(out) == 0 else out


def vomma(S, K, T, r, q, sigma, kind=None):
    """Vomma (volga) = d(vega_raw)/d(sigma) = vega_raw * d1 * d2 / sigma,
    where vega_raw is per 1.00 of sigma (so vomma is per 1.00 of sigma)."""
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    vega_raw = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    out = vega_raw * d1 * d2 / sigma
    return float(out) if np.ndim(out) == 0 else out


def charm(S, K, T, r, q, sigma, kind):
    """Charm = delta decay per calendar day of time passing = -d(delta)/dT / 365.

    Derived exactly: with A = ln(S/K) + (r-q)T and d1 = A/(sigma sqrt(T))
    + sigma sqrt(T)/2,
        dd1/dT = (r-q)/(sigma sqrt(T)) - A/(2 sigma T^1.5) + sigma/(4 sqrt(T))
    and d(delta_call)/dT = -q e^{-qT} N(d1) + e^{-qT} phi(d1) dd1/dT,
        d(delta_put)/dT  = +q e^{-qT} N(-d1) + e^{-qT} phi(d1) dd1/dT
    (the phi term keeps the same sign for puts: delta_put = -e^{-qT} N(-d1)
    and d/dT N(-d1) = -phi(d1) dd1/dT, so the two minus signs cancel).
    Validated against finite differences in the test suite.
    """
    kind = _check_kind(kind)
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    sqrt_t = np.sqrt(T)
    A = np.log(S / K) + (r - q) * T
    dd1_dT = ((r - q) / (sigma * sqrt_t)
              - A / (2.0 * sigma * T * sqrt_t)
              + sigma / (4.0 * sqrt_t))
    disc_q = np.exp(-q * T)
    if kind == "CE":
        d_delta_dT = -q * disc_q * norm.cdf(d1) + disc_q * norm.pdf(d1) * dd1_dT
    else:
        d_delta_dT = q * disc_q * norm.cdf(-d1) + disc_q * norm.pdf(d1) * dd1_dT
    out = -d_delta_dT / 365.0
    return float(out) if np.ndim(out) == 0 else out


def all_greeks(S, K, T, r, q, sigma, kind):
    """Every Greek for one option, as a dict."""
    return {
        "delta": delta(S, K, T, r, q, sigma, kind),
        "gamma": gamma(S, K, T, r, q, sigma),
        "theta": theta(S, K, T, r, q, sigma, kind),
        "vega": vega(S, K, T, r, q, sigma),
        "rho": rho(S, K, T, r, q, sigma, kind),
        "vanna": vanna(S, K, T, r, q, sigma),
        "charm": charm(S, K, T, r, q, sigma, kind),
        "vomma": vomma(S, K, T, r, q, sigma),
    }


def finite_diff_greek(name, S, K, T, r, q, sigma, kind, bump=None):
    """Bump-and-reprice Greek from Black-Scholes, for validation.

    Bumps are sized per Greek; results are converted into the same units
    as the analytic functions above.
    """
    name = name.lower()
    if name == "delta":
        h = bump or max(1e-4 * S, 1e-6)
        return (bs_price(S + h, K, T, r, q, sigma, kind)
                - bs_price(S - h, K, T, r, q, sigma, kind)) / (2 * h)
    if name == "gamma":
        h = bump or max(1e-3 * S, 1e-5)
        return (bs_price(S + h, K, T, r, q, sigma, kind)
                - 2 * bs_price(S, K, T, r, q, sigma, kind)
                + bs_price(S - h, K, T, r, q, sigma, kind)) / h**2
    if name == "theta":                      # per day, time passing
        h = bump or (0.5 / 365.0)
        return (bs_price(S, K, T - h, r, q, sigma, kind)
                - bs_price(S, K, T + h, r, q, sigma, kind)) / (2 * h * 365.0)
    if name == "vega":                       # per 1 vol point
        h = bump or 1e-4
        return (bs_price(S, K, T, r, q, sigma + h, kind)
                - bs_price(S, K, T, r, q, sigma - h, kind)) / (2 * h) / 100.0
    if name == "rho":                        # per 1 rate point
        h = bump or 1e-5
        return (bs_price(S, K, T, r + h, q, sigma, kind)
                - bs_price(S, K, T, r - h, q, sigma, kind)) / (2 * h) / 100.0
    if name == "vanna":                      # d(delta)/d(sigma), per 1.00
        h = bump or 1e-4
        return (delta(S, K, T, r, q, sigma + h, kind)
                - delta(S, K, T, r, q, sigma - h, kind)) / (2 * h)
    if name == "vomma":                      # d(vega_raw)/d(sigma), per 1.00
        h = bump or 1e-4
        vega_raw = lambda s: vega(S, K, T, r, q, s) * 100.0
        return (vega_raw(sigma + h) - vega_raw(sigma - h)) / (2 * h)
    if name == "charm":                      # delta decay per day
        h = bump or (0.5 / 365.0)
        return -(delta(S, K, T + h, r, q, sigma, kind)
                 - delta(S, K, T - h, r, q, sigma, kind)) / (2 * h) / 365.0
    raise ValueError(f"unknown greek {name!r}")
