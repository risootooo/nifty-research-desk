"""Implied volatility solver.

Newton's method on Black-Scholes vega converges in a handful of steps for
sane options but explodes for deep ITM/OTM contracts where vega ~ 0. The
solver therefore runs Newton first and falls back to bisection on
[1e-4, 4.0], which is bracket-safe because BS price is monotone in sigma.

Arbitrage handling: a price below intrinsic value (with q=0 assumptions
violated in the real feed, this *does* happen on NSE settlement data) has
no implied vol. The solver returns NaN for those instead of a fantasy
number, and ``iv_diagnostics`` tells you why.
"""

from __future__ import annotations

import numpy as np

from .pricing import bs_price
from . import greeks as _g

IV_FLOOR, IV_CAP = 1e-4, 4.0


def intrinsic(S, K, kind):
    return max(S - K, 0.0) if kind == "CE" else max(K - S, 0.0)


def implied_vol(price, S, K, T, r, q, kind, tol=1e-10, max_iter=100):
    """Implied vol for one option, or NaN if no arbitrage-consistent vol exists."""
    if price is None or np.isnan(price) or T <= 0:
        return float("nan")
    # No-arb lower bound under BSM with dividends: discounted intrinsic of
    # the forward. Use the simple spot-intrinsic bound as a hard floor --
    # prices below it cannot come from any BSM vol with q >= 0.
    lower = max(intrinsic(S * np.exp(-q * T), K * np.exp(-r * T), kind), 0.0)
    if price < lower - 1e-9:
        return float("nan")
    if price <= lower + 1e-12:
        return 0.0
    # Newton from a Brenner-Subrahmanyam style seed.
    sigma = max(np.sqrt(2 * np.pi / T) * price / S, 0.05) if S > 0 else 0.2
    sigma = min(max(sigma, IV_FLOOR), IV_CAP)
    for _ in range(max_iter):
        diff = bs_price(S, K, T, r, q, sigma, kind) - price
        if abs(diff) < tol:
            return float(sigma)
        v = _g.vega(S, K, T, r, q, sigma) * 100.0     # per 1.00 of sigma
        if v < 1e-10:
            break
        step = diff / v
        sigma -= step
        if sigma <= IV_FLOOR or sigma >= IV_CAP or not np.isfinite(sigma):
            break
    # Bisection fallback (monotone in sigma -> always brackets if possible).
    lo, hi = IV_FLOOR, IV_CAP
    p_lo = bs_price(S, K, T, r, q, lo, kind)
    p_hi = bs_price(S, K, T, r, q, hi, kind)
    if not (p_lo <= price <= p_hi):
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, r, q, mid, kind) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return float(0.5 * (lo + hi))


def implied_vol_series(prices, S, K, T, r, q, kinds):
    """Vectorised wrapper over a chain (plain loop -- chains are small)."""
    return np.array([
        implied_vol(p, S, k, T, r, q, kd)
        for p, k, kd in zip(prices, K, kinds)
    ], dtype=float)
