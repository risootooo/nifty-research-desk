"""Option pricing: Black-Scholes-Merton, CRR binomial, Monte Carlo.

All functions use the same conventions:
  S     spot price of the underlying
  K     strike
  T     time to expiry in *years* (actual/365 in the chain tools)
  r     continuously compounded risk-free rate (annual)
  q     continuously compounded dividend yield (annual)
  sigma annualised volatility
  kind  'CE' (call) or 'PE' (put)  -- NSE option-type codes
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["bs_price", "binomial_price", "mc_price", "d1_d2"]


def _check_kind(kind: str) -> str:
    k = kind.upper()
    if k in ("CE", "C", "CALL"):
        return "CE"
    if k in ("PE", "P", "PUT"):
        return "PE"
    raise ValueError(f"unknown option kind {kind!r}, use 'CE' or 'PE'")


def d1_d2(S, K, T, r, q, sigma):
    """Black-Scholes d1, d2 (array-safe)."""
    S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def bs_price(S, K, T, r, q, sigma, kind):
    """Black-Scholes-Merton price with continuous dividend yield q."""
    kind = _check_kind(kind)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    # Degenerate expiry: intrinsic value.
    intrinsic = np.maximum(S - K, 0.0) if kind == "CE" else np.maximum(K - S, 0.0)
    live = (T > 0) & (sigma > 0)
    if not np.any(live):
        return float(intrinsic) if np.ndim(intrinsic) == 0 else intrinsic
    d1, d2 = d1_d2(np.where(live, S, 1.0), np.where(live, K, 1.0),
                   np.where(live, T, 1.0), r, q, np.where(live, sigma, 1.0))
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    if kind == "CE":
        price = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    else:
        price = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    price = np.where(live, price, intrinsic)
    return float(price) if np.ndim(price) == 0 else price


def binomial_price(S, K, T, r, q, sigma, kind, steps=500, american=False):
    """Cox-Ross-Rubinstein binomial price.

    European by default; ``american=True`` enables early-exercise checks at
    every node (this matters for puts on high-rate underlyings like NIFTY).
    """
    kind = _check_kind(kind)
    if T <= 0:
        return float(max(S - K, 0.0) if kind == "CE" else max(K - S, 0.0))
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 <= p <= 1.0:
        raise ValueError("risk-neutral probability outside [0,1]; "
                         "increase steps or check inputs")
    j = np.arange(steps + 1)
    spot_t = S * u**j * d**(steps - j)
    if kind == "CE":
        values = np.maximum(spot_t - K, 0.0)
    else:
        values = np.maximum(K - spot_t, 0.0)
    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[1:i + 2] + (1.0 - p) * values[0:i + 1])
        if american:
            spot_i = S * u**np.arange(i + 1) * d**(i - np.arange(i + 1))
            exercise = (np.maximum(spot_i - K, 0.0) if kind == "CE"
                        else np.maximum(K - spot_i, 0.0))
            values = np.maximum(values, exercise)
    return float(values[0])


def mc_price(S, K, T, r, q, sigma, kind, n_paths=200_000, seed=42):
    """Monte Carlo European price with antithetic variates.

    Returns (price, standard_error). Antithetic pairing (z, -z) cuts the
    standard error materially versus plain sampling at the same path count.
    """
    kind = _check_kind(kind)
    if T <= 0:
        return float(max(S - K, 0.0) if kind == "CE" else max(K - S, 0.0)), 0.0
    rng = np.random.default_rng(seed)
    half = n_paths // 2
    z = rng.standard_normal(half)
    z = np.concatenate([z, -z])                       # antithetic pairs
    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T)
    st = S * np.exp(drift + diffusion * z)
    payoff = (np.maximum(st - K, 0.0) if kind == "CE"
              else np.maximum(K - st, 0.0))
    disc = np.exp(-r * T)
    # Pair-average the antithetic twins before estimating the SE so the
    # standard error reflects the variance reduction actually achieved.
    pair_mean = 0.5 * (payoff[:half] + payoff[half:2 * half])
    price = disc * float(pair_mean.mean())
    se = disc * float(pair_mean.std(ddof=1) / np.sqrt(half))
    return price, se
