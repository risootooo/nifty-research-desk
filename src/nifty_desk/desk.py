"""Snapshot assembly for the NIFTY Research Desk.

``build_snapshot`` runs the vendored engines over the index history and one
option chain and returns ONE plain dict — the contract the brief writer and
the chart layer code against:

    asof_index_date  str    last date in the NIFTY history (YYYY-MM-DD)
    chain_date       str    bhavcopy trade date (YYYY-MM-DD)
    expiry           str    chain expiry used (YYYY-MM-DD)
    spot             float  chain underlying print
    nifty_last_close float
    nifty_ret_5d     float  simple return, decimal
    nifty_ret_21d    float  simple return, decimal
    regime           dict   p_high_vol_today (Hamilton *filtered* prob, no
                            look-ahead), low_vol_ann / high_vol_ann
                            (annualised regime vols, decimal),
                            dur_low_days / dur_high_days (expected
                            durations, trading days), share_high_vol_1y
                            (share of last 252 days classified high-vol by
                            smoothed prob > 0.5)
    vol              dict   realized_21d_ann (decimal), garch_forecast_5d_ann
                            (list of 5 annualised vols, decimal),
                            garch_params {omega, alpha, beta, persistence}
    chain            dict   net_gex_cr_per_1pct (₹ crore per 1% move),
                            gamma_flip, max_pain, pc_oi_ratio (PE OI / CE OI),
                            atm_iv (decimal), skew_minus10 (IV at −10%
                            moneyness minus ATM IV, volatility points),
                            total_oi (contracts, both kinds)
    banknifty        dict   last_close, ret_21d, realized_21d_ann
    regime_series    dict   {dates, p_high} — smoothed P(high-vol), full
                            history, for charting
    vol_series       dict   {dates, realized_21d} — rolling 21d annualised
                            realised vol, full history
    gex_profile      dict   {strikes, gex_cr} — net GEX per strike, ₹ crore
    smile            dict   {moneyness_pct, iv} — OTM smile; iv in percent

Conventions: returns are log returns unless named ``ret_*`` (simple);
annualisation is ×√252 on daily quantities; the options engine is the
vendored greeks-lab chain module with its stated conventions (r = 6.5%,
q = 0, premium = settlement close, GEX in the canonical contracts × lot
form).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .engines import chain as ch
from .engines import garch as garch_engine
from .engines import regimes as regime_engine

TRADING_DAYS = 252
SQRT_TD = float(np.sqrt(TRADING_DAYS))
CRORE = 1e7


def _closes(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    date_col = next(c for c in df.columns if c.lower() == "date")
    close_col = next(c for c in df.columns if c.lower() == "close")
    dates = pd.to_datetime(df[date_col])
    closes = pd.to_numeric(df[close_col], errors="coerce")
    out = pd.DataFrame({"date": dates, "close": closes}).dropna()
    out = out.sort_values("date").reset_index(drop=True)
    return out["date"], out["close"].astype(float)


def _log_returns(closes: pd.Series) -> np.ndarray:
    return np.diff(np.log(closes.values))


def _realized_ann(returns: np.ndarray, window: int = 21) -> float:
    return float(np.std(returns[-window:], ddof=1) * SQRT_TD)


def _prepare_chain(chain_input, expiry_choice: str) -> pd.DataFrame:
    """Accept a bhavcopy path, a raw bhavcopy DataFrame, or a tidy chain."""
    if isinstance(chain_input, (str, Path)):
        raw = ch.load_bhavcopy(str(chain_input))
        return _extract(raw, expiry_choice)
    if isinstance(chain_input, pd.DataFrame):
        if "TckrSymb" in chain_input.columns:  # raw bhavcopy frame
            return _extract(chain_input, expiry_choice)
        return chain_input  # already a tidy chain from extract_chain
    raise TypeError(f"unsupported chain input: {type(chain_input)!r}")


def _extract(raw: pd.DataFrame, expiry_choice: str) -> pd.DataFrame:
    if expiry_choice == "nearest":
        return ch.extract_chain(raw, symbol="NIFTY", expiry=None, min_days=1)
    expiry = pd.to_datetime(expiry_choice).date()
    return ch.extract_chain(raw, symbol="NIFTY", expiry=expiry, min_days=0)


def build_snapshot(nifty_df: pd.DataFrame, banknifty_df: pd.DataFrame,
                   chain_df_or_path, expiry_choice: str = "nearest") -> dict:
    n_dates, n_close = _closes(nifty_df)
    b_dates, b_close = _closes(banknifty_df)
    n_ret = _log_returns(n_close)
    b_ret = _log_returns(b_close)
    ret_dates = n_dates.iloc[1:].reset_index(drop=True)

    # ---- Regime (Hamilton HMM on the full NIFTY history) ----------------
    params = regime_engine.fit(n_ret)
    smoothed = regime_engine.smoothed_probs(n_ret, params)[:, 1]
    filtered = regime_engine.filtered_probs(n_ret, params)[:, 1]
    durations = regime_engine.expected_durations(params)
    last_year = smoothed[-TRADING_DAYS:]
    regime = {
        "p_high_vol_today": float(filtered[-1]),
        "low_vol_ann": float(params["sigma"][0] * SQRT_TD),
        "high_vol_ann": float(params["sigma"][1] * SQRT_TD),
        "dur_low_days": float(durations[0]),
        "dur_high_days": float(durations[1]),
        "share_high_vol_1y": float(np.mean(last_year > 0.5)),
    }
    regime_series = {
        "dates": [d.strftime("%Y-%m-%d") for d in ret_dates],
        "p_high": [float(x) for x in smoothed],
    }

    # ---- Volatility (GARCH(1,1) on the last 1000 returns) ---------------
    fit = garch_engine.garch_fit(n_ret[-1000:])
    fc_var = garch_engine.forecast(fit.params, "garch", fit.last_var,
                                   fit.last_ret, horizon=5)
    vol = {
        "realized_21d_ann": _realized_ann(n_ret),
        "garch_forecast_5d_ann": [float(np.sqrt(v) * SQRT_TD) for v in fc_var],
        "garch_params": {**{k: float(v) for k, v in fit.params.items()},
                         "persistence": float(fit.persistence)},
    }
    rv = pd.Series(n_ret).rolling(21).std(ddof=1) * SQRT_TD
    vol_series = {
        "dates": [d.strftime("%Y-%m-%d") for d in ret_dates[rv.notna()]],
        "realized_21d": [float(x) for x in rv.dropna()],
    }

    # ---- Option chain (vendored greeks-lab analytics) -------------------
    chain = _prepare_chain(chain_df_or_path, expiry_choice)
    civ = ch.add_iv(chain)
    spot = float(civ["spot"].iloc[0])
    atm, _ = ch.atm_iv(civ)

    gex_df = ch.gamma_exposure(civ)
    gex_profile = {
        "strikes": [float(s) for s in gex_df.index],
        "gex_cr": [float(v / CRORE) for v in gex_df["gex_rs"]],
    }

    smile_df = ch.smile(civ)
    moneyness_pct = (smile_df["strike"].values / spot - 1.0) * 100.0
    smile_iv_pct = smile_df["iv"].values * 100.0
    below = smile_df["strike"].values < spot
    if below.any() and np.isfinite(atm):
        j = int(np.argmin(np.abs(moneyness_pct[below] + 10.0)))
        skew_minus10 = float(smile_iv_pct[below][j] - atm * 100.0)
    else:
        skew_minus10 = float("nan")

    ce_oi = civ.loc[civ["kind"] == "CE", "oi"].sum()
    pe_oi = civ.loc[civ["kind"] == "PE", "oi"].sum()
    lot = float(civ["lot_size"].iloc[0])
    chain_block = {
        "net_gex_cr_per_1pct": float(gex_df["gex_rs"].sum() / CRORE),
        "gamma_flip": float(ch.gamma_flip_level(gex_df)),
        "max_pain": float(ch.max_pain(chain)),
        "pc_oi_ratio": float(pe_oi / ce_oi) if ce_oi else float("nan"),
        "atm_iv": float(atm),
        "skew_minus10": skew_minus10,
        "total_oi": float((ce_oi + pe_oi) / lot),
    }
    smile_block = {
        "moneyness_pct": [float(x) for x in moneyness_pct],
        "iv": [float(x) for x in smile_iv_pct],
    }

    banknifty = {
        "last_close": float(b_close.iloc[-1]),
        "ret_21d": float(b_close.iloc[-1] / b_close.iloc[-22] - 1.0),
        "realized_21d_ann": _realized_ann(b_ret),
    }

    return {
        "asof_index_date": n_dates.iloc[-1].strftime("%Y-%m-%d"),
        "chain_date": str(chain["trade_date"].iloc[0]),
        "expiry": str(chain["expiry"].iloc[0]),
        "spot": spot,
        "nifty_last_close": float(n_close.iloc[-1]),
        "nifty_ret_5d": float(n_close.iloc[-1] / n_close.iloc[-6] - 1.0),
        "nifty_ret_21d": float(n_close.iloc[-1] / n_close.iloc[-22] - 1.0),
        "regime": regime,
        "vol": vol,
        "chain": chain_block,
        "banknifty": banknifty,
        "regime_series": regime_series,
        "vol_series": vol_series,
        "gex_profile": gex_profile,
        "smile": smile_block,
    }
