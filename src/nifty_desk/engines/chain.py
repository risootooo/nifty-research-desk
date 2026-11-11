"""NSE bhavcopy parsing and option-chain analytics.

Input format: the NSE F&O bhavcopy CSV (daily file
``BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv``, also distributed inside the
dated ``fo_YYYYMMDD.zip`` archives). The columns used are TradDt,
FinInstrmTp ('IDO' = index options), TckrSymb, XpryDt, StrkPric, OptnTp
('CE'/'PE'), ClsPric, UndrlygPric, OpnIntrst, TtlTradgVol, NewBrdLotQty.

Conventions used in this module (all stated, none hidden):
  * Premium used for IV: ClsPric (the exchange close / settlement print).
  * Rate: 6.5% p.a. unless overridden. Dividend yield q = 0 (NSE index
    options are cash-settled on the index; the parity diagnostic below
    measures how far the feed deviates from that assumption).
  * Days to expiry: actual calendar days, T = days/365.
  * "Nearest expiry" = the nearest expiry with at least 1 day left. On
    expiry day itself T = 0 and IV is undefined, so the expiring series is
    never used as the headline chain.
  * GEX per strike (dealer gamma exposure, ₹):
        GEX_K = (Gamma_CE(K) * OI_CE(K) - Gamma_PE(K) * OI_PE(K))
                * LOT * S^2 * 0.01
    where OI is in *contracts* and LOT is the contract multiplier. The US
    literature writes this with a 100 multiplier because SPX options
    control 100 shares; the NIFTY lot (65 in this file, read from
    NewBrdLotQty) plays exactly that role here, and the bhavcopy's
    share-denominated OI is converted to contracts first (dividing by the
    lot and multiplying back is deliberate -- it keeps the formula in the
    canonical contracts x multiplier form). Calls contribute positive
    dealer gamma, puts negative, gamma is computed at each strike's own
    implied vol, and the result is the ₹ gamma exposure per 1% spot move.
    This is a *model* of dealer positioning (dealers assumed short the
    OI), not an observable -- see the README limitations.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date

import numpy as np
import pandas as pd

from .iv import implied_vol_series
from . import greeks as _g

BHAV_COLUMNS = ["TradDt", "FinInstrmTp", "TckrSymb", "XpryDt", "StrkPric",
                "OptnTp", "ClsPric", "UndrlygPric", "OpnIntrst",
                "TtlTradgVol", "NewBrdLotQty"]
DEFAULT_RATE = 0.065


def load_bhavcopy(source):
    """Load a bhavcopy from a .csv path, a .zip path (dated NSE archive),
    or a file-like object. Returns the raw DataFrame (all symbols)."""
    if isinstance(source, (str, bytes)) and str(source).endswith(".zip"):
        with zipfile.ZipFile(source) as z:
            name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
            raw = z.read(name)
        return pd.read_csv(io.BytesIO(raw), usecols=lambda c: c in BHAV_COLUMNS)
    return pd.read_csv(source, usecols=lambda c: c in BHAV_COLUMNS)


def extract_chain(df, symbol="NIFTY", expiry=None, min_days=1):
    """Extract one option chain as a tidy DataFrame.

    Columns out: trade_date, expiry, strike, kind, premium, spot, oi,
    volume, lot_size, days_to_expiry, T.
    """
    sub = df[(df["TckrSymb"] == symbol) & (df["FinInstrmTp"] == "IDO")].copy()
    if sub.empty:
        raise ValueError(f"no index options for {symbol!r} in frame")
    sub["TradDt"] = pd.to_datetime(sub["TradDt"]).dt.date
    sub["XpryDt"] = pd.to_datetime(sub["XpryDt"]).dt.date
    trade_date = sub["TradDt"].iloc[0]
    expiries = sorted(d for d in sub["XpryDt"].unique()
                      if (d - trade_date).days >= min_days)
    if expiry is None:
        if not expiries:
            raise ValueError("no expiry with >= min_days left")
        expiry = expiries[0]
    chain = sub[sub["XpryDt"] == expiry].copy()
    days = (expiry - trade_date).days
    out = pd.DataFrame({
        "trade_date": trade_date,
        "expiry": expiry,
        "strike": chain["StrkPric"].astype(float),
        "kind": chain["OptnTp"],
        "premium": chain["ClsPric"].astype(float),
        "spot": chain["UndrlygPric"].astype(float),
        "oi": chain["OpnIntrst"].astype(float),
        "volume": chain["TtlTradgVol"].astype(float),
        "lot_size": chain["NewBrdLotQty"].astype(float),
        "days_to_expiry": days,
        "T": days / 365.0,
    }).sort_values(["strike", "kind"]).reset_index(drop=True)
    return out


def add_iv(chain, r=DEFAULT_RATE, q=0.0, liquid_only=True):
    """Add per-strike implied vol + moneyness; optionally keep OI > 0 rows."""
    out = chain.copy()
    if liquid_only:
        out = out[out["oi"] > 0].reset_index(drop=True)
    spot = float(out["spot"].iloc[0])
    T = float(out["T"].iloc[0])
    out["iv"] = implied_vol_series(out["premium"].values, spot,
                                   out["strike"].values, T, r, q,
                                   out["kind"].values)
    out["moneyness"] = out["strike"] / spot
    out["otm"] = np.where(out["kind"] == "CE", out["strike"] >= spot,
                          out["strike"] <= spot)
    return out


def smile(chain_iv):
    """The IV smile on OTM options (the market convention): puts below
    spot, calls above spot, nearest strike counted once."""
    spot = float(chain_iv["spot"].iloc[0])
    otm = chain_iv[chain_iv["otm"] & chain_iv["iv"].notna()].copy()
    otm["side"] = np.where(otm["strike"] < spot, "PE", "CE")
    atm_row = otm.iloc[(otm["strike"] - spot).abs().argsort().iloc[0]]
    others = otm.drop(index=atm_row.name)
    smile_df = pd.concat([others, atm_row.to_frame().T]).sort_values("strike")
    return smile_df[["strike", "kind", "premium", "oi", "iv"]].reset_index(drop=True)


def atm_iv(chain_iv):
    """IV of the strike nearest spot (average of CE/PE where both solve)."""
    spot = float(chain_iv["spot"].iloc[0])
    near = chain_iv.iloc[(chain_iv["strike"] - spot).abs().argsort().iloc[:4]]
    near = near[(near["strike"] == near["strike"].iloc[
        (near["strike"] - spot).abs().argsort().iloc[0]]) & near["iv"].notna()]
    if near.empty:
        return float("nan"), float("nan")
    return float(near["iv"].mean()), float(near["strike"].iloc[0])


def parity_diagnostics(chain):
    """Put-call parity check: regress (C - P) on K across strikes.

    Under BSM, C - P = e^{-rT} (F - K): slope gives the discount factor,
    intercept/slope the implied forward F. Deviations of F from spot (or
    of the fitted discount from e^{-rT}) quantify how far the feed's
    settlement prints sit from textbook parity -- a data-quality signal
    most chain tools never show.
    """
    piv = chain.pivot_table(index="strike", columns="kind",
                            values="premium").dropna()
    if len(piv) < 5 or "CE" not in piv or "PE" not in piv:
        return {}
    d = (piv["CE"] - piv["PE"]).values
    K = piv.index.values.astype(float)
    A = np.vstack([np.ones_like(K), K]).T
    a, b = np.linalg.lstsq(A, d, rcond=None)[0]
    resid = d - (a + b * K)
    spot = float(chain["spot"].iloc[0])
    T = float(chain["T"].iloc[0])
    return {
        "implied_forward": float(-a / b) if b != 0 else float("nan"),
        "fitted_discount": float(-b),
        "expected_discount": float(np.exp(-DEFAULT_RATE * T)),
        "residual_std_points": float(np.std(resid)),
        "spot": spot,
        "forward_minus_spot": float(-a / b - spot) if b != 0 else float("nan"),
        "n_strikes": int(len(piv)),
    }


def _gamma_by_strike(chain_iv, r=DEFAULT_RATE, q=0.0):
    """BS gamma per (strike, kind) at that strike's IV; NaN IV falls back
    to the ATM IV so deep strikes still contribute sensibly."""
    out = chain_iv.copy()
    spot = float(out["spot"].iloc[0])
    T = float(out["T"].iloc[0])
    fallback, _ = atm_iv(out)
    iv_used = out["iv"].fillna(fallback)
    out["gamma"] = _g.gamma(spot, out["strike"].values, T, r, q,
                            iv_used.values)
    return out


def gamma_exposure(chain_iv, r=DEFAULT_RATE):
    """Net dealer gamma exposure per strike (₹), plus cumulative profile.

    Returns a DataFrame indexed by strike with gex_rs and cum_gex_rs.
    """
    g = _gamma_by_strike(chain_iv, r=r)
    spot = float(g["spot"].iloc[0])
    lot = float(g["lot_size"].iloc[0])
    oi_contracts = g["oi"] / lot
    sign = np.where(g["kind"] == "CE", 1.0, -1.0)
    g["gex_rs"] = sign * g["gamma"] * oi_contracts * lot * spot**2 * 0.01
    by_strike = g.groupby("strike")["gex_rs"].sum().sort_index()
    out = by_strike.to_frame()
    out["cum_gex_rs"] = out["gex_rs"].cumsum()
    return out


def gamma_flip_level(gex_df):
    """Strike where cumulative net GEX crosses zero (linear interpolation),
    i.e. the level separating positive- from negative-gamma regimes."""
    strikes = gex_df.index.values.astype(float)
    cum = gex_df["cum_gex_rs"].values
    for i in range(1, len(cum)):
        if cum[i - 1] == 0:
            return float(strikes[i - 1])
        if cum[i - 1] * cum[i] < 0:
            frac = cum[i - 1] / (cum[i - 1] - cum[i])
            return float(strikes[i - 1] + frac * (strikes[i] - strikes[i - 1]))
    return float("nan")


def max_pain(chain):
    """Max pain strike: the expiry level minimising total intrinsic payout
    to option holders, weighted by open interest."""
    ce = chain[chain["kind"] == "CE"]
    pe = chain[chain["kind"] == "PE"]
    strikes = np.sort(chain["strike"].unique())
    pain = []
    for level in strikes:
        payout = (ce["oi"] * np.maximum(level - ce["strike"], 0)).sum() \
            + (pe["oi"] * np.maximum(pe["strike"] - level, 0)).sum()
        pain.append(payout)
    pain = np.array(pain, dtype=float)
    return float(strikes[int(np.argmin(pain))])


def straddle_series(day_frames, expiry, r=DEFAULT_RATE):
    """Track the ATM straddle of one fixed expiry across trading days.

    ``day_frames``: list of raw bhavcopy DataFrames (one per day, any
    symbols). For each day the strike nearest that day's spot is used
    (rolling ATM -- the standard theta-bleed exhibit, because desks roll
    to the money). Also records the fixed strike chosen on day 1 so the
    pure time decay of one contract pair can be compared.

    Returns a DataFrame: date, spot, days_to_expiry, atm_strike, ce, pe,
    straddle, straddle_pct_change, fixed_strike, fixed_ce, fixed_pe,
    fixed_straddle.
    """
    rows = []
    fixed_strike = None
    for df in day_frames:
        chain = extract_chain(df, expiry=expiry, min_days=0)
        spot = float(chain["spot"].iloc[0])
        strikes = np.sort(chain["strike"].unique())
        atm = float(strikes[np.argmin(np.abs(strikes - spot))])
        if fixed_strike is None:
            fixed_strike = atm

        def prem(strike, kind):
            row = chain[(chain["strike"] == strike) & (chain["kind"] == kind)]
            return float(row["premium"].iloc[0]) if len(row) else float("nan")

        ce, pe = prem(atm, "CE"), prem(atm, "PE")
        fce, fpe = prem(fixed_strike, "CE"), prem(fixed_strike, "PE")
        rows.append({
            "date": chain["trade_date"].iloc[0],
            "spot": spot,
            "days_to_expiry": int(chain["days_to_expiry"].iloc[0]),
            "atm_strike": atm,
            "ce": ce, "pe": pe, "straddle": ce + pe,
            "fixed_strike": fixed_strike,
            "fixed_ce": fce, "fixed_pe": fpe,
            "fixed_straddle": fce + fpe,
        })
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["straddle_pct_change"] = out["straddle"].pct_change() * 100.0
    out["fixed_straddle_pct_change"] = out["fixed_straddle"].pct_change() * 100.0
    return out
