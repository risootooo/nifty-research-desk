"""Toy-chain tests for the vendored greeks-lab chain engine.

The max-pain expectation is computed by hand in the test docstring; the GEX
sign test pins the desk's core convention (calls contribute positive dealer
gamma, puts negative).
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from nifty_desk.engines import chain as ch
from nifty_desk.engines.pricing import bs_price

SPOT = 100.0
T_DAYS = 30
LOT = 50


def _toy_chain(oi_by_strike_kind, sigma=0.15):
    """Tidy chain (extract_chain schema) over strikes 90/100/110 with
    Black-Scholes premiums at a flat 15% vol."""
    rows = []
    for (strike, kind), oi in oi_by_strike_kind.items():
        rows.append({
            "trade_date": date(2026, 1, 1), "expiry": date(2026, 1, 31),
            "strike": float(strike), "kind": kind,
            "premium": float(bs_price(SPOT, strike, T_DAYS / 365.0,
                                      0.065, 0.0, sigma, kind)),
            "spot": SPOT, "oi": float(oi), "volume": 0.0,
            "lot_size": float(LOT), "days_to_expiry": T_DAYS,
            "T": T_DAYS / 365.0,
        })
    return pd.DataFrame(rows)


def test_max_pain_hand_computed():
    """Symmetric OI: 10 (thousand) at each of 90/100/110 for CE and PE.

    Payout to holders at settlement level L:
      L=90 : CE 0                    + PE (100-90)*10 + (110-90)*10 = 300
      L=100: CE (100-90)*10 = 100    + PE (110-100)*10 = 100        = 200
      L=110: CE (110-90)*10+(110-100)*10 = 300 + PE 0               = 300
    Minimum is at L=100 -> max pain must be exactly 100.
    """
    oi = {(k, kind): 10_000 for k in (90, 100, 110) for kind in ("CE", "PE")}
    chain = _toy_chain(oi)
    assert ch.max_pain(chain) == 100.0


def test_max_pain_shifts_away_from_call_wall():
    """Heavy CE OI at 90 makes settling above 90 expensive: those calls pay
    (L-90) x 10^7 at any level L > 90, so the payout minimum collapses to
    the lowest candidate level, 90 itself."""
    oi = {(k, kind): 10_000 for k in (90, 100, 110) for kind in ("CE", "PE")}
    oi[(90, "CE")] = 10_000_000
    chain = _toy_chain(oi)
    assert ch.max_pain(chain) == 90.0


def test_gex_sign_convention():
    """A lone call strike must contribute POSITIVE GEX and a lone put
    strike NEGATIVE GEX (dealers short the OI: long call gamma hedges
    against the move, put gamma with it)."""
    calls = ch.add_iv(_toy_chain({(100, "CE"): 100_000}))
    gex_calls = ch.gamma_exposure(calls)
    assert gex_calls["gex_rs"].sum() > 0

    puts = ch.add_iv(_toy_chain({(100, "PE"): 100_000}))
    gex_puts = ch.gamma_exposure(puts)
    assert gex_puts["gex_rs"].sum() < 0


def test_gex_matches_stated_formula():
    """GEX_K must equal (Gamma*OI_contracts)*LOT*S^2*0.01 exactly — the
    convention stated in PROVENANCE.md."""
    from nifty_desk.engines import greeks as g

    chain = ch.add_iv(_toy_chain({(100, "CE"): 100_000}))
    gamma = float(g.gamma(SPOT, 100.0, T_DAYS / 365.0, 0.065, 0.0, 0.15))
    expected = gamma * (100_000 / LOT) * LOT * SPOT**2 * 0.01
    got = float(ch.gamma_exposure(chain)["gex_rs"].iloc[0])
    assert got == pytest.approx(expected, rel=1e-9)
    assert got > 0
