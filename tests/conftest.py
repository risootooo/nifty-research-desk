"""Shared fixtures: src-layout path setup and a FAKE snapshot.

The fake snapshot follows the compute-layer contract
(``nifty_desk.desk.build_snapshot``) with plausible values for a calm
November 2026 session.  It is deliberately *internally consistent* —
no sceptic check should fire on it — so tests can tell the difference
between a quiet day and a planted tension.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CHART_FILES = ["regime.png", "vol.png", "gex.png", "smile.png"]


def make_fake_snapshot() -> dict:
    return {
        "asof_index_date": "2026-11-11",
        "chain_date": "2026-11-11",
        "expiry": "2026-11-17",
        "spot": 25910.25,
        "nifty_last_close": 25910.25,
        "nifty_ret_5d": 0.0124,
        "nifty_ret_21d": -0.0213,
        "regime": {
            "p_high_vol_today": 0.07,
            "low_vol_ann": 0.108,
            "high_vol_ann": 0.289,
            "dur_low_days": 38.3,
            "dur_high_days": 6.8,
            "share_high_vol_1y": 0.16,
        },
        "vol": {
            "realized_21d_ann": 0.131,
            "garch_forecast_5d_ann": [0.126, 0.127, 0.128, 0.129, 0.130],
            "garch_params": {"omega": 0.0000029, "alpha": 0.07, "beta": 0.91},
        },
        "chain": {
            "net_gex_cr_per_1pct": 412.6,
            "gamma_flip": 25750.0,
            "max_pain": 26000.0,
            "pc_oi_ratio": 0.87,
            "atm_iv": 0.112,
            "skew_minus10": 4.6,
            "total_oi": 18452310,
        },
        "banknifty": {
            "last_close": 58120.40,
            "ret_21d": -0.0089,
            "realized_21d_ann": 0.142,
        },
        # Series are consumed by the charts layer; small stand-ins here.
        "regime_series": [0.05, 0.06, 0.07, 0.07],
        "vol_series": {"realized": [0.128, 0.131], "forecast": [0.126, 0.127]},
        "gex_profile": [
            {"strike": 25750.0, "gex": -120.5},
            {"strike": 26000.0, "gex": 260.25},
        ],
        "smile": [
            {"strike": 23319.0, "iv": 0.158},
            {"strike": 25910.0, "iv": 0.112},
        ],
    }


@pytest.fixture()
def fake_snapshot() -> dict:
    return copy.deepcopy(make_fake_snapshot())
