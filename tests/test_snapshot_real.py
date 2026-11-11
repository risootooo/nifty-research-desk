"""End-to-end snapshot assembly on the real cached index history and the
real 3 Nov 2026 NSE bhavcopy — the same file behind greeks-lab's verified
case study (net GEX +Rs 12,619 cr, gamma flip 23,872, max pain 23,600 on
the 9 Nov 2026 expiry, spot 23,810.85). The chain assertions cross-check
the vendored engine against those published numbers.
"""

from pathlib import Path

import numpy as np
import pytest

from nifty_desk import charts, data
from nifty_desk.desk import build_snapshot

BHAVCOPY = Path(
    "/home/hatch/workspace/trading/india-options-paper/data/bhavcopy/"
    "BhavCopy_NSE_FO_0_0_0_20261103_F_0000.csv"
)
SNAPSHOT_KEYS = {
    "asof_index_date", "chain_date", "expiry", "spot", "nifty_last_close",
    "nifty_ret_5d", "nifty_ret_21d", "regime", "vol", "chain", "banknifty",
    "regime_series", "vol_series", "gex_profile", "smile",
}


@pytest.fixture(scope="module")
def snapshot():
    nifty = data.load_index_history("NIFTY")
    bank = data.load_index_history("BANKNIFTY")
    if not BHAVCOPY.exists():
        pytest.skip("real bhavcopy not present on this machine")
    return build_snapshot(nifty, bank, BHAVCOPY)


def test_snapshot_contract(snapshot):
    assert set(snapshot) == SNAPSHOT_KEYS
    assert snapshot["chain_date"] == "2026-11-03"
    assert len(snapshot["vol"]["garch_forecast_5d_ann"]) == 5
    assert len(snapshot["regime_series"]["dates"]) == \
        len(snapshot["regime_series"]["p_high"])
    assert len(snapshot["gex_profile"]["strikes"]) == \
        len(snapshot["gex_profile"]["gex_cr"])
    assert 0.0 <= snapshot["regime"]["p_high_vol_today"] <= 1.0
    assert snapshot["regime"]["low_vol_ann"] < snapshot["regime"]["high_vol_ann"]


def test_chain_matches_greeks_lab_verified_case(snapshot):
    """Cross-check against greeks-lab CASES.md (9 Nov 2026 expiry)."""
    assert snapshot["expiry"] == "2026-11-09"
    assert snapshot["spot"] == pytest.approx(23810.85)
    assert snapshot["chain"]["net_gex_cr_per_1pct"] == pytest.approx(12619, rel=0.01)
    assert snapshot["chain"]["gamma_flip"] == pytest.approx(23872, rel=0.005)
    assert snapshot["chain"]["max_pain"] == pytest.approx(23600)
    assert snapshot["chain"]["atm_iv"] > 0
    assert snapshot["chain"]["total_oi"] > 0


def test_charts_render(snapshot, tmp_path):
    for fn, name in [(charts.regime_chart, "regime.png"),
                     (charts.vol_chart, "vol.png"),
                     (charts.gex_chart, "gex.png"),
                     (charts.smile_chart, "smile.png")]:
        out = fn(snapshot, tmp_path / name)
        assert out.exists() and out.stat().st_size > 10_000
