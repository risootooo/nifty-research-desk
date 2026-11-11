"""Chart layer for the NIFTY Research Desk.

Four PNG charts, GitHub-dark themed to match the repo README:
background #0d1117, panel #161b22, text #e6edf3, grid #30363d,
accent #58a6ff, green #3fb950, red #f85149. Every function takes the
snapshot dict from :mod:`nifty_desk.desk` plus an output path and saves
a wide PNG at dpi=200.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BG = "#0d1117"
PANEL = "#161b22"
TEXT = "#e6edf3"
GRID = "#30363d"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
RED = "#f85149"
FIGSIZE = (12.0, 4.6)


def _style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def _new_fig():
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=200)
    fig.patch.set_facecolor(BG)
    _style_ax(ax)
    return fig, ax


def _save(fig, out_path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return out


def _dates(strings) -> pd.DatetimeIndex:
    return pd.to_datetime(list(strings))


def regime_chart(snapshot: dict, out_path) -> Path:
    """Smoothed P(high-vol regime), last 3 years, area-filled."""
    dates = _dates(snapshot["regime_series"]["dates"])
    p = np.asarray(snapshot["regime_series"]["p_high"], dtype=float)
    cutoff = dates[-1] - pd.DateOffset(years=3)
    mask = dates >= cutoff
    d, p = dates[mask], p[mask]
    fig, ax = _new_fig()
    ax.fill_between(d, p, color=RED, alpha=0.45)
    ax.plot(d, p, color=RED, linewidth=0.9)
    ax.axhline(0.5, color=TEXT, linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("P(high-vol regime)")
    ax.set_title("NIFTY regime — smoothed probability of the high-volatility state (3y)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    return _save(fig, out_path)


def vol_chart(snapshot: dict, out_path) -> Path:
    """Realised 21d annualised vol (last year) + the 5 GARCH forecast points."""
    dates = _dates(snapshot["vol_series"]["dates"])
    rv = np.asarray(snapshot["vol_series"]["realized_21d"], dtype=float) * 100.0
    cutoff = dates[-1] - pd.DateOffset(years=1)
    mask = dates >= cutoff
    d, rv = dates[mask], rv[mask]
    fig, ax = _new_fig()
    ax.plot(d, rv, color=ACCENT, linewidth=1.2, label="Realised 21d vol (ann.)")
    fc = np.asarray(snapshot["vol"]["garch_forecast_5d_ann"], dtype=float) * 100.0
    fc_dates = pd.bdate_range(d[-1] + pd.Timedelta(days=1), periods=len(fc))
    ax.plot(fc_dates, fc, color=GREEN, linewidth=1.4, marker="o",
            markersize=5, label="GARCH(1,1) forecast — next 5 sessions")
    ax.scatter([d[-1]], [rv[-1]], color=ACCENT, zorder=5)
    ax.set_ylabel("Volatility (% annualised)")
    ax.set_title("NIFTY volatility — realised 21d vs GARCH forecast")
    leg = ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
    leg.get_frame().set_alpha(0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    return _save(fig, out_path)


def gex_chart(snapshot: dict, out_path) -> Path:
    """Net dealer GEX by strike (₹ cr per 1%): green +ve, red −ve bars,
    vertical markers at spot and at the gamma-flip level."""
    strikes = np.asarray(snapshot["gex_profile"]["strikes"], dtype=float)
    gex = np.asarray(snapshot["gex_profile"]["gex_cr"], dtype=float)
    spot = float(snapshot["spot"])
    flip = float(snapshot["chain"]["gamma_flip"])
    width = float(np.median(np.diff(strikes))) * 0.85 if len(strikes) > 1 else 40.0
    fig, ax = _new_fig()
    colors = [GREEN if v >= 0 else RED for v in gex]
    ax.bar(strikes, gex, width=width, color=colors)
    ax.axhline(0, color=TEXT, linewidth=0.8)
    ax.axvline(spot, color=ACCENT, linewidth=1.6,
               label=f"Spot {spot:,.0f}")
    if np.isfinite(flip):
        ax.axvline(flip, color=TEXT, linewidth=1.2, linestyle="--",
                   label=f"Gamma flip {flip:,.0f}")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Net GEX (₹ crore per 1% move)")
    ax.set_title(f"Dealer gamma exposure by strike — expiry {snapshot['expiry']}")
    leg = ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
    leg.get_frame().set_alpha(0.9)
    return _save(fig, out_path)


def smile_chart(snapshot: dict, out_path) -> Path:
    """IV smile: implied vol (%) against moneyness (%), ATM marked."""
    mny = np.asarray(snapshot["smile"]["moneyness_pct"], dtype=float)
    iv = np.asarray(snapshot["smile"]["iv"], dtype=float)
    ok = np.isfinite(iv)
    mny, iv = mny[ok], iv[ok]
    order = np.argsort(mny)
    mny, iv = mny[order], iv[order]
    fig, ax = _new_fig()
    ax.plot(mny, iv, color=ACCENT, linewidth=1.4, marker="o", markersize=4)
    ax.axvline(0, color=TEXT, linewidth=0.8, linestyle="--", alpha=0.7)
    atm_iv = float(snapshot["chain"]["atm_iv"]) * 100.0
    if np.isfinite(atm_iv):
        ax.scatter([0], [atm_iv], color=GREEN, s=70, zorder=5,
                   label=f"ATM IV {atm_iv:.1f}%")
        leg = ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
        leg.get_frame().set_alpha(0.9)
    ax.set_xlabel("Moneyness (strike / spot − 1, %)")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title(f"IV smile — expiry {snapshot['expiry']} (OTM options)")
    return _save(fig, out_path)
