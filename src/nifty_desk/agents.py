"""Editorial layer of the NIFTY Research Desk: three deterministic agents.

The desk's hard rule is that **no number appears in a published brief
unless it came out of the day's snapshot** — the dict produced by
``nifty_desk.desk.build_snapshot``.  This module enforces that rule in
two places:

* :class:`AnalystAgent` and :class:`EditorAgent` build every sentence
  from snapshot values via the formatting helpers below.  Prose words
  carry no numerals; every digit in the output is a formatted snapshot
  scalar (or a rounding / percent form of one, a window length encoded
  in a field name such as ``nifty_ret_21d``, or a date component).
* :func:`verify_numbers` re-reads a finished brief, extracts every
  numeral in it, and checks each one against the snapshot.  The
  publisher refuses to publish when verification fails — that is the
  desk's anti-hallucination guard, and it is what would catch an LLM
  polish pass (or a human edit) inventing a figure.

The three agents:

* ``AnalystAgent`` — turns the snapshot into structured findings, one
  per panel, each carrying the exact numbers it used.
* ``ScepticAgent`` — cross-checks the panels against each other and
  flags internal tensions (calm regime vs a rising vol path, long
  dealer gamma vs a steep crash premium, ...).
* ``EditorAgent`` — composes the publishable markdown brief.

Everything here is deterministic and offline: no external APIs, no
randomness.  An optional LLM polish hook exists (:func:`polish`) but is
env-gated and a no-op stub — and even if it were implemented, its
output would still have to pass :func:`verify_numbers` before the
publisher writes anything.
"""

from __future__ import annotations

import os
import re
from datetime import date
from typing import Any, Iterable, Mapping, Sequence

Snapshot = Mapping[str, Any]
Finding = dict[str, Any]
Flag = dict[str, Any]

# ---------------------------------------------------------------------------
# Formatting helpers — the ONLY way numbers enter brief prose.
# Each helper renders one snapshot scalar; verify_numbers() knows how to
# recognise exactly these renderings (raw, rounded, percent, absolute).
# ---------------------------------------------------------------------------


def fmt_price(x: float) -> str:
    """Index level with two decimals, e.g. ``25,910.25``."""
    return f"{x:,.2f}"


def fmt_level(x: float) -> str:
    """A strike-like level rounded to whole points, e.g. ``26,000``."""
    return f"{x:,.0f}"


def fmt_pct(frac: float, digits: int = 1) -> str:
    """A decimal fraction as a percent, e.g. ``0.112`` -> ``11.2%``."""
    return f"{frac * 100:.{digits}f}%"


def fmt_signed_pct(frac: float, digits: int = 1) -> str:
    """A return as a signed percent, e.g. ``-0.0213`` -> ``-2.1%``."""
    return f"{frac * 100:+.{digits}f}%"


def fmt_cr(x: float) -> str:
    """A rupee-crore amount rounded to whole crores, e.g. ``₹4,856 cr``."""
    sign = "-" if x < 0 else ""
    return f"{sign}₹{abs(x):,.0f} cr"


def fmt_signed_cr(x: float) -> str:
    """A rupee-crore amount with an explicit sign for gains/losses."""
    sign = "+" if x > 0 else ""
    return f"{sign}{fmt_cr(x)}"


def fmt_ratio(x: float) -> str:
    """A unitless ratio with two decimals, e.g. ``0.87``."""
    return f"{x:.2f}"


def fmt_days(x: float) -> str:
    """An expected duration rounded to whole days, e.g. ``38``."""
    return f"{x:.0f}"


def fmt_points(x: float) -> str:
    """A difference quoted in points, one decimal, e.g. ``4.6``."""
    return f"{x:.1f}"


# ---------------------------------------------------------------------------
# Regime vocabulary (words only — thresholds live in the sceptic/analyst
# logic, never printed as numerals in the brief).
# ---------------------------------------------------------------------------


def regime_state(p_high_vol: float) -> str:
    """Short label for the regime implied by today's high-vol probability."""
    if p_high_vol < 0.30:
        return "Calm (low-volatility regime)"
    if p_high_vol < 0.70:
        return "Transitional (regime unresolved)"
    return "Turbulent (high-volatility regime)"


# ---------------------------------------------------------------------------
# Analyst
# ---------------------------------------------------------------------------


class AnalystAgent:
    """Produces one structured finding per panel, from snapshot values only.

    A finding is ``{"panel", "headline", "detail", "numbers"}`` where
    ``numbers`` lists the raw snapshot values the prose was built from —
    kept so a reader (or a test) can audit every sentence.
    """

    PANELS = ("Regime", "Volatility", "Dealer gamma / positioning", "Smile", "BankNifty")

    def findings(self, snapshot: Snapshot) -> list[Finding]:
        return [
            self._regime(snapshot),
            self._volatility(snapshot),
            self._gamma(snapshot),
            self._smile(snapshot),
            self._banknifty(snapshot),
        ]

    # -- panels -----------------------------------------------------------

    def _regime(self, s: Snapshot) -> Finding:
        r = s["regime"]
        p = float(r["p_high_vol_today"])
        state = regime_state(p)
        if p < 0.30:
            headline = f"Regime: calm — high-volatility probability just {fmt_pct(p, 0)}"
        elif p < 0.70:
            headline = f"Regime: unresolved — high-volatility probability {fmt_pct(p, 0)}"
        else:
            headline = f"Regime: turbulent — high-volatility probability {fmt_pct(p, 0)}"
        detail = (
            f"The two-state Markov-switching model puts the probability of the "
            f"high-volatility state today at {fmt_pct(p)}. In the calm state "
            f"annualised volatility runs at {fmt_pct(r['low_vol_ann'])} with an "
            f"expected duration of {fmt_days(r['dur_low_days'])} sessions; in the "
            f"turbulent state it runs at {fmt_pct(r['high_vol_ann'])} with an "
            f"expected duration of {fmt_days(r['dur_high_days'])} sessions. Over "
            f"the past year the index spent {fmt_pct(r['share_high_vol_1y'], 0)} of "
            f"sessions in the high-volatility state. Current state: {state}."
        )
        return {
            "panel": "Regime",
            "headline": headline,
            "detail": detail,
            "numbers": [
                p,
                float(r["low_vol_ann"]),
                float(r["high_vol_ann"]),
                float(r["dur_low_days"]),
                float(r["dur_high_days"]),
                float(r["share_high_vol_1y"]),
            ],
        }

    def _volatility(self, s: Snapshot) -> Finding:
        v = s["vol"]
        fc = [float(x) for x in v["garch_forecast_5d_ann"]]
        realized = float(v["realized_21d_ann"])
        if fc[-1] > fc[0] * 1.02:
            path = "rising"
        elif fc[-1] < fc[0] * 0.98:
            path = "easing"
        else:
            path = "broadly flat"
        if realized > fc[0]:
            gap = "Realised volatility is running above the model's near-term path"
        elif realized < fc[0]:
            gap = "Realised volatility is running below the model's near-term path"
        else:
            gap = "Realised volatility is in line with the model's near-term path"
        article = "an" if path[0] in "aeiou" else "a"
        detail = (
            f"Realised volatility over the last 21 sessions annualises to "
            f"{fmt_pct(realized)}. The GARCH forecast for the next five sessions "
            f"starts at {fmt_pct(fc[0])} and ends at {fmt_pct(fc[-1])} — {article} {path} "
            f"path. {gap}."
        )
        params = v.get("garch_params") or {}
        numbers = [realized, fc[0], fc[-1]]
        # garch_params schema is owned by the compute layer; quote the two
        # persistence parameters when they are present under their usual names.
        if isinstance(params, Mapping) and "alpha" in params and "beta" in params:
            detail += (
                f" The fitted shock and persistence parameters are "
                f"{fmt_ratio(float(params['alpha']))} and "
                f"{fmt_ratio(float(params['beta']))} respectively."
            )
            numbers += [float(params["alpha"]), float(params["beta"])]
        return {
            "panel": "Volatility",
            "headline": f"Volatility: realised {fmt_pct(realized)}, forecast path {path}",
            "detail": detail,
            "numbers": numbers,
        }

    def _gamma(self, s: Snapshot) -> Finding:
        c = s["chain"]
        spot = float(s["spot"])
        gex = float(c["net_gex_cr_per_1pct"])
        flip = float(c["gamma_flip"])
        max_pain = float(c["max_pain"])
        if gex > 0:
            posture = (
                "dealers are net long gamma, a posture that usually dampens "
                "moves — hedging flows lean against the trend and ranges tend "
                "to be sold"
            )
        elif gex < 0:
            posture = (
                "dealers are net short gamma, a posture that usually amplifies "
                "moves — hedging flows chase the trend in both directions"
            )
        else:
            posture = "dealer gamma is balanced, so hedging flows are neutral"
        if spot > flip:
            side = "above"
            flip_note = "spot sits above the flip, in the zone where dealer hedging is normally stabilising"
        elif spot < flip:
            side = "below"
            flip_note = "spot sits below the flip, in the zone where dealer hedging can amplify moves"
        else:
            side = "at"
            flip_note = "spot sits exactly on the flip"
        if max_pain > spot:
            pain_note = "max pain sits above spot, so the expiry pin — the level at which aggregate option-holder losses peak — is overhead"
        elif max_pain < spot:
            pain_note = "max pain sits below spot, so the expiry pin — the level at which aggregate option-holder losses peak — is underneath"
        else:
            pain_note = "max pain sits exactly at spot"
        detail = (
            f"Net dealer gamma exposure is {fmt_signed_cr(gex)} for a one-percent "
            f"move in the index: {posture}. The gamma flip level is "
            f"{fmt_level(flip)} and spot at {fmt_price(spot)} is {side} it — "
            f"{flip_note}. Max pain for the {s['expiry']} expiry is "
            f"{fmt_level(max_pain)}; {pain_note}. The put-call open-interest "
            f"ratio is {fmt_ratio(c['pc_oi_ratio'])} across total open interest "
            f"of {int(c['total_oi']):,} lots."
        )
        return {
            "panel": "Dealer gamma / positioning",
            "headline": (
                f"Positioning: net GEX {fmt_signed_cr(gex)} per one-percent move, "
                f"flip at {fmt_level(flip)}"
            ),
            "detail": detail,
            "numbers": [
                gex,
                flip,
                max_pain,
                float(c["pc_oi_ratio"]),
                float(c["total_oi"]),
                spot,
            ],
        }

    def _smile(self, s: Snapshot) -> Finding:
        c = s["chain"]
        atm = float(c["atm_iv"])
        skew = float(c["skew_minus10"])
        if skew > 8:
            tone = (
                "that is a steep crash premium — downside protection is being "
                "bid hard relative to the at-the-money level"
            )
        elif skew > 4:
            tone = "a firm downside premium, in line with a market paying up for protection"
        elif skew > 0:
            tone = "a mild downside premium — protection costs a little more than the at-the-money level"
        else:
            tone = "no downside premium at all, which is unusual for an equity index"
        detail = (
            f"At-the-money implied volatility is {fmt_pct(atm)}. The implied "
            f"volatility of the option struck ten percent below spot sits "
            f"{fmt_points(skew)} points above that level: {tone}."
        )
        return {
            "panel": "Smile",
            "headline": f"Smile: ATM IV {fmt_pct(atm)}, downside skew {fmt_points(skew)} points",
            "detail": detail,
            "numbers": [atm, skew],
        }

    def _banknifty(self, snapshot: Snapshot) -> Finding:
        b = snapshot["banknifty"]
        n = snapshot["vol"]
        b_vol = float(b["realized_21d_ann"])
        n_vol = float(n["realized_21d_ann"])
        if b_vol > n_vol:
            rel = "running hotter than the headline index"
        elif b_vol < n_vol:
            rel = "running calmer than the headline index"
        else:
            rel = "running in line with the headline index"
        detail = (
            f"BANKNIFTY closed at {fmt_price(b['last_close'])}, a "
            f"{fmt_signed_pct(b['ret_21d'])} move over the last 21 sessions. Its "
            f"realised volatility over the same window annualises to "
            f"{fmt_pct(b_vol)}, {rel} (NIFTY realised {fmt_pct(n_vol)})."
        )
        return {
            "panel": "BankNifty",
            "headline": (
                f"BankNifty: {fmt_signed_pct(b['ret_21d'])} over 21 sessions, "
                f"realised vol {fmt_pct(b_vol)}"
            ),
            "detail": detail,
            "numbers": [float(b["last_close"]), float(b["ret_21d"]), b_vol, n_vol],
        }


# ---------------------------------------------------------------------------
# Sceptic
# ---------------------------------------------------------------------------

#: Net GEX above this (₹ cr per one-percent move) counts as "strongly
#: positive" for the positioning/skew tension check.  A judgement call by
#: the desk, documented here; the threshold itself is never printed in a
#: brief (only snapshot numbers are).
GEX_STRONG_CR = 1000.0


class ScepticAgent:
    """Cross-checks the desk's own panels against each other.

    Returns a list of flags — possibly empty.  Each flag is
    ``{"check", "tension", "numbers"}``: one line explaining the
    disagreement, built from both of the numbers that disagree.
    """

    def check(self, snapshot: Snapshot, findings: Sequence[Finding]) -> list[Flag]:
        # `findings` is accepted so the sceptic can be extended to check
        # prose claims later; today's checks run on the snapshot itself,
        # which is the single source the findings were derived from.
        del findings
        s = snapshot
        flags: list[Flag] = []

        p_high = float(s["regime"]["p_high_vol_today"])
        fc = [float(x) for x in s["vol"]["garch_forecast_5d_ann"]]
        realized = float(s["vol"]["realized_21d_ann"])
        gex = float(s["chain"]["net_gex_cr_per_1pct"])
        skew = float(s["chain"]["skew_minus10"])
        spot = float(s["spot"])
        max_pain = float(s["chain"]["max_pain"])

        # 1) Calm regime call, but the GARCH path climbs steeply.
        if p_high < 0.30 and fc[0] > 0 and fc[-1] > fc[0] * 1.20:
            flags.append(
                {
                    "check": "regime-vs-vol-path",
                    "tension": (
                        f"The regime model calls calm (high-volatility "
                        f"probability {fmt_pct(p_high)}), yet the GARCH path "
                        f"climbs from {fmt_pct(fc[0])} to {fmt_pct(fc[-1])} "
                        f"across the next five sessions — the two models "
                        f"disagree about how quiet the week will be."
                    ),
                    "numbers": [p_high, fc[0], fc[-1]],
                }
            )

        # 2) Dealers long gamma (dampening) but a steep crash premium.
        if gex > GEX_STRONG_CR and skew > 8:
            flags.append(
                {
                    "check": "gamma-vs-skew",
                    "tension": (
                        f"Dealers are strongly long gamma (net GEX "
                        f"{fmt_signed_cr(gex)} per one-percent move), which "
                        f"usually dampens swings — yet the downside skew is "
                        f"priced at {fmt_points(skew)} points, a crash premium "
                        f"that calm positioning alone does not explain."
                    ),
                    "numbers": [gex, skew],
                }
            )

        # 3) Realised vol far above the GARCH forecast: model under-reacting.
        if fc[0] > 0 and realized > 1.5 * fc[0]:
            flags.append(
                {
                    "check": "realized-vs-forecast",
                    "tension": (
                        f"Realised volatility of {fmt_pct(realized)} is far "
                        f"above the GARCH near-term forecast of "
                        f"{fmt_pct(fc[0])} — the model is under-reacting to "
                        f"the current tape and its path should be read with "
                        f"that lag in mind."
                    ),
                    "numbers": [realized, fc[0]],
                }
            )

        # 4) Max pain well away from spot: an expiry pull worth noting.
        if spot > 0 and abs(max_pain - spot) / spot > 0.03:
            flags.append(
                {
                    "check": "max-pain-distance",
                    "tension": (
                        f"Max pain for the {s['expiry']} expiry sits at "
                        f"{fmt_level(max_pain)} against spot "
                        f"{fmt_price(spot)} — far enough away that the expiry "
                        f"pin, if it asserts itself, pulls against the "
                        f"current level."
                    ),
                    "numbers": [max_pain, spot],
                }
            )

        return flags


# ---------------------------------------------------------------------------
# Editor
# ---------------------------------------------------------------------------

#: Chart file each panel leads with, when the chart was rendered.
PANEL_CHARTS = {
    "Regime": "regime.png",
    "Volatility": "vol.png",
    "Dealer gamma / positioning": "gex.png",
    "Smile": "smile.png",
}


def _long_date(iso: str) -> str:
    """``2026-11-11`` -> ``11 November 2026`` (numerals stay date-derived)."""
    y, m, d = (int(part) for part in iso.split("-"))
    return date(y, m, d).strftime("%d %B %Y")


class EditorAgent:
    """Composes the publishable markdown brief from findings and flags."""

    def compose(
        self,
        snapshot: Snapshot,
        findings: Sequence[Finding],
        flags: Sequence[Flag],
        chart_files: Sequence[str],
    ) -> str:
        s = snapshot
        c = s["chain"]
        r = s["regime"]
        fc = [float(x) for x in s["vol"]["garch_forecast_5d_ann"]]
        charts = set(chart_files)
        by_panel = {f["panel"]: f for f in findings}

        lines: list[str] = []
        lines.append(
            f"# NIFTY Research Desk — Daily Brief · {_long_date(s['asof_index_date'])}"
        )
        lines.append("")
        lines.append(
            "An open-source AI analyst desk for Indian markets: regime, "
            "volatility, dealer gamma and the options smile on NIFTY & "
            "BANKNIFTY, computed from end-of-day data and published free "
            "every morning."
        )
        lines.append("")
        lines.append("## At a glance")
        lines.append("")
        lines.append("| Metric | Reading |")
        lines.append("|---|---|")
        lines.append(f"| NIFTY spot | {fmt_price(s['spot'])} |")
        lines.append(
            f"| Regime | {regime_state(float(r['p_high_vol_today']))} — "
            f"high-volatility probability {fmt_pct(r['p_high_vol_today'])} |"
        )
        lines.append(
            f"| NIFTY return, five sessions | {fmt_signed_pct(s['nifty_ret_5d'])} |"
        )
        lines.append(
            f"| NIFTY return, 21 sessions | {fmt_signed_pct(s['nifty_ret_21d'])} |"
        )
        lines.append(
            f"| GARCH vol forecast, next session | {fmt_pct(fc[0])} annualised |"
        )
        lines.append(
            f"| Realised vol, 21 sessions | {fmt_pct(s['vol']['realized_21d_ann'])} annualised |"
        )
        lines.append(
            f"| Net dealer GEX | {fmt_signed_cr(c['net_gex_cr_per_1pct'])} per one-percent move |"
        )
        lines.append(f"| Gamma flip | {fmt_level(c['gamma_flip'])} |")
        lines.append(f"| Max pain (expiry {s['expiry']}) | {fmt_level(c['max_pain'])} |")
        lines.append(f"| ATM implied vol | {fmt_pct(c['atm_iv'])} |")
        lines.append("")

        for panel in AnalystAgent.PANELS:
            finding = by_panel[panel]
            lines.append(f"## {panel}")
            lines.append("")
            chart = PANEL_CHARTS.get(panel)
            if chart and chart in charts:
                lines.append(f"![{panel} chart](assets/{chart})")
                lines.append("")
            lines.append(f"**{finding['headline']}**")
            lines.append("")
            lines.append(finding["detail"])
            lines.append("")

        lines.append("## Where the desk's own numbers disagree")
        lines.append("")
        if flags:
            for flag in flags:
                lines.append(f"- {flag['tension']}")
        else:
            lines.append("No material disagreements today.")
        lines.append("")

        lines.append("## Method and data")
        lines.append("")
        lines.append(
            f"Index data as of {s['asof_index_date']} (Yahoo Finance end-of-day "
            f"closes). Options chain dated {s['chain_date']} (NSE end-of-day "
            f"bhavcopy, settlement marks), expiry {s['expiry']}. The regime "
            f"panel is a two-state Gaussian Markov-switching model (Hamilton) "
            f"fitted by expectation-maximisation; the volatility panel is a "
            f"GARCH forecast beside realised volatility; dealer gamma, the "
            f"gamma flip and max pain are computed from open interest and "
            f"Black-Scholes greeks across the chain; the smile is read from "
            f"implied volatilities inverted strike by strike. Every number in "
            f"this brief is produced by those engines and is machine-checked "
            f"against the underlying snapshot before publication — the desk's "
            f"agents may phrase the numbers, but they may not invent them."
        )
        lines.append("")
        lines.append("## Limitations")
        lines.append("")
        lines.append(
            "- End-of-day data only: nothing here is intraday, and the chain "
            "is a settlement snapshot, not a live book."
        )
        lines.append(
            "- Settlement marks for illiquid strikes are exchange-computed "
            "theoretical prices, so the far wings of the smile are indicative."
        )
        lines.append(
            "- The models are deliberately textbook and low-dimensional: two "
            "regimes, one volatility factor, one-factor gamma. They simplify "
            "the market to make its state readable, at the cost of everything "
            "they leave out."
        )
        lines.append(
            "- Gamma exposure rests on the convention that option sellers are "
            "predominantly dealers; positioning estimates are inferences, "
            "not observations."
        )
        lines.append("")
        lines.append("---")
        lines.append(
            "*Research and education only — not investment advice. This brief "
            "describes the state of the market as measured by the desk's "
            "models; it is not a recommendation to buy or sell any security. "
            "All data is end-of-day.*"
        )
        lines.append("")

        return polish("\n".join(lines))


# ---------------------------------------------------------------------------
# Optional LLM polish — env-gated stub.
# ---------------------------------------------------------------------------


def polish(text: str) -> str:
    """Optional LLM copy-polish hook.  Currently a deliberate no-op.

    Returns ``text`` unchanged unless the environment opts in with
    ``NIFTY_DESK_LLM=1`` *and* an API key is present (looked up in
    ``NIFTY_DESK_LLM_API_KEY``, ``ANTHROPIC_API_KEY`` or
    ``OPENAI_API_KEY``).  Even then, this stub performs **no network
    calls** and returns the text unchanged — it marks the single place a
    real implementation would slot in.  Any future implementation must
    preserve every numeral verbatim: the publisher runs
    :func:`verify_numbers` on the composed brief and refuses to publish
    if a polished number no longer matches the snapshot.
    """
    if os.environ.get("NIFTY_DESK_LLM") != "1":
        return text
    key = (
        os.environ.get("NIFTY_DESK_LLM_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not key:
        return text
    # Stub interface only: a real LLM polish call would be made here.
    return text


# ---------------------------------------------------------------------------
# The number guard.
# ---------------------------------------------------------------------------

_NUMERAL_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _collect_snapshot_values(snapshot: Any) -> list[float]:
    """Every numeric the brief is allowed to print, in raw form.

    Collects: all numeric scalars anywhere in the snapshot (including
    inside the chart series), the digit groups encoded in field names
    (window lengths such as the ``21`` in ``realized_21d_ann`` or the
    ``10`` in ``skew_minus10``), and the components of any ISO date
    strings (as-of date, chain date, expiry).
    """
    values: list[float] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, bool):
            return
        if isinstance(obj, (int, float)):
            values.append(float(obj))
        elif isinstance(obj, Mapping):
            for key, val in obj.items():
                for group in re.findall(r"\d+", str(key)):
                    values.append(float(group))
                walk(val)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                walk(item)
        elif isinstance(obj, str):
            # Strings in a snapshot are dates/timestamps (as-of date,
            # chain date, expiry, series dates) — whitelist their digit
            # groups so both "2026-11-17" and "2026-11-17 00:00:00"
            # renderings verify.
            for group in re.findall(r"\d+", obj):
                values.append(float(group))

    walk(snapshot)
    return values


def _printable_forms(values: Iterable[float]) -> set[float]:
    """All renderings of the raw values the formatting helpers can emit."""
    forms: set[float] = set()
    for v in values:
        for transformed in (v, v * 100.0):  # raw, and percent form
            for digits in (0, 1, 2, 3):
                forms.add(round(transformed, digits))
    return forms


def verify_numbers(brief_text: str, snapshot: Snapshot) -> list[float]:
    """Check every numeral in ``brief_text`` against the snapshot.

    Returns the list of offending numerals (as floats, commas stripped);
    an empty list means the brief passed.  A numeral passes if it equals
    a snapshot value, a percent form of one (value × 100), or a rounding
    of either at up to three decimals — exactly the forms the desk's
    formatting helpers produce — or a window length encoded in a field
    name, or a component of one of the snapshot's date strings.
    """
    forms = _printable_forms(_collect_snapshot_values(snapshot))
    offenders: list[float] = []
    for token in _NUMERAL_RE.findall(brief_text):
        try:
            n = float(token.replace(",", ""))
        except ValueError:
            continue
        if n in forms or -n in forms:
            continue
        if any(round(n, d) in forms or round(-n, d) in forms for d in range(4)):
            continue
        offenders.append(n)
    return offenders
