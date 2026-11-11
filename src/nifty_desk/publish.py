"""Publisher for the NIFTY Research Desk.

Usage::

    python -m nifty_desk.publish --chain <bhavcopy.csv> [--date YYYY-MM-DD]

Pipeline:

1. Load index history and the options chain (``nifty_desk.data``),
2. build the day's snapshot (``nifty_desk.desk.build_snapshot``),
3. render the four charts into ``briefs/assets/`` (``nifty_desk.charts``),
4. run the three editorial agents over the snapshot,
5. **verify every number in the composed brief against the snapshot** —
   if :func:`nifty_desk.agents.verify_numbers` finds a single numeral
   the engines did not produce, publishing aborts loudly and nothing is
   written,
6. write ``briefs/<asof>.md``, regenerate the static ``docs/index.html``
   dashboard, and refresh the LATEST-BRIEF block in ``README.md``.

Compute-layer API (as landed — the original build contract described
``render_all`` / single-history signatures that the finished modules do
not have, so this publisher codes against the real ones):

* ``data.fetch_all()`` -> ``{"NIFTY": df, "BANKNIFTY": df}`` of Yahoo
  end-of-day histories (cached on disk by the data layer itself).
* ``data.load_chain(path)`` -> the extracted NIFTY chain DataFrame.
* ``desk.build_snapshot(nifty_df, banknifty_df, chain, expiry_choice)``
  -> the snapshot dict.  ``--date``, when given, is passed as
  ``expiry_choice`` (an ISO expiry date instead of ``"nearest"``).
* ``charts`` exposes one function per chart —
  ``regime_chart`` / ``vol_chart`` / ``gex_chart`` / ``smile_chart``,
  each ``(snapshot, out_path) -> Path``.

The compute-layer imports are done lazily inside :func:`main` so the
pure helpers in this module (README block, index.html builder) can be
imported and tested without the compute layer present.
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import agents

Snapshot = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
BRIEFS_DIR = REPO_ROOT / "briefs"
ASSETS_DIR = BRIEFS_DIR / "assets"
DOCS_DIR = REPO_ROOT / "docs"
README_PATH = REPO_ROOT / "README.md"

MARKER_START = "<!--LATEST-BRIEF-->"
MARKER_END = "<!--/LATEST-BRIEF-->"


# ---------------------------------------------------------------------------
# README latest-brief block (pure helpers — unit-tested)
# ---------------------------------------------------------------------------


def latest_brief_block(snapshot: Snapshot, brief_filename: str) -> str:
    """The compact summary inserted between the README markers."""
    c = snapshot["chain"]
    r = snapshot["regime"]
    fc = snapshot["vol"]["garch_forecast_5d_ann"]
    state = agents.regime_state(float(r["p_high_vol_today"]))
    lines = [
        MARKER_START,
        f"### Latest brief — {snapshot['asof_index_date']}",
        "",
        f"![Dealer gamma chart](briefs/assets/gex.png)",
        "",
        f"- **NIFTY spot:** {agents.fmt_price(snapshot['spot'])} "
        f"({agents.fmt_signed_pct(snapshot['nifty_ret_5d'])} over five sessions, "
        f"{agents.fmt_signed_pct(snapshot['nifty_ret_21d'])} over 21 sessions)",
        f"- **Regime:** {state} — high-volatility probability "
        f"{agents.fmt_pct(r['p_high_vol_today'])}",
        f"- **Volatility:** realised {agents.fmt_pct(snapshot['vol']['realized_21d_ann'])} "
        f"(21 sessions); GARCH forecast {agents.fmt_pct(fc[0])} for the next session",
        f"- **Dealer gamma:** net GEX {agents.fmt_signed_cr(c['net_gex_cr_per_1pct'])} "
        f"per one-percent move; gamma flip {agents.fmt_level(c['gamma_flip'])}; "
        f"max pain {agents.fmt_level(c['max_pain'])} for the {snapshot['expiry']} expiry",
        f"- **Smile:** ATM IV {agents.fmt_pct(c['atm_iv'])}, downside skew "
        f"{agents.fmt_points(c['skew_minus10'])} points",
        "",
        f"[Read the full brief](briefs/{brief_filename}) · "
        f"[All briefs](briefs/) · [Dashboard](docs/index.html)",
        MARKER_END,
    ]
    return "\n".join(lines)


def update_readme_text(readme_text: str, block: str) -> str:
    """Replace the text between the LATEST-BRIEF markers with ``block``."""
    start = readme_text.index(MARKER_START)
    end = readme_text.index(MARKER_END) + len(MARKER_END)
    return readme_text[:start] + block + readme_text[end:]


# ---------------------------------------------------------------------------
# Static dashboard (pure helper — unit-tested)
# ---------------------------------------------------------------------------

_INDEX_CSS = """
:root { color-scheme: dark; }
body { background:#0d1117; color:#e6edf3; font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif; margin:0; }
main { max-width:980px; margin:0 auto; padding:32px 20px 64px; }
h1 { font-size:28px; margin:0 0 4px; } h2 { font-size:20px; margin:36px 0 12px; border-bottom:1px solid #30363d; padding-bottom:6px; }
.sub { color:#8b949e; margin:0 0 8px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:20px 0; }
.stat { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:12px 14px; }
.stat .k { color:#8b949e; font-size:12px; text-transform:uppercase; letter-spacing:.05em; }
.stat .v { font-size:20px; margin-top:4px; }
img.chart { width:100%; background:#161b22; border:1px solid #30363d; border-radius:8px; margin:8px 0 20px; }
.flag { background:#2d1f12; border:1px solid #6e4b16; border-radius:8px; padding:10px 14px; margin:8px 0; }
.ok { background:#12241a; border:1px solid #1f5130; border-radius:8px; padding:10px 14px; margin:8px 0; }
a { color:#58a6ff; } li { margin:4px 0; } .foot { color:#8b949e; font-size:13px; margin-top:48px; }
""".strip()


def build_index_html(
    snapshot: Snapshot,
    findings: Sequence[Mapping[str, Any]],
    flags: Sequence[Mapping[str, Any]],
    chart_files: Sequence[str],
    brief_filenames: Sequence[str],
) -> str:
    """Render docs/index.html: stats, charts, findings, flags, archive."""
    c = snapshot["chain"]
    r = snapshot["regime"]
    fc = snapshot["vol"]["garch_forecast_5d_ann"]
    esc = html.escape

    stats = [
        ("NIFTY spot", agents.fmt_price(snapshot["spot"])),
        ("Regime high-vol prob.", agents.fmt_pct(r["p_high_vol_today"])),
        ("GARCH forecast (next)", agents.fmt_pct(fc[0])),
        ("Realised vol (21d)", agents.fmt_pct(snapshot["vol"]["realized_21d_ann"])),
        ("Net GEX / 1% move", agents.fmt_signed_cr(c["net_gex_cr_per_1pct"])),
        ("Gamma flip", agents.fmt_level(c["gamma_flip"])),
        ("Max pain", agents.fmt_level(c["max_pain"])),
        ("ATM IV", agents.fmt_pct(c["atm_iv"])),
    ]
    stat_html = "\n".join(
        f'<div class="stat"><div class="k">{esc(k)}</div>'
        f'<div class="v">{esc(v)}</div></div>'
        for k, v in stats
    )
    chart_html = "\n".join(
        f'<img class="chart" src="../briefs/assets/{esc(name)}" alt="{esc(name)}">'
        for name in chart_files
    )
    findings_html = "\n".join(
        f"<li><strong>{esc(f['headline'])}</strong></li>" for f in findings
    )
    if flags:
        flags_html = "\n".join(
            f'<div class="flag">{esc(f["tension"])}</div>' for f in flags
        )
    else:
        flags_html = '<div class="ok">No material disagreements today.</div>'
    archive_html = "\n".join(
        f'<li><a href="../briefs/{esc(name)}">{esc(name)}</a></li>'
        for name in brief_filenames
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NIFTY Research Desk — {esc(snapshot['asof_index_date'])}</title>
<style>{_INDEX_CSS}</style>
</head>
<body>
<main>
<h1>NIFTY Research Desk</h1>
<p class="sub">Open-source AI analyst desk for Indian markets — brief of {esc(snapshot['asof_index_date'])}.
Index data as of {esc(snapshot['asof_index_date'])}; options chain dated {esc(snapshot['chain_date'])}, expiry {esc(snapshot['expiry'])}.</p>
<div class="grid">
{stat_html}
</div>
<h2>Charts</h2>
{chart_html}
<h2>Findings</h2>
<ul>
{findings_html}
</ul>
<h2>Where the desk's own numbers disagree</h2>
{flags_html}
<h2>Brief archive</h2>
<ul>
{archive_html}
</ul>
<p class="foot">Research and education only — not investment advice. All data is end-of-day.
Charts are referenced from <code>briefs/assets/</code>; if you are viewing this page outside the
repository tree, open the briefs themselves for the full charts and method.</p>
</main>
</body>
</html>
"""


def collect_brief_filenames(briefs_dir: Path) -> list[str]:
    """All published brief file names, newest first."""
    if not briefs_dir.exists():
        return []
    return sorted((p.name for p in briefs_dir.glob("*.md")), reverse=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nifty_desk.publish",
        description="Publish today's NIFTY Research Desk brief.",
    )
    parser.add_argument(
        "--chain",
        required=True,
        help="Path to the NSE F&O bhavcopy CSV for the chain date.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Optional expiry date (YYYY-MM-DD) for the chain; "
        "default is the nearest live expiry.",
    )
    args = parser.parse_args(argv)

    # Lazy imports: the compute layer is a separate module set (API in
    # the module docstring); importing it lazily keeps this module's
    # helpers importable and testable on their own.
    from . import charts
    from .data import fetch_all, load_chain
    from .desk import build_snapshot

    histories = fetch_all()
    chain = load_chain(args.chain)
    snapshot = build_snapshot(
        histories["NIFTY"],
        histories["BANKNIFTY"],
        chain,
        expiry_choice=args.date or "nearest",
    )

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    chart_files = [
        Path(charts.regime_chart(snapshot, ASSETS_DIR / "regime.png")).name,
        Path(charts.vol_chart(snapshot, ASSETS_DIR / "vol.png")).name,
        Path(charts.gex_chart(snapshot, ASSETS_DIR / "gex.png")).name,
        Path(charts.smile_chart(snapshot, ASSETS_DIR / "smile.png")).name,
    ]

    analyst = agents.AnalystAgent()
    findings = analyst.findings(snapshot)
    flags = agents.ScepticAgent().check(snapshot, findings)
    brief = agents.EditorAgent().compose(snapshot, findings, flags, chart_files)

    offenders = agents.verify_numbers(brief, snapshot)
    if offenders:
        print(
            "PUBLISH ABORTED: number-guard verification failed.\n"
            "The composed brief contains numerals that do not match any "
            f"value in the snapshot: {offenders}\n"
            "Nothing was published. Fix the template or the snapshot "
            "before re-running — the desk never prints an unverified number.",
            file=sys.stderr,
        )
        return 1

    brief_filename = f"{snapshot['asof_index_date']}.md"
    (BRIEFS_DIR / brief_filename).write_text(brief, encoding="utf-8")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    index_html = build_index_html(
        snapshot,
        findings,
        flags,
        chart_files,
        collect_brief_filenames(BRIEFS_DIR),
    )
    (DOCS_DIR / "index.html").write_text(index_html, encoding="utf-8")

    readme = README_PATH.read_text(encoding="utf-8")
    README_PATH.write_text(
        update_readme_text(readme, latest_brief_block(snapshot, brief_filename)),
        encoding="utf-8",
    )

    print(
        f"Published briefs/{brief_filename} "
        f"(charts: {', '.join(chart_files)}; sceptic flags: {len(flags)})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
