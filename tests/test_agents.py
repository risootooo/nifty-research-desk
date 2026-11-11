"""Tests for the editorial layer: analyst, sceptic, editor, number guard."""

from __future__ import annotations

import re

from nifty_desk import agents
from nifty_desk.agents import AnalystAgent, EditorAgent, ScepticAgent

from conftest import CHART_FILES


def compose(snapshot, flags=None):
    findings = AnalystAgent().findings(snapshot)
    if flags is None:
        flags = ScepticAgent().check(snapshot, findings)
    return EditorAgent().compose(snapshot, findings, flags, CHART_FILES), findings, flags


# ---------------------------------------------------------------------------
# Analyst
# ---------------------------------------------------------------------------


def test_findings_cover_all_five_panels(fake_snapshot):
    findings = AnalystAgent().findings(fake_snapshot)
    assert [f["panel"] for f in findings] == list(AnalystAgent.PANELS)
    for f in findings:
        assert f["headline"] and f["detail"] and f["numbers"]


def test_findings_are_built_from_snapshot_values(fake_snapshot):
    # Change a value and the prose must change with it — nothing hardcoded.
    before = {f["panel"]: f["detail"] for f in AnalystAgent().findings(fake_snapshot)}
    fake_snapshot["regime"]["p_high_vol_today"] = 0.85
    after = {f["panel"]: f["detail"] for f in AnalystAgent().findings(fake_snapshot)}
    assert before["Regime"] != after["Regime"]
    assert "85.0%" in after["Regime"]  # detail formats the probability at one decimal


# ---------------------------------------------------------------------------
# Number guard
# ---------------------------------------------------------------------------


def test_composed_brief_passes_number_guard(fake_snapshot):
    brief, _, _ = compose(fake_snapshot)
    assert agents.verify_numbers(brief, fake_snapshot) == []


def test_corrupted_brief_fails_number_guard(fake_snapshot):
    brief, _, _ = compose(fake_snapshot)
    spot_text = agents.fmt_price(fake_snapshot["spot"])
    assert spot_text in brief
    corrupted = brief.replace(spot_text, agents.fmt_price(fake_snapshot["spot"] + 137.0), 1)
    offenders = agents.verify_numbers(corrupted, fake_snapshot)
    assert offenders, "guard must reject a brief with an invented spot level"
    assert fake_snapshot["spot"] + 137.0 in offenders


def test_invented_percentage_fails_number_guard(fake_snapshot):
    brief, _, _ = compose(fake_snapshot)
    corrupted = brief.replace(
        "No material disagreements today.",
        "No material disagreements today. Sensex breadth improved 63.4 percent.",
    )
    assert agents.verify_numbers(corrupted, fake_snapshot) != []


# ---------------------------------------------------------------------------
# Sceptic
# ---------------------------------------------------------------------------


def test_sceptic_silent_on_consistent_snapshot(fake_snapshot):
    findings = AnalystAgent().findings(fake_snapshot)
    assert ScepticAgent().check(fake_snapshot, findings) == []


def test_sceptic_flags_calm_regime_with_rising_vol_path(fake_snapshot):
    # Calm regime call (p = 0.07 in the fake) but the GARCH path climbs
    # 26% across the five-day horizon — a planted tension.
    fake_snapshot["vol"]["garch_forecast_5d_ann"] = [0.100, 0.106, 0.113, 0.119, 0.126]
    findings = AnalystAgent().findings(fake_snapshot)
    flags = ScepticAgent().check(fake_snapshot, findings)
    assert [f["check"] for f in flags] == ["regime-vs-vol-path"]
    tension = flags[0]["tension"]
    assert "10.0%" in tension and "12.6%" in tension  # both sides of the tension
    # The flagged brief must still pass the number guard.
    brief = EditorAgent().compose(fake_snapshot, findings, flags, CHART_FILES)
    assert agents.verify_numbers(brief, fake_snapshot) == []


def test_sceptic_flags_long_gamma_with_steep_skew(fake_snapshot):
    fake_snapshot["chain"]["net_gex_cr_per_1pct"] = 4856.0
    fake_snapshot["chain"]["skew_minus10"] = 9.4
    findings = AnalystAgent().findings(fake_snapshot)
    flags = ScepticAgent().check(fake_snapshot, findings)
    assert "gamma-vs-skew" in [f["check"] for f in flags]


def test_sceptic_flags_model_underreacting(fake_snapshot):
    fake_snapshot["vol"]["realized_21d_ann"] = 0.240  # > 1.5 x forecast 0.126
    findings = AnalystAgent().findings(fake_snapshot)
    flags = ScepticAgent().check(fake_snapshot, findings)
    assert "realized-vs-forecast" in [f["check"] for f in flags]


def test_sceptic_flags_distant_max_pain(fake_snapshot):
    fake_snapshot["chain"]["max_pain"] = 27500.0  # > 3% away from spot
    findings = AnalystAgent().findings(fake_snapshot)
    flags = ScepticAgent().check(fake_snapshot, findings)
    assert "max-pain-distance" in [f["check"] for f in flags]


# ---------------------------------------------------------------------------
# Editor / brief structure
# ---------------------------------------------------------------------------


def test_brief_contains_all_panels_dates_and_footer(fake_snapshot):
    brief, _, _ = compose(fake_snapshot)
    for panel in AnalystAgent.PANELS:
        assert f"## {panel}" in brief
    # Both data-date stamps (index as-of and chain date) and the expiry.
    assert fake_snapshot["asof_index_date"] in brief
    assert fake_snapshot["chain_date"] in brief
    assert fake_snapshot["expiry"] in brief
    # At-a-glance essentials.
    assert agents.fmt_price(fake_snapshot["spot"]) in brief
    assert agents.fmt_level(fake_snapshot["chain"]["gamma_flip"]) in brief
    # Not-advice footer.
    assert "not investment advice" in brief
    assert "end-of-day" in brief.lower()
    # Disagreement section present, quiet-day wording.
    assert "## Where the desk's own numbers disagree" in brief
    assert "No material disagreements today." in brief


def test_brief_links_all_four_charts(fake_snapshot):
    brief, _, _ = compose(fake_snapshot)
    for name in CHART_FILES:
        assert f"assets/{name}" in brief


def test_brief_lists_flags_when_present(fake_snapshot):
    fake_snapshot["vol"]["garch_forecast_5d_ann"] = [0.100, 0.106, 0.113, 0.119, 0.126]
    brief, _, flags = compose(fake_snapshot)
    assert flags
    assert "No material disagreements today." not in brief
    assert flags[0]["tension"] in brief


# ---------------------------------------------------------------------------
# Polish stub
# ---------------------------------------------------------------------------


def test_polish_is_noop_without_env(monkeypatch):
    monkeypatch.delenv("NIFTY_DESK_LLM", raising=False)
    assert agents.polish("some text 12.3") == "some text 12.3"


def test_polish_is_noop_even_when_enabled_without_key(monkeypatch):
    monkeypatch.setenv("NIFTY_DESK_LLM", "1")
    for var in ("NIFTY_DESK_LLM_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert agents.polish("some text 12.3") == "some text 12.3"


def test_number_guard_ignores_prose_without_numerals(fake_snapshot):
    assert agents.verify_numbers("A quiet morning on the desk.", fake_snapshot) == []


def test_brief_has_no_unguarded_stray_numbers(fake_snapshot):
    # Belt-and-braces: every numeral extracted by an independent regex
    # sweep (not the guard's own) must be one the guard accepts.
    brief, _, _ = compose(fake_snapshot)
    tokens = re.findall(r"\d[\d,]*(?:\.\d+)?", brief)
    assert tokens, "brief should contain numbers — the formatted snapshot values"
    assert agents.verify_numbers(brief, fake_snapshot) == []
