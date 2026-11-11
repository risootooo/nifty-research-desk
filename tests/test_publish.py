"""Tests for the publisher's pure helpers (README block, index.html).

The end-to-end publish run needs the compute layer
(``nifty_desk.data`` / ``nifty_desk.desk`` / ``nifty_desk.charts``),
which is a separate module set; these tests cover everything in
``nifty_desk.publish`` that does not.
"""

from __future__ import annotations

from nifty_desk import agents, publish

from conftest import CHART_FILES


def _findings_and_flags(snapshot):
    findings = agents.AnalystAgent().findings(snapshot)
    flags = agents.ScepticAgent().check(snapshot, findings)
    return findings, flags


def test_latest_brief_block_is_marker_wrapped_and_numeric(fake_snapshot):
    block = publish.latest_brief_block(fake_snapshot, "2026-11-11.md")
    assert block.startswith(publish.MARKER_START)
    assert block.endswith(publish.MARKER_END)
    assert "briefs/assets/gex.png" in block
    assert "briefs/2026-11-11.md" in block
    assert agents.fmt_price(fake_snapshot["spot"]) in block


def test_update_readme_text_replaces_only_the_block(fake_snapshot):
    readme = (
        "# Title\n\nbefore\n\n"
        + publish.MARKER_START
        + "\nold content\n"
        + publish.MARKER_END
        + "\n\nafter\n"
    )
    block = publish.latest_brief_block(fake_snapshot, "2026-11-11.md")
    updated = publish.update_readme_text(readme, block)
    assert "old content" not in updated
    assert updated.startswith("# Title\n\nbefore\n\n")
    assert updated.endswith("\n\nafter\n")
    assert "Latest brief — 2026-11-11" in updated


def test_current_repo_readme_has_markers():
    readme = publish.README_PATH.read_text(encoding="utf-8")
    assert publish.MARKER_START in readme
    assert publish.MARKER_END in readme


def test_build_index_html(fake_snapshot):
    findings, flags = _findings_and_flags(fake_snapshot)
    page = publish.build_index_html(
        fake_snapshot,
        findings,
        flags,
        CHART_FILES,
        ["2026-11-11.md", "2026-11-10.md"],
    )
    assert "<style>" in page and "http" not in page.split("<style>")[1].split("</style>")[0]
    assert agents.fmt_price(fake_snapshot["spot"]) in page
    for name in CHART_FILES:
        assert f"../briefs/assets/{name}" in page
    for f in findings:
        assert f["headline"] in page
    assert "No material disagreements today." in page
    assert "../briefs/2026-11-11.md" in page
    assert "../briefs/2026-11-10.md" in page
    assert "not investment advice" in page


def test_build_index_html_lists_flags(fake_snapshot):
    fake_snapshot["vol"]["garch_forecast_5d_ann"] = [0.100, 0.106, 0.113, 0.119, 0.126]
    findings, flags = _findings_and_flags(fake_snapshot)
    assert flags
    page = publish.build_index_html(
        fake_snapshot, findings, flags, CHART_FILES, ["2026-11-11.md"]
    )
    assert flags[0]["tension"] in page
    assert "No material disagreements today." not in page


def test_collect_brief_filenames(tmp_path):
    (tmp_path / "2026-11-10.md").write_text("x")
    (tmp_path / "2026-11-11.md").write_text("x")
    (tmp_path / "notes.txt").write_text("x")
    assert publish.collect_brief_filenames(tmp_path) == [
        "2026-11-11.md",
        "2026-11-10.md",
    ]
