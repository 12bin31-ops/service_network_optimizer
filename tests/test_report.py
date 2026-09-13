from __future__ import annotations

import json

from snx.agents import tools as T
from snx.agents.crew import generate_report, is_agent_available
from snx.report.context import build_analysis_context, region_brief
from snx.report.render import render_template_report, save_report


def test_context_has_all_stages(settings, pipeline_run):
    ctx = build_analysis_context(settings)
    for col in ("annual_visits", "nearest_minutes", "gap_score", "gap_rank", "center_count"):
        assert col in ctx.columns
    assert ctx["gap_rank"].notna().all()


def test_region_brief_known_and_unknown(settings, pipeline_run):
    ctx = build_analysis_context(settings)
    code = ctx.iloc[0]["sigungu_code"]
    brief = region_brief(ctx, code)
    assert brief["갭순위"] == 1
    assert "error" in region_brief(ctx, "00000")


def test_template_report_contains_sections(settings, pipeline_run):
    md = render_template_report(settings)
    for heading in ("전국 요약", "갭 유형 판정", "가정과 한계"):
        assert heading in md
    assert len(md) > 500


def test_agent_tools_return_valid_json(settings, pipeline_run):
    for payload in (
        T.national_summary(settings),
        T.top_gap_regions(3, settings),
        T.siting_plan(settings),
        T.methodology_notes(settings),
    ):
        json.loads(payload)


def test_report_falls_back_without_agent(settings, pipeline_run):
    content, generator = generate_report(settings, use_agent=False)
    assert generator == "template"
    assert "전국 요약" in content


def test_agent_availability_reports_reason():
    ok, reason = is_agent_available()
    assert isinstance(ok, bool)
    assert reason


def test_save_report_persists(settings, pipeline_run):
    path = save_report(settings, "# 테스트\n본문", generator="template", filename="t.md")
    assert path.endswith("t.md")
