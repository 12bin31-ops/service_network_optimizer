"""엔드투엔드 통합 테스트 — 전 단계가 이어붙는지 확인."""

from __future__ import annotations

from pathlib import Path

from snx.storage.db import read_table, row_count
from snx.viz.maps import save_gap_map


def test_all_tables_populated(settings, pipeline_run):
    for table in (
        "regions", "vehicle_parc", "service_centers", "center_voc", "demand", "coverage",
        "gap_scores", "center_quality", "siting_runs", "plan_runs", "plan_actions",
    ):
        assert row_count(settings.db_path, table) > 0, f"{table} 이 비어 있습니다"


def test_every_region_has_demand_and_gap(settings, pipeline_run):
    regions = read_table(settings.db_path, "regions")
    for table in ("demand", "coverage", "gap_scores"):
        rows = read_table(settings.db_path, table)
        assert set(rows["sigungu_code"]) == set(regions["sigungu_code"])


def test_national_unmet_is_material_but_not_total(pipeline_run):
    """합성 데이터가 '분석할 가치가 있는' 갭을 실제로 만들어내는지 검증."""
    gaps = pipeline_run["gaps"]
    coverage = pipeline_run["coverage"]
    total = coverage["covered_visits"].sum() + gaps["unmet_visits"].sum()
    ratio = gaps["unmet_visits"].sum() / total
    assert 0.01 < ratio < 0.60, f"미충족 비율이 비현실적입니다: {ratio:.2%}"


def test_map_renders(settings, pipeline_run):
    path = Path(save_gap_map(settings, filename="test_map.html"))
    assert path.exists()
    html = path.read_text(encoding="utf-8")
    assert "folium" in html.lower() or "leaflet" in html.lower()
    assert path.stat().st_size > 10_000
