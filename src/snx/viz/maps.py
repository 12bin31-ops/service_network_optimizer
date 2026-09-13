"""지도 시각화 — 갭 · 기존 거점(품질 등급) · 신규 거점 · 증설 대상을 한 장에 올린다."""

from __future__ import annotations

import folium
import pandas as pd

from snx.config import Settings
from snx.report.context import build_analysis_context
from snx.storage.db import read_sql, read_table, table_exists

KOREA_CENTER = (36.4, 127.9)

GAP_COLORS = [
    (80, "#7f1d1d"),
    (60, "#b91c1c"),
    (45, "#ea580c"),
    (30, "#f59e0b"),
    (0, "#94a3b8"),
]


GRADE_COLORS = {"A": "#1d4ed8", "B": "#0891b2", "C": "#d97706", "D": "#b91c1c"}


def gap_color(score: float) -> str:
    for threshold, color in GAP_COLORS:
        if score >= threshold:
            return color
    return GAP_COLORS[-1][1]


def build_gap_map(settings: Settings, ctx: pd.DataFrame | None = None) -> folium.Map:
    ctx = build_analysis_context(settings) if ctx is None else ctx
    centers = read_table(settings.db_path, "service_centers")
    if table_exists(settings.db_path, "center_quality"):
        quality = read_table(settings.db_path, "center_quality")
        if not quality.empty:
            centers = centers.merge(
                quality[["center_id", "quality_grade", "issue_type", "utilization", "voc_per_1k_jobs"]],
                on="center_id",
                how="left",
            )

    fmap = folium.Map(location=KOREA_CENTER, zoom_start=7, tiles="OpenStreetMap")

    layer_gap = folium.FeatureGroup(name="갭 스코어 (시군구)", show=True)
    layer_centers = folium.FeatureGroup(name="기존 서비스 거점 (색 = 품질 등급)", show=True)
    layer_new = folium.FeatureGroup(name="신규 거점 우선순위", show=True)
    layer_expand = folium.FeatureGroup(name="증설 대상 거점 (투자안)", show=False)

    max_unmet = max(float(ctx["unmet_visits"].fillna(0).max()), 1.0)
    for r in ctx.itertuples(index=False):
        score = float(r.gap_score or 0)
        unmet = float(r.unmet_visits or 0)
        radius = 6 + 22 * (unmet / max_unmet) ** 0.5
        popup = folium.Popup(
            f"<b>{r.sido} {r.sigungu}</b><br>"
            f"갭순위 {int(r.gap_rank)} · 스코어 {score:.1f}<br>"
            f"연간 수요 {float(r.annual_visits or 0):,.0f}건<br>"
            f"미커버 {float(r.uncovered_visits or 0):,.0f} · 과부하 {float(r.overflow_visits or 0):,.0f}<br>"
            f"최근접 거점 {float(r.nearest_minutes or 0):.0f}분 · 기존 거점 {int(r.center_count)}개",
            max_width=280,
        )
        folium.CircleMarker(
            location=(r.lat, r.lon),
            radius=radius,
            color=gap_color(score),
            fill=True,
            fill_color=gap_color(score),
            fill_opacity=0.55,
            weight=1,
            popup=popup,
            tooltip=f"{r.sigungu} · 갭 {score:.0f}",
        ).add_to(layer_gap)

    for c in centers.to_dict(orient="records"):
        is_hitech = c["center_type"] == "hitech"
        grade = c.get("quality_grade")
        color = GRADE_COLORS.get(grade, "#1d4ed8" if not is_hitech else "#0f766e")
        tip = f"{c['name']} (베이 {c['bays']})"
        if isinstance(grade, str):
            tip += f" · 품질 {grade} · 부하 {c['utilization']:.0%}"
            if c.get("issue_type") and c["issue_type"] != "정상":
                tip += f" · {c['issue_type']}"
        folium.CircleMarker(
            location=(c["lat"], c["lon"]),
            radius=4 if not is_hitech else 6,
            color=color,
            fill=True,
            fill_opacity=0.9,
            weight=0 if not is_hitech else 2,
            tooltip=tip,
        ).add_to(layer_centers)

    picked = ctx[ctx["priority"].notna()].sort_values("priority")
    for r in picked.itertuples(index=False):
        folium.Marker(
            location=(r.lat, r.lon),
            icon=folium.Icon(color="green", icon="plus", prefix="fa"),
            tooltip=f"신규 {int(r.priority)}순위 · {r.sido} {r.sigungu} "
            f"(흡수 {float(r.captured_visits or 0):,.0f}건)",
        ).add_to(layer_new)

    if table_exists(settings.db_path, "plan_actions"):
        expands = read_sql(
            settings.db_path,
            """
            SELECT pa.target_id, pa.added_bays, pa.captured_visits, sc.name, sc.lat, sc.lon
            FROM plan_actions pa
            JOIN service_centers sc ON sc.center_id = pa.target_id
            WHERE pa.action = 'expand'
              AND pa.run_id = (SELECT run_id FROM plan_runs ORDER BY created_at DESC LIMIT 1)
            """,
        )
        for r in expands.itertuples(index=False):
            folium.RegularPolygonMarker(
                location=(r.lat, r.lon),
                number_of_sides=4,
                radius=5 + 2 * int(r.added_bays),
                color="#7c3aed",
                fill=True,
                fill_opacity=0.35,
                weight=1.5,
                tooltip=f"증설 +{int(r.added_bays)}베이 · {r.name} (흡수 {r.captured_visits:,.0f}건)",
            ).add_to(layer_expand)

    for layer in (layer_gap, layer_centers, layer_new, layer_expand):
        layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    _add_legend(fmap)
    return fmap


def _add_legend(fmap: folium.Map) -> None:
    html = """
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
                background: rgba(255,255,255,.94); padding: 10px 12px;
                border: 1px solid #cbd5e1; border-radius: 8px;
                font: 12px/1.5 -apple-system, 'Malgun Gothic', sans-serif;">
      <div style="font-weight:600; margin-bottom:6px;">갭 스코어</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:#7f1d1d;border-radius:50%"></span> 80+ 최우선</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:#b91c1c;border-radius:50%"></span> 60–80</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:#ea580c;border-radius:50%"></span> 45–60</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:#f59e0b;border-radius:50%"></span> 30–45</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:#94a3b8;border-radius:50%"></span> 30 미만</div>
      <div style="margin-top:6px; color:#64748b;">원 크기 = 미충족 수요</div>
      <div style="font-weight:600; margin:8px 0 4px;">거점 품질 등급</div>
      <div><span style="color:#1d4ed8">●</span> A <span style="color:#0891b2">●</span> B
           <span style="color:#d97706">●</span> C <span style="color:#b91c1c">●</span> D</div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(html))


def save_gap_map(settings: Settings, filename: str = "gap_map.html") -> str:
    path = settings.outputs_dir / filename
    build_gap_map(settings).save(str(path))
    return str(path)
