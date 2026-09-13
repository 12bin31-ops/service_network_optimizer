"""지도 시각화 — 갭 · 기존 거점 · 신규 거점 후보를 한 장에 올린다."""

from __future__ import annotations

import folium
import pandas as pd

from snx.config import Settings
from snx.report.context import build_analysis_context
from snx.storage.db import read_table

KOREA_CENTER = (36.4, 127.9)

GAP_COLORS = [
    (80, "#7f1d1d"),
    (60, "#b91c1c"),
    (45, "#ea580c"),
    (30, "#f59e0b"),
    (0, "#94a3b8"),
]


def gap_color(score: float) -> str:
    for threshold, color in GAP_COLORS:
        if score >= threshold:
            return color
    return GAP_COLORS[-1][1]


def build_gap_map(settings: Settings, ctx: pd.DataFrame | None = None) -> folium.Map:
    ctx = build_analysis_context(settings) if ctx is None else ctx
    centers = read_table(settings.db_path, "service_centers")

    fmap = folium.Map(location=KOREA_CENTER, zoom_start=7, tiles="OpenStreetMap")

    layer_gap = folium.FeatureGroup(name="갭 스코어 (시군구)", show=True)
    layer_centers = folium.FeatureGroup(name="기존 서비스 거점", show=True)
    layer_new = folium.FeatureGroup(name="신규 거점 우선순위", show=True)

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

    for c in centers.itertuples(index=False):
        is_hitech = c.center_type == "hitech"
        folium.CircleMarker(
            location=(c.lat, c.lon),
            radius=4 if not is_hitech else 6,
            color="#1d4ed8" if not is_hitech else "#0f766e",
            fill=True,
            fill_opacity=0.9,
            weight=0,
            tooltip=f"{c.name} (베이 {c.bays})",
        ).add_to(layer_centers)

    picked = ctx[ctx["priority"].notna()].sort_values("priority")
    for r in picked.itertuples(index=False):
        folium.Marker(
            location=(r.lat, r.lon),
            icon=folium.Icon(color="green", icon="plus", prefix="fa"),
            tooltip=f"신규 {int(r.priority)}순위 · {r.sido} {r.sigungu} "
            f"(흡수 {float(r.captured_visits or 0):,.0f}건)",
        ).add_to(layer_new)

    for layer in (layer_gap, layer_centers, layer_new):
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
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(html))


def save_gap_map(settings: Settings, filename: str = "gap_map.html") -> str:
    path = settings.outputs_dir / filename
    build_gap_map(settings).save(str(path))
    return str(path)
