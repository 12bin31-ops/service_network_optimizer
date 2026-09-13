"""리포트 생성용 컨텍스트 조립.

에이전트든 템플릿이든 **같은 근거표**를 본다. 서술의 출처를 하나로 묶어
"에이전트가 지어낸 숫자"가 리포트에 섞이는 것을 막는다.
"""

from __future__ import annotations

import pandas as pd

from snx.config import Settings
from snx.demand.model import demand_drivers
from snx.storage.db import read_sql, read_table


def build_analysis_context(settings: Settings) -> pd.DataFrame:
    """지역 단위 통합 근거표 (regions × demand × coverage × gap × siting)."""
    db = settings.db_path
    sql = """
        SELECT
            r.sigungu_code, r.sido, r.sigungu, r.urban_class, r.population,
            r.lat, r.lon,
            d.annual_visits, d.ev_visits, d.hitech_visits, d.warranty_visits, d.paid_visits,
            c.nearest_minutes, c.reachable_centers, c.reachable_capacity,
            c.covered_visits, c.uncovered_visits, c.overflow_visits, c.avg_utilization,
            g.unmet_visits, g.gap_score, g.gap_rank
        FROM regions r
        LEFT JOIN demand   d ON d.sigungu_code = r.sigungu_code
        LEFT JOIN coverage c ON c.sigungu_code = r.sigungu_code
        LEFT JOIN gap_scores g ON g.sigungu_code = r.sigungu_code
        ORDER BY g.gap_rank
    """
    ctx = read_sql(db, sql)

    centers = read_table(db, "service_centers")
    if not centers.empty:
        agg = centers.groupby("sigungu_code").agg(
            center_count=("center_id", "count"), total_bays=("bays", "sum")
        )
        ctx = ctx.merge(agg, on="sigungu_code", how="left")
    ctx["center_count"] = ctx.get("center_count", pd.Series(dtype=float)).fillna(0).astype(int)
    ctx["total_bays"] = ctx.get("total_bays", pd.Series(dtype=float)).fillna(0).astype(int)

    parc = read_table(db, "vehicle_parc")
    if not parc.empty:
        drivers = demand_drivers(parc, settings)[
            ["sigungu_code", "vehicles_total", "vehicles_aged", "aged_vehicle_share"]
        ]
        ctx = ctx.merge(drivers, on="sigungu_code", how="left")

    latest_run = read_sql(db, "SELECT run_id FROM siting_runs ORDER BY created_at DESC LIMIT 1")
    if not latest_run.empty:
        run_id = latest_run.iloc[0]["run_id"]
        siting = read_sql(
            db,
            "SELECT sigungu_code, priority, captured_visits FROM siting_results WHERE run_id = ?",
            (run_id,),
        )
        ctx = ctx.merge(siting, on="sigungu_code", how="left")
    else:
        ctx["priority"] = pd.NA
        ctx["captured_visits"] = pd.NA

    return ctx


def region_brief(ctx: pd.DataFrame, sigungu_code: str) -> dict:
    """단일 지역 요약 dict — 에이전트 도구가 반환하는 단위."""
    row = ctx.loc[ctx["sigungu_code"] == str(sigungu_code)]
    if row.empty:
        return {"error": f"지역코드 {sigungu_code} 를 찾을 수 없습니다"}
    r = row.iloc[0]
    return {
        "지역": f"{r['sido']} {r['sigungu']}",
        "지역코드": r["sigungu_code"],
        "도시등급": r["urban_class"],
        "인구": int(r["population"]),
        "자사_보유대수": _f(r.get("vehicles_total")),
        "노후차량_비중": _pct(r.get("aged_vehicle_share")),
        "연간_정비수요": _f(r.get("annual_visits")),
        "보증수리_비중": _pct(_ratio(r.get("warranty_visits"), r.get("annual_visits"))),
        "기존_거점수": int(r.get("center_count", 0)),
        "총_워크베이": int(r.get("total_bays", 0)),
        "최근접_거점_소요시간_분": _f(r.get("nearest_minutes"), 1),
        "30분내_접근가능_거점수": int(r.get("reachable_centers", 0)),
        "평균_부하율": _pct(r.get("avg_utilization")),
        "미커버_수요": _f(r.get("uncovered_visits")),
        "과부하_수요": _f(r.get("overflow_visits")),
        "미충족_수요_합": _f(r.get("unmet_visits")),
        "갭스코어": _f(r.get("gap_score"), 2),
        "갭순위": int(r["gap_rank"]) if pd.notna(r.get("gap_rank")) else None,
        "신규거점_선정여부": bool(pd.notna(r.get("priority"))),
        "신규거점_흡수수요": _f(r.get("captured_visits")),
    }


def _ratio(num, den):
    if num is None or den is None or pd.isna(num) or pd.isna(den) or float(den) == 0:
        return None
    return float(num) / float(den)


def _f(value, digits: int = 0):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _pct(value):
    if value is None or pd.isna(value):
        return None
    return f"{float(value) * 100:.1f}%"
