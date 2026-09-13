"""서비스망 갭 대시보드 (Streamlit).

실행:  streamlit run app/dashboard.py
사전:  snx pipeline --source sample
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snx.config import load_settings  # noqa: E402
from snx.report.context import build_analysis_context  # noqa: E402
from snx.storage.db import read_sql, read_table, table_exists  # noqa: E402
from snx.viz.maps import build_gap_map  # noqa: E402

st.set_page_config(page_title="서비스망 갭 분석", page_icon="🔧", layout="wide")

SETTINGS = load_settings()


@st.cache_data(show_spinner=False)
def load_context() -> pd.DataFrame:
    return build_analysis_context(SETTINGS)


@st.cache_data(show_spinner=False)
def load_centers() -> pd.DataFrame:
    return read_table(SETTINGS.db_path, "service_centers")


@st.cache_data(show_spinner=False)
def load_run() -> pd.DataFrame:
    return read_sql(SETTINGS.db_path, "SELECT * FROM siting_runs ORDER BY created_at DESC LIMIT 1")


def main() -> None:
    st.title("정비 서비스망 커버리지 갭 분석")
    st.caption(
        "차령 구조 기반 정비 수요 추정 → 30분 상권 커버리지 진단 → 용량제약 최대커버링 입지 최적화"
    )

    if not table_exists(SETTINGS.db_path, "gap_scores"):
        st.error("분석 결과가 없습니다. 터미널에서 `snx pipeline --source sample` 을 먼저 실행하세요.")
        st.stop()

    ctx = load_context()
    if ctx.empty:
        st.error("분석 결과가 비어 있습니다. 파이프라인을 다시 실행하세요.")
        st.stop()

    # --- 사이드바 필터 ----------------------------------------------------
    with st.sidebar:
        st.header("필터")
        sidos = sorted(ctx["sido"].unique())
        picked_sido = st.multiselect("시도", sidos, default=sidos)
        urban = st.multiselect(
            "도시등급", ["metro", "city", "rural"], default=["metro", "city", "rural"]
        )
        only_gap = st.checkbox("미충족 수요가 있는 지역만", value=False)
        st.divider()
        st.caption(
            f"접근 기준 {SETTINGS.coverage.max_travel_minutes:.0f}분 · "
            f"건전 부하율 {SETTINGS.capacity.healthy_utilization:.0%}"
        )

    view = ctx[ctx["sido"].isin(picked_sido) & ctx["urban_class"].isin(urban)]
    if only_gap:
        view = view[view["unmet_visits"].fillna(0) > 0]

    # --- KPI --------------------------------------------------------------
    total_demand = float(view["annual_visits"].fillna(0).sum())
    total_unmet = float(view["unmet_visits"].fillna(0).sum())
    blanks = int((view["reachable_centers"].fillna(0) == 0).sum())
    overloaded = int(
        (view["avg_utilization"].fillna(0) > SETTINGS.capacity.healthy_utilization).sum()
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("연간 정비 수요", f"{total_demand:,.0f}건")
    c2.metric(
        "미충족 수요",
        f"{total_unmet:,.0f}건",
        f"{100 * total_unmet / max(total_demand, 1):.1f}%",
        delta_color="inverse",
    )
    c3.metric("30분 접근 공백 지역", f"{blanks}곳")
    c4.metric("과부하 상권", f"{overloaded}곳")

    run = load_run()
    if not run.empty:
        r = run.iloc[0]
        st.info(
            f"신규 거점 **{int(r['n_new_sites'])}개소** 배치 시 미충족 수요 "
            f"**{r['objective_value']:,.0f}건** 흡수 — 전체 미충족의 "
            f"**{100 * r['objective_value'] / max(r['baseline_unmet'], 1):.1f}%** 해소 "
            f"(솔버: {r['solver_status']})"
        )

    tab_map, tab_rank, tab_load, tab_report = st.tabs(
        ["지도", "갭 순위", "거점 부하", "리포트"]
    )

    # --- 지도 -------------------------------------------------------------
    with tab_map:
        try:
            from streamlit_folium import st_folium

            st_folium(build_gap_map(SETTINGS, ctx), height=620, use_container_width=True)
        except ImportError:
            st.components.v1.html(
                build_gap_map(SETTINGS, ctx).get_root().render(), height=620, scrolling=False
            )

    # --- 갭 순위 -----------------------------------------------------------
    with tab_rank:
        left, right = st.columns([3, 2])
        with left:
            cols = [
                "gap_rank", "sido", "sigungu", "annual_visits", "uncovered_visits",
                "overflow_visits", "nearest_minutes", "center_count", "avg_utilization", "gap_score",
            ]
            st.dataframe(
                view[cols]
                .sort_values("gap_rank")
                .rename(
                    columns={
                        "gap_rank": "순위", "sido": "시도", "sigungu": "시군구",
                        "annual_visits": "연간수요", "uncovered_visits": "미커버",
                        "overflow_visits": "과부하", "nearest_minutes": "최근접(분)",
                        "center_count": "거점수", "avg_utilization": "평균부하율",
                        "gap_score": "갭스코어",
                    }
                ),
                use_container_width=True,
                hide_index=True,
                height=520,
            )
        with right:
            top = view.nsmallest(15, "gap_rank")
            fig = px.bar(
                top,
                x="gap_score",
                y=top["sido"] + " " + top["sigungu"],
                orientation="h",
                labels={"gap_score": "갭 스코어", "y": ""},
                title="갭 상위 15개 지역",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=520)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("갭 유형 분해")
        melted = (
            view.nsmallest(15, "gap_rank")
            .assign(region=lambda d: d["sido"] + " " + d["sigungu"])
            .melt(
                id_vars="region",
                value_vars=["uncovered_visits", "overflow_visits"],
                var_name="유형",
                value_name="건수",
            )
            .replace({"uncovered_visits": "접근성 갭", "overflow_visits": "용량 갭"})
        )
        fig2 = px.bar(melted, x="region", y="건수", color="유형", labels={"region": ""})
        st.plotly_chart(fig2, use_container_width=True)

    # --- 거점 부하 ---------------------------------------------------------
    with tab_load:
        load_path = SETTINGS.outputs_dir / "center_load.csv"
        if load_path.exists():
            loads = pd.read_csv(load_path)
            centers = load_centers()[["center_id", "sigungu_code"]]
            loads = loads.merge(centers, on="center_id", how="left", suffixes=("", "_c"))
            loads = loads.merge(
                ctx[["sigungu_code", "sido", "sigungu"]], on="sigungu_code", how="left"
            )
            loads = loads[loads["sido"].isin(picked_sido)]

            fig3 = px.histogram(
                loads, x="utilization", nbins=40, labels={"utilization": "부하율"},
                title="거점 부하율 분포",
            )
            fig3.add_vline(
                x=SETTINGS.capacity.healthy_utilization,
                line_dash="dash",
                annotation_text="건전 부하율",
            )
            st.plotly_chart(fig3, use_container_width=True)

            st.dataframe(
                loads.sort_values("utilization", ascending=False)[
                    ["name", "sido", "sigungu", "center_type", "capacity", "assigned_visits", "utilization"]
                ].head(50),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("center_load.csv 가 없습니다 — `snx analyze` 를 실행하세요.")

    # --- 리포트 ------------------------------------------------------------
    with tab_report:
        rows = read_sql(
            SETTINGS.db_path, "SELECT * FROM reports ORDER BY created_at DESC LIMIT 1"
        )
        if rows.empty:
            st.warning("리포트가 없습니다 — `snx report` 를 실행하세요.")
        else:
            row = rows.iloc[0]
            st.caption(f"생성기: `{row['generator']}` · {row['created_at']}")
            st.markdown(row["content_md"])
            st.download_button(
                "리포트 내려받기 (.md)",
                row["content_md"],
                file_name="service_network_gap_report.md",
            )


if __name__ == "__main__":
    main()
