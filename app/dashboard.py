"""Service Network Optimizer 대시보드 (Streamlit).

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

st.set_page_config(page_title="Service Network Optimizer", page_icon="🔧", layout="wide")

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


@st.cache_data(show_spinner=False)
def load_quality() -> pd.DataFrame:
    if not table_exists(SETTINGS.db_path, "center_quality"):
        return pd.DataFrame()
    return read_sql(
        SETTINGS.db_path,
        """
        SELECT q.*, sc.name, sc.center_type, sc.bays, r.sido, r.sigungu
        FROM center_quality q
        JOIN service_centers sc ON sc.center_id = q.center_id
        JOIN regions r ON r.sigungu_code = q.sigungu_code
        """,
    )


@st.cache_data(show_spinner=False)
def load_plan() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not table_exists(SETTINGS.db_path, "plan_runs"):
        return pd.DataFrame(), pd.DataFrame()
    run = read_sql(SETTINGS.db_path, "SELECT * FROM plan_runs ORDER BY created_at DESC LIMIT 1")
    if run.empty:
        return run, pd.DataFrame()
    actions = read_sql(
        SETTINGS.db_path,
        """
        SELECT pa.*, r.sido, r.sigungu, sc.name
        FROM plan_actions pa
        JOIN regions r ON r.sigungu_code = pa.sigungu_code
        LEFT JOIN service_centers sc ON sc.center_id = pa.target_id
        WHERE pa.run_id = ? ORDER BY pa.priority
        """,
        (run.iloc[0]["run_id"],),
    )
    return run, actions


@st.cache_data(show_spinner="시나리오 계산 중…")
def run_scenario(overrides: tuple[tuple[str, float], ...]):
    from snx.policy.scenario import simulate_loyalty

    return simulate_loyalty(SETTINGS, dict(overrides))


def main() -> None:
    st.title("Service Network Optimizer")
    st.caption(
        "차령 기반 정비 수요 추정 → 30분 상권 커버리지 진단 → 거점 품질(VOC) 진단 → "
        "신설 · 증설 투자 최적화 → 서비스 정책 시나리오"
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

    warranty = float(view["warranty_visits"].fillna(0).sum()) if "warranty_visits" in view else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("연간 정비 수요", f"{total_demand:,.0f}건")
    c2.metric(
        "미충족 수요",
        f"{total_unmet:,.0f}건",
        f"{100 * total_unmet / max(total_demand, 1):.1f}%",
        delta_color="inverse",
    )
    c3.metric("30분 접근 공백 지역", f"{blanks}곳")
    c4.metric("과부하 상권", f"{overloaded}곳")
    c5.metric("보증수리 비중", f"{100 * warranty / max(total_demand, 1):.1f}%")

    run = load_run()
    if not run.empty:
        r = run.iloc[0]
        st.info(
            f"신규 거점 **{int(r['n_new_sites'])}개소** 배치 시 미충족 수요 "
            f"**{r['objective_value']:,.0f}건** 흡수 — 전체 미충족의 "
            f"**{100 * r['objective_value'] / max(r['baseline_unmet'], 1):.1f}%** 해소 "
            f"(솔버: {r['solver_status']})"
        )

    tab_map, tab_rank, tab_load, tab_quality, tab_plan, tab_policy, tab_report = st.tabs(
        ["지도", "갭 순위", "거점 부하", "거점 품질", "투자안", "정책 시나리오", "리포트"]
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
            loads = pd.read_csv(load_path, dtype={"sigungu_code": str, "center_id": str})
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

    # --- 거점 품질 ---------------------------------------------------------
    with tab_quality:
        quality = load_quality()
        if quality.empty:
            st.warning("품질 진단 결과가 없습니다 — `snx quality` 를 실행하세요.")
        else:
            from snx.quality.scorecard import PRESCRIPTIONS, load_complaint_correlation

            qv = quality[quality["sido"].isin(picked_sido)]
            counts = qv["issue_type"].value_counts()
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("과부하형 품질저하", f"{counts.get('과부하형 품질저하', 0)}곳", help="붐벼서 불만 → 증설 · 분산")
            k2.metric("운영형 품질저하", f"{counts.get('운영형 품질저하', 0)}곳", help="한가한데 불만 → 품질 점검 · 교육")
            k3.metric("대기 리스크", f"{counts.get('대기 리스크', 0)}곳", help="아직 불만은 없지만 부하 초과")
            rho = load_complaint_correlation(qv)
            k4.metric("부하율–VOC 순위상관", "-" if rho is None else f"{rho:.2f}")

            fig_q = px.scatter(
                qv,
                x="utilization",
                y="voc_per_1k_jobs",
                color="issue_type",
                size="bays",
                hover_data=["name", "quality_grade", "est_wait_days", "comeback_rate"],
                labels={"utilization": "부하율", "voc_per_1k_jobs": "정비 1천 건당 VOC", "issue_type": "원인"},
                title="부하율 × VOC — 같은 불만, 다른 원인",
                color_discrete_map={
                    "과부하형 품질저하": "#b91c1c",
                    "운영형 품질저하": "#7c3aed",
                    "대기 리스크": "#d97706",
                    "정상": "#94a3b8",
                },
            )
            fig_q.add_vline(
                x=SETTINGS.capacity.healthy_utilization, line_dash="dash", annotation_text="건전 부하율"
            )
            st.plotly_chart(fig_q, use_container_width=True)

            issue_pick = st.selectbox(
                "관리 대상 유형", ["과부하형 품질저하", "운영형 품질저하", "대기 리스크", "정상"]
            )
            st.caption(f"처방: {PRESCRIPTIONS.get(issue_pick, '-')}")
            st.dataframe(
                qv[qv["issue_type"] == issue_pick]
                .sort_values("quality_risk", ascending=False)[
                    ["name", "sido", "sigungu", "quality_grade", "quality_risk", "utilization",
                     "est_wait_days", "voc_per_1k_jobs", "comeback_rate"]
                ]
                .rename(
                    columns={
                        "name": "거점", "sido": "시도", "sigungu": "시군구", "quality_grade": "등급",
                        "quality_risk": "리스크", "utilization": "부하율", "est_wait_days": "대기(일)",
                        "voc_per_1k_jobs": "VOC/천건", "comeback_rate": "재입고율",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    # --- 투자안 -------------------------------------------------------------
    with tab_plan:
        plan_run, actions = load_plan()
        if plan_run.empty:
            st.warning("투자안이 없습니다 — `snx plan` 을 실행하세요.")
        else:
            pr = plan_run.iloc[0]
            unit = SETTINGS.investment.cost_unit
            compare = pd.DataFrame(
                {
                    "투자안": ["신설만", "증설만", "신설 + 증설 최적"],
                    "흡수 수요": [pr["new_only_value"], pr["expand_only_value"], pr["objective_value"]],
                }
            )
            left, right = st.columns([2, 3])
            with left:
                st.metric(
                    f"예산 {pr['budget']:,.0f}{unit} 최적 조합",
                    f"{pr['objective_value']:,.0f}건",
                    f"신설만 대비 {pr['objective_value'] / max(pr['new_only_value'], 1):.2f}배",
                )
                st.write(
                    f"신설 **{int(pr['n_new_sites'])}개소** · 증설 **{int(pr['n_expansions'])}곳"
                    f"(+{int(pr['added_bays'])}베이)** · 집행 {pr['spent']:,.0f}{unit}"
                )
                st.caption(
                    f"단가 가정 — 신규 거점 {SETTINGS.investment.new_site_cost}{unit} · "
                    f"증설 {SETTINGS.investment.bay_expansion_cost}{unit}/베이. "
                    "`snx plan --budget <억원>` 으로 예산별 재계산."
                )
            with right:
                st.plotly_chart(
                    px.bar(compare, x="투자안", y="흡수 수요", text_auto=",.0f", title="같은 예산, 다른 수단"),
                    use_container_width=True,
                )
            if not actions.empty:
                actions["대상"] = actions.apply(
                    lambda r: f"{r['sido']} {r['sigungu']} (신설)" if r["action"] == "new" else r["name"], axis=1
                )
                st.dataframe(
                    actions[["priority", "action", "대상", "added_bays", "cost", "captured_visits"]].rename(
                        columns={
                            "priority": "우선순위", "action": "조치", "added_bays": "추가 베이",
                            "cost": f"투자({unit})", "captured_visits": "흡수 수요",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    # --- 정책 시나리오 -------------------------------------------------------
    with tab_policy:
        st.markdown(
            "보증 연장 · 정비 멤버십 같은 서비스 프로그램은 **공식망 유입률**을 올린다. "
            "유입률을 바꿔 보고, 늘어난 입고를 현 서비스망이 감당하는지 확인한다."
        )
        loyalty = SETTINGS.demand.official_network_loyalty
        preset = SETTINGS.policy_scenario.loyalty_overrides
        cols = st.columns(4)
        picked: dict[str, float] = {}
        for col, bucket in zip(cols, ["0-2", "3-5", "6-10", "11+"], strict=True):
            picked[bucket] = col.slider(
                f"{bucket}년 차 유입률",
                0.0,
                1.0,
                float(preset.get(bucket, loyalty[bucket])),
                0.01,
                help=f"기준값 {loyalty[bucket]:.2f}",
            )
        changed = tuple((b, v) for b, v in picked.items() if abs(v - loyalty[b]) > 1e-9)
        if not changed:
            st.info("기준값과 같습니다. 슬라이더로 유입률을 바꿔 보세요.")
        else:
            result = run_scenario(changed)
            sm = result.summary
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("입고 증가", f"{sm['delta_visits']:+,.0f}건")
            m2.metric("미충족 증가", f"{sm['delta_unmet']:+,.0f}건", delta_color="inverse")
            m3.metric("누수율", f"{sm['leak_ratio']:.0%}", help="늘어난 입고 중 대기 · 이탈로 새는 비율")
            m4.metric(
                "과부하 상권",
                f"{sm['overloaded_regions_scenario']}곳",
                f"{sm['overloaded_regions_scenario'] - sm['overloaded_regions_base']:+d}",
                delta_color="inverse",
            )
            top = result.regions.head(15).assign(region=lambda d: d["sido"] + " " + d["sigungu"])
            fig_p = px.bar(
                top.melt(id_vars="region", value_vars=["delta_visits", "delta_unmet"], var_name="구분", value_name="건수")
                .replace({"delta_visits": "입고 증가", "delta_unmet": "미충족 증가"}),
                x="region",
                y="건수",
                color="구분",
                barmode="group",
                labels={"region": ""},
                title="미충족 증가 상위 15개 지역",
            )
            st.plotly_chart(fig_p, use_container_width=True)

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
                file_name="service_network_report.md",
            )


if __name__ == "__main__":
    main()
