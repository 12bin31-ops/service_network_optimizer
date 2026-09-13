"""리포트 렌더링.

LLM 키가 없어도 저장소가 항상 완주하도록, 결정론적 템플릿 리포트를 기본값으로
둔다. 에이전트 리포트는 이 템플릿을 '사실 기반'으로 삼아 서술을 덧붙인다.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pandas as pd

from snx.config import Settings
from snx.report.context import build_analysis_context
from snx.storage.db import get_conn, read_sql, read_table, table_exists


def render_template_report(settings: Settings, ctx: pd.DataFrame | None = None) -> str:
    ctx = build_analysis_context(settings) if ctx is None else ctx
    top_n = settings.report.top_n_regions

    total_demand = float(ctx["annual_visits"].fillna(0).sum())
    total_unmet = float(ctx["unmet_visits"].fillna(0).sum())
    blank_regions = int((ctx["reachable_centers"].fillna(0) == 0).sum())
    overloaded = int((ctx["avg_utilization"].fillna(0) > settings.capacity.healthy_utilization).sum())

    runs = read_sql(
        settings.db_path,
        "SELECT * FROM siting_runs ORDER BY created_at DESC LIMIT 1",
    )

    lines: list[str] = []
    lines.append("# 정비 서비스망 진단 — Service Network Optimizer")
    lines.append("")
    lines.append(f"_생성일시: {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M')}_")
    lines.append("")
    lines.append("## 1. 전국 요약")
    lines.append("")
    lines.append(f"- 연간 정비 입고 수요: **{total_demand:,.0f}건**")
    lines.append(
        f"- 미충족 수요: **{total_unmet:,.0f}건 ({100 * total_unmet / max(total_demand, 1):.1f}%)**"
    )
    lines.append(f"- 30분 내 접근 가능 거점이 없는 시군구: **{blank_regions}곳**")
    lines.append(
        f"- 건전 부하율({settings.capacity.healthy_utilization:.0%}) 초과 상권: **{overloaded}곳**"
    )
    if "warranty_visits" in ctx.columns:
        warranty = float(ctx["warranty_visits"].fillna(0).sum())
        lines.append(
            f"- 보증수리(무상) 입고: **{warranty:,.0f}건 ({100 * warranty / max(total_demand, 1):.1f}%)** "
            "— 나머지는 유상 정비"
        )
    lines.append("")

    lines.append(f"## 2. 갭 상위 {top_n}개 지역")
    lines.append("")
    lines.append("| 순위 | 지역 | 연간수요 | 미커버 | 과부하 | 최근접(분) | 거점수 | 갭스코어 |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in ctx.head(top_n).itertuples(index=False):
        lines.append(
            f"| {int(r.gap_rank)} | {r.sido} {r.sigungu} | {_n(r.annual_visits)} | "
            f"{_n(r.uncovered_visits)} | {_n(r.overflow_visits)} | "
            f"{_n(r.nearest_minutes, 1)} | {int(r.center_count)} | {_n(r.gap_score, 1)} |"
        )
    lines.append("")

    lines.append("### 갭 유형 판정")
    lines.append("")
    for r in ctx.head(top_n).itertuples(index=False):
        uncovered = float(r.uncovered_visits or 0)
        overflow = float(r.overflow_visits or 0)
        if uncovered > overflow:
            verdict = "**접근성 갭** — 30분 내 도달 가능한 거점이 없음 → 신규 거점 신설"
        elif overflow > 0:
            verdict = (
                "**용량 갭** — 30분 내 접근 가능한 거점이 이미 포화 "
                "→ 기존 거점 워크베이 증설 또는 인근 신설"
            )
        else:
            verdict = "구조적 공백 없음 — 모니터링 대상"
        lines.append(f"- {r.sido} {r.sigungu}: {verdict} (자체 거점 {int(r.center_count)}개)")
    lines.append("")

    if not runs.empty:
        run = runs.iloc[0]
        lines.append("## 3. 신규 거점 우선순위 (용량제약 최대커버링 최적화)")
        lines.append("")
        lines.append(
            f"- 신규 거점 {int(run['n_new_sites'])}개소 배치 시 미충족 수요 "
            f"**{run['objective_value']:,.0f}건** 흡수 "
            f"(전체 미충족의 {100 * run['objective_value'] / max(run['baseline_unmet'], 1):.1f}%)"
        )
        lines.append(f"- 후보지 풀 {int(run['candidate_count'])}곳 · 솔버 상태 `{run['solver_status']}`")

        site_capacity = settings.capacity.annual_capacity(settings.optimization.new_site_bays)
        budget_capacity = site_capacity * max(int(run["n_new_sites"]), 1)
        if run["objective_value"] >= budget_capacity * 0.98:
            lines.append(
                f"- **제약은 후보지가 아니라 예산이다.** 선정된 {int(run['n_new_sites'])}개소가 "
                f"모두 설계 용량({site_capacity:,.0f}건/년)을 100% 채운다. "
                f"거점 수를 늘리면 흡수 수요는 선형으로 더 늘어난다 — "
                f"`snx optimize --p <n>` 으로 투자 규모별 민감도를 확인할 것."
            )
        lines.append("")
        picked = ctx[ctx["priority"].notna()].sort_values("priority")
        if not picked.empty:
            lines.append("| 우선순위 | 지역 | 흡수 수요 | 기존 거점수 | 갭순위 |")
            lines.append("|---:|---|---:|---:|---:|")
            for r in picked.itertuples(index=False):
                lines.append(
                    f"| {int(r.priority)} | {r.sido} {r.sigungu} | {_n(r.captured_visits)} | "
                    f"{int(r.center_count)} | {int(r.gap_rank)} |"
                )
            lines.append("")

        skipped = ctx.head(top_n * 2)
        skipped = skipped[skipped["priority"].isna()]
        if not skipped.empty:
            lines.append("### 갭 상위인데 신설에서 제외된 지역")
            lines.append("")
            lines.append(
                "갭 순위대로 짓는 것과 포트폴리오 최적해는 다르다. "
                "아래 지역은 순위는 높지만 신설 대상이 아니다."
            )
            lines.append("")
            site_cap = settings.capacity.annual_capacity(settings.optimization.new_site_bays)
            for r in skipped.head(4).itertuples(index=False):
                unmet = float(r.unmet_visits or 0)
                overflow = float(r.overflow_visits or 0)
                uncovered = float(r.uncovered_visits or 0)
                if unmet < site_cap:
                    why = (
                        f"미충족 수요 {unmet:,.0f}건이 거점 설계용량 {site_cap:,.0f}건의 "
                        f"{100 * unmet / site_cap:.0f}% 수준 — 신설하면 용량이 남는다. "
                        "같은 예산으로 더 큰 수요를 덮는 후보가 우선되며, "
                        "이 지역은 인근 거점 증설·출장정비·협력 정비소 제휴가 대안"
                    )
                elif overflow > uncovered:
                    why = "용량 갭이 주된 원인이므로 신설보다 기존 거점 워크베이 증설이 비용 효율적"
                else:
                    why = "인접 후보지에 선정된 신규 거점이 이 지역 수요를 이미 흡수"
                lines.append(f"- **{r.sido} {r.sigungu}** (갭 {int(r.gap_rank)}위): {why}")
            lines.append("")

    lines.extend(_quality_section(settings))
    lines.extend(_investment_section(settings))
    lines.extend(_policy_section(settings))

    lines.append("## 7. 가정과 한계")
    lines.append("")
    lines.append(
        "- 소요시간은 직선거리에 우회계수와 도시등급별 평균속도를 적용한 근사치다. "
        "실제 라우팅 API 로 교체하면 산악 지형 지역의 접근성이 더 나쁘게 나올 가능성이 크다."
    )
    lines.append(
        "- 차령별 정비 원단위 · 공식망 유입률 · 보증수리 비중은 시나리오 가정값이다. "
        "실적 데이터로 교체하면 절대 수준은 바뀌지만 지역 간 상대 순위는 유지되는 구조다."
    )
    lines.append(
        "- 신설 · 증설 단가는 부지비를 제외한 가정값이다. 단가 비율이 바뀌면 최적 조합이 달라지므로 "
        "`snx plan --budget` 과 `config/settings.yaml` 의 investment 절로 민감도를 확인할 것."
    )
    lines.append("")
    return "\n".join(lines)


def _quality_section(settings: Settings) -> list[str]:
    from snx.quality.scorecard import (
        ISSUE_OPERATION,
        ISSUE_OVERLOAD,
        ISSUE_WAIT,
        PRESCRIPTIONS,
        load_complaint_correlation,
    )

    db = settings.db_path
    if not table_exists(db, "center_quality"):
        return []
    quality = read_table(db, "center_quality")
    if quality.empty:
        return []
    centers = read_table(db, "service_centers")[["center_id", "name"]]
    quality = quality.merge(centers, on="center_id", how="left")
    counts = quality["issue_type"].value_counts()

    lines = ["## 4. 거점 품질 진단 (부하 · VOC · 재입고)", ""]
    lines.append(
        f"- 진단 거점 {len(quality)}곳 — **과부하형 품질저하 {counts.get(ISSUE_OVERLOAD, 0)}곳** · "
        f"**운영형 품질저하 {counts.get(ISSUE_OPERATION, 0)}곳** · 대기 리스크 {counts.get(ISSUE_WAIT, 0)}곳"
    )
    rho = load_complaint_correlation(quality)
    if rho is not None:
        lines.append(
            f"- 부하율과 VOC 의 순위상관 **{rho:.2f}** — 용량 부족이 고객 불만으로 번지는 구조가 "
            + ("확인된다. 증설 투자는 곧 품질 투자다." if rho >= 0.2 else "약하다. 불만의 주원인은 운영 쪽이다.")
        )
    lines.append(
        "- 같은 '불만 많은 거점'이라도 처방이 다르다: 과부하형은 워크베이 증설 · 예약 분산(투자), "
        "운영형은 정비 품질 점검 · 기술 교육(운영)."
    )
    lines.append("")

    for issue, title in ((ISSUE_OVERLOAD, "과부하형"), (ISSUE_OPERATION, "운영형")):
        sub = quality[quality["issue_type"] == issue].sort_values("quality_risk", ascending=False).head(5)
        if sub.empty:
            continue
        lines.append(f"**{title} 상위 거점** → {PRESCRIPTIONS[issue]}")
        lines.append("")
        lines.append("| 거점 | 등급 | 리스크 | 부하율 | 대기(일) | VOC/천건 | 재입고율 |")
        lines.append("|---|:-:|---:|---:|---:|---:|---:|")
        for r in sub.itertuples(index=False):
            lines.append(
                f"| {r.name} | {r.quality_grade} | {r.quality_risk:.0f} | {r.utilization:.0%} | "
                f"{r.est_wait_days:.1f} | {_n(r.voc_per_1k_jobs, 1)} | "
                f"{'-' if pd.isna(r.comeback_rate) else f'{r.comeback_rate:.1%}'} |"
            )
        lines.append("")
    return lines


def _investment_section(settings: Settings) -> list[str]:
    from snx.optimize.network_plan import budget_sensitivity

    db = settings.db_path
    runs = read_sql(db, "SELECT * FROM plan_runs ORDER BY created_at DESC LIMIT 1") if table_exists(
        db, "plan_runs"
    ) else pd.DataFrame()
    if runs.empty:
        return []
    run = runs.iloc[0]
    inv = settings.investment
    unit = inv.cost_unit
    base = max(float(run["baseline_unmet"]), 1.0)

    lines = ["## 5. 신설 + 증설 투자안 (예산 제약 동시 최적화)", ""]
    lines.append(
        f"가정: 신규 거점(6베이) {inv.new_site_cost:,.0f}{unit} · 워크베이 증설 {inv.bay_expansion_cost:,.1f}{unit}/베이 "
        f"· 거점당 최대 +{inv.max_added_bays_per_center}베이"
    )
    lines.append("")
    lines.append(f"| 예산 {float(run['budget']):,.0f}{unit} | 흡수 수요 | 해소율 |")
    lines.append("|---|---:|---:|")
    for name, value in (
        ("신설만", run["new_only_value"]),
        ("증설만", run["expand_only_value"]),
        ("**신설 + 증설 최적 조합**", run["objective_value"]),
    ):
        lines.append(f"| {name} | {float(value):,.0f} | {100 * float(value) / base:.1f}% |")
    lines.append("")

    uplift = float(run["objective_value"]) / max(float(run["new_only_value"]), 1.0)
    lines.append(
        f"- 최적 조합은 신설 **{int(run['n_new_sites'])}개소** · 증설 **{int(run['n_expansions'])}곳"
        f"(+{int(run['added_bays'])}베이)** — 같은 예산을 신설에만 쓸 때보다 **{uplift:.2f}배** 흡수한다."
    )
    if int(run["n_new_sites"]) == 0:
        lines.append(
            "- 이 예산 규모에서는 신설이 한 곳도 선택되지 않는다. 현재 미충족의 대부분이 "
            "'거점은 닿지만 붐비는' 용량 갭이라, 베이당 단가가 낮은 증설이 먼저 소진된다."
        )

    sens = budget_sensitivity(settings, multipliers=(0.5, 1.0, 2.0))
    if not sens.empty:
        ceiling = float(sens.attrs.get("expand_ceiling", 0.0))
        lines.append("")
        lines.append("| 예산 | 신설 | 증설(베이) | 흡수 수요 | 신설만 대비 |")
        lines.append("|---:|---:|---:|---:|---:|")
        for r in sens.itertuples(index=False):
            lines.append(
                f"| {r.budget:,.0f}{unit} | {r.n_new_sites} | {r.n_expansions}({r.added_bays}) | "
                f"{r.objective_value:,.0f} | {r.uplift:.2f}배 |"
            )
        if ceiling > 0:
            lines.append("")
            lines.append(
                f"- 증설만으로 해소 가능한 상한은 **{ceiling:,.0f}건(미충족의 {100 * ceiling / base:.1f}%)** 이다. "
                "나머지는 30분 밖 수요이거나 증설 한도를 넘는 집중 과부하라 신설로만 풀린다 — "
                "예산이 커질수록 신설 비중이 올라가는 이유다."
            )

    actions = read_sql(
        db,
        """
        SELECT pa.priority, pa.action, pa.target_id, pa.added_bays, pa.cost, pa.captured_visits,
               r.sido, r.sigungu, sc.name
        FROM plan_actions pa
        JOIN regions r ON r.sigungu_code = pa.sigungu_code
        LEFT JOIN service_centers sc ON sc.center_id = pa.target_id
        WHERE pa.run_id = ? ORDER BY pa.captured_visits DESC LIMIT 8
        """,
        (run["run_id"],),
    )
    if not actions.empty:
        lines.append("")
        lines.append("**흡수량 상위 조치**")
        lines.append("")
        lines.append("| 조치 | 대상 | 베이 | 투자 | 흡수 수요 |")
        lines.append("|---|---|---:|---:|---:|")
        for r in actions.itertuples(index=False):
            label = "신설" if r.action == "new" else "증설"
            target = f"{r.sido} {r.sigungu}" if r.action == "new" else f"{r.name}"
            lines.append(
                f"| {label} | {target} | +{int(r.added_bays)} | {r.cost:,.1f}{unit} | {r.captured_visits:,.0f} |"
            )
    lines.append("")
    return lines


def _policy_section(settings: Settings) -> list[str]:
    from snx.policy.scenario import simulate_loyalty

    ps = settings.policy_scenario
    if not ps.loyalty_overrides:
        return []
    try:
        result = simulate_loyalty(settings, ps.loyalty_overrides, ps.label)
    except RuntimeError:
        return []
    sm = result.summary

    lines = ["## 6. 서비스 정책 시나리오", ""]
    lines.append(f"**{ps.label or ps.loyalty_overrides}**")
    lines.append("")
    lines.append(
        f"- 연간 입고 **+{sm['delta_visits']:,.0f}건** (보증수리 +{sm['delta_warranty_visits']:,.0f}건) — "
        f"그중 **{sm['leak_ratio']:.0%}({sm['delta_unmet']:,.0f}건)** 는 서비스망이 받지 못해 대기 · 이탈로 샌다"
    )
    lines.append(
        f"- 건전 부하율 초과 상권 {sm['overloaded_regions_base']}곳 → **{sm['overloaded_regions_scenario']}곳**"
    )
    lines.append(
        "- 유입률을 올리는 정책은 네트워크 증설 계획과 한 세트로 짜야 한다. "
        "정책 효과의 절반이 거점 용량에서 막히면 고객 경험은 오히려 나빠진다."
    )
    lines.append("")
    lines.append("| 지역 | 입고 증가 | 미충족 증가 | 누수율 |")
    lines.append("|---|---:|---:|---:|")
    for r in result.regions.head(5).itertuples(index=False):
        lines.append(
            f"| {r.sido} {r.sigungu} | {r.delta_visits:,.0f} | {r.delta_unmet:,.0f} | {r.leak_ratio:.0%} |"
        )
    lines.append("")
    return lines


def save_report(
    settings: Settings,
    content: str,
    *,
    generator: str,
    scope: str = "national",
    run_id: str | None = None,
    filename: str = "report.md",
) -> str:
    report_id = uuid.uuid4().hex[:12]
    row = pd.DataFrame(
        [
            {
                "report_id": report_id,
                "run_id": run_id,
                "scope": scope,
                "generator": generator,
                "content_md": content,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        ]
    )
    with get_conn(settings.db_path) as conn:
        row.to_sql("reports", conn, if_exists="append", index=False)

    path = settings.outputs_dir / filename
    path.write_text(content, encoding="utf-8")
    return str(path)


def _n(value, digits: int = 0) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.{digits}f}"
