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
from snx.storage.db import get_conn, read_sql


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
    lines.append("# 정비 서비스망 커버리지 갭 진단")
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

    lines.append("## 4. 가정과 한계")
    lines.append("")
    lines.append(
        "- 소요시간은 직선거리에 우회계수와 도시등급별 평균속도를 적용한 근사치다. "
        "실제 라우팅 API 로 교체하면 산악 지형 지역의 접근성이 더 나쁘게 나올 가능성이 크다."
    )
    lines.append(
        "- 차령별 정비 원단위와 공식망 유입률은 시나리오 가정값이다. "
        "실적 데이터로 교체하면 절대 수준은 바뀌지만 지역 간 상대 순위는 유지되는 구조다."
    )
    lines.append("")
    return "\n".join(lines)


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
