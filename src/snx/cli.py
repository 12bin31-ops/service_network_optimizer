"""snx CLI — 파이프라인 단계별 실행.

    snx pipeline --source sample --p 15     전 단계 일괄 실행
    snx ingest / demand / analyze / quality / optimize / plan / report / map   단계별 실행
    snx scenario --loyalty 6-10=0.50        서비스 정책 시나리오 (DB 변경 없음)
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from snx.config import load_settings
from snx.logging_conf import setup_logging

app = typer.Typer(
    add_completion=False,
    help="Service Network Optimizer — 정비 수요 예측 · 커버리지 갭 · 거점 품질 · 투자 최적화",
)
console = Console()

ConfigOpt = typer.Option(None, "--config", "-c", help="설정 파일 경로")


def _settings(config: str | None):
    s = load_settings(config)
    setup_logging(s.logging.level)
    return s


@app.command()
def init(config: str | None = ConfigOpt) -> None:
    """DB 스키마를 생성한다."""
    from snx.storage.db import init_db

    s = _settings(config)
    init_db(s.db_path)
    console.print(f"[green]DB 초기화 완료[/] → {s.db_path}")


@app.command()
def ingest(
    source: str = typer.Option("sample", "--source", help="sample | live"),
    seed: int = typer.Option(20260914, "--seed", help="sample 모드 난수 시드"),
    config: str | None = ConfigOpt,
) -> None:
    """차량 모수와 서비스망을 수집해 적재한다."""
    from snx.ingest.pipeline import ingest as run_ingest

    s = _settings(config)
    result = run_ingest(s, source=source, seed=seed)
    console.print(
        f"[green]수집 완료[/] 지역 {len(result.regions)} · "
        f"거점 {len(result.service_centers)} · 모수 {len(result.vehicle_parc)}행"
    )


@app.command()
def demand(config: str | None = ConfigOpt) -> None:
    """지역별 연간 정비 수요를 추정한다."""
    from snx.demand.model import run_demand_stage

    s = _settings(config)
    df = run_demand_stage(s)
    console.print(f"[green]수요 추정 완료[/] 전국 {df['annual_visits'].sum():,.0f}건/년")


@app.command()
def analyze(config: str | None = ConfigOpt) -> None:
    """커버리지를 계산하고 갭 스코어를 산출한다."""
    from snx.coverage.gap import run_coverage_stage

    s = _settings(config)
    _, gaps = run_coverage_stage(s)
    _print_gap_table(s, gaps.head(10))


@app.command()
def quality(config: str | None = ConfigOpt) -> None:
    """거점 품질(부하 · VOC · 재입고)을 진단하고 관리 대상 거점을 가려낸다."""
    from snx.quality.scorecard import run_quality_stage

    s = _settings(config)
    q = run_quality_stage(s)
    _print_quality_table(s, q.head(10))


@app.command()
def optimize(
    p: int | None = typer.Option(None, "--p", help="신규 거점 개수 (기본: 설정값)"),
    config: str | None = ConfigOpt,
) -> None:
    """신규 거점 입지를 최적화한다."""
    from snx.optimize.siting import run_siting_stage

    s = _settings(config)
    sol = run_siting_stage(s, n_new_sites=p)

    table = Table(title=f"신규 거점 우선순위 (run {sol.run_id})")
    for col in ("우선순위", "지역", "흡수 수요", "갭스코어"):
        table.add_column(col, justify="right" if col != "지역" else "left")
    for r in sol.selected.itertuples(index=False):
        table.add_row(
            str(r.priority), f"{r.sido} {r.sigungu}", f"{r.captured_visits:,.0f}", f"{r.gap_score:.1f}"
        )
    console.print(table)
    console.print(f"미충족 수요 해소율 [bold]{sol.relief_ratio:.1%}[/]")


@app.command()
def plan(
    budget: float | None = typer.Option(None, "--budget", help="투자 예산 (기본: 설정값, 단위 억원)"),
    config: str | None = ConfigOpt,
) -> None:
    """같은 예산으로 신설과 증설을 함께 최적화한다."""
    from snx.optimize.network_plan import run_plan_stage

    s = _settings(config)
    _print_plan(s, run_plan_stage(s, budget=budget))


@app.command()
def scenario(
    loyalty: list[str] | None = typer.Option(  # noqa: B008
        None, "--loyalty", help="차령버킷=유입률 (반복 가능). 예: --loyalty 6-10=0.50"
    ),
    label: str = typer.Option("", "--label", help="시나리오 이름"),
    config: str | None = ConfigOpt,
) -> None:
    """공식망 유입률 정책 시나리오 — 늘어난 입고를 서비스망이 감당하는지 본다."""
    from snx.policy.scenario import parse_overrides, simulate_loyalty

    s = _settings(config)
    if loyalty:
        overrides, name = parse_overrides(loyalty), label or " · ".join(loyalty)
    else:
        overrides, name = s.policy_scenario.loyalty_overrides, label or s.policy_scenario.label
    result = simulate_loyalty(s, overrides, name)
    sm = result.summary

    console.print(f"[bold]{name}[/]")
    console.print(
        f"  연간 입고 {sm['demand_base']:,.0f} → {sm['demand_scenario']:,.0f}건 "
        f"([green]+{sm['delta_visits']:,.0f}[/])"
    )
    console.print(
        f"  미충족 {sm['unmet_base']:,.0f} → {sm['unmet_scenario']:,.0f}건 "
        f"([red]+{sm['delta_unmet']:,.0f}[/]) — 늘어난 입고의 [bold]{sm['leak_ratio']:.1%}[/] 가 대기 · 이탈"
    )
    console.print(
        f"  과부하 상권 {sm['overloaded_regions_base']} → {sm['overloaded_regions_scenario']}곳 · "
        f"보증수리 +{sm['delta_warranty_visits']:,.0f}건"
    )

    table = Table(title="미충족 증가 상위 지역")
    for col in ("지역", "입고 증가", "미충족 증가", "누수율"):
        table.add_column(col, justify="left" if col == "지역" else "right")
    for r in result.regions.head(10).itertuples(index=False):
        table.add_row(
            f"{r.sido} {r.sigungu}", f"{r.delta_visits:,.0f}", f"{r.delta_unmet:,.0f}", f"{r.leak_ratio:.0%}"
        )
    console.print(table)
    path = s.outputs_dir / "scenario_regions.csv"
    result.regions.to_csv(path, index=False)
    console.print(f"[green]저장[/] → {path}")


@app.command()
def report(
    agent: bool = typer.Option(True, "--agent/--no-agent", help="CrewAI 에이전트 리포트 사용"),
    config: str | None = ConfigOpt,
) -> None:
    """진단 리포트를 생성한다 (에이전트 불가 시 템플릿으로 폴백)."""
    from snx.agents.crew import generate_report
    from snx.report.render import save_report

    s = _settings(config)
    content, generator = generate_report(s, use_agent=agent)
    path = save_report(s, content, generator=generator)
    console.print(f"[green]리포트 생성 완료[/] ({generator}) → {path}")


@app.command("map")
def render_map(config: str | None = ConfigOpt) -> None:
    """갭 지도를 HTML 로 저장한다."""
    from snx.viz.maps import save_gap_map

    s = _settings(config)
    path = save_gap_map(s)
    console.print(f"[green]지도 저장 완료[/] → {path}")


@app.command()
def pipeline(
    source: str = typer.Option("sample", "--source", help="sample | live"),
    p: int | None = typer.Option(None, "--p", help="신규 거점 개수"),
    budget: float | None = typer.Option(None, "--budget", help="신설+증설 투자 예산 (억원)"),
    agent: bool = typer.Option(True, "--agent/--no-agent"),
    seed: int = typer.Option(20260914, "--seed"),
    config: str | None = ConfigOpt,
) -> None:
    """수집 → 수요 → 커버리지 → 품질 → 입지 · 투자 최적화 → 리포트 → 지도 를 한 번에 실행한다."""
    from snx.agents.crew import generate_report
    from snx.coverage.gap import run_coverage_stage
    from snx.demand.model import run_demand_stage
    from snx.ingest.pipeline import ingest as run_ingest
    from snx.optimize.network_plan import run_plan_stage
    from snx.optimize.siting import run_siting_stage
    from snx.quality.scorecard import run_quality_stage
    from snx.report.render import save_report
    from snx.storage.db import init_db
    from snx.viz.maps import save_gap_map

    s = _settings(config)
    console.rule("[bold]1/8 수집")
    init_db(s.db_path)
    run_ingest(s, source=source, seed=seed)

    console.rule("[bold]2/8 수요 추정")
    run_demand_stage(s)

    console.rule("[bold]3/8 커버리지 · 갭")
    _, gaps = run_coverage_stage(s)
    _print_gap_table(s, gaps.head(10))

    console.rule("[bold]4/8 거점 품질")
    run_quality_stage(s)

    console.rule("[bold]5/8 신규 거점 입지")
    sol = run_siting_stage(s, n_new_sites=p)

    console.rule("[bold]6/8 신설 + 증설 투자안")
    _print_plan(s, run_plan_stage(s, budget=budget))

    console.rule("[bold]7/8 리포트")
    content, generator = generate_report(s, use_agent=agent)
    report_path = save_report(s, content, generator=generator, run_id=sol.run_id)

    console.rule("[bold]8/8 지도")
    map_path = save_gap_map(s)

    console.print()
    console.print("[bold green]파이프라인 완료[/]")
    console.print(f"  리포트 : {report_path}")
    console.print(f"  지도   : {map_path}")
    console.print(f"  DB     : {s.db_path}")
    console.print(f"  대시보드: streamlit run {Path('app/dashboard.py')}")


def _print_quality_table(settings, quality) -> None:
    from snx.quality.scorecard import PRESCRIPTIONS
    from snx.storage.db import read_table

    centers = read_table(settings.db_path, "service_centers")[["center_id", "name"]]
    merged = quality.merge(centers, on="center_id", how="left")

    table = Table(title="품질 리스크 상위 거점")
    for col in ("거점", "등급", "리스크", "부하율", "VOC/천건", "재입고", "원인", "처방"):
        table.add_column(col, justify="left" if col in ("거점", "원인", "처방") else "right")
    for r in merged.itertuples(index=False):
        table.add_row(
            r.name,
            r.quality_grade,
            f"{r.quality_risk:.0f}",
            f"{r.utilization:.0%}",
            "-" if r.voc_per_1k_jobs != r.voc_per_1k_jobs else f"{r.voc_per_1k_jobs:.1f}",
            "-" if r.comeback_rate != r.comeback_rate else f"{r.comeback_rate:.1%}",
            r.issue_type,
            PRESCRIPTIONS.get(r.issue_type, "-"),
        )
    console.print(table)


def _print_plan(settings, plan) -> None:
    unit = settings.investment.cost_unit
    table = Table(title=f"예산 {plan.budget:,.0f}{unit} — 수단별 흡수 수요 비교")
    table.add_column("투자안")
    table.add_column("흡수 수요", justify="right")
    table.add_column("해소율", justify="right")
    base = max(plan.baseline_unmet, 1)
    for name, value in (
        ("신설만", plan.new_only_value),
        ("증설만", plan.expand_only_value),
        ("신설 + 증설 최적 조합", plan.objective_value),
    ):
        table.add_row(name, f"{value:,.0f}", f"{value / base:.1%}")
    console.print(table)
    console.print(
        f"최적 조합: 신설 [bold]{plan.n_new_sites}[/]개소 · 증설 [bold]{plan.n_expansions}[/]곳"
        f"(+{plan.added_bays}베이) · 집행 {plan.spent:,.0f}{unit}"
    )


def _print_gap_table(settings, gaps) -> None:
    from snx.storage.db import read_table

    regions = read_table(settings.db_path, "regions")[["sigungu_code", "sido", "sigungu"]]
    merged = gaps.merge(regions, on="sigungu_code", how="left")

    table = Table(title="갭 스코어 상위 지역")
    table.add_column("순위", justify="right")
    table.add_column("지역")
    table.add_column("미충족 수요", justify="right")
    table.add_column("갭스코어", justify="right")
    for r in merged.itertuples(index=False):
        table.add_row(
            str(int(r.gap_rank)),
            f"{r.sido} {r.sigungu}",
            f"{r.unmet_visits:,.0f}",
            f"{r.gap_score:.1f}",
        )
    console.print(table)


if __name__ == "__main__":
    app()
