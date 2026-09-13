"""snx CLI — 파이프라인 단계별 실행.

    snx pipeline --source sample --p 15     전 단계 일괄 실행
    snx ingest / demand / analyze / optimize / report / map   단계별 실행
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
    help="정비 서비스망 수요 예측 · 커버리지 갭 분석 · 신규 거점 입지 최적화",
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
    agent: bool = typer.Option(True, "--agent/--no-agent"),
    seed: int = typer.Option(20260914, "--seed"),
    config: str | None = ConfigOpt,
) -> None:
    """수집 → 수요 → 커버리지 → 최적화 → 리포트 → 지도 를 한 번에 실행한다."""
    from snx.agents.crew import generate_report
    from snx.coverage.gap import run_coverage_stage
    from snx.demand.model import run_demand_stage
    from snx.ingest.pipeline import ingest as run_ingest
    from snx.optimize.siting import run_siting_stage
    from snx.report.render import save_report
    from snx.storage.db import init_db
    from snx.viz.maps import save_gap_map

    s = _settings(config)
    console.rule("[bold]1/6 수집")
    init_db(s.db_path)
    run_ingest(s, source=source, seed=seed)

    console.rule("[bold]2/6 수요 추정")
    run_demand_stage(s)

    console.rule("[bold]3/6 커버리지 · 갭")
    _, gaps = run_coverage_stage(s)
    _print_gap_table(s, gaps.head(10))

    console.rule("[bold]4/6 입지 최적화")
    sol = run_siting_stage(s, n_new_sites=p)

    console.rule("[bold]5/6 리포트")
    content, generator = generate_report(s, use_agent=agent)
    report_path = save_report(s, content, generator=generator, run_id=sol.run_id)

    console.rule("[bold]6/6 지도")
    map_path = save_gap_map(s)

    console.print()
    console.print("[bold green]파이프라인 완료[/]")
    console.print(f"  리포트 : {report_path}")
    console.print(f"  지도   : {map_path}")
    console.print(f"  DB     : {s.db_path}")
    console.print(f"  대시보드: streamlit run {Path('app/dashboard.py')}")


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
