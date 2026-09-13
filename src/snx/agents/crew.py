"""CrewAI 멀티에이전트 리포트 생성.

구성: 수요 애널리스트 → 서비스망 전략가 → 보고서 작성자 (순차 프로세스)

설계 원칙
---------
- **숫자는 도구가, 서술은 에이전트가.** 에이전트에게 DB 조회 도구만 주고,
  수치 생성은 코드 쪽에 가둔다.
- **키가 없으면 조용히 폴백.** CrewAI 미설치 또는 API 키 부재 시 결정론적
  템플릿 리포트를 반환한다. 저장소는 어떤 환경에서도 완주한다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from snx.config import Settings
from snx.logging_conf import get_logger
from snx.report.render import render_template_report

log = get_logger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


def is_agent_available() -> tuple[bool, str]:
    """에이전트 실행 가능 여부와 사유."""
    try:
        import crewai  # noqa: F401
    except ImportError:
        return False, "crewai 미설치 (pip install -e '.[agents]')"
    if not os.getenv("OPENAI_API_KEY"):
        return False, "OPENAI_API_KEY 미설정"
    return True, "ok"


def _load_prompts() -> tuple[dict[str, Any], dict[str, Any]]:
    agents = yaml.safe_load((PROMPTS_DIR / "agents.yaml").read_text(encoding="utf-8"))
    tasks = yaml.safe_load((PROMPTS_DIR / "tasks.yaml").read_text(encoding="utf-8"))
    return agents, tasks


def _build_tools(settings: Settings) -> list:
    """순수 함수를 CrewAI 도구로 감싼다 (import 는 이 시점에만)."""
    from crewai.tools import tool

    from snx.agents import tools as T

    @tool("national_summary")
    def national_summary_tool() -> str:
        """전국 정비 수요 · 미충족 수요 · 접근 공백 지역 수 등 요약 지표를 조회한다."""
        return T.national_summary(settings)

    @tool("top_gap_regions")
    def top_gap_regions_tool(n: int = 5) -> str:
        """갭 스코어 상위 n개 시군구의 수요 · 커버리지 · 부하율 상세 지표를 조회한다."""
        return T.top_gap_regions(n, settings)

    @tool("region_detail")
    def region_detail_tool(sigungu_code: str) -> str:
        """특정 시군구 코드의 상세 지표를 조회한다."""
        return T.region_detail(sigungu_code, settings)

    @tool("siting_plan")
    def siting_plan_tool() -> str:
        """최신 신규 거점 입지 최적화 결과와 우선순위를 조회한다."""
        return T.siting_plan(settings)

    @tool("methodology_notes")
    def methodology_tool() -> str:
        """분석에 사용한 모형 · 계수 · 한계를 조회한다."""
        return T.methodology_notes(settings)

    return [
        national_summary_tool,
        top_gap_regions_tool,
        region_detail_tool,
        siting_plan_tool,
        methodology_tool,
    ]


def build_crew(settings: Settings):
    """Crew 객체를 구성해 반환 (crewai 설치 환경에서만 호출 가능)."""
    from crewai import Agent, Crew, Process, Task

    agent_cfg, task_cfg = _load_prompts()
    toolbox = _build_tools(settings)
    model = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

    agents = {
        key: Agent(
            role=cfg["role"],
            goal=cfg["goal"],
            backstory=cfg["backstory"],
            allow_delegation=cfg.get("allow_delegation", False),
            tools=toolbox,
            llm=model,
            verbose=False,
        )
        for key, cfg in agent_cfg.items()
    }

    ordered = ["diagnose_demand", "evaluate_network", "write_report"]
    tasks: list = []
    for key in ordered:
        cfg = task_cfg[key]
        tasks.append(
            Task(
                description=cfg["description"].replace("{top_n}", str(settings.report.top_n_regions)),
                expected_output=cfg["expected_output"],
                agent=agents[cfg["agent"]],
                context=list(tasks),
            )
        )

    return Crew(agents=list(agents.values()), tasks=tasks, process=Process.sequential, verbose=False)


def generate_report(settings: Settings, use_agent: bool = True) -> tuple[str, str]:
    """리포트 본문과 생성기 이름을 반환한다.

    Returns
    -------
    (content_markdown, generator)  generator ∈ {"crewai", "template"}
    """
    baseline = render_template_report(settings)

    if not use_agent:
        return baseline, "template"

    ok, reason = is_agent_available()
    if not ok:
        log.warning("에이전트 리포트 건너뜀 (%s) — 템플릿 리포트로 대체", reason)
        return baseline, "template"

    try:
        crew = build_crew(settings)
        result = crew.kickoff()
        content = str(getattr(result, "raw", result)).strip()
        if len(content) < 200:
            raise ValueError("에이전트 응답이 비정상적으로 짧습니다")
    except Exception as exc:  # noqa: BLE001 — 리포트 실패가 파이프라인을 막지 않게
        log.warning("에이전트 실행 실패 (%s) — 템플릿 리포트로 대체", exc)
        return baseline, "template"

    merged = (
        f"{content}\n\n---\n\n"
        "<details>\n<summary>부록 — 자동 생성 근거표</summary>\n\n"
        f"{baseline}\n\n</details>\n"
    )
    return merged, "crewai"
