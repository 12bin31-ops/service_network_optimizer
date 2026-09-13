"""에이전트가 호출하는 데이터 조회 함수.

원칙: **에이전트는 숫자를 만들지 않는다.** 모든 수치는 여기서 DB 를 읽어
돌려주고, 에이전트는 그 위에 해석과 서술만 얹는다. 그래서 이 함수들은
CrewAI 와 무관한 순수 함수로 두고, 크루 구성 시점에만 도구로 감싼다.
(테스트도 CrewAI 설치 없이 그대로 돌아간다.)
"""

from __future__ import annotations

import json

import pandas as pd

from snx.config import Settings, load_settings
from snx.report.context import build_analysis_context, region_brief
from snx.storage.db import read_sql


def _ctx(settings: Settings) -> pd.DataFrame:
    return build_analysis_context(settings)


def national_summary(settings: Settings | None = None) -> str:
    """전국 단위 진단 요약 지표를 JSON 으로 반환."""
    s = settings or load_settings()
    ctx = _ctx(s)
    total_demand = float(ctx["annual_visits"].fillna(0).sum())
    total_unmet = float(ctx["unmet_visits"].fillna(0).sum())
    payload = {
        "분석_지역수": int(len(ctx)),
        "연간_정비수요_합": round(total_demand),
        "미충족_수요_합": round(total_unmet),
        "미충족_비율": f"{100 * total_unmet / max(total_demand, 1):.1f}%",
        "접근공백_지역수": int((ctx["reachable_centers"].fillna(0) == 0).sum()),
        "과부하_상권수": int(
            (ctx["avg_utilization"].fillna(0) > s.capacity.healthy_utilization).sum()
        ),
        "기준_접근시간_분": s.coverage.max_travel_minutes,
        "건전_부하율": s.capacity.healthy_utilization,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def top_gap_regions(n: int = 5, settings: Settings | None = None) -> str:
    """갭 스코어 상위 n개 지역의 상세 지표를 JSON 배열로 반환."""
    s = settings or load_settings()
    ctx = _ctx(s)
    codes = ctx.head(int(n))["sigungu_code"].tolist()
    return json.dumps([region_brief(ctx, c) for c in codes], ensure_ascii=False, indent=2)


def region_detail(sigungu_code: str, settings: Settings | None = None) -> str:
    """특정 시군구의 수요 · 커버리지 · 갭 지표를 JSON 으로 반환."""
    s = settings or load_settings()
    return json.dumps(region_brief(_ctx(s), str(sigungu_code)), ensure_ascii=False, indent=2)


def siting_plan(settings: Settings | None = None) -> str:
    """최신 입지 최적화 결과(신규 거점 우선순위)를 JSON 으로 반환."""
    s = settings or load_settings()
    runs = read_sql(s.db_path, "SELECT * FROM siting_runs ORDER BY created_at DESC LIMIT 1")
    if runs.empty:
        return json.dumps({"error": "최적화 실행 이력이 없습니다"}, ensure_ascii=False)
    run = runs.iloc[0]
    rows = read_sql(
        s.db_path,
        """
        SELECT sr.priority, r.sido, r.sigungu, sr.sigungu_code, sr.captured_visits
        FROM siting_results sr JOIN regions r ON r.sigungu_code = sr.sigungu_code
        WHERE sr.run_id = ? ORDER BY sr.priority
        """,
        (run["run_id"],),
    )
    payload = {
        "신규거점_개수": int(run["n_new_sites"]),
        "후보지_풀": int(run["candidate_count"]),
        "흡수_미충족수요": round(float(run["objective_value"])),
        "최적화_이전_미충족수요": round(float(run["baseline_unmet"])),
        "해소율": f"{100 * run['objective_value'] / max(run['baseline_unmet'], 1):.1f}%",
        "솔버_상태": run["solver_status"],
        "우선순위": rows.to_dict(orient="records"),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def quality_watchlist(n: int = 10, settings: Settings | None = None) -> str:
    """품질 관리 대상 거점(과부하형 · 운영형)과 원인별 처방을 JSON 으로 반환."""
    from snx.quality.scorecard import PRESCRIPTIONS, load_complaint_correlation

    s = settings or load_settings()
    quality = read_sql(
        s.db_path,
        """
        SELECT q.*, sc.name, r.sido, r.sigungu
        FROM center_quality q
        JOIN service_centers sc ON sc.center_id = q.center_id
        JOIN regions r ON r.sigungu_code = q.sigungu_code
        ORDER BY q.quality_risk DESC
        """,
    )
    if quality.empty:
        return json.dumps({"error": "거점 품질 진단 결과가 없습니다"}, ensure_ascii=False)
    flagged = quality[quality["issue_type"].isin(["과부하형 품질저하", "운영형 품질저하"])].head(int(n))
    rho = load_complaint_correlation(quality)
    payload = {
        "진단_거점수": int(len(quality)),
        "원인별_거점수": quality["issue_type"].value_counts().to_dict(),
        "부하율_VOC_순위상관": None if rho is None else round(rho, 2),
        "관리대상": [
            {
                "거점": r.name,
                "지역": f"{r.sido} {r.sigungu}",
                "등급": r.quality_grade,
                "품질리스크": r.quality_risk,
                "부하율": f"{r.utilization:.0%}",
                "추정대기_일": r.est_wait_days,
                "VOC_천건당": r.voc_per_1k_jobs,
                "재입고율": None if pd.isna(r.comeback_rate) else f"{r.comeback_rate:.1%}",
                "원인": r.issue_type,
                "처방": PRESCRIPTIONS.get(r.issue_type),
            }
            for r in flagged.itertuples(index=False)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def investment_plan(settings: Settings | None = None) -> str:
    """최신 신설+증설 동시 최적화 결과(수단별 흡수량 비교 · 상위 조치)를 JSON 으로 반환."""
    s = settings or load_settings()
    runs = read_sql(s.db_path, "SELECT * FROM plan_runs ORDER BY created_at DESC LIMIT 1")
    if runs.empty:
        return json.dumps({"error": "투자안 최적화 이력이 없습니다"}, ensure_ascii=False)
    run = runs.iloc[0]
    actions = read_sql(
        s.db_path,
        """
        SELECT pa.action, pa.target_id, sc.name, r.sido, r.sigungu, pa.added_bays, pa.cost,
               pa.captured_visits
        FROM plan_actions pa
        JOIN regions r ON r.sigungu_code = pa.sigungu_code
        LEFT JOIN service_centers sc ON sc.center_id = pa.target_id
        WHERE pa.run_id = ? ORDER BY pa.captured_visits DESC LIMIT 15
        """,
        (run["run_id"],),
    )
    unit = s.investment.cost_unit
    base = max(float(run["baseline_unmet"]), 1.0)
    payload = {
        "예산": f"{run['budget']:,.0f}{unit}",
        "가정_단가": {
            "신규거점_6베이": f"{s.investment.new_site_cost}{unit}",
            "증설_베이당": f"{s.investment.bay_expansion_cost}{unit}",
        },
        "수단별_흡수수요": {
            "신설만": round(float(run["new_only_value"])),
            "증설만": round(float(run["expand_only_value"])),
            "최적조합": round(float(run["objective_value"])),
        },
        "최적조합_해소율": f"{100 * float(run['objective_value']) / base:.1f}%",
        "최적조합_구성": {
            "신설_개소": int(run["n_new_sites"]),
            "증설_거점수": int(run["n_expansions"]),
            "추가_워크베이": int(run["added_bays"]),
        },
        "상위_조치": actions.to_dict(orient="records"),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def policy_scenario(settings: Settings | None = None) -> str:
    """설정된 서비스 정책 시나리오(공식망 유입률 변화)의 네트워크 영향을 JSON 으로 반환."""
    from snx.policy.scenario import simulate_loyalty

    s = settings or load_settings()
    ps = s.policy_scenario
    if not ps.loyalty_overrides:
        return json.dumps({"error": "설정된 정책 시나리오가 없습니다"}, ensure_ascii=False)
    result = simulate_loyalty(s, ps.loyalty_overrides, ps.label)
    sm = result.summary
    payload = {
        "시나리오": ps.label,
        "유입률_변경": ps.loyalty_overrides,
        "입고_증가": round(sm["delta_visits"]),
        "보증수리_증가": round(sm["delta_warranty_visits"]),
        "미충족_증가": round(sm["delta_unmet"]),
        "누수율": f"{100 * sm['leak_ratio']:.1f}%",
        "과부하_상권수_변화": [sm["overloaded_regions_base"], sm["overloaded_regions_scenario"]],
        "미충족_증가_상위지역": result.regions.head(5)[
            ["sido", "sigungu", "delta_visits", "delta_unmet"]
        ].round(0).to_dict(orient="records"),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def methodology_notes(settings: Settings | None = None) -> str:
    """분석에 사용한 가정과 한계를 JSON 으로 반환 (리포트의 '유의점' 절 근거)."""
    s = settings or load_settings()
    return json.dumps(
        {
            "수요_모형": "차령 버킷별 정비 원단위 × 공식 서비스망 유입률",
            "차령별_원단위_회년": s.demand.visits_per_vehicle_year,
            "공식망_유입률": s.demand.official_network_loyalty,
            "전기차_정비빈도_배수": s.demand.ev_visit_multiplier,
            "접근성_기준_분": s.coverage.max_travel_minutes,
            "우회계수": s.coverage.detour_factor,
            "도시등급별_평균속도_kmh": s.coverage.avg_speed_kmh,
            "수요분배_모형": f"Huff (거리민감도 {s.coverage.huff_distance_decay})",
            "최적화_모형": "용량제약 최대커버링 (Capacitated MCLP, CBC) + 신설·증설 예산 동시 최적화 (MILP)",
            "보증수리_비중": s.demand.warranty_share.model_dump(),
            "품질_리스크": "대기일수(부하율 대기행렬 근사) · VOC · 재입고율 가중합, 원인은 부하 초과 × 불만 신호로 분리",
            "한계": [
                "소요시간은 직선거리 기반 근사치로 실제 도로망·지형을 반영하지 않는다",
                "정비 원단위와 유입률은 시나리오 가정값이며 실적 데이터로 교체해야 한다",
                "후보지는 시군구 중심점으로, 부지·임대료·대리점 계약 제약은 미반영",
                "신설·증설 단가는 부지비 제외 가정값이며 단가 비율에 따라 최적 조합이 달라진다",
            ],
        },
        ensure_ascii=False,
        indent=2,
    )
