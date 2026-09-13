"""서비스 정책 시나리오 — 정책이 성공하면 서비스망은 감당할 수 있는가.

보증 연장, 정비 멤버십, 소모품 할인 같은 대고객 서비스 프로그램은 대부분
'공식 서비스망 유입률 λ_a' 를 끌어올리는 정책이다. 정책 부서는 유입률과 매출을
보고, 네트워크 부서는 거점 부하를 본다. 둘을 한 모형에 넣으면 이런 질문에 답할 수 있다.

    "6~10년 차 유입률을 38% → 50% 로 올리면 늘어난 입고를 어느 지역 거점이 못 받는가?"

구현은 단순하다. 유입률만 바꾼 설정으로 수요 → 커버리지를 메모리에서 다시 돌려
기준선과 비교한다. DB 는 건드리지 않는다 — 시나리오는 몇 번이고 가볍게 돌려 보는 도구다.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from snx.config import AGE_BUCKETS, Settings
from snx.coverage.gap import compute_coverage
from snx.demand.model import estimate_demand
from snx.storage.db import read_table


@dataclass
class ScenarioResult:
    label: str
    overrides: dict[str, float]
    summary: dict[str, float]
    regions: pd.DataFrame


def parse_overrides(pairs: list[str]) -> dict[str, float]:
    """['6-10=0.50', '11+=0.25'] → {'6-10': 0.5, '11+': 0.25}"""
    out: dict[str, float] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"'버킷=유입률' 형식이어야 합니다: {pair!r}")
        bucket, value = (p.strip() for p in pair.split("=", 1))
        out[bucket] = float(value)
    return validate_overrides(out)


def validate_overrides(overrides: dict[str, float]) -> dict[str, float]:
    bad = set(overrides) - set(AGE_BUCKETS)
    if bad:
        raise ValueError(f"알 수 없는 차령 버킷: {sorted(bad)} — 허용값 {list(AGE_BUCKETS)}")
    for bucket, value in overrides.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"유입률은 0~1 사이여야 합니다: {bucket}={value}")
    return overrides


def apply_overrides(settings: Settings, overrides: dict[str, float]) -> Settings:
    scenario = settings.model_copy(deep=True)
    scenario.demand.official_network_loyalty.update(validate_overrides(dict(overrides)))
    return scenario


def simulate_loyalty(
    settings: Settings, overrides: dict[str, float], label: str = ""
) -> ScenarioResult:
    db = settings.db_path
    regions = read_table(db, "regions")
    parc = read_table(db, "vehicle_parc")
    centers = read_table(db, "service_centers")
    if parc.empty:
        raise RuntimeError("vehicle_parc 가 비어 있습니다 — `snx ingest` 를 먼저 실행하세요")

    base = _evaluate(regions, parc, centers, settings)
    scen = _evaluate(regions, parc, centers, apply_overrides(settings, overrides))

    merged = base.merge(scen, on="sigungu_code", suffixes=("_base", "_scen"))
    merged["delta_visits"] = merged["annual_visits_scen"] - merged["annual_visits_base"]
    merged["delta_unmet"] = merged["unmet_visits_scen"] - merged["unmet_visits_base"]
    # 늘어난 입고 중 서비스망이 받아내지 못하고 대기 · 이탈로 새는 비율
    merged["leak_ratio"] = (merged["delta_unmet"] / merged["delta_visits"].where(merged["delta_visits"] > 0)).fillna(0.0)
    merged = merged.merge(regions[["sigungu_code", "sido", "sigungu"]], on="sigungu_code")
    merged = merged.sort_values("delta_unmet", ascending=False).reset_index(drop=True)

    healthy = settings.capacity.healthy_utilization
    d_visits = float(merged["delta_visits"].sum())
    d_unmet = float(merged["delta_unmet"].sum())
    summary = {
        "demand_base": float(merged["annual_visits_base"].sum()),
        "demand_scenario": float(merged["annual_visits_scen"].sum()),
        "delta_visits": d_visits,
        "unmet_base": float(merged["unmet_visits_base"].sum()),
        "unmet_scenario": float(merged["unmet_visits_scen"].sum()),
        "delta_unmet": d_unmet,
        "leak_ratio": d_unmet / d_visits if d_visits > 0 else 0.0,
        "overloaded_regions_base": int((merged["avg_utilization_base"] > healthy).sum()),
        "overloaded_regions_scenario": int((merged["avg_utilization_scen"] > healthy).sum()),
        "delta_warranty_visits": float(
            merged["warranty_visits_scen"].sum() - merged["warranty_visits_base"].sum()
        ),
    }
    return ScenarioResult(label=label, overrides=dict(overrides), summary=summary, regions=merged)


def _evaluate(
    regions: pd.DataFrame, parc: pd.DataFrame, centers: pd.DataFrame, settings: Settings
) -> pd.DataFrame:
    demand = estimate_demand(parc, settings)
    coverage, _ = compute_coverage(regions, centers, demand, settings)
    out = demand[["sigungu_code", "annual_visits", "warranty_visits"]].merge(coverage, on="sigungu_code")
    out["unmet_visits"] = out["uncovered_visits"] + out["overflow_visits"]
    return out[
        ["sigungu_code", "annual_visits", "warranty_visits", "unmet_visits", "avg_utilization"]
    ]
