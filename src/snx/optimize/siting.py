"""신규 거점 입지 최적화 — 용량제약 최대커버링 (Capacitated MCLP).

갭 순위대로 짓지 않는다. 갭 상위 지역은 서로 인접한 경우가 많아 1·2·3위에
순서대로 지으면 같은 수요를 여러 번 덮는다. 거점 하나가 여러 지역을 덮고,
거점 용량은 유한하다는 두 제약을 함께 넣어야 포트폴리오로서 최적인 조합이 나온다.

    변수   x_j ∈ {0,1}   후보지 j 개설 여부
           z_ij ≥ 0      지역 i 의 미충족 수요 중 신규 거점 j 가 흡수하는 양

    max    Σ_ij v_ij · z_ij  +  ε · Σ_j g_j · x_j
    s.t.   Σ_j z_ij ≤ unmet_i          ∀i    지역별 미충족 수요 한도
           Σ_i z_ij ≤ C · x_j          ∀j    신규 거점 연간 처리 용량
           z_ij = 0  if  t_ij > T             접근 기준 밖은 흡수 불가
           Σ_j x_j ≤ p                        예산 = 신규 거점 개수

    v_ij = 1 + α(1 − t_ij/T)    같은 양이면 더 가까이서 흡수하는 배치를 선호
    g_j  = 후보지 갭 스코어/100  흡수량·접근성이 완전히 같을 때만 작동하는 동률 해소항

보고 지표(objective_value)는 가중치를 뺀 **실제 흡수 수요** 합이다.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pulp

from snx.config import Settings
from snx.coverage.distance import travel_time_matrix
from snx.logging_conf import get_logger
from snx.storage.db import get_conn, read_table, write_df

log = get_logger(__name__)

# 동률 해소항 계수. 흡수 1건(가중치 ≥ 1)보다 충분히 작아 흡수량과 절대 교환되지 않는다.
TIE_BREAK_EPS = 1e-3

SELECTED_COLUMNS = [
    "priority",
    "sigungu_code",
    "sido",
    "sigungu",
    "lat",
    "lon",
    "gap_score",
    "captured_visits",
    "access_value",
]


@dataclass
class SitingSolution:
    run_id: str
    n_new_sites: int
    candidate_count: int
    status: str
    objective_value: float
    baseline_unmet: float
    selected: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=SELECTED_COLUMNS))

    @property
    def relief_ratio(self) -> float:
        """미충족 수요 해소율 (0~1)."""
        if self.baseline_unmet <= 0:
            return 0.0
        return float(min(max(self.objective_value / self.baseline_unmet, 0.0), 1.0))


def build_candidates(regions: pd.DataFrame, gaps: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """후보지 풀 = 갭 스코어 상위 N개 시군구의 중심점."""
    top_n = settings.optimization.candidate_top_n
    top = gaps.nsmallest(top_n, "gap_rank")[["sigungu_code", "gap_score", "gap_rank"]]
    cands = top.merge(
        regions[["sigungu_code", "sido", "sigungu", "lat", "lon"]], on="sigungu_code", how="left"
    )
    return cands.sort_values("gap_rank").reset_index(drop=True)


def access_weights(minutes: np.ndarray, settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    """(도달 가능 마스크, 접근성 가중치 v_ij) — 신설·증설 모형이 같은 정의를 쓴다."""
    limit = settings.coverage.max_travel_minutes
    reachable = minutes <= limit
    alpha = settings.optimization.access_value_alpha
    value = 1.0 + alpha * (1.0 - np.clip(minutes / limit, 0.0, 1.0))
    return reachable, np.where(reachable, value, 0.0)


def region_to_candidate_minutes(
    regions: pd.DataFrame, cands: pd.DataFrame, settings: Settings
) -> np.ndarray:
    cov = settings.coverage
    return travel_time_matrix(
        regions["lat"].to_numpy(),
        regions["lon"].to_numpy(),
        cands["lat"].to_numpy(),
        cands["lon"].to_numpy(),
        regions["urban_class"].to_numpy(),
        cov.avg_speed_kmh,
        cov.detour_factor,
    )


def solve_siting(
    regions: pd.DataFrame,
    gaps: pd.DataFrame,
    settings: Settings,
    n_new_sites: int | None = None,
) -> SitingSolution:
    """예산(거점 개수) 제약 하에서 미충족 수요 흡수량을 최대화한다."""
    opt = settings.optimization
    p = opt.n_new_sites if n_new_sites is None else int(n_new_sites)
    run_id = uuid.uuid4().hex[:12]

    base = regions.merge(gaps[["sigungu_code", "unmet_visits"]], on="sigungu_code", how="inner")
    unmet = base["unmet_visits"].fillna(0.0).to_numpy(dtype=float)
    baseline = float(unmet.sum())
    cands = build_candidates(regions, gaps, settings)

    if p <= 0 or cands.empty or baseline <= 0:
        return SitingSolution(run_id, max(p, 0), len(cands), "Skipped", 0.0, baseline)

    minutes = region_to_candidate_minutes(base, cands, settings)
    reachable, value = access_weights(minutes, settings)
    capacity = settings.capacity.annual_capacity(opt.new_site_bays)

    n_i, n_j = minutes.shape
    prob = pulp.LpProblem("capacitated_mclp", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(n_j)]
    pairs = [(i, j) for i in range(n_i) for j in range(n_j) if reachable[i, j] and unmet[i] > 0]
    z = {(i, j): pulp.LpVariable(f"z_{i}_{j}", lowBound=0) for i, j in pairs}

    gap_norm = cands["gap_score"].fillna(0).to_numpy(dtype=float) / 100.0
    prob += pulp.lpSum(value[i, j] * z[i, j] for i, j in pairs) + TIE_BREAK_EPS * pulp.lpSum(
        gap_norm[j] * x[j] for j in range(n_j)
    )

    by_region: dict[int, list] = {}
    by_cand: dict[int, list] = {}
    for i, j in pairs:
        by_region.setdefault(i, []).append(z[i, j])
        by_cand.setdefault(j, []).append(z[i, j])
    for i, zs in by_region.items():
        prob += pulp.lpSum(zs) <= unmet[i], f"unmet_{i}"
    for j in range(n_j):
        prob += pulp.lpSum(by_cand.get(j, [])) <= capacity * x[j], f"cap_{j}"
    prob += pulp.lpSum(x) <= p, "budget"

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=opt.solver_time_limit_sec)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]

    captured = np.zeros(n_j)
    access_val = np.zeros(n_j)
    for (i, j), var in z.items():
        amount = var.value() or 0.0
        captured[j] += amount
        access_val[j] += value[i, j] * amount

    opened = np.array([(v.value() or 0.0) > 0.5 for v in x])
    selected = cands.loc[opened].copy()
    selected["captured_visits"] = np.round(captured[opened], 1)
    selected["access_value"] = np.round(access_val[opened], 1)
    # 흡수량이 0인 개설은 의미가 없다 (용량이 남는 후보를 예산 채우기로 연 경우)
    selected = selected[selected["captured_visits"] > 0]
    selected = selected.sort_values(
        ["captured_visits", "gap_score"], ascending=[False, False]
    ).reset_index(drop=True)
    selected["priority"] = np.arange(1, len(selected) + 1)

    return SitingSolution(
        run_id=run_id,
        n_new_sites=p,
        candidate_count=len(cands),
        status=status,
        objective_value=round(float(selected["captured_visits"].sum()), 1),
        baseline_unmet=round(baseline, 1),
        selected=selected[SELECTED_COLUMNS],
    )


def run_siting_stage(settings: Settings, n_new_sites: int | None = None) -> SitingSolution:
    regions = read_table(settings.db_path, "regions")
    gaps = read_table(settings.db_path, "gap_scores")
    if gaps.empty:
        raise RuntimeError("gap_scores 가 비어 있습니다 — `snx analyze` 를 먼저 실행하세요")

    sol = solve_siting(regions, gaps, settings, n_new_sites=n_new_sites)

    run_row = pd.DataFrame(
        [
            {
                "run_id": sol.run_id,
                "n_new_sites": sol.n_new_sites,
                "candidate_count": sol.candidate_count,
                "objective_value": sol.objective_value,
                "solver_status": sol.status,
                "baseline_unmet": sol.baseline_unmet,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        ]
    )
    with get_conn(settings.db_path) as conn:
        write_df(conn, run_row, "siting_runs", replace=False)
        if not sol.selected.empty:
            write_df(conn, sol.selected.assign(run_id=sol.run_id), "siting_results", replace=False)

    sol.selected.to_csv(settings.outputs_dir / "siting_priority.csv", index=False)
    log.info(
        "입지 최적화 완료 [%s] — 신규 %d개소 · 흡수 %.0f건 (미충족의 %.1f%%)",
        sol.status,
        len(sol.selected),
        sol.objective_value,
        100 * sol.relief_ratio,
    )
    return sol
