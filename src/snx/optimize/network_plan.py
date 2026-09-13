"""신설 + 증설 동시 최적화 — 같은 예산을 어디에 쓰는 게 미충족 수요를 가장 많이 줄이는가.

`siting.py` 는 '신규 거점 몇 개'만 결정한다. 그러나 용량 갭은 신설보다 기존 거점
워크베이 증설이 싸게 풀 수 있고, 접근성 갭은 신설로만 풀린다. 두 수단을 한 예산
안에서 경쟁시켜야 투자 배분이 정량화된다.

    변수   x_k ∈ {0,1}           후보지 k 신설
           y_j ∈ {0..Y}          기존 거점 j 워크베이 증설 개수
           z_ik ≥ 0              지역 i 미충족 수요 중 신설 거점 k 가 흡수
           e_ij ∈ [0, o_ij]      지역 i 가 거점 j 에서 겪는 대기(o_ij) 중 증설로 해소

    max    Σ v_ik z_ik + Σ v_ij e_ij  (+ 동률 해소항)
    s.t.   Σ_k z_ik + Σ_j e_ij ≤ unmet_i            ∀i   한 수요를 두 번 세지 않는다
           Σ_i z_ik ≤ C · x_k                        ∀k   신설 거점 설계용량
           Σ_i e_ij ≤ h · y_j                        ∀j   증설 1베이 = 건전 처리용량 h
           F · Σ x_k + c · Σ y_j ≤ B                      예산

    o_ij = D_i · w_ij · overflow_ratio_j  (커버리지 진단의 Huff 배정을 그대로 사용)

증설은 **대기(용량 갭)만** 풀 수 있고 30분 밖 수요(접근성 갭)는 못 푼다. 신설은 둘 다
풀지만 단가가 높다. 이 비대칭이 '어디는 짓고 어디는 늘릴지'를 가른다.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pulp

from snx.config import Settings
from snx.coverage.capacity import with_capacity
from snx.coverage.gap import assign_demand
from snx.logging_conf import get_logger
from snx.optimize.siting import (
    TIE_BREAK_EPS,
    access_weights,
    build_candidates,
    region_to_candidate_minutes,
    solve_siting,
)
from snx.storage.db import get_conn, read_table, write_df

log = get_logger(__name__)

# 이 값보다 작은 대기량 쌍은 변수로 만들지 않는다 (모형 크기 관리)
MIN_PAIR_VISITS = 1.0
MIP_GAP = 0.001

ACTION_COLUMNS = [
    "priority",
    "action",
    "target_id",
    "sigungu_code",
    "added_bays",
    "cost",
    "captured_visits",
]


@dataclass
class NetworkPlan:
    run_id: str
    budget: float
    status: str
    objective_value: float
    baseline_unmet: float
    new_only_value: float
    expand_only_value: float
    actions: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=ACTION_COLUMNS))

    @property
    def spent(self) -> float:
        return float(self.actions["cost"].sum()) if not self.actions.empty else 0.0

    @property
    def n_new_sites(self) -> int:
        return int((self.actions["action"] == "new").sum()) if not self.actions.empty else 0

    @property
    def n_expansions(self) -> int:
        return int((self.actions["action"] == "expand").sum()) if not self.actions.empty else 0

    @property
    def added_bays(self) -> int:
        return int(self.actions["added_bays"].sum()) if not self.actions.empty else 0

    @property
    def relief_ratio(self) -> float:
        if self.baseline_unmet <= 0:
            return 0.0
        return float(min(self.objective_value / self.baseline_unmet, 1.0))


def solve_network_plan(
    regions: pd.DataFrame,
    centers: pd.DataFrame,
    demand: pd.DataFrame,
    gaps: pd.DataFrame,
    settings: Settings,
    budget: float | None = None,
    *,
    allow_new: bool = True,
    allow_expand: bool = True,
    compare: bool = True,
) -> NetworkPlan:
    inv = settings.investment
    opt = settings.optimization
    budget = inv.budget if budget is None else float(budget)
    run_id = uuid.uuid4().hex[:12]

    base = regions.merge(demand[["sigungu_code", "annual_visits"]], on="sigungu_code", how="left")
    base = base.merge(gaps[["sigungu_code", "unmet_visits", "gap_score"]], on="sigungu_code", how="left")
    unmet = base["unmet_visits"].fillna(0.0).to_numpy(dtype=float)
    baseline = float(unmet.sum())

    new_only = expand_only = 0.0
    if compare:
        n_affordable = int(math.floor(budget / inv.new_site_cost + 1e-9))
        new_only = solve_siting(regions, gaps, settings, n_new_sites=n_affordable).objective_value
        expand_only = solve_network_plan(
            regions, centers, demand, gaps, settings, budget, allow_new=False, compare=False
        ).objective_value

    if budget <= 0 or baseline <= 0:
        return NetworkPlan(run_id, budget, "Skipped", 0.0, baseline, new_only, expand_only)

    prob = pulp.LpProblem("network_investment_plan", pulp.LpMaximize)
    objective: list = []
    region_terms: dict[int, list] = {}
    cost_terms: list = []

    # --- 신설 ------------------------------------------------------------------
    cands = build_candidates(regions, gaps, settings) if allow_new else pd.DataFrame()
    x: list = []
    z_by_site: dict[int, list] = {}
    if allow_new and not cands.empty:
        minutes = region_to_candidate_minutes(base, cands, settings)
        reachable, value = access_weights(minutes, settings)
        site_cap = settings.capacity.annual_capacity(opt.new_site_bays)
        x = [pulp.LpVariable(f"x_{k}", cat="Binary") for k in range(len(cands))]
        for i, k in zip(*np.nonzero(reachable), strict=True):
            if unmet[i] <= 0:
                continue
            var = pulp.LpVariable(f"z_{i}_{k}", lowBound=0)
            z_by_site.setdefault(k, []).append(var)
            objective.append(value[i, k] * var)
            region_terms.setdefault(i, []).append(var)
        for k in range(len(cands)):
            prob += pulp.lpSum(z_by_site.get(k, [])) <= site_cap * x[k], f"site_{k}"
        gap_norm = cands["gap_score"].fillna(0).to_numpy(dtype=float) / 100.0
        objective.append(TIE_BREAK_EPS * pulp.lpSum(gap_norm[k] * x[k] for k in range(len(cands))))
        cost_terms.append(inv.new_site_cost * pulp.lpSum(x))

    # --- 증설 ------------------------------------------------------------------
    ctr = with_capacity(centers, settings) if not centers.empty else centers
    y: dict[int, pulp.LpVariable] = {}
    e_by_center: dict[int, list] = {}
    if allow_expand and not ctr.empty and inv.max_added_bays_per_center > 0:
        assign = assign_demand(base, ctr, settings)
        overflow = assign.overflow_matrix
        _, value_c = access_weights(assign.minutes, settings)
        per_bay = settings.capacity.annual_capacity(1) * settings.capacity.healthy_utilization
        for j in np.nonzero(overflow.sum(axis=0) >= MIN_PAIR_VISITS)[0]:
            y[j] = pulp.LpVariable(
                f"y_{j}", lowBound=0, upBound=inv.max_added_bays_per_center, cat="Integer"
            )
            flows = e_by_center.setdefault(j, [])
            for i in np.nonzero(overflow[:, j] >= MIN_PAIR_VISITS)[0]:
                var = pulp.LpVariable(f"e_{i}_{j}", lowBound=0, upBound=float(overflow[i, j]))
                flows.append(var)
                objective.append(value_c[i, j] * var)
                region_terms.setdefault(i, []).append(var)
            prob += pulp.lpSum(flows) <= per_bay * y[j], f"bay_{j}"
        cost_terms.append(inv.bay_expansion_cost * pulp.lpSum(y.values()))

    if not objective or not cost_terms:
        return NetworkPlan(run_id, budget, "Skipped", 0.0, baseline, new_only, expand_only)

    prob += pulp.lpSum(objective)
    for i, terms in region_terms.items():
        prob += pulp.lpSum(terms) <= unmet[i], f"unmet_{i}"
    prob += pulp.lpSum(cost_terms) <= budget, "budget"

    # 상대 갭 0.1%: 흡수량 수백 건 차이로 수십 초를 더 쓰지 않는다
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=opt.solver_time_limit_sec, gapRel=MIP_GAP))
    status = pulp.LpStatus[prob.status]

    rows: list[dict] = []
    for k, var in enumerate(x):
        if (var.value() or 0) < 0.5:
            continue
        captured = sum((v.value() or 0.0) for v in z_by_site.get(k, []))
        if captured <= 0:
            continue
        c = cands.iloc[k]
        rows.append(
            {
                "action": "new",
                "target_id": c["sigungu_code"],
                "sigungu_code": c["sigungu_code"],
                "added_bays": opt.new_site_bays,
                "cost": inv.new_site_cost,
                "captured_visits": captured,
            }
        )
    for j, var in y.items():
        bays = int(round(var.value() or 0))
        if bays <= 0:
            continue
        captured = sum((v.value() or 0.0) for v in e_by_center[j])
        if captured <= 0:
            continue
        c = ctr.iloc[j]
        rows.append(
            {
                "action": "expand",
                "target_id": c["center_id"],
                "sigungu_code": c["sigungu_code"],
                "added_bays": bays,
                "cost": inv.bay_expansion_cost * bays,
                "captured_visits": captured,
            }
        )

    actions = pd.DataFrame(rows, columns=ACTION_COLUMNS[1:])
    if not actions.empty:
        actions["captured_visits"] = actions["captured_visits"].round(1)
        # 투자 1단위당 흡수 수요가 큰 조치부터 — 예산이 깎이면 뒤에서부터 덜어낸다
        actions["_efficiency"] = actions["captured_visits"] / actions["cost"]
        actions = actions.sort_values(["_efficiency", "captured_visits"], ascending=False)
        actions = actions.drop(columns="_efficiency").reset_index(drop=True)
    actions.insert(0, "priority", np.arange(1, len(actions) + 1))

    return NetworkPlan(
        run_id=run_id,
        budget=budget,
        status=status,
        objective_value=round(float(actions["captured_visits"].sum()) if not actions.empty else 0.0, 1),
        baseline_unmet=round(baseline, 1),
        new_only_value=new_only,
        expand_only_value=expand_only,
        actions=actions[ACTION_COLUMNS],
    )


def budget_sensitivity(
    settings: Settings, multipliers: tuple[float, ...] = (0.5, 1.0, 2.0)
) -> pd.DataFrame:
    """예산 규모별 최적 조합 — 어느 예산부터 신설이 선택되는지 본다.

    attrs['expand_ceiling'] 에 '예산 무제한일 때 증설만으로 흡수 가능한 상한'을 담는다.
    """
    db = settings.db_path
    regions = read_table(db, "regions")
    centers = read_table(db, "service_centers")
    demand = read_table(db, "demand")
    gaps = read_table(db, "gap_scores")
    if gaps.empty:
        return pd.DataFrame()

    inv = settings.investment
    rows = []
    for m in multipliers:
        budget = inv.budget * m
        plan = solve_network_plan(regions, centers, demand, gaps, settings, budget)
        rows.append(
            {
                "budget": budget,
                "n_new_sites": plan.n_new_sites,
                "n_expansions": plan.n_expansions,
                "added_bays": plan.added_bays,
                "objective_value": plan.objective_value,
                "new_only_value": plan.new_only_value,
                "uplift": plan.objective_value / max(plan.new_only_value, 1.0),
            }
        )
    out = pd.DataFrame(rows)

    unlimited = inv.bay_expansion_cost * inv.max_added_bays_per_center * max(len(centers), 1)
    out.attrs["expand_ceiling"] = solve_network_plan(
        regions, centers, demand, gaps, settings, unlimited, allow_new=False, compare=False
    ).objective_value
    return out


def run_plan_stage(settings: Settings, budget: float | None = None) -> NetworkPlan:
    db = settings.db_path
    regions = read_table(db, "regions")
    centers = read_table(db, "service_centers")
    demand = read_table(db, "demand")
    gaps = read_table(db, "gap_scores")
    if gaps.empty:
        raise RuntimeError("gap_scores 가 비어 있습니다 — `snx analyze` 를 먼저 실행하세요")

    plan = solve_network_plan(regions, centers, demand, gaps, settings, budget)

    run_row = pd.DataFrame(
        [
            {
                "run_id": plan.run_id,
                "budget": plan.budget,
                "spent": plan.spent,
                "n_new_sites": plan.n_new_sites,
                "n_expansions": plan.n_expansions,
                "added_bays": plan.added_bays,
                "objective_value": plan.objective_value,
                "new_only_value": plan.new_only_value,
                "expand_only_value": plan.expand_only_value,
                "baseline_unmet": plan.baseline_unmet,
                "solver_status": plan.status,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        ]
    )
    with get_conn(db) as conn:
        write_df(conn, run_row, "plan_runs", replace=False)
        if not plan.actions.empty:
            write_df(conn, plan.actions.assign(run_id=plan.run_id), "plan_actions", replace=False)

    names = centers.set_index("center_id")["name"]
    out = plan.actions.merge(regions[["sigungu_code", "sido", "sigungu"]], on="sigungu_code", how="left")
    out["target_name"] = np.where(
        out["action"] == "expand",
        out["target_id"].map(names),
        "신규 거점 " + out["sido"].fillna("") + " " + out["sigungu"].fillna(""),
    )
    out.to_csv(settings.outputs_dir / "investment_plan.csv", index=False)

    unit = settings.investment.cost_unit
    log.info(
        "투자안 최적화 완료 [%s] — 예산 %.0f%s 중 %.0f%s · 신설 %d개소 · 증설 %d곳(+%d베이) · 흡수 %.0f건 "
        "(신설만 %.0f건 · 증설만 %.0f건)",
        plan.status,
        plan.budget,
        unit,
        plan.spent,
        unit,
        plan.n_new_sites,
        plan.n_expansions,
        plan.added_bays,
        plan.objective_value,
        plan.new_only_value,
        plan.expand_only_value,
    )
    return plan
