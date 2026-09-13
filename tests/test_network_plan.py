from __future__ import annotations

import pytest

from snx.optimize.network_plan import budget_sensitivity, solve_network_plan
from snx.storage.db import read_table


@pytest.fixture(scope="module")
def frames(settings, pipeline_run):
    db = settings.db_path
    return tuple(read_table(db, t) for t in ("regions", "service_centers", "demand", "gap_scores"))


def test_plan_respects_budget(pipeline_run):
    plan = pipeline_run["plan"]
    assert plan.status == "Optimal"
    assert plan.spent <= plan.budget + 1e-6
    assert plan.objective_value <= plan.baseline_unmet + 1e-6


def test_mixed_plan_dominates_single_levers(pipeline_run):
    """두 수단을 함께 쓸 수 있는 모형은 어느 한 수단만 쓰는 모형보다 나빠질 수 없다."""
    plan = pipeline_run["plan"]
    tol = plan.baseline_unmet * 0.002  # MIP 상대 갭 허용
    assert plan.objective_value >= plan.new_only_value - tol
    assert plan.objective_value >= plan.expand_only_value - tol


def test_expansion_limits_respected(settings, pipeline_run):
    actions = pipeline_run["plan"].actions
    expands = actions[actions["action"] == "expand"]
    assert (expands["added_bays"] <= settings.investment.max_added_bays_per_center).all()
    per_bay = settings.capacity.annual_capacity(1) * settings.capacity.healthy_utilization
    assert (expands["captured_visits"] <= expands["added_bays"] * per_bay + 0.1).all()  # 소수 1자리 반올림


def test_zero_budget_does_nothing(settings, frames):
    plan = solve_network_plan(*frames, settings, budget=0, compare=False)
    assert plan.objective_value == 0.0
    assert plan.actions.empty


def test_expansion_only_touches_overloaded_centers(settings, frames):
    regions, centers, demand, gaps = frames
    plan = solve_network_plan(regions, centers, demand, gaps, settings, budget=100, allow_new=False, compare=False)
    assert (plan.actions["action"] == "expand").all()
    quality = read_table(settings.db_path, "center_quality").set_index("center_id")
    healthy = settings.capacity.healthy_utilization
    assert (quality.loc[plan.actions["target_id"], "utilization"] > healthy).all()


def test_expensive_expansion_shifts_to_new_sites(settings, frames):
    """증설 단가가 신설 베이당 단가보다 비싸지면 최적 조합이 신설로 이동한다."""
    pricey = settings.model_copy(deep=True)
    pricey.investment.bay_expansion_cost = 12.0
    cheap = solve_network_plan(*frames, settings, budget=300, compare=False)
    costly = solve_network_plan(*frames, pricey, budget=300, compare=False)
    assert costly.n_new_sites >= cheap.n_new_sites
    assert costly.added_bays <= cheap.added_bays


def test_sensitivity_is_monotone(settings, pipeline_run):
    small = settings.model_copy(deep=True)
    small.investment.budget = 100
    sens = budget_sensitivity(small, multipliers=(0.5, 1.0, 2.0))
    assert sens["objective_value"].is_monotonic_increasing
    ceiling = sens.attrs["expand_ceiling"]
    assert 0 < ceiling <= pipeline_run["plan"].baseline_unmet
