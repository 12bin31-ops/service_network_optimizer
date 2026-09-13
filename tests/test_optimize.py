from __future__ import annotations

import pandas as pd

from snx.optimize.siting import build_candidates, solve_siting
from snx.storage.db import read_table


def test_solution_respects_budget(pipeline_run):
    sol = pipeline_run["solution"]
    assert len(sol.selected) <= 10
    assert sol.status in {"Optimal", "Skipped"}


def test_capture_cannot_exceed_baseline(pipeline_run):
    sol = pipeline_run["solution"]
    assert sol.objective_value <= sol.baseline_unmet + 1e-6
    assert 0 <= sol.relief_ratio <= 1


def test_more_sites_capture_more(settings, pipeline_run):
    regions = read_table(settings.db_path, "regions")
    gaps = read_table(settings.db_path, "gap_scores")
    small = solve_siting(regions, gaps, settings, n_new_sites=3)
    large = solve_siting(regions, gaps, settings, n_new_sites=12)
    assert large.objective_value >= small.objective_value - 1e-6


def test_zero_budget_captures_nothing(settings):
    regions = read_table(settings.db_path, "regions")
    gaps = read_table(settings.db_path, "gap_scores")
    sol = solve_siting(regions, gaps, settings, n_new_sites=0)
    assert sol.objective_value == 0.0
    assert sol.selected.empty


def test_candidates_drawn_from_top_gaps(settings):
    regions = read_table(settings.db_path, "regions")
    gaps = read_table(settings.db_path, "gap_scores")
    cands = build_candidates(regions, gaps, settings)
    assert len(cands) <= settings.optimization.candidate_top_n
    top_codes = set(gaps.nsmallest(len(cands), "gap_rank")["sigungu_code"])
    assert set(cands["sigungu_code"]) == top_codes


def test_priorities_are_unique_and_ordered(pipeline_run):
    sel = pipeline_run["solution"].selected
    if sel.empty:
        return
    assert sel["priority"].is_unique
    assert sel["captured_visits"].is_monotonic_decreasing
    assert isinstance(sel, pd.DataFrame)
