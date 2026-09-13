from __future__ import annotations

import pytest

from snx.policy.scenario import apply_overrides, parse_overrides, simulate_loyalty


def test_parse_overrides():
    assert parse_overrides(["6-10=0.5", "11+ = 0.25"]) == {"6-10": 0.5, "11+": 0.25}


@pytest.mark.parametrize("bad", [["6-10"], ["20+=0.3"], ["6-10=1.5"]])
def test_parse_overrides_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        parse_overrides(bad)


def test_apply_overrides_does_not_mutate(settings):
    before = dict(settings.demand.official_network_loyalty)
    scen = apply_overrides(settings, {"6-10": 0.9})
    assert settings.demand.official_network_loyalty == before
    assert scen.demand.official_network_loyalty["6-10"] == 0.9


def test_no_change_means_no_delta(settings, pipeline_run):
    same = settings.demand.official_network_loyalty["3-5"]
    r = simulate_loyalty(settings, {"3-5": same})
    assert r.summary["delta_visits"] == pytest.approx(0.0, abs=1.0)
    assert r.summary["delta_unmet"] == pytest.approx(0.0, abs=1.0)


def test_higher_loyalty_raises_demand_and_unmet(settings, pipeline_run):
    r = simulate_loyalty(settings, {"6-10": 0.55, "11+": 0.30})
    sm = r.summary
    assert sm["delta_visits"] > 0
    assert sm["delta_unmet"] >= 0
    assert 0 <= sm["leak_ratio"] <= 1
    assert sm["overloaded_regions_scenario"] >= sm["overloaded_regions_base"]
    assert r.regions["delta_unmet"].is_monotonic_decreasing


def test_scenario_leaves_db_untouched(settings, pipeline_run):
    from snx.storage.db import read_table

    before = read_table(settings.db_path, "demand")["annual_visits"].sum()
    simulate_loyalty(settings, {"0-2": 0.99})
    after = read_table(settings.db_path, "demand")["annual_visits"].sum()
    assert before == pytest.approx(after)
