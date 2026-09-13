from __future__ import annotations

import pandas as pd
import pytest

from snx.demand.model import estimate_demand


def _parc(vehicles: float, bucket: str, ev: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sigungu_code": "99999",
                "age_bucket": bucket,
                "vehicles": vehicles,
                "ev_vehicles": ev,
                "source": "test",
            }
        ]
    )


def test_demand_matches_closed_form(settings):
    d = settings.demand
    out = estimate_demand(_parc(1000, "3-5"), settings)
    expected = 1000 * d.visits_per_vehicle_year["3-5"] * d.official_network_loyalty["3-5"]
    assert out.loc[0, "annual_visits"] == pytest.approx(expected, rel=1e-6)


def test_older_fleet_yields_fewer_official_visits(settings):
    """차령이 오르면 입고 원단위는 커지지만 공식망 유입률이 더 크게 떨어진다."""
    young = estimate_demand(_parc(1000, "0-2"), settings).loc[0, "annual_visits"]
    old = estimate_demand(_parc(1000, "11+"), settings).loc[0, "annual_visits"]
    assert young > old


def test_ev_reduces_visits(settings):
    ice = estimate_demand(_parc(1000, "0-2", ev=0), settings).loc[0, "annual_visits"]
    ev = estimate_demand(_parc(1000, "0-2", ev=1000), settings).loc[0, "annual_visits"]
    assert ev < ice
    assert ev == pytest.approx(ice * settings.demand.ev_visit_multiplier, rel=1e-6)


def test_ev_cannot_exceed_total_fleet(settings):
    out = estimate_demand(_parc(100, "0-2", ev=500), settings)
    assert out.loc[0, "ev_visits"] > 0
    assert out.loc[0, "ice_visits"] == pytest.approx(0.0, abs=1e-9)


def test_unknown_bucket_rejected(settings):
    with pytest.raises(ValueError):
        estimate_demand(_parc(10, "20+"), settings)


def test_stage_totals_are_positive(pipeline_run):
    demand = pipeline_run["demand"]
    assert (demand["annual_visits"] > 0).all()
    assert demand["annual_visits"].sum() > demand["ev_visits"].sum()


def test_warranty_split_matches_closed_form(settings):
    w = settings.demand.warranty_share
    out = estimate_demand(_parc(1000, "3-5", ev=200), settings).loc[0]
    expected = out["ice_visits"] * w.ice["3-5"] + out["ev_visits"] * w.ev["3-5"]
    assert out["warranty_visits"] == pytest.approx(expected, abs=0.2)
    assert out["warranty_visits"] + out["paid_visits"] == pytest.approx(out["annual_visits"], abs=0.2)


def test_aged_fleet_has_no_warranty(settings):
    out = estimate_demand(_parc(1000, "11+", ev=100), settings).loc[0]
    assert out["warranty_visits"] == pytest.approx(0.0, abs=1e-9)


def test_ev_battery_warranty_outlives_ice(settings):
    """배터리 보증이 길어 6-10 버킷에서도 전기차는 보증 비중이 남는다."""
    ice = estimate_demand(_parc(1000, "6-10", ev=0), settings).loc[0]
    ev = estimate_demand(_parc(1000, "6-10", ev=1000), settings).loc[0]
    assert ev["warranty_visits"] / ev["annual_visits"] > ice["warranty_visits"] / ice["annual_visits"]
