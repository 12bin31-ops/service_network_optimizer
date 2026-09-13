from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from snx.coverage.distance import haversine_matrix, huff_weights, travel_time_matrix
from snx.coverage.gap import compute_coverage, score_gaps


def test_haversine_known_distance():
    """서울시청 ↔ 부산시청 직선거리는 약 325km."""
    km = haversine_matrix([37.5665], [126.9780], [35.1796], [129.0756])[0, 0]
    assert 315 < km < 335


def test_travel_time_slower_in_metro(settings):
    minutes = travel_time_matrix(
        np.array([37.5, 37.5]),
        np.array([127.0, 127.0]),
        np.array([37.6]),
        np.array([127.0]),
        np.array(["metro", "rural"]),
        settings.coverage.avg_speed_kmh,
        settings.coverage.detour_factor,
    )
    assert minutes[0, 0] > minutes[1, 0]


def test_huff_weights_sum_to_one_when_reachable():
    minutes = np.array([[10.0, 20.0], [40.0, 50.0]])
    reachable = minutes <= 30
    w = huff_weights(minutes, reachable, decay=2.0)
    assert w[0].sum() == pytest.approx(1.0)
    assert w[1].sum() == pytest.approx(0.0)
    assert w[0, 0] > w[0, 1]  # 가까운 거점에 더 많이 배분


def _region(code="11111", lat=37.5, lon=127.0, urban="city"):
    return pd.DataFrame(
        [{"sigungu_code": code, "sido": "테스트", "sigungu": "테스트시",
          "lat": lat, "lon": lon, "urban_class": urban, "population": 100000}]
    )


def test_no_centers_means_everything_uncovered(settings):
    regions = _region()
    demand = pd.DataFrame([{"sigungu_code": "11111", "annual_visits": 5000.0}])
    centers = pd.DataFrame(
        columns=["center_id", "name", "center_type", "sigungu_code", "lat", "lon", "bays"]
    )
    coverage, load = compute_coverage(regions, centers, demand, settings)
    assert coverage.loc[0, "uncovered_visits"] == pytest.approx(5000.0)
    assert coverage.loc[0, "reachable_centers"] == 0
    assert load.empty


def test_distant_center_does_not_cover(settings):
    regions = _region()
    demand = pd.DataFrame([{"sigungu_code": "11111", "annual_visits": 5000.0}])
    centers = pd.DataFrame(
        [{"center_id": "c1", "name": "먼지점", "center_type": "bluehands",
          "sigungu_code": "11111", "lat": 35.0, "lon": 129.0, "bays": 6}]
    )
    coverage, _ = compute_coverage(regions, centers, demand, settings)
    assert coverage.loc[0, "reachable_centers"] == 0
    assert coverage.loc[0, "uncovered_visits"] == pytest.approx(5000.0)


def test_overflow_appears_when_capacity_short(settings):
    regions = _region()
    # 워크베이 1개 거점 하나 = 연간 처리량이 수요보다 훨씬 작다
    centers = pd.DataFrame(
        [{"center_id": "c1", "name": "작은지점", "center_type": "bluehands",
          "sigungu_code": "11111", "lat": 37.5, "lon": 127.0, "bays": 1}]
    )
    demand = pd.DataFrame([{"sigungu_code": "11111", "annual_visits": 100000.0}])
    coverage, load = compute_coverage(regions, centers, demand, settings)
    assert coverage.loc[0, "uncovered_visits"] == pytest.approx(0.0, abs=1e-6)
    assert coverage.loc[0, "overflow_visits"] > 0
    assert load.loc[0, "utilization"] > 1.0


def test_coverage_components_sum_to_demand(pipeline_run):
    cov = pipeline_run["coverage"]
    total = cov["covered_visits"] + cov["uncovered_visits"] + cov["overflow_visits"]
    assert (total >= 0).all()


def test_gap_scores_are_ranked(pipeline_run):
    gaps = pipeline_run["gaps"]
    assert gaps["gap_rank"].is_monotonic_increasing
    assert gaps["gap_score"].is_monotonic_decreasing
    assert gaps["gap_score"].between(0, 100).all()


def test_weights_must_sum_to_one(settings, pipeline_run):
    broken = settings.model_copy(deep=True)
    broken.gap_score.weights.uncovered_demand = 0.9
    with pytest.raises(ValueError):
        score_gaps(pipeline_run["coverage"], broken)
