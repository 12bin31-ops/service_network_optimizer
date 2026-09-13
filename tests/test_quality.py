from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from snx.quality.scorecard import (
    ISSUE_NONE,
    ISSUE_OPERATION,
    ISSUE_OVERLOAD,
    ISSUE_WAIT,
    estimated_wait_days,
    load_complaint_correlation,
    score_centers,
)


def _load(utils):
    return pd.DataFrame(
        {
            "center_id": [f"c{i}" for i in range(len(utils))],
            "sigungu_code": "11111",
            "utilization": utils,
        }
    )


def test_wait_grows_nonlinearly(settings):
    w = estimated_wait_days(np.array([0.5, 0.8, 0.95]), settings)
    assert w[0] < w[1] < w[2]
    assert (w[2] - w[1]) > (w[1] - w[0])  # 부하가 1 에 가까울수록 대기가 폭증


def test_wait_is_capped(settings):
    w = estimated_wait_days(np.array([3.0]), settings)
    assert w[0] == pytest.approx(settings.quality.max_wait_days)


def test_issue_types_separate_causes(settings):
    """과부하 × 불만 신호 2×2 가 네 가지 원인 유형으로 정확히 갈린다."""
    n = 40
    utils = [0.5] * n
    voc = [2.0] * n
    comeback = [0.02] * n
    utils[0], voc[0], comeback[0] = 1.3, 9.0, 0.09   # 붐비고 불만 많음
    utils[1], voc[1], comeback[1] = 0.4, 9.5, 0.10   # 한가한데 불만 많음
    utils[2] = 1.2                                     # 붐비지만 불만은 평이
    load = _load(utils)
    voc_df = pd.DataFrame({"center_id": load["center_id"], "voc_per_1k_jobs": voc, "comeback_rate": comeback})

    q = score_centers(load, voc_df, settings).set_index("center_id")
    assert q.loc["c0", "issue_type"] == ISSUE_OVERLOAD
    assert q.loc["c1", "issue_type"] == ISSUE_OPERATION
    assert q.loc["c2", "issue_type"] == ISSUE_WAIT
    assert q.loc["c5", "issue_type"] == ISSUE_NONE
    assert q.loc["c0", "quality_risk"] > q.loc["c5", "quality_risk"]


def test_scorecard_runs_without_voc(settings):
    q = score_centers(_load([0.3, 0.9, 1.4]), None, settings)
    assert q["quality_risk"].between(0, 100).all()
    assert q.iloc[0]["utilization"] == pytest.approx(1.4)  # VOC 가 없으면 부하 축만으로 정렬
    assert set(q["issue_type"]) <= {ISSUE_NONE, ISSUE_WAIT}


def test_grades_follow_cutoffs(pipeline_run, settings):
    q = pipeline_run["quality"]
    lo, mid, hi = settings.quality.grade_cutoffs
    assert (q.loc[q["quality_grade"] == "A", "quality_risk"] < lo).all()
    assert (q.loc[q["quality_grade"] == "D", "quality_risk"] >= hi).all()


def test_sample_overload_drives_complaints(pipeline_run):
    """합성 데이터가 '부하 → 불만' 구조와 '부하 무관 운영형' 을 함께 담고 있는지."""
    q = pipeline_run["quality"]
    rho = load_complaint_correlation(q)
    assert rho is not None and rho > 0.2
    operation = q[q["issue_type"] == ISSUE_OPERATION]
    normal = q[q["issue_type"] == ISSUE_NONE]
    assert len(operation) > 0
    assert operation["voc_per_1k_jobs"].median() > normal["voc_per_1k_jobs"].median()
