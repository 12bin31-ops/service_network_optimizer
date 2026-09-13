"""커버리지 계산과 갭 스코어링 — 이 프로젝트의 핵심 분석부.

두 종류의 갭을 구분해서 잡는다. 실무에서 처방이 완전히 다르기 때문이다.

1) **접근성 갭 (uncovered)** — 30분 내 도달 가능한 거점이 아예 없다.
   → 처방: 신규 거점 신설
2) **용량 갭 (overflow)** — 거점은 있는데 그 거점이 이미 과부하다.
   → 처방: 기존 거점 워크베이 증설 / 인근 분산

수요 분배는 Huff 모형을 쓴다. 최근접 거점에 100% 몰아주는 방식은
대도시처럼 거점이 촘촘한 지역의 부하를 과대평가한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from snx.config import Settings
from snx.coverage.capacity import utilization, with_capacity
from snx.coverage.distance import huff_weights, travel_time_matrix
from snx.logging_conf import get_logger
from snx.storage.db import get_conn, read_table, write_df

log = get_logger(__name__)

COVERAGE_COLUMNS = [
    "sigungu_code",
    "nearest_minutes",
    "reachable_centers",
    "reachable_capacity",
    "covered_visits",
    "uncovered_visits",
    "overflow_visits",
    "avg_utilization",
]


@dataclass(frozen=True)
class Assignment:
    """지역 × 거점 수요 배정 행렬 — 커버리지 진단과 증설 최적화가 같은 배정을 본다."""

    minutes: np.ndarray          # (n_regions, n_centers) 소요시간
    reachable: np.ndarray        # (n_regions, n_centers) 접근 기준 이내
    weights: np.ndarray          # (n_regions, n_centers) Huff 가중치
    demand: np.ndarray           # (n_regions,) 연간 수요
    capacity: np.ndarray         # (n_centers,) 연간 처리 용량
    assigned: np.ndarray         # (n_centers,) 배정 수요
    overflow_ratio: np.ndarray   # (n_centers,) 배정 수요 중 건전 부하율 초과로 대기가 되는 몫

    @property
    def overflow_matrix(self) -> np.ndarray:
        """(n_regions, n_centers) 지역 i 수요 중 거점 j 에서 대기로 남는 양."""
        return self.weights * self.demand[:, None] * self.overflow_ratio[None, :]


def assign_demand(
    base: pd.DataFrame, centers: pd.DataFrame, settings: Settings
) -> Assignment:
    """수요(base: regions × demand)를 접근 가능 거점에 Huff 로 배정한다."""
    cov = settings.coverage
    ctr = centers if "capacity" in centers.columns else with_capacity(centers, settings)

    minutes = travel_time_matrix(
        base["lat"].to_numpy(),
        base["lon"].to_numpy(),
        ctr["lat"].to_numpy(),
        ctr["lon"].to_numpy(),
        base["urban_class"].to_numpy(),
        cov.avg_speed_kmh,
        cov.detour_factor,
    )
    reachable = minutes <= cov.max_travel_minutes
    weights = huff_weights(minutes, reachable, cov.huff_distance_decay)

    demand = base["annual_visits"].fillna(0.0).to_numpy(dtype=float)
    assigned = (weights * demand[:, None]).sum(axis=0)
    capacity = ctr["capacity"].to_numpy(dtype=float)
    healthy = capacity * settings.capacity.healthy_utilization

    # 거점이 감당 가능한 수준을 넘긴 비율 — 그 지점으로 흘러간 수요 중 '대기'로 남는 몫
    overflow_ratio = np.divide(
        np.maximum(assigned - healthy, 0.0),
        assigned,
        out=np.zeros_like(assigned),
        where=assigned > 0,
    )
    return Assignment(minutes, reachable, weights, demand, capacity, assigned, overflow_ratio)


def compute_coverage(
    regions: pd.DataFrame,
    centers: pd.DataFrame,
    demand: pd.DataFrame,
    settings: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """지역별 커버리지와 거점별 부하를 동시에 계산한다.

    Returns
    -------
    (coverage_df, center_load_df)
    """
    base = regions.merge(demand, on="sigungu_code", how="left")
    base["annual_visits"] = base["annual_visits"].fillna(0.0)

    if centers.empty:
        empty_load = pd.DataFrame(
            columns=["center_id", "assigned_visits", "capacity", "utilization"]
        )
        coverage = base[["sigungu_code"]].copy()
        coverage["nearest_minutes"] = np.inf
        coverage["reachable_centers"] = 0
        coverage["reachable_capacity"] = 0.0
        coverage["covered_visits"] = 0.0
        coverage["uncovered_visits"] = base["annual_visits"].to_numpy()
        coverage["overflow_visits"] = 0.0
        coverage["avg_utilization"] = 0.0
        return coverage[COVERAGE_COLUMNS], empty_load

    ctr = with_capacity(centers, settings)
    a = assign_demand(base, ctr, settings)
    minutes, reachable, weights = a.minutes, a.reachable, a.weights
    assigned_j, capacity_j, overflow_ratio_j = a.assigned, a.capacity, a.overflow_ratio
    util_j = utilization(assigned_j, capacity_j)

    covered_share_i = weights.sum(axis=1)                     # 0~1
    uncovered_i = base["annual_visits"].to_numpy() * (1.0 - covered_share_i)
    overflow_i = (weights * overflow_ratio_j[None, :]).sum(axis=1) * base["annual_visits"].to_numpy()
    covered_i = base["annual_visits"].to_numpy() - uncovered_i - overflow_i

    nearest = minutes.min(axis=1)
    reachable_count = reachable.sum(axis=1)
    reachable_capacity = (weights * capacity_j[None, :]).sum(axis=1)
    avg_util = (weights * util_j[None, :]).sum(axis=1)

    coverage = pd.DataFrame(
        {
            "sigungu_code": base["sigungu_code"].to_numpy(),
            "nearest_minutes": np.round(nearest, 2),
            "reachable_centers": reachable_count.astype(int),
            "reachable_capacity": np.round(reachable_capacity, 1),
            "covered_visits": np.round(np.maximum(covered_i, 0.0), 1),
            "uncovered_visits": np.round(np.maximum(uncovered_i, 0.0), 1),
            "overflow_visits": np.round(np.maximum(overflow_i, 0.0), 1),
            "avg_utilization": np.round(avg_util, 4),
        }
    )

    center_load = pd.DataFrame(
        {
            "center_id": ctr["center_id"].to_numpy(),
            "name": ctr["name"].to_numpy(),
            "center_type": ctr["center_type"].to_numpy(),
            "sigungu_code": ctr["sigungu_code"].to_numpy(),
            "lat": ctr["lat"].to_numpy(),
            "lon": ctr["lon"].to_numpy(),
            "capacity": np.round(capacity_j, 1),
            "assigned_visits": np.round(assigned_j, 1),
            "utilization": np.round(util_j, 4),
        }
    )
    return coverage[COVERAGE_COLUMNS], center_load


def score_gaps(coverage: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """세 축을 0~1 로 정규화해 가중합 → 0~100 갭 스코어.

    - unmet_visits : 미충족 수요 (접근성 갭 + 용량 갭). 두터운 꼬리 분포라 log 변환
    - access_penalty : 최근접 거점까지 소요시간 (기준시간의 2배에서 포화)
    - load_penalty : 건전 부하율 초과분
    """
    w = settings.gap_score.weights
    total_w = w.uncovered_demand + w.access_penalty + w.load_penalty
    if not np.isclose(total_w, 1.0, atol=1e-6):
        raise ValueError(f"gap_score.weights 합이 1이 아닙니다: {total_w}")

    df = coverage.copy()
    df["unmet_visits"] = df["uncovered_visits"] + df["overflow_visits"]

    cap_minutes = settings.coverage.max_travel_minutes * 2
    access_raw = df["nearest_minutes"].replace([np.inf, -np.inf], cap_minutes).clip(0, cap_minutes)
    load_raw = (df["avg_utilization"] - settings.capacity.healthy_utilization).clip(lower=0)

    df["access_penalty"] = _minmax(access_raw).round(4)
    df["load_penalty"] = _minmax(load_raw).round(4)
    unmet_norm = _minmax(np.log1p(df["unmet_visits"]))

    df["gap_score"] = (
        100.0
        * (
            w.uncovered_demand * unmet_norm
            + w.access_penalty * df["access_penalty"]
            + w.load_penalty * df["load_penalty"]
        )
    ).round(2)

    df = df.sort_values("gap_score", ascending=False).reset_index(drop=True)
    df["gap_rank"] = np.arange(1, len(df) + 1)
    return df[
        ["sigungu_code", "unmet_visits", "access_penalty", "load_penalty", "gap_score", "gap_rank"]
    ]


def _minmax(series: pd.Series | np.ndarray) -> pd.Series:
    s = pd.Series(np.asarray(series, dtype=float))
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(hi, lo):
        return pd.Series(np.zeros(len(s)))
    return (s - lo) / (hi - lo)


def run_coverage_stage(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    regions = read_table(settings.db_path, "regions")
    centers = read_table(settings.db_path, "service_centers")
    demand = read_table(settings.db_path, "demand")
    if demand.empty:
        raise RuntimeError("demand 가 비어 있습니다 — `snx demand` 를 먼저 실행하세요")

    coverage, center_load = compute_coverage(regions, centers, demand, settings)
    gaps = score_gaps(coverage, settings)

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    coverage_out = coverage.assign(computed_at=stamp)
    gaps_out = gaps.assign(computed_at=stamp)

    with get_conn(settings.db_path) as conn:
        write_df(conn, coverage_out, "coverage")
        write_df(conn, gaps_out, "gap_scores")

    center_load.to_csv(settings.outputs_dir / "center_load.csv", index=False)

    unmet = gaps["unmet_visits"].sum()
    total = coverage["covered_visits"].sum() + unmet
    log.info(
        "커버리지 분석 완료 — 미충족 수요 %.0f건 / 전체 %.0f건 (%.1f%%), 30분 공백 지역 %d곳",
        unmet,
        total,
        100 * unmet / max(total, 1),
        int((coverage["reachable_centers"] == 0).sum()),
    )
    return coverage, gaps
