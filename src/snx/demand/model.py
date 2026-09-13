"""지역별 연간 정비 수요 추정.

핵심 아이디어
-------------
정비 수요는 '인구'나 '판매량'이 아니라 **도로 위에 굴러다니는 차량의 나이**가
결정한다. 따라서 수요는 차령 버킷별로 따로 계산하고, 공식 서비스망으로 들어오는
비율(loyalty)을 곱해 실수요로 환산한다.

    D_i = Σ_a  N_ia · r_a · λ_a          (내연기관)
        + Σ_a  N^ev_ia · r_a · λ_a · m   (전기차, m = ev_visit_multiplier)

    N_ia : 지역 i, 차령버킷 a 의 자사 보유대수
    r_a  : 차령버킷 a 의 연간 정비 입고 원단위 (회/대/년)
    λ_a  : 차령버킷 a 의 공식 서비스망 유입률
    m    : 전기차 정비 빈도 보정 (소모품 정비 감소)

이 구조 때문에 "인구는 적지만 차가 늙은 지역"이 수요 상위로 올라온다 —
서비스망 갭이 실제로 발생하는 지점이며, 판매량 기준 네트워크 배치가
놓치는 부분이다.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from snx.config import AGE_BUCKETS, Settings
from snx.logging_conf import get_logger
from snx.storage.db import get_conn, read_table, write_df

log = get_logger(__name__)


def estimate_demand(vehicle_parc: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """차량 모수 → 지역별 연간 정비 입고 수요.

    Returns
    -------
    DataFrame[sigungu_code, annual_visits, ice_visits, ev_visits, hitech_visits]
    """
    d = settings.demand
    unknown = set(vehicle_parc["age_bucket"]) - set(AGE_BUCKETS)
    if unknown:
        raise ValueError(f"알 수 없는 age_bucket: {sorted(unknown)}")

    df = vehicle_parc.copy()
    df["vehicles"] = df["vehicles"].astype(float)
    df["ev_vehicles"] = df["ev_vehicles"].astype(float).clip(lower=0)
    df["ev_vehicles"] = df[["vehicles", "ev_vehicles"]].min(axis=1)
    df["ice_vehicles"] = df["vehicles"] - df["ev_vehicles"]

    df["rate"] = df["age_bucket"].map(d.visits_per_vehicle_year)
    df["loyalty"] = df["age_bucket"].map(d.official_network_loyalty)
    if df[["rate", "loyalty"]].isna().any().any():
        raise ValueError("settings 의 차령 버킷 계수가 모든 버킷을 덮지 않습니다")

    df["ice_visits"] = df["ice_vehicles"] * df["rate"] * df["loyalty"]
    df["ev_visits"] = df["ev_vehicles"] * df["rate"] * df["loyalty"] * d.ev_visit_multiplier

    agg = (
        df.groupby("sigungu_code", as_index=False)[["ice_visits", "ev_visits"]]
        .sum()
        .assign(
            annual_visits=lambda x: x["ice_visits"] + x["ev_visits"],
            # 고전압 배터리 · 전동화 계통 정비는 하이테크센터로 흡수
            hitech_visits=lambda x: x["ev_visits"] * d.ev_hitech_ratio,
        )
    )
    return agg[
        ["sigungu_code", "annual_visits", "ice_visits", "ev_visits", "hitech_visits"]
    ].round(1)


def demand_drivers(vehicle_parc: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """지역별 수요 구성 분해 — 리포트에서 '왜 이 지역인가'를 설명하는 근거표."""
    d = settings.demand
    df = vehicle_parc.copy()
    df["rate"] = df["age_bucket"].map(d.visits_per_vehicle_year)
    df["loyalty"] = df["age_bucket"].map(d.official_network_loyalty)
    df["visits"] = df["vehicles"] * df["rate"] * df["loyalty"]

    pivot = df.pivot_table(
        index="sigungu_code", columns="age_bucket", values="visits", aggfunc="sum"
    ).fillna(0.0)
    pivot = pivot.reindex(columns=list(AGE_BUCKETS), fill_value=0.0)

    totals = pivot.sum(axis=1).replace(0, pd.NA)
    share = pivot.div(totals, axis=0).fillna(0.0)
    share.columns = [f"share_{c}" for c in share.columns]

    parc_total = df.groupby("sigungu_code")["vehicles"].sum().rename("vehicles_total")
    aged = (
        df[df["age_bucket"].isin(["6-10", "11+"])]
        .groupby("sigungu_code")["vehicles"]
        .sum()
        .rename("vehicles_aged")
    )
    out = pd.concat([pivot, share, parc_total, aged], axis=1).fillna(0.0)
    out["aged_vehicle_share"] = (out["vehicles_aged"] / out["vehicles_total"].replace(0, pd.NA)).fillna(0.0)
    return out.reset_index()


def run_demand_stage(settings: Settings) -> pd.DataFrame:
    """DB 에서 모수를 읽어 수요를 계산하고 다시 적재한다."""
    parc = read_table(settings.db_path, "vehicle_parc")
    if parc.empty:
        raise RuntimeError("vehicle_parc 가 비어 있습니다 — `snx ingest` 를 먼저 실행하세요")

    demand = estimate_demand(parc, settings)
    demand["computed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with get_conn(settings.db_path) as conn:
        write_df(conn, demand, "demand")

    log.info(
        "수요 추정 완료 — 전국 연간 %.0f건 (전기차 %.1f%%)",
        demand["annual_visits"].sum(),
        100 * demand["ev_visits"].sum() / max(demand["annual_visits"].sum(), 1),
    )
    return demand
