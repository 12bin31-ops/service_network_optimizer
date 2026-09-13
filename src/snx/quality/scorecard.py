"""거점 품질 진단 — 부하 · VOC · 재입고율로 '관리 대상 거점'을 가려낸다.

갭 분석이 '어디에 거점이 모자라는가'를 본다면, 이 모듈은 '이미 있는 거점이
제대로 돌고 있는가'를 본다. 핵심은 품질 저하의 **원인을 둘로 분리**하는 것이다.
처방이 정반대이기 때문이다.

    과부하형 품질저하   붐벼서 대기가 길고 작업이 서둘러진다
                        → 워크베이 증설 · 예약 분산 (투자 과제)
    운영형 품질저하     한가한데도 불만과 재입고가 많다
                        → 정비 품질 점검 · 기술 교육 · 협력사 평가 (운영 과제)
    대기 리스크         아직 불만은 없지만 부하가 건전 수준을 넘었다
                        → 예방적 증설 검토

대기일수는 부하율 ρ 에 대한 대기행렬식 근사를 쓴다.

    wait = base_lead_days · (1 + ρ / (1 − ρ)),   ρ ≤ 0.98,   wait ≤ max_wait_days

ρ 가 1 에 가까워질수록 대기가 비선형으로 폭증한다 — 부하율 80% 와 95% 는
체감 품질이 전혀 다른 거점이라는 뜻이다.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from snx.config import Settings
from snx.coverage.gap import compute_coverage
from snx.logging_conf import get_logger
from snx.storage.db import get_conn, read_table, write_df

log = get_logger(__name__)

RHO_CAP = 0.98

QUALITY_COLUMNS = [
    "center_id",
    "sigungu_code",
    "utilization",
    "est_wait_days",
    "voc_per_1k_jobs",
    "comeback_rate",
    "quality_risk",
    "quality_grade",
    "issue_type",
]

ISSUE_OVERLOAD = "과부하형 품질저하"
ISSUE_OPERATION = "운영형 품질저하"
ISSUE_WAIT = "대기 리스크"
ISSUE_NONE = "정상"

PRESCRIPTIONS = {
    ISSUE_OVERLOAD: "워크베이 증설 · 예약 분산",
    ISSUE_OPERATION: "정비 품질 점검 · 기술 교육",
    ISSUE_WAIT: "예방적 증설 검토",
    ISSUE_NONE: "-",
}


def estimated_wait_days(utilization: np.ndarray | pd.Series, settings: Settings) -> np.ndarray:
    q = settings.quality
    rho = np.clip(np.asarray(utilization, dtype=float), 0.0, RHO_CAP)
    wait = q.base_lead_days * (1.0 + rho / (1.0 - rho))
    return np.minimum(wait, q.max_wait_days)


def score_centers(
    center_load: pd.DataFrame, voc: pd.DataFrame | None, settings: Settings
) -> pd.DataFrame:
    """거점 부하표(+ VOC) → 거점별 품질 리스크 0~100, 등급, 원인 유형."""
    q = settings.quality
    df = center_load[["center_id", "sigungu_code", "utilization"]].copy()
    if voc is not None and not voc.empty:
        df = df.merge(voc[["center_id", "voc_per_1k_jobs", "comeback_rate"]], on="center_id", how="left")
    else:
        df["voc_per_1k_jobs"] = np.nan
        df["comeback_rate"] = np.nan

    df["est_wait_days"] = np.round(estimated_wait_days(df["utilization"], settings), 2)

    axes = {
        "wait": (_minmax(df["est_wait_days"]), q.weights.wait),
        "voc": (_minmax(df["voc_per_1k_jobs"]), q.weights.voc),
        "comeback": (_minmax(df["comeback_rate"]), q.weights.comeback),
    }
    # 실적이 없는 축은 빼고 남은 가중치로 재정규화 — VOC 없는 live 환경에서도 돈다
    usable = {k: v for k, v in axes.items() if df[_axis_column(k)].notna().any()}
    total_w = sum(w for _, w in usable.values())
    risk = sum(norm.fillna(0.0) * w for norm, w in usable.values()) / max(total_w, 1e-9)
    df["quality_risk"] = np.round(100.0 * risk, 1)

    cuts = q.grade_cutoffs
    df["quality_grade"] = pd.cut(
        df["quality_risk"], bins=[-np.inf, *cuts, np.inf], labels=["A", "B", "C", "D"], right=False
    ).astype(str)

    df["issue_type"] = classify_issues(df, settings)
    return df.sort_values("quality_risk", ascending=False).reset_index(drop=True)[QUALITY_COLUMNS]


def classify_issues(df: pd.DataFrame, settings: Settings) -> pd.Series:
    """부하 초과 여부 × 불만 신호 여부로 원인 유형을 판정한다."""
    q = settings.quality
    overloaded = df["utilization"] > settings.capacity.healthy_utilization

    # 불만 지수 = VOC · 재입고율 백분위 순위의 평균. 한 지표만 튀는 우연보다
    # 두 지표가 함께 나쁜 거점을 잡는다. 상위 (1 − percentile) 만 '불만 신호'.
    ranks = [df[c].rank(pct=True) for c in ("voc_per_1k_jobs", "comeback_rate") if df[c].notna().any()]
    if ranks:
        complaint = pd.concat(ranks, axis=1).mean(axis=1)
        # method="min": 동점 무리는 가장 낮은 순위를 받아, 전원이 같은 값이면 아무도 신호가 아니다
        signal = complaint.rank(pct=True, method="min") > q.complaint_percentile
    else:
        signal = pd.Series(False, index=df.index)

    issue = pd.Series(ISSUE_NONE, index=df.index)
    issue[overloaded & ~signal] = ISSUE_WAIT
    issue[~overloaded & signal] = ISSUE_OPERATION
    issue[overloaded & signal] = ISSUE_OVERLOAD
    return issue


def load_complaint_correlation(quality: pd.DataFrame) -> float | None:
    """부하율과 VOC 의 순위상관 — '용량 부족이 불만으로 번지는가'의 요약 지표."""
    sub = quality[["utilization", "voc_per_1k_jobs"]].dropna()
    if len(sub) < 10:
        return None
    return float(sub["utilization"].rank().corr(sub["voc_per_1k_jobs"].rank()))


def run_quality_stage(settings: Settings) -> pd.DataFrame:
    db = settings.db_path
    regions = read_table(db, "regions")
    centers = read_table(db, "service_centers")
    demand = read_table(db, "demand")
    if demand.empty:
        raise RuntimeError("demand 가 비어 있습니다 — `snx demand` 를 먼저 실행하세요")
    if centers.empty:
        log.warning("서비스 거점이 없어 품질 진단을 건너뜁니다")
        return pd.DataFrame(columns=QUALITY_COLUMNS)

    _, center_load = compute_coverage(regions, centers, demand, settings)
    voc = read_table(db, "center_voc")
    quality = score_centers(center_load, voc, settings)

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with get_conn(db) as conn:
        write_df(conn, quality.assign(computed_at=stamp), "center_quality")

    out = quality.merge(centers[["center_id", "name", "center_type", "bays"]], on="center_id")
    out["prescription"] = out["issue_type"].map(PRESCRIPTIONS)
    out.to_csv(settings.outputs_dir / "center_quality.csv", index=False)

    counts = quality["issue_type"].value_counts()
    log.info(
        "거점 품질 진단 완료 — 과부하형 %d · 운영형 %d · 대기 리스크 %d · D등급 %d곳",
        counts.get(ISSUE_OVERLOAD, 0),
        counts.get(ISSUE_OPERATION, 0),
        counts.get(ISSUE_WAIT, 0),
        int((quality["quality_grade"] == "D").sum()),
    )
    return quality


def _axis_column(axis: str) -> str:
    return {"wait": "est_wait_days", "voc": "voc_per_1k_jobs", "comeback": "comeback_rate"}[axis]


def _minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(hi, lo):
        return pd.Series(np.zeros(len(s)), index=s.index).where(s.notna())
    return (s - lo) / (hi - lo)
