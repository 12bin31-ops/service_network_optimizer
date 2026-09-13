"""거점 처리 용량 산정."""

from __future__ import annotations

import numpy as np
import pandas as pd

from snx.config import Settings


def annual_capacity(bays: pd.Series | np.ndarray, settings: Settings) -> np.ndarray:
    """워크베이 수 → 연간 처리 가능 입고 건수."""
    cap = settings.capacity
    return np.asarray(bays, dtype=float) * cap.jobs_per_bay_day * cap.workdays_per_year


def with_capacity(centers: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """service_centers 에 capacity 컬럼을 붙여 반환."""
    out = centers.copy()
    out["capacity"] = annual_capacity(out["bays"], settings)
    return out


def utilization(assigned: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """부하율 = 배정 수요 / 처리 용량."""
    capacity = np.asarray(capacity, dtype=float)
    return np.divide(
        np.asarray(assigned, dtype=float),
        capacity,
        out=np.zeros_like(capacity),
        where=capacity > 0,
    )
