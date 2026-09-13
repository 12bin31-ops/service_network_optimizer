"""거리 · 소요시간 계산.

실제 라우팅 API(OSRM/T-Map) 대신 직선거리 × 우회계수 × 도시등급별 평균속도로
소요시간을 근사한다. 전국 시군구 × 전체 거점 규모(수천 × 수십만 쌍)에서
반복 계산이 필요한데, 근사 오차보다 재현성과 실행 속도가 더 중요하기 때문이다.

정확한 주행시간이 필요해지면 :func:`travel_time_matrix` 만 라우팅 API 호출로
교체하면 되고, 상류 로직은 그대로 둘 수 있다.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088


def haversine_matrix(
    lat_a: np.ndarray, lon_a: np.ndarray, lat_b: np.ndarray, lon_b: np.ndarray
) -> np.ndarray:
    """(n_a, n_b) 직선거리 행렬 (km)."""
    lat_a = np.radians(np.asarray(lat_a, dtype=float))[:, None]
    lon_a = np.radians(np.asarray(lon_a, dtype=float))[:, None]
    lat_b = np.radians(np.asarray(lat_b, dtype=float))[None, :]
    lon_b = np.radians(np.asarray(lon_b, dtype=float))[None, :]

    dlat = lat_b - lat_a
    dlon = lon_b - lon_a
    h = np.sin(dlat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def travel_time_matrix(
    lat_a: np.ndarray,
    lon_a: np.ndarray,
    lat_b: np.ndarray,
    lon_b: np.ndarray,
    urban_class_a: np.ndarray,
    speed_kmh: dict[str, float],
    detour_factor: float,
) -> np.ndarray:
    """(n_a, n_b) 소요시간 행렬 (분).

    속도는 출발지(수요 지역)의 도시등급을 따른다. 통행의 대부분이 거주지
    주변 도로망에서 발생하기 때문이다.
    """
    km = haversine_matrix(lat_a, lon_a, lat_b, lon_b) * detour_factor
    speeds = np.array([speed_kmh[c] for c in urban_class_a], dtype=float)[:, None]
    return km / speeds * 60.0


# 지역을 하나의 점으로 다루기 때문에, 같은 시군구 안의 거점들 사이에서
# 소요시간이 0 에 가까워지면 가장 가까운 한 곳으로 수요가 비현실적으로 쏠린다.
# 실제 수요는 지역 내부에 면적으로 퍼져 있으므로 하한 통행시간을 둔다.
MIN_TRAVEL_MINUTES = 5.0


def huff_weights(minutes: np.ndarray, reachable: np.ndarray, decay: float) -> np.ndarray:
    """접근 가능한 거점들에 수요를 분배하는 Huff 가중치.

    w_ij ∝ max(t_ij, MIN)^(-decay), 도달 불가 쌍은 0.
    행 합이 1 (도달 가능 거점이 없으면 0).
    """
    safe = np.where(reachable, np.maximum(minutes, MIN_TRAVEL_MINUTES), np.nan)
    attractiveness = np.where(reachable, safe**(-decay), 0.0)
    row_sum = attractiveness.sum(axis=1, keepdims=True)
    return np.divide(
        attractiveness,
        row_sum,
        out=np.zeros_like(attractiveness),
        where=row_sum > 0,
    )
