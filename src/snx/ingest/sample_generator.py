"""재현 가능한 합성 모수 생성기.

공개 참조표(인구 · 도시등급 · 좌표) 위에, 국내 자동차 시장에서 관측되는
구조적 패턴을 규칙으로 주입해 차량 모수와 서비스망을 만든다.

주입하는 구조적 패턴 (= 이 프로젝트가 검출하려는 현상)
1) 자동차 보유율은 대도시가 낮고 지방이 높다 (대중교통 대체재 효과)
2) 차령 분포는 대도시가 젊고 지방이 늙다 (신차 구매력 · 중고차 유입)
3) 전기차 보급률은 대도시/제주에 편중된다
4) 서비스망은 '과거의 판매량'을 따라 깔려 있어, 인구는 적지만 노후 차량이
   많은 지역에서 체계적으로 과소 배치된다  ← 갭의 원천

시드를 고정하므로 누구나 같은 결과를 재현할 수 있다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from snx.config import AGE_BUCKETS, Settings
from snx.ingest.base import CENTER_COLUMNS, PARC_COLUMNS, IngestResult, load_reference_regions

# 1) 도시등급별 인구 1인당 승용차 보유대수
MOTORIZATION = {"metro": 0.34, "city": 0.44, "rural": 0.52}

# 2) 도시등급별 차령 분포 (0-2 / 3-5 / 6-10 / 11+)
AGE_MIX = {
    "metro": (0.20, 0.24, 0.31, 0.25),
    "city": (0.16, 0.22, 0.32, 0.30),
    "rural": (0.11, 0.18, 0.31, 0.40),
}

# 3) 전국 평균 대비 전기차 보급 배수
EV_MULTIPLIER = {"metro": 1.35, "city": 0.95, "rural": 0.55}
EV_SPECIAL = {"50110": 2.6, "50130": 2.4}  # 제주 — 충전 인프라 · 보조금 효과

# 4) 서비스망 배치 규칙: 대도시는 수요 대비 촘촘, 지방은 성기게
#    (실제 네트워크가 과거 판매 실적 기준으로 깔린 결과를 모사)
CENTERS_PER_10K_VEHICLES = {"metro": 1.70, "city": 1.30, "rural": 0.80}

BAY_CHOICES = {"metro": (4, 6, 8), "city": (4, 6, 8), "rural": (3, 4, 6)}

# 하이테크센터(고전압 · 사고수리 거점)는 광역 단위로만 존재
HITECH_SIDOS = {"서울", "경기", "인천", "부산", "대구", "광주", "대전", "울산", "충남", "경남"}


def generate(settings: Settings, seed: int = 20260914) -> IngestResult:
    rng = np.random.default_rng(seed)
    regions = load_reference_regions(settings.reference_dir)

    parc = _build_parc(regions, settings, rng)
    centers = _build_centers(regions, parc, rng)

    return IngestResult(
        regions=regions,
        vehicle_parc=parc,
        service_centers=centers,
        mode="sample",
    ).validate()


# ---------------------------------------------------------------------------
# 차량 모수
# ---------------------------------------------------------------------------
def _build_parc(regions: pd.DataFrame, settings: Settings, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict] = []
    share = settings.market.hyundai_share
    base_ev = settings.market.ev_share

    for r in regions.itertuples(index=False):
        motor = MOTORIZATION[r.urban_class]
        # ±6% 지역 특이 변동 — 완전한 결정론적 직선 관계를 피한다
        noise = float(rng.normal(1.0, 0.06))
        brand_vehicles = r.population * motor * share * max(noise, 0.7)

        mix = AGE_MIX[r.urban_class]
        ev_rate = base_ev * EV_MULTIPLIER[r.urban_class] * EV_SPECIAL.get(r.sigungu_code, 1.0)
        ev_rate = min(ev_rate, 0.45)

        for bucket, weight in zip(AGE_BUCKETS, mix, strict=True):
            vehicles = brand_vehicles * weight
            # 전기차는 신차에 집중 — 차령이 올라갈수록 급감
            ev_bucket_weight = {"0-2": 1.0, "3-5": 0.45, "6-10": 0.08, "11+": 0.0}[bucket]
            ev_vehicles = vehicles * ev_rate * ev_bucket_weight
            rows.append(
                {
                    "sigungu_code": r.sigungu_code,
                    "age_bucket": bucket,
                    "vehicles": round(vehicles, 1),
                    "ev_vehicles": round(min(ev_vehicles, vehicles), 1),
                    "source": "sample",
                }
            )

    return pd.DataFrame(rows, columns=PARC_COLUMNS)


# ---------------------------------------------------------------------------
# 서비스망
# ---------------------------------------------------------------------------
def _build_centers(
    regions: pd.DataFrame, parc: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    totals = parc.groupby("sigungu_code", as_index=False)["vehicles"].sum()
    merged = regions.merge(totals, on="sigungu_code", how="left")

    rows: list[dict] = []
    for r in merged.itertuples(index=False):
        rate = CENTERS_PER_10K_VEHICLES[r.urban_class]
        expected = (r.vehicles / 10_000.0) * rate
        n_centers = int(rng.poisson(max(expected, 0.25)))
        # 최소 1개는 보장하되, 인구 5만 미만 군 지역은 0개일 수 있다 (실제 공백 재현)
        if n_centers == 0 and r.population >= 50_000:
            n_centers = 1

        for k in range(n_centers):
            lat, lon = _jitter(r.lat, r.lon, r.urban_class, rng)
            rows.append(
                {
                    "center_id": f"{r.sigungu_code}-B{k + 1:02d}",
                    "name": f"블루핸즈 {r.sigungu}{k + 1}호점",
                    "center_type": "bluehands",
                    "sigungu_code": r.sigungu_code,
                    "lat": round(lat, 5),
                    "lon": round(lon, 5),
                    "bays": int(rng.choice(BAY_CHOICES[r.urban_class])),
                    "source": "sample",
                }
            )

        # 하이테크센터: 광역 거점 시도의 상위 지역에 1개
        if r.sido in HITECH_SIDOS and r.vehicles >= 60_000:
            lat, lon = _jitter(r.lat, r.lon, r.urban_class, rng)
            rows.append(
                {
                    "center_id": f"{r.sigungu_code}-H01",
                    "name": f"하이테크센터 {r.sigungu}",
                    "center_type": "hitech",
                    "sigungu_code": r.sigungu_code,
                    "lat": round(lat, 5),
                    "lon": round(lon, 5),
                    "bays": int(rng.choice((10, 12, 14))),
                    "source": "sample",
                }
            )

    return pd.DataFrame(rows, columns=CENTER_COLUMNS)


def _jitter(
    lat: float, lon: float, urban_class: str, rng: np.random.Generator
) -> tuple[float, float]:
    """행정구역 중심점 주변으로 지점을 흩뿌린다 (도시일수록 좁게)."""
    spread_km = {"metro": 2.5, "city": 5.0, "rural": 9.0}[urban_class]
    d_lat = float(rng.normal(0, spread_km / 111.0))
    d_lon = float(rng.normal(0, spread_km / (111.0 * np.cos(np.radians(lat)))))
    return lat + d_lat, lon + d_lon
