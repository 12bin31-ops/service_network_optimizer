"""수집 계층 공통 규약.

두 가지 수집 모드를 동일한 인터페이스로 다룬다.

- ``sample`` : 공개 참조표(data/reference/sigungu.csv) 위에서 재현 가능한 합성
               모수를 생성한다. API 키 없이도 저장소가 항상 end-to-end 로 돈다.
- ``live``   : data/raw 에 내려받아 둔 공공 데이터 원본을 읽는다. 파일이 없으면
               어떤 파일을 어디서 받아야 하는지 명시한 예외를 던진다.

분석 로직은 수집 모드를 알지 못한다 — 아래 세 개의 표준 프레임만 소비한다.

regions         : sigungu_code, sido, sigungu, lat, lon, urban_class, population
vehicle_parc    : sigungu_code, age_bucket, vehicles, ev_vehicles, source
service_centers : center_id, name, center_type, sigungu_code, lat, lon, bays, source
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REGION_COLUMNS = ["sigungu_code", "sido", "sigungu", "lat", "lon", "urban_class", "population"]
PARC_COLUMNS = ["sigungu_code", "age_bucket", "vehicles", "ev_vehicles", "source"]
CENTER_COLUMNS = [
    "center_id",
    "name",
    "center_type",
    "sigungu_code",
    "lat",
    "lon",
    "bays",
    "source",
]


@dataclass(frozen=True)
class IngestResult:
    regions: pd.DataFrame
    vehicle_parc: pd.DataFrame
    service_centers: pd.DataFrame
    mode: str

    def validate(self) -> IngestResult:
        _require(self.regions, REGION_COLUMNS, "regions")
        _require(self.vehicle_parc, PARC_COLUMNS, "vehicle_parc")
        _require(self.service_centers, CENTER_COLUMNS, "service_centers")

        orphan = set(self.vehicle_parc["sigungu_code"]) - set(self.regions["sigungu_code"])
        if orphan:
            raise ValueError(f"regions 에 없는 지역코드가 vehicle_parc 에 있습니다: {sorted(orphan)[:5]}")
        orphan = set(self.service_centers["sigungu_code"]) - set(self.regions["sigungu_code"])
        if orphan:
            raise ValueError(f"regions 에 없는 지역코드가 service_centers 에 있습니다: {sorted(orphan)[:5]}")
        if (self.vehicle_parc["vehicles"] < 0).any():
            raise ValueError("vehicle_parc.vehicles 에 음수가 있습니다")
        if (self.service_centers["bays"] <= 0).any():
            raise ValueError("service_centers.bays 는 1 이상이어야 합니다")
        return self


class MissingSourceFile(FileNotFoundError):
    """live 모드에서 필요한 원본 파일이 없을 때."""

    def __init__(self, path: Path, description: str, url: str) -> None:
        super().__init__(
            f"필요한 원본 파일이 없습니다: {path}\n"
            f"  내용 : {description}\n"
            f"  출처 : {url}\n"
            f"  → 위 경로에 CSV 로 저장한 뒤 다시 실행하거나, --source sample 로 실행하세요."
        )


def _require(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} 프레임에 필수 컬럼이 없습니다: {missing}")
    if df.empty:
        raise ValueError(f"{name} 프레임이 비어 있습니다")


def load_reference_regions(reference_dir: Path) -> pd.DataFrame:
    """시군구 참조표를 읽는다.

    좌표는 행정구역 중심점의 근사값이다. 정밀 분석이 필요하면
    행정안전부 행정구역 경계(SHP)의 실제 centroid 로 교체한다.
    """
    path = Path(reference_dir) / "sigungu.csv"
    if not path.exists():
        raise MissingSourceFile(
            path,
            "시군구 코드 · 중심좌표 · 도시등급 · 인구",
            "https://sgis.kostat.go.kr (행정구역 경계) / https://kosis.kr (주민등록인구)",
        )
    df = pd.read_csv(path, dtype={"sigungu_code": str})
    return df[REGION_COLUMNS].copy()
