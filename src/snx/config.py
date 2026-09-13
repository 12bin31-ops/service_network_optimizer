"""설정 로딩 및 스키마 정의.

config/settings.yaml 을 pydantic 모델로 검증해 로드한다.
모든 모듈은 전역 상수 대신 Settings 객체를 주입받아 동작한다.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

AGE_BUCKETS: tuple[str, ...] = ("0-2", "3-5", "6-10", "11+")
URBAN_CLASSES: tuple[str, ...] = ("metro", "city", "rural")


def project_root() -> Path:
    """저장소 루트 (src/snx/config.py 기준 2단계 상위)."""
    return Path(__file__).resolve().parents[2]


class PathSettings(BaseModel):
    db: str
    reference_dir: str
    raw_dir: str
    outputs_dir: str


class MarketSettings(BaseModel):
    hyundai_share: float = Field(gt=0, le=1)
    ev_share: float = Field(ge=0, le=1)


class DemandSettings(BaseModel):
    visits_per_vehicle_year: dict[str, float]
    official_network_loyalty: dict[str, float]
    ev_visit_multiplier: float
    ev_hitech_ratio: float


class CapacitySettings(BaseModel):
    jobs_per_bay_day: float
    workdays_per_year: int
    healthy_utilization: float

    def annual_capacity(self, bays: float) -> float:
        return bays * self.jobs_per_bay_day * self.workdays_per_year


class CoverageSettings(BaseModel):
    max_travel_minutes: float
    detour_factor: float
    avg_speed_kmh: dict[str, float]
    huff_distance_decay: float


class GapWeights(BaseModel):
    uncovered_demand: float
    access_penalty: float
    load_penalty: float


class GapScoreSettings(BaseModel):
    weights: GapWeights


class OptimizationSettings(BaseModel):
    n_new_sites: int
    candidate_top_n: int
    new_site_bays: int
    access_value_alpha: float = 0.0
    solver_time_limit_sec: int


class ReportSettings(BaseModel):
    top_n_regions: int
    language: str


class LoggingSettings(BaseModel):
    level: str = "INFO"


class Settings(BaseModel):
    paths: PathSettings
    market: MarketSettings
    demand: DemandSettings
    capacity: CapacitySettings
    coverage: CoverageSettings
    gap_score: GapScoreSettings
    optimization: OptimizationSettings
    report: ReportSettings
    logging: LoggingSettings = LoggingSettings()

    # --- 경로 헬퍼 -------------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self._abs(self.paths.db)

    @property
    def reference_dir(self) -> Path:
        return self._abs(self.paths.reference_dir)

    @property
    def raw_dir(self) -> Path:
        return self._abs(self.paths.raw_dir)

    @property
    def outputs_dir(self) -> Path:
        p = self._abs(self.paths.outputs_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def _abs(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else project_root() / path


DEFAULT_CONFIG = project_root() / "config" / "settings.yaml"


@lru_cache(maxsize=4)
def load_settings(path: str | os.PathLike[str] | None = None) -> Settings:
    """설정 파일을 읽어 Settings 로 반환 (경로별 캐시)."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Settings(**raw)
