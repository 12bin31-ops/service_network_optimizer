"""수집 단계 오케스트레이션 — 모드를 골라 표준 프레임을 DB 에 적재한다."""

from __future__ import annotations

from snx.config import Settings
from snx.ingest import live_sources, sample_generator
from snx.ingest.base import IngestResult
from snx.logging_conf import get_logger
from snx.storage.db import get_conn, init_db, write_df

log = get_logger(__name__)

# 외래키 역순 — regions 를 다시 쓰기 전에 비워야 하는 하류 테이블
DOWNSTREAM_TABLES = (
    "reports",
    "plan_actions",
    "plan_runs",
    "siting_results",
    "siting_runs",
    "center_quality",
    "gap_scores",
    "coverage",
    "demand",
    "center_voc",
    "service_centers",
    "vehicle_parc",
)


def collect(settings: Settings, source: str = "sample", seed: int = 20260914) -> IngestResult:
    if source == "sample":
        return sample_generator.generate(settings, seed=seed)
    if source == "live":
        return live_sources.load(settings)
    raise ValueError(f"알 수 없는 source: {source!r} — 'sample' 또는 'live'")


def ingest(settings: Settings, source: str = "sample", seed: int = 20260914) -> IngestResult:
    """수집 → 검증 → 적재."""
    init_db(settings.db_path)
    result = collect(settings, source=source, seed=seed)

    with get_conn(settings.db_path) as conn:
        # 새 수집은 하류 산출물을 모두 무효화한다. 외래키 역순으로 비운다.
        for table in DOWNSTREAM_TABLES:
            conn.execute(f"DELETE FROM {table}")
        n_regions = write_df(conn, result.regions, "regions")
        n_parc = write_df(conn, result.vehicle_parc, "vehicle_parc")
        n_centers = write_df(conn, result.service_centers, "service_centers")
        n_voc = 0
        if result.center_voc is not None and not result.center_voc.empty:
            n_voc = write_df(conn, result.center_voc, "center_voc")

    total_vehicles = result.vehicle_parc["vehicles"].sum()
    log.info(
        "수집 완료 [%s] — 지역 %d개 · 차량모수 %d행(%.0f대) · 서비스거점 %d개 · VOC %d건",
        result.mode,
        n_regions,
        n_parc,
        total_vehicles,
        n_centers,
        n_voc,
    )
    return result
