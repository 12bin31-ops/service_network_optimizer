"""공공 데이터 원본 로더 (live 모드).

원본을 코드에서 직접 크롤링하지 않는다. 통계표는 배포 형식이 자주 바뀌고
이용약관도 소스마다 다르므로, **사람이 내려받아 data/raw 에 둔 파일**을 읽어
표준 스키마로 정규화하는 책임만 이 모듈이 진다.

필요한 파일과 출처
------------------------------------------------------------------------------
data/raw/vehicle_registration.csv
    시군구 × 차령 × 연료 별 승용차 등록대수
    국토교통 통계누리 「자동차 등록현황보고」 / KOSIS 자동차등록대수 통계
    https://stat.molit.go.kr  ·  https://kosis.kr
    기대 컬럼: sigungu_code, age_bucket, vehicles, ev_vehicles

data/raw/service_centers.csv
    서비스 거점 목록 (지점명 · 좌표 · 워크베이 수)
    브랜드 공식 서비스망 안내 페이지에서 정리하거나, 사내 네트워크 마스터를 사용
    기대 컬럼: center_id, name, center_type, sigungu_code, lat, lon, bays

data/raw/center_voc.csv                                              (선택)
    거점별 VOC · 재입고율 반기 집계 — VOC 관리 시스템 · 정비 이력(DMS) 에서 추출
    기대 컬럼: center_id, period, voc_per_1k_jobs, comeback_rate
    없으면 품질 진단은 부하율(대기일수) 축만으로 계산한다.

data/reference/sigungu.csv
    시군구 코드 · 중심좌표 · 도시등급 · 인구 (저장소에 동봉)
    행정구역 경계는 SGIS, 인구는 KOSIS 주민등록인구 기준으로 갱신
------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from snx.config import AGE_BUCKETS, Settings
from snx.ingest.base import (
    CENTER_COLUMNS,
    PARC_COLUMNS,
    VOC_COLUMNS,
    IngestResult,
    MissingSourceFile,
    load_reference_regions,
)

VEHICLE_FILE = "vehicle_registration.csv"
CENTER_FILE = "service_centers.csv"
VOC_FILE = "center_voc.csv"


def load(settings: Settings) -> IngestResult:
    regions = load_reference_regions(settings.reference_dir)
    parc = _load_parc(settings.raw_dir, settings)
    centers = _load_centers(settings.raw_dir)

    # 참조표에 없는 지역은 분석 대상에서 제외 (경고 대신 명시적 필터)
    valid = set(regions["sigungu_code"])
    parc = parc[parc["sigungu_code"].isin(valid)].copy()
    centers = centers[centers["sigungu_code"].isin(valid)].copy()
    voc = _load_voc(settings.raw_dir)
    if voc is not None:
        voc = voc[voc["center_id"].isin(set(centers["center_id"]))].copy()

    return IngestResult(
        regions=regions,
        vehicle_parc=parc,
        service_centers=centers,
        mode="live",
        center_voc=voc,
    ).validate()


def _load_parc(raw_dir: Path, settings: Settings) -> pd.DataFrame:
    path = Path(raw_dir) / VEHICLE_FILE
    if not path.exists():
        raise MissingSourceFile(
            path,
            "시군구 × 차령 × 연료별 승용차 등록대수",
            "https://stat.molit.go.kr (자동차 등록현황보고) / https://kosis.kr",
        )
    df = pd.read_csv(path, dtype={"sigungu_code": str})

    required = {"sigungu_code", "age_bucket", "vehicles"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} 에 필수 컬럼이 없습니다: {sorted(missing)}")

    bad = set(df["age_bucket"]) - set(AGE_BUCKETS)
    if bad:
        raise ValueError(f"알 수 없는 age_bucket: {sorted(bad)} — 허용값 {list(AGE_BUCKETS)}")

    if "ev_vehicles" not in df.columns:
        df["ev_vehicles"] = 0.0

    # 등록대수는 전 브랜드 합계이므로 브랜드 점유율을 적용해 자사 모수로 환산
    share = settings.market.hyundai_share
    df["vehicles"] = df["vehicles"].astype(float) * share
    df["ev_vehicles"] = df["ev_vehicles"].astype(float) * share
    df["source"] = "molit"
    return df[PARC_COLUMNS]


def _load_centers(raw_dir: Path) -> pd.DataFrame:
    path = Path(raw_dir) / CENTER_FILE
    if not path.exists():
        raise MissingSourceFile(
            path,
            "서비스 거점 목록 (지점명 · 좌표 · 워크베이 수)",
            "브랜드 공식 서비스망 안내 / 사내 네트워크 마스터",
        )
    df = pd.read_csv(path, dtype={"sigungu_code": str, "center_id": str})
    if "center_type" not in df.columns:
        df["center_type"] = "bluehands"
    if "bays" not in df.columns:
        df["bays"] = 5
    df["source"] = "network_master"
    return df[CENTER_COLUMNS]


def _load_voc(raw_dir: Path) -> pd.DataFrame | None:
    """선택 파일 — 없으면 None (품질 진단이 부하 축만 사용)."""
    path = Path(raw_dir) / VOC_FILE
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"center_id": str})
    missing = {"center_id", "voc_per_1k_jobs", "comeback_rate"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} 에 필수 컬럼이 없습니다: {sorted(missing)}")
    if "period" not in df.columns:
        df["period"] = "unknown"
    # 재입고율을 % 로 넣은 경우를 비율로 환산
    if df["comeback_rate"].max() > 1:
        df["comeback_rate"] = df["comeback_rate"] / 100.0
    df["source"] = "voc_system"
    return df[VOC_COLUMNS]
