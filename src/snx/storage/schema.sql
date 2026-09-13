-- =============================================================================
-- snx 저장소 스키마 (SQLite)
-- 단계 산출물을 테이블로 물질화한다. 모든 문장은 멱등(IF NOT EXISTS).
--
--   regions ─┬─ vehicle_parc
--            ├─ service_centers ── center_voc          [수집]
--            ├─ demand                                  [1 수요]
--            ├─ coverage · gap_scores                   [2~3 커버리지·갭]
--            ├─ center_quality                          [3' 거점 품질]
--            ├─ siting_runs ── siting_results           [4 신설 최적화]
--            └─ plan_runs ── plan_actions               [4' 신설+증설 투자안]
--                                reports                [5 리포트]
-- =============================================================================

CREATE TABLE IF NOT EXISTS regions (
    sigungu_code TEXT PRIMARY KEY,
    sido         TEXT NOT NULL,
    sigungu      TEXT NOT NULL,
    lat          REAL NOT NULL,
    lon          REAL NOT NULL,
    urban_class  TEXT NOT NULL CHECK (urban_class IN ('metro', 'city', 'rural')),
    population   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS vehicle_parc (
    sigungu_code TEXT NOT NULL,
    age_bucket   TEXT NOT NULL CHECK (age_bucket IN ('0-2', '3-5', '6-10', '11+')),
    vehicles     REAL NOT NULL,
    ev_vehicles  REAL NOT NULL DEFAULT 0,
    source       TEXT NOT NULL DEFAULT 'sample',
    PRIMARY KEY (sigungu_code, age_bucket),
    FOREIGN KEY (sigungu_code) REFERENCES regions (sigungu_code)
);

CREATE TABLE IF NOT EXISTS service_centers (
    center_id    TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    center_type  TEXT NOT NULL CHECK (center_type IN ('bluehands', 'hitech')),
    sigungu_code TEXT NOT NULL,
    lat          REAL NOT NULL,
    lon          REAL NOT NULL,
    bays         INTEGER NOT NULL,
    source       TEXT NOT NULL DEFAULT 'sample',
    FOREIGN KEY (sigungu_code) REFERENCES regions (sigungu_code)
);

-- 거점별 VOC · 재입고 실적 (VOC 시스템 · 정비 이력 집계). 없으면 품질 진단이 부하 축만 쓴다.
CREATE TABLE IF NOT EXISTS center_voc (
    center_id        TEXT PRIMARY KEY,
    period           TEXT NOT NULL,
    voc_per_1k_jobs  REAL NOT NULL,
    comeback_rate    REAL NOT NULL,
    source           TEXT NOT NULL DEFAULT 'sample',
    FOREIGN KEY (center_id) REFERENCES service_centers (center_id)
);

CREATE TABLE IF NOT EXISTS demand (
    sigungu_code    TEXT PRIMARY KEY,
    annual_visits   REAL NOT NULL,
    ice_visits      REAL NOT NULL,
    ev_visits       REAL NOT NULL,
    hitech_visits   REAL NOT NULL,
    warranty_visits REAL NOT NULL DEFAULT 0,
    paid_visits     REAL NOT NULL DEFAULT 0,
    computed_at     TEXT NOT NULL,
    FOREIGN KEY (sigungu_code) REFERENCES regions (sigungu_code)
);

CREATE TABLE IF NOT EXISTS coverage (
    sigungu_code       TEXT PRIMARY KEY,
    nearest_minutes    REAL NOT NULL,
    reachable_centers  INTEGER NOT NULL,
    reachable_capacity REAL NOT NULL,
    covered_visits     REAL NOT NULL,
    uncovered_visits   REAL NOT NULL,
    overflow_visits    REAL NOT NULL,
    avg_utilization    REAL NOT NULL,
    computed_at        TEXT NOT NULL,
    FOREIGN KEY (sigungu_code) REFERENCES regions (sigungu_code)
);

CREATE TABLE IF NOT EXISTS gap_scores (
    sigungu_code     TEXT PRIMARY KEY,
    unmet_visits     REAL NOT NULL,
    access_penalty   REAL NOT NULL,
    load_penalty     REAL NOT NULL,
    gap_score        REAL NOT NULL,
    gap_rank         INTEGER NOT NULL,
    computed_at      TEXT NOT NULL,
    FOREIGN KEY (sigungu_code) REFERENCES regions (sigungu_code)
);

CREATE TABLE IF NOT EXISTS center_quality (
    center_id        TEXT PRIMARY KEY,
    sigungu_code     TEXT NOT NULL,
    utilization      REAL NOT NULL,
    est_wait_days    REAL NOT NULL,
    voc_per_1k_jobs  REAL,
    comeback_rate    REAL,
    quality_risk     REAL NOT NULL,
    quality_grade    TEXT NOT NULL CHECK (quality_grade IN ('A', 'B', 'C', 'D')),
    issue_type       TEXT NOT NULL,
    computed_at      TEXT NOT NULL,
    FOREIGN KEY (center_id) REFERENCES service_centers (center_id),
    FOREIGN KEY (sigungu_code) REFERENCES regions (sigungu_code)
);

CREATE TABLE IF NOT EXISTS siting_runs (
    run_id          TEXT PRIMARY KEY,
    n_new_sites     INTEGER NOT NULL,
    candidate_count INTEGER NOT NULL,
    objective_value REAL NOT NULL,
    solver_status   TEXT NOT NULL,
    baseline_unmet  REAL NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS siting_results (
    run_id          TEXT NOT NULL,
    priority        INTEGER NOT NULL,
    sigungu_code    TEXT NOT NULL,
    lat             REAL NOT NULL,
    lon             REAL NOT NULL,
    captured_visits REAL NOT NULL,
    PRIMARY KEY (run_id, sigungu_code),
    FOREIGN KEY (run_id) REFERENCES siting_runs (run_id),
    FOREIGN KEY (sigungu_code) REFERENCES regions (sigungu_code)
);

CREATE TABLE IF NOT EXISTS plan_runs (
    run_id            TEXT PRIMARY KEY,
    budget            REAL NOT NULL,
    spent             REAL NOT NULL,
    n_new_sites       INTEGER NOT NULL,
    n_expansions      INTEGER NOT NULL,
    added_bays        INTEGER NOT NULL,
    objective_value   REAL NOT NULL,
    new_only_value    REAL NOT NULL,
    expand_only_value REAL NOT NULL,
    baseline_unmet    REAL NOT NULL,
    solver_status     TEXT NOT NULL,
    created_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS plan_actions (
    run_id          TEXT NOT NULL,
    priority        INTEGER NOT NULL,
    action          TEXT NOT NULL CHECK (action IN ('new', 'expand')),
    target_id       TEXT NOT NULL,
    sigungu_code    TEXT NOT NULL,
    added_bays      INTEGER NOT NULL,
    cost            REAL NOT NULL,
    captured_visits REAL NOT NULL,
    PRIMARY KEY (run_id, action, target_id),
    FOREIGN KEY (run_id) REFERENCES plan_runs (run_id),
    FOREIGN KEY (sigungu_code) REFERENCES regions (sigungu_code)
);

CREATE TABLE IF NOT EXISTS reports (
    report_id    TEXT PRIMARY KEY,
    run_id       TEXT,
    scope        TEXT NOT NULL,
    generator    TEXT NOT NULL,
    content_md   TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_centers_region ON service_centers (sigungu_code);
CREATE INDEX IF NOT EXISTS idx_gap_rank ON gap_scores (gap_rank);
CREATE INDEX IF NOT EXISTS idx_siting_run ON siting_results (run_id, priority);
CREATE INDEX IF NOT EXISTS idx_plan_run ON plan_actions (run_id, priority);
