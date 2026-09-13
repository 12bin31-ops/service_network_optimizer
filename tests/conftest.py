from __future__ import annotations

import pytest

from snx.config import load_settings
from snx.ingest.pipeline import ingest


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    """임시 DB 를 쓰는 설정 — 실제 data/processed 를 건드리지 않는다."""
    s = load_settings().model_copy(deep=True)
    s.paths.db = str(tmp_path_factory.mktemp("db") / "test.db")
    s.paths.outputs_dir = str(tmp_path_factory.mktemp("out"))
    return s


@pytest.fixture(scope="session")
def ingested(settings):
    return ingest(settings, source="sample", seed=42)


@pytest.fixture(scope="session")
def pipeline_run(settings, ingested):
    from snx.coverage.gap import run_coverage_stage
    from snx.demand.model import run_demand_stage
    from snx.optimize.siting import run_siting_stage

    demand = run_demand_stage(settings)
    coverage, gaps = run_coverage_stage(settings)
    solution = run_siting_stage(settings, n_new_sites=10)
    return {
        "demand": demand,
        "coverage": coverage,
        "gaps": gaps,
        "solution": solution,
    }
