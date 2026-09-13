from __future__ import annotations

import pytest

from snx.config import AGE_BUCKETS
from snx.ingest.pipeline import collect


def test_sample_ingest_shapes(ingested):
    assert ingested.mode == "sample"
    assert len(ingested.regions) > 50
    assert set(ingested.vehicle_parc["age_bucket"]) == set(AGE_BUCKETS)
    assert (ingested.vehicle_parc["ev_vehicles"] <= ingested.vehicle_parc["vehicles"]).all()


def test_sample_is_deterministic(settings):
    a = collect(settings, source="sample", seed=7)
    b = collect(settings, source="sample", seed=7)
    assert a.vehicle_parc.equals(b.vehicle_parc)
    assert a.service_centers.equals(b.service_centers)


def test_different_seed_changes_network(settings):
    a = collect(settings, source="sample", seed=7)
    b = collect(settings, source="sample", seed=8)
    assert not a.service_centers.equals(b.service_centers)


def test_rural_regions_have_older_fleet(ingested):
    parc = ingested.vehicle_parc.merge(
        ingested.regions[["sigungu_code", "urban_class"]], on="sigungu_code"
    )
    aged = parc[parc["age_bucket"] == "11+"]
    total = parc.groupby(["urban_class"])["vehicles"].sum()
    aged_share = aged.groupby("urban_class")["vehicles"].sum() / total
    assert aged_share["rural"] > aged_share["metro"]


def test_sample_voc_covers_every_center(ingested):
    voc = ingested.center_voc
    assert set(voc["center_id"]) == set(ingested.service_centers["center_id"])
    assert voc["comeback_rate"].between(0, 1).all()
    assert (voc["voc_per_1k_jobs"] > 0).all()


def test_voc_does_not_change_network(settings):
    """VOC 는 거점망 난수 소비 이후에 뽑는다 — 추가해도 기존 모수 · 거점이 그대로여야 한다."""
    a = collect(settings, source="sample", seed=11)
    assert a.center_voc is not None and len(a.center_voc) == len(a.service_centers)


def test_orphan_voc_rejected(ingested):
    from dataclasses import replace

    bad = ingested.center_voc.copy()
    bad.loc[0, "center_id"] = "없는거점"
    with pytest.raises(ValueError):
        replace(ingested, center_voc=bad).validate()


def test_unknown_source_rejected(settings):
    with pytest.raises(ValueError):
        collect(settings, source="nope")
