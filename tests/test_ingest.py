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


def test_unknown_source_rejected(settings):
    with pytest.raises(ValueError):
        collect(settings, source="nope")
