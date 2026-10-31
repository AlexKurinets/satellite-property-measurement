"""Postgres-backed tests: seed, registry, release gate, API and the estimator loop."""

from datetime import date

import pytest
import yaml
from fastapi.testclient import TestClient

from propmeasure import cli, registry, service
from propmeasure.api import app
from propmeasure.db import bench_items, jsonb, training_arrays

from .conftest import SMALL

AUTH = {"Authorization": "Bearer local-dev-token"}


def test_seed_splits_and_labels(conn):
    counts = {r["split"]: r["n"] for r in conn.execute("SELECT split, count(*) AS n FROM pm.properties GROUP BY 1")}
    assert counts == SMALL
    images, labels = training_arrays(conn, "train")
    assert images.shape == (9, 192, 192, 3) and labels.shape == (9, 192, 192)
    assert conn.execute("SELECT count(*) AS n FROM pm.labels WHERE property_id LIKE 'live-%'").fetchone()["n"] == 0
    assert {it.market for it in bench_items(conn, "test")} == {"coastal-south", "mid-atlantic", "mountain-west"}


def test_registry_holds_production_and_candidate(conn):
    rows = {m["version"]: m for m in registry.models(conn)}
    assert rows[cli.BASE_VERSION]["status"] == "production" and rows[cli.BASE_VERSION]["pipeline"] == "v1"
    assert rows[cli.FT_VERSION]["status"] == "candidate" and rows[cli.FT_VERSION]["parent_version"] == cli.BASE_VERSION


def test_promote_refuses_without_a_gated_run_and_on_a_failed_gate(conn, tmp_path):
    res = registry.promote(conn, cli.FT_VERSION)
    assert not res["promoted"] and "no gated benchmark run" in res["reason"]
    strict = yaml.safe_load(registry.GATE_CONFIG.read_text())
    strict["max_mdape"] = {f: 0.01 for f in strict["max_mdape"]}
    path = tmp_path / "strict.yaml"
    path.write_text(yaml.safe_dump(strict))
    out = registry.run_benchmark(conn, cli.FT_VERSION, None, cli.REFERENCE_DATE, gate_path=str(path))
    assert out["gate"] is not None and not out["gate"]["passed"]
    res = registry.promote(conn, cli.FT_VERSION)
    assert not res["promoted"] and res["reason"].startswith("release gate failed")


def test_promote_after_a_passing_gate(conn, tmp_path):
    loose = yaml.safe_load(registry.GATE_CONFIG.read_text())
    loose["max_mdape"] = {f: 1000 for f in loose["max_mdape"]}
    loose["max_abs_bias"] = 1000
    path = tmp_path / "loose.yaml"
    path.write_text(yaml.safe_dump(loose))
    # 30-step models are noise, so give production a benchmark run any candidate beats; this tests the mechanics
    prod = registry.latest_run(conn, cli.BASE_VERSION, "v1")["report"]
    for scope in [prod["overall"], *prod["by"]["market"].values()]:
        for s in scope.values():
            s["mdape"] = 999.0
    conn.execute("INSERT INTO pm.eval_runs (model_version, pipeline, benchmark, report) VALUES (%s, 'v1', 'test', %s)",
                 (cli.BASE_VERSION, jsonb(prod)))
    out = registry.run_benchmark(conn, cli.FT_VERSION, None, cli.REFERENCE_DATE, gate_path=str(path))
    assert out["gate"]["passed"], out["gate"]
    assert registry.promote(conn, cli.FT_VERSION)["promoted"]
    assert registry.production(conn)["version"] == cli.FT_VERSION
    assert {m["version"]: m["status"] for m in registry.models(conn)}[cli.BASE_VERSION] == "retired"


def test_oracle_run_is_stored_without_a_model(conn):
    out = registry.run_benchmark(conn, None, "v2", cli.REFERENCE_DATE)
    assert out["report"]["oracle"] and out["gate"] is None
    assert out["report"]["overall"]["lawn_sqft"]["n"] == SMALL["test"]


@pytest.fixture()
def client(seeded):
    return TestClient(app)


def test_api_measure_overlay_and_correct(client, conn):
    assert client.get("/healthz").json()["ok"]
    assert client.post("/v1/measurements", json={"property_id": "live-cs-0000"}).status_code == 401
    r = client.post("/v1/measurements", json={"property_id": "live-cs-0000", "measured_on": "2026-10-30"},
                    headers=AUTH)
    assert r.status_code == 201, r.text
    m = r.json()
    for k in ("lawn_sqft", "lawn_under_canopy_sqft", "frontage_lawn_sqft", "bed_sqft", "pavement_sqft", "roof_sqft",
              "roof_perimeter_ft"):
        assert k in m["result"] or m["pipeline"] == "v1"
    png = client.get(f"/v1/measurements/{m['id']}/overlay.png")
    assert png.status_code == 200 and png.content[:8] == b"\x89PNG\r\n\x1a\n"
    c = client.post(f"/v1/measurements/{m['id']}/corrections", headers=AUTH,
                    json={"feature": "lawn_sqft", "corrected_value": 4200, "corrected_by": "estimator-test"})
    assert c.status_code == 201, c.text
    assert c.json()["measured_value"] == m["result"]["lawn_sqft"]
    bad = client.post(f"/v1/measurements/{m['id']}/corrections", headers=AUTH,
                      json={"feature": "garage_sqft", "corrected_value": 1, "corrected_by": "x"})
    assert bad.status_code == 422
    got = client.get(f"/v1/measurements/{m['id']}").json()
    assert got["corrections"][0]["corrected_value"] == 4200
    assert client.post("/v1/measurements", json={"property_id": "nope"}, headers=AUTH).status_code == 404


def test_estimator_simulation_feeds_edit_rates_and_label_queue(conn):
    service.simulate_estimators(conn, registry.production(conn)["version"], date(2026, 10, 16), log=lambda m: None)
    rates = service.edit_rates(conn)
    assert sum(r["measurements"] for r in rates) >= SMALL["live"]
    queue = service.label_queue(conn, 5)
    assert len(queue) <= 5
    assert all(q["property_id"].startswith("live-") for q in queue)
    markets = [q["market"] for q in queue]
    # taken in turn from each market, not all from the worst one
    assert len(set(markets[:3])) == min(3, len(set(markets)))
