"""Model registry, benchmark runs and the release gate.

A model version is stored with the pipeline it ships with, because a benchmark number only means something for a
(model, pipeline) pair. Promotion goes through the gate: the candidate's latest benchmark run must pass against the
current production pair, or promote refuses.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import yaml

from . import evaluate
from .db import bench_items, jsonb
from .model import UNet, from_bytes, to_bytes

GATE_CONFIG = Path(__file__).resolve().parents[2] / "config" / "release_gate.yaml"
BENCHMARK = "test"
_cache: dict[str, UNet] = {}


def gate_config(path: str | Path | None = None) -> dict:
    p = Path(path or os.environ.get("PROPMEASURE_GATE_CONFIG") or GATE_CONFIG)
    return yaml.safe_load(p.read_text())


def register(conn, version: str, model: UNet, *, parent: str | None, label_set: str, pipeline: str,
             train_config: dict, history: list, status: str = "candidate") -> None:
    conn.execute(
        "INSERT INTO pm.models (version, parent_version, label_set, pipeline, weights, train_config, history, status, "
        "promoted_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CASE WHEN %s = 'production' THEN now() END)",
        (version, parent, label_set, pipeline, to_bytes(model), jsonb(train_config), jsonb(history), status, status))


def load(conn, version: str) -> UNet:
    if version not in _cache:
        row = conn.execute("SELECT weights FROM pm.models WHERE version = %s", (version,)).fetchone()
        if row is None:
            raise KeyError(f"unknown model version {version}")
        _cache[version] = from_bytes(bytes(row["weights"]))
    return _cache[version]


def model_row(conn, version: str) -> dict | None:
    return conn.execute("SELECT version, parent_version, label_set, pipeline, status, train_config, history, "
                        "created_at, promoted_at FROM pm.models WHERE version = %s", (version,)).fetchone()


def production(conn) -> dict | None:
    return conn.execute("SELECT version, pipeline FROM pm.models WHERE status = 'production'").fetchone()


def models(conn) -> list[dict]:
    return conn.execute("SELECT version, parent_version, label_set, pipeline, status, created_at, promoted_at "
                        "FROM pm.models ORDER BY created_at").fetchall()


def latest_run(conn, version: str | None, pipeline: str, benchmark: str = BENCHMARK) -> dict | None:
    if version is None:
        return conn.execute("SELECT * FROM pm.eval_runs WHERE model_version IS NULL AND pipeline = %s "
                            "AND benchmark = %s ORDER BY id DESC LIMIT 1", (pipeline, benchmark)).fetchone()
    return conn.execute("SELECT * FROM pm.eval_runs WHERE model_version = %s AND pipeline = %s AND benchmark = %s "
                        "ORDER BY id DESC LIMIT 1", (version, pipeline, benchmark)).fetchone()


def run_benchmark(conn, version: str | None, pipeline: str | None, measured_on: date,
                  benchmark: str = BENCHMARK, gate_path: str | None = None) -> dict:
    """Score a model (or the traced masks when version is None) and store the run; candidates are gated."""
    items = bench_items(conn, benchmark)
    if not items:
        raise RuntimeError(f"no traced properties in split {benchmark!r}")
    gate = None
    if version is None:
        report = evaluate.run(items, None, pipeline or "v2", measured_on)
    else:
        row = model_row(conn, version)
        if row is None:
            raise KeyError(f"unknown model version {version}")
        pipeline = pipeline or row["pipeline"]
        report = evaluate.run(items, load(conn, version), pipeline, measured_on)
        prod = production(conn)
        if row["status"] == "candidate":
            prod_run = latest_run(conn, prod["version"], prod["pipeline"], benchmark) if prod else None
            gate = evaluate.gate(report, prod_run["report"] if prod_run else None, gate_config(gate_path))
    rid = conn.execute("INSERT INTO pm.eval_runs (model_version, pipeline, benchmark, report, gate) "
                       "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                       (version, pipeline, benchmark, jsonb(report), jsonb(gate) if gate else None)).fetchone()["id"]
    return {"id": rid, "report": report, "gate": gate}


def promote(conn, version: str) -> dict:
    row = model_row(conn, version)
    if row is None:
        raise KeyError(f"unknown model version {version}")
    if row["status"] == "production":
        return {"promoted": False, "reason": f"{version} is already in production"}
    run = latest_run(conn, version, row["pipeline"])
    if run is None or run["gate"] is None:
        return {"promoted": False, "reason": f"no gated benchmark run for {version}; run `propmeasure evaluate` first"}
    if not run["gate"]["passed"]:
        failed = [c["check"] for c in run["gate"]["checks"] if not c["passed"]]
        return {"promoted": False, "reason": "release gate failed: " + "; ".join(failed)}
    with conn.transaction():
        conn.execute("UPDATE pm.models SET status = 'retired' WHERE status = 'production'")
        conn.execute("UPDATE pm.models SET status = 'production', promoted_at = now() WHERE version = %s", (version,))
    return {"promoted": True, "version": version, "eval_run": run["id"]}
