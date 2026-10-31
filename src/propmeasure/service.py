"""Measurements, estimator corrections, the labelling queue and status, shared by the API and the CLI."""

from __future__ import annotations

from datetime import date

from . import registry
from .db import jsonb, property_input
from .evaluate import classify
from .measure import measure

CORRECTABLE = ("lawn_sqft", "bed_sqft", "pavement_sqft", "roof_sqft", "roof_perimeter_ft")
EDIT_FEATURES = ("lawn_sqft", "roof_sqft")  # the figures that set the price of the main services
EDIT_THRESHOLD = 0.10


class NotFound(LookupError):
    pass


def measure_property(conn, property_id: str, version: str | None = None, measured_on: date | None = None) -> dict:
    inp = property_input(conn, property_id)
    if inp is None:
        raise NotFound(f"property {property_id} not found")
    if version is None:
        prod = registry.production(conn)
        if prod is None:
            raise NotFound("no model in production yet")
        version, pipeline = prod["version"], prod["pipeline"]
    else:
        row = registry.model_row(conn, version)
        if row is None:
            raise NotFound(f"model {version} not found")
        pipeline = row["pipeline"]
    classes, proba = classify(registry.load(conn, version), inp.image)
    measured_on = measured_on or date.today()
    m = measure(inp, classes, proba, pipeline=pipeline, measured_on=measured_on)
    row = conn.execute(
        "INSERT INTO pm.measurements (property_id, model_version, pipeline, result, review_reasons, measured_on) "
        "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id, created_at",
        (property_id, version, pipeline, jsonb(m.values), jsonb(m.review_reasons), measured_on)).fetchone()
    return {"id": row["id"], "property_id": property_id, "model_version": version, "pipeline": pipeline,
            "measured_on": measured_on.isoformat(), "imagery_captured_on": inp.captured_on.isoformat(),
            "result": m.values, "review_reasons": m.review_reasons}


def get_measurement(conn, measurement_id: int) -> dict:
    row = conn.execute("SELECT * FROM pm.measurements WHERE id = %s", (measurement_id,)).fetchone()
    if row is None:
        raise NotFound(f"measurement {measurement_id} not found")
    row["corrections"] = conn.execute(
        "SELECT feature, measured_value, corrected_value, corrected_by, note, created_at FROM pm.corrections "
        "WHERE measurement_id = %s ORDER BY id", (measurement_id,)).fetchall()
    return row


def add_correction(conn, measurement_id: int, feature: str, corrected_value: float, corrected_by: str,
                   note: str | None = None) -> dict:
    if feature not in CORRECTABLE:
        raise ValueError(f"feature must be one of {', '.join(CORRECTABLE)}")
    m = conn.execute("SELECT result FROM pm.measurements WHERE id = %s", (measurement_id,)).fetchone()
    if m is None:
        raise NotFound(f"measurement {measurement_id} not found")
    measured = float(m["result"][feature])
    return conn.execute(
        "INSERT INTO pm.corrections (measurement_id, feature, measured_value, corrected_value, corrected_by, note) "
        "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (measurement_id, feature) DO UPDATE SET "
        "corrected_value = EXCLUDED.corrected_value, corrected_by = EXCLUDED.corrected_by, note = EXCLUDED.note "
        "RETURNING id, measurement_id, feature, measured_value, corrected_value, corrected_by, note",
        (measurement_id, feature, measured, corrected_value, corrected_by, note)).fetchone()


def label_queue(conn, limit: int = 20) -> list[dict]:
    """Unlabelled properties whose measurements estimators corrected the most, taken in turn from each market.

    These go to the annotators for the next fine-tune round: they are where the current model is wrong, and taking
    them market by market keeps one market's problems from filling the whole batch.
    """
    rows = conn.execute("""
        SELECT p.property_id, p.market, max(abs(c.corrected_value - c.measured_value)
               / greatest(c.corrected_value, 1)) AS largest_change,
               array_agg(DISTINCT c.feature ORDER BY c.feature) AS features, count(DISTINCT m.id) AS measurements
        FROM pm.corrections c
        JOIN pm.measurements m ON m.id = c.measurement_id
        JOIN pm.properties p ON p.property_id = m.property_id
        LEFT JOIN pm.labels l ON l.property_id = p.property_id
        WHERE l.property_id IS NULL AND m.model_version = (SELECT version FROM pm.models WHERE status = 'production')
        GROUP BY p.property_id, p.market
        ORDER BY largest_change DESC, p.property_id""").fetchall()
    by_market: dict[str, list] = {}
    for r in rows:
        r["largest_change"] = round(float(r["largest_change"]), 3)
        by_market.setdefault(r["market"], []).append(r)
    out = []
    while len(out) < limit and any(by_market.values()):
        for market in sorted(by_market):
            if by_market[market] and len(out) < limit:
                out.append(by_market[market].pop(0))
    return out


def edit_rates(conn) -> list[dict]:
    return conn.execute("""
        SELECT m.model_version, m.pipeline, count(*) AS measurements,
               count(*) FILTER (WHERE EXISTS (SELECT 1 FROM pm.corrections c WHERE c.measurement_id = m.id))
                   AS corrected,
               round(100.0 * count(*) FILTER (WHERE EXISTS (SELECT 1 FROM pm.corrections c
                   WHERE c.measurement_id = m.id)) / count(*), 1) AS corrected_pct,
               round(100.0 * count(*) FILTER (WHERE jsonb_array_length(m.review_reasons) > 0) / count(*), 1)
                   AS flagged_pct
        FROM pm.measurements m
        GROUP BY m.model_version, m.pipeline ORDER BY min(m.created_at)""").fetchall()


def simulate_estimators(conn, version: str, measured_on: date, log=print) -> dict:
    """Measure every live property with a model and have the estimator fix any checked figure off by over 10%.

    Live properties carry synthetic truth only so the simulation knows what the estimator would have typed. Where
    the quoting screen asks for a pitch, the estimator types the real one; that is an input, not a correction, so
    only a wrong roof outline counts against the model.
    """
    rows = conn.execute("SELECT p.property_id, p.market, g.* FROM pm.properties p JOIN pm.ground_truth g "
                        "USING (property_id) WHERE p.split = 'live' ORDER BY p.property_id").fetchall()
    corrected = 0
    for i, r in enumerate(rows):
        m = measure_property(conn, r["property_id"], version, measured_on)
        edited = False
        for f in EDIT_FEATURES:
            truth = float(r[f])
            shown = m["result"][f]
            if f == "roof_sqft" and m["result"].get("needs_pitch_from_estimator"):
                shown = m["result"]["roof_plan_sqft"] * r["roof_sqft"] / r["roof_plan_sqft"]
            if truth > 0 and abs(shown - truth) / truth > EDIT_THRESHOLD:
                add_correction(conn, m["id"], f, round(truth, -1), f"estimator-{r['market']}-{i % 4 + 1}",
                               "re-measured on screen before quoting")
                edited = True
        corrected += edited
    log(f"simulated estimators: {version} measured {len(rows)} live properties, {corrected} corrected")
    return {"measured": len(rows), "corrected": corrected}


def status(conn) -> dict:
    splits = conn.execute("SELECT split, count(*) AS n FROM pm.properties GROUP BY split ORDER BY split").fetchall()
    out = {"properties": {r["split"]: r["n"] for r in splits}, "models": registry.models(conn), "benchmark": []}
    # latest run of every (model, pipeline) pair: oracle runs first, then models in the order they were registered
    order = {m["version"]: i for i, m in enumerate(out["models"])}
    runs = conn.execute("SELECT DISTINCT ON (model_version, pipeline) * FROM pm.eval_runs WHERE benchmark = %s "
                        "ORDER BY model_version, pipeline, id DESC", (registry.BENCHMARK,)).fetchall()
    runs.sort(key=lambda r: (-1 if r["model_version"] is None else order.get(r["model_version"], 99), r["pipeline"]))
    for run in runs:
        out["benchmark"].append(_summary(run["model_version"] or "traced masks (oracle)", run["pipeline"], run))
    out["edit_rates"] = edit_rates(conn)
    out["corrections"] = conn.execute("SELECT count(*) AS n FROM pm.corrections").fetchone()["n"]
    return out


def _summary(name: str, pipeline: str, run: dict) -> dict:
    o = run["report"]["overall"]
    s = {"model": name, "pipeline": pipeline, "eval_run": run["id"], "properties": run["report"]["properties"]}
    for f in ("lawn_sqft", "bed_sqft", "pavement_sqft", "roof_sqft", "roof_perimeter_ft"):
        if f in o:
            s[f] = {"mdape": o[f]["mdape"], "bias": o[f]["bias"]}
    s["miou"] = run["report"].get("miou")
    s["review_share"] = run["report"].get("review_share")
    if run["gate"] is not None:
        s["gate_passed"] = run["gate"]["passed"]
    return s
