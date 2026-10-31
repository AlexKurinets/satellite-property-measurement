"""HTTP API the quoting app calls: measure a property, show the overlay, record an estimator's correction."""

from __future__ import annotations

import os
import secrets
from datetime import date

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from . import __version__, registry, service
from .db import connect, property_input
from .evaluate import classify
from .measure import measure
from .overlay import render

app = FastAPI(title="Property measurement API", version=__version__)


def conn():
    c = connect()
    try:
        yield c
    finally:
        c.close()


def require_token(authorization: str = Header(default="")):
    expected = os.environ.get("PROPMEASURE_API_TOKEN", "local-dev-token")
    if not secrets.compare_digest(authorization, f"Bearer {expected}"):
        raise HTTPException(401, "missing or wrong bearer token")


class MeasureRequest(BaseModel):
    property_id: str
    model_version: str | None = Field(None, description="defaults to the production model and its pipeline")
    measured_on: date | None = None


class CorrectionRequest(BaseModel):
    feature: str
    corrected_value: float = Field(ge=0)
    corrected_by: str = Field(min_length=1, max_length=80)
    note: str | None = Field(None, max_length=500)


@app.get("/healthz")
def healthz(c=Depends(conn)):
    prod = registry.production(c)
    return {"ok": True, "production_model": prod["version"] if prod else None}


@app.get("/v1/properties")
def properties(split: str = "live", market: str | None = None, limit: int = Query(50, le=500), c=Depends(conn)):
    sql = "SELECT property_id, market, split FROM pm.properties WHERE split = %s"
    args: list = [split]
    if market:
        sql += " AND market = %s"
        args.append(market)
    return c.execute(sql + " ORDER BY property_id LIMIT %s", (*args, limit)).fetchall()


@app.post("/v1/measurements", status_code=201, dependencies=[Depends(require_token)])
def create_measurement(req: MeasureRequest, c=Depends(conn)):
    try:
        return service.measure_property(c, req.property_id, req.model_version, req.measured_on)
    except service.NotFound as e:
        raise HTTPException(404, str(e)) from e


@app.get("/v1/measurements/{measurement_id}")
def get_measurement(measurement_id: int, c=Depends(conn)):
    try:
        return service.get_measurement(c, measurement_id)
    except service.NotFound as e:
        raise HTTPException(404, str(e)) from e


@app.get("/v1/measurements/{measurement_id}/overlay.png")
def overlay(measurement_id: int, c=Depends(conn)):
    row = c.execute("SELECT property_id, model_version, pipeline, measured_on FROM pm.measurements WHERE id = %s",
                    (measurement_id,)).fetchone()
    if row is None:
        raise HTTPException(404, f"measurement {measurement_id} not found")
    inp = property_input(c, row["property_id"])
    classes, proba = classify(registry.load(c, row["model_version"]), inp.image)
    m = measure(inp, classes, proba, pipeline=row["pipeline"], measured_on=row["measured_on"])
    return Response(render(inp.image, classes, m), media_type="image/png")


@app.post("/v1/measurements/{measurement_id}/corrections", status_code=201, dependencies=[Depends(require_token)])
def correct(measurement_id: int, req: CorrectionRequest, c=Depends(conn)):
    try:
        return service.add_correction(c, measurement_id, req.feature, req.corrected_value, req.corrected_by, req.note)
    except service.NotFound as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.get("/v1/models")
def models(c=Depends(conn)):
    return registry.models(c)


@app.get("/v1/benchmark")
def benchmark(c=Depends(conn)):
    return service.status(c)["benchmark"]


@app.get("/v1/label-queue")
def label_queue(limit: int = Query(20, le=200), c=Depends(conn)):
    return service.label_queue(c, limit)
