"""Postgres access: schema, captures, parcels, labels and truth.

The imagery and parcel adapters production used (a satellite tile API and a county parcel feed) are replaced here by
tables the seed fills; the rest of the code only sees PropertyInput and BenchItem.
"""

from __future__ import annotations

import io
import os
from importlib import resources

import numpy as np
import psycopg
from PIL import Image
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .evaluate import BenchItem
from .measure import PropertyInput

DEFAULT_URL = "postgresql://propmeasure:propmeasure@localhost:5512/propmeasure"


def url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_URL)


def connect(dsn: str | None = None) -> psycopg.Connection:
    return psycopg.connect(dsn or url(), row_factory=dict_row, autocommit=True)


def migrate(conn: psycopg.Connection) -> None:
    sql = resources.files("propmeasure").joinpath("schema.sql").read_text()
    conn.execute(sql)


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=False, compress_level=6)
    return buf.getvalue()


def png_array(data: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)))


def _input(row: dict) -> PropertyInput:
    return PropertyInput(
        property_id=row["property_id"], image=png_array(row["png"]), lat=row["lat"], zoom=row["zoom"],
        origin=(row["origin_x"], row["origin_y"]), captured_on=row["captured_on"], parcel_ring=row["ring"],
        lidar_rise=row["rise_in_12"],
    )


_INPUT_SQL = """
SELECT p.property_id, p.market, p.lat, c.zoom, c.origin_x, c.origin_y, c.captured_on, c.season, c.png, pa.ring,
       po.rise_in_12
FROM pm.properties p
JOIN pm.captures c USING (property_id)
JOIN pm.parcels pa USING (property_id)
LEFT JOIN pm.pitch_observations po USING (property_id)
"""


def property_input(conn, property_id: str) -> PropertyInput | None:
    row = conn.execute(_INPUT_SQL + " WHERE p.property_id = %s", (property_id,)).fetchone()
    return _input(row) if row else None


def bench_items(conn, split: str) -> list[BenchItem]:
    rows = conn.execute(
        _INPUT_SQL.replace("SELECT p.property_id,", "SELECT p.property_id, l.mask_png, g.*,")
        + " JOIN pm.labels l USING (property_id) JOIN pm.ground_truth g USING (property_id)"
        " WHERE p.split = %s ORDER BY p.property_id", (split,)).fetchall()
    keys = ("lawn_sqft", "bed_sqft", "pavement_sqft", "roof_sqft", "roof_plan_sqft", "roof_perimeter_ft")
    return [BenchItem(_input(r), {k: r[k] for k in keys}, png_array(r["mask_png"]), r["market"], r["season"])
            for r in rows]


def training_arrays(conn, split: str) -> tuple[np.ndarray, np.ndarray]:
    rows = conn.execute(
        "SELECT c.png, l.mask_png FROM pm.properties p JOIN pm.captures c USING (property_id) "
        "JOIN pm.labels l USING (property_id) WHERE p.split = %s ORDER BY p.property_id", (split,)).fetchall()
    if not rows:
        raise RuntimeError(f"no labelled properties in split {split!r}; run `propmeasure seed` first")
    return np.stack([png_array(r["png"]) for r in rows]), np.stack([png_array(r["mask_png"]) for r in rows])


def jsonb(value) -> Jsonb:
    return Jsonb(value)
