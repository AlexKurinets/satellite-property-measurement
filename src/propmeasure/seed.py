"""Seed synthetic properties: base-domain training data, the client's labelled sets, and live properties.

Splits:
- base: public land-cover look, labelled with the base classes; trains the original model.
- train / val: client markets, traced with the client classes; the fine-tune set and its checkpoint selection.
- test: the held-out benchmark, traced, never trained on.
- live: unlabelled properties the API measures and the simulated estimators correct.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

from . import geo
from .db import jsonb, png_bytes
from .synth import MARKETS, Scene, make_scene

MARKET_CODES = {"coastal-south": "cs", "mid-atlantic": "ma", "mountain-west": "mw"}
SPLIT_SEEDS = {"base": 10_000, "train": 20_000, "val": 30_000, "test": 40_000, "live": 50_000}
DEFAULT_SIZES = {"base": 400, "train": 240, "val": 60, "test": 150, "live": 90}


def plan(sizes: dict[str, int]) -> list[tuple[str, str, int, str, str | None]]:
    """(property_id, split, seed, domain, market) for every property to generate."""
    markets = list(MARKETS)
    jobs = []
    for split, n in sizes.items():
        for i in range(n):
            seed = SPLIT_SEEDS[split] + i
            if split == "base":
                jobs.append((f"base-{i:04d}", split, seed, "base", None))
            else:
                mk = markets[i % len(markets)]
                jobs.append((f"{split}-{MARKET_CODES[mk]}-{i:04d}", split, seed, "target", mk))
    return jobs


def _make(job) -> tuple[tuple, Scene]:
    pid, split, seed, domain, market = job
    return job, make_scene(seed, domain, market)


def _rows(job, sc: Scene):
    pid, split, _, _, _ = job
    ring = geo.capture_px_to_ring(sc.parcel_delivered, sc.origin[0], sc.origin[1], sc.zoom)
    prop = (pid, sc.market, split, sc.lat, sc.lon)
    cap = (pid, sc.captured_on, sc.season, sc.zoom, sc.origin[0], sc.origin[1], png_bytes(sc.image))
    parcel = (pid, jsonb(ring))
    pitch = (pid, sc.lidar_rise) if sc.lidar_rise is not None else None
    label = None if split == "live" else (pid, "base" if split == "base" else "client", png_bytes(sc.label))
    t = sc.truth
    truth = None if split == "base" else (
        pid, t["lawn_sqft"], t["bed_sqft"], t["pavement_sqft"], t["roof_sqft"], t["roof_plan_sqft"],
        t["roof_perimeter_ft"], "synthetic" if split == "live" else "traced")
    return prop, cap, parcel, pitch, label, truth


def seed(conn, sizes: dict[str, int] | None = None, workers: int | None = None, log=print) -> int:
    sizes = sizes or DEFAULT_SIZES
    existing = {r["property_id"] for r in conn.execute("SELECT property_id FROM pm.properties")}
    jobs = [j for j in plan(sizes) if j[0] not in existing]
    if not jobs:
        log("seed: nothing to do")
        return 0
    workers = workers or min(8, os.cpu_count() or 2)
    log(f"seed: generating {len(jobs)} properties on {workers} processes")
    done = 0
    with ProcessPoolExecutor(workers) as ex:
        batch = []
        for job, sc in ex.map(_make, jobs, chunksize=4):
            batch.append(_rows(job, sc))
            if len(batch) == 50:
                _insert(conn, batch)
                done += len(batch)
                batch = []
                log(f"seed: {done}/{len(jobs)}")
        if batch:
            _insert(conn, batch)
            done += len(batch)
    log(f"seed: {done} properties stored")
    return done


def _insert(conn, batch) -> None:
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany("INSERT INTO pm.properties (property_id, market, split, lat, lon) VALUES (%s, %s, %s, %s, %s)",
                        [b[0] for b in batch])
        cur.executemany("INSERT INTO pm.captures (property_id, captured_on, season, zoom, origin_x, origin_y, png) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s)", [b[1] for b in batch])
        cur.executemany("INSERT INTO pm.parcels (property_id, ring) VALUES (%s, %s)", [b[2] for b in batch])
        pitches = [b[3] for b in batch if b[3]]
        if pitches:
            cur.executemany("INSERT INTO pm.pitch_observations (property_id, rise_in_12) VALUES (%s, %s)", pitches)
        labels = [b[4] for b in batch if b[4]]
        if labels:
            cur.executemany("INSERT INTO pm.labels (property_id, label_set, mask_png) VALUES (%s, %s, %s)", labels)
        truths = [b[5] for b in batch if b[5]]
        if truths:
            cur.executemany("INSERT INTO pm.ground_truth (property_id, lawn_sqft, bed_sqft, pavement_sqft, roof_sqft, "
                            "roof_plan_sqft, roof_perimeter_ft, method) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                            truths)
