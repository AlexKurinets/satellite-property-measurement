"""From a class map to the square footage a quote uses.

Two pipelines share the same inputs so the benchmark can score each one on the model's output and on hand-traced
masks (the oracle run):

- v1 is the measurement code the original model shipped with: clip every class to the county parcel as delivered,
  report lawn as visible turf and roof size as plan area, and count roof perimeter along pixel edges.
- v2 snaps the parcel's front line to the back of the sidewalk, adds the mowed strip between sidewalk and curb,
  counts canopy surrounded by lawn as lawn under trees, takes whole roofs by overlap instead of clipping them,
  applies the roof pitch, measures perimeter on a simplified outline, and says when a person should look.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
from scipy import ndimage

from . import geo
from .classes import BED, CANOPY, LAWN, PAVEMENT, ROOF

PIPELINES = ("v1", "v2")
DEFAULT_PITCH = 6.0
SIDEWALK_FRACTION = 0.55  # share of a row across the frontage that must be paving to count as sidewalk
SNAP_SEARCH_M = 3.5
CANOPY_RING_M = 1.5
CANOPY_LAWN_SHARE = 0.5
ROOF_OVERLAP = 0.5
MIN_ROOF_SQFT = 40.0
LOW_CONFIDENCE = 0.85
CANOPY_REVIEW_SHARE = 0.35
STALE_IMAGERY_DAYS = 548
EIGHT = np.ones((3, 3), dtype=bool)


@dataclass
class PropertyInput:
    property_id: str
    image: np.ndarray
    lat: float
    zoom: int
    origin: tuple[float, float]
    captured_on: date
    parcel_ring: list[list[float]]  # lon/lat, as delivered by the parcel source
    lidar_rise: float | None = None

    @property
    def gsd(self) -> float:
        return geo.ground_resolution(self.lat, self.zoom)


@dataclass
class Measurement:
    property_id: str
    pipeline: str
    values: dict
    review_reasons: list[str] = field(default_factory=list)
    parcel_px: np.ndarray | None = None  # delivered
    snapped_px: np.ndarray | None = None
    roof_mask: np.ndarray | None = None
    lawn_mask: np.ndarray | None = None

    def public(self) -> dict:
        return {"property_id": self.property_id, "pipeline": self.pipeline, **self.values,
                "review_reasons": self.review_reasons}


def _frontage_columns(poly: np.ndarray, width: int) -> np.ndarray:
    x_lo, x_hi = poly[:, 0].min(), poly[:, 0].max()
    centers = np.arange(width) + 0.5
    return np.nonzero((centers >= x_lo) & (centers < x_hi))[0]


def find_street(classes: np.ndarray, poly: np.ndarray, gsd: float) -> dict:
    """Locate road edge and sidewalk along the parcel's frontage (the bottom of the capture).

    Returns row indices: road_top (first road row), and sw_top/sw_bottom when a sidewalk band is found within
    SNAP_SEARCH_M of the delivered front line.
    """
    h, w = classes.shape
    cols = _frontage_columns(poly, w)
    if len(cols) == 0:
        return {"road_top": h}
    frac = (classes[:, cols] == PAVEMENT).mean(axis=1)
    y = h - 1
    while y >= 0 and frac[y] >= SIDEWALK_FRACTION:
        y -= 1
    road_top = y + 1
    out = {"road_top": road_top}
    front = poly[:, 1].max()
    reach = SNAP_SEARCH_M / gsd
    min_rows, max_rows = max(2, int(0.8 / gsd)), int(2.6 / gsd) + 1
    # walk up through the strip, then through the sidewalk band
    y = road_top - 1
    while y >= 0 and frac[y] < SIDEWALK_FRACTION and road_top - y <= int(4.5 / gsd):
        y -= 1
    bottom = y
    while y >= 0 and frac[y] >= SIDEWALK_FRACTION:
        y -= 1
    top = y + 1
    rows = bottom - top + 1
    if bottom >= 0 and min_rows <= rows <= max_rows and abs(top - front) <= reach:
        out.update(sw_top=top, sw_bottom=bottom)
    return out


def measure(inp: PropertyInput, classes: np.ndarray, proba: np.ndarray | None = None, pipeline: str = "v2",
            measured_on: date | None = None) -> Measurement:
    if pipeline not in PIPELINES:
        raise ValueError(f"unknown pipeline {pipeline}")
    gsd = inp.gsd
    px_sqft = geo.sqft_per_pixel(inp.lat, inp.zoom)
    shape = classes.shape
    parcel_px = geo.ring_to_capture_px(inp.parcel_ring, inp.origin[0], inp.origin[1], inp.zoom)
    if pipeline == "v1":
        return _measure_v1(inp, classes, parcel_px, px_sqft, gsd)

    reasons: list[str] = []
    street = find_street(classes, parcel_px, gsd)
    snapped = parcel_px.copy()
    shift_m = 0.0
    if "sw_top" in street:
        dy = street["sw_top"] - parcel_px[:, 1].max()
        snapped[:, 1] += dy
        shift_m = float(dy * gsd)
        strip_rows = np.arange(street["sw_bottom"] + 1, street["road_top"])
    else:
        reasons.append("no sidewalk found along the frontage; parcel line used as delivered")
        front = int(round(parcel_px[:, 1].max()))
        strip_rows = np.arange(max(front, 0), street["road_top"])
    parcel = geo.rasterize(snapped, shape)

    cols = _frontage_columns(snapped, shape[1])
    strip = np.zeros(shape, dtype=bool)
    if len(strip_rows) and len(cols):
        strip[np.ix_(strip_rows, cols)] = True
    strip &= ~parcel
    frontage = strip & ((classes == LAWN) | (classes == CANOPY))

    lawn_visible = (classes == LAWN) & parcel
    canopy = (classes == CANOPY) & parcel
    under = np.zeros(shape, dtype=bool)
    lab, n = ndimage.label(canopy, structure=EIGHT)
    ring_px = max(1, int(round(CANOPY_RING_M / gsd)))
    for k in range(1, n + 1):
        comp = lab == k
        ring = ndimage.binary_dilation(comp, iterations=ring_px) & ~comp & (classes != CANOPY)
        if ring.sum() and ((classes == LAWN) & ring).sum() / ring.sum() >= CANOPY_LAWN_SHARE:
            under |= comp

    roofs = classes == ROOF
    rlab, rn = ndimage.label(roofs, structure=EIGHT)
    roof_mask = np.zeros(shape, dtype=bool)
    for k in range(1, rn + 1):
        comp = rlab == k
        size = comp.sum()
        if size * px_sqft < MIN_ROOF_SQFT:
            continue
        if (comp & parcel).sum() / size >= ROOF_OVERLAP:
            roof_mask |= comp
    roof_mask = ndimage.binary_fill_holes(roof_mask)
    if not roof_mask.any():
        reasons.append("no roof found inside the parcel")

    # pitch comes from public lidar where it covers the property; otherwise the quoting screen asks the estimator
    if inp.lidar_rise is not None:
        rise, source = float(inp.lidar_rise), "lidar"
    else:
        rise, source = DEFAULT_PITCH, "default"

    lawn_v = float(lawn_visible.sum() * px_sqft)
    lawn_u = float(under.sum() * px_sqft)
    lawn_f = float(frontage.sum() * px_sqft)
    lawn_total = lawn_v + lawn_u + lawn_f
    roof_plan = float(roof_mask.sum() * px_sqft)
    values = {
        "lawn_sqft": round(lawn_total, 1),
        "lawn_visible_sqft": round(lawn_v, 1),
        "lawn_under_canopy_sqft": round(lawn_u, 1),
        "frontage_lawn_sqft": round(lawn_f, 1),
        "bed_sqft": round(float(((classes == BED) & parcel).sum() * px_sqft), 1),
        "pavement_sqft": round(float(((classes == PAVEMENT) & parcel).sum() * px_sqft), 1),
        "roof_plan_sqft": round(roof_plan, 1),
        "roof_sqft": round(roof_plan * geo.pitch_factor(rise), 1),
        "roof_pitch": f"{rise:g}/12",
        "pitch_source": source,
        "needs_pitch_from_estimator": source == "default",
        "roof_perimeter_ft": round(geo.simplified_perimeter_px(roof_mask) * gsd * geo.FT_PER_M, 1),
        "parcel_shift_m": round(shift_m, 2),
        "gsd_m": round(gsd, 4),
    }
    if proba is not None and parcel.any():
        conf = float(np.asarray(proba).max(axis=0)[parcel].mean())
        values["confidence"] = round(conf, 3)
        if conf < LOW_CONFIDENCE:
            reasons.append(f"model confidence {conf:.2f} inside the parcel is below {LOW_CONFIDENCE}")
    if lawn_total > 0 and lawn_u / lawn_total > CANOPY_REVIEW_SHARE:
        reasons.append(f"{lawn_u / lawn_total:.0%} of the lawn is under tree canopy")
    measured_on = measured_on or date.today()
    age = (measured_on - inp.captured_on).days
    values["imagery_age_days"] = age
    if age > STALE_IMAGERY_DAYS:
        reasons.append(f"imagery captured {inp.captured_on.isoformat()} is over 18 months old")
    return Measurement(inp.property_id, "v2", values, reasons, parcel_px, snapped, roof_mask,
                       lawn_visible | under | frontage)


def _measure_v1(inp, classes, parcel_px, px_sqft, gsd) -> Measurement:
    parcel = geo.rasterize(parcel_px, classes.shape)
    roof = (classes == ROOF) & parcel
    lawn = (classes == LAWN) & parcel
    plan = float(roof.sum() * px_sqft)
    values = {
        "lawn_sqft": round(float(lawn.sum() * px_sqft), 1),
        "bed_sqft": round(float(((classes == BED) & parcel).sum() * px_sqft), 1),
        "pavement_sqft": round(float(((classes == PAVEMENT) & parcel).sum() * px_sqft), 1),
        "roof_plan_sqft": round(plan, 1),
        "roof_sqft": round(plan, 1),
        "roof_perimeter_ft": round(geo.staircase_perimeter_px(roof) * gsd * geo.FT_PER_M, 1),
        "gsd_m": round(gsd, 4),
    }
    return Measurement(inp.property_id, "v1", values, [], parcel_px, parcel_px, roof, lawn)
