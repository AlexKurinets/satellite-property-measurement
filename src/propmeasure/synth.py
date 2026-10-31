"""Synthetic residential properties: satellite-style imagery, label masks, parcels and measured truth.

Each scene is laid out in metres with the street along the bottom edge (production rotated every capture so the
parcel's frontage faces down), drawn on a grid four times finer than the imagery, then averaged down so boundary
pixels are mixtures the way real 30 cm pixels are. Truth areas come from the fine grid, which stands in for the
hand tracing on sharper imagery that production's benchmark used.

Two domains:
- "base": the public land-cover look the original model was trained on. Summer lawns, mulch beds, short shadows.
- "target": the client's three markets, each with the failure that mattered there. Dormant warm-season lawns in
  winter captures (coastal-south), heavy canopy over lawns (mid-atlantic), gravel beds and long low-sun shadows
  (mountain-west). County parcel lines are delivered offset from the imagery by up to 2.5 m.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F

from . import geo
from .classes import BED, CANOPY, LAWN, N_CLASSES, OTHER, PAVEMENT, ROOF

N = 192  # imagery pixels per side
S = 4  # supersampling factor for the truth grid
NF = N * S
ZOOM = 19

# materials
SOIL, LAWN_GREEN, LAWN_DORMANT, MULCH_DARK, MULCH_RED, GRAVEL, CONCRETE, ASPHALT, ROOF_MAT, POOL = range(10)
MAT_CLASS_TARGET = np.array([OTHER, LAWN, LAWN, BED, BED, BED, PAVEMENT, PAVEMENT, ROOF, OTHER], dtype=np.uint8)
MAT_CLASS_BASE = MAT_CLASS_TARGET.copy()
MAT_CLASS_BASE[SOIL] = BED  # public "bare_soil" includes dirt

BASE_RGB = np.array([
    (150, 121, 90),  # soil
    (82, 124, 56),  # lawn, growing
    (166, 147, 104),  # lawn, dormant warm-season grass
    (80, 60, 45),  # dark mulch
    (122, 64, 47),  # red mulch
    (166, 162, 152),  # gravel
    (186, 184, 178),  # concrete
    (80, 80, 84),  # asphalt
    (0, 0, 0),  # roof, coloured per house
    (70, 148, 182),  # pool
], dtype=np.float32)
# texture amplitudes per material: fine grain, medium, coarse blotches, speckle
TEXTURE = np.array([
    (8, 8, 4, 0),
    (6, 5, 2, 0),
    (8, 6, 2, 0),
    (5, 4, 13, 0),
    (5, 4, 12, 0),
    (3, 3, 2, 20),
    (3, 2, 1, 0),
    (4, 3, 1, 0),
    (3, 1, 0, 0),
    (2, 3, 0, 0),
], dtype=np.float32)
ROOF_RGB = [(72, 72, 78), (112, 88, 72), (152, 84, 62), (150, 150, 148), (96, 104, 112)]
CANOPY_RGB = (46, 80, 42)

PITCH_RISES = [4, 5, 6, 7, 8, 9, 10, 12]
PITCH_WEIGHTS = np.array([1, 2, 3, 3, 2, 1, 1, 1], dtype=float)

MARKETS = {
    "coastal-south": dict(lat=30.4, lon=-40.0, dormant=0.45, overseed=0.15, trees=(1, 3), tree_r=(2.0, 3.5),
                          beds={MULCH_DARK: 0.5, MULCH_RED: 0.3, GRAVEL: 0.2}, sun_elev=(35, 70), shadow=0.62,
                          sidewalk=0.9, lidar=0.6, pool=0.15, dirt=0.1, width=(15, 22)),
    "mid-atlantic": dict(lat=39.0, lon=-41.0, dormant=0.0, overseed=0.0, trees=(2, 4), tree_r=(2.5, 4.5),
                         beds={MULCH_DARK: 0.7, MULCH_RED: 0.25, GRAVEL: 0.05}, sun_elev=(35, 65), shadow=0.6,
                         sidewalk=0.9, lidar=0.8, pool=0.03, dirt=0.1, width=(14, 20)),
    "mountain-west": dict(lat=40.6, lon=-42.0, dormant=0.0, overseed=0.0, trees=(0, 2), tree_r=(2.0, 3.5),
                          beds={GRAVEL: 0.6, MULCH_DARK: 0.3, MULCH_RED: 0.1}, sun_elev=(24, 42), shadow=0.5,
                          sidewalk=0.9, lidar=0.7, pool=0.05, dirt=0.1, width=(15, 22)),
}
BASE_DOMAIN = dict(lat=(32.0, 42.0), lon=-45.0, dormant=0.0, overseed=0.0, trees=(1, 3), tree_r=(2.0, 4.0),
                   beds={MULCH_DARK: 0.6, MULCH_RED: 0.4}, sun_elev=(38, 72), shadow=0.68, sidewalk=0.85,
                   lidar=0.0, pool=0.0, dirt=0.3, width=(14, 22))


@dataclass
class Scene:
    image: np.ndarray  # (N, N, 3) uint8
    label: np.ndarray  # (N, N) uint8 visible class
    domain: str
    market: str
    lat: float
    lon: float
    zoom: int
    origin: tuple[float, float]  # world pixel of the top-left corner
    parcel_true: np.ndarray  # (4, 2) capture pixel coordinates
    parcel_delivered: np.ndarray
    pitch_rise: int
    lidar_rise: float | None
    season: str
    captured_on: date
    truth: dict = field(default_factory=dict)
    layout: dict = field(default_factory=dict)  # street geometry in metres, for tests

    @property
    def gsd(self) -> float:
        return geo.ground_resolution(self.lat, self.zoom)


class _Canvas:
    def __init__(self, gsd: float):
        self.fs = gsd / S
        self.mat = np.full((NF, NF), LAWN_GREEN, dtype=np.uint8)
        self.parcel = np.full((NF, NF), -1, dtype=np.int8)
        self.owner = np.full((NF, NF), -1, dtype=np.int8)  # whose lawn tint a pixel takes, including the strip
        self.house = np.full((NF, NF), -1, dtype=np.int8)
        self.tree = np.zeros((NF, NF), dtype=np.int16)
        self.shade = np.ones((NF, NF), dtype=np.float32)

    def _box(self, x0, y0, x1, y1):
        fs = self.fs
        c0, c1 = max(0, int(math.floor(x0 / fs))), min(NF, int(math.ceil(x1 / fs)) + 1)
        r0, r1 = max(0, int(math.floor(y0 / fs))), min(NF, int(math.ceil(y1 / fs)) + 1)
        if c0 >= c1 or r0 >= r1:
            return None
        xs = (np.arange(c0, c1) + 0.5) * fs
        ys = (np.arange(r0, r1) + 0.5) * fs
        X, Y = np.meshgrid(xs, ys)
        return (slice(r0, r1), slice(c0, c1)), X, Y

    def rect(self, x0, y0, x1, y1):
        b = self._box(x0, y0, x1, y1)
        if b is None:
            return None, None
        sl, X, Y = b
        return sl, (X >= x0) & (X < x1) & (Y >= y0) & (Y < y1)

    def orect(self, cx, cy, hw, hh, theta):
        r = math.hypot(hw, hh)
        b = self._box(cx - r, cy - r, cx + r, cy + r)
        if b is None:
            return None, None, None, None
        sl, X, Y = b
        c, s = math.cos(theta), math.sin(theta)
        u = (X - cx) * c + (Y - cy) * s
        v = -(X - cx) * s + (Y - cy) * c
        return sl, (np.abs(u) <= hw) & (np.abs(v) <= hh), u, v

    def blob(self, cx, cy, r, rng):
        b = self._box(cx - 1.25 * r, cy - 1.25 * r, cx + 1.25 * r, cy + 1.25 * r)
        if b is None:
            return None, None
        sl, X, Y = b
        a = np.arctan2(Y - cy, X - cx)
        p1, p2 = rng.uniform(0, 2 * math.pi, 2)
        rr = r * (1 + 0.12 * np.sin(3 * a + p1) + 0.07 * np.sin(5 * a + p2))
        return sl, (X - cx) ** 2 + (Y - cy) ** 2 <= rr**2

    def paint(self, sl, m, mat):
        if sl is not None:
            self.mat[sl][m] = mat


def _pick(rng, weights: dict):
    keys = list(weights)
    p = np.array([weights[k] for k in keys], dtype=float)
    return keys[rng.choice(len(keys), p=p / p.sum())]


def _shift(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = np.zeros_like(mask)
    h, w = mask.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = mask[ys, xs]
    return out


@dataclass
class _House:
    cx: float
    cy: float
    hw: float
    hh: float
    theta: float
    wing: tuple[int, float, float] | None  # side, half width, protrusion toward the street
    hip: bool


def _house_parts(h: _House):
    """(cx, cy, hw, hh, ridge_along_u) for the main body and the optional garage wing, in world metres."""
    parts = [(h.cx, h.cy, h.hw, h.hh, h.hw >= h.hh, h.hip)]
    if h.wing:
        side, gw, p = h.wing
        uw = side * (h.hw - gw)
        v0, v1 = h.hh - 1.0, h.hh + p
        vw, hhw = (v0 + v1) / 2, (v1 - v0) / 2
        c, s = math.cos(h.theta), math.sin(h.theta)
        parts.append((h.cx + uw * c - vw * s, h.cy + uw * s + vw * c, gw, hhw, False, False))
    return parts


def _house_perimeter_m(h: _House) -> float:
    """Exact outline length of the house in its own frame (axis-aligned there, so edge counting is exact)."""
    res = 0.05
    vmax = h.hh + (h.wing[2] if h.wing else 0.0)
    us = (np.arange(int(2 * h.hw / res) + 4) - 2 + 0.5) * res - h.hw
    vs = (np.arange(int((vmax + h.hh) / res) + 4) - 2 + 0.5) * res - h.hh
    U, V = np.meshgrid(us, vs)
    m = (np.abs(U) <= h.hw) & (np.abs(V) <= h.hh)
    if h.wing:
        side, gw, p = h.wing
        uw = side * (h.hw - gw)
        m |= (np.abs(U - uw) <= gw) & (V >= h.hh - 1.0) & (V <= h.hh + p)
    return geo.staircase_perimeter_px(m) * res


def _draw_house(cv: _Canvas, hid: int, h: _House, sun_dir: np.ndarray):
    c, s = math.cos(h.theta), math.sin(h.theta)
    ax_u, ax_v = np.array([c, s]), np.array([-s, c])
    for cx, cy, hw, hh, ridge_u, hip in _house_parts(h):
        sl, m, u, v = cv.orect(cx, cy, hw, hh, h.theta)
        if sl is None:
            continue
        if hip:
            end = (hw - np.abs(u)) < (hh - np.abs(v)) if ridge_u else (hh - np.abs(v)) < (hw - np.abs(u))
            along_u = end if ridge_u else ~end
        else:
            along_u = np.zeros_like(m) if ridge_u else np.ones_like(m)
        # plane facing direction: +/- u axis or +/- v axis
        nu = np.sign(u) * (ax_u @ sun_dir)
        nv = np.sign(v) * (ax_v @ sun_dir)
        shade = 1.0 + 0.2 * np.where(along_u, nu, nv)
        ridge = (np.abs(v) < 0.1) if ridge_u else (np.abs(u) < 0.1)
        if not hip:
            shade = shade + 0.08 * ridge
        cv.mat[sl][m] = ROOF_MAT
        cv.house[sl][m] = hid
        cv.shade[sl][m] = shade[m]


def _furnish(cv, rng, cfg, pid, x0, x1, y_back, y_lot, front, layout, houses, trees, domain):
    """Beds, paving, house and trees on one parcel. front is True for parcels facing this street."""
    w, d = x1 - x0, y_lot - y_back
    bed_mat = _pick(rng, cfg["beds"])
    hw = min(w * rng.uniform(0.48, 0.68), rng.uniform(9.0, 15.0)) / 2
    setback = rng.uniform(6.0, 10.0) if front else rng.uniform(3.0, 6.0)
    hh = min(rng.uniform(8.0, 12.5), max(6.0, d - setback - 5.0)) / 2
    side_margin = 1.5
    cx = x0 + side_margin + hw + rng.uniform(0, max(0.01, w - 2 * hw - 2 * side_margin))
    theta = 0.0
    if rng.random() < 0.3:
        theta = math.radians(rng.choice([-1, 1]) * rng.uniform(5, 20))
    wing = None
    if front and rng.random() < 0.55 and setback > 4.0:
        wing = (int(rng.choice([-1, 1])), rng.uniform(1.8, 3.2), rng.uniform(2.0, min(4.5, setback - 2.0)))
    if front:
        cy = y_lot - setback - hh
    else:
        cy = y_back + setback + hh
        theta += math.pi  # faces the other street
    house = _House(cx, cy, hw, hh, theta, wing, bool(rng.random() < 0.4))
    c, s = math.cos(theta), math.sin(theta)

    # foundation bed along the front face, drawn in the house frame
    bed_d = rng.uniform(1.0, 2.0)
    bcx, bcy = cx - (hh + bed_d / 2) * s, cy + (hh + bed_d / 2) * c
    sl, m, _, _ = cv.orect(bcx, bcy, hw, bed_d / 2, theta)
    cv.paint(sl, m, bed_mat)
    if front and rng.random() < 0.4:
        bw, bh = rng.uniform(2.0, 4.0), rng.uniform(2.0, 3.5)
        bx = x0 + 0.4 if rng.random() < 0.5 else x1 - 0.4 - bw
        sl, m = cv.rect(bx, y_lot - 0.6 - bh, bx + bw, y_lot - 0.6)
        cv.paint(sl, m, bed_mat)
    # patio behind the house
    if rng.random() < 0.5:
        pw, pd = rng.uniform(1.5, 2.5), rng.uniform(1.5, 2.2)
        pu = rng.uniform(-hw + pw, hw - pw)
        pcx, pcy = cx + pu * c + (hh + pd) * s, cy + pu * s - (hh + pd) * c
        sl, m, _, _ = cv.orect(pcx, pcy, pw, pd, theta)
        cv.paint(sl, m, CONCRETE)
    if front and rng.random() < cfg["pool"]:
        pcx, pcy = cx + rng.uniform(-hw / 2, hw / 2), cy - hh - 5.0
        if pcy - 3.0 > y_back + 1.0:
            sl, m, _, _ = cv.orect(pcx, pcy, 3.0, 2.2, 0.0)
            cv.paint(sl, m, CONCRETE)
            sl, m, _, _ = cv.orect(pcx, pcy, 2.2, 1.4, 0.0)
            cv.paint(sl, m, POOL)
    # island beds in the lawn: more boundary per square foot, which is where area error comes from
    for _ in range(int(rng.integers(0, 3)) if front else 0):
        sl, m = cv.blob(rng.uniform(x0 + 2, x1 - 2), rng.uniform(y_back + 2, y_lot - 2), rng.uniform(0.8, 2.0), rng)
        cv.paint(sl, m, bed_mat)
    if rng.random() < cfg["dirt"]:
        sl, m = cv.blob(rng.uniform(x0 + 2, x1 - 2), rng.uniform(y_back + 2, y_lot - 2), rng.uniform(0.8, 1.8), rng)
        cv.paint(sl, m, SOIL)

    if front:
        # walkway and driveway run to the street; the sidewalk is painted over them afterwards
        front_y = cy + hh
        if rng.random() < 0.6:
            sl, m = cv.rect(cx - 0.55, front_y, cx + 0.55, layout["walk_to"])
            cv.paint(sl, m, CONCRETE)
        if wing:
            side, gw, p = wing
            dcx, dw, top = cx + side * (hw - gw) * c, gw, cy + hh + p
        else:
            dw = rng.uniform(1.5, 2.7)
            side = rng.choice([-1, 1])
            dcx, top = cx + side * (hw - dw) * c, front_y
        dcx = min(max(dcx, x0 + dw + 0.3), x1 - dw - 0.3)
        sl, m = cv.rect(dcx - dw, top - 0.5, dcx + dw, layout["y_curb"])
        cv.paint(sl, m, CONCRETE if rng.random() < 0.7 else ASPHALT)
    houses.append(house)

    # trees, kept far enough from the house that canopy rarely hides roof or foundation bed
    n_trees = int(rng.integers(cfg["trees"][0], cfg["trees"][1] + 1))
    placed = 0
    for _ in range(40):
        if placed >= n_trees:
            break
        r = rng.uniform(*cfg["tree_r"])
        tx, ty = rng.uniform(x0 + 0.5, x1 - 0.5), rng.uniform(y_back + 0.5, y_lot - 1.0)
        du, dv = (tx - cx) * c + (ty - cy) * s, -(tx - cx) * s + (ty - cy) * c
        if abs(du) < hw + 0.7 * r + 0.5 and abs(dv) < hh + 0.7 * r + 0.5:
            continue
        # canopy goes over lawn: skip spots with paving or beds under most of the crown
        sl, m = cv.blob(tx, ty, 0.8 * r, rng)
        if sl is not None and m.any():
            under = cv.mat[sl][m]
            if np.isin(under, (CONCRETE, ASPHALT, POOL, bed_mat)).mean() > 0.1:
                continue
        trees.append((tx, ty, r))
        placed += 1
    return bed_mat


def make_scene(seed: int, domain: str = "target", market: str | None = None) -> Scene:
    rng = np.random.default_rng(seed)
    if domain == "base":
        cfg = BASE_DOMAIN
        market = "base"
        lat = float(rng.uniform(*cfg["lat"]))
        lon = cfg["lon"] + seed % 997 * 0.002
    else:
        cfg = MARKETS[market]
        lat = cfg["lat"] + (seed % 97) * 1e-4
        lon = cfg["lon"] + (seed % 991) * 0.002
    gsd = geo.ground_resolution(lat, ZOOM)
    cm = N * gsd
    cv = _Canvas(gsd)

    # street along the bottom edge
    road_w = rng.uniform(3.0, 4.5)
    y_curb = cm - road_w
    has_sidewalk = rng.random() < cfg["sidewalk"]
    if has_sidewalk:
        strip_w, walk_w = rng.uniform(1.0, 2.5), rng.uniform(1.2, 1.8)
        y_sw_top = y_curb - strip_w - walk_w
        y_lot = y_sw_top - rng.uniform(0.0, 0.3)
    else:
        y_sw_top = None
        y_lot = y_curb - rng.uniform(2.0, 4.0)
    layout = {"y_curb": y_curb, "walk_to": y_sw_top if has_sidewalk else y_curb}

    w = min(rng.uniform(*cfg["width"]), cm - 10.0)
    d = min(rng.uniform(22.0, 34.0), y_lot - 5.0)
    xc = cm / 2 + rng.uniform(-2.0, 2.0)
    px0, px1, py0 = xc - w / 2, xc + w / 2, y_lot - d
    left = min(px0 - rng.uniform(14, 20), -1.0)
    right = max(px1 + rng.uniform(14, 20), cm + 1.0)
    # target, two side neighbours at least as deep, and the rear neighbour facing the next street
    parcels = [(px0, px1, py0, y_lot, True), (left, px0, py0 - rng.uniform(0, 3), y_lot, True),
               (px1, right, py0 - rng.uniform(0, 3), y_lot, True),
               (-5.0, cm + 5.0, py0 - rng.uniform(18, 26), py0, False)]
    for pid in (3, 0, 1, 2):
        a, b, top, bottom, _ = parcels[pid]
        sl, m = cv.rect(a, top, b, bottom)
        if sl is not None:
            cv.parcel[sl][m] = pid
            cv.owner[sl][m] = pid
    # right-of-way: strip lawn takes the tint of the parcel it fronts
    for pid in (0, 1, 2):
        a, b = parcels[pid][0], parcels[pid][1]
        sl, m = cv.rect(a, y_lot, b, cm + 1)
        if sl is not None:
            cv.owner[sl][m] = pid
    sl, m = cv.rect(-1, y_curb, cm + 1, cm + 1)
    cv.paint(sl, m, ASPHALT)

    houses: list[_House] = []
    trees: list[tuple[float, float, float]] = []
    bed_mats = []
    for pid, (a, b, top, bottom, front) in enumerate(parcels):
        bed_mats.append(_furnish(cv, rng, cfg, pid, a, b, top, bottom, front, layout, houses, trees, domain))
    if has_sidewalk:
        sl, m = cv.rect(-1, y_sw_top, cm + 1, y_sw_top + walk_w)
        cv.paint(sl, m, CONCRETE)
    sun_az = rng.uniform(0, 2 * math.pi)
    sun_dir = np.array([math.cos(sun_az), math.sin(sun_az)])
    for hid, h in enumerate(houses):
        _draw_house(cv, hid, h, sun_dir)
    for tid, (tx, ty, r) in enumerate(trees):
        sl, m = cv.blob(tx, ty, r, rng)
        if sl is not None:
            cv.tree[sl][m] = tid + 1

    # season
    season = "growing"
    if domain == "target" and rng.random() < cfg["dormant"]:
        season = "dormant"
    dormant_owner = np.array([season == "dormant" and rng.random() >= cfg["overseed"] for _ in range(4)])
    lawn = cv.mat == LAWN_GREEN
    dorm = lawn & (cv.owner >= 0) & dormant_owner[np.clip(cv.owner, 0, 3)]
    cv.mat[dorm] = LAWN_DORMANT

    # truth on the fine grid
    fa_sqft = (cv.fs**2) * geo.SQFT_PER_M2
    mat_class = MAT_CLASS_BASE if domain == "base" else MAT_CLASS_TARGET
    ground = mat_class[cv.mat]
    parcel0 = cv.parcel == 0
    fsl, fm = cv.rect(px0, y_lot, px1, y_curb)
    frontage = np.zeros_like(parcel0)
    if fsl is not None:
        frontage[fsl] = fm
    rise = int(rng.choice(PITCH_RISES, p=PITCH_WEIGHTS / PITCH_WEIGHTS.sum()))
    roof_plan = float((cv.house == 0).sum() * fa_sqft)
    truth = {
        "lawn_sqft": float(((ground == LAWN) & (parcel0 | frontage)).sum() * fa_sqft),
        "bed_sqft": float(((ground == BED) & parcel0).sum() * fa_sqft),
        "pavement_sqft": float(((ground == PAVEMENT) & parcel0).sum() * fa_sqft),
        "roof_plan_sqft": roof_plan,
        "roof_sqft": roof_plan * geo.pitch_factor(rise),
        "roof_perimeter_ft": _house_perimeter_m(houses[0]) * geo.FT_PER_M,
        "lawn_under_canopy_sqft": float(((ground == LAWN) & parcel0 & (cv.tree > 0)).sum() * fa_sqft),
        "frontage_lawn_sqft": float(((ground == LAWN) & frontage).sum() * fa_sqft),
    }

    # label: majority visible class in each imagery pixel
    visible = ground.copy()
    visible[cv.tree > 0] = CANOPY
    counts = np.stack([(visible == k).reshape(N, S, N, S).sum(axis=(1, 3)) for k in range(N_CLASSES)])
    label = counts.argmax(axis=0).astype(np.uint8)

    image = _render(cv, rng, cfg, domain, sun_az, trees)

    # parcels in capture pixels; the county line arrives offset from the imagery
    true_px = np.array([(px0, py0), (px1, py0), (px1, y_lot), (px0, y_lot)]) / gsd
    if domain == "target":
        off = np.array([rng.uniform(-0.6, 0.6), rng.uniform(-2.5, 2.5)]) / gsd
    else:
        off = np.zeros(2)
    lidar = None
    if rng.random() < cfg["lidar"]:
        lidar = round(rise + float(rng.uniform(-0.3, 0.3)), 1)
    xc_w, yc_w = geo.lonlat_to_world_px(lon, lat, ZOOM)
    origin = (math.floor(xc_w) - N / 2, math.floor(yc_w) - N / 2)
    return Scene(
        image=image, label=label, domain=domain, market=market, lat=lat, lon=lon, zoom=ZOOM, origin=origin,
        parcel_true=true_px, parcel_delivered=true_px + off, pitch_rise=rise, lidar_rise=lidar, season=season,
        captured_on=_capture_date(rng, season, market), truth=truth,
        layout={"has_sidewalk": has_sidewalk, "y_lot_m": y_lot, "y_curb_m": y_curb, "y_sidewalk_top_m": y_sw_top},
    )


def _capture_date(rng, season: str, market: str) -> date:
    """Mostly the last twelve months before the benchmark date (30 Oct 2026); a few captures are years old."""
    if season == "dormant" or (market == "mountain-west" and rng.random() < 0.6):
        month = int(rng.choice([1, 2, 11, 12]))
    else:
        month = int(rng.integers(4, 11))
    year = 2025 if month > 10 else 2026
    if rng.random() < 0.04:
        year -= 2
    return date(year, month, int(rng.integers(1, 29)))


def _render(cv: _Canvas, rng, cfg, domain, sun_az, trees) -> np.ndarray:
    def noise(sigma, factor=1):
        n = gaussian_filter(rng.standard_normal((NF // factor, NF // factor)).astype(np.float32), sigma / factor)
        if factor > 1:
            t = torch.from_numpy(n)[None, None]
            n = F.interpolate(t, scale_factor=factor, mode="bilinear", align_corners=False)[0, 0].numpy()
        return n / (n.std() + 1e-6)

    n_fine, n_med, n_coarse, n_speck = noise(1.0), noise(4.0, 4), noise(12.0, 8), noise(0.6)
    mat = cv.mat
    rgb = BASE_RGB[mat].copy()
    tex = TEXTURE[mat]
    lum = tex[..., 0] * n_fine + tex[..., 1] * n_med + tex[..., 2] * n_coarse + tex[..., 3] * n_speck
    rgb += lum[..., None]

    # lawns: a tint per parcel and mowing stripes
    lawn = (mat == LAWN_GREEN) | (mat == LAWN_DORMANT)
    tints = rng.uniform(-9, 9, size=(5, 3)).astype(np.float32)
    tints[:, 1] += rng.uniform(-6, 6, size=5)
    owner = np.clip(cv.owner, -1, 3) + 1
    rgb[lawn] += tints[owner[lawn]]
    fs = cv.fs
    yy, xx = np.mgrid[0:NF, 0:NF].astype(np.float32) * fs
    stripes = np.zeros((NF, NF), dtype=np.float32)
    for p in range(-1, 4):
        sel = lawn & (cv.owner == p)
        if not sel.any():
            continue
        coord = yy if rng.random() < 0.5 else xx
        stripes[sel] = np.tanh(3 * np.sin(2 * math.pi * coord[sel] / 1.6 + rng.uniform(0, 6.3))) * 5
    rgb += stripes[..., None]

    # roofs
    roof = mat == ROOF_MAT
    roof_colors = np.array(ROOF_RGB, dtype=np.float32)
    hues = rng.integers(len(ROOF_RGB), size=int(cv.house.max()) + 2)
    rgb[roof] = roof_colors[hues[cv.house[roof]]] * cv.shade[roof][:, None] + 3 * n_fine[roof][:, None]

    # canopy over everything below it
    canopy = cv.tree > 0
    if canopy.any():
        tint = rng.uniform(-8, 8, size=(len(trees) + 1, 3)).astype(np.float32)
        tex_c = 11 * n_med + 9 * n_coarse + 6 * n_fine
        rgb[canopy] = np.array(CANOPY_RGB, dtype=np.float32) + tint[cv.tree[canopy]] + tex_c[canopy][:, None]

    # shadows of houses and trees, cast away from the sun
    elev = math.radians(rng.uniform(*cfg["sun_elev"]))
    away = -np.array([math.cos(sun_az), math.sin(sun_az)])
    shadow = np.zeros((NF, NF), dtype=bool)
    for height, obj in ((6.0, cv.house >= 0), (9.0, canopy)):
        length = height / math.tan(elev) / fs
        steps = max(1, int(length // 10))
        for k in range(1, steps + 1):
            frac = k / steps
            shadow |= _shift(obj, int(round(away[0] * length * frac)), int(round(away[1] * length * frac)))
    shadow &= ~roof & ~canopy
    rgb[shadow] *= cfg["shadow"]

    img = rgb.reshape(N, S, N, S, 3).mean(axis=(1, 3))
    img = gaussian_filter(img, sigma=(0.75, 0.75, 0))
    if domain == "base":
        gain = rng.uniform(0.97, 1.03, size=3)
        offset = rng.uniform(-2, 2, size=3)
    else:
        gain = rng.uniform(0.92, 1.08, size=3)
        offset = rng.uniform(-6, 6, size=3)
    if domain == "target":  # haze and sensor contrast differences between captures
        contrast = rng.uniform(0.8, 1.0)
        img = img.mean(axis=(0, 1)) + (img - img.mean(axis=(0, 1))) * contrast
    img = img * gain + offset + rng.normal(0, 4.0, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)
