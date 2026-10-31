"""Web Mercator pixel geometry, area conversion, roof pitch and outline perimeter.

Imagery arrives on the Web Mercator tile grid (256 px tiles), where a pixel's ground size shrinks with cos(latitude).
Every area in this package is counted in pixels and converted with the ground resolution at the capture's latitude,
never with the grid's nominal resolution.
"""

from __future__ import annotations

import math

import numpy as np
from skimage.draw import polygon2mask
from skimage.measure import approximate_polygon, find_contours

EARTH_RADIUS_M = 6378137.0
TILE_PX = 256
SQFT_PER_M2 = 10.763910417
FT_PER_M = 3.280839895


def ground_resolution(lat: float, zoom: int) -> float:
    """Metres on the ground per pixel at this latitude and zoom."""
    return math.cos(math.radians(lat)) * 2 * math.pi * EARTH_RADIUS_M / (TILE_PX * 2**zoom)


def sqft_per_pixel(lat: float, zoom: int) -> float:
    return ground_resolution(lat, zoom) ** 2 * SQFT_PER_M2


def lonlat_to_world_px(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    scale = TILE_PX * 2**zoom
    x = (lon + 180.0) / 360.0 * scale
    s = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * scale
    return x, y


def world_px_to_lonlat(x: float, y: float, zoom: int) -> tuple[float, float]:
    scale = TILE_PX * 2**zoom
    lon = x / scale * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / scale))))
    return lon, lat


def ring_to_capture_px(ring: list[list[float]], origin_x: float, origin_y: float, zoom: int) -> np.ndarray:
    """Lon/lat ring to (x, y) pixel coordinates inside a capture whose top-left world pixel is the origin."""
    pts = [lonlat_to_world_px(lon, lat, zoom) for lon, lat in ring]
    return np.array([(x - origin_x, y - origin_y) for x, y in pts], dtype=float)


def capture_px_to_ring(pts: np.ndarray, origin_x: float, origin_y: float, zoom: int) -> list[list[float]]:
    return [list(world_px_to_lonlat(x + origin_x, y + origin_y, zoom)) for x, y in pts]


def rasterize(poly_xy: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Boolean mask of the pixels whose centres fall inside the polygon.

    Polygon coordinates are continuous pixel coordinates, with pixel (r, c) covering [c, c+1) x [r, r+1).
    """
    rc = np.column_stack([poly_xy[:, 1] - 0.5, poly_xy[:, 0] - 0.5])
    return polygon2mask(shape, rc)


def pitch_factor(rise_in_12: float) -> float:
    """Roof surface area over plan area for a roof whose planes all share this pitch."""
    return math.sqrt(1.0 + (rise_in_12 / 12.0) ** 2)


def staircase_perimeter_px(mask: np.ndarray) -> float:
    """Perimeter counted along pixel edges. On an edge at angle a it runs long by |cos a| + |sin a|."""
    m = np.pad(mask.astype(np.int8), 1)
    return float(np.abs(np.diff(m, axis=0)).sum() + np.abs(np.diff(m, axis=1)).sum())


def outline_polygons(mask: np.ndarray, tolerance_px: float = 0.9) -> list[np.ndarray]:
    """Outer outlines of a mask as simplified (x, y) polygons in continuous pixel coordinates."""
    padded = np.pad(mask.astype(float), 1)
    polys = []
    for contour in find_contours(padded, 0.5):
        if len(contour) < 8:
            continue
        simple = approximate_polygon(contour, tolerance=tolerance_px)
        # find_contours runs through pixel centres (offset by the pad); move to pixel-edge coordinates
        polys.append(np.column_stack([simple[:, 1] - 0.5, simple[:, 0] - 0.5]))
    return polys


def polygon_perimeter(poly_xy: np.ndarray) -> float:
    closed = np.vstack([poly_xy, poly_xy[:1]]) if not np.allclose(poly_xy[0], poly_xy[-1]) else poly_xy
    return float(np.sqrt((np.diff(closed, axis=0) ** 2).sum(axis=1)).sum())


def simplified_perimeter_px(mask: np.ndarray, tolerance_px: float = 0.9) -> float:
    """Perimeter of the vectorised, simplified outline: what a crew would measure along the eaves."""
    return sum(polygon_perimeter(p) for p in outline_polygons(mask, tolerance_px))
