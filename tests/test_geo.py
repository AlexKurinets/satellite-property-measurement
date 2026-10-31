import math

import numpy as np
import pytest

from propmeasure import geo


def test_ground_resolution_shrinks_with_latitude():
    assert geo.ground_resolution(0, 19) == pytest.approx(0.298582, rel=1e-5)
    assert geo.ground_resolution(60, 19) == pytest.approx(0.298582 / 2, rel=1e-5)
    # counting area with the nominal grid size instead would overstate it by 1/cos^2(lat): 33% at 30 degrees north
    assert (geo.ground_resolution(0, 19) / geo.ground_resolution(30, 19)) ** 2 == pytest.approx(4 / 3, rel=1e-6)


def test_lonlat_world_pixel_round_trip():
    for lon, lat in [(-40.0, 30.4), (-41.2, 39.0), (-42.0, 40.6)]:
        x, y = geo.lonlat_to_world_px(lon, lat, 19)
        lon2, lat2 = geo.world_px_to_lonlat(x, y, 19)
        assert lon2 == pytest.approx(lon, abs=1e-10)
        assert lat2 == pytest.approx(lat, abs=1e-10)


def test_capture_ring_round_trip():
    pts = np.array([(10.25, 20.5), (80.0, 20.5), (80.0, 150.75), (10.25, 150.75)])
    ring = geo.capture_px_to_ring(pts, 12345678.0, 23456789.0, 19)
    back = geo.ring_to_capture_px(ring, 12345678.0, 23456789.0, 19)
    assert np.allclose(back, pts, atol=1e-6)


def test_rasterize_counts_pixel_centers():
    poly = np.array([(10, 20), (30, 20), (30, 30), (10, 30)], dtype=float)
    assert geo.rasterize(poly, (64, 64)).sum() == 200


def test_pitch_factor():
    assert geo.pitch_factor(0) == 1.0
    assert geo.pitch_factor(6) == pytest.approx(1.1180, abs=1e-4)
    assert geo.pitch_factor(12) == pytest.approx(math.sqrt(2))


def _rotated_rect(w, h, deg, size=160):
    yy, xx = np.mgrid[0:size, 0:size] + 0.5
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    u = (xx - size / 2) * c + (yy - size / 2) * s
    v = -(xx - size / 2) * s + (yy - size / 2) * c
    return (np.abs(u) <= w / 2) & (np.abs(v) <= h / 2)


def test_staircase_perimeter_runs_long_on_diagonal_edges():
    assert geo.staircase_perimeter_px(_rotated_rect(40, 60, 0)) == 200
    true = 200.0
    stair = geo.staircase_perimeter_px(_rotated_rect(40, 60, 20))
    assert stair / true == pytest.approx(math.cos(math.radians(20)) + math.sin(math.radians(20)), abs=0.04)


@pytest.mark.parametrize("deg", [0, 7, 20, 33])
def test_simplified_perimeter_is_close_to_true(deg):
    assert geo.simplified_perimeter_px(_rotated_rect(40, 60, deg)) == pytest.approx(200, rel=0.03)
