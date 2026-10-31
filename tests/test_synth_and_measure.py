from datetime import date

import numpy as np
import pytest

from propmeasure import geo
from propmeasure.classes import CANOPY, LAWN, PAVEMENT, ROOF
from propmeasure.measure import PropertyInput, measure
from propmeasure.synth import MARKETS, make_scene

DAY = date(2026, 10, 30)


@pytest.fixture(scope="module")
def scenes():
    markets = list(MARKETS)
    return [make_scene(70_000 + i, "target", markets[i % 3]) for i in range(18)]


def _inp(sc, lidar=True):
    ring = geo.capture_px_to_ring(sc.parcel_delivered, sc.origin[0], sc.origin[1], sc.zoom)
    return PropertyInput("p", sc.image, sc.lat, sc.zoom, sc.origin, sc.captured_on, ring,
                         sc.lidar_rise if lidar else None)


def _err(m, sc, f):
    return (m.values[f] - sc.truth[f]) / sc.truth[f]


def test_scene_is_deterministic():
    a, b = make_scene(5, "target", "mid-atlantic"), make_scene(5, "target", "mid-atlantic")
    assert np.array_equal(a.image, b.image) and np.array_equal(a.label, b.label)
    assert a.truth == b.truth


def test_truth_is_consistent(scenes):
    for sc in scenes:
        t = sc.truth
        assert t["roof_sqft"] == pytest.approx(t["roof_plan_sqft"] * geo.pitch_factor(sc.pitch_rise))
        assert 0 < t["lawn_under_canopy_sqft"] + t["frontage_lawn_sqft"] < t["lawn_sqft"]
        assert sc.image.shape == (192, 192, 3) and sc.image.dtype == np.uint8
        off = (sc.parcel_delivered - sc.parcel_true) * sc.gsd
        assert abs(off[0, 0]) <= 0.6 and abs(off[0, 1]) <= 2.5


def test_dormant_lawns_are_tan():
    sc = next(s for s in (make_scene(i, "target", "coastal-south") for i in range(40)) if s.season == "dormant")
    lawn = sc.image[sc.label == LAWN].astype(float).mean(axis=0)
    assert lawn[0] > lawn[1] * 0.95  # red at least level with green: brown, not green


def test_front_line_snaps_to_back_of_sidewalk(scenes):
    snapped = 0
    for sc in scenes:
        if not sc.layout["has_sidewalk"]:
            continue
        m = measure(_inp(sc), sc.label, pipeline="v2", measured_on=DAY)
        delivered_off_m = (sc.parcel_delivered[:, 1].max() - sc.parcel_true[:, 1].max()) * sc.gsd
        # the lot line sits 0 to 0.3 m behind the sidewalk, and pixels are about a quarter metre
        assert m.values["parcel_shift_m"] + delivered_off_m == pytest.approx(0.15, abs=0.5)
        snapped += 1
    assert snapped >= 10


def test_oracle_v2_beats_v1_on_lawn_and_roof(scenes):
    errs = {"v1": {"lawn_sqft": [], "roof_sqft": []}, "v2": {"lawn_sqft": [], "roof_sqft": []}}
    for sc in scenes:
        for pipe in ("v1", "v2"):
            m = measure(_inp(sc), sc.label, pipeline=pipe, measured_on=DAY)
            for f in errs[pipe]:
                errs[pipe][f].append(_err(m, sc, f))
    med = {p: {f: float(np.median(np.abs(v))) for f, v in d.items()} for p, d in errs.items()}
    assert med["v2"]["lawn_sqft"] < 0.06 and med["v2"]["roof_sqft"] < 0.05
    assert med["v1"]["lawn_sqft"] > 2 * med["v2"]["lawn_sqft"]
    # v1 reports plan area as roof size, so it always comes in low
    assert np.median(errs["v1"]["roof_sqft"]) < -0.05


def test_roof_selection_takes_the_whole_house_and_no_neighbours(scenes):
    for sc in scenes:
        m = measure(_inp(sc), sc.label, pipeline="v2", measured_on=DAY)
        assert _err(m, sc, "roof_plan_sqft") == pytest.approx(0, abs=0.12)


def test_perimeter_on_simplified_outline(scenes):
    v1, v2 = [], []
    for sc in scenes:
        v1.append(_err(measure(_inp(sc), sc.label, pipeline="v1", measured_on=DAY), sc, "roof_perimeter_ft"))
        v2.append(_err(measure(_inp(sc), sc.label, pipeline="v2", measured_on=DAY), sc, "roof_perimeter_ft"))
    assert abs(np.median(v2)) < 0.05
    assert np.median(v1) > np.median(v2)


def test_missing_pitch_defaults_and_asks_the_estimator(scenes):
    m = measure(_inp(scenes[0], lidar=False), scenes[0].label, pipeline="v2", measured_on=DAY)
    assert m.values["pitch_source"] == "default" and m.values["roof_pitch"] == "6/12"
    assert m.values["needs_pitch_from_estimator"]
    assert m.values["roof_sqft"] == pytest.approx(m.values["roof_plan_sqft"] * geo.pitch_factor(6), abs=0.2)


def test_stale_imagery_is_flagged(scenes):
    m = measure(_inp(scenes[0]), scenes[0].label, pipeline="v2", measured_on=date(2029, 1, 1))
    assert any("over 18 months old" in r for r in m.review_reasons)


def _toy(center_class):
    """Square parcel on lawn with a canopy disk in the middle, sitting on a ring of center_class."""
    classes = np.full((96, 96), LAWN, dtype=np.uint8)
    classes[88:, :] = PAVEMENT  # road
    yy, xx = np.mgrid[0:96, 0:96]
    d = np.hypot(yy - 40, xx - 48)
    classes[d < 18] = center_class
    classes[d < 12] = CANOPY
    classes[5:15, 5:15] = ROOF
    ring = geo.capture_px_to_ring(np.array([(4.0, 4.0), (92.0, 4.0), (92.0, 80.0), (4.0, 80.0)]), 0, 0, 19)
    inp = PropertyInput("toy", np.zeros((96, 96, 3), np.uint8), 30.0, 19, (0, 0), DAY, ring, 6.0)
    return measure(inp, classes, pipeline="v2", measured_on=DAY)


def test_canopy_counts_as_lawn_only_when_lawn_surrounds_it():
    on_lawn = _toy(LAWN)
    on_paving = _toy(PAVEMENT)
    assert on_lawn.values["lawn_under_canopy_sqft"] > 0
    assert on_paving.values["lawn_under_canopy_sqft"] == 0
