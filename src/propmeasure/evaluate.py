"""Benchmark scoring in square feet, not pixels.

Every model and pipeline is scored on the same held-out properties as percentage area error per feature: median
absolute error, median signed error (bias, which is what makes quotes run high or low across a whole market),
90th percentile, and the share of properties within 10%. The oracle run feeds the hand-traced masks through a
pipeline instead of model output, so whatever error remains belongs to the pipeline and not to the model.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date

import numpy as np

from .classes import N_CLASSES
from .measure import PropertyInput, measure
from .model import UNet, predict_proba

FEATURES = ("lawn_sqft", "bed_sqft", "pavement_sqft", "roof_sqft", "roof_perimeter_ft")
SELECTION_FEATURES = ("lawn_sqft", "bed_sqft", "pavement_sqft", "roof_plan_sqft")
STRATA = ("market", "season")


@dataclass
class BenchItem:
    inp: PropertyInput
    truth: dict
    label: np.ndarray
    market: str
    season: str


def classify(model: UNet, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    proba = predict_proba(model, image).numpy()
    return proba.argmax(axis=0).astype(np.uint8), proba


def _stats(errs: list[float]) -> dict:
    a = np.array(errs, dtype=float)
    return {
        "n": int(a.size),
        "mdape": round(float(np.median(np.abs(a))) * 100, 2),
        "bias": round(float(np.median(a)) * 100, 2),
        "p90": round(float(np.percentile(np.abs(a), 90)) * 100, 2),
        "within10": round(float((np.abs(a) <= 0.10).mean()) * 100, 1),
    }


def score(rows: list[dict], features=FEATURES) -> dict:
    """rows carry market, season, and per feature measured/truth values."""
    out = {"overall": {}, "by": {k: {} for k in STRATA}}
    for f in features:
        errs = [r["err"][f] for r in rows if f in r["err"]]
        if errs:
            out["overall"][f] = _stats(errs)
    for key in STRATA:
        groups: dict[str, list] = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        for g, members in sorted(groups.items()):
            out["by"][key][g] = {f: _stats([r["err"][f] for r in members if f in r["err"]]) for f in features
                                 if any(f in r["err"] for r in members)}
    return out


def run(items: list[BenchItem], model: UNet | None, pipeline: str, measured_on: date,
        features=FEATURES) -> dict:
    """Score a model (or the traced masks when model is None) through a pipeline."""
    rows = []
    inter = np.zeros(N_CLASSES)
    union = np.zeros(N_CLASSES)
    for it in items:
        if model is None:
            classes, proba = it.label, None
        else:
            classes, proba = classify(model, it.inp.image)
            for k in range(N_CLASSES):
                p, t = classes == k, it.label == k
                inter[k] += (p & t).sum()
                union[k] += (p | t).sum()
        m = measure(it.inp, classes, proba, pipeline=pipeline, measured_on=measured_on)
        err = {}
        for f in features:
            t = it.truth.get(f)
            if t and t > 0 and f in m.values:
                err[f] = (m.values[f] - t) / t
        rows.append({"property_id": it.inp.property_id, "market": it.market, "season": it.season, "err": err,
                     "review": bool(m.review_reasons)})
    report = score(rows, features)
    report["pipeline"] = pipeline
    report["oracle"] = model is None
    report["properties"] = len(items)
    report["review_share"] = round(100 * float(np.mean([r["review"] for r in rows])), 1) if rows else 0.0
    if model is not None:
        iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
        report["iou"] = [None if np.isnan(v) else round(float(v), 4) for v in iou]
        report["miou"] = round(float(np.nanmean(iou)), 4)
    return report


def validator(items: list[BenchItem], measured_on: date):
    """Validation hook for training: mean MdAPE over the selection features through pipeline v2, plus mIoU."""

    def _validate(model: UNet) -> dict:
        rep = run(items, model, "v2", measured_on, SELECTION_FEATURES)
        area = float(np.mean([rep["overall"][f]["mdape"] for f in SELECTION_FEATURES]))
        return {"area_mdape": round(area, 2), "lawn_mdape": rep["overall"]["lawn_sqft"]["mdape"],
                "miou": rep["miou"]}

    return _validate


def gate(candidate: dict, production: dict | None, cfg: dict) -> dict:
    """Release gate: the candidate must beat production overall and not lose ground in any market."""
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    for f in cfg["gated_features"]:
        c = candidate["overall"][f]
        check(f"{f} median abs error <= {cfg['max_mdape'][f]}%", c["mdape"] <= cfg["max_mdape"][f],
              f"{c['mdape']}%")
        check(f"{f} bias within +/-{cfg['max_abs_bias']}%", abs(c["bias"]) <= cfg["max_abs_bias"], f"{c['bias']}%")
        if production is not None:
            p = production["overall"][f]
            check(f"{f} better than production", c["mdape"] < p["mdape"], f"{c['mdape']}% vs {p['mdape']}%")
            for market, feats in candidate["by"]["market"].items():
                pm = production["by"]["market"].get(market, {}).get(f)
                if pm is None:
                    continue
                cm = feats[f]["mdape"]
                check(f"{f} in {market} no worse than production by {cfg['max_market_regression']} pts",
                      cm <= pm["mdape"] + cfg["max_market_regression"], f"{cm}% vs {pm['mdape']}%")
    return {"passed": all(c["passed"] for c in checks), "checks": checks}
