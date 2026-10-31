"""propmeasure command line: seed, train, fine-tune, benchmark, promote, measure, and the bootstrap that runs them."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

import torch

from . import registry, service
from .classes import BASE_CLASSES, CLIENT_CLASSES
from .db import bench_items, connect, migrate, property_input, training_arrays
from .evaluate import classify, validator
from .measure import measure
from .model import UNet, swap_head
from .overlay import render
from .seed import DEFAULT_SIZES, seed
from .train import TrainConfig, train

BASE_VERSION = "lc-unet-1.0"
FT_VERSION = "prop-unet-2.0"
# benchmark runs use a fixed "today" so imagery-age flags, and therefore results, don't drift with the calendar
REFERENCE_DATE = date(2026, 10, 30)
BEFORE_MONTH = date(2026, 8, 14)
AFTER_MONTH = date(2026, 10, 16)
BASE_STEPS = int(os.environ.get("PROPMEASURE_BASE_STEPS", "1500"))
FT_STEPS = int(os.environ.get("PROPMEASURE_FT_STEPS", "1000"))


def _threads():
    torch.set_num_threads(int(os.environ.get("PROPMEASURE_THREADS", str(min(4, os.cpu_count() or 1)))))


def log(msg: str) -> None:
    print(msg, flush=True)


def cmd_train_base(c, version=BASE_VERSION, steps=BASE_STEPS, production=True):
    _threads()
    images, labels = training_arrays(c, "base")
    cfg = TrainConfig(steps=steps, lr=3e-3, eval_every=max(1, steps // 5), seed=1)
    log(f"train-base: {len(images)} base-domain properties, {steps} steps")
    model, history = train(UNet(len(BASE_CLASSES)), images, labels, cfg, log=lambda m: log(f"  {m}"))
    registry.register(c, version, model, parent=None, label_set="base", pipeline="v1",
                      train_config=cfg.as_dict() | {"classes": BASE_CLASSES, "properties": len(images)},
                      history=history, status="production" if production else "candidate")
    log(f"train-base: registered {version}" + (" as production" if production else ""))


def cmd_finetune(c, parent=BASE_VERSION, version=FT_VERSION, steps=FT_STEPS):
    _threads()
    images, labels = training_arrays(c, "train")
    val = bench_items(c, "val")
    model = swap_head(registry.load(c, parent), len(CLIENT_CLASSES))
    cfg = TrainConfig(steps=steps, lr=3e-3, body_lr_scale=0.3, eval_every=max(1, steps // 5), seed=2)
    log(f"finetune: {len(images)} traced client properties, {len(val)} for checkpoint selection, {steps} steps")
    model, history = train(model, images, labels, cfg, validate=validator(val, REFERENCE_DATE),
                           log=lambda m: log(f"  {m}"))
    best = min(history, key=lambda h: h["area_mdape"])
    best_iou = max(history, key=lambda h: h["miou"])
    log(f"finetune: kept step {best['step']} (area error {best['area_mdape']}%); best mIoU was step "
        f"{best_iou['step']} ({best_iou['miou']})")
    registry._cache.pop(version, None)
    registry.register(c, version, model, parent=parent, label_set="client", pipeline="v2",
                      train_config=cfg.as_dict() | {"classes": CLIENT_CLASSES, "properties": len(images),
                                                    "selected_step": best["step"],
                                                    "best_miou_step": best_iou["step"]},
                      history=history)
    log(f"finetune: registered {version} as candidate")


def cmd_evaluate(c, version=None, pipeline=None, quiet=False):
    _threads()
    out = registry.run_benchmark(c, version, pipeline, REFERENCE_DATE)
    if not quiet:
        name = version or "traced masks (oracle)"
        o = out["report"]["overall"]
        log(f"evaluate: {name} with pipeline {out['report']['pipeline']} on {out['report']['properties']} "
            f"benchmark properties (run {out['id']})")
        for f, s in o.items():
            log(f"  {f:<18} median abs error {s['mdape']:>6.2f}%  bias {s['bias']:>+6.2f}%  p90 {s['p90']:>6.2f}%  "
                f"within 10% {s['within10']:>5.1f}%")
        if "miou" in out["report"]:
            log(f"  mIoU {out['report']['miou']}")
        if out["gate"] is not None:
            log(f"  release gate {'PASSED' if out['gate']['passed'] else 'FAILED'}")
            for ch in out["gate"]["checks"]:
                log(f"    [{'x' if ch['passed'] else ' '}] {ch['check']}: {ch['detail']}")
    return out


def print_status(st: dict) -> None:
    log("properties: " + ", ".join(f"{k} {v}" for k, v in st["properties"].items()))
    log("models:")
    for m in st["models"]:
        log(f"  {m['version']:<15} {m['status']:<11} label set {m['label_set']:<7} pipeline {m['pipeline']}")
    log("benchmark (median abs error / bias, %):")
    for b in st["benchmark"]:
        cells = "  ".join(f"{f.replace('_sqft', '').replace('_ft', '')} {b[f]['mdape']:.1f}/{b[f]['bias']:+.1f}"
                          for f in ("lawn_sqft", "bed_sqft", "pavement_sqft", "roof_sqft", "roof_perimeter_ft")
                          if f in b)
        extra = f"  flagged {b['review_share']}%" if b.get("review_share") else ""
        log(f"  {b['model']:<22} {b['pipeline']}  {cells}{extra}")
    if st["edit_rates"]:
        log("estimator edits on live measurements:")
        for e in st["edit_rates"]:
            log(f"  {e['model_version']:<15} {e['pipeline']}  {e['corrected']}/{e['measurements']} corrected "
                f"({e['corrected_pct']}%), {e['flagged_pct']}% flagged for review")


def bootstrap(c) -> None:
    migrate(c)
    seed(c, log=log)
    have = {m["version"] for m in registry.models(c)}
    if BASE_VERSION not in have:
        cmd_train_base(c)
    for pipe in ("v1", "v2"):
        if registry.latest_run(c, None, pipe) is None:
            cmd_evaluate(c, None, pipe)
    # the base model through both pipelines: the second run is the "fix the pipeline, don't retrain" ablation
    for pipe in ("v1", "v2"):
        if registry.latest_run(c, BASE_VERSION, pipe) is None:
            cmd_evaluate(c, BASE_VERSION, pipe)
    if FT_VERSION not in have:
        cmd_finetune(c)
    if registry.latest_run(c, FT_VERSION, "v2") is None:
        cmd_evaluate(c, FT_VERSION, "v2")
    if registry.model_row(c, FT_VERSION)["status"] == "candidate":
        log(f"promote: {json.dumps(registry.promote(c, FT_VERSION))}")
    if not c.execute("SELECT 1 FROM pm.measurements LIMIT 1").fetchone():
        service.simulate_estimators(c, BASE_VERSION, BEFORE_MONTH, log=log)
        service.simulate_estimators(c, FT_VERSION, AFTER_MONTH, log=log)
    print_status(service.status(c))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="propmeasure")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate")
    s = sub.add_parser("seed")
    for k, v in DEFAULT_SIZES.items():
        s.add_argument(f"--{k}", type=int, default=v)
    s.add_argument("--workers", type=int)
    s = sub.add_parser("train-base")
    s.add_argument("--version", default=BASE_VERSION)
    s.add_argument("--steps", type=int, default=BASE_STEPS)
    s = sub.add_parser("finetune")
    s.add_argument("--from", dest="parent", default=BASE_VERSION)
    s.add_argument("--version", default=FT_VERSION)
    s.add_argument("--steps", type=int, default=FT_STEPS)
    s = sub.add_parser("evaluate", help="score a model, or the traced masks with --oracle, on the benchmark")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--model")
    g.add_argument("--oracle", action="store_true")
    s.add_argument("--pipeline", choices=["v1", "v2"])
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("promote")
    s.add_argument("version")
    s = sub.add_parser("measure")
    s.add_argument("property_id")
    s.add_argument("--model")
    s.add_argument("--overlay", help="write the overlay PNG here")
    s = sub.add_parser("simulate-estimators")
    s.add_argument("--model", required=True)
    s.add_argument("--measured-on", type=date.fromisoformat, default=REFERENCE_DATE)
    s = sub.add_parser("label-queue")
    s.add_argument("--limit", type=int, default=20)
    s = sub.add_parser("status")
    s.add_argument("--json", action="store_true")
    sub.add_parser("bootstrap", help="migrate, seed, train, fine-tune, benchmark, promote, simulate a live month")
    a = p.parse_args(argv)

    with connect() as c:
        if a.cmd == "migrate":
            migrate(c)
            log("migrated")
        elif a.cmd == "seed":
            seed(c, {k: getattr(a, k) for k in DEFAULT_SIZES}, a.workers, log=log)
        elif a.cmd == "train-base":
            cmd_train_base(c, a.version, a.steps, production=registry.production(c) is None)
        elif a.cmd == "finetune":
            cmd_finetune(c, a.parent, a.version, a.steps)
        elif a.cmd == "evaluate":
            out = cmd_evaluate(c, a.model, a.pipeline, quiet=a.json)
            if a.json:
                print(json.dumps(out, default=str, indent=2))
        elif a.cmd == "promote":
            res = registry.promote(c, a.version)
            log(json.dumps(res))
            return 0 if res["promoted"] else 1
        elif a.cmd == "measure":
            res = service.measure_property(c, a.property_id, a.model, REFERENCE_DATE)
            print(json.dumps(res, indent=2, default=str))
            if a.overlay:
                inp = property_input(c, a.property_id)
                classes, proba = classify(registry.load(c, res["model_version"]), inp.image)
                m = measure(inp, classes, proba, res["pipeline"], REFERENCE_DATE)
                Path(a.overlay).write_bytes(render(inp.image, classes, m))
                log(f"overlay written to {a.overlay}")
        elif a.cmd == "simulate-estimators":
            service.simulate_estimators(c, a.model, a.measured_on, log=log)
        elif a.cmd == "label-queue":
            for r in service.label_queue(c, a.limit):
                log(f"{r['property_id']:<16} {r['market']:<14} largest change {r['largest_change']:.0%}  "
                    f"corrected: {', '.join(r['features'])}")
        elif a.cmd == "status":
            st = service.status(c)
            if a.json:
                print(json.dumps(st, default=str, indent=2))
            else:
                print_status(st)
        elif a.cmd == "bootstrap":
            bootstrap(c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
