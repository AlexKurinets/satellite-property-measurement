import numpy as np
import pytest
import torch

from propmeasure import evaluate
from propmeasure.model import UNet, from_bytes, predict_proba, swap_head, to_bytes
from propmeasure.synth import make_scene
from propmeasure.train import TrainConfig, train


def test_unet_returns_full_resolution_logits():
    model = UNet(6, (8, 16, 32))
    assert model(torch.zeros(2, 3, 96, 96)).shape == (2, 6, 96, 96)
    proba = predict_proba(model, np.zeros((190, 190, 3), np.uint8))
    assert proba.shape == (6, 190, 190)
    assert torch.allclose(proba.sum(0), torch.ones(190, 190), atol=1e-5)


def test_swap_head_keeps_the_pretrained_body():
    model = UNet(6, (8, 16, 32))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    old_head = model.head.weight.clone()
    swap_head(model, 6, seed=3)
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before if not k.startswith("head."))
    assert not torch.equal(old_head, model.head.weight)


def test_weights_round_trip():
    model = UNet(6, (8, 16, 32)).eval()
    clone = from_bytes(to_bytes(model))
    x = torch.randn(1, 3, 32, 32)
    assert torch.allclose(model(x), clone(x))


def test_short_training_run_lowers_the_loss():
    scenes = [make_scene(90_000 + i, "base") for i in range(4)]
    images = np.stack([s.image for s in scenes])
    labels = np.stack([s.label for s in scenes])
    _, history = train(UNet(6, (8, 16, 32)), images, labels, TrainConfig(steps=60, batch=4, crop=64, eval_every=20),
                       log=lambda m: None)
    assert history[-1]["loss"] < history[0]["loss"]


def test_stats():
    s = evaluate._stats([0.05, -0.02, 0.20, -0.10, 0.01])
    assert s == {"n": 5, "mdape": 5.0, "bias": 1.0, "p90": 16.0, "within10": 80.0}


def _report(lawn, roof, by_market):
    def st(m, b=0.0):
        return {"n": 10, "mdape": m, "bias": b, "p90": m * 2, "within10": 50.0}
    return {"overall": {"lawn_sqft": st(*lawn), "roof_sqft": st(*roof)},
            "by": {"market": {k: {"lawn_sqft": st(v[0]), "roof_sqft": st(v[1])} for k, v in by_market.items()}}}


CFG = {"gated_features": ["lawn_sqft", "roof_sqft"], "max_mdape": {"lawn_sqft": 8, "roof_sqft": 6},
       "max_abs_bias": 3, "max_market_regression": 2}


def test_gate_passes_a_better_candidate():
    prod = _report((17, -10), (14, -12), {"a": (15, 13), "b": (19, 15)})
    cand = _report((6, 1), (5, -1), {"a": (5, 4), "b": (7, 6)})
    assert evaluate.gate(cand, prod, CFG)["passed"]


def test_gate_blocks_a_market_regression_hidden_by_a_better_average():
    prod = _report((9, 0), (7, 0), {"a": (4, 4), "b": (14, 10)})
    cand = _report((6, 0), (5, 0), {"a": (7, 5), "b": (5, 5)})
    res = evaluate.gate(cand, prod, CFG)
    assert not res["passed"]
    assert [c["check"] for c in res["checks"] if not c["passed"]] == [
        "lawn_sqft in a no worse than production by 2 pts"]


@pytest.mark.parametrize("bias", [3.5, -4.0])
def test_gate_blocks_bias(bias):
    cand = _report((6, bias), (5, 0), {"a": (6, 5)})
    assert not evaluate.gate(cand, None, CFG)["passed"]
