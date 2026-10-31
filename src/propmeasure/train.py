"""Training loop for base pretraining and for the fine-tune.

The fine-tune keeps the pretrained encoder and decoder, starts a new class head, and trains the head at a higher
learning rate than the pretrained layers so the first few hundred steps don't wash out what they already know.
Checkpoints are chosen by area error on the validation properties, measured through the same pipeline the API uses,
because that is the number a quote depends on; mean IoU is logged next to it.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .model import UNet, normalize


@dataclass
class TrainConfig:
    steps: int = 900
    batch: int = 16
    crop: int = 64
    lr: float = 3e-3  # head, and every layer when training from scratch
    body_lr_scale: float = 1.0  # fine-tune: pretrained layers train at lr * body_lr_scale
    weight_decay: float = 1e-4
    dice_weight: float = 0.5
    color_jitter: float = 0.08
    eval_every: int = 150
    seed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def _batch(images: np.ndarray, labels: np.ndarray, rng: np.random.Generator, cfg: TrainConfig):
    n, h, w = labels.shape
    idx = rng.integers(0, n, size=cfg.batch)
    ys = rng.integers(0, h - cfg.crop + 1, size=cfg.batch)
    xs = rng.integers(0, w - cfg.crop + 1, size=cfg.batch)
    xb = np.empty((cfg.batch, cfg.crop, cfg.crop, 3), dtype=np.float32)
    yb = np.empty((cfg.batch, cfg.crop, cfg.crop), dtype=np.int64)
    for b in range(cfg.batch):
        img = images[idx[b], ys[b]:ys[b] + cfg.crop, xs[b]:xs[b] + cfg.crop]
        lab = labels[idx[b], ys[b]:ys[b] + cfg.crop, xs[b]:xs[b] + cfg.crop]
        k = int(rng.integers(4))
        img, lab = np.rot90(img, k), np.rot90(lab, k)
        if rng.random() < 0.5:
            img, lab = img[:, ::-1], lab[:, ::-1]
        gain = 1 + rng.uniform(-cfg.color_jitter, cfg.color_jitter, size=3)
        offset = rng.uniform(-8, 8, size=3)
        xb[b] = img * gain + offset
        yb[b] = lab
    x = normalize(torch.from_numpy(np.clip(xb, 0, 255)))
    return x, torch.from_numpy(yb)


def loss_fn(logits: torch.Tensor, target: torch.Tensor, dice_weight: float) -> torch.Tensor:
    ce = F.cross_entropy(logits, target)
    if dice_weight == 0:
        return ce
    probs = torch.softmax(logits, dim=1)
    onehot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()
    inter = (probs * onehot).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
    present = onehot.sum(dim=(0, 2, 3)) > 0
    dice = 1 - (2 * inter + 1) / (denom + 1)
    return ce + dice_weight * dice[present].mean()


def train(model: UNet, images: np.ndarray, labels: np.ndarray, cfg: TrainConfig,
          validate: Callable[[UNet], dict] | None = None, log: Callable[[str], None] = print):
    """Train in place; returns (best model by validation area error, history)."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    head = list(model.head.parameters())
    groups = [{"params": head, "lr": cfg.lr}, {"params": model.body_parameters(), "lr": cfg.lr * cfg.body_lr_scale}]
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    base_lrs = [g["lr"] for g in opt.param_groups]
    warmup = max(1, cfg.steps // 20)
    history: list[dict] = []
    best_state, best_score = None, math.inf
    t0 = time.time()
    running, since = 0.0, 0
    for step in range(1, cfg.steps + 1):
        if step <= warmup:
            scale = step / warmup
        else:
            scale = 0.5 * (1 + math.cos(math.pi * (step - warmup) / (cfg.steps - warmup)))
        for g, lr in zip(opt.param_groups, base_lrs, strict=True):
            g["lr"] = lr * scale
        model.train()
        x, y = _batch(images, labels, rng, cfg)
        loss = loss_fn(model(x), y, cfg.dice_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running += loss.item()
        since += 1
        if step % cfg.eval_every == 0 or step == cfg.steps:
            entry = {"step": step, "loss": round(running / since, 4), "seconds": round(time.time() - t0, 1)}
            running, since = 0.0, 0
            if validate is not None:
                entry.update(validate(model))
                score = entry["area_mdape"]
                if score < best_score:
                    best_score, best_state = score, copy.deepcopy(model.state_dict())
            history.append(entry)
            log(" ".join(f"{k}={v}" for k, v in entry.items()))
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history
