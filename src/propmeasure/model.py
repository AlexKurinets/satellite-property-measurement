"""U-Net segmentation model and the head swap used for fine-tuning.

A U-Net decodes back to full resolution through skip connections. At 30 cm a model that predicts at a quarter of the
input resolution and upsamples smears every boundary over about a meter, and on a 4,000 sq ft lawn a one-pixel
boundary error is already about 6% of the area, so full-resolution output matters more here than a bigger encoder.
Production used an ImageNet-pretrained ResNet-34 encoder; this repo uses the same layout at a width that trains on a
CPU in minutes.
"""

from __future__ import annotations

import io

import torch
from torch import nn
from torch.nn import functional as F

IMAGE_MEAN = torch.tensor([118.0, 122.0, 104.0]).view(1, 3, 1, 1)
IMAGE_STD = torch.tensor([42.0, 38.0, 40.0]).view(1, 3, 1, 1)


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, n_classes: int, widths: tuple[int, ...] = (16, 32, 64, 128)):
        super().__init__()
        self.widths = tuple(widths)
        self.down = nn.ModuleList()
        cin = 3
        for w in widths:
            self.down.append(_block(cin, w))
            cin = w
        self.up = nn.ModuleList()
        for w_skip in reversed(widths[:-1]):
            self.up.append(_block(cin + w_skip, w_skip))
            cin = w_skip
        self.head = nn.Conv2d(cin, n_classes, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.down):
            x = block(x)
            if i < len(self.down) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for block, skip in zip(self.up, reversed(skips), strict=True):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x)

    def body_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("head.")]


def normalize(images_uint8: torch.Tensor) -> torch.Tensor:
    """(B, H, W, 3) uint8 to normalized (B, 3, H, W) float."""
    x = images_uint8.permute(0, 3, 1, 2).float()
    return (x - IMAGE_MEAN) / IMAGE_STD


def swap_head(model: UNet, n_classes: int, seed: int = 0) -> UNet:
    """Keep the pretrained encoder and decoder, start a new class head for the client's label set."""
    torch.manual_seed(seed)
    model.head = nn.Conv2d(model.head.in_channels, n_classes, 1)
    return model


def to_bytes(model: UNet) -> bytes:
    buf = io.BytesIO()
    torch.save({"widths": model.widths, "n_classes": model.head.out_channels, "state": model.state_dict()}, buf)
    return buf.getvalue()


def from_bytes(data: bytes) -> UNet:
    blob = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    model = UNet(blob["n_classes"], tuple(blob["widths"]))
    model.load_state_dict(blob["state"])
    model.eval()
    return model


@torch.no_grad()
def predict_proba(model: UNet, image) -> torch.Tensor:
    """Class probabilities (C, H, W) for one (H, W, 3) uint8 image, padded to a multiple of 8 and cropped back."""
    x = normalize(torch.as_tensor(image).unsqueeze(0))
    h, w = x.shape[-2:]
    ph, pw = (-h) % 8, (-w) % 8
    x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    model.eval()
    logits = model(x)[0, :, :h, :w]
    return torch.softmax(logits, dim=0)
