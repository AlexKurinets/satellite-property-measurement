"""Side-by-side PNG of a capture and what the pipeline measured on it, for estimators and for review."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import geo
from .classes import BED, CANOPY, CLASS_COLORS, LAWN, PAVEMENT, ROOF
from .measure import Measurement

SCALE = 3


def _poly(draw, pts, color, width=2, dash=False):
    pts = [(float(x) * SCALE, float(y) * SCALE) for x, y in pts]
    pts.append(pts[0])
    if not dash:
        draw.line(pts, fill=color, width=width)
        return
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:], strict=True):
        n = max(1, int(np.hypot(x1 - x0, y1 - y0) // 8))
        for k in range(0, n, 2):
            a, b = k / n, min(1.0, (k + 1) / n)
            draw.line([(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a), (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)],
                      fill=color, width=width)


def _legend(draw, y, pipeline):
    font = ImageFont.load_default(size=13)
    items = [(CLASS_COLORS[LAWN][:3], "lawn counted"), (CLASS_COLORS[ROOF][:3], "roof"),
             (CLASS_COLORS[BED][:3], "bed"), (CLASS_COLORS[PAVEMENT][:3], "paving"),
             (CLASS_COLORS[CANOPY][:3], "canopy not counted")]
    x = 6
    for color, text in items:
        draw.rectangle([x, y + 1, x + 13, y + 14], fill=color)
        draw.text((x + 18, y), text, fill=(40, 40, 40), font=font)
        x += 26 + int(draw.textlength(text, font=font))
    lines = [((255, 80, 80), "parcel as delivered (dashed)"), ((255, 255, 255), "parcel used")]
    if pipeline == "v2":
        lines.append(((255, 220, 0), "roof outline"))
    for color, text in lines:
        draw.line([x - 1, y + 8, x + 19, y + 8], fill=(90, 90, 90), width=5)
        draw.line([x, y + 8, x + 18, y + 8], fill=color, width=3)
        draw.text((x + 24, y), text, fill=(40, 40, 40), font=font)
        x += 32 + int(draw.textlength(text, font=font))


def render(image: np.ndarray, classes: np.ndarray, m: Measurement) -> bytes:
    h, w = classes.shape
    base = Image.fromarray(image).resize((w * SCALE, h * SCALE), Image.NEAREST).convert("RGBA")
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    for k, color in CLASS_COLORS.items():
        layer[classes == k] = color
    # what was counted: lawn (including under canopy and the frontage strip) and the selected roof
    if m.lawn_mask is not None:
        layer[m.lawn_mask] = CLASS_COLORS[LAWN][:3] + (150,)
    if m.roof_mask is not None:
        layer[m.roof_mask] = CLASS_COLORS[ROOF][:3] + (160,)
    over = base.copy()
    over.alpha_composite(Image.fromarray(layer, "RGBA").resize(base.size, Image.NEAREST))
    draw = ImageDraw.Draw(over)
    if m.parcel_px is not None:
        _poly(draw, m.parcel_px, (255, 80, 80, 255), 2, dash=True)
    if m.snapped_px is not None:
        _poly(draw, m.snapped_px, (255, 255, 255, 255), 2)
    if m.roof_mask is not None and m.roof_mask.any() and m.pipeline == "v2":
        for p in geo.outline_polygons(m.roof_mask):
            _poly(draw, p, (255, 220, 0, 255), 2)
    legend_h = 30
    out = Image.new("RGB", (base.width * 2 + 8, base.height + legend_h), (255, 255, 255))
    out.paste(base.convert("RGB"), (0, 0))
    out.paste(over.convert("RGB"), (base.width + 8, 0))
    _legend(ImageDraw.Draw(out), base.height + 8, m.pipeline)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()
