from __future__ import annotations

import io
import math
from typing import Literal

from PIL import Image, ImageEnhance, ImageFilter

CorruptionKind = Literal["blur", "jpeg", "low_light", "occlusion"]


def apply_corruption(
    image: Image.Image, kind: CorruptionKind, severity: float
) -> Image.Image:
    result = image.convert("RGB")
    if kind == "blur":
        return result.filter(ImageFilter.GaussianBlur(radius=float(severity)))
    if kind == "jpeg":
        buffer = io.BytesIO()
        result.save(buffer, format="JPEG", quality=int(severity))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if kind == "low_light":
        return ImageEnhance.Brightness(result).enhance(float(severity))
    if kind == "occlusion":
        fraction = float(severity)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("Occlusion fraction must be between 0 and 1")
        side = int(math.sqrt(fraction) * min(result.size))
        left = (result.width - side) // 2
        top = (result.height - side) // 2
        result.paste((0, 0, 0), (left, top, left + side, top + side))
        return result
    raise ValueError(f"Unknown corruption kind: {kind}")
