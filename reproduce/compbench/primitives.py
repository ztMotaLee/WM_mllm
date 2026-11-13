"""Historical RGB change-map primitives and diagnostic helpers.

Modified for a portable release: plain local functions replace historical imports.
"""
from __future__ import annotations
import math
import numpy as np
from scipy import ndimage

OPENING_STRUCTURE = np.ones((3, 3), dtype=bool)
EIGHT = np.ones((3, 3), dtype=bool)
LINE_LENGTH = 5
GEN_THRESHOLD = 25.0
GEN_MIN_PX = 64
GEN_ADJACENCY_PX = 24.0

def mean_rgb_abs_diff(source_rgb: np.ndarray, direct_rgb: np.ndarray) -> np.ndarray:
    """mean_RGB_abs(source - Direct) as float32 HxW."""
    return np.abs(
        np.asarray(source_rgb, dtype=np.float32)
        - np.asarray(direct_rgb, dtype=np.float32)
    ).mean(axis=2)


def _line_structures(length: int = LINE_LENGTH) -> list[np.ndarray]:
    horizontal = np.zeros((1, length), dtype=bool)
    horizontal[0, :] = True
    diagonal = np.eye(length, dtype=bool)
    return [horizontal, horizontal.T.copy(), diagonal,
            np.fliplr(diagonal).copy()]


LINE_STRUCTURES = _line_structures()

def thin_safe_noise_floor(strong: np.ndarray) -> np.ndarray:
    """Opening by the union of a 3x3 square and 4 length-5 line elements."""
    kept = ndimage.binary_opening(strong, structure=OPENING_STRUCTURE)
    for structure in LINE_STRUCTURES:
        kept |= ndimage.binary_opening(strong, structure=structure)
    return kept


def norm_box_to_pixels(
    box: list[float], size: tuple[int, int]
) -> list[int]:
    if len(box) != 4:
        raise ValueError(f"invalid normalized box: {box}")
    width, height = size
    x0, y0, x1, y1 = [float(value) for value in box]
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"out-of-range normalized box: {box}")
    result = [
        max(0, min(width - 1, int(math.floor(x0 * width)))),
        max(0, min(height - 1, int(math.floor(y0 * height)))),
        max(1, min(width, int(math.ceil(x1 * width)))),
        max(1, min(height, int(math.ceil(y1 * height)))),
    ]
    if not (result[0] < result[2] and result[1] < result[3]):
        raise ValueError(f"degenerate pixel box: {result}")
    return result


def bool_box(size: tuple[int, int], box: list[int]) -> np.ndarray:
    width, height = size
    result = np.zeros((height, width), dtype=bool)
    x0, y0, x1, y1 = box
    result[y0:y1, x0:x1] = True
    return result


def authorised_region_from_grounding(grounding, size):
    """Union of the unpadded prompt boxes named by the grounding."""
    out = np.zeros((size[1], size[0]), dtype=bool)
    for box in grounding.get("edit_regions_norm") or []:
        out |= bool_box(size, norm_box_to_pixels(box, size))
    return out


def open3(mask: np.ndarray) -> np.ndarray:
    return ndimage.binary_opening(mask, structure=np.ones((3, 3)))


def general_fragments(diff, core, protected, *, min_px=GEN_MIN_PX,
                      adjacency_px=GEN_ADJACENCY_PX):
    """The GENERAL ungated criterion (the MEASUREMENT).  Unchanged."""
    strong = open3(diff > GEN_THRESHOLD)
    residual = strong & ~core
    out = []
    if not (residual.any() and core.any()):
        return out
    dist = ndimage.distance_transform_edt(~core)
    labels, count = ndimage.label(residual, structure=EIGHT)
    for lab in range(1, count + 1):
        cc = labels == lab
        size_px = int(cc.sum())
        if size_px < min_px:
            continue
        d = float(dist[cc].min())
        if d > adjacency_px:
            continue
        pf = float((cc & protected).sum() / size_px)
        ys, xs = np.where(cc)
        out.append({
            "size_px": size_px, "min_dist": d, "protected_fraction": pf,
            "nonprotected": pf < 0.5,
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1,
                     int(ys.max()) + 1],
            "_mask": cc,
        })
    return out

