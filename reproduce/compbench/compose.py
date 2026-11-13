"""Historical CompBench composition, independent of the session engine.

Modified for portable explicit inputs. Pixel arithmetic and scope decisions
retain the historical functions. Diagnostic masks do not authorize any pixel.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageChops, ImageFilter
from scipy import ndimage
from .primitives import mean_rgb_abs_diff, thin_safe_noise_floor, general_fragments
from .inputs import RowInputs

STRONG_CHANGE_THRESHOLD = 25.0
SUPPORT_CC_MIN_PX = 64
TARGET_DILATE_PX = 7
PROTECTED_DILATE_PX = 3
FEATHER_PX = 3
EIGHT = np.ones((3, 3), dtype=bool)
SUPPORT_CEILING_S = 0.56

def dilate_mask(mask: np.ndarray, radius: int) -> Image.Image:
    image = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8))
    if radius > 0:
        image = image.filter(ImageFilter.MaxFilter(2 * radius + 1))
    return image


def _soft(mask: np.ndarray, dilate_px: int, feather_px: int,
          half: bool) -> np.ndarray:
    core = dilate_mask(mask, dilate_px)
    soft = core
    if feather_px > 0:
        radius = max(1, feather_px // 2) if half else feather_px
        soft = core.filter(ImageFilter.GaussianBlur(radius=radius))
        soft = ImageChops.lighter(soft, core)
    return np.asarray(soft, dtype=np.uint8)


def commit_alpha_v14(support: np.ndarray, *,
                     target_dilate_px: int = TARGET_DILATE_PX,
                     feather_px: int = FEATHER_PX) -> np.ndarray:
    """Build the target-side commit alpha without a protected-region term.

    The compatibility variant uses min(target_soft, 255 - protected_soft).
    This variant retains the full target-side alpha.
    """
    return _soft(support, target_dilate_px, feather_px, half=False)


def commit_alpha_v11_compat(target: np.ndarray, protected: np.ndarray, *,
                            target_dilate_px: int = TARGET_DILATE_PX,
                            protected_dilate_px: int = PROTECTED_DILATE_PX,
                            feather_px: int = FEATHER_PX) -> np.ndarray:
    """Frozen protected-region commit alpha for compatibility replay."""
    t = _soft(target, target_dilate_px, feather_px, half=False)
    if not bool(protected.any()):
        return t
    p = _soft(protected, protected_dilate_px, feather_px, half=True)
    return np.minimum(t, 255 - p).astype(np.uint8)


def locality_commit(source: np.ndarray, direct: np.ndarray,
                    alpha_u8: np.ndarray) -> np.ndarray:
    """PIL composite, so rounding is bit-identical to the frozen runs."""
    return np.asarray(Image.composite(
        Image.fromarray(direct), Image.fromarray(source),
        Image.fromarray(alpha_u8.astype(np.uint8), "L")), dtype=np.uint8)


def necessary_support(diff: np.ndarray):
    """Strong-change connected components of |Direct - source|.

    No protected carve, no envelope clip, no area cap, no adjacency test.
    Returns (support_mask, labels) where `labels` labels the reconstructed
    change map, so a caller can address individual components.
    """
    strong = diff > STRONG_CHANGE_THRESHOLD
    seed = thin_safe_noise_floor(strong)
    recon = ndimage.binary_propagation(seed, mask=strong, structure=EIGHT)
    labels, count = ndimage.label(recon, structure=EIGHT)
    if not count:
        return np.zeros_like(strong), labels
    sizes = ndimage.sum_labels(recon, labels, range(1, count + 1))
    keep = np.zeros(count + 1, dtype=bool)
    keep[1:] = sizes >= SUPPORT_CC_MIN_PX
    return keep[labels], labels


def audit_components(diff, core, protected):
    """THE GRADED CRITERION, imported unmodified (fixlib.general_fragments).

    general = non-protected-agnostic 8-connected components of
              open3(|diff|>25) & ~core, >= 64 px, within 24 px of core
    visible = those with protected_fraction < 0.5 and >= 300 px
    The same call is the fallback's legality test, so the gate and the
    measurement use the same implementation without a 24 px proxy.
    """
    gen = general_fragments(diff, core, protected)
    vis = [f for f in gen if f["nonprotected"] and f["size_px"] >= 300]
    return {
        "general_components": len(gen),
        "general_px": int(sum(f["size_px"] for f in gen)),
        "visible_fragments": len(vis),
        "visible_fragment_px": int(sum(f["size_px"] for f in vis)),
        "visible_detail": [{k: f[k] for k in
                            ("size_px", "min_dist", "protected_fraction",
                             "bbox")} for f in vis],
    }


def frozen_roi(base_target: np.ndarray, dilate_px: int = TARGET_DILATE_PX
               ) -> np.ndarray:
    """Arm-independent ROI from the stored base SAM target, dilated by 7 px.

    Computed once per row from the same frozen artifact for every arm. It
    does not depend on an arm's alpha or core. Fragment counts subtract each
    arm's own core, so this companion statistic uses a fixed region.
    """
    return np.asarray(dilate_mask(base_target, dilate_px), dtype=np.uint8) > 127


def roi_source_retention(output: np.ndarray, source: np.ndarray,
                         roi: np.ndarray) -> dict:
    """How much of the frozen ROI still LOOKS LIKE the source, per arm.

    On a removal row a high retention means the named object is still there in
    that output.  Arm-independent: the ROI is fixed, nothing is subtracted, and
    the statistic is read off the shipped raster.  Companion, NOT a verdict --
    on `replace` rows every arm is expected low, so it is comparable across
    arms on the same row and never absolute.
    """
    roi_px = int(roi.sum())
    if not roi_px:
        return {"roi_px": 0, "roi_source_retention": None,
                "roi_changed_px": 0}
    d = mean_rgb_abs_diff(output, source)
    kept = int(((d <= STRONG_CHANGE_THRESHOLD) & roi).sum())
    return {"roi_px": roi_px,
            "roi_source_retention": float(kept / roi_px),
            "roi_changed_px": roi_px - kept}


def drift_diagnostic(diff: np.ndarray, support: np.ndarray,
                     support_components_n: int) -> dict:
    """DIFFUSE-GLOBAL-DRIFT regime report.

    The criterion family is bistable at |diff| = 25: a uniform drift just under
    it produces NO support (the whole edit dropped, q = 1) and just over it
    produces ONE frame-sized component (the whole frame committed, q = 0).  A
    mixed drift produces a support shattered into many components whose
    dilation floods the frame.  No threshold is added here and none is tuned:
    the three statistics that separate the regime are REPORTED for every row so
    the regime is visible rather than silently absorbed.
    """
    frame = int(diff.size)
    strong = diff > STRONG_CHANGE_THRESHOLD
    mid = (diff > 10.0) & ~strong
    return {
        "strong_frame_fraction": float(strong.sum() / frame),
        "mid_band_frame_fraction": float(mid.sum() / frame),
        "support_components": int(support_components_n),
        "support_px_per_component": (float(support.sum() / support_components_n)
                                     if support_components_n else None),
        "mean_abs_diff": float(diff.mean()),
    }


def exceedance_report(support: np.ndarray, core: np.ndarray,
                      declared_box: np.ndarray) -> dict:
    """REPORTED, never gated on.

    How far the necessary support exceeds the box the grounding declared,
    and whether the excess is contiguous with it.  Every ratio is published
    alongside its absolute pixel count AND its frame fraction. A box ranges 3,276-263,112 px against a fixed 409,600 px
    frame, so a bare `/box` ratio hides the denominator.
    """
    box_px = int(declared_box.sum())
    outside = support & ~declared_box
    outside_px = int(outside.sum())
    contiguous_px = 0
    detached = 0
    if outside_px:
        lab, n = ndimage.label(support, structure=EIGHT)
        for i in range(1, n + 1):
            cc = lab == i
            if bool((cc & declared_box).any()):
                contiguous_px += int((cc & ~declared_box).sum())
            elif bool((cc & ~declared_box).any()):
                detached += 1
    frame_px = int(support.size)
    return {
        "declared_box_px": box_px,
        "frame_px": frame_px,
        "box_frame_fraction": float(box_px / frame_px),
        "support_px": int(support.sum()),
        "core_px": int(core.sum()),
        "support_frame_fraction": float(support.sum() / frame_px),
        "core_frame_fraction": float(core.sum() / frame_px),
        "outside_box_frame_fraction": float(outside_px / frame_px),
        "support_over_box": (int(support.sum()) / box_px) if box_px else None,
        "core_over_box": (int(core.sum()) / box_px) if box_px else None,
        "outside_box_over_box": (outside_px / box_px) if box_px else None,
        "outside_box_px": outside_px,
        "outside_box_contiguous_px": contiguous_px,
        "outside_box_detached_px": outside_px - contiguous_px,
        "outside_box_detached_components": detached,
        "contiguous_share_of_outside": (contiguous_px / outside_px)
                                       if outside_px else None,
    }


def compose_row(row: RowInputs) -> dict:
    """Full necessary support + its guarantee + the exceedance report.

    THE FULL SUPPORT IS ALWAYS WHAT IS PROPOSED AND ALWAYS WHAT SHIPS.  There
    is no pruning stage any more (see the selector): the pre-registered ceiling
    is a SCOPE CLASSIFIER, not a pixel editor.
    """
    frame_px = int(row.diff.size)
    support, labels = necessary_support(row.diff)
    n_components = int(len(
        [v for v in np.unique(np.where(support, labels, 0)) if v != 0]))
    alpha = commit_alpha_v14(support)
    core = alpha >= 128
    final = locality_commit(row.source, row.direct, alpha)
    guarded = alpha == 0
    s = float(support.sum() / frame_px)

    # ---- SCOPE, classified on the pixels, BEFORE any commit claim ----------
    # the reviewed build raised RuntimeError
    # on an empty guaranteed set, so a global edit shipped NOTHING -- it
    # crashed.  A crash is not a shipping outcome.  A row whose necessary
    # support runs away past the pre-registered ceiling, or whose guaranteed
    # set is empty, is now classified `out_of_scope_global_edit`: a LABELLED
    # terminal state, distinct from a commit AND from unguarded Direct, which
    # ships its artefacts, declares `ships_with_guarantee: False`, and is
    # counted OUT of the guarantee headline.  Not silently absorbed, not
    # crashed, not laundered as a commit.
    scope_reasons = []
    if s > SUPPORT_CEILING_S:
        scope_reasons.append(f"support_fraction {s:.4f} > ceiling "
                             f"{SUPPORT_CEILING_S} (pre-registered s rule)")
    if int(guarded.sum()) == 0:
        scope_reasons.append("guaranteed set is EMPTY: support covers the "
                             "frame, so a byte guarantee cannot be claimed")
    scope = "out_of_scope_global_edit" if scope_reasons else "in_scope"

    # FAIL-CLOSED byte guarantee.  Wherever alpha == 0 the composite must be
    # the source byte for byte.  This is the property the whole system exists
    # for; on a non-empty guarded set it raises rather than reports.  A
    # vacuously-empty guarded set is handled by the scope classifier above,
    # NOT by this invariant (which would pass vacuously).
    if not np.array_equal(final[guarded], row.source[guarded]):
        raise RuntimeError(f"Composition byte guarantee violated on {row.rid}")
    if not bool((support & ~core).sum() == 0):
        raise RuntimeError(f"Composition core does not contain support on {row.rid}")

    audit = audit_components(row.diff, core, row.protected)
    strong_px = int((row.diff > STRONG_CHANGE_THRESHOLD).sum())
    rec = {
        "row_id": row.rid, "split": row.split, "operation": row.operation,
        "instruction": row.instruction, "frame_px": frame_px,
        "source_path": str(row.source_path),
        "direct_path": str(row.direct_path),
        "source_sha256": row.source_sha256,
        "direct_sha256": row.direct_sha256,
        "strong_change_px": strong_px,
        "support_px": int(support.sum()),
        "support_fraction": s,
        # len(unique)-1 assumes a background label is
        # present; a full-frame support has none and would report one
        # component too few.
        "support_components": n_components,
        "core_px": int(core.sum()),
        "core_fraction": float(core.sum() / frame_px),
        "alpha_nonzero_px": int((alpha > 0).sum()),
        # the guarantee boundary is {alpha == 0},
        # NOT the complement of the binary support PNG.  dilate 7 + feather 3
        # write outside the support, so "outside support" overstates the
        # guaranteed set.  Both are reported and the gap is named.
        "guaranteed_px": int(guarded.sum()),
        "guaranteed_fraction": float(guarded.sum() / frame_px),
        "outside_support_px": int((~support).sum()),
        "nonguaranteed_outside_support_px": int((~support & (alpha > 0)).sum()),
        "fringe_px_0_lt_alpha_lt_128": int(((alpha > 0) & ~core).sum()),
        "scope": scope,
        "scope_reasons": scope_reasons,
        "support_ceiling_s": SUPPORT_CEILING_S,
        "strong_change_containment": (float((( row.diff >
            STRONG_CHANGE_THRESHOLD) & core).sum() / strong_px)
            if strong_px else None),
        "protected_px": int(row.protected.sum()),
        "protected_written_px": int((row.protected & core).sum()),
        "v11_committed": bool(row.v11_committed),
    }
    rec.update(audit)
    rec["exceedance"] = exceedance_report(support, core, row.declared_box)
    rec["drift"] = drift_diagnostic(row.diff, support, n_components)
    # arm-independent companion, measured on this arm's own composite inside a
    # frozen ROI that no arm can move
    roi = frozen_roi(row.base_target)
    rec["roi_companion"] = roi_source_retention(final, row.source, roi)
    rec["roi_companion"]["direct"] = roi_source_retention(
        row.direct, row.source, roi)["roi_source_retention"]
    # partition identity: guaranteed + core + fringe == frame, asserted not
    # assumed
    part = (rec["guaranteed_px"] + rec["core_px"]
            + rec["fringe_px_0_lt_alpha_lt_128"])
    if part != frame_px:
        raise RuntimeError(f"alpha partition broken on {row.rid}: "
                           f"{part} != {frame_px}")
    return rec, support, labels, alpha, final


def compose_pixels(source: np.ndarray, direct: np.ndarray):
    """Return support, labels, alpha and RGB composite from equal-sized RGB8 arrays."""
    source = np.asarray(source)
    direct = np.asarray(direct)
    if source.dtype != np.uint8 or direct.dtype != np.uint8:
        raise ValueError('composition requires uint8 RGB inputs')
    if source.ndim != 3 or source.shape[-1] != 3 or source.shape != direct.shape or not source.size:
        raise ValueError('composition requires matching nonempty HxWx3 inputs')
    support, labels = necessary_support(mean_rgb_abs_diff(source, direct))
    alpha = commit_alpha_v14(support)
    return support, labels, alpha, locality_commit(source, direct, alpha)
