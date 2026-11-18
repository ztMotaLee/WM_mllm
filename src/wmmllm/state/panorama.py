"""Deterministic multiresolution panorama state and immutable run-local assets.

Views sample the immutable base followed by ordered native-resolution patches,
then round once. Screen padding remains separate from the world textures.
"""
from __future__ import annotations

import hashlib
import io as _bytes_io
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from . import projection
from ..image_io import encode_rgb_png
from ..session.io import (_local, array_sha256, file_ref, pixel_sha256, read_json, resolve_run_path,
                          rgb_array, sha256_file, verify_ref, write_bytes_once, write_once)

MULTIRES_SCHEMA = "wmmllm.panorama.v1"
# Stable hash-domain identifier preserves the original canonical hash arithmetic.
_HASH_SCHEMA = "aiwm.w34.multires_panorama.v3_immutable_render_base"


def is_panorama_state(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get("schema") == MULTIRES_SCHEMA


def erp_array(value: Any) -> np.ndarray:
    """Return the ERP preview/base; native patches remain part of state authority."""
    return rgb_array(value["erp"] if is_panorama_state(value) else value)


def panorama_state_sha256(value: Any) -> str:
    if not is_panorama_state(value):
        return pixel_sha256(value)
    record = {"schema": _HASH_SCHEMA, "erp_pixel_sha256": pixel_sha256(value["erp"]),
              "immutable_render_base_pixel_sha256": pixel_sha256(value["render_base"]),
              "viewport_profile": value["viewport_profile"],
              "patches": [{"pixel_sha256": pixel_sha256(patch["pixels"]),
                           "alpha_sha256": array_sha256(patch["alpha"]),
                           "camera": patch["camera"], "role": patch["role"]}
                          for patch in value["patches"]]}
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def create_panorama_state(pano: Any, source: Any = None, *, width: int = 4096,
                          height: int = 2048, source_metadata: Any = None,
                          source_content_box: Any = None) -> dict[str, Any]:
    """Create a real multires panorama: ERP plus native geometric front texture.

    The source texture has a fixed world camera/frustum. Every rendered ray,
    at any output pose, projects into it through that camera. This preserves
    official source sampling without a special-case output frame or pose.
    No absent panorama is synthesized. Native textures are ordered overlays.
    """
    if source is None:
        array = erp_array(pano)
        if width != 2 * height or min(width, height) < 1:
            raise ValueError("ERP dimensions must satisfy W=2H")
        if array.shape[:2] != (height, width):
            array = np.asarray(Image.fromarray(array).resize((width, height), Image.Resampling.LANCZOS))
        patches = []
        profile = {"width": 1344, "height": 768, "content_box": [0, 0, 1344, 768],
                   "format": "fixed_screen_letterbox", "padding_ltrb": [0, 0, 0, 0]}
    else:
        full_source = rgb_array(source)
        profile = source_profile(source, source_metadata=source_metadata,
                                  source_content_box=source_content_box)
        left, top, right, bottom = profile["content_box"]
        source_array = np.ascontiguousarray(full_source[top:bottom, left:right])
        # The official source is the front view: yaw 0, pitch 0, 90-degree HFOV.
        array = lift_front(pano, source_array, width=width, height=height)
        patches = [{"pixels": source_array,
                    "alpha": np.ones(source_array.shape[:2], dtype=np.float32),
                    "camera": crop_camera(0, 0, 90, source_array.shape[1], source_array.shape[0]),
                    "role": "official_initial_native_front_texture"}]
    return {"schema": MULTIRES_SCHEMA, "erp": array, "patches": patches,
            "render_base": array,
            "viewport_profile": profile,
            "coordinate_convention": "canonical_u=.5+yaw/(2*pi)",
            "renderer": "ray_projected_multires_panorama",
            "pose_specific_override": False, "_asset_refs": {}}


def source_profile(source: Any, *, source_metadata: Any = None,
                    source_content_box: Any = None) -> dict[str, Any]:
    """Resolve fixed screen padding from explicit verified metadata or bounds.

    Black border checks validate declared padding, without inferring content.
    The session layer verifies any metadata/source references before this call.
    """
    pixels = rgb_array(source)
    height, width = pixels.shape[:2]
    if source_content_box is None and source_metadata is not None and "padding_ltrb" in source_metadata:
        left, top, right_pad, bottom_pad = map(int, source_metadata["padding_ltrb"])
        source_content_box = [left, top, width - right_pad, height - bottom_pad]
    if source_content_box is None:
        has_black_border = any(np.all(edge == 0) for edge in
                               (pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
        if has_black_border:
            raise ValueError("black outer source pixels require frozen padding metadata or explicit content bounds")
        source_content_box = [0, 0, width, height]
    if len(source_content_box) != 4 or any(int(x) != x for x in source_content_box):
        raise ValueError("source content box must have four integer edge coordinates")
    left, top, right, bottom = map(int, source_content_box)
    if not 0 <= left < right <= width or not 0 <= top < bottom <= height:
        raise ValueError("source content box is outside the source canvas")
    outside = np.ones((height, width), dtype=bool)
    outside[top:bottom, left:right] = False
    if np.any(pixels[outside] != 0):
        raise ValueError("declared frozen screen padding contains nonblack source pixels")
    profile = {"width": width, "height": height, "content_box": [left, top, right, bottom],
               "padding_ltrb": [left, top, width - right, height - bottom],
               "format": "fixed_screen_letterbox", "padding_is_world_texture": False,
               "padding_metadata_ref": None, "preprocess_ref": None}
    return profile


def profile_content_box(profile: Mapping[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = profile["content_box"]
    return (round(left * width / profile["width"]), round(top * height / profile["height"]),
            round(right * width / profile["width"]), round(bottom * height / profile["height"]))


def _project_patch(directions: np.ndarray, patch: Mapping[str, Any]):
    geom = projection
    lon, lat, tan_x, tan_y, width, height = geom._cam_params(patch["camera"])
    cx, cy, cz = geom._world_to_camera_components(
        directions[..., 0], directions[..., 1], directions[..., 2], lon, lat)
    with np.errstate(divide="ignore", invalid="ignore"):
        nx = cx / cz / tan_x
        ny = cy / cz / tan_y
    inside = ((cz > 1e-9) & np.isfinite(nx) & np.isfinite(ny) &
              (np.abs(nx) <= 1) & (np.abs(ny) <= 1))
    x = np.where(inside, (nx + 1) * .5 * width - .5, 0)
    y = np.where(inside, (1 - ny) * .5 * height - .5, 0)
    alpha_array = np.asarray(patch["alpha"], dtype=np.float32)
    alpha_height, alpha_width = alpha_array.shape
    ax = np.where(inside, (nx + 1) * .5 * alpha_width - .5, 0)
    ay = np.where(inside, (1 - ny) * .5 * alpha_height - .5, 0)
    if np.all(alpha_array == 1):
        alpha = np.ones_like(x, dtype=np.float64)
    else:
        alpha = _sample_xy(alpha_array, ax, ay)[..., 0]
    alpha = np.where(inside, np.clip(alpha, 0, 1), 0)
    # Stabilize exact mathematical mask plateaus against ray-roundtrip noise.
    alpha = np.where(alpha < 1e-12, 0, np.where(alpha > 1-1e-12, 1, alpha))
    return x, y, alpha


def _sample_xy(array: np.ndarray, x: np.ndarray, y: np.ndarray, *, wrap_x=False) -> np.ndarray:
    """Exact bilinear colours; gather only requested texels without base casts."""
    source = np.asarray(array)
    if source.ndim == 2:
        source = source[..., None]
    height, width = source.shape[:2]
    x = np.asarray(x, dtype=np.float64)
    y = np.clip(np.asarray(y, dtype=np.float64), 0, height - 1)
    if not wrap_x:
        x = np.clip(x, 0, width - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    x1, y1 = x0 + 1, np.minimum(y0 + 1, height - 1)
    if wrap_x:
        x0, x1 = x0 % width, x1 % width
    else:
        x1 = np.minimum(x1, width - 1)
    top = source[y0, x0] * (1 - fx) + source[y0, x1] * fx
    bottom = source[y1, x0] * (1 - fx) + source[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


@lru_cache(maxsize=16)
def _camera_grid(hfov: float, width: int, height: int):
    tan_x = math.tan(math.radians(hfov) / 2)
    tan_y = tan_x * height / width
    xs = ((np.arange(width) + .5) / width * 2 - 1) * tan_x
    ys = (1 - (np.arange(height) + .5) / height * 2) * tan_y
    return xs[None, :], ys[:, None]


def crop_camera(yaw: float = 0, pitch: float = 0, hfov: float = 90,
                width: int = 1344, height: int = 768) -> dict[str, Any]:
    """Convert a public horizontal-FOV camera to the app's crop camera."""
    yaw, pitch, hfov = float(yaw), float(pitch), float(hfov)
    if not all(math.isfinite(x) for x in (yaw, pitch, hfov)):
        raise ValueError("camera angles must be finite")
    if not -89.9 <= pitch <= 89.9 or not 0 < hfov <= 170:
        raise ValueError("pitch must be +/-89.9 and horizontal FOV in (0,170]")
    if int(width) != width or int(height) != height or min(width, height) < 1:
        raise ValueError("view dimensions must be positive integers")
    width, height = int(width), int(height)
    vfov = 2 * math.atan(math.tan(math.radians(hfov) / 2) * height / width)
    if vfov > math.radians(170):
        raise ValueError("vertical FOV exceeds the gnomonic camera limit")
    return {"center_lon": math.radians(yaw), "center_lat": math.radians(pitch),
            "fov_x": math.radians(hfov), "fov_y": vfov,
            "out_w": width, "out_h": height}


def _render_content_view(pano: Any, yaw: float = 0, pitch: float = 0, hfov: float = 90,
                width: int = 1344, height: int = 768) -> np.ndarray:
    """Render RGB uint8 perspective directly from the real ERP panorama.

    ERP pixel centres are u=(column+.5)/W and v=(row+.5)/H. Horizontal
    samples wrap; vertical samples clamp. Identical panorama and pose produce
    identical pixel hashes, including after lossless scene restoration.
    """
    array = rgb_array(pano["render_base"]) if is_panorama_state(pano) else erp_array(pano)
    geom = projection
    camera = crop_camera(yaw, pitch, hfov, width, height)
    x, y = _camera_grid(float(hfov), int(width), int(height))
    wx, wy, wz = geom._camera_to_world_components(x, y, np.ones((height, width)),
                                                camera["center_lon"], camera["center_lat"])
    wx, wy, wz = np.broadcast_arrays(wx, wy, wz)
    directions = np.stack((wx, wy, wz), axis=-1)
    mappings = []
    base_needed = np.ones((height, width), dtype=bool)
    if is_panorama_state(pano):
        for patch in pano["patches"]:
            px, py, alpha = _project_patch(directions, patch)
            mappings.append((patch, px, py, alpha))
            base_needed &= alpha < 1
    samples = np.zeros((height, width, 3), dtype=np.float64)
    if np.any(base_needed):
        u, v = geom.dir_to_uv(directions[base_needed])
        samples[base_needed] = _sample_xy(array, u * array.shape[1] - .5,
                                         v * array.shape[0] - .5, wrap_x=True)
    for patch, px, py, alpha in mappings:
        active = alpha > 0
        if not np.any(active):
            continue
        texture = _sample_xy(patch["pixels"], px[active], py[active])
        weight = alpha[active, None]
        samples[active] = texture * weight + samples[active] * (1 - weight)
    return np.clip(np.rint(samples), 0, 255).astype(np.uint8)


def render_view(pano: Any, yaw: float = 0, pitch: float = 0, hfov: float = 90,
                width: int = 1344, height: int = 768) -> np.ndarray:
    """Render every pose through the same official fixed-screen content window.

    Native source content is a world texture. Frozen letterbox padding is
    output formatting and remains screen fixed at all poses. There is no
    pose-specific or frame-specific branch.
    """
    if not is_panorama_state(pano):
        return _render_content_view(pano, yaw, pitch, hfov, width, height)
    left, top, right, bottom = profile_content_box(pano["viewport_profile"], width, height)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[top:bottom, left:right] = _render_content_view(
        pano, yaw, pitch, hfov, right - left, bottom - top)
    return canvas


def _support_alpha(support: Any, width: int, height: int) -> np.ndarray:
    if support is None:
        return np.ones((height, width), dtype=np.float32)
    alpha = np.asarray(support, dtype=np.float32)
    if alpha.ndim != 2 or not alpha.size or not np.isfinite(alpha).all():
        raise ValueError("support must be a finite nonempty 2D mask")
    if alpha.min() < 0 or alpha.max() > 255:
        raise ValueError("support values must be in [0,1] or [0,255]")
    if alpha.max() > 1:
        alpha = alpha / 255
    return alpha


def _composite_content_edit(pano: Any, edited: Any, yaw: float = 0,
                        pitch: float = 0, hfov: float = 90, support: Any = None):
    """Reproject a scoped perspective edit and preserve every zero-alpha pixel.

    Support is a 2D mask in [0,1] or [0,255]; it may have a different
    resolution, since its normalized frustum remains the same. ``None``
    selects the full edited perspective footprint.
    Returns (panorama, ERP float alpha, JSON metadata). Pixels outside ERP
    alpha>0 stay exactly unchanged.
    """
    array, edit = erp_array(pano), rgb_array(edited)
    height, width = edit.shape[:2]
    alpha = _support_alpha(support, width, height)
    result, erp_alpha = projection.reproject_crop_to_erp(
        array, edit, crop_camera(yaw, pitch, hfov, width, height), alpha)
    outside = erp_alpha == 0
    violations = int(np.count_nonzero(np.any(result[outside] != array[outside], axis=-1)))
    if violations:
        raise RuntimeError(f"scoped reprojection changed {violations} zero-alpha pixels")
    metadata = {"method": "app_ray_scoped_crop_reprojection",
                "coordinate_convention": "canonical_u=.5+yaw/(2*pi)",
                "camera_degrees": {"yaw": float(yaw), "pitch": float(pitch), "hfov": float(hfov)},
                "pano_size": [array.shape[1], array.shape[0]],
                "edited_size": [width, height],
                "erp_support_fraction": float(np.mean(erp_alpha > 0)),
                "outside_support_pixels": int(outside.sum()),
                "outside_support_changed_pixels": violations,
                "outside_support_max_error": 0,
                "input_pixel_sha256": pixel_sha256(array),
                "output_pixel_sha256": pixel_sha256(result)}
    if is_panorama_state(pano):
        updated = dict(pano)
        updated["erp"] = result
        updated["_asset_refs"] = dict(pano.get("_asset_refs", {}))
        # The patch frustum and support are geometric. Older native textures
        # remain losslessly stored underneath this later accepted edit layer.
        # Each output ray applies the same ordered alpha composition.
        patch = {"pixels": edit, "alpha": alpha.astype(np.float32),
                 "camera": crop_camera(yaw, pitch, hfov, width, height),
                 "role": "accepted_scoped_endpoint_edit_native_texture"}
        updated["patches"] = list(pano["patches"]) + [patch]
        metadata.update({"method": "multires_geometric_scoped_texture_overlay",
                         "native_prior_patches_unchanged": True,
                         "native_prior_patch_changed_pixels": 0,
                         "native_edit_support_nonzero_pixels": int(np.count_nonzero(alpha)),
                         "input_panorama_state_sha256": panorama_state_sha256(pano),
                         "output_panorama_state_sha256": panorama_state_sha256(updated)})
        result = updated
    return result, erp_alpha, metadata


def composite_view_edit(pano: Any, edited: Any, yaw: float = 0,
                        pitch: float = 0, hfov: float = 90, support: Any = None):
    """Strip fixed screen padding before geometric scoped edit reprojection."""
    image = rgb_array(edited)
    height, width = image.shape[:2]
    left, top, right, bottom = profile_content_box(pano["viewport_profile"], width, height)
    alpha = _support_alpha(support, width, height)
    if alpha.shape != (height, width):
        alpha = np.asarray(Image.fromarray(alpha, mode="F").resize((width, height), Image.Resampling.BILINEAR))
    content = np.ascontiguousarray(image[top:bottom, left:right])
    content_alpha = np.ascontiguousarray(alpha[top:bottom, left:right])
    result, erp_alpha, metadata = _composite_content_edit(
        pano, content, yaw, pitch, hfov, content_alpha)
    metadata.update({"viewport_profile": pano["viewport_profile"],
                     "edited_profile_size": [width, height],
                     "edited_content_box": [left, top, right, bottom],
                     "screen_padding_reprojected_as_world_texture": False})
    return result, erp_alpha, metadata


def lift_front(pano: Any, source: Any, *, width: int = 4096,
               height: int = 2048, yaw: float = 0, pitch: float = 0, hfov: float = 90):
    """Resize supplied ERP and inverse-project source into its front frustum.

    ``pano`` must be actual panorama content from the bootstrap image model.
    No missing panorama is synthesized. No source override occurs on rendering.
    """
    if int(width) != width or int(height) != height or width != 2 * height:
        raise ValueError("ERP dimensions must be positive integers with W=2H")
    if min(width, height) < 1:
        raise ValueError("ERP dimensions must be positive")
    pano_array, source_array = rgb_array(pano), rgb_array(source)
    if [pano_array.shape[1], pano_array.shape[0]] != [width, height]:
        pano_array = np.asarray(Image.fromarray(pano_array).resize(
            (int(width), int(height)), Image.Resampling.LANCZOS))
    return _composite_content_edit(pano_array, source_array, yaw=yaw, pitch=pitch, hfov=hfov)[0]


def image_error(reference: Any, image: Any) -> dict[str, Any]:
    ref, result = rgb_array(reference), rgb_array(image)
    if ref.shape != result.shape:
        raise ValueError("error measurement requires identical image shapes")
    error = np.abs(ref.astype(np.int16) - result.astype(np.int16))
    return {"mae_8bit": float(error.mean()), "max_8bit": int(error.max()),
            "p95_8bit": float(np.percentile(error, 95)),
            "p99_8bit": float(np.percentile(error, 99)),
            "exact_pixel_fraction": float(np.mean(np.all(error == 0, axis=-1))),
            "psnr_db": float(-10 * math.log10(float(np.mean(error.astype(np.float64) ** 2)) / 255 ** 2))
            if np.any(error) else None,
            "pixel_identical": bool(not np.any(error))}


def _persist_asset(state: dict[str, Any], directory: Path, kind: str,
                   value: np.ndarray, *, run_root: Path) -> dict[str, Any]:
    image_asset = kind == "rgb"
    array = rgb_array(value) if image_asset else np.asarray(value, dtype=np.float32)
    digest = pixel_sha256(array) if image_asset else array_sha256(array)
    asset_key = f"{kind}:{digest}"
    cache = state.setdefault("_asset_refs", {})
    cached = cache.get(asset_key)
    if cached is not None:
        # Reuse only verified assets in this run, never an adjacent cache.
        _read_asset(cached, kind, run_root=run_root)
        return dict(cached)
    path = _local(directory / (digest + (".png" if image_asset else ".npy")),
                     run_root=run_root, must_exist=False)
    if path.exists():
        ref = file_ref(path, run_root=run_root, image=image_asset)
        ref.update({"pixel_sha256" if image_asset else "array_sha256": digest,
                    "shape": list(array.shape)})
        if not image_asset:
            ref["dtype"] = str(array.dtype)
        _read_asset(ref, kind, run_root=run_root)
    else:
        if image_asset:
            payload, _conversion = encode_rgb_png(Image.fromarray(array))
        else:
            encoded = _bytes_io.BytesIO()
            np.save(encoded, array, allow_pickle=False)
            payload = encoded.getvalue()
        write_bytes_once(path, payload, run_root=run_root, image=image_asset)
        ref = file_ref(path, run_root=run_root, image=image_asset)
        ref.update({"pixel_sha256" if image_asset else "array_sha256": digest,
                    "shape": list(array.shape)})
        if not image_asset:
            ref["dtype"] = str(array.dtype)
        verify_ref(ref, run_root=run_root)
    cache[asset_key] = dict(ref)
    return ref


def save_panorama_state(state: dict[str, Any], directory: Path, *, run_root: Path) -> dict[str, Any]:
    """Publish an immutable manifest and content-addressed lossless assets.

    The preview, immutable render base, every ordered patch and float32 alpha
    retain separate encoded and decoded hashes. Unchanged assets are reused.
    """
    if not is_panorama_state(state):
        raise ValueError("save_panorama_state requires a multires panorama state")
    target = _local(directory, run_root=run_root, must_exist=False)
    pool = target / "panorama_assets"
    record = {"schema": MULTIRES_SCHEMA, "renderer": state["renderer"],
              "viewport_profile": state["viewport_profile"],
              "coordinate_convention": state["coordinate_convention"],
              "pose_specific_override": False,
              "erp": _persist_asset(state, pool, "rgb", state["erp"], run_root=run_root),
              "render_base": _persist_asset(state, pool, "rgb", state["render_base"], run_root=run_root),
              "patches": []}
    for patch in state["patches"]:
        record["patches"].append({
            "pixels": _persist_asset(state, pool, "rgb", patch["pixels"], run_root=run_root),
            "alpha": _persist_asset(state, pool, "alpha", patch["alpha"], run_root=run_root),
            "camera": patch["camera"], "role": patch["role"]})
    record["panorama_state_sha256"] = panorama_state_sha256(state)
    ref = write_once(target / "PANORAMA_STATE.json", record, run_root=run_root)
    return {**ref, "panorama_state_sha256": record["panorama_state_sha256"],
            "erp_preview_ref": record["erp"], "native_patch_count": len(record["patches"]),
            "schema": MULTIRES_SCHEMA}


def _read_asset(ref: Mapping[str, Any], kind: str, *, run_root: Path) -> np.ndarray:
    path = verify_ref(ref, run_root=run_root)
    if kind == "rgb":
        array = rgb_array(path)
        digest, expected = pixel_sha256(array), ref["pixel_sha256"]
    else:
        array = np.load(path, allow_pickle=False)
        if array.dtype != np.float32 or array.ndim != 2 or not array.size or not np.isfinite(array).all():
            raise ValueError("stored panorama alpha must be finite nonempty float32 HxW")
        if np.any(array < 0) or np.any(array > 1):
            raise ValueError("stored panorama alpha is outside [0,1]")
        digest, expected = array_sha256(array), ref["array_sha256"]
    if digest != expected or (ref.get("shape") is not None and list(array.shape) != ref["shape"]):
        raise ValueError("stored panorama asset pixel/array hash or shape changed")
    return array


def load_panorama_state(path: Path | Mapping[str, Any], *, run_root: Path) -> dict[str, Any]:
    """Restore the exact geometric state after validating every referenced asset."""
    ref = path if isinstance(path, Mapping) else {"path": path}
    resolved = (resolve_run_path(run_root, ref["path"]) if isinstance(path, Mapping)
                else _local(ref["path"], run_root=run_root))
    if ref.get("sha256") and sha256_file(resolved) != ref["sha256"]:
        raise ValueError("panorama state manifest byte hash changed")
    # Read by path to avoid verify_ref's canonical-state recursion.
    record = read_json(resolved, run_root=run_root)
    if record.get("schema") != MULTIRES_SCHEMA:
        raise ValueError("unsupported panorama state schema")
    erp = _read_asset(record["erp"], "rgb", run_root=run_root)
    render_base = _read_asset(record["render_base"], "rgb", run_root=run_root)
    cache = {f"rgb:{pixel_sha256(erp)}": record["erp"],
             f"rgb:{pixel_sha256(render_base)}": record["render_base"]}
    patches = []
    for entry in record["patches"]:
        pixels = _read_asset(entry["pixels"], "rgb", run_root=run_root)
        alpha = _read_asset(entry["alpha"], "alpha", run_root=run_root)
        cache[f"rgb:{pixel_sha256(pixels)}"] = entry["pixels"]
        cache[f"alpha:{array_sha256(alpha)}"] = entry["alpha"]
        patches.append({"pixels": pixels, "alpha": alpha,
                        "camera": entry["camera"], "role": entry["role"]})
    state = {"schema": MULTIRES_SCHEMA, "renderer": record["renderer"],
             "viewport_profile": record["viewport_profile"],
             "coordinate_convention": record["coordinate_convention"],
             "pose_specific_override": False, "erp": erp, "render_base": render_base,
             "patches": patches, "_asset_refs": cache}
    digest = panorama_state_sha256(state)
    if digest != record["panorama_state_sha256"] or (ref.get("panorama_state_sha256") and digest != ref["panorama_state_sha256"]):
        raise ValueError("restored panorama state hash changed")
    return state


def erp_outpaint_input(context: Any, *, hfov: float, width: int = 2048,
                       height: int = 1024):
    """Embed accepted wide native context before generating unknown directions."""
    canvas = np.full((height,width,3),127,dtype=np.uint8)
    canvas, alpha, meta = _composite_content_edit(canvas, context, hfov=hfov)
    mask = (alpha == 0).astype(np.uint8) * 255
    meta.update({'known_region':'wide perspective context including exact official front',
                 'mask_rule':'255 editable; 0 preserved',
                 'unknown_fraction':float(np.mean(mask > 0))})
    return canvas,mask,meta
