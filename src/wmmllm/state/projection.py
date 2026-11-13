"""CPU panorama rays, wrap-safe samplers, crop reprojection and depth navigation.

ERP longitude is ``(u-.5)*2*pi``; latitude is ``(.5-v)*pi``. Crop
camera angles and the depth-forward yaw are radians. Public view poses in
``panorama`` use degrees. Navigation requires explicit real radial depth.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import numpy as np
from PIL import Image, ImageFilter

TWO_PI = 2.0 * math.pi
_CENTER_LAT_CLAMP = math.radians(89.9)
_EPS_Z = 1.0e-9
STRETCH_THRESHOLD_JMIN_MULTIPLE = 1.10


def uv_to_lonlat(u: Any, v: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert ERP UV coordinates to longitude/latitude radians."""

    u_arr, v_arr = np.broadcast_arrays(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64))
    lon = (u_arr - 0.5) * TWO_PI
    lat = (0.5 - v_arr) * math.pi
    return lon, lat


def lonlat_to_uv(lon: Any, lat: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert longitude/latitude radians to ERP UV coordinates."""

    lon_arr, lat_arr = np.broadcast_arrays(
        np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    )
    u = np.mod(lon_arr / TWO_PI + 0.5, 1.0)
    v = 0.5 - lat_arr / math.pi
    return u, v


def lonlat_to_dir(lon: Any, lat: Any) -> np.ndarray:
    """Convert longitude/latitude radians to y-up right-handed directions."""

    lon_arr, lat_arr = np.broadcast_arrays(
        np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    )
    cos_lat = np.cos(lat_arr)
    x = cos_lat * np.sin(lon_arr)
    y = np.sin(lat_arr)
    z = cos_lat * np.cos(lon_arr)
    return np.stack((x, y, z), axis=-1)


def uv_to_dir(u: Any, v: Any) -> np.ndarray:
    """Convert ERP UV coordinates to direction(s).

    Scalars return shape ``(3,)``. Array inputs return shape ``(..., 3)``.
    """

    lon, lat = uv_to_lonlat(u, v)
    return lonlat_to_dir(lon, lat)


def dir_to_lonlat(d: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert direction vector(s) to longitude/latitude radians."""

    arr = np.asarray(d, dtype=np.float64)
    if arr.shape == ():
        raise ValueError("d must have shape (3,) or (..., 3)")
    if arr.shape[-1] != 3:
        raise ValueError(f"d must have shape (3,) or (..., 3), got {arr.shape}")

    x = arr[..., 0]
    y = arr[..., 1]
    z = arr[..., 2]
    norm = np.linalg.norm(arr, axis=-1)
    if np.any(norm <= 0.0):
        raise ValueError("direction vectors must be non-zero")

    lon = np.arctan2(x, z)
    lat = np.arcsin(np.clip(y / norm, -1.0, 1.0))
    return lon, lat


def dir_to_uv(d: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert direction vector(s) to ERP UV coordinates.

    ``u`` is always wrapped into ``[0, 1)``. Inputs may be shape ``(3,)`` or
    ``(..., 3)``; returned ``u`` and ``v`` match the input's leading shape.
    """

    lon, lat = dir_to_lonlat(d)
    return lonlat_to_uv(lon, lat)


def _camera_to_world_components(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    yaw: float,
    pitch: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply pitch around camera x, then yaw around world y.

    The pitch sign is chosen so positive pitch maps the center ray toward
    positive world y (higher latitude).
    """

    cp = math.cos(pitch)
    sp = math.sin(pitch)
    py = y * cp + z * sp
    pz = -y * sp + z * cp
    px = x

    cy = math.cos(yaw)
    sy = math.sin(yaw)
    wx = px * cy + pz * sy
    wy = py
    wz = -px * sy + pz * cy
    return wx, wy, wz


def _world_to_camera_components(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    yaw: float,
    pitch: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of ``_camera_to_world_components``."""

    cy = math.cos(yaw)
    sy = math.sin(yaw)
    ux = x * cy - z * sy
    uy = y
    uz = x * sy + z * cy

    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cx = ux
    cy_cam = uy * cp - uz * sp
    cz = uy * sp + uz * cp
    return cx, cy_cam, cz


def _bilinear_sample_erp(image_arr: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Sample an ERP image with horizontal wrap and vertical clamp."""

    src_h, src_w = image_arr.shape[:2]
    channels = 1 if image_arr.ndim == 2 else image_arr.shape[2]
    pixels = image_arr.reshape(src_h, src_w, channels).astype(np.float64)

    x = np.mod(u, 1.0) * float(src_w)
    y = np.clip(v, 0.0, 1.0) * float(src_h)
    y = np.clip(y, 0.0, float(src_h - 1))

    x_floor = np.floor(x).astype(np.int64)
    y_floor = np.floor(y).astype(np.int64)
    x0 = np.mod(x_floor, src_w)
    x1 = np.mod(x_floor + 1, src_w)
    y0 = np.clip(y_floor, 0, src_h - 1)
    y1 = np.clip(y_floor + 1, 0, src_h - 1)
    tx = (x - x_floor)[..., None]
    ty = (y - y_floor)[..., None]

    c00 = pixels[y0, x0]
    c10 = pixels[y0, x1]
    c01 = pixels[y1, x0]
    c11 = pixels[y1, x1]
    top = c00 * (1.0 - tx) + c10 * tx
    bottom = c01 * (1.0 - tx) + c11 * tx
    out = top * (1.0 - ty) + bottom * ty
    # Round (not truncate) so resampling round-trips stay within +-0.5 LSB.
    out = np.clip(np.rint(out), 0.0, 255.0).astype(np.uint8)
    if image_arr.ndim == 2:
        return out[..., 0]
    return out


def _bilinear_sample_xy_clamped(image_arr: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinearly sample a plain (non-wrapping) image at float pixel coords.

    Coordinates are clamped to the valid range on both axes. Returns float64
    with shape ``x.shape + (channels,)`` (no quantization — callers blend in
    float and quantize once).
    """

    src_h, src_w = image_arr.shape[:2]
    channels = 1 if image_arr.ndim == 2 else image_arr.shape[2]
    pixels = image_arr.reshape(src_h, src_w, channels).astype(np.float64)

    xc = np.clip(np.asarray(x, dtype=np.float64), 0.0, float(src_w - 1))
    yc = np.clip(np.asarray(y, dtype=np.float64), 0.0, float(src_h - 1))

    x0 = np.floor(xc).astype(np.int64)
    y0 = np.floor(yc).astype(np.int64)
    x1 = np.minimum(x0 + 1, src_w - 1)
    y1 = np.minimum(y0 + 1, src_h - 1)
    tx = (xc - x0)[..., None]
    ty = (yc - y0)[..., None]

    c00 = pixels[y0, x0]
    c10 = pixels[y0, x1]
    c01 = pixels[y1, x0]
    c11 = pixels[y1, x1]
    top = c00 * (1.0 - tx) + c10 * tx
    bottom = c01 * (1.0 - tx) + c11 * tx
    return top * (1.0 - ty) + bottom * ty


def _cam_params(cam: Mapping[str, Any]) -> tuple[float, float, float, float, int, int]:
    center_lon = float(cam["center_lon"])
    center_lat = float(min(max(float(cam["center_lat"]), -_CENTER_LAT_CLAMP), _CENTER_LAT_CLAMP))
    fov_x = float(cam["fov_x"])
    fov_y = float(cam["fov_y"])
    if not (0.0 < fov_x < math.pi and 0.0 < fov_y < math.pi):
        raise ValueError("crop_cam fov_x/fov_y must be in the open interval (0, pi)")
    out_w = int(cam["out_w"])
    out_h = int(cam["out_h"])
    if out_w <= 0 or out_h <= 0:
        raise ValueError("crop_cam out_w/out_h must be positive")
    return center_lon, center_lat, math.tan(fov_x / 2.0), math.tan(fov_y / 2.0), out_w, out_h


def _crop_pixel_erp_uv(cam: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """ERP UV coordinates of every crop pixel center (forward ray cast)."""

    center_lon, center_lat, tan_x, tan_y, out_w, out_h = _cam_params(cam)
    xs = ((np.arange(out_w, dtype=np.float64) + 0.5) / out_w * 2.0 - 1.0) * tan_x
    ys = (1.0 - (np.arange(out_h, dtype=np.float64) + 0.5) / out_h * 2.0) * tan_y
    xx, yy = np.meshgrid(xs, ys)
    zz = np.ones_like(xx)
    wx, wy, wz = _camera_to_world_components(xx, yy, zz, center_lon, center_lat)
    return dir_to_uv(np.stack((wx, wy, wz), axis=-1))


def render_gnomonic_crop(erp_img: np.ndarray, cam: Mapping[str, Any]) -> np.ndarray:
    """Render a rectilinear (gnomonic) crop of an ERP image. Wrap-safe."""

    u, v = _crop_pixel_erp_uv(cam)
    return _bilinear_sample_erp(np.asarray(erp_img), u, v)


def erp_mask_to_crop_mask(
    erp_mask: np.ndarray,
    cam: Mapping[str, Any],
    *,
    threshold: int = 127,
) -> np.ndarray:
    """Project an ERP-space binary mask into crop space (uint8, 0/255)."""

    mask_arr = np.asarray(erp_mask)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]
    u, v = _crop_pixel_erp_uv(cam)
    sampled = _bilinear_sample_erp(mask_arr.astype(np.uint8), u, v)
    return (sampled.astype(np.int64) >= int(threshold)).astype(np.uint8) * 255


def reproject_crop_to_erp(
    erp_img: np.ndarray,
    edited_crop: np.ndarray,
    cam: Mapping[str, Any],
    blend_feather_px: int = 16,
    crop_alpha: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Paste an edited gnomonic crop back into the ERP panorama.

    Every ERP pixel ray is projected into the crop camera; pixels inside the
    crop frustum sample the edited crop bilinearly, pixels outside stay EXACT
    identity. ``blend_feather_px`` linearly feathers alpha within that many
    crop pixels of the crop border (0 = hard edge). Wrap-safe and pole-safe
    because the inverse path operates through rays, never ERP x-spans.

    ``crop_alpha`` (optional): a crop-space blend-weight map (H, W) — e.g. the
    scope alpha of a region edit. It is bilinearly sampled per ERP pixel and
    MULTIPLIED into the footprint alpha, so ERP pixels whose crop_alpha is 0
    keep EXACT identity even inside the crop footprint (no resampling blur on
    unedited pixels). Accepts uint8 (0-255) or float (0-1) arrays of any
    resolution (normalized frustum coords are resolution independent).

    Returns ``(result_erp_uint8, blend_alpha_float)`` where alpha has
    shape ``(erp_h, erp_w)`` in ``[0, 1]`` (footprint x crop_alpha).
    """

    erp_arr = np.asarray(erp_img)
    erp_h, erp_w = erp_arr.shape[:2]
    channels = 1 if erp_arr.ndim == 2 else erp_arr.shape[2]
    erp_f = erp_arr.reshape(erp_h, erp_w, channels).astype(np.float64)

    crop_arr = np.asarray(edited_crop)
    crop_ch = 1 if crop_arr.ndim == 2 else crop_arr.shape[2]
    crop_f = crop_arr.reshape(crop_arr.shape[0], crop_arr.shape[1], crop_ch)
    if crop_ch != channels:
        if crop_ch > channels:
            crop_f = crop_f[:, :, :channels]
        elif crop_ch == 1:
            crop_f = np.repeat(crop_f, channels, axis=2)
        else:
            raise ValueError(
                f"edited_crop has {crop_ch} channels but erp_img has {channels}"
            )
    # The editor may return the crop at a different resolution; normalized
    # frustum coords are resolution independent, so use the ACTUAL crop size.
    ec_h, ec_w = crop_f.shape[:2]

    ca_f = None
    if crop_alpha is not None:
        ca_f = np.asarray(crop_alpha, dtype=np.float64)
        if ca_f.ndim == 3:
            ca_f = ca_f[..., 0]
        if ca_f.ndim != 2 or ca_f.shape[0] < 1 or ca_f.shape[1] < 1:
            raise ValueError("crop_alpha must be a 2D (or HxWx1) array")
        if ca_f.max() > 1.0:
            ca_f = ca_f / 255.0
        ca_f = np.clip(ca_f, 0.0, 1.0)[..., None]

    center_lon, center_lat, tan_x, tan_y, _, _ = _cam_params(cam)
    feather = float(blend_feather_px)

    result = erp_f.copy()
    alpha = np.zeros((erp_h, erp_w), dtype=np.float64)
    u_cols = (np.arange(erp_w, dtype=np.float64) + 0.5) / float(erp_w)

    chunk_rows = max(1, min(256, erp_h))
    for row0 in range(0, erp_h, chunk_rows):
        row1 = min(row0 + chunk_rows, erp_h)
        v_rows = (np.arange(row0, row1, dtype=np.float64) + 0.5) / float(erp_h)
        uu, vv = np.meshgrid(u_cols, v_rows)
        dirs = uv_to_dir(uu, vv)
        cam_x, cam_y, cam_z = _world_to_camera_components(
            dirs[..., 0], dirs[..., 1], dirs[..., 2], center_lon, center_lat
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            nx = (cam_x / cam_z) / tan_x
            ny = (cam_y / cam_z) / tan_y
        inside = (
            (cam_z > _EPS_Z)
            & np.isfinite(nx)
            & np.isfinite(ny)
            & (np.abs(nx) <= 1.0)
            & (np.abs(ny) <= 1.0)
        )
        if not np.any(inside):
            continue

        crop_x = np.where(inside, (nx + 1.0) * 0.5 * ec_w - 0.5, 0.0)
        crop_y = np.where(inside, (1.0 - ny) * 0.5 * ec_h - 0.5, 0.0)
        sample = _bilinear_sample_xy_clamped(crop_f, crop_x, crop_y)

        if feather > 0.0:
            dist = np.minimum.reduce(
                [crop_x + 0.5, ec_w - 0.5 - crop_x, crop_y + 0.5, ec_h - 0.5 - crop_y]
            )
            a = np.clip(dist / feather, 0.0, 1.0)
        else:
            a = np.ones_like(crop_x)
        a = np.where(inside, a, 0.0)

        if ca_f is not None:
            ca_h, ca_w = ca_f.shape[:2]
            ax = np.where(inside, (nx + 1.0) * 0.5 * ca_w - 0.5, 0.0)
            ay = np.where(inside, (1.0 - ny) * 0.5 * ca_h - 0.5, 0.0)
            a = a * _bilinear_sample_xy_clamped(ca_f, ax, ay)[..., 0]

        blend = a[..., None]
        result[row0:row1] = sample * blend + result[row0:row1] * (1.0 - blend)
        alpha[row0:row1] = a

    out = np.clip(np.rint(result), 0.0, 255.0).astype(np.uint8)
    return out.reshape(erp_arr.shape), alpha


def _bilinear_sample(arr: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Sample ``arr`` (H, W[, C]) at fractional UV. u wraps, v clamps."""
    a = np.asarray(arr)
    single = a.ndim == 2
    if single:
        a = a[:, :, None]
    h, w = a.shape[:2]
    src = a.astype(np.float32)

    x = np.mod(np.asarray(u, dtype=np.float64), 1.0) * w - 0.5
    y = np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0) * h - 0.5

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0).astype(np.float32)[..., None]
    fy = (y - y0).astype(np.float32)[..., None]

    x0m = np.mod(x0, w)
    x1m = np.mod(x0 + 1, w)
    y0m = np.clip(y0, 0, h - 1)
    y1m = np.clip(y0 + 1, 0, h - 1)

    p00 = src[y0m, x0m]
    p01 = src[y0m, x1m]
    p10 = src[y1m, x0m]
    p11 = src[y1m, x1m]

    top = p00 * (1.0 - fx) + p01 * fx
    bot = p10 * (1.0 - fx) + p11 * fx
    out = top * (1.0 - fy) + bot * fy
    return out[:, :, 0] if single else out


def _wrap_delta(d: np.ndarray) -> np.ndarray:
    """Map a difference of u coordinates into [-0.5, 0.5)."""
    return (np.asarray(d) + 0.5) % 1.0 - 0.5


def _blend_wrap_seam(image: Image.Image, band_px: int) -> Image.Image:
    """Make the two ERP edge samples agree with a narrow circular taper."""
    band = max(0, min(int(band_px), image.size[0] // 4))
    if band == 0:
        return image
    arr = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    left = arr[:, :band]
    right = arr[:, -band:][:, ::-1]
    t = (np.arange(band, dtype=np.float32) + 0.5) / float(band)
    weight = (1.0 - t * t * (3.0 - 2.0 * t))[None, :, None]
    adjustment = 0.5 * (left - right) * weight
    arr[:, :band] = left - adjustment
    arr[:, -band:] = (right + adjustment)[:, ::-1]
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), "RGB")


class ReprojectResult:
    """Warped panorama + the hole mask the move revealed."""

    __slots__ = ("image", "hole_mask", "stats", "depth_map")

    def __init__(
        self,
        image: Image.Image,
        hole_mask: Image.Image,
        stats: dict,
        depth_map: Optional[np.ndarray] = None,
    ):
        self.image = image
        self.hole_mask = hole_mask
        self.stats = stats
        self.depth_map = depth_map


def _raycast_radial_depth_surface(
    depth: np.ndarray,
    *,
    dx: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
    cx: float,
    cz: float,
    iterations: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect destination rays with a source-camera radial-depth surface.

    The source depth map defines the surface ``|P| = D(P / |P|)``.  For each
    ray ``P(t) = C_new + t*d`` we solve that equation by vectorized bisection.
    This keeps foreground silhouettes and upper-scene parallax tied to the
    same 3D surface instead of sampling depth once at the unshifted pixel.
    """
    surface = np.asarray(depth, dtype=np.float32)
    camera_radius = math.hypot(float(cx), float(cz))
    upper = float(np.max(surface)) + 2.0 * camera_radius + 1.0
    lo = np.zeros_like(dx, dtype=np.float32)
    hi = np.full_like(dx, upper, dtype=np.float32)

    for _ in range(max(4, int(iterations))):
        mid = (lo + hi) * 0.5
        qx = cx + mid * dx
        qy = mid * dy
        qz = cz + mid * dz
        radius = np.sqrt(qx * qx + qy * qy + qz * qz)
        safe_radius = np.maximum(radius, 1e-7)
        lon = np.arctan2(qx, qz)
        lat = np.arcsin(np.clip(qy / safe_radius, -1.0, 1.0))
        source_u = np.mod(lon / TWO_PI + 0.5, 1.0)
        source_v = 0.5 - lat / math.pi
        source_depth = _bilinear_sample(surface, source_u, source_v)
        outside = radius >= source_depth
        hi = np.where(outside, mid, hi)
        lo = np.where(outside, lo, mid)

    distance = (lo + hi) * 0.5
    qx = cx + distance * dx
    qy = distance * dy
    qz = cz + distance * dz
    radius = np.sqrt(qx * qx + qy * qy + qz * qz)
    safe_radius = np.maximum(radius, 1e-7)
    source_u = np.mod(np.arctan2(qx, qz) / TWO_PI + 0.5, 1.0)
    source_v = 0.5 - np.arcsin(
        np.clip(qy / safe_radius, -1.0, 1.0)
    ) / math.pi
    return distance.astype(np.float32), source_u, source_v


def min_area_ratio_for_step(step: float) -> float:
    """Closed form for the SMALLEST solid-angle ratio this warp can produce at
    a given step (it occurs at the new nadir):

        J_min = ((sqrt(a^2 + 4) - a) / 2)^3,  a = step / eye height

    A fixed stretch threshold is therefore almost inert at small steps and
    swallows the frame at large ones: J_min is 0.33 at a=0.75 but 0.047 at
    a=2.5. The auto threshold below is expressed as a multiple of J_min so the
    mask keeps flagging the same worst-magnified region at every step.
    (Verified against the measured `area_ratio_p01`: 0.344 vs 0.333 at a=0.75.)
    """
    a = abs(float(step))
    return float(((math.sqrt(a * a + 4.0) - a) / 2.0) ** 3)


def reproject_forward(
    src: Image.Image,
    *,
    step: float,
    yaw: float = 0.0,
    stretch_threshold: Optional[float] = None,
    back_wedge_deg: float = 60.0,
    back_wedge_lat_deg: float = 28.0,
    dilate_px: int = 10,
    feather_px: int = 14,
    depth_map: Optional[np.ndarray | Image.Image] = None,
    far_depth_eye_heights: float = 8.0,
    seam_blend_px: Optional[int] = None,
) -> ReprojectResult:
    """Reproject ``src`` to a camera translated ``step`` eye heights forward.

    ``yaw`` is the walking direction in radians using the repo's ERP
    convention (``lon = (u - 0.5) * 2*pi``), so ``yaw = 0`` walks toward the
    centre column of the panorama.

    ``stretch_threshold`` is a SOLID-ANGLE ratio: destination pixels whose
    source solid angle per destination solid angle falls below it are pixels
    where a tiny source patch is being smeared over a large destination area,
    i.e. where the warp is inventing detail rather than moving it. ``None``
    (the default) scales it with the step via ``min_area_ratio_for_step``.

    ``depth_map`` is radial depth from the source camera in eye-height units.
    It may be an array or a floating-point PIL image and is resized to the ERP
    grid. Invalid values fall back to ``far_depth_eye_heights``. Destination
    rays are intersected with that radial surface by vectorized bisection.
    Large translations still need a layered representation for surfaces that
    were occluded in the source view.
    """
    depth = np.asarray(depth_map, dtype=np.float32) if depth_map is not None else None
    expected_shape = (src.height, src.width)
    if depth is None or depth.shape != expected_shape or not np.isfinite(depth).all() or np.any(depth <= 0.05):
        raise ValueError("depth_map requires exact-shape finite radial depth above 0.05")
    if not math.isfinite(float(step)) or not math.isfinite(float(yaw)):
        raise ValueError("step and yaw must be finite")
    if stretch_threshold is None:
        stretch_threshold = (
            STRETCH_THRESHOLD_JMIN_MULTIPLE * min_area_ratio_for_step(step)
            if abs(float(step)) > 1e-9 else 0.0
        )
    rgb = src.convert("RGB")
    w, h = rgb.size
    arr = np.asarray(rgb, dtype=np.uint8)
    far_depth = float(far_depth_eye_heights)
    if not math.isfinite(far_depth) or far_depth <= 0.25:
        raise ValueError("far_depth_eye_heights must be finite and > 0.25")

    # destination pixel centres -> destination ray directions
    uu = (np.arange(w, dtype=np.float64) + 0.5) / w
    vv = (np.arange(h, dtype=np.float64) + 0.5) / h
    u_d, v_d = np.meshgrid(uu, vv)
    lon_d = (u_d - 0.5) * TWO_PI
    lat_d = (0.5 - v_d) * math.pi
    cos_lat_d = np.cos(lat_d)
    dx = cos_lat_d * np.sin(lon_d)
    dy = np.sin(lat_d)
    dz = cos_lat_d * np.cos(lon_d)

    # camera translation, eye height = 1, ground plane y = -1
    step = float(step)
    cx = step * math.sin(yaw)
    cz = step * math.cos(yaw)

    below = dy < -1e-9
    output_depth: Optional[np.ndarray] = None
    if depth_map is None:
        # The old implementation used t=0 above the horizon, which made that
        # whole half of the panorama an exact identity while the ground was
        # translated. A finite shell gives every scene ray a valid 3D point.
        ground_t = np.where(below, 1.0 / np.where(below, -dy, 1.0), far_depth)
        t = np.minimum(ground_t, far_depth)
        depth_source = "ground_plane_capped_by_finite_scene_shell"
    else:
        raw_depth = (
            np.asarray(depth_map, dtype=np.float32)
            if not isinstance(depth_map, Image.Image)
            else np.asarray(depth_map, dtype=np.float32)
        )
        if raw_depth.ndim != 2:
            raise ValueError(
                f"depth_map must be HxW, got shape {raw_depth.shape}"
            )
        if raw_depth.shape != (h, w):
            raw_depth = np.asarray(
                Image.fromarray(raw_depth, mode="F").resize(
                    (w, h), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )
        valid_depth = np.isfinite(raw_depth) & (raw_depth > 0.05)
        surface_depth = np.where(valid_depth, raw_depth, far_depth)
        surface_depth = np.clip(surface_depth, 0.05, far_depth).astype(np.float32)
        t, u_s, v_s = _raycast_radial_depth_surface(
            surface_depth,
            dx=dx,
            dy=dy,
            dz=dz,
            cx=cx,
            cz=cz,
        )
        output_depth = t
        depth_source = "per_pixel_radial_depth"

    if depth_map is None:
        # source direction = (C_new + t*d) - C_old, with C_old at the origin.
        sx = cx + t * dx
        sy = t * dy
        sz = cz + t * dz
        norm = np.sqrt(sx * sx + sy * sy + sz * sz)
        norm = np.where(norm < 1e-12, 1.0, norm)
        sx, sy, sz = sx / norm, sy / norm, sz / norm

        lat_s = np.arcsin(np.clip(sy, -1.0, 1.0))
        lon_s = np.arctan2(sx, sz)
        u_s = np.mod(lon_s / TWO_PI + 0.5, 1.0)
        v_s = 0.5 - lat_s / math.pi
    else:
        lat_s = (0.5 - v_s) * math.pi

    warped = _bilinear_sample(arr, u_s, v_s)
    warped_img = Image.fromarray(np.clip(warped, 0, 255).astype(np.uint8), "RGB")
    if seam_blend_px is None:
        seam_blend_px = max(2, min(16, w // 128))
    if abs(step) > 1e-9:
        warped_img = _blend_wrap_seam(warped_img, seam_blend_px)

    # ---- solid-angle Jacobian of the inverse map -------------------------- #
    # d(u_s, v_s) / d(u_d, v_d) by central differences; u differences are
    # unwrapped so the seam column is not a false discontinuity.
    du_s_du = _wrap_delta(np.roll(u_s, -1, axis=1) - np.roll(u_s, 1, axis=1)) / 2.0
    dv_s_du = (np.roll(v_s, -1, axis=1) - np.roll(v_s, 1, axis=1)) / 2.0
    du_s_dv = _wrap_delta(u_s[2:, :] - u_s[:-2, :]) / 2.0
    dv_s_dv = (v_s[2:, :] - v_s[:-2, :]) / 2.0
    du_s_dv = np.vstack([du_s_dv[:1], du_s_dv, du_s_dv[-1:]])
    dv_s_dv = np.vstack([dv_s_dv[:1], dv_s_dv, dv_s_dv[-1:]])

    # differences above are per PIXEL INDEX; d(u_d) = 1/w and d(v_d) = 1/h per
    # index, so det(d(u_s,v_s)/d(u_d,v_d)) = w*h * det(per-index). The identity
    # map then gives exactly 1.0.
    det = float(w * h) * np.abs(du_s_du * dv_s_dv - du_s_dv * dv_s_du)
    # ERP solid angle element is proportional to cos(lat) du dv.
    cos_s = np.clip(np.cos(lat_s), 1e-4, None)
    cos_d = np.clip(cos_lat_d, 1e-4, None)
    area_ratio = det * (cos_s / cos_d)

    stretch_hole = area_ratio < float(stretch_threshold)
    # the bottom pole row of an ERP is a single point; never trust it.
    stretch_hole[:2, :] = False
    stretch_hole[-2:, :] = True if step > 0 else False

    # ---- the wedge behind the viewer -------------------------------------- #
    back_lon = np.abs(_wrap_delta((lon_d - (yaw + math.pi)) / TWO_PI)) * TWO_PI
    wedge = (back_lon <= math.radians(back_wedge_deg) / 2.0) & (
        np.abs(lat_d) <= math.radians(back_wedge_lat_deg)
    )
    if step <= 0:
        wedge[:] = False

    hole = stretch_hole | wedge
    hole_u8 = (hole.astype(np.uint8)) * 255
    mask_img = Image.fromarray(hole_u8, "L")

    # dilate then feather: the inpainter must own a margin around the hole so
    # the mask outline does not print itself into the result.
    if dilate_px > 0:
        # MaxFilter needs an odd kernel; wrap the seam by tiling horizontally.
        k = int(dilate_px) * 2 + 1
        pad = k
        tiled = np.hstack([hole_u8[:, -pad:], hole_u8, hole_u8[:, :pad]])
        grown = Image.fromarray(tiled, "L").filter(ImageFilter.MaxFilter(k))
        mask_img = Image.fromarray(np.asarray(grown)[:, pad:pad + w], "L")
    if feather_px > 0:
        pad = int(feather_px) * 3
        m = np.asarray(mask_img)
        tiled = np.hstack([m[:, -pad:], m, m[:, :pad]])
        blur = Image.fromarray(tiled, "L").filter(
            ImageFilter.GaussianBlur(float(feather_px))
        )
        mask_img = Image.fromarray(np.asarray(blur)[:, pad:pad + w], "L")

    final_mask = np.asarray(mask_img)
    stats = {
        "step_eye_heights": round(step, 4),
        "yaw_rad": round(float(yaw), 5),
        "hole_fraction_raw": round(float(hole.mean()), 5),
        "hole_fraction_dilated": round(float((final_mask > 8).mean()), 5),
        "wedge_fraction": round(float(wedge.mean()), 5),
        "stretch_fraction": round(float(stretch_hole.mean()), 5),
        "stretch_threshold": float(stretch_threshold),
        "area_ratio_p01": round(float(np.percentile(area_ratio, 1)), 5),
        "area_ratio_median": round(float(np.median(area_ratio)), 5),
        "back_wedge_deg": float(back_wedge_deg),
        "back_wedge_lat_deg": float(back_wedge_lat_deg),
        "dilate_px": int(dilate_px),
        "feather_px": int(feather_px),
        "seam_blend_px": int(seam_blend_px),
        "model": depth_source,
        "depth_source": depth_source,
        "far_depth_eye_heights": round(far_depth, 4),
        "depth_min_eye_heights": round(float(np.min(t)), 4),
        "depth_median_eye_heights": round(float(np.median(t)), 4),
        "depth_max_eye_heights": round(float(np.max(t)), 4),
    }
    if output_depth is not None:
        stats["depth_raycast_iterations"] = 12
        stats["source_depth_min_eye_heights"] = round(
            float(np.min(surface_depth)), 4
        )
        stats["source_depth_median_eye_heights"] = round(
            float(np.median(surface_depth)), 4
        )
        stats["source_depth_max_eye_heights"] = round(
            float(np.max(surface_depth)), 4
        )
    return ReprojectResult(warped_img, mask_img, stats, output_depth)


def wrap_continuity_ratio(img: Image.Image) -> float:
    """Column difference at the wrap join over the typical column difference
    in the same frame. A genuine 360 panorama reads about 1.00; a frame whose
    left and right edges do not belong together reads well above it."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    if arr.shape[1] < 4:
        return float("nan")
    join = float(np.mean(np.abs(arr[:, 0, :] - arr[:, -1, :])))
    interior = float(np.mean(np.abs(arr[:, 1:, :] - arr[:, :-1, :])))
    if interior <= 1e-6:
        return float("nan")
    return round(join / interior, 4)


def hole_fill_instruction(scene_prompt: str = "", travel_hint: str = "") -> str:
    """The image model is NOT asked to move the camera — the camera has already
    moved. It is asked to complete the pixels the move exposed."""
    where = (travel_hint or "").strip()
    scene = (scene_prompt or "").strip()
    lines = [
        "This panorama has ALREADY been reprojected to a viewpoint a few steps "
        "forward; the camera move is done and must not be repeated.",
        "Only the transparent/marked regions are missing: ground newly walked "
        "onto near the bottom, small gaps beside near objects, and the wedge "
        "directly behind the viewer.",
        "Fill those regions so they continue the surrounding pixels exactly: "
        "same surfaces, same materials, same perspective, same lighting and "
        "shadow direction, same colour grade.",
        "Behind the viewer, show what the walker has just passed, receding.",
        "Do NOT move the viewpoint, do not re-render the scene, do not add new "
        "landmarks or people, and do not change anything outside the marked "
        "regions.",
    ]
    if scene:
        lines.append(f"The place is: {scene[:280]}.")
    if where:
        lines.append(f"The walk went toward {where[:160]}.")
    lines.append("Return one continuous 2:1 equirectangular 360 panorama.")
    return " ".join(lines)


def perspective_view(
    img: Image.Image,
    *,
    lon: float = 0.0,
    lat: float = 0.0,
    fov_deg: float = 110.0,
    out_w: int = 960,
    out_h: int = 640,
) -> Image.Image:
    """Rectilinear view of an ERP panorama, ``fov_deg`` horizontal."""
    # The crop renderer is defined in this module.

    fov_x = math.radians(float(fov_deg))
    # keep pixels square: match the vertical FOV to the output aspect
    fov_y = 2.0 * math.atan(math.tan(fov_x / 2.0) * (float(out_h) / float(out_w)))
    cam: Mapping[str, Any] = {
        "center_lon": float(lon),
        "center_lat": float(lat),
        "fov_x": fov_x,
        "fov_y": fov_y,
        "out_w": int(out_w),
        "out_h": int(out_h),
    }
    arr = render_gnomonic_crop(np.asarray(img.convert("RGB")), cam)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
