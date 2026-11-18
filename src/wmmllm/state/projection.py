"""CPU panorama rays, wrap-safe samplers and crop reprojection.

ERP longitude is ``(u-.5)*2*pi``; latitude is ``(.5-v)*pi``. Crop camera
angles are radians. Public view poses in ``panorama`` use degrees.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

TWO_PI = 2.0 * math.pi
_CENTER_LAT_CLAMP = math.radians(89.9)
_EPS_Z = 1.0e-9


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



def reproject_crop_to_erp(
    erp_img: np.ndarray,
    edited_crop: np.ndarray,
    cam: Mapping[str, Any],
    crop_alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Paste an RGB gnomonic crop into the RGB ERP through rays, weighted by crop_alpha in [0,1].

    Returns the uint8 ERP and the (erp_h, erp_w) blend alpha; zero-alpha pixels stay exact."""
    erp_arr = np.asarray(erp_img)
    erp_h, erp_w = erp_arr.shape[:2]
    erp_f = erp_arr.astype(np.float64)
    crop_f = np.asarray(edited_crop)
    # Normalized frustum coords are resolution independent; use the actual crop size.
    ec_h, ec_w = crop_f.shape[:2]
    ca_f = np.asarray(crop_alpha, np.float64)[..., None]

    center_lon, center_lat, tan_x, tan_y, _, _ = _cam_params(cam)

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
        ca_h, ca_w = ca_f.shape[:2]
        ax = np.where(inside, (nx + 1.0) * 0.5 * ca_w - 0.5, 0.0)
        ay = np.where(inside, (1.0 - ny) * 0.5 * ca_h - 0.5, 0.0)
        a = np.where(inside, _bilinear_sample_xy_clamped(ca_f, ax, ay)[..., 0], 0.0)

        blend = a[..., None]
        result[row0:row1] = sample * blend + result[row0:row1] * (1.0 - blend)
        alpha[row0:row1] = a

    out = np.clip(np.rint(result), 0.0, 255.0).astype(np.uint8)
    return out, alpha
