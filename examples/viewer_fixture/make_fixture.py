# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The WM^MLLM Authors
"""Create a small, authored procedural playback fixture without model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image


PANORAMA_SIZE = (1024, 512)
VIEW_SIZE = (480, 270)
NUM_FRAMES = 24
FPS = 12
PROFILE = {
    "width": VIEW_SIZE[0], "height": VIEW_SIZE[1],
    "content_box": [0, 0, *VIEW_SIZE], "padding_ltrb": [0, 0, 0, 0],
}


def pixel_sha256(pixels: np.ndarray) -> str:
    """Hash decoded RGB pixels using the session image-reference convention."""
    array = np.ascontiguousarray(pixels, dtype=np.uint8)
    header = f"RGB uint8 {array.shape[1]} {array.shape[0]}\n".encode("ascii")
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode()


def _file_ref(root: Path, path: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _write_json(root: Path, relative: str, value: object) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_json_bytes(value))
    return _file_ref(root, path)


def _write_png(root: Path, relative: str, pixels: np.ndarray) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.ascontiguousarray(pixels, dtype=np.uint8))
    with path.open("xb") as handle:
        image.save(handle, format="PNG", optimize=True)
    return {**_file_ref(root, path), "pixel_sha256": pixel_sha256(pixels),
            "size": list(image.size)}


def _panorama_rays() -> np.ndarray:
    width, height = PANORAMA_SIZE
    longitude = ((np.arange(width) + 0.5) / width - 0.5) * 2 * np.pi
    latitude = (0.5 - (np.arange(height) + 0.5) / height) * np.pi
    lon, lat = np.meshgrid(longitude, latitude)
    return np.stack((np.cos(lat) * np.sin(lon), np.sin(lat),
                     np.cos(lat) * np.cos(lon)), axis=-1)


class _Scene:
    """Small analytic ray caster for authored boxes and rounded pottery/plants."""

    def __init__(self) -> None:
        self.rays = _panorama_rays()
        self.eye = np.array([0.0, 1.6, 0.0])
        self.depth = np.full(self.rays.shape[:2], np.inf)
        sky = np.clip((self.rays[..., 1] + 0.05) / 0.9, 0, 1)[..., None]
        self.colors = np.array([229, 237, 222]) * (1 - sky) + np.array([130, 190, 210]) * sky
        self.light = np.array([-0.36, 0.85, -0.38])

    def _paint(self, distance: np.ndarray, normal: np.ndarray,
               color: tuple | np.ndarray, valid: np.ndarray) -> None:
        visible = valid & (distance > 0.0001) & (distance < self.depth)
        shade = 0.73 + 0.27 * np.maximum(np.sum(normal * self.light, axis=-1), 0)
        shaded = np.asarray(color) * shade[..., None]
        self.colors[visible] = np.broadcast_to(shaded, self.colors.shape)[visible]
        self.depth[visible] = distance[visible]

    def box(self, low: tuple, high: tuple, color: tuple, *, room: bool = False) -> None:
        near = (np.asarray(low) - self.eye) / self.rays
        far = (np.asarray(high) - self.eye) / self.rays
        lower, upper = np.minimum(near, far), np.maximum(near, far)
        entry, leave = lower.max(axis=-1), upper.min(axis=-1)
        inside = entry <= 0
        distance = np.where(inside, leave, entry)
        axis = np.where(inside, upper.argmin(axis=-1), lower.argmax(axis=-1))
        normal = np.zeros_like(self.rays)
        signs = np.take_along_axis(np.sign(self.rays), axis[..., None], axis=-1)[..., 0]
        np.put_along_axis(normal, axis[..., None], (signs * np.where(inside, 1, -1))[..., None], axis=-1)
        if room:
            normal *= -1
            point = self.eye + distance[..., None] * self.rays
            colors = np.broadcast_to(np.array(color, dtype=float), self.colors.shape).copy()
            floor = point[..., 1] < 0.01
            ceiling = point[..., 1] > high[1] - 0.01
            colors[floor] = [192, 159, 119]
            planks = (np.mod(point[..., 0] + 4, 0.55) < 0.018)
            joints = (np.mod(point[..., 2] + np.floor((point[..., 0] + 4) / 0.55) % 2, 2.2) < 0.016)
            colors[floor & (planks | joints)] = [164, 135, 102]
            colors[ceiling] = [251, 245, 225]
            colors[(point[..., 0] < low[0] + 0.01) & ~floor & ~ceiling] = [183, 209, 196]
            color = colors
        self._paint(distance, normal, color, leave >= np.maximum(entry, 0))

    def ellipsoid(self, center: tuple, radius: tuple, color: tuple) -> None:
        radius = np.asarray(radius)
        relative = (self.eye - np.asarray(center)) / radius
        directions = self.rays / radius
        a = np.sum(directions * directions, axis=-1)
        b = np.sum(directions * relative, axis=-1)
        c = float(np.sum(relative * relative) - 1)
        discriminant = b * b - a * c
        distance = (-b - np.sqrt(np.maximum(discriminant, 0))) / a
        point = self.eye + distance[..., None] * self.rays
        normal = (point - np.asarray(center)) / (radius * radius)
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
        self._paint(distance, normal, color, discriminant >= 0)

    def ground(self) -> None:
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = -self.eye[1] / self.rays[..., 1]
        point = self.eye + distance[..., None] * self.rays
        normal = np.zeros_like(self.rays)
        normal[..., 1] = 1
        colors = np.broadcast_to(np.array([218., 211., 183.]), self.colors.shape).copy()
        lines = ((np.mod(point[..., 0], 0.8) < 0.025)
                 | (np.mod(point[..., 2], 0.8) < 0.025))
        colors[lines] = [183, 181, 159]
        self._paint(distance, normal, colors, self.rays[..., 1] < -0.0001)

    def pixels(self) -> np.ndarray:
        return np.clip(np.rint(self.colors), 0, 255).astype(np.uint8)


def _room(*, coral_vase: bool) -> np.ndarray:
    s = _Scene()
    s.box((-4, 0, -4), (4, 3.6, 4), (237, 225, 202), room=True)
    # Window, ochre curtains, and a low sideboard.
    s.box((-3.12, 1.15, 3.90), (-0.38, 2.96, 3.99), (46, 96, 91))
    s.box((-2.98, 1.29, 3.87), (-0.52, 2.82, 3.90), (152, 210, 222))
    s.ellipsoid((-1.02, 2.44, 3.84), (0.19, 0.19, 0.035), (255, 225, 142))
    s.box((-2.98, 1.29, 3.83), (-0.52, 1.58, 3.87), (127, 165, 134))
    for x in (-2.2, -1.38):
        s.box((x - 0.025, 1.29, 3.80), (x + 0.025, 2.82, 3.87), (244, 237, 211))
    s.box((-2.98, 2.04, 3.80), (-0.52, 2.10, 3.87), (244, 237, 211))
    for x in (-3.36, -0.25):
        s.box((x - 0.13, 1.10, 3.76), (x + 0.13, 3.04, 3.90), (211, 166, 87))
    s.box((-3.13, 0.08, 2.98), (-0.43, 0.85, 3.78), (180, 124, 74))
    s.box((-3.19, 0.85, 2.92), (-0.37, 0.93, 3.83), (223, 177, 114))
    for x in (-2.25, -1.34):
        s.box((x, 0.14, 2.96), (x + 0.018, 0.78, 2.99), (132, 89, 58))
        s.ellipsoid((x + 0.13, 0.52, 2.93), (0.035, 0.035, 0.035), (241, 208, 135))
    # A sage sofa, soft rug, and a table with the persistent edited object.
    s.box((-3.5, 0.19, 0.3), (-1.8, 0.63, 1.63), (80, 126, 115))
    s.box((-3.5, 0.63, 1.3), (-1.8, 1.24, 1.7), (92, 143, 129))
    s.box((-3.62, 0.40, 0.28), (-3.33, 1.0, 1.72), (104, 154, 136))
    s.box((-1.95, 0.40, 0.28), (-1.67, 1.0, 1.72), (104, 154, 136))
    s.ellipsoid((-2.95, 0.85, 1.22), (0.24, 0.27, 0.14), (230, 174, 116))
    s.box((-1.25, 0.01, 0.55), (2.65, 0.025, 3.0), (219, 201, 165))
    for z in (0.67, 2.84):
        s.box((-1.14, 0.027, z), (2.54, 0.029, z + 0.045), (168, 139, 113))
    for x in (0.78, 2.13):
        for z in (1.88, 2.63):
            s.box((x, 0.03, z), (x + 0.10, 0.87, z + 0.10), (135, 93, 64))
    s.box((0.58, 0.84, 1.72), (2.42, 0.96, 2.88), (221, 170, 111))
    s.box((0.77, 0.96, 2.15), (1.15, 1.01, 2.62), (61, 116, 125))
    s.box((0.82, 1.01, 2.17), (1.19, 1.07, 2.58), (230, 207, 157))
    vase = (234, 118, 94) if coral_vase else (59, 136, 179)
    s.ellipsoid((1.64, 1.22, 2.20), (0.33, 0.31, 0.33), vase)
    s.ellipsoid((1.64, 1.50, 2.20), (0.16, 0.22, 0.16), vase)
    s.ellipsoid((1.64, 1.70, 2.20), (0.18, 0.035, 0.18), vase)
    s.ellipsoid((1.64, 1.715, 2.20), (0.11, 0.012, 0.11), (83, 72, 64))
    # Plant, pendant, rear gallery, and shelving complete all look directions.
    s.ellipsoid((3.1, 0.32, 2.96), (0.35, 0.34, 0.35), (208, 144, 103))
    s.box((3.07, 0.4, 2.93), (3.13, 1.65, 2.99), (97, 111, 70))
    for x, y, z, r in ((2.85, 1.55, 2.95, .42), (3.25, 1.9, 3.02, .44),
                       (3.0, 2.2, 2.97, .36)):
        s.ellipsoid((x, y, z), (r, r * .65, r * .6), (83, 135, 96))
    s.box((-0.025, 3.02, 1.1), (0.025, 3.6, 1.15), (129, 115, 89))
    s.ellipsoid((0, 2.98, 1.125), (.38, .18, .38), (242, 199, 119))
    for x, tone in ((-1.5, (112, 155, 159)), (0, (212, 143, 109)), (1.5, (159, 177, 133))):
        s.box((x - .5, 1.3, -3.96), (x + .5, 2.5, -3.88), (125, 92, 67))
        s.box((x - .43, 1.37, -3.87), (x + .43, 2.43, -3.85), (245, 230, 199))
        s.ellipsoid((x, 1.95, -3.82), (.31, .32, .035), tone)
    s.box((3.52, .1, -1.9), (3.98, 2.55, -.35), (152, 108, 77))
    for y in (.55, 1.15, 1.75, 2.35):
        s.box((3.42, y, -1.95), (3.99, y + .07, -.30), (218, 171, 111))
        for i, tone in enumerate(((69, 115, 129), (206, 157, 102), (156, 166, 124))):
            s.box((3.50, y + .08, -1.77 + .43 * i), (3.88, y + .43, -1.49 + .43 * i), tone)
    return s.pixels()


def _courtyard() -> np.ndarray:
    s = _Scene()
    s.ground()
    s.box((-5, 0, 4.95), (5, 2.9, 5.1), (225, 173, 139))
    s.box((-5.1, 0, -5), (-5, 2.9, 5), (232, 188, 151))
    s.box((5, 0, -5), (5.1, 2.9, 5), (232, 188, 151))
    s.box((-5, 0, -5.1), (5, 2.9, -5), (225, 173, 139))
    # Mint doorway and pale stone surround.
    s.box((-.90, 0, 4.83), (.90, 2.45, 4.95), (248, 228, 185))
    s.box((-.72, 0, 4.78), (.72, 2.28, 4.83), (69, 135, 124))
    s.box((-.58, .18, 4.73), (.58, 1.08, 4.78), (89, 159, 144))
    s.box((-.58, 1.20, 4.73), (.58, 2.10, 4.78), (89, 159, 144))
    s.ellipsoid((.50, 1.12, 4.68), (.045, .045, .045), (231, 191, 111))
    for x in (-3.25, 2.55):
        s.box((x, 1.38, 4.82), (x + 1.05, 2.32, 4.95), (241, 216, 177))
        s.box((x + .12, 1.50, 4.77), (x + .93, 2.20, 4.82), (87, 140, 133))
        for k in range(4):
            s.box((x + .16 + .19 * k, 1.53, 4.73), (x + .20 + .19 * k, 2.17, 4.77), (62, 113, 108))
    # Leafy tree and flowering planter.
    s.box((-3.4, 0, 1.95), (-1.3, .42, 3.6), (188, 124, 90))
    s.box((-2.58, .38, 2.69), (-2.32, 2.30, 2.95), (130, 98, 65))
    for center, radius, color in (
        ((-2.5, 2.9, 2.8), (1.15, .95, .95), (83, 137, 98)),
        ((-3.20, 2.55, 2.7), (.9, .72, .8), (104, 158, 105)),
        ((-1.7, 2.65, 2.85), (.82, .73, .9), (122, 170, 111)),
    ):
        s.ellipsoid(center, radius, color)
    for i in range(7):
        x = -3.15 + .27 * i
        s.ellipsoid((x, .62, 2.10), (.2, .22, .2), (94, 146, 96))
        s.ellipsoid((x, .83, 2.08), (.09, .08, .09), (239, 184, 133))
    # Low pool and wood bench make the destination visibly different.
    s.box((.30, 0, 1.42), (2.15, .32, 3.05), (181, 185, 164))
    s.box((.43, .32, 1.55), (2.02, .36, 2.92), (119, 180, 184))
    s.ellipsoid((1.24, .38, 2.22), (.49, .035, .42), (169, 211, 203))
    for x in (3.05, 3.95):
        s.box((x, .02, .42), (x + .10, .55, 2.55), (106, 105, 79))
    for z in (.42, .82, 1.22, 1.62, 2.02, 2.42):
        s.box((2.92, .55, z), (4.18, .65, z + .23), (181, 133, 85))
    for z in (.45, 2.4):
        s.ellipsoid((3.56, .82, z), (.32, .17, .21), (231, 191, 123))
    s.ellipsoid((3.8, .25, -1.9), (.4, .27, .4), (201, 139, 96))
    for x in (3.5, 3.8, 4.05):
        s.ellipsoid((x, .68, -1.9), (.24, .5, .24), (98, 150, 106))
    return s.pixels()


def render_view(panorama: np.ndarray, yaw: float, pitch: float = 0,
                hfov: float = 90) -> np.ndarray:
    """Perspective sampling with ERP pixel centers, wrap, and bilinear sampling."""
    width, height = VIEW_SIZE
    extent = math.tan(math.radians(hfov) / 2)
    x = (2 * (np.arange(width) + .5) / width - 1) * extent
    y = (1 - 2 * (np.arange(height) + .5) / height) * extent * height / width
    x, y = np.meshgrid(x, y)
    yaw, pitch = math.radians(yaw), math.radians(pitch)
    # Camera basis: positive yaw turns right; positive pitch looks up.
    wx = x * math.cos(yaw) + (math.cos(pitch) - y * math.sin(pitch)) * math.sin(yaw)
    wy = y * math.cos(pitch) + math.sin(pitch)
    wz = -x * math.sin(yaw) + (math.cos(pitch) - y * math.sin(pitch)) * math.cos(yaw)
    u = (np.arctan2(wx, wz) / (2 * np.pi) + .5) * panorama.shape[1] - .5
    v = (.5 - np.arctan2(wy, np.hypot(wx, wz)) / np.pi) * panorama.shape[0] - .5
    v = np.clip(v, 0, panorama.shape[0] - 1)
    left, top = np.floor(u).astype(int), np.floor(v).astype(int)
    dx, dy = (u - left)[..., None], (v - top)[..., None]
    left %= panorama.shape[1]
    right = (left + 1) % panorama.shape[1]
    bottom = np.minimum(top + 1, panorama.shape[0] - 1)
    pixels = ((panorama[top, left] * (1 - dx) + panorama[top, right] * dx) * (1 - dy)
              + (panorama[bottom, left] * (1 - dx) + panorama[bottom, right] * dx) * dy)
    return np.clip(np.rint(pixels), 0, 255).astype(np.uint8)


def _require_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("FFmpeg is required for the procedural rotation clips. Install FFmpeg and rerun this command.")
    return executable


def _rotation(root: Path, relative: str, panorama: np.ndarray,
              start_yaw: float, end_yaw: float) -> dict:
    """Encode actual intermediate perspective renders, including both end poses."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [render_view(panorama, float(yaw))
              for yaw in np.linspace(start_yaw, end_yaw, NUM_FRAMES)]
    command = [
        _require_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "480x270",
        "-framerate", str(FPS), "-i", "pipe:0", "-an", "-map_metadata", "-1",
        "-map_chapters", "-1", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "21", "-profile:v", "baseline", "-level:v", "3.0",
        "-pix_fmt", "yuv420p", "-threads", "1", "-fflags", "+bitexact",
        "-flags:v", "+bitexact", "-metadata", "encoder=",
        "-metadata:s:v:0", "encoder=", "-metadata", "creation_time=",
        "-metadata:s:v:0", "creation_time=", "-movflags", "+faststart", str(path),
    ]
    try:
        result = subprocess.run(command, input=b"".join(frame.tobytes() for frame in frames),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        path.unlink(missing_ok=True)
        raise RuntimeError("Procedural rotation encoding failed. Check the local FFmpeg installation and retry.") from None
    if result.returncode or not path.is_file() or path.stat().st_size == 0:
        path.unlink(missing_ok=True)
        raise RuntimeError("Procedural rotation encoding failed. FFmpeg must support the libx264 encoder.")
    # Stream-copy remux removes the encoder tag that a transcode adds even
    # when its metadata option is empty. MP4's structural format tags remain.
    cleaned = path.with_suffix(".clean.mp4")
    command = [
        _require_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
        "-i", str(path), "-map", "0:v:0", "-an", "-map_metadata", "-1",
        "-map_chapters", "-1", "-c:v", "copy", "-threads", "1",
        "-fflags", "+bitexact", "-metadata", "encoder=",
        "-metadata:s:v:0", "encoder=", "-metadata:s:v:0", "handler_name=",
        "-movflags", "+faststart", str(cleaned),
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=30, check=False)
        if result.returncode or not cleaned.is_file() or cleaned.stat().st_size == 0:
            raise RuntimeError("Procedural video metadata removal failed. Check the local FFmpeg installation.")
        cleaned.replace(path)
    except (OSError, subprocess.TimeoutExpired):
        path.unlink(missing_ok=True)
        raise RuntimeError("Procedural video metadata removal failed. Check the local FFmpeg installation.") from None
    finally:
        cleaned.unlink(missing_ok=True)
    return {"status": "rendered", "backend": "cpu_panorama_rotation",
            "video": _file_ref(root, path), "reason": None,
            "num_frames": NUM_FRAMES, "fps": FPS, "duration_seconds": NUM_FRAMES / FPS}


def _no_video(reason: str) -> dict:
    return {"status": "none", "backend": "none", "video": None, "reason": reason,
            "num_frames": None, "fps": None, "duration_seconds": None}


def make_run(run_root: Path) -> Path:
    """Write an isolated synthetic session and return its session.json path.

    This authors playback records directly. It does not run the agent, fabricate
    model responses, or claim that a semantic model verified these images.
    """
    _require_ffmpeg()
    root = Path(run_root)
    if root.is_symlink() or (root.exists() and (not root.is_dir() or any(root.iterdir()))):
        raise ValueError("The fixture run directory must be absent or empty.")
    root.mkdir(parents=True, exist_ok=True)
    room_blue, room_coral, courtyard = _room(coral_vase=False), _room(coral_vase=True), _courtyard()
    panoramas = {"room_blue": room_blue, "room_coral": room_coral, "courtyard": courtyard}
    state_refs = {}
    for name, pixels in panoramas.items():
        origin, revision = ("origin0001", 1) if name == "courtyard" else ("origin0000", int(name == "room_coral"))
        directory = f"states/{origin}/revision{revision:04d}"
        image_ref = _write_png(root, f"{directory}/panorama_assets/erp.png", pixels)
        # The full engine's numerical state is outside this playback example.
        state = {"schema": "wmmllm.fixture.panorama.v1", "procedural_fixture": True,
                 "erp": image_ref, "render_base": image_ref, "patches": [],
                 "viewport_profile": PROFILE}
        state_hash = hashlib.sha256(_json_bytes(state)).hexdigest()
        state_ref = _write_json(root, f"{directory}/PANORAMA_STATE.json", state)
        state_refs[name] = (image_ref, {**state_ref, "panorama_state_sha256": state_hash})
    # Five nodes, two places, three physical panorama states. The final turn
    # restores f0002 itself; it does not manufacture a replacement revision.
    specifications = [
        ("f0000", None, "room_blue", -16.0),
        ("f0001", "f0000", "room_blue", 33.0),
        ("f0002", "f0001", "room_coral", 33.0),
        ("f0003", "f0002", "room_coral", -16.0),
        ("f0004", "f0003", "courtyard", -12.0),
    ]
    nodes = []
    for node_id, parent_id, name, yaw in specifications:
        facts = ["The room vase is coral." if name == "room_coral" else
                 "The room vase is blue." if name == "room_blue" else
                 "The courtyard has a tree and a low pool."]
        memory = _write_json(root, f"memory/snapshots/{node_id}.json", {
            "schema": "wmmllm.fixture.memory.v1", "provenance": "procedural_authored",
            "node_id": node_id, "facts": facts,
        })
        image_ref, state_ref = state_refs[name]
        node = {"node_id": node_id, "parent_node_id": parent_id,
                "origin_id": "origin0001" if name == "courtyard" else "origin0000",
                "physical_revision": int(name != "room_blue"),
                "view_identity": "first_person|normal|procedural",
                "pose": {"yaw": yaw, "pitch": 0.0, "hfov": 90.0},
                "viewport_profile": PROFILE, "panorama_state": state_ref,
                "panorama": image_ref,
                "view": _write_png(root, f"views/{node_id}.png", render_view(panoramas[name], yaw)),
                "memory_snapshot": memory, "seam_status": "not_checked"}
        nodes.append(node)
        _write_json(root, f"nodes/{node_id}.json", node)
    clips = [
        _rotation(root, "transitions/turn0001/video.mp4", room_blue, -16, 33),
        _no_video("Local edits have no generated video in this release."),
        _rotation(root, "transitions/turn0003/video.mp4", room_coral, 33, -16),
        _no_video("Navigation has no generated video in this release."),
        _no_video("The saved room is restored. Cross-place return geometry is unavailable; no video is rendered."),
    ]
    actions = [
        ("f0000", "f0001", "Look right toward the vase.", "rotate"),
        ("f0001", "f0002", "Make the blue vase coral.", "local_edit"),
        ("f0002", "f0003", "Look left toward the window.", "rotate"),
        ("f0003", "f0004", "Go out to the courtyard.", "navigate"),
        ("f0004", "f0002", "Return to the room and the coral vase.", "revisit"),
    ]
    by_id = {node["node_id"]: node for node in nodes}
    transitions = []
    for index, (source, target, caption, operation) in enumerate(actions, 1):
        transition = {
            "transition_id": f"turn{index:04d}", "index": index,
            "from_node_id": source, "to_node_id": target, "caption": caption,
            "operations": [operation], "status": "accepted",
            "semantic": {"status": "UNCERTAIN", "attempt_count": 0,
                         "semantic_accepted": False, "uncertain_accepted_and_labeled": True,
                         "verification": None},
            "presentation": clips[index - 1], "decision": None,
            "memory_after": by_id[target]["memory_snapshot"],
        }
        transitions.append(transition)
        _write_json(root, f"transitions/turn{index:04d}/transition.json", transition)
    memory = _write_json(root, "memory/MEMORY.json", {
        "schema": "wmmllm.fixture.memory.v1", "provenance": "procedural_authored",
        "restored_node_id": "f0002", "facts": ["The room vase is coral."],
    })
    session = {
        "schema": "wmmllm.session.v1", "session_id": "procedural_playback",
        "title": "Procedural playback fixture", "seed": 0, "status": "ready",
        "state_authority": "accepted_panorama_and_endpoint", "video_writes_state": False,
        "initial_node_id": "f0000", "current_node_id": "f0002",
        "nodes": nodes, "transitions": transitions, "memory": memory,
        "extensions": {"wmmllm:procedural_fixture": True,
                       "wmmllm:semantic_verification_performed": False},
    }
    _write_json(root, "session.json", session)
    return root / "session.json"


def main(argv: list[str] | None = None) -> int:
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds-dir", type=Path, default=project / "viewer" / "worlds")
    parser.add_argument("--world-id", default="procedural")
    parser.add_argument("--work-dir", type=Path, help="Parent for the automatically removed temporary run.")
    args = parser.parse_args(argv)
    try:
        _require_ffmpeg()
        if args.work_dir is not None:
            args.work_dir.mkdir(parents=True, exist_ok=True)
        from wmmllm import export_viewer as exporter
        with tempfile.TemporaryDirectory(prefix="viewer_fixture_", dir=args.work_dir) as temporary:
            run_root = Path(temporary) / "run"
            make_run(run_root)
            world = exporter.export_world(run_root, args.worlds_dir, world_id=args.world_id,
                                          procedural_fixture=True)
        print(f"Procedural playback fixture exported: {world}")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"Fixture generation stopped: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
