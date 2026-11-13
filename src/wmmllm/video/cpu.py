"""CPU panorama camera paths and explicitly supplied two-view geometry."""
from __future__ import annotations
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from wmmllm.state.panorama import (crop_camera, erp_array, render_view, pixel_sha256,
                                  panorama_state_sha256, is_panorama_state)
from wmmllm.memory.store import checked_path
from wmmllm.session.io import file_ref, verify_ref, write_atomic, sha256_file

SCHEMA = 'wmmllm.known_return_geometry.v1'
CONVENTION = {
    'extrinsics': 'world_to_camera', 'camera_axes': 'x_right_y_down_z_forward',
    'depth': 'camera_z', 'pixel_coordinates': 'integer_pixel_centres',
    'world_scale': 'joint_DA3_relative_scale_not_metric',
}
class UnsupportedGeometry(RuntimeError):
    """Supplied geometry cannot meet its observed-support gates."""

def rotation_poses(start_yaw: float = 0, end_yaw: float = 45,
                   start_pitch: float = 0, end_pitch: float = 0,
                   *, num_frames: int = 124, shortest_yaw: bool = False,
                   easing: str = "linear") -> list[dict[str, float]]:
    """Return both endpoint poses. Linear motion matches constant camera speed."""
    if int(num_frames) != num_frames or num_frames < 2:
        raise ValueError("num_frames must be an integer >=2")
    for yaw, pitch in ((start_yaw, start_pitch), (end_yaw, end_pitch)):
        crop_camera(yaw, pitch)
    delta = float(end_yaw) - float(start_yaw)
    if shortest_yaw:
        delta = (delta + 180) % 360 - 180
    if easing not in ("linear", "smoothstep"):
        raise ValueError("easing must be linear or smoothstep")
    poses = []
    for i in range(int(num_frames)):
        progress = i / (num_frames - 1)
        if easing == "smoothstep":
            progress = progress * progress * (3 - 2 * progress)
        poses.append({"yaw": float(start_yaw + progress * delta),
                      "pitch": float(start_pitch + progress * (end_pitch - start_pitch))})
    return poses

def homogeneous_extrinsics(value):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (3, 4):
        matrix = np.vstack((matrix, [0., 0., 0., 1.]))
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise UnsupportedGeometry('DA3 extrinsics are not finite 3x4/4x4 matrices')
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5):
        raise UnsupportedGeometry('DA3 homogeneous extrinsic row is invalid')
    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=2e-3) or not np.isclose(np.linalg.det(rotation), 1., atol=2e-3):
        raise UnsupportedGeometry('DA3 extrinsic rotation is outside SO(3)')
    return matrix

def validate_intrinsics(value, width, height):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise UnsupportedGeometry('DA3 intrinsics are not finite 3x3 matrices')
    fx, fy = matrix[0, 0], matrix[1, 1]
    if not (0.1 * width <= fx <= 10 * width and 0.1 * height <= fy <= 10 * height):
        raise UnsupportedGeometry('DA3 predicted focal lengths are outside valid bounds')
    if not 0.75 <= fx / fy <= 1.34:
        raise UnsupportedGeometry('DA3 predicted fx/fy is outside the measured pinhole plausibility range')
    if not (0 <= matrix[0, 2] <= width and 0 <= matrix[1, 2] <= height):
        raise UnsupportedGeometry('DA3 principal point is outside the processed view')
    if not np.allclose(matrix[2], [0, 0, 1], atol=1e-6):
        raise UnsupportedGeometry('DA3 intrinsic homogeneous row is invalid')
    return matrix

def unproject_points(depth, intrinsics, world_to_camera, *, valid=None):
    """The same camera-z and inverse-w2c construction as DA3 GLB export."""
    depth = np.asarray(depth, dtype=np.float32)
    height, width = depth.shape
    k = validate_intrinsics(intrinsics, width, height)
    c2w = np.linalg.inv(homogeneous_extrinsics(world_to_camera))
    valid = np.isfinite(depth) & (depth > 0) if valid is None else np.asarray(valid, dtype=bool) & np.isfinite(depth) & (depth > 0)
    y, x = np.nonzero(valid)
    rays = np.column_stack((x, y, np.ones(len(x)))) @ np.linalg.inv(k).T
    camera = rays * depth[y, x, None]
    points = camera @ c2w[:3, :3].T + c2w[:3, 3]
    return points.astype(np.float32), (y, x)

def project_points(points, intrinsics, world_to_camera):
    extrinsics = homogeneous_extrinsics(world_to_camera)
    camera = np.asarray(points) @ extrinsics[:3, :3].T + extrinsics[:3, 3]
    projected = camera @ np.asarray(intrinsics, dtype=np.float64).T
    z = camera[:, 2]
    safe = np.where(np.abs(projected[:, 2]) > 1e-12, projected[:, 2], np.nan)
    return projected[:, :2] / safe[:, None], z

def interpolate_camera(source_extrinsic, target_extrinsic, t):
    """SLERP camera-to-world orientation; linearly interpolate camera centres."""
    from scipy.spatial.transform import Rotation, Slerp
    if not 0 <= t <= 1:
        raise ValueError('camera interpolation parameter must lie in [0,1]')
    source = np.linalg.inv(homogeneous_extrinsics(source_extrinsic))
    target = np.linalg.inv(homogeneous_extrinsics(target_extrinsic))
    rotations = Rotation.from_matrix(np.stack((source[:3, :3], target[:3, :3])))
    c2w = np.eye(4)
    c2w[:3, :3] = Slerp([0., 1.], rotations)([float(t)]).as_matrix()[0]
    c2w[:3, 3] = (1 - t) * source[:3, 3] + t * target[:3, 3]
    return np.linalg.inv(c2w)

def _depth_mesh(depth, intrinsics, extrinsics, texture, valid, *, stride=3,
                max_relative_depth_jump=.12):
    """Triangulate actual predicted depth samples, cutting depth discontinuities.

    Coarse triangle vertices carry their exact native image coordinates. RGB
    remains a native-resolution texture sampled by perspective-correct UVs.
    The outer mesh vertices lie on native pixel centres; no hidden shell or
    extrapolated geometry exists outside the observed image rectangle.
    """
    height, width = depth.shape
    ys = np.unique(np.append(np.arange(0, height, stride), height - 1)).astype(int)
    xs = np.unique(np.append(np.arange(0, width, stride), width - 1)).astype(int)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    d = np.asarray(depth[yy, xx], dtype=np.float64).reshape(-1)
    uv = np.column_stack((xx.reshape(-1), yy.reshape(-1))).astype(np.float64)
    rays = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(intrinsics).T
    c2w = np.linalg.inv(homogeneous_extrinsics(extrinsics))
    camera = rays * d[:, None]
    points = camera @ c2w[:3, :3].T + c2w[:3, 3]
    ids = np.arange(len(uv)).reshape(len(ys), len(xs))
    a, b, c, e = ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1], ids[1:, 1:]
    triangles = np.concatenate((np.stack((a, b, c), axis=-1).reshape(-1, 3),
                                np.stack((b, e, c), axis=-1).reshape(-1, 3)))
    sampled_valid = np.asarray(valid[yy, xx], dtype=bool).reshape(-1)
    positive = sampled_valid[triangles].all(axis=1) & np.isfinite(d[triangles]).all(axis=1) & (d[triangles] > 0).all(axis=1)
    depths = d[triangles]
    safe_min = np.maximum(depths.min(axis=1), 1e-12)
    connected = (depths.max(axis=1) - depths.min(axis=1)) / safe_min <= max_relative_depth_jump
    triangles = triangles[positive & connected]
    return {'points': np.ascontiguousarray(points), 'uv': np.ascontiguousarray(uv),
            'triangles': np.ascontiguousarray(triangles), 'texture': np.ascontiguousarray(texture),
            'triangle_count': int(len(triangles)), 'sample_stride_native_pixels': int(stride),
            'discarded_depth_discontinuity_triangles': int((positive & ~connected).sum()),
            'max_relative_depth_jump': float(max_relative_depth_jump)}

def _triangle_kernel(xy, depths, uv, triangles, texture, output, zbuffer, mask):
    """Ordinary CPU triangle rasterizer; optional compilation retains this arithmetic."""
    height, width = output.shape[:2]
    tex_height, tex_width = texture.shape[:2]
    for triangle in triangles:
        ia, ib, ic = triangle[0], triangle[1], triangle[2]
        za, zb, zc = depths[ia], depths[ib], depths[ic]
        if za <= 1e-6 or zb <= 1e-6 or zc <= 1e-6:
            continue
        ax, ay = xy[ia, 0], xy[ia, 1]
        bx, by = xy[ib, 0], xy[ib, 1]
        cx, cy = xy[ic, 0], xy[ic, 1]
        if not (np.isfinite(ax) and np.isfinite(ay) and np.isfinite(bx) and np.isfinite(by) and np.isfinite(cx) and np.isfinite(cy)):
            continue
        denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(denominator) <= 1e-10:
            continue
        xmin = max(0, int(np.ceil(min(ax, bx, cx) - 1e-7)))
        xmax = min(width - 1, int(np.floor(max(ax, bx, cx) + 1e-7)))
        ymin = max(0, int(np.ceil(min(ay, by, cy) - 1e-7)))
        ymax = min(height - 1, int(np.floor(max(ay, by, cy) + 1e-7)))
        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                wa = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / denominator
                wb = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / denominator
                wc = 1. - wa - wb
                if wa < -1e-7 or wb < -1e-7 or wc < -1e-7:
                    continue
                inverse_z = wa / za + wb / zb + wc / zc
                if inverse_z <= 0:
                    continue
                z = 1. / inverse_z
                if z >= zbuffer[y, x]:
                    continue
                u = (wa * uv[ia, 0] / za + wb * uv[ib, 0] / zb + wc * uv[ic, 0] / zc) * z
                v = (wa * uv[ia, 1] / za + wb * uv[ib, 1] / zb + wc * uv[ic, 1] / zc) * z
                if u < -1e-6 or u > tex_width - 1 + 1e-6 or v < -1e-6 or v > tex_height - 1 + 1e-6:
                    continue
                u = max(0., min(float(tex_width - 1), u))
                v = max(0., min(float(tex_height - 1), v))
                u0, v0 = int(np.floor(u)), int(np.floor(v))
                u1, v1 = min(u0 + 1, tex_width - 1), min(v0 + 1, tex_height - 1)
                du, dv = u - u0, v - v0
                for channel in range(3):
                    color = ((1 - du) * (1 - dv) * texture[v0, u0, channel]
                             + du * (1 - dv) * texture[v0, u1, channel]
                             + (1 - du) * dv * texture[v1, u0, channel]
                             + du * dv * texture[v1, u1, channel])
                    output[y, x, channel] = np.uint8(min(255., max(0., np.rint(color))))
                zbuffer[y, x] = z
                mask[y, x] = True

def _content_box(value, width, height):
    box = [0, 0, width, height] if value is None else [int(x) for x in value]
    if len(box) != 4 or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height:
        raise ValueError('endpoint content box must be a valid native-pixel rectangle')
    return box

def verify_encoded_endpoints(path, source, target, num_frames):
    """Decode actual first/last encoded frames and verify exact endpoint RGB."""
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('CPU ffmpeg is required for encoded-arrival verification')
    height, width = source.shape[:2]
    selected = f'select=eq(n\\,0)+eq(n\\,{num_frames - 1})'
    done = subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-i', str(path),
        '-vf', selected, '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180, check=False)
    expected_bytes = 2 * height * width * 3
    if done.returncode or len(done.stdout) != expected_bytes:
        raise UnsupportedGeometry('encoded geometry video did not decode exactly two endpoint frames')
    decoded = np.frombuffer(done.stdout, dtype=np.uint8).reshape(2, height, width, 3)
    first_exact = bool(np.array_equal(decoded[0], source))
    last_exact = bool(np.array_equal(decoded[1], target))
    if not first_exact or not last_exact:
        raise UnsupportedGeometry('encoded geometric video endpoint RGB differs from immutable stored endpoints')
    return {'first_decoded_frame_pixel_sha256': pixel_sha256(decoded[0]),
            'last_decoded_frame_pixel_sha256': pixel_sha256(decoded[1]),
            'first_decoded_frame_pixel_identical': first_exact,
            'last_decoded_frame_pixel_identical': last_exact,
            'verification': 'actual ffmpeg decode of encoded first and last frame'}

def geometry_alignment_metrics(arrays, *, max_reprojection_error=8., min_feature_inliers=20):
    """Check predicted two-view pose against classical image correspondences."""
    import cv2
    images = np.asarray(arrays['processed_images'], dtype=np.uint8)
    detector = cv2.SIFT_create(nfeatures=8000)
    keypoints, descriptors = [], []
    for image in images:
        key, desc = detector.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None)
        keypoints.append(key); descriptors.append(desc)
    if any(desc is None for desc in descriptors):
        raise UnsupportedGeometry('joint two-view images have no correspondence descriptors')
    pairs = cv2.BFMatcher().knnMatch(descriptors[0], descriptors[1], k=2)
    matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .72 * pair[1].distance]
    if len(matches) < min_feature_inliers:
        raise UnsupportedGeometry('too few cross-view image correspondences for geometric support')
    src = np.array([keypoints[0][m.queryIdx].pt for m in matches], dtype=np.float64)
    dst = np.array([keypoints[1][m.trainIdx].pt for m in matches], dtype=np.float64)
    height, width = arrays['depth'].shape[1:]
    xs = np.clip(np.rint(src[:, 0]).astype(int), 0, width - 1)
    ys = np.clip(np.rint(src[:, 1]).astype(int), 0, height - 1)
    k = validate_intrinsics(arrays['intrinsics'][0], width, height)
    c2w = np.linalg.inv(homogeneous_extrinsics(arrays['extrinsics'][0]))
    depth = arrays['depth'][0, ys, xs]
    camera = np.column_stack((src, np.ones(len(src)))) @ np.linalg.inv(k).T * depth[:, None]
    world = camera @ c2w[:3, :3].T + c2w[:3, 3]
    reprojection, z = project_points(world, arrays['intrinsics'][1], arrays['extrinsics'][1])
    errors = np.linalg.norm(reprojection - dst, axis=1)
    inlier = np.isfinite(errors) & (z > 0) & (depth > 0) & (errors <= max_reprojection_error)
    count = int(inlier.sum())
    if count < min_feature_inliers or count / len(matches) < .35:
        raise UnsupportedGeometry('predicted DA3 pose/depth disagrees with cross-view image correspondences')
    return {'feature_matches': len(matches), 'predicted_geometry_inliers': count,
            'predicted_geometry_inlier_fraction': count / len(matches),
            'inlier_reprojection_rmse_processed_pixels': float(np.sqrt(np.mean(errors[inlier] ** 2))),
            'max_inlier_reprojection_error_processed_pixels': float(max_reprojection_error),
            'classical_method': 'SIFT_ratio_matches_checked_against_joint_DA3_depth_and_pose'}


_COMPILED_TRIANGLES = None


def _raster_meshes(meshes, intrinsics, extrinsics, width, height):
    """Use the source triangle arithmetic; compilation is an optional acceleration."""
    global _COMPILED_TRIANGLES
    if _COMPILED_TRIANGLES is None:
        try:
            from numba import njit
        except (ImportError, AttributeError, RuntimeError):
            _COMPILED_TRIANGLES = _triangle_kernel
        else:
            _COMPILED_TRIANGLES = njit(cache=False)(_triangle_kernel)
    output = np.zeros((height, width, 3), dtype=np.uint8)
    zbuffer = np.full((height, width), np.inf, dtype=np.float64)
    mask = np.zeros((height, width), dtype=np.bool_)
    for mesh in meshes:
        xy, depths = project_points(mesh['points'], intrinsics, extrinsics)
        try:
            _COMPILED_TRIANGLES(np.ascontiguousarray(xy), np.ascontiguousarray(depths),
                mesh['uv'], mesh['triangles'], mesh['texture'], output, zbuffer, mask)
        except Exception:
            if _COMPILED_TRIANGLES is _triangle_kernel:
                raise
            # An unavailable compiler does not change the underlying CPU algorithm.
            _COMPILED_TRIANGLES = _triangle_kernel
            _COMPILED_TRIANGLES(np.ascontiguousarray(xy), np.ascontiguousarray(depths),
                mesh['uv'], mesh['triangles'], mesh['texture'], output, zbuffer, mask)
    return output, mask


def _publish(temporary: Path, output: Path, *, overwrite: bool, run_root: Path):
    checked_path(output, Path(run_root).resolve(), must_exist=False)
    if overwrite:
        os.replace(temporary, output)
    else:
        os.link(temporary, output)
        temporary.unlink()


def _encode_frames(frames, output: Path, *, width: int, height: int, fps: float,
                   lossless: bool, crf: int, overwrite: bool, run_root: Path):
    """Reap the bounded encoder and remove its exact partial artifact on failure."""
    import uuid
    output = checked_path(output, Path(run_root).resolve(), must_exist=False)
    if output.exists() and not overwrite:
        raise FileExistsError('video output already exists')
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('CPU ffmpeg is required to encode MP4')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name('.' + output.stem + '.' + uuid.uuid4().hex + '.tmp.mp4')
    encoder = 'libx264rgb' if lossless else 'libx264'
    argv = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
            '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
            '-framerate', str(fps), '-i', 'pipe:0', '-an', '-c:v', encoder,
            '-threads', '2', '-preset', 'fast', '-crf', str(0 if lossless else int(crf)),
            '-pix_fmt', 'rgb24' if lossless else 'yuv420p', '-movflags', '+faststart', str(temporary)]
    process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE)
    hashes, first, last = [], None, None
    try:
        assert process.stdin is not None
        for frame in frames:
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            if frame.shape != (height, width, 3):
                raise ValueError('CPU video frame dimensions differ from the profile')
            if first is None:
                first = frame.copy()
            last = frame.copy()
            hashes.append(pixel_sha256(frame))
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        # communicate reaps the process with a deadline without retaining remote text.
        process.stdin = None
        _, _ = process.communicate(timeout=300)
        if process.returncode:
            raise RuntimeError('CPU ffmpeg encoding failed')
        if len(hashes) < 1 or not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError('CPU ffmpeg produced no video')
        check = verify_encoded_endpoints(temporary, first, last, len(hashes)) if lossless and len(hashes) > 1 else None
        _publish(temporary, output, overwrite=overwrite, run_root=run_root)
        return hashes, check
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        temporary.unlink(missing_ok=True)


def render_rotation_video(pano: Any, path: Path, *, start_yaw: float = 0,
                          end_yaw: float = 45, start_pitch: float = 0,
                          end_pitch: float = 0, hfov: float = 90,
                          width: int = 1344, height: int = 768,
                          num_frames: int = 124, fps: float = 24,
                          shortest_yaw: bool = False, easing: str = 'linear',
                          crf: int = 18, overwrite: bool = False,
                          lossless: bool = False, run_root: Path) -> dict[str, Any]:
    """Sample one unchanged panorama along the source endpoint-inclusive camera path."""
    array = erp_array(pano)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be finite and positive')
    if width % 2 or height % 2:
        raise ValueError('video dimensions must be even for yuv420p')
    crop_camera(start_yaw, start_pitch, hfov, width, height)
    poses = rotation_poses(start_yaw, end_yaw, start_pitch, end_pitch,
                           num_frames=num_frames, shortest_yaw=shortest_yaw, easing=easing)
    output = checked_path(path, Path(run_root).resolve(), must_exist=False)
    before = panorama_state_sha256(pano)
    started = time.monotonic()
    frames = (render_view(pano, **pose, hfov=hfov, width=width, height=height) for pose in poses)
    video, frame_refs, encoded_check = None, [], None
    if shutil.which('ffmpeg'):
        hashes, encoded_check = _encode_frames(frames, output, width=width, height=height,
            fps=fps, lossless=lossless, crf=crf, overwrite=overwrite, run_root=run_root)
        video = file_ref(output, run_root=run_root)
    else:
        # These are actual rendered frames. The public transition remains video-unavailable.
        directory = output.with_suffix('.frames')
        checked_path(directory, Path(run_root).resolve(), must_exist=False)
        directory.mkdir(parents=True, exist_ok=True)
        hashes = []
        for index, frame in enumerate(frames):
            destination = checked_path(directory / ('%06d.png' % index), Path(run_root).resolve(), must_exist=False)
            if destination.exists() and not overwrite:
                raise FileExistsError('rendered frame output already exists')
            Image.fromarray(frame).save(destination, format='PNG')
            hashes.append(pixel_sha256(frame))
            frame_refs.append(file_ref(destination, run_root=run_root, image=True))
    after = panorama_state_sha256(pano)
    if before != after:
        raise RuntimeError('CPU presentation changed authoritative panorama state')
    report = {'schema': 'wmmllm.cpu_rotation.v1', 'status': 'READY' if video else 'FRAMES_ONLY',
              'video': video, 'frames': frame_refs,
              'reason': None if video else 'CPU ffmpeg is unavailable; rendered PNG frames were saved.',
              'num_frames': int(num_frames), 'fps': float(fps), 'duration_seconds': num_frames / fps,
              'width': int(width), 'height': int(height), 'hfov_degrees': float(hfov),
              'pano_size': [array.shape[1], array.shape[0]], 'pano_pixel_sha256': pixel_sha256(array),
              'panorama_state_sha256_before': before, 'panorama_state_sha256_after': after,
              'lossless': bool(lossless), 'video_writes_state': False,
              'pose_interpolation': easing, 'poses': poses, 'raw_frame_pixel_sha256': hashes,
              'first_frame_pixel_sha256': hashes[0], 'last_frame_pixel_sha256': hashes[-1],
              'encoded_endpoint_check': encoded_check, 'render_and_encode_seconds': time.monotonic() - started}
    write_atomic(output.with_suffix('.geometry.json'), report, run_root=run_root)
    return report


def encode_hold_video(source: dict, profile: dict, destination: Path, *, run_root: Path) -> dict:
    """Encode repeated accepted pixels; a hold writes no requested-action fact."""
    path = verify_ref(source, run_root=run_root)
    with Image.open(path) as image:
        frame = np.asarray(image.convert('RGB')).copy()
    width, height = int(profile['width']), int(profile['height'])
    count, fps = int(profile['num_frames']), float(profile['fps'])
    if frame.shape != (height, width, 3) or count < 2 or not math.isfinite(fps) or fps <= 0:
        raise ValueError('Failure hold source/profile dimensions or timing differ')
    output = checked_path(destination, Path(run_root).resolve(), must_exist=False)
    hashes, check = _encode_frames((frame for _ in range(count)), output, width=width,
        height=height, fps=fps, lossless=True, crf=0, overwrite=False, run_root=run_root)
    report = {'schema': 'wmmllm.failure_hold_video.v1', 'status': 'READY',
              'source': dict(source), 'video': file_ref(output, run_root=run_root),
              'num_frames': count, 'fps': fps, 'duration_seconds': count / fps,
              'raw_frame_pixel_sha256': hashes, 'encoded_endpoint_check': check,
              'video_writes_state': False, 'accepted_state_unchanged': True}
    write_atomic(output.with_suffix('.failure_hold.json'), report, run_root=run_root)
    return report


def _load_geometry(record, source_ref, target_ref, *, run_root):
    if record.get('status') != 'READY' or record.get('convention') != CONVENTION:
        raise UnsupportedGeometry('supplied two-view geometry is unavailable or has a different convention')
    if not record.get('model_execution_measured'):
        raise UnsupportedGeometry('measured two-view geometry is required')
    for role, actual in (('source', source_ref), ('target', target_ref)):
        verify_ref(actual, run_root=run_root)
        if record.get(role, {}).get('sha256') != actual['sha256']:
            raise ValueError('two-view endpoint binding differs: ' + role)
    product_path = verify_ref(record['product'], run_root=run_root)
    with np.load(product_path, allow_pickle=False) as product:
        arrays = {key: np.asarray(product[key]).copy() for key in product.files}
    required = {'depth', 'conf', 'intrinsics', 'extrinsics', 'processed_images'}
    if not required <= set(arrays):
        raise UnsupportedGeometry('two-view geometry lacks depth, confidence, calibration or RGB')
    if arrays['depth'].ndim != 3 or arrays['depth'].shape[0] != 2:
        raise UnsupportedGeometry('joint geometry must contain exactly two views')
    if arrays['conf'].shape != arrays['depth'].shape or arrays['processed_images'].shape != (*arrays['depth'].shape, 3):
        raise UnsupportedGeometry('two-view depth, confidence and RGB shapes disagree')
    return arrays


def render_known_return_video(depth_record, source_ref, target_node, path, *,
                              width=1344, height=768, num_frames=124, fps=24,
                              min_coverage=.995, require_translation=True,
                              overwrite=False, run_root: Path):
    """Render a real camera transition from both stored RGB point clouds.

    Every intermediate frame must meet the coverage gate. Unobserved rays
    produce UNSUPPORTED and no published video. First/last frames are the exact
    saved endpoint bytes decoded as RGB; the target panorama reference remains
    byte-identical and is returned as state authority. No image or video generator is called.
    """
    started = time.monotonic()
    output = checked_path(path, Path(run_root).resolve(), must_exist=False)
    target_ref = target_node.get('endpoint', target_node.get('view'))
    from wmmllm.state.graph import verify_saved_node, _node_revision
    result = {'schema': SCHEMA, 'status': 'UNSUPPORTED', 'rgb_generation_calls': 0,
              'video_writes_state': False, 'geometry_backend': 'two_view_depth_mesh_cpu_zbuffer',
              'convention': CONVENTION, 'geometry_product': depth_record.get('product'),
              'source': dict(source_ref), 'target': dict(target_ref), 'video': None}
    temporary = None
    process = None
    try:
        if output.exists() and not overwrite:
            raise FileExistsError(output)
        if width % 2 or height % 2 or num_frames < 3 or not math.isfinite(fps) or fps <= 0:
            raise ValueError('video requires even dimensions, at least three frames and positive FPS')
        saved = verify_saved_node(target_node, run_root=run_root)
        arrays = _load_geometry(depth_record, source_ref, target_ref, run_root=run_root)
        if depth_record['physical_revision'] != _node_revision(target_node):
            raise ValueError('two-view geometry belongs to another physical revision')
        if depth_record['target_state_sha256'] != saved['panorama_state']['panorama_state_sha256']:
            raise ValueError('two-view geometry belongs to another saved panorama state')
        processed_height, processed_width = arrays['depth'].shape[1:]
        for view in range(2):
            validate_intrinsics(arrays['intrinsics'][view], processed_width, processed_height)
            homogeneous_extrinsics(arrays['extrinsics'][view])
        alignment = geometry_alignment_metrics(arrays)
        source = np.asarray(Image.open(verify_ref(source_ref, run_root=run_root)).convert('RGB')).copy()
        target = np.asarray(Image.open(verify_ref(target_ref, run_root=run_root)).convert('RGB')).copy()
        if source.shape != (height, width, 3) or target.shape != source.shape:
            raise UnsupportedGeometry('saved native endpoints must match exact requested video dimensions')
        boxes = [
            _content_box(depth_record.get('source_content_box'), width, height),
            _content_box(depth_record.get('target_content_box'), width, height),
        ]
        if boxes[0] != boxes[1]:
            raise UnsupportedGeometry('two stored endpoint viewport content boxes differ')
        left, top, right, bottom = boxes[0]
        native_width, native_height = right - left, bottom - top
        padding = np.ones((height, width), dtype=bool)
        padding[top:bottom, left:right] = False
        if not np.array_equal(source[padding], target[padding]):
            raise UnsupportedGeometry('stored endpoint non-world padding differs across a known return')
        native_k, meshes = [], []
        import cv2
        for view, image in enumerate((source, target)):
            k = np.asarray(arrays['intrinsics'][view], dtype=np.float64).copy()
            sx, sy = native_width / processed_width, native_height / processed_height
            # DA3 upper_bound_resize uses OpenCV pixel-centre resizing.
            k[0, :] *= sx; k[1, :] *= sy
            k[0, 2] += .5 * (sx - 1); k[1, 2] += .5 * (sy - 1)
            depth = cv2.resize(arrays['depth'][view].astype(np.float32), (native_width, native_height), interpolation=cv2.INTER_LINEAR)
            confidence = cv2.resize(arrays['conf'][view].astype(np.float32), (native_width, native_height), interpolation=cv2.INTER_LINEAR)
            valid = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence) & (confidence > 0)
            if valid.mean() < min_coverage:
                raise UnsupportedGeometry('joint DA3 depth/confidence has insufficient finite positive native support')
            mesh = _depth_mesh(depth, k, arrays['extrinsics'][view], image[top:bottom, left:right], valid)
            if mesh['triangle_count'] < 32:
                raise UnsupportedGeometry('predicted depth has too few connected real surface triangles')
            meshes.append(mesh)
            native_k.append(k)
        centres = [np.linalg.inv(homogeneous_extrinsics(e))[:3, 3] for e in arrays['extrinsics']]
        baseline = float(np.linalg.norm(centres[1] - centres[0]))
        scene_depth = float(np.median(arrays['depth'][np.isfinite(arrays['depth']) & (arrays['depth'] > 0)]))
        if require_translation and baseline <= max(1e-6, scene_depth * 1e-4):
            raise UnsupportedGeometry('different-origin return has no measured nonzero camera baseline')
        endpoint_reconstruction = []
        for view, image in enumerate((source, target)):
            reconstruction, support = _raster_meshes(meshes, native_k[view], arrays['extrinsics'][view], native_width, native_height)
            reference = image[top:bottom, left:right]
            if float(support.mean()) < min_coverage:
                raise UnsupportedGeometry('two-view mesh does not reconstruct a saved endpoint with sufficient observed coverage')
            difference = np.abs(reconstruction[support].astype(float) - reference[support].astype(float))
            mae = float(difference.mean())
            p95 = float(np.percentile(difference.mean(axis=1), 95))
            if mae > 8. or p95 > 35.:
                raise UnsupportedGeometry('two-view occlusion resolve disagrees with saved endpoint pixels')
            endpoint_reconstruction.append({'view_index': view, 'valid_coverage': float(support.mean()),
                'rgb_mae': mae, 'pixel_rgb_mean_error_p95': p95,
                'max_rgb_mae_gate': 8., 'max_pixel_mean_error_p95_gate': 35.})
        coverage_records, camera_records = [], []
        def frames():
            for index in range(num_frames):
                t = index / (num_frames - 1)
                e = interpolate_camera(arrays['extrinsics'][0], arrays['extrinsics'][1], t)
                k = (1 - t) * native_k[0] + t * native_k[1]
                if index == 0:
                    frame, fraction = source, 1.
                elif index == num_frames - 1:
                    frame, fraction = target, 1.
                else:
                    content, mask = _raster_meshes(meshes, k, e, native_width, native_height)
                    fraction = float(mask.mean())
                    if fraction < min_coverage:
                        raise UnsupportedGeometry('intermediate frame coverage is below the required gate')
                    if not np.all(mask):
                        raise UnsupportedGeometry('intermediate frame has unobserved rays; geometric hole filling is unavailable')
                    frame = source.copy()
                    frame[top:bottom, left:right] = content
                coverage_records.append(fraction)
                camera_records.append({'index': index, 't': t, 'world_to_camera': e.tolist(), 'intrinsics_native': k.tolist()})
                yield frame
        hashes, encoded_endpoint_check = _encode_frames(frames(), output, width=width,
            height=height, fps=fps, lossless=True, crf=0, overwrite=overwrite, run_root=run_root)
        result.update({'status': 'READY', 'reason': 'supplied two-view geometry meets every intermediate support gate',
            'video': file_ref(output, run_root=run_root), 'raw_frame_pixel_sha256': hashes,
            'first_frame_pixel_identical': hashes[0] == pixel_sha256(source),
            'last_frame_pixel_identical': hashes[-1] == pixel_sha256(target),
            'encoded_endpoint_check': encoded_endpoint_check, 'coverage_by_frame': coverage_records,
            'min_valid_coverage': min(coverage_records), 'camera_records': camera_records,
            'camera_baseline_geometry_units': baseline, 'endpoint_mesh_reconstruction': endpoint_reconstruction,
            'depth_meshes': [{key: mesh[key] for key in ('triangle_count', 'sample_stride_native_pixels',
                'discarded_depth_discontinuity_triangles', 'max_relative_depth_jump')} for mesh in meshes],
            'baseline_to_median_depth': baseline / scene_depth, 'alignment': alignment,
            'num_frames': int(num_frames), 'fps': float(fps), 'duration_seconds': num_frames / fps,
            'lossless': True, 'endpoint': saved['endpoint'], 'panorama': saved['panorama'],
            'panorama_state': saved['panorama_state'], 'restore_byte_identical': True,
            'restore_pixel_identical': True, 'rgb_generation': False,
            'predicted_depth_points_per_view': [len(mesh['points']) for mesh in meshes]})
    except UnsupportedGeometry as error:
        result['reason'] = str(error)
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
    write_atomic(output.with_suffix('.GEOMETRY.json'), result, run_root=run_root)
    return result

