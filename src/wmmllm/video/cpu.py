"""CPU panorama camera paths and failure-hold clips."""
from __future__ import annotations
import math
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from wmmllm.state.panorama import crop_camera, erp_array, render_view, pixel_sha256, panorama_state_sha256
from wmmllm.session.io import _local, file_ref, verify_ref, write_atomic


class UnsupportedGeometry(RuntimeError):
    """An encoded clip does not reproduce its exact endpoints."""


def rotation_poses(start_yaw: float = 0, end_yaw: float = 45, start_pitch: float = 0,
                   end_pitch: float = 0, *, num_frames: int = 124) -> list[dict[str, float]]:
    """Return both endpoint poses. Linear motion matches constant camera speed."""
    if int(num_frames) != num_frames or num_frames < 2:
        raise ValueError("num_frames must be an integer >=2")
    for yaw, pitch in ((start_yaw, start_pitch), (end_yaw, end_pitch)):
        crop_camera(yaw, pitch)
    delta = float(end_yaw) - float(start_yaw)
    poses = []
    for i in range(int(num_frames)):
        progress = i / (num_frames - 1)
        poses.append({"yaw": float(start_yaw + progress * delta),
                      "pitch": float(start_pitch + progress * (end_pitch - start_pitch))})
    return poses


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
    if done.returncode or len(done.stdout) != 2 * height * width * 3:
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


def _encode_frames(frames, output: Path, *, width: int, height: int, fps: float, run_root: Path):
    """Losslessly encode, verify both endpoints and remove partial output on failure."""
    output = _local(output, run_root=run_root, must_exist=False)
    if output.exists():
        raise FileExistsError('video output already exists')
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('CPU ffmpeg is required to encode MP4')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name('.' + output.stem + '.' + uuid.uuid4().hex + '.tmp.mp4')
    argv = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
            '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
            '-framerate', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264rgb',
            '-threads', '2', '-preset', 'fast', '-crf', '0',
            '-pix_fmt', 'rgb24', '-movflags', '+faststart', str(temporary)]
    process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE)
    hashes, first, last = [], None, None
    try:
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
        process.stdin = None
        process.communicate(timeout=300)
        if process.returncode:
            raise RuntimeError('CPU ffmpeg encoding failed')
        if not hashes or not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError('CPU ffmpeg produced no video')
        check = verify_encoded_endpoints(temporary, first, last, len(hashes)) if len(hashes) > 1 else None
        _local(output, run_root=run_root, must_exist=False)
        os.link(temporary, output)
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


def render_rotation_video(pano: Any, path: Path, *, start_yaw: float = 0, end_yaw: float = 45,
                          start_pitch: float = 0, end_pitch: float = 0, hfov: float = 90,
                          width: int = 1344, height: int = 768, num_frames: int = 124,
                          fps: float = 24, run_root: Path) -> dict[str, Any]:
    """Sample one unchanged panorama along the source endpoint-inclusive camera path."""
    array = erp_array(pano)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be finite and positive')
    if width % 2 or height % 2:
        raise ValueError('video dimensions must be even for yuv420p')
    crop_camera(start_yaw, start_pitch, hfov, width, height)
    poses = rotation_poses(start_yaw, end_yaw, start_pitch, end_pitch, num_frames=num_frames)
    output = _local(path, run_root=run_root, must_exist=False)
    before = panorama_state_sha256(pano)
    started = time.monotonic()
    frames = (render_view(pano, **pose, hfov=hfov, width=width, height=height) for pose in poses)
    video, frame_refs, encoded_check = None, [], None
    if shutil.which('ffmpeg'):
        hashes, encoded_check = _encode_frames(frames, output, width=width, height=height,
                                               fps=fps, run_root=run_root)
        video = file_ref(output, run_root=run_root)
    else:
        # These are actual rendered frames. The public transition remains video-unavailable.
        directory = _local(output.with_suffix('.frames'), run_root=run_root, must_exist=False)
        directory.mkdir(parents=True, exist_ok=True)
        hashes = []
        for index, frame in enumerate(frames):
            destination = _local(directory / ('%06d.png' % index), run_root=run_root, must_exist=False)
            if destination.exists():
                raise FileExistsError('rendered frame output already exists')
            Image.fromarray(frame).save(destination, format='PNG')
            hashes.append(pixel_sha256(frame))
            frame_refs.append(file_ref(destination, run_root=run_root, image=True))
    after = panorama_state_sha256(pano)
    report = {'schema': 'wmmllm.cpu_rotation.v1', 'status': 'READY' if video else 'FRAMES_ONLY',
              'video': video, 'frames': frame_refs,
              'reason': None if video else 'CPU ffmpeg is unavailable; rendered PNG frames were saved.',
              'num_frames': int(num_frames), 'fps': float(fps), 'duration_seconds': num_frames / fps,
              'width': int(width), 'height': int(height), 'hfov_degrees': float(hfov),
              'pano_size': [array.shape[1], array.shape[0]], 'pano_pixel_sha256': pixel_sha256(array),
              'panorama_state_sha256_before': before, 'panorama_state_sha256_after': after,
              'lossless': True, 'video_writes_state': False,
              'pose_interpolation': 'linear', 'poses': poses, 'raw_frame_pixel_sha256': hashes,
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
    output = _local(destination, run_root=run_root, must_exist=False)
    hashes, check = _encode_frames((frame for _ in range(count)), output, width=width,
                                   height=height, fps=fps, run_root=run_root)
    report = {'schema': 'wmmllm.failure_hold_video.v1', 'status': 'READY',
              'source': dict(source), 'video': file_ref(output, run_root=run_root),
              'num_frames': count, 'fps': fps, 'duration_seconds': count / fps,
              'raw_frame_pixel_sha256': hashes, 'encoded_endpoint_check': check,
              'video_writes_state': False, 'accepted_state_unchanged': True}
    write_atomic(output.with_suffix('.failure_hold.json'), report, run_root=run_root)
    return report
