"""Present the accepted candidate before observation and commit, without writing state."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping
import math
import numpy as np
from PIL import Image

from wmmllm.providers.types import VideoTransition, VideoBackend, PresentationResult
from wmmllm.state.panorama import render_view, panorama_state_sha256, pixel_sha256
from wmmllm.memory.store import checked_path
from wmmllm.session.io import file_ref, verify_ref, write_once, read_json
from .cpu import render_rotation_video, render_known_return_video, encode_hold_video


def _output(transition: VideoTransition, run_root: Path) -> Path:
    path = transition.output_path or Path(run_root) / 'transitions' / transition.transition_id / 'video.mp4'
    return checked_path(path, Path(run_root).resolve(), must_exist=False)


def _node_image(node: Mapping[str, Any]):
    return node.get('view', node.get('endpoint'))


def _pixels(path: Path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB')).copy()


def _same_camera_origin(source, target):
    return (source.get('origin_id') is not None
            and source.get('origin_id') == target.get('origin_id')
            and source.get('physical_revision') is not None
            and source.get('physical_revision') == target.get('physical_revision')
            and source.get('view_identity') == target.get('view_identity')
            and source.get('viewport_profile') == target.get('viewport_profile')
            and source.get('pose', {}).get('hfov', 90) == target.get('pose', {}).get('hfov', 90))


def _verify_camera_endpoints(panorama, source_node, target_node, transition, profile):
    width, height = int(profile['width']), int(profile['height'])
    for node, path in ((source_node, transition.first_frame), (target_node, transition.last_frame)):
        pose = node['pose']
        rendered = render_view(panorama, yaw=pose['yaw'], pitch=pose['pitch'],
                               hfov=pose.get('hfov', 90), width=width, height=height)
        if not np.array_equal(rendered, _pixels(path)):
            raise ValueError('CPU presentation endpoint differs from the authoritative panorama render')


def _presentation_value(result, transition, profile, *, run_root):
    video = None
    if result.video_path is not None:
        path = checked_path(result.video_path, Path(run_root).resolve())
        if path.stat().st_size == 0:
            raise ValueError('Presentation video is empty')
        video = file_ref(path, run_root=run_root)
    rendered = result.status in ('rendered', 'generated')
    if rendered and video is None:
        raise ValueError('Rendered presentation requires an actual video file')
    if result.status == 'none' and (video is not None or not result.reason):
        raise ValueError('Unavailable presentation requires a reason and no video')
    count, fps = int(profile['num_frames']), float(profile['fps'])
    return {'status': result.status, 'backend': result.backend, 'video': video,
            'reason': result.reason, 'num_frames': count, 'fps': fps,
            'duration_seconds': count / fps}


def present_transition(transition: VideoTransition, *, panorama: dict,
                       source_node: Mapping[str, Any], target_node: Mapping[str, Any],
                       backend: VideoBackend, profile: Mapping[str, Any], run_root: Path,
                       depth_record: Mapping[str, Any] | None = None) -> PresentationResult:
    """Dispatch CPU camera paths or the injected backend and bind unchanged authority."""
    root = Path(run_root).resolve()
    if (int(profile['num_frames']) < 2 or not math.isfinite(float(profile['fps']))
            or float(profile['fps']) <= 0):
        raise ValueError('Presentation requires at least two frames and positive finite FPS')
    output = _output(transition, root)
    for path in (transition.first_frame, transition.last_frame):
        checked_path(path, root)
    first_ref = file_ref(transition.first_frame, run_root=root, image=True)
    last_ref = file_ref(transition.last_frame, run_root=root, image=True)
    for node, actual in ((source_node, first_ref), (target_node, last_ref)):
        ref = _node_image(node)
        if ref is not None:
            verify_ref(ref, run_root=root)
            if ref['sha256'] != actual['sha256']:
                raise ValueError('Presentation input differs from the accepted node endpoint')
    before = panorama_state_sha256(panorama)
    binding = {'transition_id': transition.transition_id, 'operation': transition.operation,
               'source': first_ref, 'endpoint': last_ref,
               'panorama_state_sha256': before, 'profile': dict(profile),
               'source_origin_id': source_node.get('origin_id'),
               'target_origin_id': target_node.get('origin_id'),
               'source_physical_revision': source_node.get('physical_revision'),
               'target_physical_revision': target_node.get('physical_revision')}
    receipt_path = output.with_suffix('.presentation.json')
    if receipt_path.exists():
        cached = read_json(receipt_path, run_root=root)
        if cached['binding'] != binding:
            raise ValueError('Existing presentation belongs to another accepted transition')
        info = cached['presentation']
        video_path = verify_ref(info['video'], run_root=root) if info['video'] else None
        return PresentationResult(info['status'], video_path, info['backend'], info['reason'])
    transition = replace(transition, output_path=output)
    if transition.operation in ('rotate', 'revisit') and _same_camera_origin(source_node, target_node):
        _verify_camera_endpoints(panorama, source_node, target_node, transition, profile)
        start, end = source_node['pose'], target_node['pose']
        report = render_rotation_video(panorama, output,
            start_yaw=start['yaw'], end_yaw=end['yaw'], start_pitch=start['pitch'],
            end_pitch=end['pitch'], hfov=end.get('hfov', 90), width=int(profile['width']),
            height=int(profile['height']), num_frames=int(profile['num_frames']),
            fps=float(profile['fps']), lossless=True, run_root=root)
        if report['video'] is None:
            result = PresentationResult('none', None, 'none', report['reason'])
        else:
            result = PresentationResult('rendered', output,
                'cpu_panorama_rotation' if transition.operation == 'rotate' else 'cpu_known_return')
    elif transition.operation == 'revisit' and depth_record is not None:
        # A cross-origin clip additionally binds both supplied geometry states.
        source_state = source_node.get('panorama_state', {}).get('panorama_state_sha256')
        target_state = target_node.get('panorama_state', {}).get('panorama_state_sha256')
        eligible = (source_state is not None and target_state is not None
            and depth_record.get('source_state_sha256') == source_state
            and depth_record.get('target_state_sha256') == target_state
            and depth_record.get('source_physical_revision') == source_node.get('physical_revision')
            and depth_record.get('physical_revision') == target_node.get('physical_revision')
            and depth_record.get('source_viewport_profile') == source_node.get('viewport_profile')
            and depth_record.get('target_viewport_profile') == target_node.get('viewport_profile'))
        if eligible:
            report = render_known_return_video(depth_record, first_ref, target_node, output,
                width=int(profile['width']), height=int(profile['height']),
                num_frames=int(profile['num_frames']), fps=float(profile['fps']), run_root=root)
            if report['status'] == 'READY':
                result = PresentationResult('rendered', output, 'cpu_known_return')
            else:
                result = backend.present(transition)
                if result.status == 'none':
                    result = replace(result, reason='Supplied return geometry cannot support every intermediate ray; generated video is unavailable.')
        else:
            result = backend.present(transition)
            if result.status == 'none':
                result = replace(result, reason='Return geometry is unavailable for these exact states and viewport profiles; generated video is unavailable.')
    elif transition.operation == 'failure_hold':
        if first_ref['pixel_sha256'] != last_ref['pixel_sha256']:
            raise ValueError('Failure hold endpoints differ')
        import shutil
        if shutil.which('ffmpeg'):
            encode_hold_video(first_ref, dict(profile), output, run_root=root)
            result = PresentationResult('rendered', output, 'cpu_failure_hold')
        else:
            result = PresentationResult('none', None, 'none', 'CPU ffmpeg is unavailable; the accepted endpoint remains available.')
    else:
        result = backend.present(transition)
    after = panorama_state_sha256(panorama)
    if after != before:
        raise RuntimeError('Presentation changed the authoritative panorama')
    value = {'schema': 'wmmllm.presentation_authority.v1', 'binding': binding,
             'presentation': _presentation_value(result, transition, profile, run_root=root),
             'panorama_state_sha256_before': before, 'panorama_state_sha256_after': after,
             'authority': 'accepted_panorama_and_endpoint', 'video_writes_state': False,
             'next_source': last_ref, 'presentation_before_observation_and_commit': True}
    write_once(receipt_path, value, run_root=root)
    return result


def read_presentation_receipt(transition: VideoTransition, *, run_root: Path) -> dict:
    return read_json(_output(transition, run_root).with_suffix('.presentation.json'), run_root=run_root)
