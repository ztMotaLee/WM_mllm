"""Immutable accepted-view records and revision-aware visited-state lookup."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .panorama import erp_array, load_panorama_state, panorama_state_sha256, pixel_sha256, render_view
from ..session.io import _ID, public_ref, verify_ref, write_once


def _identifier(value, field):
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError('invalid ' + field)
    return value


def canonical_identity(identity: str | Mapping[str, str]) -> str:
    if isinstance(identity, Mapping):
        if not identity or not all(isinstance(k, str) and isinstance(v, str) for k, v in identity.items()):
            raise ValueError('view identity requires string-valued fields')
        return json.dumps(dict(identity), sort_keys=True, separators=(',', ':'))
    if not isinstance(identity, str) or not identity:
        raise ValueError('view identity must be a nonempty string or string-valued mapping')
    return identity


def visit_key(physical_rev: int, identity: str | Mapping[str, str], pose: Mapping) -> tuple[int, str, float, float]:
    yaw, pitch = float(pose.get('yaw', 0.)), float(pose.get('pitch', 0.))
    if not math.isfinite(yaw) or not math.isfinite(pitch):
        raise ValueError('visited yaw/pitch must be finite')
    if type(physical_rev) is not int or physical_rev < 0:
        raise ValueError('physical revision must be a nonnegative integer')
    return (physical_rev, canonical_identity(identity), round(yaw % 360., 6), round(pitch, 6))


def _profile(profile):
    if not isinstance(profile, Mapping):
        return None
    width, height = profile.get('width'), profile.get('height')
    box, padding = profile.get('content_box'), profile.get('padding_ltrb')
    if type(width) is not int or type(height) is not int or min(width, height) < 1:
        return None
    if not isinstance(box, (tuple, list)) or len(box) != 4 or any(type(x) is not int for x in box):
        return None
    left, top, right, bottom = box
    if not 0 <= left < right <= width or not 0 <= top < bottom <= height:
        return None
    if (not isinstance(padding, (tuple, list)) or list(padding) != [left, top, width - right, height - bottom]
            or any(type(x) is not int for x in padding)):
        return None
    return {'width': width, 'height': height, 'content_box': list(box), 'padding_ltrb': list(padding)}


def public_viewport_profile(profile: Mapping) -> dict:
    result = _profile(profile)
    if result is None:
        raise ValueError('invalid viewport profile')
    return result


def verify_saved_node(node: Mapping, *, run_root: Path) -> dict[str, Any]:
    """Verify encoded assets, decoded pixels, canonical state and arrival view; return the state."""
    identity = node['view_identity']
    if 'unresolved_' in canonical_identity(identity):
        raise ValueError('saved view identity is unresolved')
    revision = node['physical_revision']
    if type(revision) is not int or revision < 0:
        raise ValueError('saved physical revision is unresolved')
    if not node.get('origin_id') or 'unresolved_' in str(node['origin_id']):
        raise ValueError('saved origin is unresolved')
    _identifier(node['origin_id'], 'saved origin ID')
    refs = {}
    for name, public_name in (('endpoint', 'view'), ('panorama', 'panorama')):
        ref = node[public_name]
        if not isinstance(ref, Mapping) or not {'path', 'sha256', 'pixel_sha256', 'size'} <= set(ref):
            raise ValueError('saved ' + public_name + ' requires encoded and decoded provenance')
        verify_ref(ref, run_root=run_root)
        refs[name] = dict(ref)
    state_ref = node['panorama_state']
    if not isinstance(state_ref, Mapping) or not {'path', 'sha256', 'panorama_state_sha256'} <= set(state_ref):
        raise ValueError('saved multires panorama state provenance is incomplete')
    state = load_panorama_state(state_ref, run_root=run_root)
    if panorama_state_sha256(state) != state_ref['panorama_state_sha256']:
        raise ValueError('restored authoritative panorama state differs')
    if (_profile(node.get('viewport_profile')) is None
            or _profile(node.get('viewport_profile')) != _profile(state['viewport_profile'])):
        raise ValueError('saved viewport profile differs from authoritative state')
    if pixel_sha256(erp_array(state)) != refs['panorama']['pixel_sha256']:
        raise ValueError('saved ERP preview differs from authoritative state')
    pose = node.get('pose', {})
    if not isinstance(pose, Mapping) or not {'yaw', 'pitch', 'hfov'} <= set(pose):
        raise ValueError('saved camera pose is unresolved')
    width, height = refs['endpoint']['size']
    arrival = render_view(state, yaw=pose['yaw'], pitch=pose['pitch'], hfov=pose['hfov'],
                          width=width, height=height)
    if pixel_sha256(arrival) != refs['endpoint']['pixel_sha256']:
        raise ValueError('saved arrival view differs from authoritative renderer')
    return state


def node_record(*, node_id: str, parent_node_id: str | None,
                origin_id: str, physical_revision: int, view_identity: str | Mapping[str, str],
                pose: Mapping, panorama_state: Mapping, panorama: Mapping,
                view: Mapping, memory_snapshot: Mapping, viewport_profile: Mapping,
                seam_status: str = 'not_checked') -> dict:
    """Build a public node record; ``SceneGraph.add_node`` verifies and persists it."""
    _identifier(node_id, 'node ID')
    _identifier(origin_id, 'origin ID')
    if parent_node_id is not None:
        _identifier(parent_node_id, 'parent node ID')
    camera = {'yaw': float(pose.get('yaw', 0)), 'pitch': float(pose.get('pitch', 0)),
              'hfov': float(pose.get('hfov', 90))}
    if seam_status not in ('resolved', 'seam_unresolved', 'not_checked'):
        raise ValueError('invalid seam status')
    return {'node_id': node_id, 'parent_node_id': parent_node_id, 'origin_id': origin_id,
            'physical_revision': physical_revision, 'view_identity': deepcopy(view_identity),
            'pose': camera, 'viewport_profile': public_viewport_profile(viewport_profile),
            'panorama_state': public_ref(panorama_state, state=True), 'panorama': public_ref(panorama, image=True),
            'view': public_ref(view, image=True), 'memory_snapshot': public_ref(memory_snapshot), 'seam_status': seam_status}


class SceneGraph:
    """Small persistent graph of actual producer records, preserving parents."""
    def __init__(self, *, run_root: Path) -> None:
        self.run_root = Path(run_root)
        self._nodes: dict[str, dict] = {}

    def add_node(self, node: Mapping) -> None:
        key = _identifier(node.get('node_id'), 'node ID')
        _identifier(node.get('origin_id'), 'origin ID')
        if key in self._nodes:
            if self._nodes[key] != node:
                raise ValueError('immutable graph node changed')
            return
        parent = node.get('parent_node_id')
        if parent is not None and parent not in self._nodes:
            raise ValueError('parent node is absent from graph')
        if parent is None and self._nodes:
            raise ValueError('graph already has a root node')
        verify_saved_node(node, run_root=self.run_root)
        write_once(self.run_root / 'nodes' / (key + '.json'), dict(node), run_root=self.run_root)
        self._nodes[key] = deepcopy(dict(node))
