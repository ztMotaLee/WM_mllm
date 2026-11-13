"""Immutable accepted-view records and revision-aware visited-state lookup."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from .panorama import (crop_camera, erp_array, load_panorama_state,
                       panorama_state_sha256, pixel_sha256, render_view, rgb_array)


def _official(row):
    turn = row.get('official_turn', {}) or {}
    interaction = turn.get('official_interaction', row.get('official_interaction', {})) or {}
    return turn, interaction


def _normal_text(value):
    value = re.sub(r'\s+', ' ', str(value or '').strip().lower())
    value = value.replace('first-person', 'first_person').replace('third-person', 'third_person')
    return value.rstrip(' .')


def _actor_identity(row, after):
    _, interaction = _official(row)
    action = _normal_text(interaction.get('action', row.get('action', '')))
    if not action.startswith(('fp_to_fp:', 'tp_to_tp:')):
        return 'default_subject'
    caption = _normal_text(row.get('current_caption', row.get('caption', '')))
    if after:
        destination = action.split(':', 1)[1].strip()
        destination = re.sub(r'^(?:cut|switch)\s+(?:back\s+)?to\s+', '', destination)
        destination = re.sub(r'^(?:follow|following)\s+', '', destination)
        return destination or 'unresolved_actor_destination'
    match = re.search(r'\bswitches\s+(?:back\s+)?from\s+(.+)\s+to\s+(.+)$', caption)
    if not match:
        return 'unresolved_actor_origin'
    origin = re.sub(r'^(?:follow|following)\s+', '', match.group(1))
    return origin or 'unresolved_actor_origin'


def view_identity(row, before_or_after='after'):
    """Preserve explicit perspective/actor identity in benchmark metadata."""
    aliases = {'before': False, 'current': False, 'source': False,
               'after': True, 'resulting': True, 'target': True}
    if before_or_after not in aliases:
        raise ValueError('before_or_after must be before/current or after/resulting')
    after = aliases[before_or_after]
    turn, _ = _official(row)
    prefix = 'resulting' if after else 'current'
    perspective = _normal_text(turn.get(prefix + '_perspective', row.get(prefix + '_perspective')))
    mode = _normal_text(turn.get(prefix + '_view_mode', row.get(prefix + '_view_mode', 'normal')))
    perspective = {'fp': 'first_person', 'tp': 'third_person'}.get(perspective, perspective)
    return '|'.join((perspective or 'unresolved_perspective', mode or 'normal', _actor_identity(row, after)))


def canonical_identity(identity: str | Mapping[str, str]) -> str:
    if isinstance(identity, Mapping):
        if not identity or not all(isinstance(k, str) and isinstance(v, str) for k, v in identity.items()):
            raise ValueError('view identity requires string-valued fields')
        return json.dumps(dict(identity), sort_keys=True, separators=(',', ':'))
    if not isinstance(identity, str) or not identity:
        raise ValueError('view identity must be a nonempty string or string-valued mapping')
    return identity


def camera_only_switch(row):
    turn, interaction = _official(row)
    if interaction.get('type') != 'perspective_switch':
        return False
    if row.get('physical_state_changes') or interaction.get('physical_state_changes'):
        return False
    action = _normal_text(interaction.get('action', row.get('action', '')))
    if action in {'fp_to_tp', 'tp_to_fp'}:
        expected = ('first_person', 'third_person') if action == 'fp_to_tp' else ('third_person', 'first_person')
        actual = tuple(_normal_text(turn.get(k)) for k in ('current_perspective', 'resulting_perspective'))
        return actual == expected and turn.get('current_view_mode', 'normal') == turn.get('resulting_view_mode', 'normal')
    if action.startswith(('fp_to_fp:', 'tp_to_tp:')):
        before, after = view_identity(row, 'before'), view_identity(row, 'after')
        return 'unresolved_perspective' not in before + after and 'unresolved_actor_destination' not in after
    return False


def visit_key(physical_rev: int, identity: str | Mapping[str, str], pose: Mapping) -> tuple[int, str, float, float]:
    yaw, pitch = float(pose.get('yaw', 0.)), float(pose.get('pitch', 0.))
    if not math.isfinite(yaw) or not math.isfinite(pitch):
        raise ValueError('visited yaw/pitch must be finite')
    if type(physical_rev) is not int or physical_rev < 0:
        raise ValueError('physical revision must be a nonnegative integer')
    return (physical_rev, canonical_identity(identity), round(yaw % 360., 6), round(pitch, 6))


def _node_revision(node):
    metadata = node.get('metadata', {}) or {}
    return node.get('physical_revision', node.get('physical_rev', metadata.get('physical_revision', metadata.get('scene_rev'))))


def _node_identity(node):
    return node.get('view_identity', (node.get('metadata', {}) or {}).get('view_identity'))


def location_identity(pose):
    if pose.get('location_id') is not None:
        return 'location_id:' + str(pose['location_id'])
    xyz = pose.get('rig_position_world_xyz')
    if xyz is None:
        return None
    xyz = np.asarray(xyz, dtype=float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError('rig position must be a finite world xyz triple')
    if pose.get('position_authority') != 'measured_geometry':
        return None
    return 'measured_xyz:' + json.dumps(np.round(xyz, 6).tolist(), separators=(',', ':'))


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


def same_render_origin(source: Mapping, target: Mapping) -> bool:
    """A CPU rotation requires explicit matching origin, revision and profile."""
    if _node_identity(source) is None or _node_identity(target) is None:
        return False
    return bool(source.get('origin_id') and source.get('origin_id') == target.get('origin_id')
                and _node_revision(source) == _node_revision(target)
                and canonical_identity(_node_identity(source)) == canonical_identity(_node_identity(target))
                and _profile(source.get('viewport_profile')) is not None
                and _profile(source.get('viewport_profile')) == _profile(target.get('viewport_profile')))


def public_viewport_profile(profile: Mapping) -> dict:
    result = _profile(profile)
    if result is None:
        raise ValueError('invalid viewport profile')
    return result


def resolve_return(visits, row: Mapping, physical_rev: int, pose: Mapping) -> dict | None:
    """Resolve the source's explicit camera switch, with current-origin bindings.

    Opposite navigation instructions never establish a geometric return.
    Supply the current origin/profile in the row or pose and the requested
    HFOV in pose. Unresolved bindings cannot authorize a saved-state restore.
    Saved-file verification remains a separate call.
    """
    if not camera_only_switch(row):
        return None
    identity = view_identity(row, 'after')
    if 'unresolved_' in identity:
        return None
    wanted = visit_key(physical_rev, identity, pose)
    candidates = []
    if isinstance(visits, Mapping):
        direct = visits.get(wanted)
        if direct is not None:
            candidates.append((wanted, direct))
        candidates.extend((key, node) for key, node in visits.items() if key != wanted)
    else:
        candidates.extend((None, node) for node in visits)
    wanted_origin = pose.get('origin_id', row.get('origin_id'))
    wanted_profile = pose.get('viewport_profile', row.get('viewport_profile'))
    if (not wanted_origin or 'unresolved_' in str(wanted_origin)
            or _profile(wanted_profile) is None or 'hfov' not in pose):
        return None
    for key, node in candidates:
        if not isinstance(node, Mapping) or _node_revision(node) != physical_rev:
            continue
        node_identity = _node_identity(node)
        if node_identity is None and isinstance(key, tuple) and len(key) == 4:
            node_identity = key[1]
        if node_identity is None or canonical_identity(node_identity) != canonical_identity(identity):
            continue
        if node.get('origin_id') != wanted_origin:
            continue
        if _profile(node.get('viewport_profile')) != _profile(wanted_profile):
            continue
        wanted_location = location_identity(pose)
        node_location = location_identity(node.get('pose', node))
        if (wanted_location is not None or node_location is not None) and wanted_location != node_location:
            continue
        if 'hfov' not in node.get('pose', node) or float(node.get('pose', node)['hfov']) != float(pose['hfov']):
            continue
        if visit_key(physical_rev, node_identity, node.get('pose', node)) != wanted:
            continue
        return dict(node)
    return None


def verify_saved_node(node: Mapping, *, run_root: Path) -> dict[str, Any]:
    """Verify encoded assets, decoded pixels, canonical state and arrival view."""
    from ..session.io import verify_ref

    identity = _node_identity(node)
    if identity is None or 'unresolved_' in canonical_identity(identity):
        raise ValueError('saved view identity is unresolved')
    if type(_node_revision(node)) is not int or _node_revision(node) < 0:
        raise ValueError('saved physical revision is unresolved')
    if not node.get('origin_id') or 'unresolved_' in str(node['origin_id']):
        raise ValueError('saved origin is unresolved')
    _identifier(node['origin_id'], 'saved origin ID')
    refs = {}
    for name, public_name in (('endpoint', 'view'), ('panorama', 'panorama')):
        ref = node.get(public_name, node.get(name))
        if not isinstance(ref, Mapping) or not {'path', 'sha256', 'pixel_sha256', 'size'} <= set(ref):
            raise ValueError('saved ' + public_name + ' requires encoded and decoded provenance')
        verify_ref(ref, run_root=run_root)
        refs[name] = dict(ref)
    state_ref = node.get('panorama_state', node.get('panorama_state_ref'))
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
    arrival = render_view(state, yaw=pose.get('yaw', 0), pitch=pose.get('pitch', 0),
                          hfov=pose.get('hfov', 90), width=width, height=height)
    if pixel_sha256(arrival) != refs['endpoint']['pixel_sha256']:
        raise ValueError('saved arrival view differs from authoritative renderer')
    return {**refs, 'view': dict(refs['endpoint']), 'panorama_state': dict(state_ref), 'state': state,
            'restore_pixel_identical': True, 'restore_byte_identical': True,
            'physical_revision': _node_revision(node), 'view_identity': identity}


_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}')


def _identifier(value, field):
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError('invalid ' + field)
    return value


def node_record(*, run_root: Path, node_id: str, parent_node_id: str | None,
                origin_id: str, physical_revision: int, view_identity: str | Mapping[str, str],
                pose: Mapping, panorama_state: Mapping, panorama: Mapping,
                view: Mapping, memory_snapshot: Mapping, viewport_profile: Mapping,
                seam_status: str = 'not_checked') -> dict:
    from ..session.io import verify_ref, write_once, public_ref

    _identifier(node_id, 'node ID')
    _identifier(origin_id, 'origin ID')
    if parent_node_id is not None:
        _identifier(parent_node_id, 'parent node ID')
    visit_key(physical_revision, view_identity, pose)
    camera = {'yaw': float(pose.get('yaw', 0)), 'pitch': float(pose.get('pitch', 0)),
              'hfov': float(pose.get('hfov', 90))}
    crop_camera(**camera)
    if seam_status not in ('resolved', 'seam_unresolved', 'not_checked'):
        raise ValueError('invalid seam status')
    for ref in (panorama_state, panorama, view, memory_snapshot):
        verify_ref(ref, run_root=run_root)
    node = {'node_id': node_id, 'parent_node_id': parent_node_id, 'origin_id': origin_id,
            'physical_revision': physical_revision, 'view_identity': deepcopy(view_identity),
            'pose': camera, 'viewport_profile': public_viewport_profile(viewport_profile),
            'panorama_state': public_ref(panorama_state, state=True), 'panorama': public_ref(panorama, image=True),
            'view': public_ref(view, image=True), 'memory_snapshot': public_ref(memory_snapshot), 'seam_status': seam_status}
    verify_saved_node(node, run_root=run_root)
    write_once(Path(run_root) / 'nodes' / (node_id + '.json'), node, run_root=run_root)
    return node


class SceneGraph:
    """Small persistent graph of actual producer records, preserving parents."""
    def __init__(self, *, run_root: Path) -> None:
        self.run_root = Path(run_root)
        self._nodes: dict[str, dict] = {}

    def add_node(self, node: Mapping) -> None:
        from ..session.io import read_json, write_once

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
        ref = write_once(self.run_root / 'nodes' / (key + '.json'), dict(node), run_root=self.run_root)
        if read_json(ref, run_root=self.run_root) != node:
            raise ValueError('persisted graph node differs')
        self._nodes[key] = deepcopy(dict(node))

    def get_node(self, node_id: str) -> dict:
        return deepcopy(self._nodes[_identifier(node_id, 'node ID')])

    def to_record(self) -> dict[str, Any]:
        return {'schema': 'wmmllm.graph.v1', 'nodes': deepcopy(list(self._nodes.values()))}

    @classmethod
    def from_record(cls, value: Mapping, *, run_root: Path) -> 'SceneGraph':
        if value.get('schema') not in ('wmmllm.graph.v1', 'wmmllm.session.v1') or not isinstance(value.get('nodes'), list):
            raise ValueError('unsupported graph record')
        graph = cls(run_root=run_root)
        remaining = list(value['nodes'])
        ids = [_identifier(node.get('node_id'), 'node ID') for node in remaining]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate graph node ID')
        while remaining:
            ready = [node for node in remaining if node.get('parent_node_id') is None
                     or node.get('parent_node_id') in graph._nodes]
            if not ready:
                raise ValueError('graph has missing parents or a cycle')
            for node in ready:
                graph.add_node(node)
                remaining.remove(node)
        return graph
