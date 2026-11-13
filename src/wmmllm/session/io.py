"""Atomic records and verified references contained in an explicit run root."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import uuid
from typing import Any

import numpy as np
from PIL import Image

from ..image_io import encode_rgb_png


class IntegrityError(ValueError):
    """A durable asset or its binding differs from the accepted record."""


class ImmutableRecordChanged(IntegrityError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve_run_path(run_root: Path, relative_path: str, *, must_exist: bool = True) -> Path:
    """Reject malformed serialized paths before touching their targets."""
    if (not isinstance(relative_path, str) or not relative_path or
            any(c in relative_path for c in ('\\', '\x00', '?', '#', ':')) or
            relative_path.startswith('/') or
            any(p in ('', '.', '..') for p in relative_path.split('/'))):
        raise IntegrityError('Invalid run-relative path')
    root = Path(run_root).resolve(strict=True)
    path = root.joinpath(*PurePosixPath(relative_path).parts)
    try:
        actual = path.resolve(strict=must_exist)
        actual.relative_to(root)
        # strict=False also resolves every existing parent, including symlinks.
        if must_exist and not actual.is_file():
            raise IntegrityError('Reference is not a regular file: ' + relative_path)
        ancestor = path
        while ancestor != root:
            if ancestor.is_symlink() and not ancestor.exists():
                raise IntegrityError('Dangling reference: ' + relative_path)
            ancestor = ancestor.parent
    except (OSError, RuntimeError, ValueError) as error:
        raise IntegrityError('Reference containment failed: ' + relative_path) from None
    return path


def _local(path: Path | str, *, run_root: Path, must_exist: bool = True) -> Path:
    root = Path(run_root).resolve(strict=True)
    value = Path(path)
    if value.is_absolute():
        try:
            relative = value.relative_to(root).as_posix()
        except ValueError:
            raise IntegrityError('Path lies outside the run root') from None
    else:
        relative = value.as_posix()
    return resolve_run_path(root, relative, must_exist=must_exist)


def file_ref(path: Path, *, run_root: Path, image: bool = False) -> dict:
    path = _local(path, run_root=run_root)
    result = {'path': path.relative_to(Path(run_root).resolve()).as_posix(),
              'sha256': sha256_file(path)}
    if image:
        pixels = np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
        digest = hashlib.sha256(f'RGB uint8 {pixels.shape[1]} {pixels.shape[0]}\n'.encode('ascii'))
        digest.update(pixels.tobytes())
        result.update(pixel_sha256=digest.hexdigest(),
                      size=[pixels.shape[1], pixels.shape[0]])
    return result


def public_ref(ref: Mapping, *, image: bool = False, state: bool = False) -> dict:
    """Select the fields declared by the session.v1 file-reference schema."""
    keys = ('path', 'sha256')
    if image:
        keys += ('pixel_sha256', 'size')
    if state:
        keys += ('panorama_state_sha256',)
    return {key: ref[key] for key in keys}


def verify_ref(ref: Mapping, *, run_root: Path) -> Path:
    if not isinstance(ref, Mapping) or not re.fullmatch('[a-f0-9]{64}', str(ref.get('sha256', ''))):
        raise IntegrityError('Invalid file reference')
    path = resolve_run_path(run_root, ref.get('path'))
    if sha256_file(path) != ref['sha256']:
        raise IntegrityError('Encoded hash mismatch: ' + ref['path'])
    if 'pixel_sha256' in ref or 'size' in ref:
        actual = file_ref(path, run_root=run_root, image=True)
        if any(actual.get(key) != ref.get(key) for key in ('pixel_sha256', 'size')):
            raise IntegrityError('Pixel hash or size mismatch: ' + ref['path'])
    if 'array_sha256' in ref:
        array = np.load(path, allow_pickle=False)
        digest = hashlib.sha256(f'{array.dtype.str} {array.shape}\n'.encode('ascii'))
        digest.update(np.ascontiguousarray(array).tobytes())
        if (digest.hexdigest() != ref['array_sha256'] or
                list(array.shape) != ref.get('shape') or str(array.dtype) != ref.get('dtype')):
            raise IntegrityError('Array hash or shape mismatch: ' + ref['path'])
    if 'panorama_state_sha256' in ref:
        from ..state.panorama import load_panorama_state, panorama_state_sha256
        # A path avoids recursive validation of this same outer reference.
        state = load_panorama_state(path, run_root=run_root)
        if panorama_state_sha256(state) != ref['panorama_state_sha256']:
            raise IntegrityError('Panorama canonical hash mismatch: ' + ref['path'])
    return path


def read_json(path: Path | Mapping, *, run_root: Path) -> Any:
    target = verify_ref(path, run_root=run_root) if isinstance(path, Mapping) else _local(path, run_root=run_root)
    return json.loads(target.read_text(encoding='utf-8'))


def verify_references(value: Any, *, run_root: Path) -> None:
    if isinstance(value, Mapping):
        if 'path' in value and 'sha256' in value:
            verify_ref(value, run_root=run_root)
        else:
            for child in value.values():
                verify_references(child, run_root=run_root)
    elif isinstance(value, (list, tuple)):
        for child in value:
            verify_references(child, run_root=run_root)


def _write(path: Path, value: Any, *, run_root: Path, immutable: bool) -> dict:
    path = _local(path, run_root=run_root, must_exist=False)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                         allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    path = _local(path, run_root=run_root, must_exist=False)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _local(path, run_root=run_root, must_exist=False)
        if immutable:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if canonical_json(read_json(path, run_root=run_root)) != canonical_json(value):
                    raise ImmutableRecordChanged('Immutable record changed: ' + path.name)
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return file_ref(path, run_root=run_root)


def write_atomic(path: Path, value: Any, *, run_root: Path) -> dict:
    return _write(path, value, run_root=run_root, immutable=False)


def write_once(path: Path, value: Any, *, run_root: Path) -> dict:
    return _write(path, value, run_root=run_root, immutable=True)


def write_bytes_once(path: Path, value: bytes, *, run_root: Path, image: bool = False) -> dict:
    path = _local(path, run_root=run_root, must_exist=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path = _local(path, run_root=run_root, must_exist=False)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        _local(path, run_root=run_root, must_exist=False)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if sha256_file(path) != hashlib.sha256(value).hexdigest():
                raise ImmutableRecordChanged('Immutable asset changed: ' + path.name)
    finally:
        temporary.unlink(missing_ok=True)
    return file_ref(path, run_root=run_root, image=image)


def save_png(path: Path, pixels: Any, *, run_root: Path) -> dict:
    image = pixels if isinstance(pixels, Image.Image) else Image.fromarray(np.asarray(pixels, dtype=np.uint8))
    if image.mode in ('L', 'LA', 'RGBA'):
        # Masks are typed pixel data: changing L to RGB prevents provider mask
        # validation, and dropping alpha changes its editing semantics.
        image.load()
        clean = Image.frombytes(image.mode, image.size, image.tobytes())
        stream = io.BytesIO()
        clean.save(stream, format='PNG')
        encoded = stream.getvalue()
        with Image.open(io.BytesIO(encoded)) as decoded:
            decoded.load()
            if (decoded.mode != clean.mode or decoded.size != clean.size or
                    decoded.tobytes() != clean.tobytes() or decoded.info or len(decoded.getexif())):
                raise IntegrityError('Mask PNG round trip changed pixels or retained metadata')
    else:
        encoded, _conversion = encode_rgb_png(image)
    return write_bytes_once(path, encoded, run_root=run_root, image=True)


@contextmanager
def file_lock(path: Path, *, run_root: Path, blocking: bool = True):
    path = _local(path, run_root=run_root, must_exist=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path = _local(path, run_root=run_root, must_exist=False)
    with path.open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def save_pending(path: Path, packet: Mapping, *, run_root: Path) -> dict:
    return write_once(path, dict(packet), run_root=run_root)


def recover_pending(path: Path, *, session) -> dict | None:
    path = _local(path, run_root=session.run_root, must_exist=False)
    if not path.exists():
        return None
    return session._recover_packet(read_json(path, run_root=session.run_root), path)


_ROOT_FIELDS = {'schema', 'session_id', 'title', 'seed', 'status', 'state_authority',
                'video_writes_state', 'initial_node_id', 'current_node_id', 'nodes',
                'transitions', 'memory'}
_NODE_FIELDS = {'node_id', 'parent_node_id', 'origin_id', 'physical_revision',
                'view_identity', 'pose', 'viewport_profile', 'panorama_state',
                'panorama', 'view', 'memory_snapshot', 'seam_status'}
_TURN_FIELDS = {'transition_id', 'index', 'from_node_id', 'to_node_id', 'caption',
                'operations', 'status', 'semantic', 'presentation', 'decision', 'memory_after'}
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}')


def validate_session(value: Mapping, *, run_root: Path, verify_assets: bool = True) -> dict:
    """Validate the v1 viewer handoff and its authoritative asset references."""
    def require(condition, reason):
        if not condition:
            raise IntegrityError('Invalid session: ' + reason)
    def identifier(item):
        return isinstance(item, str) and _ID.fullmatch(item)
    def asset(ref):
        require(isinstance(ref, Mapping), 'missing file reference')
        resolve_run_path(run_root, ref.get('path'), must_exist=verify_assets)
        require(bool(re.fullmatch('[a-f0-9]{64}', str(ref.get('sha256', '')))), 'file hash')
        if verify_assets:
            verify_ref(ref, run_root=run_root)
    require(isinstance(value, Mapping), 'root object')
    require(_ROOT_FIELDS <= value.keys() and not (value.keys() - _ROOT_FIELDS - {'extensions'}), 'root fields')
    require(value['schema'] == 'wmmllm.session.v1', 'schema')
    require(identifier(value['session_id']), 'session ID')
    require(value['status'] in ('ready', 'incomplete', 'failed'), 'status')
    require(value['state_authority'] == 'accepted_panorama_and_endpoint' and value['video_writes_state'] is False, 'state authority')
    require(isinstance(value['nodes'], list) and isinstance(value['transitions'], list), 'graph arrays')
    nodes = {}
    for node in value['nodes']:
        require(set(node) == _NODE_FIELDS, 'node fields')
        nid = node['node_id']
        require(identifier(nid) and nid not in nodes and identifier(node['origin_id']), 'node ID')
        require(type(node['physical_revision']) is int and node['physical_revision'] >= 0, 'revision')
        require(node['seam_status'] in ('resolved', 'seam_unresolved', 'not_checked'), 'seam status')
        require(set(node['pose']) == {'yaw', 'pitch', 'hfov'} and all(type(x) in (int, float) and math.isfinite(x) for x in node['pose'].values()), 'pose')
        profile = node['viewport_profile']
        require(set(profile) == {'width', 'height', 'content_box', 'padding_ltrb'}, 'viewport profile')
        require(all(type(profile[k]) is int and profile[k] > 0 for k in ('width', 'height')), 'viewport size')
        for key in ('panorama_state', 'panorama', 'view', 'memory_snapshot'):
            asset(node[key])
        nodes[nid] = node
    if not nodes:
        require(value['status'] in ('incomplete', 'failed') and value['initial_node_id'] is None and value['current_node_id'] is None and not value['transitions'] and value['memory'] is None, 'uninitialized graph')
        return dict(value)
    require(value['initial_node_id'] in nodes and value['current_node_id'] in nodes, 'active node')
    for node in nodes.values():
        require(node['parent_node_id'] is None or node['parent_node_id'] in nodes, 'parent node')
    require(nodes[value['initial_node_id']]['parent_node_id'] is None, 'root parent')
    tids = set()
    current = value['initial_node_id']
    for index, transition in enumerate(value['transitions'], 1):
        require(set(transition) == _TURN_FIELDS, 'transition fields')
        tid = transition['transition_id']
        require(identifier(tid) and tid not in tids and transition['index'] == index, 'transition ID/index')
        tids.add(tid)
        require(transition['from_node_id'] == current, 'transition continuity')
        require(transition['status'] in ('accepted', 'held', 'failed'), 'transition status')
        if transition['status'] == 'accepted':
            require(transition['to_node_id'] in nodes, 'accepted node')
            current = transition['to_node_id']
        elif transition['status'] == 'held':
            require(transition['to_node_id'] == current, 'held node')
        else:
            require(transition['to_node_id'] is None, 'failed node')
        require(all(op in ('rotate', 'revisit', 'navigate', 'local_edit', 'global_change', 'subject_action') for op in transition['operations']), 'operations')
        semantic = transition['semantic']
        require(set(semantic) == {'status','attempt_count','semantic_accepted','uncertain_accepted_and_labeled','verification'}, 'semantic fields')
        require(semantic['status'] in ('PASS','FAIL','UNCERTAIN','NOT_CHECKED') and semantic['attempt_count'] in (0,1,2), 'semantic status')
        accepted = transition['status'] == 'accepted'
        require(semantic['semantic_accepted'] is (accepted and semantic['status'] == 'PASS'), 'semantic acceptance')
        require(semantic['uncertain_accepted_and_labeled'] is (accepted and semantic['status'] == 'UNCERTAIN'), 'uncertain acceptance')
        require(not accepted or semantic['status'] in ('PASS','UNCERTAIN'), 'accepted semantic status')
        if semantic['verification'] is not None:
            asset(semantic['verification'])
        presentation = transition['presentation']
        require(set(presentation) == {'status','backend','video','reason','num_frames','fps','duration_seconds'}, 'presentation fields')
        require(identifier(presentation['backend']), 'presentation backend')
        if presentation['status'] == 'none':
            require(presentation['video'] is None and isinstance(presentation['reason'], str) and bool(presentation['reason']), 'no-video result')
        else:
            require(presentation['status'] in ('rendered','generated'), 'presentation status')
            asset(presentation['video'])
            require(type(presentation['num_frames']) is int and presentation['num_frames'] > 0 and presentation['fps'] > 0, 'presentation profile')
            require(presentation['duration_seconds'] == presentation['num_frames'] / presentation['fps'], 'presentation duration')
        if transition['decision'] is not None:
            asset(transition['decision'])
        if transition['memory_after'] is not None:
            asset(transition['memory_after'])
        require(transition['status'] == 'failed' or transition['memory_after'] is not None, 'memory after')
    require(current == value['current_node_id'], 'settled current node')
    asset(value['memory'])
    return dict(value)
