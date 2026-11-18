"""Single-writer local runner with accepted-state transaction recovery."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image

from .io import (IntegrityError, canonical_sha256, file_lock, file_ref, read_json,
                 resolve_run_path, validate_session, verify_ref, verify_references,
                 write_atomic, write_bytes_once, write_once)
from ..logging_utils import redact
from ..image_io import load_rgb_image, encode_rgb_png
from ..providers.retry import raise_if_fatal_provider_error, safe_provider_error
from ..agent import bootstrap
from ..agent.executor import execute_transition
from ..agent.failures import classify_failure, exception_record
from ..memory.store import MemoryStore
from ..state.graph import SceneGraph, node_record, visit_key
from ..state.panorama import load_panorama_state


def portable_settings(config) -> dict:
    """Persist algorithm/budget settings; credentials and endpoint identities stay external."""
    value = config.safe_dict()
    value['chat'].pop('model_configured')
    value['image'].pop('model_configured')
    value['budget'] = {'max_cost_usd': config.budget.max_cost_usd,
                       'prices': {role: asdict(price) for role, price in config.budget.prices.items()}}
    return value


class SessionRunner:
    def __init__(self, run_root: Path, *, deps):
        self.run_root = Path(run_root).resolve()
        self.run_root.mkdir(parents=True, exist_ok=True)
        if self.run_root != Path(deps.run_root).resolve():
            raise IntegrityError('Engine dependencies belong to another run root')
        self.deps = deps
        self.path = self.run_root / 'session.json'
        self.record = None
        self.store = None
        self.graph = None

    @property
    def current_node(self) -> dict:
        return next(node for node in self.record['nodes'] if node['node_id'] == self.record['current_node_id'])

    @property
    def profile(self) -> dict:
        v = self.deps.config.video
        return {'width': v.size[0], 'height': v.size[1], 'num_frames': v.num_frames, 'fps': v.fps}

    @property
    def context(self) -> dict:
        return read_json(self.run_root / 'bootstrap' / 'CONTEXT.json', run_root=self.run_root)

    @property
    def preceding(self) -> list[str]:
        return [commit['caption'] for commit in self.store.snapshot().commits]

    @property
    def visits(self) -> dict:
        result = {}
        for node in self.record['nodes']:
            result[visit_key(node['physical_revision'], node['view_identity'], node['pose'])] = node
        return result

    def _open_store(self):
        self.store = MemoryStore(self.run_root / 'memory', task_id=self.record['session_id'],
                                 setting='wmmllm.core.v1', scene_id='scene0000',
                                 state_indices=(0, 1), run_root=self.run_root)

    def _publish(self):
        validate_session(self.record, run_root=self.run_root)
        write_atomic(self.path, self.record, run_root=self.run_root)

    def _bootstrap_models(self) -> dict:
        # Keep configured model identities bound without serializing their names.
        return {role: canonical_sha256({'role': role, 'model': getattr(self.deps.config, role).model})
                for role in ('chat', 'image')}

    def initialize(self, initial_image: Path, *, captions: Sequence[str], seed: int = 0,
                   title: str = '', panorama_width: int = 4096) -> dict:
        """Start or retry initialization with the exact checkpoint-bound inputs."""
        if type(panorama_width) is not int or panorama_width < 64 or panorama_width % 2:
            raise ValueError('Panorama width must be an even integer of at least 64')
        if not all(isinstance(caption, str) for caption in captions):
            raise ValueError('Bootstrap captions must be text')
        captions = list(captions)
        with file_lock(self.run_root / 'session.lock', run_root=self.run_root):
            pixels, conversion = load_rgb_image(initial_image)
            target = (1344, 768)
            source_profile = {'content_box': [0, 0, target[0], target[1]], 'padding_ltrb': [0, 0, 0, 0]}
            if pixels.size != target:
                scale = min(target[0] / pixels.width, target[1] / pixels.height)
                size = (max(1, round(pixels.width * scale)), max(1, round(pixels.height * scale)))
                content = pixels.resize(size, Image.Resampling.LANCZOS)
                left, top = (target[0] - size[0]) // 2, (target[1] - size[1]) // 2
                pixels = Image.new('RGB', target, (0, 0, 0))
                pixels.paste(content, (left, top))
                source_profile = {'content_box': [left, top, left + size[0], top + size[1]],
                    'padding_ltrb': [left, top, target[0]-left-size[0], target[1]-top-size[1]]}
            encoded, _ = encode_rgb_png(pixels)
            if self.path.exists():
                self.record = validate_session(read_json(self.path, run_root=self.run_root), run_root=self.run_root)
                if self.record['nodes']:
                    raise IntegrityError('Session already exists; resume it instead')
                checkpoint = self._bootstrap_checkpoint()
                if (checkpoint['initial_image']['sha256'] != hashlib.sha256(encoded).hexdigest() or
                        checkpoint['captions'] != captions or checkpoint['seed'] != seed or
                        checkpoint['panorama_width'] != panorama_width or
                        read_json(checkpoint['source_profile'], run_root=self.run_root) != source_profile):
                    raise IntegrityError('Bootstrap retry inputs differ from the initialization checkpoint')
                return self._continue_initialization(checkpoint)
            initial = write_bytes_once(self.run_root / 'inputs' / 'initial.png', encoded,
                                       run_root=self.run_root, image=True)
            identity = canonical_sha256({'image': initial, 'seed': seed})[:20]
            self.record = {'schema': 'wmmllm.session.v1', 'session_id': 'session-' + identity,
                'title': redact(title or 'Local panorama session'), 'seed': int(seed),
                'status': 'incomplete', 'state_authority': 'accepted_panorama_and_endpoint',
                'video_writes_state': False, 'initial_node_id': None, 'current_node_id': None,
                'nodes': [], 'transitions': [], 'memory': None}
            settings = write_once(self.run_root / 'runtime_config.json', portable_settings(self.deps.config), run_root=self.run_root)
            profile = write_once(self.run_root / 'inputs' / 'source_profile.json', source_profile, run_root=self.run_root)
            checkpoint = {'schema': 'wmmllm.bootstrap_checkpoint.v1',
                'session_id': self.record['session_id'], 'initial_image': initial,
                'source_profile': profile, 'captions': captions, 'seed': int(seed),
                'panorama_width': panorama_width, 'render_profile': self.profile,
                'runtime_settings': settings, 'models': self._bootstrap_models(),
                'image_ingestion': conversion}
            checkpoint['binding_sha256'] = canonical_sha256(checkpoint)
            write_once(self.run_root / 'bootstrap' / 'CHECKPOINT.json', checkpoint, run_root=self.run_root)
            self.record['extensions'] = {'bootstrap': {'status': 'PENDING',
                'checkpoint': file_ref(self.run_root / 'bootstrap' / 'CHECKPOINT.json', run_root=self.run_root),
                'image_ingestion': conversion}}
            self._publish()
            return self._continue_initialization(checkpoint)

    def _bootstrap_checkpoint(self) -> dict:
        checkpoint_path = self.run_root / 'bootstrap' / 'CHECKPOINT.json'
        checkpoint = read_json(checkpoint_path, run_root=self.run_root)
        payload = {key: value for key, value in checkpoint.items() if key != 'binding_sha256'}
        if (checkpoint.get('schema') != 'wmmllm.bootstrap_checkpoint.v1' or
                checkpoint.get('binding_sha256') != canonical_sha256(payload) or
                checkpoint.get('session_id') != self.record['session_id'] or
                checkpoint.get('seed') != self.record['seed']):
            raise IntegrityError('Bootstrap initialization checkpoint binding differs')
        reference = self.record.get('extensions', {}).get('bootstrap', {}).get('checkpoint')
        if reference != file_ref(checkpoint_path, run_root=self.run_root):
            raise IntegrityError('Session initialization checkpoint reference differs')
        for key in ('initial_image', 'source_profile', 'runtime_settings'):
            verify_ref(checkpoint[key], run_root=self.run_root)
        if (checkpoint['render_profile'] != self.profile or
                read_json(checkpoint['runtime_settings'], run_root=self.run_root) != portable_settings(self.deps.config)):
            raise IntegrityError('Bootstrap runtime settings differ from the initialization checkpoint')
        if checkpoint.get('models') != self._bootstrap_models():
            raise IntegrityError('Bootstrap model identities differ from the initialization checkpoint')
        return checkpoint

    def _continue_initialization(self, checkpoint: Mapping) -> dict:
        # Revalidate before every attempt, including an in-process retry.
        checkpoint = self._bootstrap_checkpoint()
        captions = checkpoint['captions']
        source_profile = read_json(checkpoint['source_profile'], run_root=self.run_root)
        row = {'key': self.record['session_id'], 'task_id': self.record['session_id'],
            'segment': 1, 'seed': checkpoint['seed'],
            'initial_image': checkpoint['initial_image'],
            'current_caption': captions[0] if captions else 'Remain at the initial viewpoint.', 'profile': self.profile,
            '_item_instructions': ([{'segment': i + 1, 'caption': text} for i, text in enumerate(captions)]
                                   if captions else [{'segment': 0, 'caption': 'Remain at the initial viewpoint.'}]),
            'source_metadata': source_profile}
        # A fresh attempt never overwrites immutable artifacts from a failed
        # provider call or registration. A committed bootstrap is reused exactly.
        attempts = self.run_root / 'bootstrap' / 'attempts'
        attempts.mkdir(parents=True, exist_ok=True)
        existing = sorted(attempts.glob('attempt*.json'))
        for number, path in enumerate(existing, 1):
            attempt = read_json(path, run_root=self.run_root)
            relative = '.' if number == 1 else f'bootstrap/retries/attempt{number:03d}'
            expected = {'schema': 'wmmllm.bootstrap_attempt.v1', 'attempt': number,
                        'checkpoint_sha256': checkpoint['binding_sha256'], 'output_root': relative}
            if path.name != f'attempt{number:03d}.json' or attempt != expected:
                raise IntegrityError('Bootstrap attempt belongs to another initialization')
        active = read_json(existing[-1], run_root=self.run_root) if existing else None
        if active is not None:
            output = self.run_root if active['output_root'] == '.' else resolve_run_path(self.run_root, active['output_root'], must_exist=False)
            committed = output / row['key'] / 'bootstrap' / 'COMMITTED.json'
        else:
            committed = None
        if active is None or (not committed.is_file() and not (self.run_root / 'bootstrap' / 'ACCEPTED_READY.json').is_file()):
            number = len(existing) + 1
            relative = '.' if number == 1 else f'bootstrap/retries/attempt{number:03d}'
            active = {'schema': 'wmmllm.bootstrap_attempt.v1', 'attempt': number,
                      'checkpoint_sha256': checkpoint['binding_sha256'], 'output_root': relative}
            write_once(attempts / f'attempt{number:03d}.json', active, run_root=self.run_root)
            output = self.run_root if relative == '.' else resolve_run_path(self.run_root, relative, must_exist=False)
        result_path = self.run_root / 'bootstrap' / 'RESULT.json'
        ready_path = self.run_root / 'bootstrap' / 'ACCEPTED_READY.json'
        unpublished = deepcopy(self.record)
        try:
            if ready_path.exists():
                ready = read_json(ready_path, run_root=self.run_root)
                payload = {key: value for key, value in ready.items() if key != 'sha256'}
                if (ready.get('checkpoint_sha256') != checkpoint['binding_sha256'] or
                        ready.get('sha256') != canonical_sha256(payload)):
                    raise IntegrityError('Saved bootstrap result initialization binding differs')
                ref = self.record['extensions']['bootstrap'].get('accepted_ready')
                if ref is not None and verify_ref(ref, run_root=self.run_root) != ready_path:
                    raise IntegrityError('Saved bootstrap result reference differs')
                boot = ready['bootstrap']
            else:
                boot = bootstrap.initialize(row, output, width=checkpoint['panorama_width'], deps=self.deps)
            for key in ('panorama_state', 'panorama', 'endpoint'):
                verify_ref(boot[key], run_root=self.run_root)
            ready = {'schema': 'wmmllm.bootstrap_ready.v1',
                     'checkpoint_sha256': checkpoint['binding_sha256'], 'bootstrap': boot}
            ready['sha256'] = canonical_sha256(ready)
            ready_ref = write_once(ready_path, ready, run_root=self.run_root)
            self.record['extensions']['bootstrap']['accepted_ready'] = ready_ref
            write_once(self.run_root / 'bootstrap' / 'CONTEXT.json', boot['context'], run_root=self.run_root)
            write_once(result_path, boot, run_root=self.run_root)
            self._open_store()
            self.store.initialize(boot['endpoint'])
            memory = self.store.snapshot_to(self.run_root / 'memory' / 'snapshots' / 'node0000.json')
            state = load_panorama_state(boot['panorama_state'], run_root=self.run_root)
            pose = {**boot['camera'], 'hfov': 90.0}
            node = node_record(node_id='node0000', parent_node_id=None,
                origin_id='origin0000', physical_revision=0,
                view_identity={'camera_origin': 'origin0000'}, pose=pose,
                panorama_state=boot['panorama_state'], panorama=boot['panorama'], view=boot['endpoint'],
                memory_snapshot=memory, viewport_profile=state['viewport_profile'],
                seam_status='seam_unresolved' if boot.get('seam_status') == 'seam_unresolved' else 'resolved')
            self.graph = SceneGraph(run_root=self.run_root)
            self.graph.add_node(node)
            self.record.update(status='ready', initial_node_id='node0000', current_node_id='node0000',
                               nodes=[node], memory=file_ref(self.store.path, run_root=self.run_root))
            label = self.record['extensions']['bootstrap']
            for stale in ('hold', 'source', 'endpoint', 'error'):
                label.pop(stale, None)
            label.update(status='ACCEPTED', attempt=active['attempt'])
            self._publish()
        except Exception as error:
            # A failure while publishing the first node must remain an
            # uninitialized checkpoint. Any saved endpoint/memory is replayed
            # idempotently from ACCEPTED_READY on resume.
            progress = deepcopy(self.record['extensions']['bootstrap'])
            self.record = unpublished
            self.record['extensions']['bootstrap'] = progress
            error_record = safe_provider_error(error)
            write_atomic(self.run_root / 'bootstrap' / 'ERROR.json', error_record, run_root=self.run_root)
            self.record['status'] = 'failed'
            self.record['extensions']['bootstrap'].update(status='FAILED', attempt=active['attempt'], error=error_record)
            self._publish()
            raise_if_fatal_provider_error(error)
            classification = classify_failure(type(error).__name__, str(error), exception_record(error))
            if (classification['hold_eligible'] and
                    classification['classification'] in ('bootstrap_registration', 'provider_refusal') and
                    not self.record['nodes']):
                hold = {'schema': 'wmmllm.bootstrap_hold.v1', 'status': 'BOOTSTRAP_HOLD',
                    'attempt': active['attempt'], 'checkpoint_sha256': checkpoint['binding_sha256'],
                    'source': checkpoint['initial_image'], 'endpoint': checkpoint['initial_image'],
                    'failure_classification': classification['classification'], 'error': error_record,
                    'world_initialized': False, 'accepted_panorama_created': False,
                    'panorama_state': None, 'panorama': None, 'memory': None,
                    'new_facts': 0, 'state_or_memory_commit_calls': 0,
                    'semantic_memory_observation_calls': 0, 'video_backend_calls': 0,
                    'accepted_state_unchanged': True, 'action_completed': False}
                hold_ref = write_once(self.run_root / 'bootstrap' / f'HOLD{active["attempt"]:03d}.json', hold, run_root=self.run_root)
                self.record['status'] = 'incomplete'
                self.record['extensions']['bootstrap'].update(status='BOOTSTRAP_HOLD',
                    hold=hold_ref, source=checkpoint['initial_image'], endpoint=checkpoint['initial_image'])
                self._publish()
                return self.session_record()
            raise
        return self.session_record()

    def _load(self):
        pending = self.run_root / 'pending'
        has_pending = pending.exists() and any(pending.glob('turn*.json'))
        self.record = validate_session(read_json(self.path, run_root=self.run_root), run_root=self.run_root,
                                       verify_assets=not has_pending)
        if has_pending:
            # MEMORY.json may already contain the pending commit. All immutable
            # state/endpoint/snapshot references remain mandatory.
            for node in self.record['nodes']:
                for key in ('panorama_state', 'panorama', 'view', 'memory_snapshot'):
                    verify_ref(node[key], run_root=self.run_root)
        if not self.record['nodes']:
            raise IntegrityError('Session has no accepted bootstrap state')
        self._open_store()
        self.graph = SceneGraph(run_root=self.run_root)
        for node in self.record['nodes']:
            self.graph.add_node(node)

    def resume(self) -> dict:
        with file_lock(self.run_root / 'session.lock', run_root=self.run_root):
            self.record = validate_session(read_json(self.path, run_root=self.run_root),
                                           run_root=self.run_root, verify_assets=False)
            if not self.record['nodes']:
                return self._continue_initialization(self._bootstrap_checkpoint())
            self._load()
            self._recover_pending()
            return self.session_record()

    def step(self, caption: str) -> dict:
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError('Instruction must be nonempty text')
        with file_lock(self.run_root / 'session.lock', run_root=self.run_root):
            self._load()
            self._recover_pending()
            if self.record['status'] != 'ready':
                raise IntegrityError('Session is not ready for another instruction')
            if (self.profile['width'], self.profile['height']) != (self.current_node['viewport_profile']['width'], self.current_node['viewport_profile']['height']):
                raise IntegrityError('Session viewport differs from the configured rendering profile')
            index = len(self.record['transitions']) + 1
            seed = int(hashlib.sha256((self.record['session_id'] + ':' + str(index)).encode()).hexdigest()[:8], 16) % (2**31)
            row = {'key': self.record['session_id'], 'task_id': self.record['session_id'],
                'segment': index, 'seed': seed, 'current_caption': caption, 'profile': self.profile}
            return execute_transition(row, session=self, deps=self.deps)

    def _recover_pending(self):
        for path in sorted((self.run_root / 'pending').glob('turn*.json')):
            self._recover_packet(read_json(path, run_root=self.run_root), path)

    def _recover_packet(self, packet: Mapping, path: Path) -> dict:
        """Replay only accepted, fully bound packets; no provider/presentation calls."""
        if packet.get('schema') != 'wmmllm.pending.v1':
            raise IntegrityError('Unknown pending transaction schema')
        binding = packet['binding']
        packet_hash = canonical_sha256({k:v for k,v in packet.items() if k != 'packet_sha256'})
        if packet_hash != packet.get('packet_sha256'):
            raise IntegrityError('Pending packet checksum mismatch')
        if binding.get('session_id') != self.record['session_id'] or binding.get('method') != 'wmmllm.core.v1':
            raise IntegrityError('Pending transaction belongs to another session/method')
        transition = deepcopy(packet['transition'])
        commit = packet['memory_commit']
        held = packet['held']
        if type(held) is not bool or transition['status'] != ('held' if held else 'accepted'):
            raise IntegrityError('Pending status and hold flag differ')
        if (binding['decision'] != transition['decision'] or binding['caption'] != commit['caption'] or
                transition['index'] != commit['segment'] or transition['from_node_id'] != binding['from_node_id']):
            raise IntegrityError('Pending decision/caption/segment binding differs')
        if held:
            if packet['node'] is not None or transition['to_node_id'] != transition['from_node_id'] or commit['source'] != binding['source']:
                raise IntegrityError('Pending hold changes accepted state')
            if transition['semantic']['semantic_accepted'] or transition['semantic']['uncertain_accepted_and_labeled']:
                raise IntegrityError('Pending hold claims semantic acceptance')
        else:
            node = packet['node']
            semantic = transition['semantic']
            if (node is None or transition['to_node_id'] != node['node_id'] or commit['endpoint'] != node['view'] or
                    commit['start'] != binding['source'] or semantic['status'] not in ('PASS','UNCERTAIN') or
                    semantic['semantic_accepted'] is not (semantic['status']=='PASS') or
                    semantic['uncertain_accepted_and_labeled'] is not (semantic['status']=='UNCERTAIN')):
                raise IntegrityError('Pending accepted endpoint/semantic binding differs')
        # The whole packet contains only existing assets and explicit nulls for
        # the not-yet-created memory snapshot. Even a settled replay verifies it.
        verify_references(packet, run_root=self.run_root)
        prior = next((t for t in self.record['transitions'] if t['transition_id'] == transition['transition_id']), None)
        if prior is not None:
            candidate = {**transition, 'memory_after': prior['memory_after']}
            if prior != candidate:
                raise IntegrityError('Pending transaction conflicts with its settled transition')
            path.unlink()
            return deepcopy(prior)
        current = self.current_node
        if (binding['from_node_id'] != current['node_id'] or binding['source'] != current['view'] or
                binding['panorama_state'] != current['panorama_state'] or
                binding['physical_revision'] != current['physical_revision'] or
                binding['profile'] != self.profile or transition['index'] != len(self.record['transitions']) + 1):
            raise IntegrityError('Pending input/revision/profile binding differs')
        # Validate the final record shape before advancing the memory journal.
        # Its future immutable snapshot occupies an already verified snapshot
        # reference for this structural validation only.
        projected = deepcopy(self.record)
        projected_turn = {**transition, 'memory_after': current['memory_snapshot']}
        if not held:
            projected['nodes'].append({**packet['node'], 'memory_snapshot': current['memory_snapshot']})
            projected['current_node_id'] = packet['node']['node_id']
        projected['transitions'].append(projected_turn)
        validate_session(projected, run_root=self.run_root, verify_assets=False)
        if packet['held']:
            self.store.commit_hold(**packet['memory_commit'])
            memory_ref = self.store.snapshot_to(self.run_root / 'memory' / 'snapshots' / (transition['transition_id'] + '-held.json'))
        else:
            self.store.commit(**packet['memory_commit'])
            if packet.get('restore_semantics'):
                self.store.restore_semantics(packet['restore_semantics'], self.run_root / 'transitions' / transition['transition_id'] / 'MEMORY_RESTORE.json')
            nid = packet['node']['node_id']
            memory_ref = self.store.snapshot_to(self.run_root / 'memory' / 'snapshots' / (nid + '.json'))
            spec = deepcopy(packet['node'])
            spec['memory_snapshot'] = memory_ref
            node = node_record(**spec)
            self.graph.add_node(node)
            self.record['nodes'].append(node)
            self.record['current_node_id'] = nid
        transition['memory_after'] = memory_ref
        write_once(self.run_root / 'transitions' / transition['transition_id'] / 'transition.json', transition, run_root=self.run_root)
        self.record['transitions'].append(transition)
        self.record['memory'] = file_ref(self.store.path, run_root=self.run_root)
        self._publish()
        path.unlink()
        return deepcopy(transition)

    def settle(self, packet: dict) -> dict:
        packet['packet_sha256'] = canonical_sha256(packet)
        path = self.run_root / 'pending' / (packet['transition']['transition_id'] + '.json')
        write_once(path, dict(packet), run_root=self.run_root)
        return self._recover_packet(packet, path)

    def session_record(self) -> dict:
        if self.record is None:
            self._load()
        return deepcopy(self.record)
