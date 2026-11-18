"""Real session transactions with explicit test-only procedural providers (not models).

Only bootstrap generation is patched; every later stage runs the production code.
"""
from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import pytest

from wmmllm.agent.executor import EngineDeps, EndpointRejected
from wmmllm.budget import BudgetController
from wmmllm.config import AppConfig, ChatConfig, ImageConfig, VideoConfig
from wmmllm.providers.errors import TransportExhausted
from wmmllm.providers.factory import Providers
from wmmllm.providers.types import AttemptReceipt, CallReceipt, ChatResponse, ImageEditResult
from wmmllm.providers.video import NoneVideoBackend
from wmmllm.session.io import IntegrityError, canonical_json, read_json, save_png, sha256_file, write_atomic
from wmmllm.session.runner import SessionRunner
from wmmllm.state.panorama import create_panorama_state, render_view, save_panorama_state

SEMANTIC_FIELDS = ('state_matches_source', 'action_preserves_request',
                   'endpoint_matches_action', 'scene_identity_preserved')
POLYGON = [[.35, .30], [.65, .30], [.65, .70], [.35, .70]]


def _receipt(kind, role, value):
    digest = hashlib.sha256(canonical_json(value)).hexdigest()
    return CallReceipt(kind, role, 'test-procedural-not-a-model', 'test-procedural-not-a-model',
        'known', None, 'unknown', 'complete', 'test-only', 0., digest, digest,
        (), (AttemptReceipt(1, 'complete', 0.),))


class TestOnlyQueuedChat:
    """Not a model. Explicit queued decisions exercise the real engine."""
    __test__ = False
    route = 'test-only-procedural-not-a-model'

    def __init__(self):
        self.routes, self.verdicts, self.observer_errors = deque(), deque(), deque()
        self.requests, self.active, self.caption = [], None, ''

    def counts(self):
        return Counter(request.response_format['json_schema']['name'] for request in self.requests)

    def complete(self, request):
        self.requests.append(request)
        name = request.response_format['json_schema']['name']
        if name == 'aiwm_operation_plan':
            if not self.routes:
                raise AssertionError('A test operation must be queued explicitly')
            self.active = dict(self.routes.popleft())
            self.caption = request.payload['instruction']
            operation = self.active['operation']
            pose = {'yaw': float(self.active.get('yaw', request.payload['current_camera']['yaw'])),
                    'pitch': float(self.active.get('pitch', request.payload['current_camera']['pitch']))}
            local = operation == 'local_edit'
            value = {'operations': [{'operation': operation,
                'targets': ['yellow marker'] if local else [],
                'scope': 'local' if local else ('global' if operation == 'navigate' else 'camera_only'),
                'instruction': self.caption}], 'ending_camera': pose,
                'editing_camera': dict(request.payload['current_camera']), 'editing_hfov': 90.,
                'restore_node': self.active.get('restore_node', ''),
                'physical_state_changed': self.active.get('physical_state_changed', local),
                'endpoint_instruction': 'Paint a yellow marker on the middle tile.' if local else
                    'Show the procedural room at the requested camera location.',
                'decision_reason': 'Explicit test fixture decision; not a model.'}
        elif name in ('source_grounded_transition_action', 'visible_state_remaining_action'):
            value = {'current_state': 'A procedural room contains tiled geometric marks.',
                'remaining_action': self.caption,
                'endpoint_instruction': ('A yellow marker is on the middle tile.'
                    if self.active['operation'] == 'local_edit' else
                    'The procedural room is visible from the destination location.'),
                'scope_bbox': [.30, .25, .70, .75], 'support_polygons': [POLYGON],
                'final_view_entity_bboxes': [{'entity': 'yellow marker', 'bbox': [.42, .40, .58, .60]}]
                    if self.active['operation'] == 'local_edit' else []}
        elif name == 'actual_generated_edit_support':
            value = {'support_polygons': [POLYGON]}
        elif name in ('endpoint_semantic_verification', 'w34_combined_navigation_verification'):
            status = self.verdicts.popleft() if self.verdicts else 'pass'
            semantic = {**{field: status for field in SEMANTIC_FIELDS},
                        'reason': 'Explicit procedural test verdict; not a model.'}
            value = semantic if name == 'endpoint_semantic_verification' else {
                'criteria': {key: status.upper() for key in ('projection', 'movement', 'spatial', 'entity')},
                'semantic': semantic, 'reason': semantic['reason'], 'retry_instruction': ''}
        elif name == 'accepted_state_memory_observation':
            if self.observer_errors:
                raise self.observer_errors.popleft()
            value = {'summary': 'Geometric marks are visible in the supplied endpoints.', 'facts': [{
                'subject_id': 'room', 'label': 'room', 'key': 'appearance', 'kind': 'appearance',
                'value': 'procedural geometric marks', 'evidence': 'Marks appear in both supplied endpoint images.',
                'frame_indices': [0, 1]}]}
        else:
            raise AssertionError('Unexpected test schema: ' + name)
        return ChatResponse(json.dumps(value), 'test-procedural-not-a-model',
                            _receipt('chat', request.role, value))

    def close(self):
        pass


class TestOnlyDrawnImage:
    """Not a model. Draws the explicitly labeled procedural test proposal."""
    __test__ = False
    route = 'test-only-drawn-not-a-model'

    def __init__(self):
        self.requests = []

    def edit(self, request):
        self.requests.append(request)
        image = Image.open(request.images[0].path).convert('RGB')
        if request.panorama:
            pixels = np.roll(np.asarray(image), max(1, image.width // 12), axis=1).copy()
            pixels[:, image.width // 3:image.width // 2, 1] = 190
            image = Image.fromarray(pixels)
        else:
            w, h = image.size
            ImageDraw.Draw(image).rectangle((int(w * .42), int(h * .4), int(w * .58), int(h * .6)), fill=(245, 205, 40))
        Path(request.output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(request.output_dir) / 'endpoint.png'
        with path.open('xb') as stream:
            image.save(stream, format='PNG')
        return ImageEditResult(path,
            _receipt('image', 'editor', {'pixel_bytes_sha256': hashlib.sha256(np.asarray(image).tobytes()).hexdigest()}))

    def close(self):
        pass


class TestOnlyNoneVideo(NoneVideoBackend):
    __test__ = False
    def __init__(self):
        self.requests = []
    def present(self, transition):
        self.requests.append(transition)
        return super().present(transition)


def _procedural_pixels(width, height):
    y, x = np.indices((height, width))
    return np.stack((40 + (x // 37 % 3) * 40 + (y // 53 % 2) * 12,
                     50 + (y // 43 % 3) * 35 + (x // 59 % 2) * 10,
                     65 + (x // 61 % 2) * 60 + (y // 71 % 2) * 15), axis=-1).astype(np.uint8)


@pytest.fixture
def make_session(tmp_path, monkeypatch):
    """Patch generation-only bootstrap and retain all later real state logic."""
    from wmmllm.agent import bootstrap

    def procedural_bootstrap(row, output_root, context=None, context_ref=None, *, width=4096, deps):
        # A small real ERP plus the native procedural source is sufficient for
        # these transaction invariants. This is not a replacement live backend.
        source = Image.open(deps.run_root / row['initial_image']['path']).convert('RGB')
        size = deps.config.video.size
        source = np.asarray(source.resize(size, Image.Resampling.LANCZOS))
        erp = _procedural_pixels(512, 256)
        state = create_panorama_state(erp, source, width=512, height=256,
                                       source_content_box=[0, 0, size[0], size[1]])
        saved = save_panorama_state(state, deps.run_root / 'states/origin0000/revision0000', run_root=deps.run_root)
        endpoint = save_png(deps.run_root / 'bootstrap/endpoint.png', render_view(state, width=size[0], height=size[1]), run_root=deps.run_root)
        path = {'camera_path': [{'segment': i, 'yaw': float(yaw), 'pitch': 0.}
                               for i, yaw in enumerate((0, 45, 90, 135, 180, 225, 270, 315, 360, 0))]}
        return {'panorama_state': saved, 'panorama': saved['erp_preview_ref'], 'endpoint': endpoint,
                'camera': {'yaw': 0., 'pitch': 0.}, 'seam_status': 'resolved',
                'seam_gate': {'planned_path': path},
                'context': {'scene_description': 'A procedural tiled room.',
                            'style_tokens': 'Geometric flat colors',
                            'entity_identity_properties': 'Repeated geometric marks'}}

    monkeypatch.setattr(bootstrap, 'initialize', procedural_bootstrap)
    made = []

    def create(*, canonical=False):
        root = tmp_path / ('session' + str(len(made)))
        root.mkdir()
        seed = tmp_path / ('seed' + str(len(made)) + '.png')
        Image.fromarray(_procedural_pixels(1344, 768)).save(seed)
        size = (1344, 768) if canonical else (320, 180)
        config = AppConfig(ChatConfig('test-only', 'test-not-a-model', base_url='http://offline.invalid/v1'),
            ImageConfig('test-only', 'test-not-a-model', base_url='http://offline.invalid/v1'), video=VideoConfig(size=size, num_frames=3, fps=3.))
        chat, image, video = TestOnlyQueuedChat(), TestOnlyDrawnImage(), TestOnlyNoneVideo()
        providers = Providers(chat, image, video)
        budget = BudgetController(config.budget,
            checkpoint=lambda value: write_atomic(root / 'budget/checkpoint.json', value, run_root=root))
        deps = EngineDeps(config, providers, budget, root, None)
        runner = SessionRunner(root, deps=deps)
        record = runner.initialize(seed, captions=['Follow the explicit test operations.'],
                                   title='TEST PROCEDURAL FIXTURE - not a model')
        fixture = SimpleNamespace(runner=runner, root=root, deps=deps, chat=chat, image=image,
                                  video=video, initial=record['nodes'][0])
        made.append(fixture)
        return fixture

    yield create
    for fixture in made:
        fixture.deps.providers.close()


def _step(fixture, operation, caption, *, yaw=None, restore_node='', physical_state_changed=None):
    route = {'operation': operation, 'restore_node': restore_node}
    if yaw is not None:
        route['yaw'] = yaw
    if physical_state_changed is not None:
        route['physical_state_changed'] = physical_state_changed
    fixture.chat.routes.append(route)
    return fixture.runner.step(caption)


def _calls(fixture):
    return dict(fixture.chat.counts()), len(fixture.image.requests), len(fixture.video.requests)


def test_session_camera_return_is_same_revision_pixel_and_byte_exact(make_session):
    f = make_session()
    initial = f.initial
    assert _step(f, 'rotate', 'Look right.', yaw=45)['status'] == 'accepted'
    returned = _step(f, 'revisit', 'Return to the initial viewing direction.', yaw=0)
    assert returned['status'] == 'accepted'
    node = f.runner.current_node
    assert node['physical_revision'] == initial['physical_revision'] == 0
    assert node['origin_id'] == initial['origin_id']; assert node['view'] == initial['view']
    assert node['panorama_state'] == initial['panorama_state']
    assert f.runner.store.snapshot().completed_segments == 2; assert len(f.image.requests) == 0
    receipts = [json.loads(path.read_text()) for path in (f.root / 'receipts').glob('*.json')]
    assert len(receipts) == len(f.chat.requests)
    assert all('returned_model' in receipt and 'requested_model' in receipt for receipt in receipts)
    assert all(not {'payload', 'system', 'images'} & receipt.keys() for receipt in receipts)


def test_local_edit_revision_then_turnback_uses_new_pixels_and_rejects_stale_target(make_session):
    f = make_session(canonical=True)
    initial = deepcopy(f.initial)
    edit = _step(f, 'local_edit', 'Paint a yellow marker on the middle tile.')
    assert edit['status'] == 'accepted'
    edited = deepcopy(f.runner.current_node)
    assert edited['physical_revision'] == 1
    assert edited['view']['pixel_sha256'] != initial['view']['pixel_sha256']
    _step(f, 'rotate', 'Look right after painting.', yaw=45)
    _step(f, 'revisit', 'Look back toward the edited tile.', yaw=0)
    assert f.runner.current_node['physical_revision'] == 1; assert f.runner.current_node['view'] == edited['view']
    before = deepcopy(f.runner.session_record())
    with pytest.raises(IntegrityError, match='stale physical revision'):
        _step(f, 'revisit', 'Restore the stale pre-edit view.', yaw=0, restore_node=initial['node_id'])
    assert f.runner.session_record() == before; assert f.runner.store.snapshot().completed_segments == 3
    assert len(f.image.requests) == 1


def test_navigation_new_origin_then_exact_current_revision_return(make_session):
    f = make_session()
    initial = deepcopy(f.initial)
    first = _step(f, 'navigate', 'Walk to a new camera location.', physical_state_changed=False)
    assert first['status'] == 'accepted'
    destination = deepcopy(f.runner.current_node)
    assert destination['origin_id'] != initial['origin_id']
    assert destination['physical_revision'] == initial['physical_revision'] == 0
    assert destination['panorama_state'] != initial['panorama_state']
    returned = _step(f, 'revisit', 'Return to the exact saved initial location.', yaw=0,
                     restore_node=initial['node_id'])
    assert returned['status'] == 'accepted'; assert f.runner.current_node['origin_id'] == initial['origin_id']
    assert f.runner.current_node['view'] == initial['view']
    assert f.runner.current_node['panorama_state'] == initial['panorama_state']
    assert f.runner.current_node['physical_revision'] == 0
    assert f.runner.store.snapshot().completed_segments == 2; assert len(f.image.requests) == 1
    verifier = [r for r in f.chat.requests if r.response_format['json_schema']['name'] == 'endpoint_semantic_verification'][-1]
    assert verifier.images[0].sha256 == destination['view']['sha256']


@pytest.mark.parametrize('verdicts,status', [(['fail', 'fail'], 'held'), (['fail', 'pass'], 'accepted')])
def test_generated_semantic_retry_is_exactly_two_proposals(make_session, verdicts, status):
    f = make_session(canonical=True)
    f.chat.verdicts.extend(verdicts)
    result = _step(f, 'local_edit', 'Paint a yellow marker on the middle tile.')
    counts = f.chat.counts()
    assert result['status'] == status and result['semantic']['attempt_count'] == 2
    assert len(f.image.requests) == 2; assert counts['aiwm_operation_plan'] == 1
    assert counts['actual_generated_edit_support'] == 2; assert counts['endpoint_semantic_verification'] == 2
    assert counts['source_grounded_transition_action'] + counts['visible_state_remaining_action'] == 2
    assert f.runner.store.snapshot().completed_segments == 1
    if status == 'held':
        assert f.runner.current_node == f.initial; assert len(f.runner.session_record()['nodes']) == 1
        assert counts['accepted_state_memory_observation'] == 0; assert not f.runner.store.snapshot().subjects
        assert result['to_node_id'] == result['from_node_id']
    else:
        assert f.runner.current_node['physical_revision'] == 1
        assert counts['accepted_state_memory_observation'] == 1


def test_geometric_semantic_failure_holds_without_image_regeneration(make_session):
    f = make_session()
    f.chat.verdicts.append('fail')
    result = _step(f, 'rotate', 'Look right.', yaw=45)
    assert result['status'] == 'held' and result['semantic']['attempt_count'] == 1
    assert f.runner.current_node == f.initial; assert f.chat.counts()['endpoint_semantic_verification'] == 1
    assert f.chat.counts()['accepted_state_memory_observation'] == 0; assert not f.image.requests


def test_observer_transport_error_keeps_accepted_ready_and_resumes_observation_only(make_session):
    f = make_session()
    f.chat.observer_errors.extend([TransportExhausted('ReadTimeout'), TransportExhausted('ReadTimeout')])
    f.chat.routes.append({'operation': 'rotate', 'yaw': 45})
    with pytest.raises(EndpointRejected, match='Geometric observer failed'):
        f.runner.step('Look right.')
    assert f.runner.current_node == f.initial; assert f.runner.store.snapshot().completed_segments == 0
    assert f.runner.session_record()['transitions'] == []
    ready = f.root / 'transitions/turn0001/ACCEPTED_READY.json'
    assert ready.is_file(); assert not (f.root / 'transitions/turn0001/FAILURE_HOLD.json').exists()
    assert not (f.root / 'pending/turn0001.json').exists()
    before = _calls(f)
    result = f.runner.step('Look right.')
    assert result['status'] == 'accepted'
    after = _calls(f)
    expected = Counter(before[0])
    expected['accepted_state_memory_observation'] += 1
    assert Counter(after[0]) == expected; assert after[1:] == before[1:]
    assert f.runner.store.snapshot().completed_segments == 1
    resumed = read_json(f.root / 'transitions/turn0001/observation/transport_resume_001/OBSERVATION.json', run_root=f.root)
    assert resumed['status'] == 'OBSERVED'


class SimulatedPublicationCrash(RuntimeError):
    """Injected crash between durable steps."""


def test_pending_after_memory_commit_recovers_once_with_zero_new_calls(make_session, monkeypatch):
    f = make_session()

    def crash_after_commit():
        assert f.runner.store.snapshot().completed_segments == 1
        raise SimulatedPublicationCrash('injected after memory commit, before session publication')

    monkeypatch.setattr(f.runner, '_publish', crash_after_commit)
    with pytest.raises(SimulatedPublicationCrash):
        _step(f, 'rotate', 'Look right.', yaw=45)
    pending = f.root / 'pending/turn0001.json'
    packet_bytes = pending.read_bytes()
    assert f.runner.store.snapshot().completed_segments == 1
    assert read_json(f.root / 'session.json', run_root=f.root)['transitions'] == []
    before = _calls(f)
    recovered = SessionRunner(f.root, deps=f.deps)
    record = recovered.resume()
    assert _calls(f) == before; assert len(record['transitions']) == 1 and len(record['nodes']) == 2
    assert recovered.store.snapshot().completed_segments == 1; assert not pending.exists()
    # A duplicate durable packet after publication is an idempotent cleanup.
    pending.write_bytes(packet_bytes)
    record2 = recovered.resume()
    assert record2 == record; assert _calls(f) == before
    assert recovered.store.snapshot().completed_segments == 1; assert not pending.exists()


@pytest.mark.parametrize('field', ['memory_endpoint', 'node_endpoint'])
def test_pending_rejects_valid_but_crossbound_endpoint_before_memory_commit(make_session, monkeypatch, field):
    f = make_session()

    def stop_before_commit(packet, path):
        raise SimulatedPublicationCrash('injected after pending write, before memory commit')

    monkeypatch.setattr(f.runner, '_recover_packet', stop_before_commit)
    with pytest.raises(SimulatedPublicationCrash):
        _step(f, 'rotate', 'Look right.', yaw=45)
    pending = f.root / 'pending/turn0001.json'
    packet = json.loads(pending.read_text())
    if field == 'memory_endpoint':
        packet['memory_commit']['endpoint'] = deepcopy(f.initial['view'])
    else:
        packet['node']['view'] = deepcopy(f.initial['view'])
    packet['packet_sha256'] = hashlib.sha256(canonical_json({k: v for k, v in packet.items() if k != 'packet_sha256'})).hexdigest()
    pending.write_text(json.dumps(packet))
    before = _calls(f)
    memory_hash = sha256_file(f.root / 'memory/MEMORY.json')
    recovered = SessionRunner(f.root, deps=f.deps)
    with pytest.raises(IntegrityError):
        recovered.resume()
    assert _calls(f) == before; assert recovered.store.snapshot().completed_segments == 0
    assert sha256_file(f.root / 'memory/MEMORY.json') == memory_hash
