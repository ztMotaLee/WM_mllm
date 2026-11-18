"""Initialization checkpoints, method holds and optional provider failures."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from wmmllm.agent import bootstrap, executor
from wmmllm.config import AppConfig, ChatConfig, ImageConfig
from wmmllm.providers.errors import ProviderError, ProviderRefusal, TerminalQuotaError, TransportExhausted
from wmmllm.session.io import (IntegrityError, canonical_json, file_ref, read_json,
                               save_png, verify_ref)
from wmmllm.session.runner import SessionRunner
from wmmllm.state.panorama import create_panorama_state, render_view, save_panorama_state
from wmmllm.state.registration import RegistrationFailure


def _runner(tmp_path):
    root = tmp_path / 'run'
    root.mkdir()
    seed = tmp_path / 'seed.png'
    y, x = np.indices((768, 1344))
    pixels = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)
    Image.fromarray(pixels).save(seed)
    config = AppConfig(ChatConfig('unused', 'test-only', base_url='http://offline.invalid/v1'), ImageConfig('unused', 'test-only', base_url='http://offline.invalid/v1'))
    deps = SimpleNamespace(config=config, run_root=root)
    return SessionRunner(root, deps=deps), seed


def _accepted(row, output_root, *, width, deps):
    source = np.asarray(Image.open(verify_ref(row['initial_image'], run_root=deps.run_root)))
    state = create_panorama_state(np.full((64, 128, 3), 64, np.uint8), source,
                                  width=128, height=64,
                                  source_metadata=row['source_metadata'])
    state_ref = save_panorama_state(state, Path(output_root) / 'accepted-state', run_root=deps.run_root)
    endpoint = save_png(Path(output_root) / 'accepted-view.png', render_view(state), run_root=deps.run_root)
    return {'panorama_state': state_ref, 'panorama': state_ref['erp_preview_ref'],
            'endpoint': endpoint, 'camera': {'yaw': 0., 'pitch': 0.},
            'context': {'scene_description': 'Test procedural marks.',
                        'style_tokens': 'Geometric colors', 'entity_identity_properties': ''}}


def _registration():
    return RegistrationFailure('Rejected test fit', stage='native', candidates=[],
                               source_shape=(768, 1344, 3), target_shape=(768, 1344, 3))


def _transport_failure(monkeypatch):
    calls = []
    def fail(*a, **kw):
        calls.append(True)
        raise TransportExhausted('ReadTimeout')
    monkeypatch.setattr(bootstrap, 'initialize', fail)
    return calls


def test_registration_hold_keeps_original_image_without_world_or_facts(tmp_path, monkeypatch):
    runner, seed = _runner(tmp_path)
    monkeypatch.setattr(bootstrap, 'initialize', lambda *a, **kw: (_ for _ in ()).throw(_registration()))
    record = runner.initialize(seed, captions=['Look left.'], seed=7)
    assert record['status'] == 'incomplete'; assert record['nodes'] == record['transitions'] == []
    assert record['memory'] is record['initial_node_id'] is record['current_node_id'] is None
    label = record['extensions']['bootstrap']
    assert label['status'] == 'BOOTSTRAP_HOLD' and label['source'] == label['endpoint']
    hold = read_json(label['hold'], run_root=runner.run_root)
    assert hold['source'] == file_ref(runner.run_root / 'inputs/initial.png', run_root=runner.run_root, image=True)
    assert hold['failure_classification'] == 'bootstrap_registration'
    assert hold['new_facts'] == hold['state_or_memory_commit_calls'] == hold['semantic_memory_observation_calls'] == 0
    assert not hold['world_initialized'] and not hold['accepted_panorama_created'] and not hold['action_completed']
    assert hold['panorama'] is hold['panorama_state'] is hold['memory'] is None
    assert not (runner.run_root / 'memory').exists()


def test_policy_refusal_bootstrap_hold_is_explicit_in_cli_summary(tmp_path, monkeypatch):
    from wmmllm.cli import _session_summary
    runner, seed = _runner(tmp_path)
    error = ProviderRefusal('content_policy_violation', http_status=400)
    monkeypatch.setattr(bootstrap, 'initialize', lambda *a, **kw: (_ for _ in ()).throw(error))
    record = runner.initialize(seed, captions=['Look left.'])
    assert _session_summary(record) == {'status': 'incomplete', 'current_node_id': None,
        'bootstrap_status': 'BOOTSTRAP_HOLD', 'world_initialized': False, 'action_completed': False}
    hold = read_json(record['extensions']['bootstrap']['hold'], run_root=runner.run_root)
    assert hold['failure_classification'] == 'provider_refusal' and not (runner.run_root / 'memory').exists()


@pytest.mark.parametrize('failure', ['registration', 'transport'])
@pytest.mark.parametrize('retry', ['initialize', 'resume'])
def test_initialization_retry_and_resume_use_bound_checkpoint(tmp_path, monkeypatch, failure, retry):
    runner, seed = _runner(tmp_path)
    requests = []

    def generation(row, output_root, *, width, deps):
        requests.append((deepcopy(row), Path(output_root), width))
        if len(requests) == 1:
            # A partial immutable attempt must not collide with the next one.
            save_png(Path(output_root) / 'partial.png', Image.new('RGB', (8, 8)), run_root=deps.run_root)
            raise _registration() if failure == 'registration' else TransportExhausted('ReadTimeout')
        return _accepted(row, output_root, width=width, deps=deps)

    monkeypatch.setattr(bootstrap, 'initialize', generation)
    if failure == 'transport':
        with pytest.raises(TransportExhausted):
            runner.initialize(seed, captions=['Look left.'], seed=7, panorama_width=128)
        assert runner.record['status'] == 'failed'
    else:
        assert runner.initialize(seed, captions=['Look left.'], seed=7, panorama_width=128)['extensions']['bootstrap']['status'] == 'BOOTSTRAP_HOLD'
    fresh = SessionRunner(runner.run_root, deps=runner.deps)
    record = (fresh.resume() if retry == 'resume' else
              fresh.initialize(seed, captions=['Look left.'], seed=7, panorama_width=128))
    assert record['status'] == 'ready' and record['current_node_id'] == 'node0000'
    assert requests[0][0] == requests[1][0] and requests[0][2] == requests[1][2] == 128
    assert requests[0][1] == runner.run_root and requests[1][1] != runner.run_root
    assert fresh.store.snapshot().completed_segments == 0 and not fresh.store.snapshot().subjects
    assert fresh.resume() == record and len(requests) == 2


@pytest.mark.parametrize('changed', ['caption', 'seed', 'width', 'source'])
def test_retry_rejects_changed_inputs_before_generation(tmp_path, monkeypatch, changed):
    runner, seed = _runner(tmp_path)
    calls = _transport_failure(monkeypatch)
    with pytest.raises(TransportExhausted):
        runner.initialize(seed, captions=['Look left.'], seed=7, panorama_width=128)
    arguments = {'captions': ['Look left.'], 'seed': 7, 'panorama_width': 128}
    if changed == 'caption': arguments['captions'] = ['Look right.']
    if changed == 'seed': arguments['seed'] = 8
    if changed == 'width': arguments['panorama_width'] = 256
    if changed == 'source': Image.new('RGB', (1344, 768), 'white').save(seed)
    with pytest.raises(IntegrityError, match='retry inputs differ'):
        runner.initialize(seed, **arguments)
    assert len(calls) == 1


def test_resume_rejects_mutated_checkpoint_and_runtime_binding(tmp_path, monkeypatch):
    runner, seed = _runner(tmp_path)
    calls = _transport_failure(monkeypatch)
    with pytest.raises(TransportExhausted): runner.initialize(seed, captions=['Look left.'])
    path = runner.run_root / 'bootstrap/CHECKPOINT.json'
    value = json.loads(path.read_text())
    value['captions'] = ['Look right.']
    value['binding_sha256'] = hashlib.sha256(canonical_json({k: v for k, v in value.items() if k != 'binding_sha256'})).hexdigest()
    path.write_text(json.dumps(value))
    with pytest.raises(IntegrityError, match='checkpoint reference differs'): runner.resume()
    assert len(calls) == 1


@pytest.mark.parametrize('crash', ['before_memory', 'before_publication'])
def test_resume_finishes_saved_bootstrap_without_new_generation(tmp_path, monkeypatch, crash):
    runner, seed = _runner(tmp_path)
    calls = []
    def generate(*a, **kw):
        calls.append(True)
        return _accepted(*a, **kw)
    monkeypatch.setattr(bootstrap, 'initialize', generate)
    if crash == 'before_memory':
        monkeypatch.setattr(runner, '_open_store', lambda: (_ for _ in ()).throw(RuntimeError('Injected before memory')))
    else:
        publish, crashed = runner._publish, []
        def fail_first_node_publication():
            if runner.record['nodes'] and not crashed:
                crashed.append(True)
                raise RuntimeError('Injected after initial memory, before node publication')
            publish()
        monkeypatch.setattr(runner, '_publish', fail_first_node_publication)
    with pytest.raises(RuntimeError): runner.initialize(seed, captions=['Look left.'])
    durable = read_json(runner.path, run_root=runner.run_root)
    assert durable['status'] == 'failed' and not durable['nodes'] and durable['memory'] is None
    assert (runner.run_root / 'bootstrap/ACCEPTED_READY.json').is_file()
    fresh = SessionRunner(runner.run_root, deps=runner.deps)
    assert fresh.resume()['status'] == 'ready'
    assert len(calls) == 1 and fresh.store.snapshot().completed_segments == 0


@pytest.mark.parametrize('changed', ['video', 'chat', 'image'])
def test_resume_rejects_changed_settings_or_model_identity_before_generation(tmp_path, monkeypatch, changed):
    from dataclasses import replace
    runner, seed = _runner(tmp_path)
    calls = _transport_failure(monkeypatch)
    with pytest.raises(TransportExhausted): runner.initialize(seed, captions=[])
    config = runner.deps.config
    if changed == 'video':
        runner.deps.config, message = replace(config, video=replace(config.video, fps=30.)), 'runtime settings differ'
    else:
        checkpoint = read_json(runner.run_root / 'bootstrap/CHECKPOINT.json', run_root=runner.run_root)
        assert checkpoint['models'][changed] != getattr(config, changed).model
        replacement = replace(getattr(config, changed), model='different-test-only-model')
        runner.deps.config, message = replace(config, **{changed: replacement}), 'model identities differ'
    with pytest.raises(IntegrityError, match=message): runner.resume()
    assert len(calls) == 1


@pytest.mark.parametrize('site', ['repair', 'final_gate'])
def test_bootstrap_optional_handlers_preserve_fatal_errors(tmp_path, monkeypatch, site):
    error = ProviderError('AuthenticationError', http_status=401)
    row, deps = _optional_bootstrap(tmp_path, monkeypatch)
    gate_calls = []
    def gate(*a, **kw):
        gate_calls.append(True)
        if len(gate_calls) > 1 and site == 'final_gate': raise error
        return False, {'verdict': 'FAIL', 'detail': '', 'repair_views': [{'yaw': 45., 'pitch': 0.}]}, {'metrics': []}
    def repair(*a, **kw):
        raise error if site == 'repair' else ProviderError('BadRequestError', http_status=400)
    monkeypatch.setattr(bootstrap, 'gate', gate)
    monkeypatch.setattr(bootstrap, 'repair', repair)
    with pytest.raises(type(error)) as caught: bootstrap.initialize(row, tmp_path, width=128, deps=deps)
    assert caught.value is error


def _optional_bootstrap(tmp_path, monkeypatch):
    seed = tmp_path / 'seed.png'; Image.new('RGB', (1344, 768)).save(seed)
    config = AppConfig(ChatConfig('unused', 'test-only', base_url='http://offline.invalid/v1'), ImageConfig('unused', 'test-only', base_url='http://offline.invalid/v1'))
    deps = SimpleNamespace(config=config, run_root=tmp_path)
    row = {'key': 'fixture', 'task_id': 'fixture', 'initial_image': file_ref(seed, run_root=tmp_path),
           'segment': 1, 'current_caption': 'Look left.'}
    monkeypatch.setattr(bootstrap, 'camera_path', lambda *a, **kw: {'camera_path': []})
    monkeypatch.setattr(bootstrap, '_initial', lambda *a, **kw: {'context': {}, 'panorama_state': {}})
    monkeypatch.setattr(bootstrap.geo, 'load_panorama_state', lambda *a, **kw: {})
    return row, deps


def test_optional_http400_stays_handled_with_only_safe_class(tmp_path, monkeypatch):
    row, deps = _optional_bootstrap(tmp_path, monkeypatch)
    monkeypatch.setattr(bootstrap, 'gate', lambda *a, **kw: (False, {'verdict': 'FAIL', 'detail': '', 'repair_views': [{'yaw': 45., 'pitch': 0.}]}, {'metrics': []}))
    error = ProviderError('BadRequestError', http_status=400)
    error.body = {'error': {'message': 'sensitive remote details'}}
    monkeypatch.setattr(bootstrap, 'repair', lambda *a, **kw: (_ for _ in ()).throw(error))
    record = bootstrap.initialize(row, tmp_path, width=128, deps=deps)
    assert record['repair_error'] == {'class': 'BadRequestError'}
    assert record['seam_status'] == 'seam_unresolved'; assert 'sensitive' not in json.dumps(record)


@pytest.mark.parametrize('fatal', [True, False])
def test_reachable_gate_optional_handler(tmp_path, monkeypatch, fatal):
    session, row, deps = _optional_reachable(tmp_path, monkeypatch)
    error = TerminalQuotaError('insufficient_quota', http_status=429) if fatal else ProviderError('BadRequestError', http_status=400)
    monkeypatch.setattr(bootstrap, 'gate', lambda *a, **kw: (_ for _ in ()).throw(error))
    check = lambda: executor.check_actual_reachable_pose(row, tmp_path / 'turn', session, {}, {'yaw': 90., 'pitch': 0.}, deps=deps)
    if fatal:
        with pytest.raises(type(error)) as caught: check()
        assert caught.value is error
        return
    assert check() == 'seam_unresolved'
    report = read_json(tmp_path / 'turn/ACTUAL_REACHABLE_GATE.json', run_root=tmp_path)
    assert report['review']['error'] == {'class': 'BadRequestError'}


def _optional_reachable(tmp_path, monkeypatch):
    from wmmllm.session.io import write_once
    write_once(tmp_path / 'bootstrap/RESULT.json', {'seam_gate': {'planned_path': {'camera_path': []}}}, run_root=tmp_path)
    current = {'pose': {'yaw': 0., 'pitch': 0.}, 'panorama_state': {}, 'panorama': {}, 'seam_status': 'resolved'}
    session = SimpleNamespace(current_node=current)
    return session, {'segment': 1}, SimpleNamespace(run_root=tmp_path)
