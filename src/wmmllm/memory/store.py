"""Atomic accepted-state memory, continuity checks and semantic snapshots."""
from __future__ import annotations
import copy
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping
from . import facts as fact_store
from wmmllm.session.io import canonical_json, file_lock, read_json, write_atomic, write_once, file_ref, verify_ref, sha256_file, resolve_run_path
FileRef = dict[str, Any]


def checked_path(path: Path, run_root: Path, *, must_exist: bool = True) -> Path:
    path = Path(path)
    if path.is_absolute():
        try:
            relative = path.relative_to(run_root).as_posix()
        except ValueError:
            raise ValueError('memory path: outside run root') from None
    else:
        relative = path.as_posix()
    return resolve_run_path(run_root, relative, must_exist=must_exist)

MEMORY_NAME = 'MEMORY.json'


class MemoryConflict(ValueError):
    """A commit or read does not fit the committed chain of this item."""


@dataclass
class World:
    """Minimal world object for the frozen fact store: only ``memory`` (no scene graph store)."""

    memory: dict = field(default_factory=dict)


@dataclass(frozen=True)
class StateImage:
    """An accepted state image: (segment, 0) = start S_n, (segment, 1) = accepted end E_n."""

    segment: int
    state_index: Literal[0, 1]
    image: FileRef
    facts: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class MemorySnapshot:
    """Read-only view of MEMORY.json; ``sha256`` is the SHA-256 of the file bytes."""

    task_id: str
    setting: str
    completed_segments: int
    initial_state: FileRef
    commits: tuple[Mapping[str, Any], ...]
    states: tuple[StateImage, ...]
    subjects: Mapping[str, Any]
    latest_state: Mapping[str, Any]
    state_change_segment: Mapping[str, int]
    world_memory: Mapping[str, Any]
    sha256: str

    def world(self) -> World:
        """A private copy of the fact store for the frozen app functions."""
        return World(copy.deepcopy(dict(self.world_memory)))


@dataclass(frozen=True)
class MemoryCommit:
    """Result of committing one segment's observation."""

    segment: int
    memory_after_commit: FileRef
    facts: int
    states_registered: tuple[tuple[int, int], ...]


def _snapshot(state: Mapping[str, Any], sha256: str) -> MemorySnapshot:
    states = tuple(StateImage(s['segment'], s['state_index'], dict(s['image']), tuple(s['facts']))
                   for s in state['states'])
    return MemorySnapshot(state['task_id'], state['setting'], state['completed_segments'],
                          dict(state['initial_state']), tuple(state['commits']), states, state['subjects'],
                          state['latest_state'], state['state_change_segment'], state['world_memory'], sha256)


class MemoryStore:
    """Memory of one item in ``directory`` (MEMORY.json + memory.lock).

    ``state_indices`` are the observer's indices of (S_n, E_n) (config memory.observer_frame_indices).
    """

    def __init__(self, directory: Path, *, task_id: str, setting: str, scene_id: str,
                 state_indices: tuple[int, int] = (0, 1), run_root: Path) -> None:
        self.run_root = Path(run_root).resolve()
        self.directory = checked_path(directory, self.run_root, must_exist=False)
        self.task_id, self.setting, self.scene_id = task_id, setting, scene_id
        self.state_indices = tuple(state_indices)
        self.path = self.directory / MEMORY_NAME
        self.lock = self.directory / 'memory.lock'

    def initialize(self, initial_state: FileRef) -> FileRef:
        """Create the memory with the official letterboxed first state (idempotent for the same state)."""
        verify_ref(initial_state, run_root=self.run_root)
        with file_lock(self.lock, run_root=self.run_root):
            if self.path.exists():
                if self._load()['initial_state'] != dict(initial_state):
                    raise MemoryConflict('Memory exists with another initial state')
                return file_ref(self.path, run_root=self.run_root)
            world = World()
            fact_store.ensure_world_memory(world)
            state = {'task_id': self.task_id, 'setting': self.setting, 'scene_id': self.scene_id,
                     'completed_segments': 0, 'initial_state': dict(initial_state), 'commits': [],
                     'states': [{'segment': 1, 'state_index': self.state_indices[0], 'image': dict(initial_state),
                                 'facts': []}],
                     'subjects': {}, 'latest_state': {}, 'state_change_segment': {}, 'world_memory': world.memory}
            return self._save(state)

    def snapshot(self) -> MemorySnapshot:
        with file_lock(self.lock, run_root=self.run_root):
            state = self._load()
            return _snapshot(state, sha256_file(self.path))

    def known_subjects(self, segment: int) -> Mapping[str, Any]:
        """Subject catalog the observer of ``segment`` (the next uncommitted one) may reuse."""
        snapshot = self.snapshot()
        if segment != snapshot.completed_segments + 1:
            raise MemoryConflict('Segment %d is not the next uncommitted segment' % segment)
        return copy.deepcopy(dict(snapshot.subjects))

    def commit(self, *, segment: int, caption: str, start: FileRef, endpoint: FileRef,
               observation: Mapping[str, Any], observation_record: FileRef,
               fact_source_type: str = 'accepted_state_observation') -> MemoryCommit:
        """Commit the observation of [S_n, E_n]; repeating the last commit returns it unchanged."""
        for ref in (start, endpoint, observation_record):
            verify_ref(ref, run_root=self.run_root)
        checked = fact_store.validate_observation(dict(observation),
                                                                         available_indices=self.state_indices)
        observed_record = read_json(observation_record, run_root=self.run_root)
        if (observed_record.get('status', 'OBSERVED') != 'OBSERVED'
                or observed_record.get('observation', observed_record) != checked):
            raise MemoryConflict('Observation is not the validated accepted-state record')
        with file_lock(self.lock, run_root=self.run_root):
            state = self._load()
            last = state['commits'][-1] if state['commits'] else None
            if last is not None and last['segment'] == segment and last['observation'] == dict(observation_record):
                saved_observation = read_json(observation_record, run_root=self.run_root)
                if saved_observation.get('observation', saved_observation) != checked:
                    raise MemoryConflict('Recovered observation value differs from its receipt')
                if (last['caption'], last['start'], last['endpoint']) != (caption, dict(start), dict(endpoint)):
                    raise MemoryConflict('Recovered commit belongs to other inputs')
                return MemoryCommit(segment, file_ref(self.path, run_root=self.run_root), last['facts'],
                                    tuple(tuple(x) for x in last['states_registered']))
            expected = state['initial_state'] if segment == 1 else (last or {}).get('endpoint')
            if segment != state['completed_segments'] + 1 or dict(start) != expected:
                source_name = 'S_1' if segment == 1 else 'E_%d' % (segment - 1)
                raise MemoryConflict('Segment %d must follow the last commit and start from %s' % (segment, source_name))
            self._apply_facts(state, segment, checked['facts'], observation_record, fact_source_type)
            registered = self._register_states(state, segment, checked['facts'], zip(self.state_indices, (start, endpoint)))
            state['completed_segments'] = segment
            state['commits'].append({'segment': segment, 'caption': caption, 'start': dict(start),
                                     'endpoint': dict(endpoint), 'observation': dict(observation_record),
                                     'facts': len(checked['facts']),
                                     'states_registered': [list(x) for x in registered]})
            ref = self._save(state)
            return MemoryCommit(segment, ref, len(checked['facts']), registered)

    def _apply_facts(self, state: dict, segment: int, facts: list, observation_record: FileRef,
                     source_type: str) -> None:
        app = fact_store
        world = World(state['world_memory'])
        app.ensure_world_memory(world)
        for f in facts:
            subject = f['subject_id']
            label = state['subjects'].get(subject, {}).get('label', f['label'])
            state['subjects'].setdefault(subject, {'label': label, 'properties': []})
            slot = subject + '|' + f['kind'] + '|' + f['key'].casefold()
            old = state['latest_state'].get(slot)
            if f['kind'] == 'state' and old and old['value'].casefold() != f['value'].casefold():
                state['state_change_segment'][subject] = segment
            text = (label + ' ' + f['key'] + ': ' + f['value'])[:190]
            fact = app.append_fact(
                world, kind=f['kind'], fact=text, scene_id=self.scene_id, scope='world',
                slots={'subject_id': subject, 'key': f['key'], 'value': f['value']},
                source={'type': source_type, 'segment': segment, 'observation': dict(observation_record),
                        'state_indices': f['frame_indices'], 'evidence': f['evidence']},
                confidence='model_observed', replaces=[old['fact_id']] if old and old.get('fact_id') else [])
            stored = app.sanitize(text, 200)
            fact = fact or next((x for x in world.memory['facts'] if (x['kind'], x['fact']) == (f['kind'], stored)),
                                None)
            if fact is None:
                raise RuntimeError('The frozen fact store did not retain an observed fact')
            state['latest_state'][slot] = {'value': f['value'], 'fact_id': fact['id'], 'segment': segment}
            properties = state['subjects'][subject].setdefault('properties', [])
            properties[:] = [p for p in properties
                             if (p['kind'], p['key'].casefold()) != (f['kind'], f['key'].casefold())]
            properties.append({'kind': f['kind'], 'key': f['key'], 'value': f['value']})
        state['world_memory'] = world.memory

    def commit_hold(self, *, segment: int, caption: str, source: FileRef,
                    failure_record: FileRef) -> MemoryCommit:
        """Advance the request journal while retaining every accepted fact and image."""
        for ref in (source, failure_record):
            verify_ref(ref, run_root=self.run_root)
        receipt = read_json(failure_record, run_root=self.run_root)
        if (receipt.get('status') != 'FAILURE_HOLD'
                or receipt.get('rejected_candidate_committed') is not False
                or receipt.get('hold_eligible') is not True
                or (receipt.get('segment'), receipt.get('caption'), receipt.get('source'),
                    receipt.get('endpoint')) != (segment, caption, dict(source), dict(source))):
            raise MemoryConflict('Failure hold requires its bound classified rejection receipt')
        with file_lock(self.lock, run_root=self.run_root):
            state = self._load()
            last = state['commits'][-1] if state['commits'] else None
            entry = {'segment': segment, 'caption': caption, 'start': dict(source),
                     'endpoint': dict(source), 'observation': dict(failure_record),
                     'facts': 0, 'states_registered': []}
            if last is not None and last['segment'] == segment:
                if last != entry:
                    raise MemoryConflict('Recovered failure hold belongs to other inputs')
                return MemoryCommit(segment, file_ref(self.path, run_root=self.run_root), 0, ())
            expected = state['initial_state'] if segment == 1 else (last or {}).get('endpoint')
            if segment != state['completed_segments'] + 1 or dict(source) != expected:
                raise MemoryConflict('Failure hold must follow the accepted endpoint')
            before = copy.deepcopy(state)
            state['completed_segments'] = segment
            state['commits'].append(entry)
            if any(state[k] != before[k] for k in state if k not in ('completed_segments', 'commits')):
                raise MemoryConflict('Failure hold changed accepted world memory')
            ref = self._save(state)
            if canonical_json(self._load()) != canonical_json(state):
                raise MemoryConflict('Failure hold persistence readback differs')
            return MemoryCommit(segment, ref, 0, ())

    def snapshot_to(self, path: Path) -> FileRef:
        """Publish an immutable accepted-node memory packet."""
        destination = checked_path(path, self.run_root, must_exist=False)
        with file_lock(self.lock, run_root=self.run_root):
            state = self._load()
            packet = {'schema': 'wmmllm.memory.v1', 'state': state,
                      'state_sha256': hashlib.sha256(canonical_json(state)).hexdigest()}
        return write_once(destination, packet, run_root=self.run_root)

    def restore_semantics(self, snapshot_ref: FileRef, receipt_path: Path) -> FileRef:
        """Restore saved facts without rolling back the sequential accepted journal."""
        verify_ref(snapshot_ref, run_root=self.run_root)
        packet = read_json(snapshot_ref, run_root=self.run_root)
        saved = packet['state']
        if hashlib.sha256(canonical_json(saved)).hexdigest() != packet['state_sha256']:
            raise MemoryConflict('Saved semantic memory checksum differs')
        if (saved['task_id'], saved['setting'], saved['scene_id']) != (
                self.task_id, self.setting, self.scene_id):
            raise MemoryConflict('Saved semantic memory belongs to another world')
        fields = ('world_memory', 'subjects', 'latest_state', 'state_change_segment')
        semantic = {key: copy.deepcopy(saved[key]) for key in fields}
        semantic_sha = hashlib.sha256(canonical_json(semantic)).hexdigest()
        destination = checked_path(receipt_path, self.run_root, must_exist=False)
        with file_lock(self.lock, run_root=self.run_root):
            current = self._load()
            journal = {key: copy.deepcopy(value) for key, value in current.items() if key not in fields}
            journal_sha = hashlib.sha256(canonical_json(journal)).hexdigest()
            current.update(semantic)
            self._save(current)
            restored = self._load()
            if {key: restored[key] for key in fields} != semantic:
                raise MemoryConflict('Saved semantic memory restore differs')
            if hashlib.sha256(canonical_json({key: value for key, value in restored.items()
                                             if key not in fields})).hexdigest() != journal_sha:
                raise MemoryConflict('Saved semantic restore changed the accepted journal')
            packet = read_json(self.path, run_root=self.run_root)
        after = write_once(destination.with_name(destination.stem + '.MEMORY.json'),
                           packet, run_root=self.run_root)
        receipt = {'schema': 'wmmllm.semantic_restore.v1', 'saved_memory': dict(snapshot_ref),
                   'semantic_fields': list(fields), 'saved_semantic_sha256': semantic_sha,
                   'restored_semantic_sha256': semantic_sha, 'semantic_memory_exact': True,
                   'retained_journal_sha256': journal_sha,
                   'retained_completed_segments': journal['completed_segments'],
                   'retained_last_endpoint': journal['commits'][-1]['endpoint']
                   if journal['commits'] else journal['initial_state'],
                   'journal_rolled_back': False, 'memory_after_restore': after,
                   'video_writes_memory': False}
        return write_once(destination, receipt, run_root=self.run_root)

    @staticmethod
    def _register_states(state: dict, segment: int, facts: list,
                         images: Iterable[tuple[int, FileRef]]) -> tuple[tuple[int, int], ...]:
        registered = []
        for index, image in images:
            cited = [f for f in facts if index in f['frame_indices']]
            if not cited:
                continue
            entry = {'segment': segment, 'state_index': index, 'image': dict(image), 'facts': cited}
            existing = [s for s in state['states'] if (s['segment'], s['state_index']) == (segment, index)]
            if existing:
                existing[0].update(entry)
            else:
                state['states'].append(entry)
            registered.append((segment, index))
        return tuple(registered)

    def _load(self) -> dict:
        packet = read_json(self.path, run_root=self.run_root)
        state = packet['state']
        if hashlib.sha256(canonical_json(state)).hexdigest() != packet['state_sha256']:
            raise MemoryConflict('MEMORY.json checksum mismatch')
        if (state['task_id'], state['setting'], state['scene_id']) != (self.task_id, self.setting, self.scene_id):
            raise MemoryConflict('MEMORY.json belongs to another item, setting or scene')
        for ref in [state['initial_state']] + [ref for c in state['commits']
                for ref in (c['start'], c['endpoint'], c['observation'])] + [
                s['image'] for s in state['states']]:
            verify_ref(ref, run_root=self.run_root)
        return state

    def _save(self, state: dict) -> FileRef:
        packet = {'schema': 'wmmllm.memory.v1', 'state': state,
                  'state_sha256': hashlib.sha256(canonical_json(state)).hexdigest()}
        return write_atomic(self.path, packet, run_root=self.run_root)
