"""Observed fact context and at most one current historical image."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from . import facts
from .store import MemoryConflict, MemorySnapshot, MemoryStore, StateImage
from wmmllm.agent.planning import HistoryReference
FileRef = dict[str, Any]

def _plain(value):
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value

@dataclass(frozen=True)
class MemoryContext:
    """Retrieval result for segment ``target_segment`` (MEMORY_CONTEXT.json without planner_payload)."""

    target_segment: int
    source: FileRef
    preceding_captions: tuple[str, ...]
    memory_text: str
    historical_reference: HistoryReference | None
    selection: Mapping[str, Any]
    memory_sha256: str

    def to_record(self, planner_payload: Mapping[str, Any]) -> dict:
        history = self.historical_reference
        return _plain({'schema': 'wmmllm.memory_context.v1', 'target_segment': self.target_segment,
                      'source': self.source, 'preceding_captions': list(self.preceding_captions),
                      'memory_text': self.memory_text,
                      'historical_reference': None if history is None else history.to_record(),
                      'selection': self.selection, 'memory_sha256': self.memory_sha256,
                      'planner_payload': planner_payload})


def _candidates(snapshot: MemorySnapshot, source: FileRef) -> tuple[list[StateImage], list[dict]]:
    changes = snapshot.state_change_segment
    eligible, excluded = [], []
    for state in snapshot.states:
        if state.image['sha256'] == source['sha256']:
            continue
        if any(state.segment < changes.get(f['subject_id'], 0) for f in state.facts):
            excluded.append({'segment': state.segment, 'state_index': state.state_index,
                             'reason': 'older_than_observed_state_change'})
            continue
        eligible.append(state)
    return eligible, excluded


def _identity(eligible: list[StateImage], caption: str) -> tuple[HistoryReference | None, int]:
    tokens = facts._tokens
    query = tokens(caption)
    ranked = []
    for state in eligible:
        labels = tokens(' '.join(f['label'] for f in state.facts))
        content = tokens(' '.join(f['key'] + ' ' + f['value'] for f in state.facts))
        score = 4 * len(query & labels) + len(query & content)
        if score > 0:
            ranked.append((score, state.segment, state.state_index, state))
    ranked.sort(key=lambda row: row[:3], reverse=True)
    if not ranked:
        return None, 0
    best = ranked[0][3]
    return HistoryReference(best.image, best.segment, best.state_index, 'identity'), len(ranked)


def _initial_view(snapshot: MemorySnapshot, source: FileRef) -> HistoryReference | None:
    entries = [{'image': s.image, 'facts': list(s.facts), 'segment': s.segment, 'frame_index': s.state_index}
               for s in snapshot.states]
    chosen = facts.select_initial_reference(
        entries, dict(snapshot.state_change_segment), dict(source))
    if chosen is None or (chosen['segment'], chosen['frame_index']) != (1, 0):
        return None
    return HistoryReference(chosen['image'], 1, 0, 'initial_viewpoint')


def retrieve(store: MemoryStore, *, segment: int, caption: str, source: FileRef,
             preceding_captions: Sequence[str]) -> MemoryContext:
    """Context for the next uncommitted segment, whose source must be S_1 or the last accepted E."""
    if not isinstance(caption, str) or not caption.strip() or len(caption) > 10000:
        raise ValueError('current caption must be a nonempty string of at most 10000 characters')
    snapshot = store.snapshot()
    if segment != snapshot.completed_segments + 1:
        raise MemoryConflict('Retrieve only for the next uncommitted segment')
    expected = snapshot.initial_state if segment == 1 else snapshot.commits[-1]['endpoint']
    if dict(source) != expected:
        raise MemoryConflict('Segment %d source must be the %s' % (
            segment, 'official initial state' if segment == 1 else 'accepted endpoint of segment %d' % (segment - 1)))
    if list(preceding_captions) != [c['caption'] for c in snapshot.commits]:
        raise MemoryConflict('Preceding captions differ from the committed segments')
    text = facts.memory_context(snapshot.world_memory)
    eligible, excluded = _candidates(snapshot, source)
    initial_requested = facts.asks_initial_view(caption)
    if initial_requested:
        reference, ranked = _initial_view(snapshot, source), None
    else:
        reference, ranked = _identity(eligible, caption)
    selection = {'policy': 'explicit_initial_view_v2' if initial_requested else 'observed_subject_overlap_latest_state_v1',
                 'initial_view_requested': initial_requested,
                 'initial_view_available': (reference is not None) if initial_requested else None,
                 'ranked_candidates': ranked, 'excluded': excluded,
                 'selected': None if reference is None else [reference.segment, reference.state_index],
                 'requested_reference_role': None if reference is None else reference.role}
    return MemoryContext(segment, dict(source), tuple(preceding_captions), text, reference, selection,
                         snapshot.sha256)

