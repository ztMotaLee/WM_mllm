"""Observe actual accepted state images; preserve image order and resumable bindings."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from .facts import validate_observation
from .store import checked_path
from wmmllm.providers.types import ChatClient, ChatRequest, ImageInput, Usage
from wmmllm.providers.errors import ProviderError, TransportError
from wmmllm.providers.retry import raise_if_fatal_provider_error, safe_provider_error
from wmmllm.session.io import canonical_json, file_ref, verify_ref, read_json, write_once

RESULT_NAME = 'OBSERVATION.json'
AUXILIARY_NOTE = (' The auxiliary physical views are deterministic crops of the same accepted panorama states. '
    'State index0 includes the auxiliary starting physical view; state index1 includes the auxiliary accepted ending '
    'physical view. A viewer pan may hide the edited object; use these actual state crops to observe physical changes. '
    'Do not infer completion from the caption.')


@dataclass(frozen=True)
class ObserverDeps:
    chat: ChatClient
    caps: Mapping[str, int]
    technical_retries: int
    frame_indices: tuple[int, int]
    image_roles: tuple[str, str]
    system: str
    response_format: Mapping[str, Any]
    now: Callable[[], float]
    run_root: Path


@dataclass(frozen=True)
class ObservationResult:
    status: Literal['OBSERVED', 'ERROR']
    observation: Mapping[str, Any] | None
    usage: Usage | None
    technical_retries: int
    error: Mapping[str, str] | None
    record: dict[str, Any]

    @classmethod
    def from_record(cls, value, record):
        usage = None if value['usage'] is None else Usage(**value['usage'])
        return cls(value['status'], value['observation'], usage,
                   value['technical_retries'], value['error'], record)


def observer_request(caption: str, start: Mapping[str, Any], endpoint: Mapping[str, Any],
                     known_subjects: Mapping[str, Any], deps: ObserverDeps) -> ChatRequest:
    payload = {'current_caption_requested_not_observed': caption,
               'known_subjects': dict(known_subjects), 'frame_indices': list(deps.frame_indices)}
    images = tuple(ImageInput(verify_ref(ref, run_root=deps.run_root), role, ref['sha256'])
                   for ref, role in zip((start, endpoint), deps.image_roles))
    return ChatRequest('observer', deps.system, images, payload, deps.response_format)


def _request_hash(request: ChatRequest, *, run_root: Path) -> str:
    record = {'role': request.role, 'system': request.system,
              'images': [{'image': file_ref(image.path, run_root=run_root), 'role': image.label}
                         for image in request.images], 'payload': request.payload,
              'response_format': request.response_format}
    return hashlib.sha256(canonical_json(record)).hexdigest()


def _usage_total(usages: Sequence[Usage | None]) -> Usage | None:
    if not usages or any(usage is None for usage in usages):
        return None
    totals = {}
    for key in Usage(0, 0).to_record():
        values = [usage.to_record()[key] for usage in usages]
        totals[key] = None if any(value is None for value in values) else sum(values)
    return Usage(**totals)


def observe_state(*, task_id: str, segment: int, caption: str, start: Mapping[str, Any],
                  endpoint: Mapping[str, Any], known_subjects: Mapping[str, Any],
                  deps: ObserverDeps, out_dir: Path,
                  auxiliary_images: Sequence[ImageInput] = ()) -> ObservationResult:
    """Observe once, with one explicit stage technical retry and no inferred facts."""
    if deps.frame_indices != (0, 1) or len(deps.image_roles) != 2:
        raise ValueError('Observer requires source index 0 and endpoint index 1')
    if deps.technical_retries not in (0, 1):
        raise ValueError('Observer technical retry limit must be zero or one')
    out_dir = checked_path(out_dir, Path(deps.run_root).resolve(), must_exist=False)
    result_path = out_dir / RESULT_NAME
    request = observer_request(caption, start, endpoint, known_subjects, deps)
    for image in auxiliary_images:
        ref = file_ref(checked_path(image.path, Path(deps.run_root).resolve()), run_root=deps.run_root)
        if image.sha256 is not None and image.sha256 != ref['sha256']:
            raise ValueError('Observer auxiliary image checksum differs')
    if auxiliary_images:
        request = replace(request, system=request.system + AUXILIARY_NOTE,
                          images=request.images + tuple(auxiliary_images),
                          payload={**request.payload, 'auxiliary_images_are_accepted_state_crops': True})
    binding = _request_hash(request, run_root=deps.run_root)
    if result_path.exists():
        cached = read_json(result_path, run_root=deps.run_root)
        if cached.get('request_sha256') != binding:
            raise ValueError('Cached observation is bound to another accepted transition')
        if cached.get('status') != 'ERROR' or (cached.get('error') or {}).get('kind') != 'transport':
            return ObservationResult.from_record(cached, file_ref(result_path, run_root=deps.run_root))
        # Resume observation alone. All earlier failed records remain immutable.
        index = 1
        while (out_dir / ('transport_resume_%03d' % index) / RESULT_NAME).exists():
            previous = out_dir / ('transport_resume_%03d' % index)
            prior = read_json(previous / RESULT_NAME, run_root=deps.run_root)
            if prior.get('request_sha256') != binding:
                raise ValueError('Resumed observation request binding differs')
            if prior.get('status') != 'ERROR' or (prior.get('error') or {}).get('kind') != 'transport':
                return ObservationResult.from_record(prior, file_ref(previous / RESULT_NAME, run_root=deps.run_root))
            index += 1
        selected = out_dir / ('transport_resume_%03d' % index)
        write_once(selected / 'RESUME_BINDING.json', {
            'schema': 'wmmllm.observer_resume.v1',
            'preserved_error': file_ref(result_path, run_root=deps.run_root),
            'accepted_request_sha256': binding, 'source': dict(start), 'endpoint': dict(endpoint),
            'observer_only': True, 'additional_semantic_proposals': 0}, run_root=deps.run_root)
        return observe_state(task_id=task_id, segment=segment, caption=caption, start=start,
                             endpoint=endpoint, known_subjects=known_subjects, deps=deps,
                             out_dir=selected, auxiliary_images=auxiliary_images)
    observation, error, retries, calls = None, None, 0, 0
    usages, receipts = [], []
    cap = min(2, int(deps.caps.get('chat', 2)))
    while observation is None and error is None:
        if calls >= cap:
            error = {'kind': 'call_cap', 'class': 'ObserverCallCapExceeded',
                     'message': 'Observer stage chat cap exceeded'}
            break
        calls += 1
        try:
            response = deps.chat.complete(request)
            usages.append(response.usage)
            receipts.append(response.receipt.to_record())
            try:
                observation = validate_observation(json.loads(response.text), deps.frame_indices)
            except (ValueError, TypeError, KeyError):
                error = {'kind': 'output', 'class': 'InvalidObservation',
                         'message': 'Observer output failed the observation schema'}
        except ProviderError as exc:
            raise_if_fatal_provider_error(exc)
            usages.append(exc.usage)
            if exc.receipt is not None:
                receipts.append(exc.receipt.to_record())
            if isinstance(exc, TransportError) and retries < deps.technical_retries:
                retries += 1
            else:
                error = {'kind': exc.kind, **safe_provider_error(exc),
                         'message': 'Observer provider call failed'}
    usage = _usage_total(usages)
    value = {'schema': 'wmmllm.observation.v1', 'status': 'OBSERVED' if error is None else 'ERROR',
             'task_id': task_id, 'segment': segment, 'current_caption': caption,
             'frames': [{'index': index, 'image': dict(ref)}
                        for index, ref in zip(deps.frame_indices, (start, endpoint))],
             'known_subjects': dict(known_subjects), 'request_sha256': binding,
             'observation': observation, 'technical_retries': retries,
             'usage': None if usage is None else usage.to_record(),
             'call_counts': {'chat': calls, 'image': 0}, 'receipts': receipts, 'error': error}
    ref = write_once(result_path, value, run_root=deps.run_root)
    return ObservationResult.from_record(value, ref)
