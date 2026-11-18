"""Observe actual accepted state images; preserve image order and resumable bindings."""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .facts import validate_observation
from wmmllm.prompts import read_json_data, read_text
from wmmllm.providers.types import ChatClient, ChatRequest, ImageInput, Usage
from wmmllm.providers.errors import ProviderError, TransportError
from wmmllm.providers.retry import raise_if_fatal_provider_error, safe_provider_error
from wmmllm.session.io import _local, canonical_sha256, file_ref, verify_ref, read_json, write_once

RESULT_NAME = 'OBSERVATION.json'
FRAME_INDICES = (0, 1)
IMAGE_ROLES = ('segment start state (index0)', 'accepted end state (index1)')
TECHNICAL_RETRIES = 1
SYSTEM = read_text('observer_system.txt')
RESPONSE_FORMAT = read_json_data('observer_response_format.json')
AUXILIARY_NOTE = (' The auxiliary physical views are deterministic crops of the same accepted panorama states. '
    'State index0 includes the auxiliary starting physical view; state index1 includes the auxiliary accepted ending '
    'physical view. A viewer pan may hide the edited object; use these actual state crops to observe physical changes. '
    'Do not infer completion from the caption.')


@dataclass(frozen=True)
class ObserverDeps:
    chat: ChatClient
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
               'known_subjects': dict(known_subjects), 'frame_indices': list(FRAME_INDICES)}
    images = tuple(ImageInput(verify_ref(ref, run_root=deps.run_root), role, ref['sha256'])
                   for ref, role in zip((start, endpoint), IMAGE_ROLES))
    return ChatRequest('observer', SYSTEM, images, payload, RESPONSE_FORMAT)


def _request_hash(request: ChatRequest, *, run_root: Path) -> str:
    record = {'role': request.role, 'system': request.system,
              'images': [{'image': file_ref(image.path, run_root=run_root), 'role': image.label}
                         for image in request.images], 'payload': request.payload,
              'response_format': request.response_format}
    return canonical_sha256(record)


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
    out_dir = _local(out_dir, run_root=deps.run_root, must_exist=False)
    result_path = out_dir / RESULT_NAME
    request = observer_request(caption, start, endpoint, known_subjects, deps)
    for image in auxiliary_images:
        ref = file_ref(image.path, run_root=deps.run_root)
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
    while observation is None and error is None:
        calls += 1
        try:
            response = deps.chat.complete(request)
            usages.append(response.usage)
            receipts.append(response.receipt.to_record())
            try:
                observation = validate_observation(json.loads(response.text), FRAME_INDICES)
            except (ValueError, TypeError, KeyError):
                error = {'kind': 'output', 'class': 'InvalidObservation',
                         'message': 'Observer output failed the observation schema'}
        except ProviderError as exc:
            raise_if_fatal_provider_error(exc)
            usages.append(exc.usage)
            if exc.receipt is not None:
                receipts.append(exc.receipt.to_record())
            if isinstance(exc, TransportError) and retries < TECHNICAL_RETRIES:
                retries += 1
            else:
                error = {'kind': exc.kind, **safe_provider_error(exc),
                         'message': 'Observer provider call failed'}
    usage = _usage_total(usages)
    value = {'schema': 'wmmllm.observation.v1', 'status': 'OBSERVED' if error is None else 'ERROR',
             'task_id': task_id, 'segment': segment, 'current_caption': caption,
             'frames': [{'index': index, 'image': dict(ref)}
                        for index, ref in zip(FRAME_INDICES, (start, endpoint))],
             'known_subjects': dict(known_subjects), 'request_sha256': binding,
             'observation': observation, 'technical_retries': retries,
             'usage': None if usage is None else usage.to_record(),
             'call_counts': {'chat': calls, 'image': 0}, 'receipts': receipts, 'error': error}
    ref = write_once(result_path, value, run_root=deps.run_root)
    return ObservationResult.from_record(value, ref)
