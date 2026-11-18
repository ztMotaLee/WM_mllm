"""Source-grounded endpoint requests, schema repair and semantic checking."""
from __future__ import annotations
import copy, json, re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Literal, Mapping
import numpy as np
from PIL import Image, ImageDraw
from ..providers.types import ChatRequest, ImageInput
from ..prompts import read_text, read_json_data
from ..prompts import policy
from ..session.io import canonical_sha256, verify_ref, write_once

@dataclass(frozen=True)
class HistoryReference:
    image: Mapping[str, Any]
    segment: int
    state_index: Literal[0, 1]
    role: Literal['identity', 'initial_viewpoint']

    def to_record(self):
        return {'image': dict(self.image), 'segment': self.segment, 'state_index': self.state_index, 'role': self.role}

@dataclass(frozen=True)
class SegmentRequest:
    segment: int
    current_caption: str
    preceding_captions: tuple[str, ...]
    source: Mapping[str, Any]
    memory_payload: Mapping[str, Any]
    memory_context: Mapping[str, Any]
    historical_reference: HistoryReference | None
    planner_duration_seconds: float
    run_root: Path | None = field(default=None, repr=False, compare=False)

    def verify_inputs(self, *, run_root):
        for ref in [self.source, self.memory_context] + ([self.historical_reference.image] if self.historical_reference else []):
            verify_ref(ref, run_root=run_root)
        if len(self.preceding_captions) != self.segment - 1:
            raise ValueError('preceding_captions must hold the %d earlier captions' % (self.segment - 1))
        object.__setattr__(self, 'run_root', Path(run_root))

def image_input(ref, label='', *, run_root):
    if isinstance(ref, ImageInput):
        return ref
    if run_root is None:
        raise ValueError('An explicit run_root is required to resolve request images')
    return ImageInput(verify_ref(ref, run_root=run_root), label, ref['sha256'])

def labelled_images(request, endpoint=None):
    roles = read_json_data('image_roles.json')
    images = (image_input(request.source, roles['source'], run_root=request.run_root),)
    if endpoint is not None:
        images += (image_input(endpoint, roles['endpoint'], run_root=request.run_root),)
    if request.historical_reference:
        images += (image_input(request.historical_reference.image, roles['history'][request.historical_reference.role], run_root=request.run_root),)
    return images

@dataclass(frozen=True)
class Plan:
    current_state: str
    remaining_action: str
    endpoint_instruction: str
    h3_shots: tuple[Mapping[str, Any], ...] | None = None

    def as_policy_dict(self):
        v = {'current_state': self.current_state, 'remaining_action': self.remaining_action, 'endpoint_instruction': self.endpoint_instruction}
        if self.h3_shots is not None:
            v['h3_shots'] = [dict(s) for s in self.h3_shots]
        return v

    @classmethod
    def from_policy_dict(cls, value):
        return cls(value['current_state'], value['remaining_action'], value['endpoint_instruction'], None if value.get('h3_shots') is None else tuple((dict(s) for s in value['h3_shots'])))

@dataclass(frozen=True)
class SemanticDecision:
    status: Literal['PASS', 'FAIL', 'UNCERTAIN']
    verification: Mapping[str, str]
    failed: tuple[str, ...]
    uncertain: tuple[str, ...]
    reason: str

    def to_record(self):
        return {'status': self.status, 'verification': dict(self.verification), 'failed': list(self.failed), 'uncertain': list(self.uncertain), 'reason': self.reason}

    @classmethod
    def from_record(cls, value):
        return None if value is None else cls(value['status'], dict(value['verification']), tuple(value['failed']), tuple(value['uncertain']), value['reason'])

def verifier_request(request, plan, endpoint):
    context = policy.planner_input(list(request.preceding_captions), request.current_caption, dict(request.memory_payload))
    return ChatRequest('verifier', policy.verifier_system(request.current_caption), labelled_images(request, endpoint), policy.verification_input(context, plan.as_policy_dict()), policy.VERIFIER_RESPONSE_FORMAT)

def endpoint_request(request, *, decision, payload, editing_crop, physical_pano_ref, history=None, feedback=None):
    """Planner request; ``feedback`` is the checked retry feedback of attempt 2."""
    from .contracts import UI_WORLD_POLICY, planner_feedback
    settings = read_json_data('endpoint_multiple_response_format.json' if policy.explicit_multiple_shots(request.current_caption) else 'endpoint_single_response_format.json')
    system = policy.planner_system(request.current_caption, request.planner_duration_seconds) + read_text('endpoint_planner_scope.txt') + read_text('endpoint_planner_final_bbox.txt') + UI_WORLD_POLICY
    p_in = policy.planner_input(list(request.preceding_captions), request.current_caption, payload)
    if feedback is not None:
        p_in['previous_endpoint_failure'] = feedback['complete_verifier_failure']
    images = [image_input(request.source, 'current perspective source state', run_root=request.run_root)]
    history = history or request.historical_reference
    if history:
        images.append(image_input(history.image, 'earlier accepted identity reference; preserve latest physical properties', run_root=request.run_root))
    images.extend((image_input(editing_crop, 'EDITING_VIEW: actual latest physical source at editing_camera; bbox and image edit refer to this view', run_root=request.run_root), image_input(physical_pano_ref, 'current entire ERP panorama for scene context; EDITING_VIEW defines the edit camera', run_root=request.run_root)))
    result = ChatRequest('planner', system, tuple(images), p_in, settings)
    if feedback is not None:
        result = planner_feedback(result, feedback, run_root=request.run_root)
    return result

def support_mask(bbox, width=1344, height=768):
    b = np.asarray(bbox, dtype=float)
    if b.shape != (4,) or not np.isfinite(b).all() or np.any(b < 0) or np.any(b > 1) or (b[2] <= b[0]) or (b[3] <= b[1]):
        raise ValueError('planner scope_bbox must be a nonempty normalized rectangle')
    out = np.zeros((height, width), dtype=np.uint8)
    x0, y0 = np.floor(b[:2] * [width, height]).astype(int)
    x1, y1 = np.ceil(b[2:] * [width, height]).astype(int)
    out[y0:y1, x0:x1] = 255
    return out

def draw_polygons(polygons, width, height):
    image = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        draw.polygon([tuple(p) for p in np.asarray(polygon, dtype=float) * [width - 1, height - 1]], fill=255)
    return image

def _validate_response(text, *, decision, caption, run_root):
    raw = copy.deepcopy(policy.parse_response(text))
    if not isinstance(raw, dict):
        raise ValueError('planner response must be a JSON object')
    bbox = raw.pop('scope_bbox')
    polygons = raw.pop('support_polygons')
    boxes = raw.pop('final_view_entity_bboxes')
    metadata = bound_entity_metadata(decision, raw, run_root=run_root)
    plan, repair = repair_plan_schema(raw, entity_metadata=metadata, validate_plan=policy.validate_plan, run_root=run_root)
    policy.validate_requested_shots(plan, caption)
    permitted = support_mask(bbox)
    if not isinstance(polygons, list):
        raise ValueError('invalid normalized support polygon')
    for polygon in polygons:
        pts = np.asarray(polygon, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3 or (not np.isfinite(pts).all()) or np.any(pts < 0) or np.any(pts > 1):
            raise ValueError('invalid normalized support polygon')
    if not polygons:
        raise ValueError('empty actual permitted support')
    height, width = permitted.shape
    if not np.any(np.minimum(permitted, np.asarray(draw_polygons(polygons, width, height)))):
        raise ValueError('empty actual permitted support')
    if not isinstance(boxes, list):
        raise ValueError('invalid final_view_entity_bboxes')
    for box in boxes:
        if not isinstance(box, dict) or not isinstance(box.get('entity'), str) or (not box['entity'].strip()):
            raise ValueError('invalid final_view_entity_bboxes entity')
        support_mask(box['bbox'])
    return (bbox, polygons, boxes, plan, repair)

def endpoint_plan(*, chat, request, decision, caption, out, deps):
    """One schema-only correction; never spends a semantic regeneration."""
    response = chat.complete(request)
    receipts = [response.receipt.to_record()]
    try:
        values = _validate_response(response.text, decision=decision, caption=caption, run_root=deps.run_root)
    except (ValueError, KeyError, TypeError) as error:
        error_text = f'{type(error).__name__}: {error}'
        payload = copy.deepcopy(dict(request.payload))
        payload['technical_schema_correction'] = {'validation_error': error_text, 'invalid_planner_response': response.text, 'original_request_sha256': response.receipt.request_sha256, 'technical_reask_limit': 1, 'semantic_attempt': request.payload.get('previous_failed_proposal_provenance', {}).get('current_attempt', 1), 'does_not_consume_semantic_retry': True}
        correction = replace(request, payload=payload, system=request.system + ' Your previous response failed schema validation: ' + error_text + '. Return one corrected complete response with the same current action, accepted source, operation decision and response schema. Resolve generic subject/object references to a concrete entity in the supplied accepted view and operation targets. scope_bbox must contain four finite values [left,top,right,bottom] in [0,1], with right>left and bottom>top.')
        corrected = chat.complete(correction)
        receipts.append(corrected.receipt.to_record())
        values = _validate_response(corrected.text, decision=decision, caption=caption, run_root=deps.run_root)
        write_once(Path(out) / 'PLANNER_TECHNICAL_REASK.json', {'schema': 'wmmllm.planner_schema_reask.v1', 'status': 'VALIDATED', 'technical_reask_limit': 1, 'does_not_consume_semantic_retry': True, 'receipts': receipts}, run_root=deps.run_root)
    bbox, polygons, boxes, plan, repair = values
    return {'scope_bbox': bbox, 'support_polygons': polygons, 'final_view_entity_bboxes': boxes, 'plan': plan, 'repair': repair, 'receipts': receipts}

GENERIC_ENTITY = re.compile('\\b(the|this|that)\\s+(subject|object)\\b', re.I)
GENERIC_ERROR = 'endpoint_instruction must name the affected entity instead of an unresolved subject/object'

def _normalized(text: str) -> str:
    return ' '.join(re.findall('[a-z0-9]+', text.lower()))

def bound_entity_metadata(decision_ref: Mapping[str, Any], plan: Mapping[str, Any], *, run_root=None) -> dict:
    """Derive only unambiguous concrete target bindings already named in a plan.

    The two observed failures involve one person/figure subject_action target.
    This intentionally limited resolver does not infer an animal, group, other
    generic subject, or a target absent from the endpoint text. Ambiguous cases
    require a new planner response. The returned name comes verbatim from the
    operation decision, with its file SHA and exact target JSON pointer.
    """
    decision = json.loads(verify_ref(decision_ref, run_root=run_root).read_text(encoding='utf-8')) if 'path' in decision_ref else dict(decision_ref)
    endpoint = ' ' + _normalized(plan['endpoint_instruction']) + ' '
    candidates: dict[str, list[dict]] = {'subject': [], 'object': []}
    for i, operation in enumerate(decision.get('operations', [])):
        if operation.get('operation') not in {'subject_action', 'local_edit'}:
            continue
        targets = operation.get('targets', [])
        for j, name in enumerate(targets):
            if not isinstance(name, str) or not name.strip() or GENERIC_ENTITY.search(name):
                continue
            normalized = _normalized(name)
            core = re.sub('^(?:the |this |that |central )+', '', normalized)
            if ' ' + core + ' ' not in endpoint:
                continue
            entry = {'name': name, 'source': dict(decision_ref), 'json_pointer': f'/operations/{i}/targets/{j}', 'operation': operation['operation']}
            if operation['operation'] == 'subject_action' and core.split()[-1] in {'person', 'figure'}:
                candidates['subject'].append(entry)
            if operation['operation'] == 'local_edit' and len(targets) == 1:
                candidates['object'].append(entry)
    result = {}
    for role, entries in candidates.items():
        unique = {entry['name']: entry for entry in entries}
        if len(unique) == 1:
            result[role] = next(iter(unique.values()))
    return result

def repair_plan_schema(plan: Mapping[str, Any], *, entity_metadata: Mapping[str, Any], validate_plan: Callable[[dict], dict], run_root=None) -> tuple[dict, dict]:
    """Repair only the frozen validator's generic entity failure, then revalidate.

    Every replacement resolves to a concrete, SHA-bound operation target. No
    other plan field, physical operation, camera parameter, action wording,
    timing, or support geometry is changed. Unknown or ambiguous bindings fail.
    """
    original = copy.deepcopy(dict(plan))
    try:
        accepted = validate_plan(copy.deepcopy(original))
    except ValueError as exc:
        if str(exc) != GENERIC_ERROR:
            raise
    else:
        return (accepted, {'schema': 'aiwm.w34.planner_schema_repair.v5', 'changed': False, 'plan_sha256': canonical_sha256(original), 'additional_API_calls': 0})
    substitutions = []

    def replace(match: re.Match) -> str:
        role = match.group(2).lower()
        # ``entity_metadata`` comes from ``bound_entity_metadata`` of this same plan and decision.
        binding = entity_metadata.get(role)
        if binding is None:
            raise ValueError('No unique metadata binding for unresolved ' + role)
        name = binding['name']
        prefix = 'The ' if match.group(1)[0].isupper() else 'the '
        text = prefix + re.sub('^(?:the|this|that)\\s+', '', name, flags=re.I)
        substitutions.append({'span': list(match.span()), 'before': match.group(0), 'after': text, 'entity_metadata': copy.deepcopy(dict(binding))})
        return text
    repaired = copy.deepcopy(original)
    repaired['endpoint_instruction'] = GENERIC_ENTITY.sub(replace, original['endpoint_instruction'])
    accepted = validate_plan(copy.deepcopy(repaired))
    for key in original:
        if key != 'endpoint_instruction' and repaired[key] != original[key]:
            raise AssertionError('Planner repair modified an unrelated field')
    return (accepted, {'schema': 'aiwm.w34.planner_schema_repair.v5', 'changed': True, 'original_plan_sha256': canonical_sha256(original), 'repaired_plan_sha256': canonical_sha256(repaired), 'original_endpoint_instruction': original['endpoint_instruction'], 'repaired_endpoint_instruction': repaired['endpoint_instruction'], 'substitutions': substitutions, 'additional_API_calls': 0, 'remaining_action_unchanged': repaired['remaining_action'] == original['remaining_action'], 'requires_normal_endpoint_semantic_check': True})
