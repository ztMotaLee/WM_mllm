"""The injected language model chooses the ordered operation route."""
from __future__ import annotations
import copy, json, math
from dataclasses import replace
from pathlib import Path
from ..providers.types import ChatRequest
from ..prompts import read_text, read_json_data
from ..session.io import write_once, file_ref
from .planning import image_input

def json_format(name, properties):
    return {'type': 'json_schema', 'json_schema': {'name': name, 'strict': True, 'schema': {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}}}
OPS = ('rotate', 'revisit', 'navigate', 'local_edit', 'global_change', 'subject_action')
OP_SCHEMA = {'type': 'object', 'properties': {'operation': {'type': 'string', 'enum': list(OPS)}, 'targets': {'type': 'array', 'items': {'type': 'string'}}, 'scope': {'type': 'string', 'enum': ['camera_only', 'local', 'global']}, 'instruction': {'type': 'string'}}, 'required': ['operation', 'targets', 'scope', 'instruction'], 'additionalProperties': False}
POSE = {'type': 'object', 'properties': {'yaw': {'type': 'number'}, 'pitch': {'type': 'number'}}, 'required': ['yaw', 'pitch'], 'additionalProperties': False}

def validate_operation_response(text, visits):
    """Validate the declared operation schema before choosing an execution path."""
    decision = json.loads(text)
    if not isinstance(decision, dict):
        raise ValueError('operation planner response must be a JSON object')
    required = {'operations', 'ending_camera', 'editing_camera', 'editing_hfov', 'restore_node', 'physical_state_changed', 'endpoint_instruction', 'decision_reason'}
    if set(decision) != required:
        raise ValueError('operation planner response fields differ from schema')
    for key in ('ending_camera', 'editing_camera'):
        camera = decision[key]
        if not isinstance(camera, dict) or set(camera) != {'yaw', 'pitch'}:
            raise ValueError('invalid planner camera schema')
        if not all((type(camera[k]) in (int, float) and math.isfinite(camera[k]) for k in ('yaw', 'pitch'))):
            raise ValueError('nonfinite planner camera')
        if abs(camera['pitch']) > 80:
            raise ValueError('planner pitch outside supported camera')
    if type(decision['editing_hfov']) not in (int, float) or not math.isfinite(decision['editing_hfov']) or (not 90 <= decision['editing_hfov'] <= 150):
        raise ValueError('unsupported editing aperture')
    if not isinstance(decision['operations'], list) or not decision['operations']:
        raise ValueError('operation planner operations must be a nonempty array')
    for operation in decision['operations']:
        if not isinstance(operation, dict) or set(operation) != {'operation', 'targets', 'scope', 'instruction'}:
            raise ValueError('invalid operation schema')
        if operation['operation'] not in OPS:
            raise ValueError('unknown operation')
        if not isinstance(operation['targets'], list) or not all((isinstance(target, str) for target in operation['targets'])):
            raise ValueError('invalid operation targets schema')
        if operation['scope'] not in ('camera_only', 'local', 'global') or not isinstance(operation['instruction'], str):
            raise ValueError('invalid operation scope/instruction schema')
    if not isinstance(decision['restore_node'], str) or type(decision['physical_state_changed']) is not bool or (not isinstance(decision['endpoint_instruction'], str)) or (not isinstance(decision['decision_reason'], str)):
        raise ValueError('invalid operation planner scalar schema')
    kinds = [operation['operation'] for operation in decision['operations']]
    if decision['restore_node'] and (not any((v.get('node_id', v.get('node_path')) == decision['restore_node'] for v in visits.values()))):
        raise ValueError('planner restore target not in saved scene graph')
    if all((kind in ('rotate', 'revisit') for kind in kinds)) and decision['physical_state_changed']:
        raise ValueError('camera-only plan cannot change physical state')
    return decision

def primitive_decision(decision):
    return {key: decision[key] for key in ('operations', 'ending_camera', 'editing_camera', 'editing_hfov', 'restore_node', 'physical_state_changed', 'endpoint_instruction', 'decision_reason')}

def operation_request(row, source, pano_ref, pose, memory, subjects, context, preceding, visits, *, run_root):
    from .contracts import bind_operation_planner_inputs, bind_official_row
    payload = {'instruction': row['current_caption'], 'current_camera': pose, 'preceding_instructions': list(preceding), 'memory': memory.memory_text, 'latest_subjects': subjects, 'scene_context': context, 'saved_views': [{'node_path': v.get('node_id', v.get('node_path')), 'pose': v['pose'], 'physical_revision': v['physical_revision'], 'view_identity': v.get('view_identity')} for v in list(visits.values())[-24:]]}
    if row.get('prior_failure_holds'):
        payload['prior_failure_holds'] = row['prior_failure_holds']
    payload, binding = bind_operation_planner_inputs(payload, row, bind_official_row)
    return (ChatRequest('planner', read_text('operation_system.txt'), (image_input(source, 'actual current accepted source view', run_root=run_root), image_input(pano_ref, 'current accepted entire ERP panorama', run_root=run_root)), payload, read_json_data('operation_response_format.json')), binding)

def decide(row, out, source, pano_ref, pose, store, context, preceding, visits, *, deps):
    from ..memory.retrieval import retrieve, known_subjects
    out = Path(out) / 'operation_planner'
    out.mkdir(parents=True, exist_ok=True)
    memory = retrieve(store, segment=row['segment'], caption=row['current_caption'], source=source, preceding_captions=preceding, text_purpose='navigate')
    memory_ref = write_once(out / 'MEMORY_READ.json', memory.to_record({'root_context': context}), run_root=deps.run_root)
    request, binding = operation_request(row, source, pano_ref, pose, memory, known_subjects(store, row['segment']), context, preceding, visits, run_root=deps.run_root)
    write_once(out / 'OFFICIAL_MOTION_BINDING.json', binding, run_root=deps.run_root)
    response = deps.providers.chat.complete(request)
    receipts = [response.receipt.to_record()]
    try:
        decision = validate_operation_response(response.text, visits)
    except (ValueError, KeyError, TypeError) as error:
        error_text = f'{type(error).__name__}: {error}'
        payload = copy.deepcopy(dict(request.payload))
        payload['technical_schema_correction'] = {'validation_error': error_text, 'invalid_planner_response': response.text, 'original_request_sha256': response.receipt.request_sha256, 'technical_reask_limit': 1, 'semantic_attempt': 1, 'does_not_consume_semantic_retry': True}
        correction = replace(request, payload=payload, system=request.system + ' Your previous response failed schema validation: ' + error_text + '. Return one corrected complete response with the same current action, accepted source, saved views, memory and response schema. Do not add a new physical action. Use a nonempty operations array, finite supported camera values and an exact supplied saved node_path for restore_node, or an empty string.')
        corrected = deps.providers.chat.complete(correction)
        receipts.append(corrected.receipt.to_record())
        decision = validate_operation_response(corrected.text, visits)
        write_once(out / 'OPERATION_PLANNER_TECHNICAL_REASK.json', {'schema': 'wmmllm.operation_schema_reask.v1', 'status': 'VALIDATED', 'technical_reask_limit': 1, 'does_not_consume_semantic_retry': True, 'receipts': receipts}, run_root=deps.run_root)
    decision.update(planner_receipts=receipts, memory_reads=[memory_ref], source_sha256=source['sha256'], panorama_sha256=pano_ref['sha256'], reasoning_effort=deps.config.chat.reasoning_effort)
    kinds = [op['operation'] for op in decision['operations']]
    prior = row.get('ae_classification', {}).get('primary_class', row.get('class_top'))
    diag = 'pure_rotation' if all((k in ('rotate', 'revisit') for k in kinds)) else 'mixed' if len(kinds) > 1 else {'navigate': 'translation_navigation', 'local_edit': 'local_edit', 'global_change': 'global_change', 'subject_action': 'subject_action'}.get(kinds[0], kinds[0])
    decision['rule_diagnostic'] = {'prior': prior, 'planner_class': diag, 'agreement': prior == diag, 'routing_authority': False}
    write_once(out / 'DECISION.json', decision, run_root=deps.run_root)
    return decision
