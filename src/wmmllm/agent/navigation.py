"""Combined navigation destination verification; generated ERP is the navigation endpoint."""
from __future__ import annotations
import copy, hashlib, json, re
from ..prompts import read_text, read_json_data
from ..prompts.policy import SEMANTIC_FIELDS
from ..providers.types import ChatRequest
from .planning import image_input
CRITERIA = ('projection', 'movement', 'spatial', 'entity')

def navigation_decision(value):
    """Gate on every criterion; UNCERTAIN continues and is labeled."""
    required = {'criteria', 'semantic', 'reason', 'retry_instruction'}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError('Invalid combined verifier response schema.')
    if set(value['criteria']) != set(CRITERIA) or any((v not in ('PASS', 'FAIL', 'UNCERTAIN') for v in value['criteria'].values())):
        raise ValueError('Invalid navigation criteria.')
    sem = value['semantic']
    if set(sem) != set(SEMANTIC_FIELDS) | {'reason'} or any((sem[k] not in ('pass', 'fail', 'uncertain') for k in SEMANTIC_FIELDS)):
        raise ValueError('Invalid endpoint semantic fields.')
    if not all((isinstance(v, str) for v in (sem['reason'], value['reason'], value['retry_instruction']))):
        raise ValueError('Verifier explanations must be strings.')
    failed = [k for k, v in value['criteria'].items() if v == 'FAIL'] + [k for k in SEMANTIC_FIELDS if sem[k] == 'fail']
    uncertain = [k for k, v in value['criteria'].items() if v == 'UNCERTAIN'] + [k for k in SEMANTIC_FIELDS if sem[k] == 'uncertain']
    status = 'FAIL' if failed else 'UNCERTAIN' if uncertain else 'PASS'
    return {'status': status, 'criteria': dict(value['criteria']), 'verification': dict(sem), 'failed': failed, 'uncertain': uncertain, 'continuation_allowed': status != 'FAIL', 'semantic_accepted': status == 'PASS', 'uncertain_accepted_and_labeled': status == 'UNCERTAIN', 'reason': value['reason'], 'retry_instruction': value['retry_instruction'], 'adopted_improvement': False}

def actual_motion_text(caption):
    """Sessions bind no camera-local motion contract; an orbit request adds the pivot rule."""
    parts = ['Current bound action: ' + caption, 'Official camera-local motion contract: {}']
    if 'orbit' in caption.lower():
        parts.append('For this subject-pivot motion, rotation_axis_cam and rotation_sign describe the camera rig moving around its supplied pivot. Judge the changed physical viewpoint and coherent near/far landmark parallax. A stationary subject keeps its world orientation; spinning the subject does not satisfy camera orbit. Do not substitute a camera-centre pan for a subject-pivot orbit.')
    parts.append('The two still images cannot prove exact distance or continuous video motion. Mark an observable contradiction FAIL and insufficient observable evidence UNCERTAIN. Requested pose, action or anchor changes remain permitted; unrelated identity or topology changes remain FAIL.')
    return '\n'.join(parts)

def adapted_specs(specs, *, binding):
    result = copy.deepcopy(specs)
    motion = actual_motion_text(binding['current_caption'])
    result['movement']['prompt'] += '\n\nCurrent-turn motion-specific interpretation:\n' + motion
    result['entity']['prompt'] = result['entity']['prompt'].replace('same current action or body pose when visible', 'action and body pose compatible with the requested current action when visible').replace('changed action/pose/prop, duplicated', 'unrequested action/pose/prop changes, duplicated')
    result['entity']['prompt'] += '\nRequested subject movement or pose change is allowed; preserve identity and unchanged entity properties.'
    result['projection']['prompt'] = 'Judge the actual CANDIDATE_ERP physical-world projection. Use SOURCE_ERP as a separately labeled baseline when supplied. PASS requires observable coherent 2:1 full-sphere equirectangular coverage and wrapping; aspect ratio alone is insufficient. FAIL requires a concrete visible projection defect, such as broken left/right wrapping, a discontinuous seam or polar geometry, split/stacked panels, reference-image collage, rectilinear borders, or fixed screen-space UI baked into ERP. Name the location and defect in reason and retry_instruction. Unreadable or unconvincing pole/nadir coverage or insufficient stretching evidence alone is UNCERTAIN. Report inherited source defects separately; a matching source defect does not excuse a concrete invalid candidate. HUD, timestamp, REC/battery icons, watermarks and recording borders are UI. Physical LCD displays, signs and cameras are in-world scene content. The contact sheet and labeled reference views are diagnostic inputs and never candidate projection panels. Use the final combined response schema.'
    for key, value in result.items():
        value['inherited_resolved_prompt_sha256'] = specs[key]['resolved_prompt_sha256']
        value['resolved_prompt_sha256'] = hashlib.sha256(value['prompt'].encode()).hexdigest()
        value['adaptation'] = 'navigation_policy_round2'
    return result

def app_navigation_specs(*, subjects=(), context=''):
    # No spatial anchors are recorded; the anchor list stays empty.
    values = {'__ANCHORS_JSON__': '[]', '__SUBJECTS_JSON__': json.dumps(list(subjects), ensure_ascii=False, sort_keys=True), '__CONTEXT__': str(context), '__CONTEXT_ONE_LINE__': ' '.join(str(context).split())[:500]}
    result = {}
    for category in CRITERIA:
        template = read_text(f'navigation_{category}.txt')
        prompt = re.sub('__ANCHORS_JSON__|__SUBJECTS_JSON__|__CONTEXT_ONE_LINE__|__CONTEXT__', lambda m: values[m.group()], template)
        result[category] = {'prompt': prompt, 'component': category, 'resolved_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
    return result

def combined_navigation_request(frozen_verifier_request, *, candidate_erp_ref, comparison_sheet_ref, source_erp_ref=None, subjects=(), context='', run_root=None):
    if frozen_verifier_request.role != 'verifier':
        raise ValueError('Frozen base request must have role verifier.')
    specs = app_navigation_specs(subjects=subjects, context=context)
    bound = frozen_verifier_request.payload['memory_context']['endpoint_constraints']['official_row_binding']
    specs = adapted_specs(specs, binding=bound)
    response_format = read_json_data('navigation_response_format.json')
    system = frozen_verifier_request.system + '\n\n' + '\n\n'.join((f"APP_{key.upper()}_CRITERION:\n{specs[key]['prompt']}" for key in CRITERIA))
    system += '\nThe component output examples describe their original local protocols. For this combined request return ONLY the final strict response schema. Report criteria as PASS/FAIL/UNCERTAIN with insufficient evidence labeled UNCERTAIN. Missing spatial anchors or entity records are insufficient evidence, not measured success. Projection uses CANDIDATE_ERP and the separately labeled SOURCE_ERP baseline; movement uses original source/endpoint; spatial and entity use the labeled comparison contact sheet. Populate semantic with the unchanged frozen M1 four checks and lowercase pass/fail/uncertain. No criterion may imply unseen video execution. Fixed screen-space overlays are presentation metadata excluded from the physical ERP world. Their removal does not change physical scene or entity identity; preserve physical LCD displays, signs and cameras.'
    payload = {**dict(frozen_verifier_request.payload), 'navigation_motion_spec': {}, 'spatial_anchors': [], 'entity_subjects': list(subjects), 'navigation_context': context, 'component_source_refs': {k: {kk: vv for kk, vv in value.items() if kk != 'prompt'} for k, value in specs.items()}}
    source_images = () if source_erp_ref is None else (image_input(source_erp_ref, 'SOURCE_ERP: accepted source panorama; projection baseline only', run_root=run_root),)
    images = tuple(frozen_verifier_request.images) + source_images + (image_input(candidate_erp_ref, 'CANDIDATE_ERP: full panorama; projection criterion only', run_root=run_root), image_input(comparison_sheet_ref, 'SOURCE/TARGET perspective comparison contact sheet: spatial and entity continuity', run_root=run_root))
    return ChatRequest('verifier', system, images, payload, response_format)

def verify_navigation(request, *, chat_client):
    response = chat_client.complete(request)
    decision = navigation_decision(json.loads(response.text))
    return {**decision, 'requested_model': response.receipt.requested_model, 'returned_model': response.model, 'receipt': response.receipt.to_record(), 'state_committed': False}
