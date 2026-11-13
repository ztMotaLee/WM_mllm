"""Depth-bound CPU navigation and combined destination verification."""
from __future__ import annotations
import copy, hashlib, json, math, re
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from PIL import Image
from ..prompts import read_text, read_json_data
from ..providers.types import ChatRequest, ImageInput, ImageEditRequest
from ..state import panorama as geometry
from ..state import projection
from ..session.io import file_ref, verify_ref, resolve_run_path
from .contracts import verified_official_binding
from .planning import image_input
SEMANTIC_FIELDS = ('state_matches_source', 'action_preserves_request', 'endpoint_matches_action', 'scene_identity_preserved')
CRITERIA = ('projection', 'movement', 'spatial', 'entity')

def depth_not_ready(reason):
    return {'status': 'depth_not_ready', 'depth_ready': False, 'reason': reason, 'generative_navigation_required': True, 'recommended_next_lane': 'verified_generative_navigation', 'dispatch_performed': False, 'endpoint_complete': False}

def navigation_decision(value, *, allow_uncertain_continuation=True):
    if type(allow_uncertain_continuation) is not bool:
        raise ValueError('UNCERTAIN continuation permission must be an explicit boolean.')
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
    return {'status': status, 'criteria': dict(value['criteria']), 'verification': dict(sem), 'failed': failed, 'uncertain': uncertain, 'continuation_allowed': status == 'PASS' or (status == 'UNCERTAIN' and bool(allow_uncertain_continuation)), 'semantic_accepted': status == 'PASS', 'uncertain_accepted_and_labeled': status == 'UNCERTAIN' and bool(allow_uncertain_continuation), 'reason': value['reason'], 'retry_instruction': value['retry_instruction'], 'adopted_improvement': False}

def actual_motion_text(binding):
    bound = verified_official_binding(binding)
    motion = bound['official_motion_context']
    caption = bound['current_caption']
    pivot = str(motion.get('pivot') or '')
    direction = motion.get('translation_dir_cam')
    tracked = bool(motion.get('subject_screen_fixed'))
    orbit = 'orbit' in caption.lower() or ('subject point' in pivot.lower() and motion.get('rotation_axis_cam') is not None)
    parts = ['Current bound action: ' + caption, 'Official camera-local motion contract: ' + json.dumps(motion, ensure_ascii=False, sort_keys=True, separators=(',', ':'))]
    if orbit:
        parts.append('For this subject-pivot motion, rotation_axis_cam and rotation_sign describe the camera rig moving around its supplied pivot. Judge the changed physical viewpoint and coherent near/far landmark parallax. A stationary subject keeps its world orientation; spinning the subject does not satisfy camera orbit. Do not substitute a camera-centre pan for a subject-pivot orbit.')
    if isinstance(direction, (list, tuple)) and len(direction) == 3:
        dx, dy, dz = map(float, direction)
        if dz > 0:
            parts.append('This turn contains forward translation. Nearby stationary landmarks should have compatible approach scale, occlusion and parallax; distant scenery can have little measurable parallax.')
        elif dz < 0:
            parts.append('This turn contains backward translation. Judge retreat/pull-back evidence with compatible stationary-landmark scale, occlusion and parallax. Do not require forward-walking evidence.')
        if dx:
            side = 'right' if dx > 0 else 'left'
            parts.append(f'This turn contains camera-local {side} translation. Judge lateral viewpoint change and depth-dependent landmark parallax; a flat horizontal image shift or camera-centre pan alone does not establish translation.')
        if dy:
            parts.append('This turn contains vertical translation. Judge the changed physical eye height and depth-dependent parallax, together with the supplied rotation/pivot contract.')
    if tracked:
        parts.append('This is subject-following/tracking with subject_screen_fixed. Keep the named subject centered at the requested source-relative framing and size unless the current caption explicitly changes it. When subject_translation_equals_camera_translation is true, the subject and camera translate equally; that requested subject relocation does not violate its previous world anchor.')
    if motion.get('ego_attached_foreground_screen_fixed'):
        parts.append('Camera-attached hands/equipment are attached foreground. Preserve them once in source-relative screen framing unless the current action changes them; judge stationary world landmarks separately.')
    parts.append('The two still images cannot prove exact distance or continuous video motion. Mark an observable contradiction FAIL and insufficient observable evidence UNCERTAIN. Requested pose, action or anchor changes remain permitted; unrelated identity or topology changes remain FAIL.')
    return '\n'.join(parts)

def adapted_specs(specs, *, binding):
    result = copy.deepcopy(specs)
    motion = actual_motion_text(binding)
    result['movement']['prompt'] += '\n\nCurrent-turn motion-specific interpretation:\n' + motion
    result['entity']['prompt'] = result['entity']['prompt'].replace('same current action or body pose when visible', 'action and body pose compatible with the requested current action when visible').replace('changed action/pose/prop, duplicated', 'unrequested action/pose/prop changes, duplicated')
    result['entity']['prompt'] += '\nRequested subject movement or pose change is allowed; preserve identity and unchanged entity properties.'
    result['projection']['prompt'] = 'Judge the actual CANDIDATE_ERP physical-world projection. Use SOURCE_ERP as a separately labeled baseline when supplied. PASS requires observable coherent 2:1 full-sphere equirectangular coverage and wrapping; aspect ratio alone is insufficient. FAIL requires a concrete visible projection defect, such as broken left/right wrapping, a discontinuous seam or polar geometry, split/stacked panels, reference-image collage, rectilinear borders, or fixed screen-space UI baked into ERP. Name the location and defect in reason and retry_instruction. Unreadable or unconvincing pole/nadir coverage or insufficient stretching evidence alone is UNCERTAIN. Report inherited source defects separately; a matching source defect does not excuse a concrete invalid candidate. HUD, timestamp, REC/battery icons, watermarks and recording borders are UI. Physical LCD displays, signs and cameras are in-world scene content. The contact sheet and labeled reference views are diagnostic inputs and never candidate projection panels. Use the final combined response schema.'
    for key, value in result.items():
        value['inherited_resolved_prompt_sha256'] = specs[key]['resolved_prompt_sha256']
        value['resolved_prompt_sha256'] = hashlib.sha256(value['prompt'].encode()).hexdigest()
        value['adaptation'] = 'navigation_policy_round2'
    return result

def app_navigation_specs(*, anchors=(), subjects=(), context=''):
    values = {'__ANCHORS_JSON__': json.dumps(list(anchors), ensure_ascii=False, sort_keys=True), '__SUBJECTS_JSON__': json.dumps(list(subjects), ensure_ascii=False, sort_keys=True), '__CONTEXT__': str(context), '__CONTEXT_ONE_LINE__': ' '.join(str(context).split())[:500]}
    result = {}
    for category in CRITERIA:
        template = read_text(f'navigation_{category}.txt')
        prompt = re.sub('__ANCHORS_JSON__|__SUBJECTS_JSON__|__CONTEXT_ONE_LINE__|__CONTEXT__', lambda m: values[m.group()], template)
        result[category] = {'prompt': prompt, 'component': category, 'resolved_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
    return result

def combined_navigation_request(frozen_verifier_request, *, candidate_erp_ref, comparison_sheet_ref, source_erp_ref=None, anchors=(), subjects=(), motion_spec=None, context='', run_root=None):
    if frozen_verifier_request.role != 'verifier':
        raise ValueError('Frozen base request must have role verifier.')
    specs = app_navigation_specs(anchors=anchors, subjects=subjects, context=context)
    bound = frozen_verifier_request.payload['memory_context']['endpoint_constraints']['official_row_binding']
    specs = adapted_specs(specs, binding=bound)
    response_format = read_json_data('navigation_response_format.json')
    system = frozen_verifier_request.system + '\n\n' + '\n\n'.join((f"APP_{key.upper()}_CRITERION:\n{specs[key]['prompt']}" for key in CRITERIA))
    system += '\nThe component output examples describe their original local protocols. For this combined request return ONLY the final strict response schema. Report criteria as PASS/FAIL/UNCERTAIN with insufficient evidence labeled UNCERTAIN. Missing spatial anchors or entity records are insufficient evidence, not measured success. Projection uses CANDIDATE_ERP and the separately labeled SOURCE_ERP baseline; movement uses original source/endpoint; spatial and entity use the labeled comparison contact sheet. Populate semantic with the unchanged frozen M1 four checks and lowercase pass/fail/uncertain. No criterion may imply unseen video execution. Fixed screen-space overlays are presentation metadata excluded from the physical ERP world. Their removal does not change physical scene or entity identity; preserve physical LCD displays, signs and cameras.'
    payload = {**dict(frozen_verifier_request.payload), 'navigation_motion_spec': dict(motion_spec or {}), 'spatial_anchors': list(anchors), 'entity_subjects': list(subjects), 'navigation_context': context, 'component_source_refs': {k: {kk: vv for kk, vv in value.items() if kk != 'prompt'} for k, value in specs.items()}}
    source_images = () if source_erp_ref is None else (image_input(source_erp_ref, 'SOURCE_ERP: accepted source panorama; projection baseline only', run_root=run_root),)
    images = tuple(frozen_verifier_request.images) + source_images + (image_input(candidate_erp_ref, 'CANDIDATE_ERP: full panorama; projection criterion only', run_root=run_root), image_input(comparison_sheet_ref, 'SOURCE/TARGET perspective comparison contact sheet: spatial and entity continuity', run_root=run_root))
    return ChatRequest('verifier', system, images, payload, response_format)

def verify_navigation(request, *, chat_client, allow_uncertain_continuation=True):
    response = chat_client.complete(request)
    decision = navigation_decision(json.loads(response.text), allow_uncertain_continuation=allow_uncertain_continuation)
    return {**decision, 'requested_model': response.receipt.requested_model, 'returned_model': response.model, 'receipt': response.receipt.to_record(), 'state_committed': False}

def validate_depth(state, depth_record, *, run_root):
    if depth_record is None:
        return depth_not_ready('No state-bound radial depth product supplied.')
    required = {'path', 'sha256', 'panorama_state_sha256', 'erp_pixel_sha256', 'units'}
    if not required.issubset(depth_record):
        return depth_not_ready('Depth provenance is incomplete.')
    if depth_record['units'] != 'radial_eye_heights':
        return depth_not_ready('Depth units must be radial_eye_heights.')
    if depth_record['panorama_state_sha256'] != geometry.panorama_state_sha256(state):
        return depth_not_ready('Depth belongs to another panorama state.')
    erp = geometry.erp_array(state)
    if depth_record['erp_pixel_sha256'] != geometry.pixel_sha256(erp):
        return depth_not_ready('Depth RGB binding differs from current ERP pixels.')
    path = resolve_run_path(run_root, depth_record['path'])
    try:
        if hashlib.sha256(path.read_bytes()).hexdigest() != depth_record['sha256']:
            return depth_not_ready('Depth encoded-byte hash changed.')
        product = np.load(path, allow_pickle=False)
        if isinstance(product, np.lib.npyio.NpzFile):
            try:
                key = next((k for k in ('depth', 'depth_map') if k in product), None)
                if key is None:
                    return depth_not_ready('Depth npz lacks depth/depth_map.')
                depth = np.asarray(product[key], dtype=np.float32)
            finally:
                product.close()
        else:
            depth = np.asarray(product, dtype=np.float32)
    except (OSError, ValueError, TypeError):
        return depth_not_ready('Depth product cannot be read.')
    if depth.shape != erp.shape[:2]:
        return depth_not_ready('Depth shape differs from exact ERP shape; no resize permitted.')
    if not np.isfinite(depth).all() or np.any(depth <= 0.05):
        return depth_not_ready('Depth must be finite and greater than .05 eye heights at every pixel.')
    return {'status': 'depth_ready', 'depth_ready': True, 'depth': depth, 'depth_ref': dict(depth_record), 'source_state_sha256': geometry.panorama_state_sha256(state), 'source_erp_pixel_sha256': geometry.pixel_sha256(erp), 'dispatch_performed': False}

def prepare_depth_forward(state, depth_record, *, yaw=0.0, step=0.45, run_root):
    validated = validate_depth(state, depth_record, run_root=run_root)
    if not validated['depth_ready']:
        return validated
    if not math.isfinite(step) or step <= 0:
        raise ValueError('Forward step must be finite and positive.')
    if not math.isfinite(yaw):
        raise ValueError('Walking yaw must be finite degrees.')
    depth = validated['depth']
    result = projection.reproject_forward(Image.fromarray(geometry.erp_array(state)), depth_map=depth, step=float(step), yaw=math.radians(float(yaw)), far_depth_eye_heights=max(8.0, float(depth.max())))
    return {'status': 'depth_warp_prepared', 'depth_ready': True, 'warped_erp': np.asarray(result.image).copy(), 'hole_mask': np.asarray(result.hole_mask).copy(), 'warped_depth': result.depth_map, 'stats': result.stats, 'source_depth_ref': dict(depth_record), 'source_state_sha256': validated['source_state_sha256'], 'yaw_degrees': float(yaw), 'step_eye_heights': float(step), 'endpoint_complete': not bool(np.any(np.asarray(result.hole_mask) > 0)), 'fill_required': bool(np.any(np.asarray(result.hole_mask) > 0)), 'generative_navigation_required': False, 'dispatch_performed': False, 'viewport_profile': dict(state['viewport_profile'])}

def fill_forward_holes(prepared, *, image_client, output_dir, scene_prompt='', travel_hint='', deps):
    if prepared.get('status') == 'depth_not_ready':
        return dict(prepared)
    if prepared.get('status') != 'depth_warp_prepared':
        raise ValueError('A validated prepared depth warp is required.')
    warped = geometry.rgb_array(prepared['warped_erp'])
    mask = np.asarray(prepared['hole_mask'])
    if mask.dtype != np.uint8 or mask.shape != warped.shape[:2]:
        raise ValueError('Actual app hole mask must be aligned uint8 L.')
    if not np.any(mask):
        return {**prepared, 'status': 'depth_endpoint_candidate', 'candidate_erp': warped, 'verification_required': True, 'dispatch_performed': False, 'depth_reestimate_required': False}
    if np.all(mask > 0):
        raise ValueError('No protected geometry remains; use explicitly verified generative navigation.')
    directory = resolve_run_path(deps.run_root, Path(output_dir).relative_to(Path(deps.run_root)).as_posix(), must_exist=False)
    directory.mkdir(parents=True, exist_ok=True)
    w = warped.shape[1]
    yaw = math.radians(prepared['yaw_degrees'])
    wedge_u = ((yaw + math.pi) / (2 * math.pi) + 0.5) % 1
    roll = int(round((0.5 - wedge_u) % 1 * w))
    image_path = directory / 'DEPTH_WARP_ROLLED.png'
    mask_path = directory / 'REAL_HOLES_ROLLED.png'
    from ..session.io import save_png
    source_ref = save_png(image_path, np.roll(warped, roll, axis=1), run_root=deps.run_root)
    mask_ref = save_png(mask_path, np.roll(mask, roll, axis=1), run_root=deps.run_root)
    prompt = projection.hole_fill_instruction(scene_prompt, travel_hint)
    fill_dir = directory / 'image_fill'
    resolve_run_path(deps.run_root, fill_dir.relative_to(deps.run_root).as_posix(), must_exist=False)
    result = image_client.edit(ImageEditRequest(prompt, (image_input(source_ref, '', run_root=deps.run_root),), fill_dir, mask=image_input(mask_ref, '', run_root=deps.run_root), panorama=True))
    filled = geometry.rgb_array(result.image)
    if filled.shape[1] != 2 * filled.shape[0]:
        raise ValueError('Fill output must retain true ERP2:1 aspect.')
    if filled.shape != warped.shape:
        filled = np.asarray(Image.fromarray(filled).resize((warped.shape[1], warped.shape[0]), Image.Resampling.LANCZOS))
    filled = np.roll(filled, -roll, axis=1)
    alpha = mask.astype(np.float64) / 255
    candidate = np.rint(warped * (1 - alpha[..., None]) + filled * alpha[..., None]).clip(0, 255).astype(np.uint8)
    outside = mask == 0
    changed = int(np.count_nonzero(np.any(candidate[outside] != warped[outside], axis=-1)))
    if changed:
        raise RuntimeError('Depth fill changed protected warped ERP pixels.')
    return {**prepared, 'status': 'depth_endpoint_candidate', 'candidate_erp': candidate, 'verification_required': True, 'dispatch_performed': True, 'image': file_ref(result.image, run_root=deps.run_root, image=True), 'image_receipt': result.receipt.to_record(), 'outside_holes_changed_pixels': changed, 'outside_holes_pixels': int(outside.sum()), 'depth_reestimate_required': True, 'warped_depth': None, 'requested_model': result.receipt.requested_model, 'returned_model': result.receipt.returned_model, 'fill_instruction_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'state_committed': False}
