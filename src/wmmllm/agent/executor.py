"""Frozen accepted-state transition algorithm, using injected provider interfaces."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping
import time

import numpy as np
from PIL import Image, ImageFilter

from ..budget import BudgetController
from ..config import AppConfig
from ..logging_utils import redact, safe_error
from ..memory.observer import ObserverDeps, observe_state
from ..memory.retrieval import retrieve
from ..providers.errors import TransportExhausted
from ..providers.factory import Providers
from ..providers.retry import raise_if_fatal_provider_error, safe_provider_error
from ..providers.types import ChatRequest, ImageEditRequest, VideoTransition
from ..session.io import (IntegrityError, canonical_sha256, file_ref, public_ref, read_json,
                          resolve_run_path, verify_ref, verify_references, write_once, save_png)
from ..session.receipts import recorded_providers
from ..state.graph import public_viewport_profile, verify_saved_node, visit_key
from ..state.panorama import (composite_view_edit, create_panorama_state, erp_array,
                              load_panorama_state, render_view, save_panorama_state)
from ..state.registration import blend_endpoint
from ..video.presentation import present_transition
from ..prompts import read_json_data, read_text, policy
from . import contracts, navigation
from . import bootstrap
from .failures import classify_failure, exception_record
from .planning import (Plan, SegmentRequest, SemanticDecision, draw_polygons, endpoint_plan,
                       endpoint_request, image_input, support_mask, verifier_request)
from .router import decide, primitive_decision


@dataclass(frozen=True)
class EngineDeps:
    config: AppConfig
    providers: Providers
    budget: BudgetController
    run_root: Path
    logger: Any

    def __post_init__(self):
        object.__setattr__(self, 'providers', recorded_providers(self.providers, run_root=self.run_root,
                                                                secrets=self.config.secrets()))


class EndpointRejected(RuntimeError):
    def __init__(self, message, *, attempts=(), verdict=None):
        super().__init__(message)
        self.attempts = list(attempts)
        self.verdict = verdict


def rasterize_support_polygons(polygons, width=1344, height=768, *, dilate_pixels=16):
    for polygon in polygons:
        points = np.asarray(polygon, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 1):
            raise ValueError('invalid actual support polygon')
    image = draw_polygons(polygons, width, height)
    if dilate_pixels:
        image = image.filter(ImageFilter.MaxFilter(dilate_pixels * 2 + 1))
    result = np.asarray(image)
    if not np.any(result):
        raise ValueError('empty actual generated support')
    return result


def _save(path, pixels, deps):
    return save_png(path, pixels, run_root=deps.run_root)


def _pixels(ref, deps):
    return np.asarray(Image.open(verify_ref(ref, run_root=deps.run_root)).convert('RGB')).copy()


def _pose(pose):
    return {key: float(pose[key]) for key in ('yaw', 'pitch')}


def _request(row, source, store, payload, out, deps, history=None):
    context_ref = write_once(out / 'MEMORY_CONTEXT.json', payload, run_root=deps.run_root)
    request = SegmentRequest(segment=row['segment'], current_caption=row['current_caption'],
        preceding_captions=tuple(c['caption'] for c in store.snapshot().commits), source=source,
        memory_payload=payload, memory_context=context_ref, historical_reference=history,
        planner_duration_seconds=row['profile']['num_frames'] / row['profile']['fps'], run_root=deps.run_root)
    request.verify_inputs(run_root=deps.run_root)
    return request


def _check(request, *, deps):
    response = deps.providers.chat.complete(request)
    checked = policy.validate_semantic_verification(policy.parse_response(response.text))
    status = policy.semantic_decision(checked)
    return SemanticDecision(status['semantic_status'], checked, tuple(status['failed_criteria']),
                            tuple(status['uncertain_criteria']), checked['reason']), response.receipt.to_record()


def _editor(request, *, deps):
    # Containment is checked before dispatch, including every existing parent.
    relative = Path(request.output_dir).relative_to(Path(deps.run_root).resolve()).as_posix()
    resolve_run_path(deps.run_root, relative, must_exist=False)
    return deps.providers.image.edit(request)


def state_endpoint(row: Mapping, *, panorama, source, pose, end_pose, store,
                   context, out: Path, deps: EngineDeps) -> dict:
    """Propose at most twice; select actual support and check the composed endpoint."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    decision = row['_planner_decision']
    edit_pose = _pose(decision['editing_camera'])
    edit_hfov = float(decision['editing_hfov'])
    pano_ref = row['_panorama_ref']
    physical_pano_ref = row.get('_execution_panorama_ref', pano_ref)
    width, height = row['profile']['width'], row['profile']['height']
    auxiliary_needed = row.get('_execution_panorama_ref') is not None or edit_pose != _pose(pose) or _pose(end_pose) != _pose(pose)
    editing_crop = _save(out / 'EDITING_VIEW.png', render_view(panorama, **edit_pose, hfov=edit_hfov, width=width, height=height), deps)
    memory = retrieve(store, segment=row['segment'], caption=row['current_caption'], source=source,
                      preceding_captions=[c['caption'] for c in store.snapshot().commits])
    constraints = contracts.build_endpoint_constraints(row, decision, source=source, panorama=pano_ref)
    write_once(out / 'ENDPOINT_CONSTRAINTS.json', constraints, run_root=deps.run_root)
    payload = {'memory_text': memory.memory_text, 'root_style_anchor': context['style_tokens'],
        'root_entity_identity_properties': context['entity_identity_properties'],
        'known_subjects': store.known_subjects(row['segment']), 'editing_camera': edit_pose,
        'editing_hfov': edit_hfov, 'geometric_camera_delta': {k:end_pose[k] - pose[k] for k in ('yaw', 'pitch')},
        'panorama_state': 'Persistent360-degree ERP; current view is a gnomonic crop.',
        'endpoint_constraints': constraints, 'operation_decision': primitive_decision(decision)}
    holds = failure_hold_history(store, deps=deps)
    if holds:
        payload['prior_failure_holds'] = holds
    request = _request(row, source, store, payload, out, deps, memory.historical_reference)
    operations = [op['operation'] for op in decision['operations']]
    global_edit = 'global_change' in operations or any(op['scope'] == 'global' for op in decision['operations'])
    is_navigation = 'navigate' in operations
    attempts = []
    feedback = None
    for attempt in (1, 2):
        stage = out / 'attempts' / f'{attempt:02d}'
        stage.mkdir(parents=True, exist_ok=True)
        if attempt == 2:
            editor_source = physical_pano_ref if global_edit or is_navigation else attempts[0]['physical_source']
            feedback = contracts.build_feedback(current_attempt=attempt, attempts=attempts, stage=out,
                constraints=constraints, actual_editor_source=editor_source, navigation=is_navigation, run_root=deps.run_root)
            write_once(out / 'SEMANTIC_RETRY_FEEDBACK.json', feedback, run_root=deps.run_root)
        prequest = endpoint_request(request, decision=decision, payload=payload, editing_crop=editing_crop,
            physical_pano_ref=physical_pano_ref, history=memory.historical_reference, feedback=feedback)
        planned = endpoint_plan(chat=deps.providers.chat, request=prequest, decision=row.get('_decision_ref',decision),
                                caption=row['current_caption'], out=stage, deps=deps)
        bbox, polygons, final_boxes = (planned[k] for k in ('scope_bbox','support_polygons','final_view_entity_bboxes'))
        plan_dict = planned['plan']
        plan = Plan.from_policy_dict(plan_dict)
        support = np.minimum(support_mask(bbox, width, height), rasterize_support_polygons(polygons, width, height, dilate_pixels=0))
        if not np.any(support):
            raise ValueError('empty actual permitted support')
        plan_ref = write_once(stage / 'PLAN.json', planned, run_root=deps.run_root)
        physical_source = _save(stage / 'EDIT_CROP.png', render_view(panorama, **edit_pose, hfov=edit_hfov, width=width, height=height), deps)
        prompt = policy.image_prompt(plan_dict)
        if any(payload['geometric_camera_delta'].values()):
            prompt += ' Keep the camera fixed; pan/tilt is handled by geometric rendering.'
        prompt += ' Preserve this root style: ' + context['style_tokens'] + '. Accepted memory: ' + memory.memory_text[:2200]
        if global_edit or is_navigation:
            prompt = contracts.build_erp_endpoint_prompt(plan_instruction=plan.endpoint_instruction, constraints=constraints,
                source_pose=pose, ending_pose=end_pose, root_style=context['style_tokens'], memory_text=memory.memory_text,
                viewport_profile=panorama['viewport_profile'])
            prompt = contracts.append_endpoint_constraints(prompt, constraints)
            coords = contracts.coordinate_record(editing_bbox=bbox, final_entity_bboxes=final_boxes,
                constraints=constraints, viewport_profile=panorama['viewport_profile'], editing_view_ref=editing_crop, source_view_ref=source)
            write_once(stage / 'ERP_EDIT_COORDINATES.json', coords, run_root=deps.run_root)
            images = (image_input(physical_pano_ref, run_root=deps.run_root), image_input(source, run_root=deps.run_root))
            prompt, images = contracts.editor_coordinate_contract(prompt, images, coords, run_root=deps.run_root)
            if feedback:
                prompt, images = contracts.editor_feedback(prompt, images, feedback, run_root=deps.run_root)
            result = _editor(ImageEditRequest(prompt, images, stage / 'image', panorama=True), deps=deps)
            image_ref = file_ref(result.image, run_root=deps.run_root, image=True)
            next_erp = np.asarray(Image.open(result.image).convert('RGB').resize((erp_array(panorama).shape[1], erp_array(panorama).shape[0]), Image.Resampling.LANCZOS))
            editor_receipt = result.receipt.to_record()
            editor_inputs = [file_ref(item.path, run_root=deps.run_root, image=True) for item in images]
            next_panorama = create_panorama_state(next_erp, None, width=erp_array(panorama).shape[1], height=erp_array(panorama).shape[0])
            next_panorama['viewport_profile'] = dict(panorama['viewport_profile'])
            compositing = {'scope':'whole_panorama', 'outside_preservation_applicable':False,
                            'depth_forward_lane':'depth_not_ready' if is_navigation else 'not_applicable'}
        else:
            mask_path = stage / 'SUPPORT.png'
            save_png(mask_path, support, run_root=deps.run_root)
            mask_ref = file_ref(mask_path, run_root=deps.run_root)
            identity, identity_record = contracts.extract_reference(physical_source, polygons, stage / 'identity',
                upscale_min_edge=256, run_root=deps.run_root)
            write_once(stage / 'IDENTITY_CONDITIONING.json', identity_record, run_root=deps.run_root)
            images = (image_input(physical_source, run_root=deps.run_root),)
            if identity is not None:
                images += (image_input(identity, run_root=deps.run_root),)
                prompt += read_text('identity_image_suffix.txt')
            prompt = contracts.append_endpoint_constraints(prompt, constraints)
            if feedback:
                prompt, images = contracts.editor_feedback(prompt, images, feedback, run_root=deps.run_root)
            mask_input = None if len(images) > 1 or np.all(support == 255) else image_input(mask_ref, run_root=deps.run_root)
            result = _editor(ImageEditRequest(prompt, images, stage / 'image', mask=mask_input), deps=deps)
            image_ref = file_ref(result.image, run_root=deps.run_root, image=True)
            editor_receipt = result.receipt.to_record()
            editor_inputs = [file_ref(item.path, run_root=deps.run_root, image=True) for item in images]
            edited = _pixels(image_ref, deps)
            if not np.all(support == 255):
                support_request = ChatRequest('planner', read_text('actual_support_system.txt'),
                    (image_input(physical_source, 'source EDITING_VIEW', run_root=deps.run_root),
                     image_input(image_ref, 'actual generated EDITING_VIEW ending', run_root=deps.run_root)),
                    {'caption':row['current_caption'], 'endpoint_instruction':plan.endpoint_instruction, 'allowed_bbox':bbox,
                     'previous_complete_verifier_failure':feedback['complete_verifier_failure'] if feedback else None},
                    read_json_data('actual_support_response_format.json'))
                selected = deps.providers.chat.complete(support_request)
                actual_polygons = json.loads(selected.text)['support_polygons']
                support = rasterize_support_polygons(actual_polygons, width, height)
                write_once(stage / 'ACTUAL_SUPPORT.json', {'polygons':actual_polygons,'allowed_bbox':bbox,
                    'source':physical_source,'generated':image_ref,'selection_receipt':selected.receipt.to_record()}, run_root=deps.run_root)
                _save(stage / 'ACTUAL_SUPPORT_RGB.png', np.repeat(support[:,:,None],3,axis=2), deps)
            edited, blend = blend_endpoint(_pixels(physical_source,deps), edited, support, feather_pixels=12,
                max_translation_pixels=3.0, max_rotation_degrees=.05, max_corner_displacement_pixels=3.0)
            write_once(stage / 'ENDPOINT_BLEND.json', blend, run_root=deps.run_root)
            _save(stage / 'BLENDED_EDIT.png', edited, deps)
            next_panorama, alpha, compositing = composite_view_edit(panorama, edited, **edit_pose,
                hfov=edit_hfov, support=support)
            outside = alpha == 0
            changed = np.any(erp_array(next_panorama)[outside] != erp_array(panorama)[outside], axis=1)
            compositing = {**compositing,'native_preblend':blend,'scope':'gnomonic_crop_edit_with_explicit_support',
                'outside_changed_pixels':int(changed.sum()),'outside_pixels':int(outside.sum()),'outside_exact':bool(not changed.any())}
            if changed.any():
                raise IntegrityError('scoped edit changed protected ERP pixels')
        endpoint = _save(stage / 'ENDPOINT.png', render_view(next_panorama, **_pose(end_pose), hfov=90., width=width, height=height), deps)
        vrequest = verifier_request(request, plan, endpoint)
        if is_navigation:
            candidate_erp = _save(stage / 'NAV_CANDIDATE_ERP.png', erp_array(next_panorama), deps)
            sheet = np.concatenate((_pixels(source,deps),_pixels(endpoint,deps)), axis=1)
            comparison = _save(stage / 'NAV_COMPARISON.png', sheet, deps)
            vrequest = navigation.combined_navigation_request(vrequest, candidate_erp_ref=candidate_erp,
                comparison_sheet_ref=comparison, source_erp_ref=physical_pano_ref,
                subjects=list(store.known_subjects(row['segment']).values()),
                context=memory.memory_text, run_root=deps.run_root)
            vrequest = contracts.spatial_verifier_with_actual_motion(vrequest, constraints['official_row_binding'])
            checked = navigation.verify_navigation(vrequest, chat_client=deps.providers.chat)
            verdict = SemanticDecision(checked['status'], checked['verification'], tuple(checked['failed']), tuple(checked['uncertain']), checked['reason'])
            verification_ref = write_once(out / f'NAVIGATION_VERIFY_{attempt}.json', checked, run_root=deps.run_root)
            verifier_receipt = checked.get('receipt')
        else:
            baseline = _save(stage / 'UNMODIFIED_VIEWER_BASELINE.png', render_view(panorama, **_pose(end_pose), hfov=90., width=width, height=height), deps)
            suffix = read_text('verifier_auxiliary_integrity.txt' if auxiliary_needed else 'verifier_simple_integrity.txt')
            suffix = ' Bootstrap seam status: ' + row.get('_seam_status','UNKNOWN') + suffix
            extra = (image_input(baseline,'unmodified accepted panorama at ending viewer pose; inherited baseline',run_root=deps.run_root),)
            if auxiliary_needed:
                ending_physical = _save(stage / 'PHYSICAL_END_VIEW.png',render_view(next_panorama,**edit_pose,hfov=edit_hfov,width=width,height=height),deps)
                extra += (image_input(physical_source,'starting physical view at editing_camera; state index0',run_root=deps.run_root),
                          image_input(ending_physical,'candidate ending physical view at editing_camera; state index1',run_root=deps.run_root))
            vrequest = replace(vrequest,system=vrequest.system+suffix,images=vrequest.images+extra)
            verdict, verifier_receipt = _check(vrequest,deps=deps)
            verification_ref = write_once(out / f'VERIFIER_{attempt}.json', verdict.to_record(),run_root=deps.run_root)
        attempts.append({'attempt':attempt,'plan':plan_ref,'image':image_ref,'endpoint':endpoint,
            'semantic':verdict.to_record(),'compositing':compositing,'physical_source':physical_source,
            'verification_ref':verification_ref,'editor_receipt':editor_receipt,'verifier_receipt':verifier_receipt,
            'editor_inputs':editor_inputs,'verifier_source':source})
        if verdict.status != 'FAIL':
            break
    write_once(out / 'ATTEMPTS.json', attempts, run_root=deps.run_root)
    if verdict.status == 'FAIL':
        raise EndpointRejected('endpoint rejected after exactly one semantic retry', attempts=attempts, verdict=verdict)
    auxiliary = ()
    if auxiliary_needed:
        physical_end = _save(out / 'PHYSICAL_END_VIEW.png',render_view(next_panorama,**edit_pose,hfov=edit_hfov,width=width,height=height),deps)
        auxiliary = (image_input(physical_source,'starting physical state view; index0',run_root=deps.run_root),
                     image_input(physical_end,'accepted ending physical state view; index1',run_root=deps.run_root))
    return {'panorama':next_panorama,'endpoint':endpoint,'verdict':verdict,'verification':verification_ref,
            'attempt_count':len(attempts),'plan':plan,'auxiliary_images':auxiliary,'attempts':attempts}


def restore_endpoint(row: Mapping, *, target, source, current_revision, store, out, deps) -> dict:
    saved_state = verify_saved_node(target, run_root=deps.run_root)
    if target['physical_revision'] != current_revision:
        raise IntegrityError('Cannot revisit a stale physical revision')
    decision = row['_planner_decision']
    if _pose(target['pose']) != _pose(decision['ending_camera']):
        raise IntegrityError('Planner return pose differs from its saved node')
    endpoint = target['view']
    expected = render_view(saved_state, **_pose(target['pose']), hfov=target['pose']['hfov'],
                           width=row['profile']['width'],height=row['profile']['height'])
    if not np.array_equal(_pixels(endpoint,deps),expected):
        raise IntegrityError('Saved revisit endpoint differs from its camera render')
    chain_memory = store.snapshot_to(out/'RETURN_CURRENT_CHAIN_MEMORY.json')
    request = _request(row,source,store,{'operation_decision':primitive_decision(decision),
        'saved_world_memory':target['memory_snapshot'],'retained_accepted_chain_memory':chain_memory},out,deps)
    plan = Plan('Current accepted scene.',row['current_caption'],decision['endpoint_instruction'])
    verdict,receipt = _check(verifier_request(request,plan,endpoint),deps=deps)
    verification = write_once(out / 'RETURN_VERIFIER.json',verdict.to_record(),run_root=deps.run_root)
    if verdict.status == 'FAIL':
        raise EndpointRejected('Saved return semantic check failed',verdict=verdict)
    return {'panorama':saved_state,'endpoint':endpoint,'verdict':verdict,'verification':verification,
        'attempt_count':1,'plan':plan,'auxiliary_images':(),'restore_semantics':target['memory_snapshot'],
        'origin_id':target['origin_id'],'view_identity':target['view_identity'],'panorama_ref':target['panorama'],
        'state_ref':target['panorama_state'],'parent_node_id':target['node_id'],
        'observer_subjects':read_json(target['memory_snapshot'],run_root=deps.run_root)['state']['subjects']}


def _classify(error):
    record = exception_record(error)
    if getattr(error, 'stage_error', None):
        record['stage_error'] = dict(error.stage_error)
    return classify_failure(type(error).__name__, str(error), record)


def _presentation_record(result, row, deps):
    profile = row['profile']
    return {'status':result.status,'backend':result.backend,
        'video':None if result.video_path is None else file_ref(result.video_path,run_root=deps.run_root),
        'reason':result.reason,'num_frames':profile['num_frames'],'fps':profile['fps'],
        'duration_seconds':profile['num_frames']/profile['fps']}


def execute_transition(row: Mapping, *, session, deps: EngineDeps) -> dict:
    current = session.current_node
    source = current['view']
    verify_ref(source,run_root=deps.run_root)
    panorama = load_panorama_state(current['panorama_state'],run_root=deps.run_root)
    pose = _pose(current['pose'])
    revision = current['physical_revision']
    index = row['segment']
    tid = f'turn{index:04d}'
    out = deps.run_root / 'transitions' / tid
    out.mkdir(parents=True,exist_ok=True)
    decision = None
    decision_ref = None
    staged = None
    try:
        if (out/'ACCEPTED_READY.json').exists():
            ready = read_json(out/'ACCEPTED_READY.json',run_root=deps.run_root)
            expected = {'session_id':session.record['session_id'],'source_node_id':current['node_id'],
                'source':source,'panorama_state':current['panorama_state'],'caption':row['current_caption'],
                'seed':row['seed'],'segment':index,'profile':row['profile'],'revision':revision}
            if ready.get('binding') != expected or ready.get('sha256') != canonical_sha256({k:v for k,v in ready.items() if k!='sha256'}):
                raise IntegrityError('Accepted-ready checkpoint binding differs')
            verify_references({k:v for k,v in ready.items() if k!='sha256'},run_root=deps.run_root)
            decision=ready['decision']; decision_ref=ready['decision_ref']
            staged=dict(ready['staged']); staged['verdict']=SemanticDecision.from_record(staged['verdict'])
            staged['auxiliary_images']=tuple(image_input(item['image'],item['label'],run_root=deps.run_root) for item in staged['auxiliary_images'])
            staged['panorama']=load_panorama_state(ready['node']['panorama_state'],run_root=deps.run_root)
            return _observe_and_settle(row,session=session,deps=deps,current=current,decision_ref=decision_ref,
                operations=ready['operations'],node=ready['node'],staged=staged,presentation=ready['presentation'],pure=ready['pure'],out=out)
        prior_holds = failure_hold_history(session.store, deps=deps)
        if prior_holds:
            row = {**row, 'prior_failure_holds': prior_holds}
        decision = decide(row,out,source,current['panorama'],pose,session.store,
                          session.context,session.preceding,session.visits,deps=deps)
        decision_ref = write_once(out / 'decision.json',decision,run_root=deps.run_root)
        row = {**row,'_planner_decision':decision,'_decision_ref':decision_ref,
               '_panorama_ref':current['panorama'],'_seam_status':current['seam_status']}
        operations = [op['operation'] for op in decision['operations']]
        end_pose = _pose(decision['ending_camera'])
        row['_seam_status'] = check_actual_reachable_pose(row, out, session, panorama, end_pose, deps=deps)
        target = next((node for node in session.record['nodes'] if node['node_id'] == decision.get('restore_node')),None)
        physical_ops = any(op in ('navigate','local_edit','global_change','subject_action') for op in operations)
        pure = not physical_ops and all(op in ('rotate','revisit') for op in operations) and (target is None or
            (target['origin_id'] == current['origin_id'] and target['view_identity'] == current['view_identity']))
        if target is not None and target['physical_revision'] != revision:
            raise IntegrityError('Planner revisit points to a stale physical revision')
        if target is not None and (_pose(target['pose']) != end_pose or target['pose']['hfov'] != 90. or
                target['viewport_profile'] != current['viewport_profile']):
            raise IntegrityError('Planner return pose/profile differs from its saved node')
        if pure:
            saved = session.visits.get(visit_key(revision,current['view_identity'],end_pose))
            if saved is not None:
                if saved['origin_id'] != current['origin_id'] or saved['viewport_profile'] != current['viewport_profile']:
                    raise IntegrityError('Saved view origin/profile mismatch')
                verify_saved_node(saved,run_root=deps.run_root)
                endpoint = saved['view']
            else:
                endpoint = _save(out/'ENDPOINT.png',render_view(panorama,**end_pose,hfov=90.,
                    width=row['profile']['width'],height=row['profile']['height']),deps)
            plan = Plan('Current accepted panorama and camera view.',row['current_caption'],decision['endpoint_instruction'])
            request = _request(row,source,session.store,{'operation_decision':primitive_decision(decision)},out,deps)
            verdict,receipt = _check(verifier_request(request,plan,endpoint),deps=deps)
            verification = write_once(out/'GEOMETRIC_VERIFIER.json',verdict.to_record(),run_root=deps.run_root)
            if verdict.status == 'FAIL':
                raise EndpointRejected('Geometric endpoint semantic verification failed: '+verdict.reason,verdict=verdict)
            staged = {'panorama':panorama,'endpoint':endpoint,'verdict':verdict,'verification':verification,
                'attempt_count':1,'plan':plan,'auxiliary_images':(),'panorama_ref':current['panorama'],'state_ref':current['panorama_state']}
        elif target is not None and not physical_ops:
            staged = restore_endpoint(row,target=target,source=source,current_revision=revision,
                store=session.store,out=out,deps=deps)
        else:
            if 'revisit' in operations and target is None:
                raise IntegrityError('Planner revisit requires an exact saved target')
            execution = panorama
            if target is not None:
                execution = verify_saved_node(target,run_root=deps.run_root)
                row = {**row,'_execution_panorama_ref':target['panorama']}
            staged = state_endpoint(row,panorama=execution,source=source,pose=pose,end_pose=end_pose,
                store=session.store,context=session.context,out=out/'endpoint',deps=deps)
            if target is not None:
                staged.update(origin_id=target['origin_id'],view_identity=target['view_identity'])
        after_revision = revision + int(bool(decision['physical_state_changed']))
        origin = staged.get('origin_id', f'origin{index:04d}' if 'navigate' in operations else current['origin_id'])
        identity = staged.get('view_identity', {'camera_origin':origin} if 'navigate' in operations else current['view_identity'])
        state_ref = staged.get('state_ref')
        if state_ref is None:
            state_ref = save_panorama_state(staged['panorama'],deps.run_root/'states'/origin/f'revision{after_revision:04d}',run_root=deps.run_root)
        panorama_ref = staged.get('panorama_ref') or _save(out/'PANORAMA.png',erp_array(staged['panorama']),deps)
        endpoint = staged['endpoint']
        nid = f'node{index:04d}'
        node = {'node_id':nid,'parent_node_id':staged.get('parent_node_id',current['node_id']),
            'origin_id':origin,'physical_revision':after_revision,'view_identity':identity,
            'pose':{**end_pose,'hfov':90.},'panorama_state':public_ref(state_ref,state=True),
            'panorama':public_ref(panorama_ref,image=True), 'view':public_ref(endpoint,image=True),
            'memory_snapshot':None,'viewport_profile':public_viewport_profile(staged['panorama']['viewport_profile']),
            'seam_status':row['_seam_status']}
        transition = VideoTransition(tid,'revisit' if target is not None and not physical_ops else ('rotate' if pure else next(op for op in operations if op not in ('rotate','revisit'))),
            verify_ref(source,run_root=deps.run_root),verify_ref(endpoint,run_root=deps.run_root),
            row['profile']['num_frames']/row['profile']['fps'],out/'video.mp4')
        presentation_result = present_transition(transition,panorama=staged['panorama'],source_node=current,
            target_node=node,backend=deps.providers.video,profile=row['profile'],run_root=deps.run_root)
        verify_ref(endpoint,run_root=deps.run_root)
        presentation = _presentation_record(presentation_result, row, deps)
        ready = {'schema':'wmmllm.accepted_ready.v1',
            'binding':{'session_id':session.record['session_id'],'source_node_id':current['node_id'],
                'source':source,'panorama_state':current['panorama_state'],'caption':row['current_caption'],
                'seed':row['seed'],'segment':index,'profile':row['profile'],'revision':revision},
            'decision':decision,'decision_ref':decision_ref,'operations':operations,'node':node,
            'staged':{'endpoint':endpoint,'verdict':staged['verdict'].to_record(),
                'verification':staged['verification'],'attempt_count':staged['attempt_count'],
                'restore_semantics':staged.get('restore_semantics'),
                'observer_subjects':staged.get('observer_subjects'),
                'auxiliary_images':[{'image':file_ref(item.path,run_root=deps.run_root,image=True),'label':item.label} for item in staged['auxiliary_images']]},
            'presentation':presentation,'pure':pure}
        ready['sha256'] = canonical_sha256(ready)
        write_once(out/'ACCEPTED_READY.json',ready,run_root=deps.run_root)
        return _observe_and_settle(row,session=session,deps=deps,current=current,decision_ref=decision_ref,
            operations=operations,node=node,staged=staged,presentation=presentation,pure=pure,out=out)
    except Exception as error:
        pending = deps.run_root/'pending'/(tid+'.json')
        if pending.exists() or session.store.snapshot().completed_segments >= index:
            raise
        classification = _classify(error)
        if not classification['hold_eligible']:
            write_once(out/'ERROR.json',safe_error(error),run_root=deps.run_root)
            raise
        # A method hold records failure only. It neither observes the requested
        # action nor installs any rejected proposal or proposed camera.
        failure = {'status':'FAILURE_HOLD','segment':index,'caption':row['current_caption'],
            'source':source,'endpoint':source,'rejected_candidate_committed':False,
            **classification,'attempts':getattr(error,'attempts',[])}
        failure_ref = write_once(out/'FAILURE_HOLD.json',failure,run_root=deps.run_root)
        verdict = getattr(error,'verdict',None)
        semantic = {'status':verdict.status if verdict and verdict.status=='FAIL' else 'NOT_CHECKED',
            'attempt_count':len(getattr(error,'attempts',[])) or (1 if verdict else 0),
            'semantic_accepted':False,'uncertain_accepted_and_labeled':False,'verification':failure_ref}
        hold_transition = VideoTransition(tid, 'failure_hold', verify_ref(source, run_root=deps.run_root),
            verify_ref(source, run_root=deps.run_root), row['profile']['num_frames']/row['profile']['fps'],
            out/'hold'/'video.mp4')
        hold_result = present_transition(hold_transition, panorama=panorama, source_node=current, target_node=current,
            backend=deps.providers.video, profile=row['profile'], run_root=deps.run_root)
        hold_presentation = _presentation_record(hold_result, row, deps)
        summary = {'transition_id':tid,'index':index,'from_node_id':current['node_id'],'to_node_id':current['node_id'],
            'caption':_display_caption(row['current_caption'], deps),'operations':[] if decision is None else [v['operation'] for v in decision['operations']],
            'status':'held','semantic':semantic,'presentation':hold_presentation,
            'decision':decision_ref,'memory_after':None}
        commit = {'segment':index,'caption':row['current_caption'],'source':source,'failure_record':failure_ref}
        return session.settle(_packet(session,row,current,summary,None,commit,held=True))


def _packet(session,row,current,summary,node,commit,*,held):
    return {'schema':'wmmllm.pending.v1','binding':{'method':'wmmllm.core.v1',
        'session_id':session.record['session_id'],'from_node_id':current['node_id'],
        'source':current['view'],'panorama_state':current['panorama_state'],
        'physical_revision':current['physical_revision'],'profile':row['profile'],
        'caption':row['current_caption'],'seed':row['seed'],'decision':summary['decision']},
        'held':held,'transition':summary,'node':node,'memory_commit':commit}


def _observe_and_settle(row, *, session, deps, current, decision_ref, operations, node, staged, presentation, pure, out):
    source=current['view']; endpoint=staged['endpoint']; index=row['segment']; tid=f'turn{index:04d}'; nid=node['node_id']
    obs_deps = ObserverDeps(chat=deps.providers.chat,run_root=deps.run_root)
    observation = observe_state(task_id=row['task_id'],segment=index,caption=row['current_caption'],
        start=source,endpoint=endpoint,known_subjects=(staged['observer_subjects'] if staged.get('observer_subjects') is not None else session.store.known_subjects(index)),deps=obs_deps,
        out_dir=out/'observation',auxiliary_images=staged['auxiliary_images'])
    if observation.status != 'OBSERVED':
        message = 'Geometric observer failed' if pure else ('Saved return observer failed' if staged.get('restore_semantics') else 'accepted endpoint observer failed')
        error = EndpointRejected(message,verdict=staged['verdict'])
        error.stage_error = observation.error
        raise error
    verdict = staged['verdict']
    semantic = {'status':verdict.status,'attempt_count':staged['attempt_count'],
        'semantic_accepted':verdict.status=='PASS','uncertain_accepted_and_labeled':verdict.status=='UNCERTAIN',
        'verification':staged['verification']}
    summary = {'transition_id':tid,'index':index,'from_node_id':current['node_id'],'to_node_id':nid,
        'caption':_display_caption(row['current_caption'], deps),'operations':operations,'status':'accepted','semantic':semantic,
        'presentation':presentation,'decision':decision_ref,'memory_after':None}
    commit = {'segment':index,'caption':row['current_caption'],'start':source,'endpoint':endpoint,
        'observation':observation.observation,'observation_record':observation.record,'fact_source_type':'accepted_state_observation'}
    packet = _packet(session,row,current,summary,node,commit,held=False)
    packet['restore_semantics'] = staged.get('restore_semantics')
    return session.settle(packet)


def _display_caption(caption, deps):
    return redact(caption,secrets=deps.config.secrets())


def check_actual_reachable_pose(row, out, session, panorama, end_pose, *, deps):
    """Check newly requested directions beyond the bootstrap's reachable schedule."""
    boot = read_json(deps.run_root/'bootstrap'/'RESULT.json',run_root=deps.run_root)
    current = session.current_node
    planned = boot['seam_gate']['planned_path']
    expected = bootstrap.reachable_poses(planned)
    actual = {**planned, 'camera_path':[{'segment':0,**_pose(current['pose'])},
                                      {'segment':row['segment'],**end_pose}]}
    covered = all(any(abs(((yaw-y+180)%360)-180)<=10 and abs(pitch-p)<=10 for y,p in expected)
                  for yaw,pitch in bootstrap.reachable_poses(actual))
    if covered:
        write_once(out/'ACTUAL_REACHABLE_GATE.json',{'passed':True,'scope':'within_initial_gate',
            'source_state':current['panorama_state'],'ending_camera':end_pose,'additional_image_calls':0},run_root=deps.run_root)
        return current['seam_status']
    record = {**boot, 'panorama_state':current['panorama_state'], 'panorama':current['panorama']}
    try:
        passed,review,report = bootstrap.gate(row,record,out/'actual_reachable_gate',actual,deps=deps,
                                  deadline_monotonic=time.monotonic()+60)
    except TransportExhausted:
        raise
    except Exception as error:
        raise_if_fatal_provider_error(error)
        passed,review,report=False,{'error':safe_provider_error(error)},{}
    write_once(out/'ACTUAL_REACHABLE_GATE.json',{'passed':passed,'review':review,'report':report,
        'source_state':current['panorama_state'],'ending_camera':end_pose},run_root=deps.run_root)
    return current['seam_status'] if passed else 'seam_unresolved'


def failure_hold_history(store, *, deps):
    result=[]
    for commit in store.snapshot().commits:
        if commit.get('facts') != 0 or not commit.get('observation'):
            continue
        receipt=read_json(commit['observation'],run_root=deps.run_root)
        if receipt.get('status') == 'FAILURE_HOLD':
            result.append({'segment':commit['segment'],'caption':commit['caption'],'status':'FAILURE_HOLD',
                'action_completed':False,'accepted_state_unchanged':True,'receipt':commit['observation']})
    return result

