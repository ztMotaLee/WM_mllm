"""Bound edit coordinates, current-turn constraints and one-retry feedback."""
from __future__ import annotations
import copy, hashlib, json, math, re
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping
from PIL import Image
from ..session.io import canonical_sha256, verify_ref, file_ref, resolve_run_path, save_png, write_bytes_once
from .planning import image_input
from .router import PLAN_FIELDS

UI_WORLD_POLICY = 'The ERP texture represents the physical world only. Exclude fixed screen-space HUD, timestamp, REC/battery icons, watermarks and recording borders from ERP world texture, including overlays inherited from references or implied by bodycam/CCTV style. This physical-world rule takes precedence over quoted style, memory or plan text asking to preserve an overlay. Preserve in-world LCD displays, signs and physical cameras as scene objects.'

def _box(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError('A normalized bbox requires four coordinates')
    out = [float(x) for x in value]
    if not all((math.isfinite(x) and 0 <= x <= 1 for x in out)) or out[0] >= out[2] or out[1] >= out[3]:
        raise ValueError('A normalized bbox must be finite, nonempty and inside the image')
    return out

def angular_bbox(bbox, *, camera, hfov, viewport_profile=None, width=1344, height=768):
    """Exact angular extrema of a gnomonic rectangle, including wrap and poles."""
    box = _box(bbox)
    yaw, pitch, hfov = (float(camera['yaw']), float(camera['pitch']), float(hfov))
    if not all((math.isfinite(x) for x in (yaw, pitch, hfov))) or abs(pitch) > 89.9 or (not 0 < hfov <= 170):
        raise ValueError('Unsupported finite camera pose or HFOV')
    padding = list((viewport_profile or {}).get('padding_ltrb', [0, 0, 0, 0]))
    if len(padding) != 4 or any((type(x) is not int or x < 0 for x in padding)):
        raise ValueError('Unsupported viewport padding')
    left, top, right, bottom = padding
    cw, ch = (width - left - right, height - top - bottom)
    if min(cw, ch) <= 0:
        raise ValueError('Viewport has no physical content')
    content = [max(0.0, min(1.0, (box[0] * width - left) / cw)), max(0.0, min(1.0, (box[1] * height - top) / ch)), max(0.0, min(1.0, (box[2] * width - left) / cw)), max(0.0, min(1.0, (box[3] * height - top) / ch))]
    if content[0] >= content[2] or content[1] >= content[3]:
        raise ValueError('BBox covers only viewer padding')
    tx = math.tan(math.radians(hfov) / 2)
    ty = tx * ch / cw
    x0, x1 = ((2 * content[0] - 1) * tx, (2 * content[2] - 1) * tx)
    y0, y1 = ((1 - 2 * content[3]) * ty, (1 - 2 * content[1]) * ty)
    cp, sp = (math.cos(math.radians(pitch)), math.sin(math.radians(pitch)))
    reference_lon = math.degrees(math.atan2((x0 + x1) / 2, -(y0 + y1) / 2 * sp + cp))

    def angles(x, y):
        py, pz = (y * cp + sp, -y * sp + cp)
        local_lon = math.degrees(math.atan2(x, pz))
        lon = yaw + reference_lon + (local_lon - reference_lon + 180) % 360 - 180
        lat = math.degrees(math.atan2(py, math.hypot(x, pz)))
        return (lon, lat)
    corners = [(x0, y1), (x1, y1), (x1, y0), (x0, y0)]
    points = list(corners)
    if x0 <= 0 <= x1:
        points.extend([(0, y0), (0, y1)])
    if abs(sp) > 1e-15:
        for x in (x0, x1):
            critical_y = cp * (x * x + 1) / sp
            if y0 <= critical_y <= y1:
                points.append((x, critical_y))
    values = [angles(x, y) for x, y in points]
    poles = []
    if abs(sp) > 1e-15 and x0 <= 0 <= x1 and (y0 <= cp / sp <= y1):
        poles.append('zenith' if sp > 0 else 'nadir')
    lo, hi = (min((p[0] for p in values)), max((p[0] for p in values)))
    low_lat, high_lat = (min((p[1] for p in values)), max((p[1] for p in values)))
    if poles:
        lo, hi = (yaw - 180.0, yaw + 180.0)
        if 'zenith' in poles:
            high_lat = 90.0
        if 'nadir' in poles:
            low_lat = -90.0
    u0, u1 = ((0.5 + lo / 360) % 1, (0.5 + hi / 360) % 1)
    wrap = not poles and math.floor((lo + 180) / 360) != math.floor((hi + 180) / 360)
    return {'image_bbox_normalized': box, 'physical_content_bbox_normalized': content, 'camera_degrees': {'yaw': yaw, 'pitch': pitch, 'hfov': hfov}, 'angular_bbox_degrees_unwrapped': [lo, low_lat, hi, high_lat], 'erp_bbox_uv': [0.0 if poles else u0, 0.5 - high_lat / 180, 1.0 if poles else u1, 0.5 - low_lat / 180], 'crosses_erp_horizontal_seam': wrap, 'includes_poles': poles, 'corner_rays_degrees': [list(angles(x, y)) for x, y in corners], 'viewport_padding_ltrb': padding, 'coordinate_semantics': 'image-edge rectangle; angular axes of accepted ERP; yaw right, pitch up'}

def coordinate_record(*, editing_bbox, final_entity_bboxes, constraints, viewport_profile, editing_view_ref, source_view_ref):
    if not isinstance(final_entity_bboxes, list):
        raise ValueError('final_view_entity_bboxes must be an array')
    entities = []
    names = set()
    for entry in final_entity_bboxes:
        if set(entry) != {'entity', 'bbox'} or not isinstance(entry['entity'], str) or (not entry['entity'].strip()):
            raise ValueError('Final entity bbox requires a literal entity and bbox')
        if entry['entity'] in names:
            raise ValueError('Duplicate final entity bbox')
        names.add(entry['entity'])
        entities.append({'entity': entry['entity'], **angular_bbox(entry['bbox'], camera=constraints['ending_camera'], hfov=90.0, viewport_profile=viewport_profile)})
    value = {'schema': 'aiwm.w34.erp_edit_coordinates.round2', 'endpoint_constraints_sha256': constraints['sha256'], 'official_current_motion_contract': {}, 'editing_view': dict(editing_view_ref), 'source_view': dict(source_view_ref), 'editing_support': angular_bbox(editing_bbox, camera=constraints['editing_camera'], hfov=constraints['editing_hfov'], viewport_profile=viewport_profile), 'final_benchmark_view': angular_bbox([0, 0, 1, 1], camera=constraints['ending_camera'], hfov=90.0, viewport_profile=viewport_profile), 'planned_final_entity_footprints': entities, 'planned_footprints_are_observations': False, 'missing_final_entity_bbox': 'not supplied; do not infer measured subject coordinates'}
    value['sha256'] = canonical_sha256(value)
    return value
_COMPOSITION = re.compile('\\b(?:wide[- ]angle|ultra[- ]wide|fisheye|perspective|vanishing point|composition|framing|close[- ]up|full[- ]body|front[- ]facing|rear[- ]facing|first[- ]person|third[- ]person|camera|eye[- ]level|low[- ]angle|high[- ]angle|centered|centred)\\b', re.IGNORECASE)

def appearance_style(root_style: str) -> str:
    """Keep literal appearance clauses; omit viewport composition clauses.

    The raw style remains in root provenance. This image prompt receives the
    appearance clauses only; full-sphere projection has its own explicit rule.
    """
    if not isinstance(root_style, str):
        raise TypeError('root_style must be a string')
    clauses = [p.strip(' .') for p in re.split('[,;\\n]+', root_style) if p.strip(' .')]
    retained = []
    for clause in clauses:
        check = clause
        if re.search('\\b(?:sunlight|sun|light|lighting|illumination)\\b', clause, re.IGNORECASE):
            check = re.sub('\\b(?:low|high)[- ]angle\\b', '', check, flags=re.IGNORECASE)
        if not _COMPOSITION.search(check) and (not re.search('\\b(?:overlay|HUD|timestamp|REC|watermark|recording border|battery icon)\\b', check, re.IGNORECASE)):
            retained.append(clause)
    return ', '.join(retained)

def build_erp_endpoint_prompt(*, plan_instruction: str, constraints: Mapping[str, Any], source_pose: Mapping[str, Any], ending_pose: Mapping[str, Any], root_style: str='', memory_text: str='', viewport_profile: Mapping[str, Any] | None=None) -> str:
    """Bind full-sphere generation to the actual 90-degree benchmark viewport.

    Caller appends the unchanged hash-checked operation constraints and uses
    the unchanged semantic verifier and exactly one semantic retry.
    """
    source, ending = ({k: float(pose[k]) for k in ('yaw', 'pitch')} for pose in (source_pose, ending_pose))
    center_u = (0.5 + ending['yaw'] / 360.0) % 1.0
    center_v = 0.5 - ending['pitch'] / 180.0
    caption = constraints['official_row_binding']['current_caption']
    appearance = appearance_style(root_style)
    parts = ['Edit Image1 into one complete 360-degree by 180-degree equirectangular panorama of the same world at the requested final state.', 'Image1 is the accepted full-sphere ERP world state. Image2 is its actual current perspective viewer crop, supplied for subject identity and benchmark framing only. Output the edited full-sphere Image1.', 'Projection: output a single 2:1 latitude-longitude image. Longitude spans 360 degrees left to right; the middle column is yaw 0, the horizontal edges meet behind the camera at yaw -180/+180. The top is the zenith and the bottom is the nadir. Retain connected front, side, rear, sky/ceiling and ground/floor coverage, continuous horizontal wrapping and spherical polar distortion.', f"Current viewer: yaw {source['yaw']:g} degrees, pitch {source['pitch']:g} degrees, horizontal field of view 90 degrees.", f"Final benchmark viewer: yaw {ending['yaw']:g} degrees, pitch {ending['pitch']:g} degrees, horizontal field of view 90 degrees. Its central ERP ray is at normalized (u={center_u:.6f}, v={center_v:.6f}); u=0.5+yaw/360 and v=0.5-pitch/180.", 'All requested centering, subject size, visibility and maintained framing apply to the final benchmark perspective viewer crop, compared with Image2. Place the subject at the correct angular location and size for that crop. The ERP canvas center and the wider EDITING_VIEW are separate coordinate systems.', 'Apply the ordered physical operations once. Represent camera translation and any subject movement with coherent landmark parallax and occlusion, while preserving stationary scene structure and unchanged entity properties.', 'Current action (verbatim): ' + caption, 'Planned final state (verbatim): ' + plan_instruction]
    if viewport_profile and viewport_profile.get('padding_ltrb'):
        parts.append('Image2 may contain benchmark letterbox bars; they are viewer formatting and must not become world texture in Image1.')
    if appearance:
        parts.append('Preserve source appearance: ' + appearance + '.')
    if memory_text:
        parts.append('Accepted state memory: ' + str(memory_text)[:2200])
    parts.append(UI_WORLD_POLICY)
    return '\n'.join(parts)

SCHEMA = 'wmmllm.identity_reference.v1'

def extract_reference(physical_source_ref, planned_support_polygons, out_dir, *, padding_pixels=4, min_size_pixels=12, max_crop_fraction=0.5, upscale_min_edge=None, run_root=None):
    """Return ``(reference FileRef or None, provenance)`` without any model call.

    Polygon points must be normalized [x,y] in the actual source editing view.
    A missing/invalid/tiny/full-scene first polygon returns UNSUPPORTED so the
    caller can choose a different policy. A changed source hash is an error.
    Default output is the exact native RGB crop. Optional Lanczos enlargement
    changes sampling pixels; both native and enlarged files are lossless PNG.
    """
    source = verify_ref(physical_source_ref, run_root=run_root)
    with Image.open(source) as loaded:
        original_mode = loaded.mode
        image = loaded.convert('RGB')
    width, height = image.size
    record = {'schema': SCHEMA, 'status': 'UNSUPPORTED', 'reason': None, 'source': file_ref(source, run_root=run_root, image=True), 'source_size': [width, height], 'source_original_mode': original_mode, 'source_RGB_pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest(), 'coordinate_convention': 'Normalized EDITING_VIEW points scale by(width-1,height-1); crop bbox is integer half-open [left,top,right,bottom].', 'selected_polygon_index': 0, 'selected_polygon_role': 'Assumed old-object support; no automatic semantic identification.', 'padding_pixels': padding_pixels, 'min_size_pixels': min_size_pixels, 'max_crop_fraction': max_crop_fraction, 'upscale_min_edge': upscale_min_edge, 'reference_only': True, 'other_pixels_may_be_included': True, 'other_pixels_policy': 'Any clamp, nearby prop or scene pixels inside the rectangular source crop are unchanged reference context only; this crop never authorizes their edit.', 'generation_calls': 0, 'GPU_calls': 0, 'source_modified': False, 'support_mask_modified': False, 'accepted_state_modified': False}
    out_dir = resolve_run_path(run_root, Path(out_dir).relative_to(Path(run_root)).as_posix(), must_exist=False)

    def finish(reason, reference=None):
        record['reason'] = reason
        data = (json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
        write_bytes_once(out_dir / 'IDENTITY_REFERENCE.json', data, run_root=run_root)
        return (reference, record)
    if not isinstance(planned_support_polygons, (list, tuple)) or not planned_support_polygons:
        return finish('No first old-object support polygon')
    polygon = planned_support_polygons[0]
    if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
        return finish('First support polygon requires at least3 points')
    points = []
    for point in polygon:
        if not isinstance(point, (list, tuple)) or len(point) != 2 or any((isinstance(v, bool) or not isinstance(v, (int, float)) or (not math.isfinite(v)) or (not 0 <= v <= 1) for v in point)):
            return finish('First support polygon has invalid or out-of-bounds normalized coordinates')
        points.append([float(point[0]), float(point[1])])
    record['planned_polygon_count'] = len(planned_support_polygons)
    record['selected_polygon_normalized'] = points
    area = abs(sum((a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1])))) / 2
    if not area:
        return finish('First support polygon has zero area')
    pixels = [[x * (width - 1), y * (height - 1)] for x, y in points]
    bbox = [max(0, math.floor(min((p[0] for p in pixels))) - padding_pixels), max(0, math.floor(min((p[1] for p in pixels))) - padding_pixels), min(width, math.ceil(max((p[0] for p in pixels))) + 1 + padding_pixels), min(height, math.ceil(max((p[1] for p in pixels))) + 1 + padding_pixels)]
    left, top, right, bottom = bbox
    crop_width, crop_height = (right - left, bottom - top)
    fraction = crop_width * crop_height / (width * height)
    record.update(selected_polygon_pixels=pixels, selected_polygon_normalized_area=area, crop_bbox=bbox, native_crop_size=[crop_width, crop_height], crop_fraction=fraction)
    if min(crop_width, crop_height) < min_size_pixels:
        return finish('First old-object reference crop is too small')
    if fraction >= max_crop_fraction:
        return finish('First support is full-scene or too broad for an object identity reference')
    native = image.crop(tuple(bbox))
    native_ref = _save_png(out_dir / 'OBJECT_REFERENCE_NATIVE.png', native, run_root=run_root)
    reference = native_ref
    record.update(native_crop=native_ref, native_crop_RGB_pixel_sha256=hashlib.sha256(native.tobytes()).hexdigest(), native_crop_source_RGB_exact=True, native_crop_encoding='lossless PNG', resampling=None)
    if upscale_min_edge is not None and min(native.size) < upscale_min_edge:
        scale = upscale_min_edge / min(native.size)
        size = [math.ceil(v * scale) for v in native.size]
        if max(size) > 4096:
            return finish('Requested enlargement is too large; retained exact native crop')
        enlarged = native.resize(tuple(size), Image.Resampling.LANCZOS)
        reference = _save_png(out_dir / 'OBJECT_REFERENCE_LANCZOS.png', enlarged, run_root=run_root)
        record['resampling'] = {'method': 'PIL.LANCZOS', 'input': native_ref, 'input_size': list(native.size), 'output_size': size, 'encoding': 'lossless PNG', 'pixels_are_interpolated': True, 'original_native_crop_retained': True}
    record.update(status='READY', reference=reference, reference_size=record['resampling']['output_size'] if record['resampling'] else list(native.size))
    return finish('Actual source object crop; conditioning reference only', reference)
VERSION = 'navigation_contract_feedback_v5_20261013'
SPATIAL_START = 'APP_SPATIAL_CRITERION:\n'
FORWARD_SENTENCE = "The camera has WALKED FORWARD from the remembered viewpoint, so subjects legitimately shift screen position or pass behind the viewer; judge only whether each subject remains at its listed LOCAL physical anchor (or is plausibly out of view behind the camera), and never report 'relocated' merely because a subject's screen position (left/right/center/height) differs from the remembered one."

def transition_binding(constraints: Mapping[str, Any]) -> dict:
    official = constraints['official_row_binding']
    return {'row_key': official['row_key'], 'segment': official['segment'], 'current_caption': official['current_caption'], 'source_sha': constraints['planner_input_binding']['source_sha256'], 'pano_sha': constraints['planner_input_binding']['panorama_sha256']}

def bind_operation_planner_inputs(payload: Mapping[str, Any], row: Mapping[str, Any]) -> tuple[dict, dict]:
    """Bind the operation planner payload to this actual current turn."""
    binding = bind_official_row(row)
    result = copy.deepcopy(dict(payload))
    result['official_motion_context'] = copy.deepcopy(binding['official_motion_context'])
    result['official_context'] = copy.deepcopy(binding['official_context'])
    result['official_current_turn_binding'] = {'row_key': binding['row_key'], 'segment': binding['segment'], 'sha256': binding['sha256']}
    return (result, binding)

def spatial_verifier_with_actual_motion(request: Any, binding: Mapping[str, Any]) -> Any:
    """Replace only the erroneous forward-only spatial-memory sentence."""
    bound = binding
    if request.payload.get('current_caption') != bound['current_caption']:
        raise ValueError('Verifier caption differs from bound current turn')
    if request.payload.get('actual_h3_action') != bound['current_caption']:
        raise ValueError('Verifier actual single-shot action differs from bound current turn')
    if SPATIAL_START not in request.system:
        raise ValueError('Combined request lacks spatial criterion')
    before, tail = request.system.split(SPATIAL_START, 1)
    spatial, after = tail.split('\n\nAPP_ENTITY_CRITERION:', 1)
    if spatial.count(FORWARD_SENTENCE) != 1:
        raise ValueError('Expected one exact source forward-only spatial sentence')
    context = {'current_caption': bound['current_caption'], 'official_motion_context': bound['official_motion_context'], 'official_context': bound['official_context']}
    replacement = 'The actual current motion is the following bound quoted context: ' + json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '. Judge each subject against its listed LOCAL physical anchor and any physical change explicitly requested in this turn. A changed screen position alone does not prove relocation. Allow visibility or screen-position changes only when they follow from this actual motion; honor explicit subject_screen_fixed and equal camera/subject translation constraints when supplied.'
    from .navigation import actual_motion_text
    replacement += '\n' + actual_motion_text(bound['current_caption'])
    updated = spatial.replace(FORWARD_SENTENCE, replacement, 1)
    payload = copy.deepcopy(dict(request.payload))
    payload['spatial_motion_context_binding'] = {'version': VERSION, 'official_binding_sha256': bound['sha256'], 'row_key': bound['row_key'], 'segment': bound['segment'], 'original_spatial_sha256': hashlib.sha256(spatial.encode()).hexdigest(), 'updated_spatial_sha256': hashlib.sha256(updated.encode()).hexdigest(), 'verdict_gate_changed': False, 'motion_spec_version': 'navigation_policy_round2'}
    return replace(request, system=before + SPATIAL_START + updated + '\n\nAPP_ENTITY_CRITERION:' + after, payload=payload)
def _require(condition, message):
    if not condition:
        raise ValueError(message)

def _text(value, name):
    _require(isinstance(value, str) and bool(value.strip()), name + ' is empty')
    _require(not any((ord(c) < 32 or ord(c) == 127 or c in '\x85\u2028\u2029' for c in value)), name + ' must be one literal paragraph')
    return value

def bind_official_row(row):
    """Bind the current interactive turn. Sessions supply no benchmark turn, motion or context,
    so those bound fields stay empty."""
    caption = _text(row['current_caption'], 'current_caption')
    binding = {'row_key': row['key'], 'task_id': row['task_id'], 'segment': row['segment'], 'seed': row.get('seed', 0), 'profile': copy.deepcopy(row.get('profile', {})), 'current_caption': caption, 'caption_sha256': hashlib.sha256(caption.encode()).hexdigest(), 'official_turn_sha256': canonical_sha256({}), 'source_refs': [], 'official_context': {}, 'official_motion_context': {}}
    binding['sha256'] = canonical_sha256(binding)
    return binding

def build_endpoint_constraints(row, decision, *, source=None, panorama=None):
    primitive = {key: copy.deepcopy(decision[key]) for key in PLAN_FIELDS}
    for ref, field in ((source, 'source_sha256'), (panorama, 'panorama_sha256')):
        if ref is not None:
            _require(decision.get(field) == ref['sha256'], 'Operation planner input ' + field + ' differs')
    operations = primitive['operations']
    full = any((op['operation'] in ('navigate', 'global_change') or op['scope'] == 'global' for op in operations))
    value = {'schema': 'wmmllm.endpoint_constraints.v1', 'routing_authority': 'configured_operation_planner', 'official_row_binding': bind_official_row(row), 'operation_decision': primitive, 'operation_decision_sha256': canonical_sha256(primitive), 'planner_input_binding': {key: decision.get(key) for key in ('source_sha256', 'panorama_sha256')}, 'endpoint_instruction': primitive['endpoint_instruction'], 'ordered_physical_operations': [op for op in operations if op['operation'] not in ('rotate', 'revisit')], 'edit_domain': 'whole_panorama' if full else 'scoped_editing_view', 'editing_camera': primitive['editing_camera'], 'editing_hfov': primitive['editing_hfov'], 'ending_camera': primitive['ending_camera'], 'outside_support_preservation_required': not full, 'video_writes_state': False}
    value['sha256'] = canonical_sha256(value)
    return value

def append_endpoint_constraints(prompt, constraints):
    _require(isinstance(prompt, str) and bool(prompt.strip()), 'image prompt is empty')
    ending = ' Operation planner final-state constraint: ' + constraints['endpoint_instruction']
    targets = list(dict.fromkeys((target for op in constraints['ordered_physical_operations'] for target in op['targets'])))
    if targets:
        ending += ' Affected entities: ' + ', '.join(targets) + '.'
    if constraints['edit_domain'] == 'whole_panorama':
        ending += ' Apply the planned change within the complete ERP panorama; preserve connected layout and unaffected entities.'
    else:
        ending += ' Keep the supplied editing-view camera fixed. Preserve every pixel outside the approved changed-entity support; render the requested viewer pose after compositing.'
    if constraints['edit_domain'] == 'whole_panorama':
        ending += ' ' + UI_WORLD_POLICY
    return prompt + ending

def editor_coordinate_contract(prompt, images, record, *, run_root=None):
    """Append Image3 (the EDITING_VIEW) and the coordinate record to the accepted ERP and source view."""
    text = '\nImage3 is the actual starting EDITING_VIEW at its recorded camera and HFOV. Image2 and Image3 are subject identity and camera references; output only the complete Image1 ERP. The editing support bbox belongs to Image3. The planned final entity bboxes belong to the final 90-degree benchmark viewer and are converted below into ERP longitude/latitude and wrap-aware UV. Keep each coordinate system separate. Apply final framing and size to the final benchmark crop. Do not paste reference panels or bake an additional pan/tilt into ERP texture. Coordinate contract (JSON): ' + json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return (prompt + text, tuple(images) + (image_input(record['editing_view'], '', run_root=run_root),))

def _save_png(path, image, *, run_root):
    # These are crops of authoritative state pixels. Color interpretation was
    # settled at ingestion; a crop must not inherit metadata or transform again.
    rgb = image.convert('RGB')
    clean = Image.frombytes('RGB', rgb.size, rgb.tobytes())
    return save_png(path, clean, run_root=run_root)

def build_feedback(*, current_attempt, attempts, stage, constraints, actual_editor_source, navigation, run_root=None):
    if current_attempt != 2 or len(attempts) != 1 or attempts[0].get('attempt') != 1:
        raise ValueError('Semantic retry is restricted to attempt1 -> attempt2')
    previous = attempts[0]
    if previous['semantic']['status'] != 'FAIL':
        raise ValueError('Only a rejected attempt may produce retry feedback')
    for ref in (previous['endpoint'], previous['image'], actual_editor_source, previous['verification_ref'], previous['verifier_source']):
        verify_ref(ref, run_root=run_root)
    full = json.loads(verify_ref(previous['verification_ref'], run_root=run_root).read_text())
    if any((full.get(k) != previous['semantic'].get(k) for k in ('status', 'verification', 'failed', 'uncertain', 'reason'))):
        raise ValueError('Actual verifier result differs from rejected prior attempt')
    if navigation and (set(full.get('criteria', {})) != {'projection', 'movement', 'spatial', 'entity'} or not full.get('retry_instruction')):
        raise ValueError('Navigation retry must retain every criterion and retry_instruction')
    bound = transition_binding(constraints)
    if previous['verifier_source']['sha256'] != bound['source_sha']:
        raise ValueError('Prior verifier accepted source differs')
    inputs = previous['editor_inputs']
    if not inputs or inputs[0]['sha256'] != actual_editor_source['sha256']:
        raise ValueError('Editor retry sources belong to different accepted physical views')
    for ref in inputs:
        verify_ref(ref, run_root=run_root)
    for key in ('editor_receipt', 'verifier_receipt'):
        receipt = previous[key]
        if receipt.get('status') != 'complete' or not receipt.get('request_sha256') or (not receipt.get('response_sha256')):
            raise ValueError('Feedback requires complete actual provider receipt hashes')
    return {'schema': 'wmmllm.complete_endpoint_retry_feedback.v1', 'previous_attempt': 1, 'current_attempt': 2, 'semantic_retry_limit': 1, 'navigation': bool(navigation), 'complete_verifier_failure': full, 'verification_ref': dict(previous['verification_ref']), 'rejected_endpoint': dict(previous['endpoint']), 'rejected_raw_image': dict(previous['image']), 'editor_receipt': copy.deepcopy(previous['editor_receipt']), 'verifier_receipt': copy.deepcopy(previous['verifier_receipt']), 'editor_inputs': copy.deepcopy(inputs), 'verifier_source': dict(previous['verifier_source']), 'transition_binding': bound, 'actual_editor_source': dict(actual_editor_source), 'failure_sha256': canonical_sha256(full), 'accepted_state_authority_unchanged': True}

def planner_feedback(request, value, *, run_root=None):
    """Attach the ``build_feedback`` retry feedback to the planner request."""
    payload = copy.deepcopy(dict(request.payload))
    payload['previous_endpoint_failure'] = value['complete_verifier_failure']
    payload['previous_failed_proposal_provenance'] = value
    return replace(request, payload=payload, system=request.system + ' The rejected endpoint and raw proposal are failed attempt1 outputs. Repair the complete supplied verifier failure in attempt2. The original accepted source remains state authority.', images=request.images + (image_input(value['rejected_endpoint'], 'actual rejected attempt1 benchmark endpoint', run_root=run_root), image_input(value['rejected_raw_image'], 'actual rejected attempt1 raw image proposal', run_root=run_root)))

def editor_feedback(prompt, images, value, *, run_root=None):
    """Append the ``build_feedback`` retry feedback to the editor request."""
    number = len(images) + 1
    text = f' Image{number} is the actual rejected attempt1 benchmark endpoint; Image{number + 1} is its raw proposal. Repair these failed proposal defects while retaining Image1 as the accepted source. Complete verifier failure (verbatim JSON): ' + json.dumps(value['complete_verifier_failure'], ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    refs = tuple((image_input(ref, '', run_root=run_root) for ref in (value['rejected_endpoint'], value['rejected_raw_image'])))
    return (prompt + text, tuple(images) + refs)
