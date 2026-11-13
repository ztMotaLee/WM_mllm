"""Two-image root construction and bounded reachable-view seam repair."""
from __future__ import annotations
import json, math, time
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from ..state import panorama as geo, projection
from ..state.registration import perspective_outpaint_input, register_native_context, register_erp_to_source, _source_homography
from ..session.io import verify_ref, file_ref, write_once, write_atomic, resolve_run_path, save_png
from ..providers.types import ChatRequest, ImageEditRequest
from ..providers.retry import raise_if_fatal_provider_error, safe_provider_error
from ..prompts import read_text, read_json_data
from .planning import image_input
from .router import json_format
from .contracts import canonical_sha

def _source_metadata(row):
    return row.get('source_profile') or row.get('source_metadata')

def corrected_single_erp_state(base, source, source_metadata, *, width=4096):
    base = geo.rgb_array(base)
    profile = geo.source_profile(source, source_metadata=source_metadata)
    l, t, r, b = profile['content_box']
    official = geo.rgb_array(source)[t:b, l:r]
    lifted, alpha, projection = geo._composite_content_edit(base, official, return_metadata=True)
    known = alpha > 0
    delta = (lifted.astype(np.float32) - base.astype(np.float32)) * known[..., None]
    mass = cv2.GaussianBlur(known.astype(np.float32), (0, 0), 3.0)
    average = cv2.GaussianBlur(delta, (0, 0), 3.0) / np.maximum(0.001, mass[..., None])
    distance, indices = distance_transform_edt(~known, return_indices=True)
    iy, ix = indices
    offset = average[iy, ix]
    small = np.mean(np.abs(offset), axis=-1) < 32
    offset = np.clip(offset, -16, 16) * small[..., None]
    weight = np.clip(1 - distance / 32, 0, 1) ** 2
    weight[known] = 0
    corrected = np.clip(np.rint(base.astype(np.float32) + offset * weight[..., None]), 0, 255).astype(np.uint8)
    if not np.array_equal(corrected[known], base[known]):
        raise RuntimeError('unknown-only collar edited known ERP')
    state = geo.create_panorama_state(corrected, source, width=width, height=width // 2, source_metadata=source_metadata)
    state['render_base'] = np.asarray(Image.fromarray(corrected).resize((width, width // 2), Image.Resampling.LANCZOS))
    return (state, {'method': 'single_registered_ERP_plus_official_native_texture', 'unknown_native_patches': 0, 'official_native_patches': 1, 'outer_native_context_join_exists': False, 'small_colour_residual_only_outside_official_frustum': True, 'ERP_collar_width_pixels': 32, 'maximum_RGB_collar_correction': 16, 'known_ERP_changed_pixels': 0, 'projection': projection})

def bake_native_collar(row, state, record):
    native = geo.rgb_array(record['_native_context_path'])
    h, w = native.shape[:2]
    hfov = record['native_registration']['context_hfov_degrees']
    cam = geo.crop_camera(0, 0, hfov, w, h)
    x, y = geo._camera_grid(hfov, w, h)
    x, y = np.broadcast_arrays(x, y)
    directions = np.stack([x, y, np.ones_like(x)], axis=-1)
    p = state['patches'][0]
    _, _, official = geo._project_patch(directions, p)
    dx = np.minimum(np.arange(w) + 0.5, w - np.arange(w) - 0.5)
    dy = np.minimum(np.arange(h) + 0.5, h - np.arange(h) - 0.5)
    distance = np.minimum(dx[None, :], dy[:, None])
    support = np.clip(distance / 32, 0, 1) * (official == 0)
    base, alpha, meta = geo._composite_content_edit(state['render_base'], native, hfov=hfov, support=support, return_metadata=True)
    preview, _, _ = geo._composite_content_edit(state['erp'], native, hfov=hfov, support=support, return_metadata=True)
    accepted = dict(state)
    accepted['render_base'] = base
    accepted['erp'] = preview
    assert geo.image_error(row['initial_image'], geo.render_view(accepted))['pixel_identical']
    return (accepted, {'native_context_baked_into_ERP_only': True, 'native_patch_count': 1, 'official_world_rays_editable': False, 'outer_feather_source_pixels': 32, 'support_metadata': meta})

def correct_colour_collar(row, state, collar=64):
    p = geo.source_profile(row['initial_image'], source_metadata=_source_metadata(row))
    l, t, r, b = p['content_box']
    source = geo.rgb_array(row['initial_image'])[t:b, l:r]
    ch, cw = source.shape[:2]
    base = geo.rgb_array(state['render_base'])
    ph, pw = base.shape[:2]
    margin = 8
    eh, ew = (ch + 2 * margin, cw + 2 * margin)
    hfov = math.degrees(2 * math.atan(ew / cw))
    extended = geo._render_content_view(base, 0, 0, hfov, ew, eh).astype(np.float32)
    edges = [source[:, 0].astype(np.float32) - extended[margin:margin + ch, margin - 1], source[:, -1].astype(np.float32) - extended[margin:margin + ch, margin + cw], source[0].astype(np.float32) - extended[margin - 1, margin:margin + cw], source[-1].astype(np.float32) - extended[margin + ch, margin:margin + cw]]
    edges = [np.clip(cv2.GaussianBlur(e[:, None], (1, 9), 2)[:, 0], -96, 96) for e in edges]
    corrected = base.copy()
    alpha = np.zeros((ph, pw), np.float32)
    lon = ((np.arange(pw) + 0.5) / pw - 0.5) * 2 * math.pi
    for first in range(0, ph, 128):
        last = min(first + 128, ph)
        lat = (0.5 - (np.arange(first, last) + 0.5) / ph) * math.pi
        dx = np.cos(lat[:, None]) * np.sin(lon[None, :])
        dy = np.broadcast_to(np.sin(lat[:, None]), dx.shape)
        dz = np.cos(lat[:, None]) * np.cos(lon[None, :])
        px = (dx / np.maximum(dz, 1e-05) + 1) * cw / 2 - 0.5
        py = (ch - 1) / 2 - dy / np.maximum(dz, 1e-05) * cw / 2
        ex = np.maximum.reduce([-px - 0.5, px - (cw - 0.5), np.zeros_like(px)])
        ey = np.maximum.reduce([-py - 0.5, py - (ch - 0.5), np.zeros_like(py)])
        dist = np.hypot(ex, ey)
        weight = (0.5 + 0.5 * np.cos(np.minimum(dist / collar, 1) * math.pi)) * (dist > 0) * (dz > 0)
        lr = np.where((px < 0)[..., None], geo._sample_xy(edges[0][:, None], np.zeros_like(px), py), geo._sample_xy(edges[1][:, None], np.zeros_like(px), py))
        tb = np.where((py < 0)[..., None], geo._sample_xy(edges[2][None], px, np.zeros_like(py)), geo._sample_xy(edges[3][None], px, np.zeros_like(py)))
        offset = (lr * ex[..., None] + tb * ey[..., None]) / np.maximum(ex + ey, 1e-06)[..., None]
        corrected[first:last] = np.clip(np.rint(base[first:last].astype(np.float32) + offset * weight[..., None]), 0, 255).astype(np.uint8)
        alpha[first:last] = weight
    out = dict(state)
    out['render_base'] = corrected
    preview = geo.rgb_array(state['erp']).copy()
    unknown = alpha > 0
    preview[unknown] = corrected[unknown]
    out['erp'] = preview
    assert not np.any(base[alpha == 0] != corrected[alpha == 0])
    assert geo.image_error(row['initial_image'], geo.render_view(out))['pixel_identical']
    return (out, {'method': 'C0_edge_residual_with_source_plane_cosine_taper', 'outside_only': True, 'width_source_pixels': collar, 'clip_RGB': 96, 'tangential_smoothing_sigma': 2, 'outside_support_changed': 0, 'native_patch_count': 1, 'edge_mean_absolute_residuals': [float(np.mean(np.abs(e))) for e in edges]})

def prior_erp_mask(state, keep_yaw):
    """Protect the actual prior-view interior, including ERP interpolation neighbors."""
    base = geo.rgb_array(state['render_base'])
    ph, pw = base.shape[:2]
    l, t, r, b = geo.profile_content_box(state['viewport_profile'], 1344, 768)
    cw, ch = (r - l, b - t)
    proxy = {'camera': geo.crop_camera(keep_yaw, 0, 90, cw, ch), 'alpha': np.ones((ch, cw), np.float32)}
    protected = np.zeros((ph, pw), np.uint8)
    lon = ((np.arange(pw) + 0.5) / pw - 0.5) * 2 * math.pi
    for first in range(0, ph, 128):
        last = min(first + 128, ph)
        lat = (0.5 - (np.arange(first, last) + 0.5) / ph) * math.pi
        wx = np.cos(lat[:, None]) * np.sin(lon[None, :])
        wy = np.broadcast_to(np.sin(lat[:, None]), wx.shape)
        wz = np.cos(lat[:, None]) * np.cos(lon[None, :])
        directions = np.stack((wx, wy, wz), axis=-1)
        px, py, alpha = geo._project_patch(directions, proxy)
        border = np.minimum.reduce([px + 0.5, cw - px - 0.5, py + 0.5, ch - py - 0.5])
        protected[first:last] = ((alpha > 0) & (border >= 96)).astype(np.uint8)
    return cv2.dilate(protected, np.ones((5, 5), np.uint8)) > 0

def preserve_prior_overlap(before, after, keep_yaw):
    if keep_yaw is None:
        return (after, {'active': False})
    protected = prior_erp_mask(before, keep_yaw)
    out = dict(after)
    for key in ('render_base', 'erp'):
        old = geo.rgb_array(before[key])
        new = geo.rgb_array(after[key]).copy()
        new[protected] = old[protected]
        out[key] = new
        assert np.array_equal(new[protected], old[protected])
    l, t, r, b = geo.profile_content_box(before['viewport_profile'], 1344, 768)
    old = geo.render_view(before, yaw=keep_yaw)
    new = geo.render_view(out, yaw=keep_yaw)
    old = old[t + 96:b - 96, l + 96:r - 96]
    new = new[t + 96:b - 96, l + 96:r - 96]
    maximum = int(np.max(np.abs(old.astype(np.int16) - new.astype(np.int16))))
    if maximum:
        raise RuntimeError('protected prior-view overlap changed after ERP seam repair')
    return (out, {'active': True, 'keep_yaw': keep_yaw, 'native_interior_margin_pixels': 96, 'protected_ERP_texels': int(protected.sum()), 'interpolation_neighbor_margin_ERP_pixels': 2, 'native_interior_max_abs_change': maximum, 'protected_ERP_max_abs_change': 0})

def reachable_poses(path):
    poses = {(0.0, 0.0)}
    last = {'yaw': 0.0, 'pitch': 0.0}
    for current in path['camera_path']:
        yaw = float(current['yaw'])
        pitch = float(current['pitch'])
        if not math.isfinite(yaw + pitch) or abs(pitch) > 80:
            raise ValueError('invalid planned camera pose')
        count = max(1, math.ceil(max(abs(yaw - last['yaw']), abs(pitch - last['pitch'])) / 20))
        for index in range(count + 1):
            a = index / count
            y = last['yaw'] + (yaw - last['yaw']) * a
            p = last['pitch'] + (pitch - last['pitch']) * a
            for dy, dp in ((0, 0), (-20, 0), (20, 0), (0, -20), (0, 20)):
                poses.add((round(y + dy, 2), round(float(np.clip(p + dp, -80, 80)), 2)))
        last = {'yaw': yaw, 'pitch': pitch}
    return sorted(poses)

def edge_metrics(state, yaw, pitch):
    width, height = (896, 512)
    image = geo.render_view(state, yaw=yaw, pitch=pitch, width=width, height=height)
    l, t, r, b = geo.profile_content_box(state['viewport_profile'], width, height)
    cw, ch = (r - l, b - t)
    cam = geo.crop_camera(yaw, pitch, 90, cw, ch)
    x, y = geo._camera_grid(90, cw, ch)
    geometry = projection
    wx, wy, wz = geometry._camera_to_world_components(x, y, np.ones((ch, cw)), cam['center_lon'], cam['center_lat'])
    directions = np.stack(np.broadcast_arrays(wx, wy, wz), axis=-1)
    official = next((p for p in state['patches'] if p['role'] == 'official_initial_native_front_texture'))
    px, py, a = geo._project_patch(directions, official)
    known = a >= 0.999999
    rgb = image[t:b, l:r].astype(np.float32)
    lap = np.abs(cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_32F))
    native = official['pixels']
    oh, ow = geo.rgb_array(native).shape[:2]
    out = []
    for axis in (0, 1):
        gradients = np.abs(np.diff(rgb, axis=axis)).mean(axis=-1)
        ij = np.argwhere(np.diff(known.astype(np.int8), axis=axis) != 0)
        if not len(ij):
            continue
        distance = np.stack((np.abs(px[ij[:, 0], ij[:, 1]]), np.abs(px[ij[:, 0], ij[:, 1]] - (ow - 1)), np.abs(py[ij[:, 0], ij[:, 1]]), np.abs(py[ij[:, 0], ij[:, 1]] - (oh - 1))), axis=1)
        edgeid = distance.argmin(axis=1)
        for eid, name in enumerate(('left', 'right', 'top', 'bottom')):
            q = ij[edgeid == eid]
            if not len(q):
                continue
            crossing = gradients[q[:, 0], q[:, 1]]
            controls = []
            sharp = [[], []]
            for d in (8, 24):
                for sign in (-1, 1):
                    z = q.copy()
                    z[:, axis] += sign * d
                    valid = (z[:, 0] >= 0) & (z[:, 0] < gradients.shape[0]) & (z[:, 1] >= 0) & (z[:, 1] < gradients.shape[1])
                    z = z[valid]
                    if len(z):
                        controls.extend(gradients[z[:, 0], z[:, 1]].tolist())
                        mask = known[z[:, 0], z[:, 1]]
                        for k in (0, 1):
                            sharp[k].extend(lap[z[mask == bool(k), 0], z[mask == bool(k), 1]].tolist())
            ratio = float(crossing.mean() / max(0.001, np.mean(controls))) if controls else None
            sharpness = None
            if all(sharp):
                values = [float(np.mean(s)) for s in sharp]
                sharpness = max(values) / max(0.001, min(values))
            out.append({'edge': name, 'axis': axis, 'boundary_pairs': len(q), 'boundary_to_nearby_ratio': ratio, 'sharpness_ratio': sharpness, 'cross_gradient_mean': float(crossing.mean()), 'nearby_gradient_mean': float(np.mean(controls)) if controls else None})
    return (image, {'yaw': yaw, 'pitch': pitch, 'edges': out, 'numeric_pass': all(((v['boundary_to_nearby_ratio'] is None or v['boundary_to_nearby_ratio'] <= 3) and (v['sharpness_ratio'] is None or v['sharpness_ratio'] <= 3) for v in out))})

def _out_path(path, *, deps):
    path = Path(path)
    if not path.is_absolute():
        path = deps.run_root / path
    return resolve_run_path(deps.run_root, path.relative_to(deps.run_root).as_posix(), must_exist=False)

def _save(path, pixels, *, deps):
    return save_png(_out_path(path, deps=deps), pixels, run_root=deps.run_root)

def input_binding(row, *, deps):
    return {'key': row['key'], 'official_source': file_ref(row['initial_image'], run_root=deps.run_root, image=True), 'instructions': row.get('_item_instructions', [{'segment': row['segment'], 'caption': row['current_caption']}]), 'method_id': 'wmmllm.full_system.v1'}

def verify_root_assets(record, row, *, deps):
    if record.get('key') != row['key'] or record.get('official_source', {}).get('sha256') != file_ref(row['initial_image'], run_root=deps.run_root, image=True)['sha256']:
        raise RuntimeError('Saved root belongs to another source')
    for name in ('official_source', 'panorama', 'endpoint', 'panorama_state'):
        verify_ref(record[name], run_root=deps.run_root)
    geo.load_panorama_state(record['panorama_state'], run_root=deps.run_root)

def _context(row, boot, context, context_ref, *, deps):
    path = boot / 'ROOT_CONTEXT.json'
    if context is not None:
        write_once(path, context, run_root=deps.run_root)
        if context_ref is not None:
            verify_ref(context_ref, run_root=deps.run_root)
        return (dict(context), context_ref or file_ref(path, run_root=deps.run_root))
    if path.exists():
        return (json.loads(path.read_text()), file_ref(path, run_root=deps.run_root))
    result = deps.providers.chat.complete(ChatRequest('planner', read_text('bootstrap_context_system.txt'), (image_input(file_ref(row['initial_image'], run_root=deps.run_root, image=True), 'official initial perspective', run_root=deps.run_root),), {'task_id': row['task_id']}, read_json_data('bootstrap_context_response_format.json')))
    context = json.loads(result.text)
    if set(context) != {'scene_description', 'style_tokens', 'entity_identity_properties'} or not all((isinstance(v, str) for v in context.values())):
        raise ValueError('Invalid bootstrap scene schema')
    write_once(path, context, run_root=deps.run_root)
    return (context, file_ref(path, run_root=deps.run_root))

def camera_path(row, boot, *, deps):
    path = boot / 'PLANNED_CAMERA_PATH.json'
    binding = input_binding(row, deps=deps)
    if path.exists():
        value = json.loads(path.read_text())
        if value.get('input_binding') != binding:
            raise RuntimeError('Saved camera schedule source/caption binding differs')
        return value
    pose = {'type': 'object', 'properties': {'segment': {'type': 'integer'}, 'yaw': {'type': 'number'}, 'pitch': {'type': 'number'}}, 'required': ['segment', 'yaw', 'pitch'], 'additionalProperties': False}
    answer = deps.providers.chat.complete(ChatRequest('planner', read_text('bootstrap_camera_system.txt'), (image_input(binding['official_source'], 'official initial state', run_root=deps.run_root),), {'instructions': binding['instructions']}, read_json_data('bootstrap_camera_response_format.json')))
    value = json.loads(answer.text)
    if set(value) != {'camera_path'} or not isinstance(value['camera_path'], list):
        raise ValueError('Invalid camera schedule schema')
    expected = [v['segment'] for v in binding['instructions']]
    if [v['segment'] for v in value['camera_path']] != expected:
        raise ValueError('Camera schedule does not cover exactly the input sequence')
    reachable_poses(value)
    value.update(input_binding=binding, planner_receipts=[answer.receipt.to_record()], margin_degrees=20, reasoning_effort=answer.receipt.reasoning_effort)
    write_once(path, value, run_root=deps.run_root)
    return value

def evidence(row, state, stage, path, *, deps, deadline_monotonic=None):
    stage = _out_path(stage, deps=deps)
    stage.mkdir(parents=True, exist_ok=True)
    poses = reachable_poses(path)
    tw, th, columns = (336, 192, 5)
    sheet = Image.new('RGB', (columns * tw, math.ceil(len(poses) / columns) * (th + 24)), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    metrics = []
    views = []
    for i, (yaw, pitch) in enumerate(poses):
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError('repair deadline during gate rendering')
        pixels, metric = edge_metrics(state, yaw, pitch)
        metrics.append(metric)
        ref = _save(stage / f'view_yaw{yaw:+07.2f}_pitch{pitch:+07.2f}.png', pixels, deps=deps)
        views.append({'yaw': yaw, 'pitch': pitch, 'image': ref})
        xx = i % columns * tw
        yy = i // columns * (th + 24)
        sheet.paste(Image.fromarray(pixels).resize((tw, th)), (xx, yy + 24))
        draw.text((xx + 4, yy + 4), f'yaw{yaw:+g} pitch{pitch:+g}', fill='white')
    sheet_ref = _save(stage / 'REACHABLE_VIEWS.png', np.asarray(sheet), deps=deps)
    report = {'schema': 'wmmllm.reachable_seam_gate.v1', 'key': row['key'], 'planned_path': path, 'margin_degrees': 20, 'views': views, 'metrics': metrics, 'numeric_threshold': 3, 'numeric_pass': all((v['numeric_pass'] for v in metrics)), 'contact_sheet': sheet_ref, 'full_sphere_required': False}
    write_once(stage / 'SEAM_REPORT.json', report, run_root=deps.run_root)
    return report

def gate(row, record, stage, path, *, deps, deadline_monotonic=None):
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise TimeoutError('repair deadline before gate')
    state = geo.load_panorama_state(record['panorama_state'], run_root=deps.run_root)
    report = evidence(row, state, stage, path, deps=deps, deadline_monotonic=deadline_monotonic)
    properties = read_json_data('bootstrap_gate_response_format.json')['json_schema']['schema']['properties']
    timeout = None if deadline_monotonic is None else min(deps.config.chat.timeout_seconds, max(0.001, deadline_monotonic - time.monotonic()))
    answer = deps.providers.chat.complete(ChatRequest('verifier', read_text('bootstrap_gate_system.txt'), (image_input(record['official_source'], 'official source', run_root=deps.run_root), image_input(report['contact_sheet'], 'actual reachable views, labelled yaw/pitch', run_root=deps.run_root)), {'numeric_metrics': report['metrics']}, read_json_data('bootstrap_gate_response_format.json'), timeout_seconds=timeout))
    review = json.loads(answer.text)
    if set(review) != set(properties) or review['verdict'] not in ('PASS', 'FAIL', 'UNCERTAIN') or any((type(review[k]) is not bool for k in ('visible_seam', 'duplicated_object', 'cut_object'))) or (not isinstance(review['detail'], str)) or (not isinstance(review['repair_views'], list)) or (len(review['repair_views']) > 2):
        raise ValueError('Invalid reachable seam verdict')
    for view in review['repair_views']:
        if set(view) != {'yaw', 'pitch'} or not all((type(v) in (int, float) and math.isfinite(v) for v in view.values())) or abs(view['pitch']) > 80:
            raise ValueError('Invalid repair camera')
    review.update(reasoning_effort=answer.receipt.reasoning_effort, receipts=[answer.receipt.to_record()], numeric_pass=report['numeric_pass'])
    passed = report['numeric_pass'] and review['verdict'] != 'FAIL' and (not any((review[k] for k in ('visible_seam', 'duplicated_object', 'cut_object'))))
    write_once(Path(stage) / 'VERDICT.json', review, run_root=deps.run_root)
    return (passed, review, report)

def _masked_image(directory, source_ref, mask_ref, prompt, *, deps, panorama=False, request_size=None, output_size=None, deadline_monotonic=None):
    directory = _out_path(directory, deps=deps)
    directory.mkdir(parents=True, exist_ok=True)
    output = _out_path(directory / 'generated', deps=deps)
    timeout = None if deadline_monotonic is None else min(deps.config.image.timeout_seconds, max(0.001, deadline_monotonic - time.monotonic()))
    result = deps.providers.image.edit(ImageEditRequest(prompt, (image_input(source_ref, '', run_root=deps.run_root),), output, mask=image_input(mask_ref, '', run_root=deps.run_root), panorama=panorama, request_size=request_size, output_size=output_size, timeout_seconds=timeout))
    return (result, result.receipt.to_record())

def _initial(row, boot, context, context_ref, *, width, deps):
    context, context_ref = _context(row, boot, context, context_ref, deps=deps)
    stage = boot / 'registered_native_ERP_C0_v2' / 'attempt001'
    stage.mkdir(parents=True, exist_ok=True)
    canvas, mask, geometry = perspective_outpaint_input(row['initial_image'], source_metadata=_source_metadata(row))
    c = _save(stage / 'NATIVE_INPUT.png', canvas, deps=deps)
    m = _save(stage / 'NATIVE_MASK.png', mask, deps=deps)
    write_once(stage / 'NATIVE_GEOMETRY.json', geometry, run_root=deps.run_root)
    w, h = geometry['canvas_size']
    x0, y0, x1, y1 = geometry['official_content_box']
    prompt = read_text('bootstrap_native_template.txt').format(x0=x0, x1=x1, y0=y0, y1=y1)
    if 'museum' in context.get('scene_description', '').lower() and 'sculpture' in context.get('scene_description', '').lower():
        prompt = 'This is architectural border completion of an existing museum gallery photograph. Preserve the existing sculptures, paintings, visitors and their clothing exactly. ' + prompt
    result, native_receipt = _masked_image(stage / 'native_call', c, m, prompt, deps=deps, request_size=f'{w}x{h}', output_size=(w, h))
    native, registration = register_native_context(result.image, canvas, mask)
    native_ref = _save(stage / 'NATIVE_CONTEXT.png', native, deps=deps)
    write_once(stage / 'NATIVE_REGISTRATION.json', registration, run_root=deps.run_root)
    erp, editable, projection_record = geo.erp_outpaint_input(native, hfov=registration['context_hfov_degrees'])
    ec = _save(stage / 'ERP_INPUT.png', erp, deps=deps)
    em = _save(stage / 'ERP_MASK.png', editable, deps=deps)
    prompt = read_text('bootstrap_erp_template.txt').format(style_tokens=context['style_tokens'])
    result, erp_receipt = _masked_image(stage / 'erp_call', ec, em, prompt, deps=deps, panorama=True)
    aligned, erp_registration = register_erp_to_source(result.image, row['initial_image'], source_metadata=_source_metadata(row))
    aligned_ref = _save(stage / 'REGISTERED_ERP.png', aligned, deps=deps)
    write_once(stage / 'ERP_REGISTRATION.json', erp_registration, run_root=deps.run_root)
    write_once(stage / 'ERP_PROJECTION.json', projection_record, run_root=deps.run_root)
    state, coherent = corrected_single_erp_state(aligned, row['initial_image'], _source_metadata(row), width=width)
    old_base = geo.rgb_array(state['render_base'])
    state['render_base'], official_base_alpha, official_base_projection = geo._composite_content_edit(old_base, state['patches'][0]['pixels'], blend_feather_px=0, return_metadata=True)
    official_base_known = official_base_alpha > 0
    if np.any(old_base[~official_base_known] != state['render_base'][~official_base_known]):
        raise RuntimeError('official base restoration modified unknown texels')
    official_base_consistency = {'known_base_texels_restored_count': int(np.any(old_base != state['render_base'], axis=-1).sum()), 'unknown_base_changed_pixels': 0, 'projection': official_base_projection}
    state, baked = bake_native_collar(row, state, {'_native_context_path': verify_ref(native_ref, run_root=deps.run_root), 'native_registration': registration})
    state, colour = correct_colour_collar(row, state)
    front = geo.render_view(state)
    error = geo.image_error(row['initial_image'], front)
    if not error['pixel_identical'] or len(state['patches']) != 1:
        raise RuntimeError('bootstrap lost the one exact native source patch')
    state_ref = geo.save_panorama_state(state, stage / 'state', run_root=deps.run_root)
    return {'schema': 'wmmllm.bootstrap.v1', 'recipe': 'registered_native_context_baked_into_coherent_ERP_with_outside_C0_collar_v2', 'key': row['key'], 'task_id': row['task_id'], 'panorama': _save(stage / 'panorama.png', geo.erp_array(state), deps=deps), 'endpoint': _save(stage / 'initial_view.png', front, deps=deps), 'official_source': file_ref(row['initial_image'], run_root=deps.run_root, image=True), 'front_preservation': {**error, 'passed': True, 'exact_source_reconstruction': True}, 'context': context, 'context_ref': context_ref, 'scene_rev': 0, 'camera': {'yaw': 0.0, 'pitch': 0.0}, 'panorama_state': state_ref, 'native_context': native_ref, 'native_receipt': native_receipt, 'erp_receipt': erp_receipt, 'generated_erp': aligned_ref, 'native_registration': registration, 'erp_registration': erp_registration, 'coherent_geometry': coherent, 'official_base_consistency': official_base_consistency, 'native_bake_geometry': baked, 'outside_colour_geometry': colour, 'native_patch_count': 1, 'additional_native_context_patch': False, 'geometry_flow_used': False, 'production_recipe_nominal_image_calls': 2, 'provider_refusal_retries_max_per_image': 0, 'no_model_fallback': True}

def repair(row, state, yaw, pitch=0, *, out_dir, context, deps, feedback='', keep_yaw=None, deadline_monotonic=None):
    out = _out_path(out_dir, deps=deps)
    out.mkdir(parents=True, exist_ok=True)
    original = geo.render_view(state, yaw=yaw, pitch=pitch)
    l, t, r, b = geo.profile_content_box(state['viewport_profile'], 1344, 768)
    cw, ch = (r - l, b - t)
    x, y = geo._camera_grid(90, cw, ch)
    cam = geo.crop_camera(yaw, pitch, 90, cw, ch)
    wx, wy, wz = projection._camera_to_world_components(x, y, np.ones((ch, cw)), cam['center_lon'], cam['center_lat'])
    directions = np.stack(np.broadcast_arrays(wx, wy, wz), axis=-1)
    _, _, alpha = geo._project_patch(directions, state['patches'][0])
    official = alpha > 0
    keep = official.copy()
    if keep_yaw is not None:
        proxy = {'camera': geo.crop_camera(keep_yaw, 0, 90, cw, ch), 'alpha': np.ones((ch, cw), np.float32)}
        px, py, prior = geo._project_patch(directions, proxy)
        border = np.minimum.reduce([px + 0.5, cw - px - 0.5, py + 0.5, ch - py - 0.5])
        keep |= (prior > 0) & (border >= 96)
    editable = ~keep
    canvas = original.copy()
    canvas[t:b, l:r][editable] = 127
    mask = np.zeros((768, 1344), np.uint8)
    mask[t:b, l:r] = editable * 255
    dx = np.minimum(np.arange(cw) + 0.5, cw - np.arange(cw) - 0.5)
    dy = np.minimum(np.arange(ch) + 0.5, ch - np.arange(ch) - 0.5)
    support = editable * np.clip(np.minimum(dx[None, :], dy[:, None]) / 64, 0, 1)
    source_ref = _save(out / 'SOURCE_ONLY_INPUT.png', canvas, deps=deps)
    mask_ref = _save(out / 'EDITABLE_MASK.png', mask, deps=deps)
    scene = context.get('scene_description', '') + ' ' + context.get('entity_identity_properties', '')
    prompt = read_text('bootstrap_repair_template.txt').format(scene=scene, feedback=feedback)
    result, receipt = _masked_image(out / 'provider_attempt001', source_ref, mask_ref, prompt, deps=deps, request_size='1792x1024', output_size=(1344, 768), deadline_monotonic=deadline_monotonic)
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise TimeoutError('repair deadline after image response')
    raw = geo.rgb_array(result.image)
    H, reg = _source_homography(canvas, raw)
    aligned = cv2.warpPerspective(raw, np.linalg.inv(H), (1344, 768), flags=cv2.INTER_LINEAR, borderValue=0)
    valid = cv2.warpPerspective(np.ones(raw.shape[:2], np.uint8), np.linalg.inv(H), (1344, 768), flags=cv2.INTER_NEAREST, borderValue=0)
    valid_distance = cv2.distanceTransform(valid, cv2.DIST_L2, 3)
    support *= np.clip(valid_distance[t:b, l:r] / 24, 0, 1)
    aligned[t:b, l:r][keep] = original[t:b, l:r][keep]
    _save(out / 'VALID_WARP_SUPPORT.png', (valid * 255).astype(np.uint8), deps=deps)
    _save(out / 'ALIGNED_SOURCE_OUTPAINT.png', aligned, deps=deps)
    base, erp_alpha, meta = geo._composite_content_edit(state['render_base'], aligned[t:b, l:r], yaw=yaw, pitch=pitch, hfov=90, support=support, return_metadata=True)
    preview, _, _ = geo._composite_content_edit(state['erp'], aligned[t:b, l:r], yaw=yaw, pitch=pitch, hfov=90, support=support, return_metadata=True)
    accepted = dict(state)
    accepted['erp'] = preview
    accepted['render_base'] = base
    accepted, c0 = correct_colour_collar(row, accepted)
    accepted, prior_preservation = preserve_prior_overlap(state, accepted, keep_yaw)
    front = geo.image_error(row['initial_image'], geo.render_view(accepted))
    if not front['pixel_identical'] or len(accepted['patches']) != 1:
        raise RuntimeError('repair changed the exact native source')
    record = {'panorama_state': geo.save_panorama_state(accepted, out / 'state', run_root=deps.run_root), 'source_pixel_preservation': front, 'prior_overlap_preservation': prior_preservation, 'registration': reg, 'valid_inverse_warp_fraction': float(valid.mean()), 'invalid_response_pixels_applied': 0, 'ERP_support': meta, 'outer_view_feather_output_pixels': 64, 'source_protected_world_frustum': True, 'additional_native_patch': False, 'outside_geometric_projection_support_changed': 0, 'C0_photometric_scope': 'outside official frustum within64 source pixels, then restore protected prior-view ERP texels', 'C0': c0, 'receipt': receipt}
    for y, p in [(-45, 0), (-60, 0), (-90, 0), (-110, 0), (-135, 0), (45, 0), (0, 20)]:
        _save(out / f'yaw{y}_pitch{p}.png', geo.render_view(accepted, yaw=y, pitch=p), deps=deps)
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise TimeoutError('repair deadline during reprojection')
    write_once(out / 'REPAIR.json', record, run_root=deps.run_root)
    return (accepted, record)

def initialize(row, output_root, context=None, context_ref=None, *, width=4096, deps):
    """Initialize one exact native source with at most one seam repair round."""
    import re
    from ..providers.errors import TransportExhausted
    from ..config import ConfigError
    if tuple(deps.config.video.size) != (1344, 768) or tuple(deps.config.image.view_output_size) != (1344, 768):
        raise ConfigError('Core sessions require a 1344x768 viewport and image output; no request was sent.')
    row = dict(row)
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', row['key']):
        raise ValueError('Invalid bootstrap key')
    initial = verify_ref(row['initial_image'], run_root=deps.run_root) if isinstance(row['initial_image'], dict) else Path(row['initial_image'])
    if not initial.is_absolute():
        initial = resolve_run_path(deps.run_root, initial.as_posix())
    else:
        initial = resolve_run_path(deps.run_root, initial.relative_to(deps.run_root).as_posix())
    row['initial_image'] = initial
    if _source_metadata(row) is None and row.get('source_initial_state_ref'):
        row['source_profile'] = json.loads(verify_ref(row['source_initial_state_ref'], run_root=deps.run_root).read_text())
    boot = _out_path(Path(output_root) / row['key'] / 'bootstrap', deps=deps)
    boot.mkdir(parents=True, exist_ok=True)
    done = boot / 'COMMITTED.json'
    binding = input_binding(row, deps=deps)
    if done.exists():
        cached = json.loads(done.read_text())
        verify_root_assets(cached, row, deps=deps)
        if cached.get('input_binding') != binding:
            raise RuntimeError('Saved bootstrap source/captions binding differs')
        return cached
    path = camera_path(row, boot, deps=deps)
    source = row.get('_bootstrap_reuse')
    if source:
        record = json.loads(verify_ref(source, run_root=deps.run_root).read_text())
        verify_root_assets(record, row, deps=deps)
        if record.get('input_binding') != binding:
            raise RuntimeError('Imported bootstrap binding differs')
        record = {**record, 'reused_bootstrap': dict(source), 'original_root_source_sha256': record['official_source']['sha256']}
    else:
        record = _initial(row, boot, context, context_ref, width=width, deps=deps)
    inherited_unresolved = bool(source and record.get('seam_status') == 'seam_unresolved')
    inherited_round_spent = bool(source and (record.get('repair_rounds_consumed', 0) >= 1 or record.get('repair_records') or record.get('repair_error') or (inherited_unresolved and record.get('repair_rounds_max', 0) >= 1)))
    passed, review, report = gate(row, record, boot / 'reachable_gate_initial', path, deps=deps)
    started = time.monotonic()
    deadline = started + 600
    repair_records = []
    repair_error = None
    repair_calls = 0
    round_consumed = inherited_round_spent
    if not passed and inherited_round_spent:
        repair_error = 'inherited_one_round_budget_already_consumed'
    elif not passed:
        round_consumed = True
        state = geo.load_panorama_state(record['panorama_state'], run_root=deps.run_root)
        views = review.get('repair_views', [])[:2]
        if not views:
            views = [{'yaw': m['yaw'], 'pitch': m['pitch']} for m in report['metrics'] if not m['numeric_pass']][:2]
        if not views:
            views = [{'yaw': -45.0, 'pitch': 0.0}, {'yaw': 45.0, 'pitch': 0.0}]
        for i, view in enumerate(views):
            if time.monotonic() >= deadline:
                repair_error = 'ten_minute_timebox'
                break
            out = boot / 'repair_round001' / f'view{i + 1:02d}'
            try:
                repair_calls += 1
                state, receipt = repair(row, state, float(view['yaw']), float(view['pitch']), out_dir=out, context=record['context'], deps=deps, feedback=review['detail'], deadline_monotonic=deadline)
                repair_records.append(file_ref(out / 'REPAIR.json', run_root=deps.run_root))
            except TransportExhausted:
                raise
            except Exception as error:
                raise_if_fatal_provider_error(error)
                repair_error = safe_provider_error(error)
                break
        if repair_records:
            record = {**record, 'panorama_state': geo.save_panorama_state(state, boot / 'repair_round001/state', run_root=deps.run_root), 'panorama': _save(boot / 'repair_round001/panorama.png', geo.erp_array(state), deps=deps), 'endpoint': _save(boot / 'repair_round001/initial_view.png', geo.render_view(state), deps=deps)}
        if time.monotonic() < deadline:
            try:
                passed, review, report = gate(row, record, boot / 'reachable_gate_final', path, deps=deps, deadline_monotonic=deadline)
            except TransportExhausted:
                raise
            except Exception as error:
                passed = False
                raise_if_fatal_provider_error(error)
                repair_error = safe_provider_error(error)
        else:
            passed = False
            repair_error = repair_error or 'ten_minute_timebox'
    if inherited_unresolved:
        passed = False
    record.update(method_id='wmmllm.full_system.v1', reasoning_effort=deps.config.chat.reasoning_effort, recipe='C0_collar_reachable_timeboxed', input_binding=binding, inherited_seam_unresolved=inherited_unresolved, inherited_repair_round_spent=inherited_round_spent, repair_rounds_consumed=int(round_consumed), repair_image_calls_this_version=repair_calls, repair_deadline_elapsed_seconds=time.monotonic() - started, seam_status='resolved' if passed else 'seam_unresolved', seam_gate=report, seam_verifier=review, repair_records=repair_records, repair_error=repair_error, production_recipe_nominal_image_calls=2, correction_image_calls_max=2, maximum_image_calls_excluding_provider_retries=4, provider_refusal_retries_max_per_image=0, repair_rounds_max=1, repair_timebox_seconds=600, rotation_remains_geometric=True, visual_acceptance=review['verdict'], camera={'yaw': 0.0, 'pitch': 0.0})
    write_once(done, record, run_root=deps.run_root)
    return record
