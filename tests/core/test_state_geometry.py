"""CPU state invariants using procedural pixels only."""
from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
from PIL import Image
import pytest

from wmmllm.state import graph, panorama as p, projection as geom, registration as reg
from wmmllm.session.io import file_ref, write_once


def texture(width=128, height=64):
    y, x = np.indices((height, width))
    return np.stack((1 + (x * 11 + y * 7) % 254, 1 + (x * 3 + y * 17) % 254,
                     1 + (x * 19 + y * 5) % 254), axis=-1).astype(np.uint8)


def state(*, padded=False):
    source = texture(48, 32)
    if padded:
        source[:4] = 0
        source[-4:] = 0
    return p.create_panorama_state(texture(), source, width=128, height=64,
                                   source_content_box=[0, 4, 48, 28] if padded else [0, 0, 48, 32])


@pytest.mark.parametrize('yaw,pitch', [(0, 0), (45, 10), (180, -20), (359, 70)])
def test_same_state_pose_profile_returns_exact_pixels(yaw, pitch):
    value = state(padded=True)
    first = p.render_view(value, yaw, pitch, width=48, height=32)
    assert np.array_equal(first, p.render_view(value, yaw + 360, pitch, width=48, height=32))
    assert np.array_equal(first[:4], np.zeros_like(first[:4]))
    assert np.array_equal(first[-4:], np.zeros_like(first[-4:]))


def test_native_front_and_ordered_patch_locality():
    value = state()
    source = p.render_view(value, width=48, height=32)
    assert np.array_equal(source, texture(48, 32))
    edit = source.copy()
    edit[12:22, 15:30] = [230, 10, 80]
    support = np.zeros((32, 48), np.uint8)
    support[10:24, 13:32] = 255
    changed, erp_alpha, metadata = p.composite_view_edit(value, edit, support=support)
    assert changed['render_base'] is value['render_base']
    assert len(changed['patches']) == len(value['patches']) + 1
    assert np.array_equal(p.erp_array(value)[erp_alpha == 0], p.erp_array(changed)[erp_alpha == 0])
    before = p.render_view(value, yaw=22, pitch=8, width=61, height=41)
    after = p.render_view(changed, yaw=22, pitch=8, width=61, height=41)
    cam = p.crop_camera(22, 8, 90, 61, 41)
    x, y = p._camera_grid(90., 61, 41)
    rays = np.stack(np.broadcast_arrays(*geom._camera_to_world_components(
        x, y, np.ones((41, 61)), cam['center_lon'], cam['center_lat'])), axis=-1)
    _, _, alpha = p._project_patch(rays, changed['patches'][-1])
    assert np.array_equal(before[alpha == 0], after[alpha == 0])
    assert metadata['outside_support_changed_pixels'] == 0


def test_padding_is_screen_fixed_and_never_reprojected():
    value = state(padded=True)
    edit = p.render_view(value, width=48, height=32)
    edit[:4] = 255
    support = np.zeros((32, 48), np.uint8)
    support[:4] = 255
    changed, alpha, _ = p.composite_view_edit(value, edit, support=support)
    assert not alpha.any(); assert np.array_equal(p.erp_array(value), p.erp_array(changed))
    assert np.array_equal(p.render_view(value, yaw=51, width=48, height=32),
                          p.render_view(changed, yaw=51, width=48, height=32))


def test_global_generation_retires_every_native_patch():
    old = state()
    changed = p.create_panorama_state(np.full((64, 128, 3), 71, np.uint8), source=None,
                                      width=128, height=64)
    changed['viewport_profile'] = deepcopy(old['viewport_profile'])
    assert not changed['patches']; assert np.all(p.render_view(changed, width=48, height=32) == 71)


def test_state_roundtrip_assets_reuse_and_immutable_manifest(tmp_path):
    value = state(padded=True)
    ref = p.save_panorama_state(value, tmp_path / 'states/rev0', run_root=tmp_path)
    # An identical reconstructed state reuses its assets and immutable manifest byte for byte.
    same = p.save_panorama_state(state(padded=True), tmp_path / 'states/rev0', run_root=tmp_path)
    assert same == ref
    loaded = p.load_panorama_state(ref, run_root=tmp_path)
    assert p.panorama_state_sha256(value) == p.panorama_state_sha256(loaded)
    for yaw in (0, 35, 180):
        assert np.array_equal(p.render_view(value, yaw=yaw, width=48, height=32),
                              p.render_view(loaded, yaw=yaw, width=48, height=32))
    second = p.save_panorama_state(loaded, tmp_path / 'states/rev1', run_root=tmp_path)
    first_record = json.loads((tmp_path / ref['path']).read_text())
    second_record = json.loads((tmp_path / second['path']).read_text())
    assert first_record['render_base'] == second_record['render_base']
    assert first_record['patches'] == second_record['patches']
    assert not (tmp_path / 'states/rev1/panorama_assets').exists()
    changed = deepcopy(loaded)
    changed['erp'][0, 0] ^= 1
    with pytest.raises(ValueError, match='Immutable record changed'):
        p.save_panorama_state(changed, tmp_path / 'states/rev0', run_root=tmp_path)


@pytest.mark.parametrize('field', ['encoded', 'decoded', 'canonical'])
def test_encoded_decoded_and_canonical_hashes_are_checked_independently(tmp_path, field):
    ref = p.save_panorama_state(state(), tmp_path / 'states/rev0', run_root=tmp_path)
    manifest = tmp_path / ref['path']
    record = json.loads(manifest.read_text())
    if field == 'encoded':
        alpha_path = tmp_path / record['patches'][0]['alpha']['path']
        array = np.load(alpha_path, allow_pickle=False)
        array[0, 0] = 0
        with alpha_path.open('wb') as stream:
            np.save(stream, array, allow_pickle=False)
    elif field == 'decoded':
        record['patches'][0]['pixels']['pixel_sha256'] = '0' * 64
    else:
        record['panorama_state_sha256'] = '0' * 64
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='(hash|Pixel)'):
        p.load_panorama_state(manifest, run_root=tmp_path)


def test_state_manifest_cannot_follow_a_symlink_escape(tmp_path):
    outside = tmp_path.parent / 'outside-state.json'
    outside.write_text('{}')
    link = tmp_path / 'state.json'
    link.symlink_to(outside)
    with pytest.raises(ValueError, match='containment'):
        p.load_panorama_state({'path': 'state.json'}, run_root=tmp_path)


def test_strict_support_blend_never_changes_outside():
    source = texture(160, 96)
    generated = np.roll(source, 1, axis=1)
    generated[33:57, 55:91] = [212, 9, 93]
    support = np.zeros((96, 160), np.uint8)
    support[20:70, 40:110] = 255
    result, meta = reg.blend_endpoint(source, generated, support, feather_pixels=12,
                                     max_translation_pixels=3, max_rotation_degrees=.05,
                                     max_corner_displacement_pixels=3)
    assert np.array_equal(source[support == 0], result[support == 0])
    assert meta['outside_exact'] and meta['outside_changed_pixels'] == 0


def _node(tmp_path, node_id='node0', rev=0):
    value = state()
    saved = p.save_panorama_state(value, tmp_path / 'states' / node_id, run_root=tmp_path)
    view = p.render_view(value, width=48, height=32)
    image_path = tmp_path / (node_id + '.png')
    Image.fromarray(view).save(image_path)
    memory = write_once(tmp_path / (node_id + '-memory.json'), {}, run_root=tmp_path)
    return graph.node_record(node_id=node_id, parent_node_id=None,
        origin_id='origin0', physical_revision=rev, view_identity='first_person|normal|default_subject',
        pose={'yaw': 0, 'pitch': 0, 'hfov': 90}, panorama_state=saved,
        panorama=saved['erp_preview_ref'], view=file_ref(image_path, run_root=tmp_path, image=True),
        memory_snapshot=memory, viewport_profile=value['viewport_profile'])


def test_saved_node_graph_roundtrip_and_exact_return(tmp_path):
    node = _node(tmp_path)
    value = graph.verify_saved_node(node, run_root=tmp_path)
    assert p.panorama_state_sha256(value) == node['panorama_state']['panorama_state_sha256']
    store = graph.SceneGraph(run_root=tmp_path)
    store.add_node(node)
    assert json.loads((tmp_path / 'nodes/node0.json').read_text()) == node
    changed = deepcopy(node)
    changed['pose']['yaw'] = 90
    with pytest.raises(ValueError, match='immutable graph node changed'):
        store.add_node(changed)
    identity, pose = node['view_identity'], {'yaw': 360, 'pitch': 0}
    assert graph.visit_key(0, identity, pose) == graph.visit_key(0, identity, {'yaw': 0, 'pitch': 0})
    assert graph.visit_key(0, identity, pose) != graph.visit_key(1, identity, pose)
    assert graph.visit_key(0, {'a': 'x', 'b': 'y'}, pose) == graph.visit_key(0, {'b': 'y', 'a': 'x'}, pose)


def test_saved_node_rejects_profile_origin_identity_or_arrival_change(tmp_path):
    original = _node(tmp_path)
    for field, value in [('origin_id', ''), ('view_identity', 'unresolved_perspective|normal|viewer'),
                          ('viewport_profile', {**original['viewport_profile'], 'width': 49}),
                          ('pose', {'yaw': 90, 'pitch': 0, 'hfov': 90})]:
        node = deepcopy(original)
        node[field] = value
        with pytest.raises(ValueError):
            graph.verify_saved_node(node, run_root=tmp_path)
