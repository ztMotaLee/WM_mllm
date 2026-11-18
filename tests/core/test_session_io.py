import numpy as np
import pytest
from wmmllm.session.io import (IntegrityError, ImmutableRecordChanged, resolve_run_path,
    verify_ref, write_atomic, write_once, save_png)
from wmmllm.state.panorama import pixel_sha256


@pytest.mark.parametrize('path', ['', '/outside.png', '../outside', 'x/../y', './x', 'x//y',
    'x\\y', 'C:/x', 'https://invalid/x', 'x?y', 'x#y', 'x\x00y'])
def test_malformed_references_rejected(tmp_path, path):
    with pytest.raises(IntegrityError):
        resolve_run_path(tmp_path, path, must_exist=False)


def test_contained_symlink_and_escape(tmp_path):
    root=tmp_path/'run';root.mkdir()
    outside=tmp_path/'outside';outside.mkdir()
    (root/'escape').symlink_to(outside, target_is_directory=True)
    with pytest.raises(IntegrityError):write_atomic(root/'escape'/'value.json',{},run_root=root)
    assert not (outside/'value.json').exists()
    (root/'dangling').symlink_to(root/'missing')
    with pytest.raises(IntegrityError):resolve_run_path(root,'dangling',must_exist=False)


def test_immutable_records_and_image_hash(tmp_path):
    path=tmp_path/'record.json'
    first=write_once(path,{'turn':1},run_root=tmp_path)
    assert write_once(path,{'turn':1},run_root=tmp_path)==first
    with pytest.raises(ImmutableRecordChanged):write_once(path,{'turn':2},run_root=tmp_path)
    pixels=np.arange(12*8*3,dtype=np.uint8).reshape(8,12,3)
    ref=save_png(tmp_path/'view.png',pixels,run_root=tmp_path)
    assert ref['pixel_sha256']==pixel_sha256(pixels); assert verify_ref(ref,run_root=tmp_path)==tmp_path/'view.png'
    (tmp_path/'view.png').write_bytes(b'changed')
    with pytest.raises(IntegrityError):verify_ref(ref,run_root=tmp_path)
