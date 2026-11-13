"""Identity crops preserve authoritative pixels and discard inherited metadata."""
import numpy as np
from PIL import Image, ImageCms
import pytest

from wmmllm.agent.contracts import extract_reference
from wmmllm.session.io import file_ref, verify_ref


@pytest.mark.parametrize('format', ['JPEG', 'PNG', 'WEBP'])
def test_identity_reference_has_no_inherited_metadata(tmp_path, format):
    array = np.random.default_rng(0).integers(0, 256, (96, 128, 3), dtype=np.uint8)
    source = tmp_path / ('source.' + format.lower())
    exif = Image.Exif()
    exif[270] = 'isolated identity crop metadata control'
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    Image.fromarray(array).save(source, format=format, icc_profile=profile, exif=exif)
    reference, record = extract_reference(
        file_ref(source, run_root=tmp_path, image=True),
        [[[0.25, 0.25], [0.50, 0.25], [0.50, 0.50], [0.25, 0.50]]],
        tmp_path / 'reference', upscale_min_edge=64, run_root=tmp_path)
    assert record['status'] == 'READY'
    assert record['native_crop_source_RGB_exact'] is True
    with Image.open(source) as original:
        expected = original.convert('RGB').crop(tuple(record['crop_bbox'])).tobytes()
    with Image.open(verify_ref(record['native_crop'], run_root=tmp_path)) as native:
        assert native.tobytes() == expected
    for ref in (record['native_crop'], reference):
        with Image.open(verify_ref(ref, run_root=tmp_path)) as output:
            assert output.mode == 'RGB'
            assert output.info == {}
            assert len(output.getexif()) == 0
