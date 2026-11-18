"""Multipart edits and source image acceptance checks, with no remote I/O."""
import base64
import io
import json
import struct
from email.parser import BytesParser
from email.policy import default

import httpx
import pytest
from PIL import Image, ImageCms

from wmmllm.config import AppConfig, ChatConfig, ImageConfig, RetryConfig
from wmmllm.image_io import encode_rgb_png, load_rgb_image, prepare_rgb_image
from wmmllm.providers import ChatRequest, ImageEditRequest, ImageInput, LocalCallError, ProviderRefusal, create_providers
from wmmllm.providers.openai_compatible import build_messages

SCHEMA = {"type": "json_schema", "json_schema": {"name": "decision", "strict": True,
    "schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}}


def config():
    return AppConfig(ChatConfig("test-secret", "test-chat", base_url="https://provider.invalid/v1"),
                     ImageConfig("test-secret", "test-image", base_url="https://provider.invalid/v1"),
                     transport=RetryConfig(jitter=0))


def encoded(size=(1792, 1024), color="green"):
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode()


def response(**kwargs):
    value = {"created": 0, "model": "test-image", "quality": "high", "data": [{"b64_json": encoded()}],
             "usage": {"input_tokens": 12, "output_tokens": 20,
                       "input_tokens_details": {"text_tokens": 3, "image_tokens": 9},
                       "output_tokens_details": {"image_tokens": 20}}}
    value.update(kwargs)
    return value


def source(tmp_path, name="source", color="red"):
    path = tmp_path / (name + ".png")
    Image.new("RGB", (32, 16), color).save(path)
    return ImageInput(path, name)


def multipart(wire):
    message = BytesParser(policy=default).parsebytes(b"Content-Type: " + wire.headers["content-type"].encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + wire.content)
    return [(part.get_param("name", header="content-disposition"), part.get_filename(), part.get_payload(decode=True)) for part in message.iter_parts()]


@pytest.mark.parametrize("panorama,size,output", [(False, "1792x1024", (1344, 768)), (True, "2048x1024", (2048, 1024))])
def test_multipart_preserves_image_order_and_normalization(tmp_path, panorama, size, output):
    inputs = (source(tmp_path, "current", "red"), source(tmp_path, "history", "blue"))
    captured = []
    def handler(wire):
        captured.append(wire)
        return httpx.Response(200, json=response(data=[{"b64_json": encoded(tuple(int(v) for v in size.split("x")))}]))
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        providers = create_providers(config(), http_client=client)
        result = providers.image.edit(ImageEditRequest("Keep action", inputs, tmp_path / "out", panorama=panorama))
    wire = captured[0]
    assert wire.url.path == "/v1/images/edits"
    parts = multipart(wire)
    text = {name: payload.decode() for name, filename, payload in parts if filename is None}
    assert text["model"] == "test-image" and text["prompt"] == "Keep action"
    assert text["size"] == size and text["quality"] == "high"; assert "input_fidelity" not in text
    files = [(filename, payload) for name, filename, payload in parts if filename]
    assert [f[0] for f in files] == ["input_0.png", "input_1.png"]
    for (_, payload), color in zip(files, ((255, 0, 0), (0, 0, 255))):
        with Image.open(io.BytesIO(payload)) as image:
            assert image.size == (32, 16) and image.getpixel((0, 0)) == color
    with Image.open(result.image) as image:
        assert image.size == output
    assert result.receipt.usage.image_input_tokens == 9
    assert list((tmp_path / "out").iterdir()) == [result.image]
    assert "Keep action" not in str(result.receipt.to_record())


def test_mask_conversion_resize_and_validation(tmp_path):
    image = source(tmp_path)
    mask_path = tmp_path / "mask.png"
    mask = Image.new("L", (32, 16), 0)
    mask.paste(255, (0, 0, 16, 16))
    mask.save(mask_path)
    captured = []
    def handler(wire):
        captured.append(wire)
        return httpx.Response(200, json=response())
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        providers = create_providers(config(), http_client=client)
        providers.image.edit(ImageEditRequest("Fill", (image,), tmp_path / "out", mask=ImageInput(mask_path)))
    files = {filename: payload for _, filename, payload in multipart(captured[0]) if filename}
    with Image.open(io.BytesIO(files["input_0.png"])) as actual:
        assert actual.size == (1792, 1024)
    with Image.open(io.BytesIO(files["mask.png"])) as actual:
        assert actual.mode == "RGBA" and actual.size == (1792, 1024)
        assert actual.getpixel((0, 0)) == (255, 255, 255, 0)
        assert actual.getpixel((1791, 0)) == (255, 255, 255, 255)


@pytest.mark.parametrize("problem", ["empty", "multi", "rgb", "size", "gray", "all-editable", "all-immutable", "lost-after-resize"])
def test_malformed_masks_fail_before_dispatch(tmp_path, problem):
    inputs = (source(tmp_path),)
    path = tmp_path / "mask.png"
    mask = Image.new("L", (32, 16), 0)
    mask.paste(255, (0, 0, 16, 16))
    if problem == "empty":
        inputs = ()
    elif problem == "multi":
        inputs += (source(tmp_path, "second"),)
    elif problem == "rgb":
        mask = mask.convert("RGB")
    elif problem == "size":
        mask = mask.resize((16, 16))
    elif problem == "gray":
        mask.putpixel((0, 0), 127)
    elif problem == "all-editable":
        mask.paste(255, (0, 0, 32, 16))
    elif problem == "all-immutable":
        mask.paste(0, (0, 0, 32, 16))
    elif problem == "lost-after-resize":
        mask.paste(0, (0, 0, 32, 16))
        mask.putpixel((0, 0), 255)
    mask.save(path)
    calls = []
    with httpx.Client(transport=httpx.MockTransport(lambda wire: calls.append(wire))) as client:
        providers = create_providers(config(), http_client=client)
        with pytest.raises(LocalCallError):
            providers.image.edit(ImageEditRequest("Fill", inputs, tmp_path / "out", mask=ImageInput(path), request_size="1x1" if problem == "lost-after-resize" else None))
    assert not calls


@pytest.mark.parametrize("patch,error", [
    ({"data": []}, "ImagePayloadInvalid"),
    ({"data": [{"url": "https://provider.invalid/image"}]}, "ImagePayloadInvalid"),
    ({"data": [{"b64_json": "not base64"}]}, "ImagePayloadInvalid"),
    ({"quality": "low"}, "ReturnedQualityMismatch"),
    ({"data": [{"b64_json": encoded((100, 100))}]}, "AspectMismatch"),
    ({"model": "wrong-model"}, "RouteMismatch"),
])
def test_image_response_rejections_do_not_write_or_retry(tmp_path, patch, error):
    calls = []
    def handler(wire):
        calls.append(wire)
        return httpx.Response(200, json=response(**patch))
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        providers = create_providers(config(), http_client=client)
        with pytest.raises(ProviderRefusal, match=error):
            providers.image.edit(ImageEditRequest("Edit", (source(tmp_path),), tmp_path / "out"))
    assert len(calls) == 1 and not (tmp_path / "out").exists()


@pytest.mark.parametrize("usage", [{"input_tokens": 100, "output_tokens": 20},
    {"input_tokens": 12, "output_tokens": 20, "input_tokens_details": {"text_tokens": 1, "image_tokens": 1}},
    {"input_tokens": 12, "output_tokens": 20, "input_tokens_details": {"text_tokens": 3, "image_tokens": 9},
     "output_tokens_details": {"image_tokens": 19}}])
def test_missing_or_inconsistent_image_usage_is_unknown(tmp_path, usage):
    with httpx.Client(transport=httpx.MockTransport(lambda wire: httpx.Response(200, json=response(usage=usage)))) as client:
        providers = create_providers(config(), http_client=client)
        answer = providers.image.edit(ImageEditRequest("Edit", (source(tmp_path),), tmp_path / "out"))
    assert answer.receipt.usage is None and answer.receipt.usage_status == "unknown"

def _profile(non_srgb):
    profile = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    if non_srgb:
        for index in range(struct.unpack_from(">I", profile, 128)[0]):
            name, offset, size = struct.unpack_from(">4sII", profile, 132 + index * 12)
            if name == b"rTRC":
                struct.pack_into(">i", profile, offset + 12, round(1.8 * 65536))
    return bytes(profile)


def _metadata_image(path, fmt, non_srgb):
    image = Image.frombytes("RGB", (32, 16), bytes((index * 29 % 256 for index in range(32 * 16 * 3))))
    exif = Image.Exif(); exif[270] = "procedural-private-metadata-canary"
    image.save(path, format=fmt, icc_profile=_profile(non_srgb), exif=exif, **({"lossless": True} if fmt == "WEBP" else {}))
    with Image.open(path) as opened:
        if not non_srgb:
            return opened.convert("RGB")
        return ImageCms.profileToProfile(opened, ImageCms.ImageCmsProfile(io.BytesIO(opened.info["icc_profile"])),
                                         ImageCms.createProfile("sRGB"), outputMode="RGB")


def _assert_clean(encoded, expected):
    with Image.open(io.BytesIO(encoded)) as decoded:
        assert decoded.mode == "RGB" and decoded.tobytes() == expected.tobytes()
        assert decoded.info == {} and not len(decoded.getexif())
    assert b"procedural-private-metadata-canary" not in encoded


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
@pytest.mark.parametrize("non_srgb", [False, True])
def test_uploads_and_endpoint_strip_metadata_and_convert_color_once(tmp_path, fmt, non_srgb):
    source = tmp_path / ("source." + fmt.lower())
    expected = _metadata_image(source, fmt, non_srgb)
    clean, record = load_rgb_image(source)
    assert record["color_conversion"] == ("embedded_icc_to_srgb" if non_srgb else "srgb_identity")
    assert record["decoded_rgb_pixels_changed"] == non_srgb and clean.tobytes() == expected.tobytes()
    _assert_clean(encode_rgb_png(clean)[0], expected)
    messages, _ = build_messages(ChatRequest("test", "Unchanged system", (ImageInput(source),), {}, SCHEMA))
    _assert_clean(base64.b64decode(messages[1]["content"][1]["image_url"]["url"].split(",", 1)[1]), expected)
    captured, answer = [], {"model": "test-image", "quality": "high", "data": [{"b64_json": base64.b64encode(source.read_bytes()).decode()}]}
    with httpx.Client(transport=httpx.MockTransport(lambda wire: captured.append(wire) or httpx.Response(200, json=answer))) as client:
        providers = create_providers(config(), http_client=client)
        result = providers.image.edit(ImageEditRequest("Unchanged edit", (ImageInput(source),),
            tmp_path / "out", request_size="32x16", output_size=(32, 16)))
    uploads = [payload for _, filename, payload in multipart(captured[0]) if filename]
    assert len(uploads) == 1
    _assert_clean(uploads[0], expected)
    _assert_clean(result.image.read_bytes(), expected)
    records = result.receipt.to_record()["image_conversions"]
    assert [item["stage"] for item in records] == ["input", "provider_endpoint"]
    assert "procedural-private-metadata-canary" not in json.dumps(records)


def test_invalid_icc_fails_ingestion_without_copying_profile():
    image = Image.new("RGB", (2, 2))
    image.info["icc_profile"] = b"procedural-private-metadata-canary"
    with pytest.raises(ValueError, match="color profile") as error:
        prepare_rgb_image(image)
    assert "procedural-private" not in str(error.value)
