"""Multipart edits and source image acceptance checks, with no remote I/O."""
import base64
import io
from email.parser import BytesParser
from email.policy import default

import httpx
import pytest
from PIL import Image

from wmmllm.config import AppConfig, ChatConfig, ImageConfig, RetryConfig
from wmmllm.providers import ImageEditRequest, ImageInput, LocalCallError, ProviderRefusal, create_providers


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
        with create_providers(config(), http_client=client) as providers:
            result = providers.image.edit(ImageEditRequest("Keep action", inputs, tmp_path / "out", panorama=panorama))
    wire = captured[0]
    assert wire.url.path == "/v1/images/edits"
    parts = multipart(wire)
    text = {name: payload.decode() for name, filename, payload in parts if filename is None}
    assert text["model"] == "test-image" and text["prompt"] == "Keep action"
    assert text["size"] == size and text["quality"] == "high"
    assert "input_fidelity" not in text
    files = [(filename, payload) for name, filename, payload in parts if filename]
    assert [f[0] for f in files] == ["input_0.png", "input_1.png"]
    for (_, payload), color in zip(files, ((255, 0, 0), (0, 0, 255))):
        with Image.open(io.BytesIO(payload)) as image:
            assert image.size == (32, 16) and image.getpixel((0, 0)) == color
    with Image.open(result.image) as image:
        assert image.size == output
    assert result.returned_quality == "high" and result.usage.image_input_tokens == 9
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
        with create_providers(config(), http_client=client) as providers:
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
        with create_providers(config(), http_client=client) as providers:
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
        with create_providers(config(), http_client=client) as providers:
            with pytest.raises(ProviderRefusal, match=error):
                providers.image.edit(ImageEditRequest("Edit", (source(tmp_path),), tmp_path / "out"))
    assert len(calls) == 1 and not (tmp_path / "out").exists()


def test_missing_image_breakdown_is_explicitly_unknown(tmp_path):
    with httpx.Client(transport=httpx.MockTransport(lambda wire: httpx.Response(200, json=response(usage={"input_tokens": 100, "output_tokens": 20})))) as client:
        with create_providers(config(), http_client=client) as providers:
            answer = providers.image.edit(ImageEditRequest("Edit", (source(tmp_path),), tmp_path / "out"))
    assert answer.usage is None and answer.receipt.usage_status == "unknown"


@pytest.mark.parametrize("details", [{"text_tokens": 1, "image_tokens": 1}, {"text_tokens": 3, "image_tokens": 9}])
def test_inconsistent_image_usage_is_unknown(tmp_path, details):
    usage = {"input_tokens": 12, "output_tokens": 20, "input_tokens_details": details,
             "output_tokens_details": {"image_tokens": 19}}
    with httpx.Client(transport=httpx.MockTransport(lambda wire: httpx.Response(200, json=response(usage=usage)))) as client:
        with create_providers(config(), http_client=client) as providers:
            answer = providers.image.edit(ImageEditRequest("Edit", (source(tmp_path),), tmp_path / "out"))
    assert answer.usage is None
