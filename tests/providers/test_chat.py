"""No-network transport contract tests using only synthetic inputs."""
import base64
import io
import json

import httpx
import pytest
from PIL import Image

from wmmllm.config import AppConfig, ChatConfig, ImageConfig, RetryConfig
from wmmllm.providers import ChatRequest, ImageInput, TokenLimitError, create_providers

SCHEMA = {"type": "json_schema", "json_schema": {"name": "decision", "strict": True,
    "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}},
               "required": ["ok"], "additionalProperties": False}}}


def config(**chat):
    return AppConfig(ChatConfig("test-chat-secret", "test-chat", base_url="https://provider.invalid/v1", **chat),
                     ImageConfig("test-image-secret", "test-image", base_url="https://provider.invalid/v1"),
                     transport=RetryConfig(jitter=0))


def request(images=(), **kwargs):
    return ChatRequest("planner", "System instructions", tuple(images), {"action": "café"}, SCHEMA, **kwargs)


def completion(**kwargs):
    value = {"id": "test-id", "object": "chat.completion", "created": 0, "model": "test-chat",
             "choices": [{"index": 0, "message": {"role": "assistant", "content": '{"ok": true}'},
                          "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 12, "completion_tokens": 5,
                       "prompt_tokens_details": {"cached_tokens": 2},
                       "completion_tokens_details": {"reasoning_tokens": 1}}}
    value.update(kwargs)
    return value


def test_multimodal_contract_and_attribution(tmp_path):
    images = []
    for name, color in (("source", "red"), ("reference", "blue")):
        path = tmp_path / (name + ".png")
        Image.new("RGB", (1800, 900), color).save(path)
        images.append(ImageInput(path, name))
    captured = []
    def handler(wire):
        captured.append((wire.url.path, json.loads(wire.content)))
        return httpx.Response(200, json=completion())
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        providers = create_providers(config(), http_client=client)
        answer = providers.chat.complete(request(images, reasoning_effort="xhigh", max_output_tokens=1234, timeout_seconds=45))
    path, body = captured[0]
    assert len(captured) == 1; assert answer.text == '{"ok": true}' and answer.model == "test-chat"
    assert answer.usage.input_tokens == 12 and answer.usage.reasoning_tokens == 1
    receipt = answer.receipt.to_record()
    assert receipt["attribution_status"] == receipt["usage_status"] == "known"
    assert len(receipt["image_sha256"]) == 2 and len(receipt["request_sha256"]) == 64
    assert "System instructions" not in json.dumps(receipt) and "data:image" not in json.dumps(receipt)
    assert not any(k in body for k in ("temperature", "top_p", "max_tokens"))
    assert path == "/v1/chat/completions" and receipt["transport"] == "chat_completions"
    assert body["reasoning_effort"] == "xhigh" and body["max_completion_tokens"] == 1234
    assert body["response_format"] == SCHEMA
    parts = body["messages"][1]["content"]
    urls = [parts[i]["image_url"]["url"] for i in (1, 3)]
    assert receipt["timing_scope"] == "unavailable" and receipt["first_byte_seconds"] is None
    assert parts[0]["text"] == "Image 1: source." and parts[2]["text"] == "Image 2: reference."
    assert parts[4]["text"] == json.dumps({"action": "café"}, ensure_ascii=False)
    for url, color in zip(urls, ((255, 0, 0), (0, 0, 255))):
        with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as image:
            assert image.size == (1024, 512) and image.getpixel((0, 0)) == color


def test_explicit_unknown_usage_and_attribution():
    def handler(wire):
        return httpx.Response(200, json=completion(model=None, usage=None))
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        providers = create_providers(config(), http_client=client)
        result = providers.chat.complete(request())
    assert result.model is None and result.usage is None
    assert result.receipt.attribution_status == result.receipt.usage_status == "unknown"


def test_token_limit_is_not_retried():
    calls = []
    def handler(wire):
        calls.append(wire)
        value = completion()
        value["choices"][0]["finish_reason"] = "length"
        return httpx.Response(200, json=value)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        providers = create_providers(config(), http_client=client)
        with pytest.raises(TokenLimitError):
            providers.chat.complete(request())
    assert len(calls) == 1


def test_ambient_openai_environment_cannot_override_explicit_settings(monkeypatch):
    for name, value in {"OPENAI_API_KEY": "ambient-secret", "OPENAI_BASE_URL": "https://ambient.invalid/v1",
        "OPENAI_ORG_ID": "ambient-org", "OPENAI_PROJECT_ID": "ambient-project", "OPENAI_ADMIN_KEY": "ambient-admin",
        "OPENAI_CUSTOM_HEADERS": "X-Ambient: should-not-be-sent\nAuthorization: Bearer ambient\nOpenAI-Organization: overwritten"}.items():
        monkeypatch.setenv(name, value)
    captured = []
    def handler(wire):
        captured.append(wire)
        return httpx.Response(200, json=completion())
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        providers = create_providers(config(), http_client=client)
        providers.chat.complete(request())
        assert providers.chat._client.max_retries == providers.image._client.max_retries == 0
    wire = captured[0]
    assert wire.url.host == "provider.invalid"; assert wire.headers["authorization"] == "Bearer test-chat-secret"
    assert wire.headers.get("openai-organization", "") == "" and wire.headers.get("openai-project", "") == ""
    assert "x-ambient" not in wire.headers
