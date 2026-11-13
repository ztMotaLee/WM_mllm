"""No-network transport contract tests using only synthetic inputs."""
import base64
import io
import json
import logging
import traceback
from dataclasses import replace

import httpx
import pytest
from PIL import Image

from wmmllm.config import AppConfig, ChatConfig, ImageConfig, RetryConfig
from wmmllm.providers import (ChatRequest, ImageInput, ProviderError, RouteMismatch,
                             TokenLimitError, create_providers)

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


def native(**kwargs):
    value = {"id": "test-response", "object": "response", "created_at": 0,
             "model": "test-chat", "status": "completed",
             "output": [{"type": "reasoning", "summary": []}, {"id": "message-id", "type": "message",
                         "role": "assistant", "status": "completed",
                         "content": [{"type": "output_text", "text": '{"ok": true}', "annotations": []}]}],
             "usage": {"input_tokens": 12, "output_tokens": 5, "total_tokens": 17,
                       "input_tokens_details": {"cached_tokens": 2},
                       "output_tokens_details": {"reasoning_tokens": 1}}}
    value.update(kwargs)
    return value


def sse(events):
    data = "".join("event: %s\ndata: %s\n\n" % (e["type"], json.dumps(e)) for e in events).encode()
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=httpx.ByteStream(data))


@pytest.mark.parametrize("transport", ["chat_completions", "responses_stream"])
def test_multimodal_contract_and_attribution(tmp_path, transport):
    images = []
    for name, color in (("source", "red"), ("reference", "blue")):
        path = tmp_path / (name + ".png")
        Image.new("RGB", (1800, 900), color).save(path)
        images.append(ImageInput(path, name))
    captured = []
    def handler(wire):
        captured.append((wire.url.path, json.loads(wire.content)))
        if transport == "chat_completions":
            return httpx.Response(200, json=completion())
        return sse([{"type": "response.created", "response": native(status="in_progress", output=[])},
                    {"type": "response.completed", "response": native()}])
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with create_providers(config(transport=transport), http_client=client) as providers:
            answer = providers.chat.complete(request(images, reasoning_effort="xhigh", max_output_tokens=1234, timeout_seconds=45))
    path, body = captured[0]
    assert len(captured) == 1
    assert answer.text == '{"ok": true}' and answer.model == "test-chat"
    assert answer.usage.input_tokens == 12 and answer.usage.reasoning_tokens == 1
    receipt = answer.receipt.to_record()
    assert receipt["attribution_status"] == receipt["usage_status"] == "known"
    assert len(receipt["image_sha256"]) == 2 and len(receipt["request_sha256"]) == 64
    assert "System instructions" not in json.dumps(receipt) and "data:image" not in json.dumps(receipt)
    assert not any(k in body for k in ("temperature", "top_p", "max_tokens"))
    if transport == "chat_completions":
        assert path == "/v1/chat/completions"
        assert body["reasoning_effort"] == "xhigh" and body["max_completion_tokens"] == 1234
        assert body["response_format"] == SCHEMA
        parts = body["messages"][1]["content"]
        urls = [parts[i]["image_url"]["url"] for i in (1, 3)]
    else:
        assert path == "/v1/responses"
        assert body["reasoning"] == {"effort": "xhigh"} and body["max_output_tokens"] == 1234
        assert body["text"]["format"] == {"type": "json_schema", **SCHEMA["json_schema"]}
        assert body["stream"] is True and body["background"] is False and body["store"] is False
        assert body["input"][0] == {"role": "system", "content": [{"type": "input_text", "text": "System instructions"}]}
        parts = body["input"][1]["content"]
        urls = [parts[i]["image_url"] for i in (1, 3)]
        assert receipt["timing_scope"] == "httpx.Response.iter_raw body bytes"
        assert receipt["first_byte_seconds"] is not None
    assert parts[0]["text"] == "Image 1: source." and parts[2]["text"] == "Image 2: reference."
    assert parts[4]["text"] == json.dumps({"action": "café"}, ensure_ascii=False)
    for url, color in zip(urls, ((255, 0, 0), (0, 0, 255))):
        with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as image:
            assert image.size == (1024, 512) and image.getpixel((0, 0)) == color


@pytest.mark.parametrize("transport", ["chat_completions", "responses_stream"])
def test_explicit_unknown_usage_and_attribution(transport):
    def handler(wire):
        if transport == "chat_completions":
            return httpx.Response(200, json=completion(model=None, usage=None))
        return sse([{"type": "response.completed", "response": native(model=None, usage=None)}])
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with create_providers(config(transport=transport), http_client=client) as providers:
            result = providers.chat.complete(request())
    assert result.model is None and result.usage is None
    assert result.receipt.attribution_status == result.receipt.usage_status == "unknown"


@pytest.mark.parametrize("events,error", [
    ([{"type": "response.output_text.delta", "delta": '{"ok": true}'}], "MissingTerminalEvent"),
    ([{"type": "response.completed", "response": native()}, {"type": "response.completed", "response": native()}], "InvalidTerminalEvent"),
    ([{"type": "response.completed", "response": native()}, {"type": "response.output_text.delta", "delta": "extra"}], "DeltaAfterTerminal"),
    ([{"type": "response.completed", "response": native(status="in_progress")}], "TerminalStatusMismatch"),
    ([{"type": "response.completed", "response": native(output=[{"type": "function_call", "name": "unexpected"}])}], "UnexpectedResponseOutput"),
    ([{"type": "response.completed", "response": native(output=[])}], "IncompleteAnswer"),
    ([{"type": "error", "message": "private payload"}], "ResponsesStreamError"),
])
def test_stream_requires_a_complete_terminal_answer(events, error):
    calls = []
    def handler(wire):
        calls.append(wire)
        return sse(events)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with create_providers(config(transport="responses_stream"), http_client=client) as providers:
            with pytest.raises(ProviderError, match=error) as failure:
                providers.chat.complete(request())
    assert len(calls) == 1 and failure.value.receipt.status == "error"


@pytest.mark.parametrize("transport", ["chat_completions", "responses_stream"])
def test_token_limit_is_not_retried(transport):
    calls = []
    def handler(wire):
        calls.append(wire)
        if transport == "chat_completions":
            value = completion()
            value["choices"][0]["finish_reason"] = "length"
            return httpx.Response(200, json=value)
        return sse([{"type": "response.incomplete", "response": native(status="incomplete", incomplete_details={"reason": "max_output_tokens"})}])
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with create_providers(config(transport=transport), http_client=client) as providers:
            with pytest.raises(TokenLimitError):
                providers.chat.complete(request())
    assert len(calls) == 1


def test_mismatched_model_is_not_retried():
    with httpx.Client(transport=httpx.MockTransport(lambda wire: httpx.Response(200, json=completion(model="another-model")))) as client:
        with create_providers(config(), http_client=client) as providers:
            with pytest.raises(RouteMismatch) as failure:
                providers.chat.complete(request())
    assert failure.value.receipt.returned_model == "another-model"
    assert failure.value.receipt.response_sha256 and failure.value.receipt.usage.input_tokens == 12


def test_missing_key_or_model_fails_before_client_construction():
    for arm, field in (("chat", "api_key"), ("chat", "model"), ("image", "api_key"), ("image", "model")):
        cfg = config()
        calls = []
        with httpx.Client(transport=httpx.MockTransport(lambda wire: calls.append(wire))) as client:
            with pytest.raises(ValueError, match="Missing WMM_"):
                cfg = replace(cfg, **{arm: replace(getattr(cfg, arm), **{field: ""})})
                create_providers(cfg, http_client=client)
        assert not calls


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
        with create_providers(config(), http_client=client) as providers:
            providers.chat.complete(request())
            assert providers.chat._client.max_retries == providers.image._client.max_retries == 0
    wire = captured[0]
    assert wire.url.host == "provider.invalid"
    assert wire.headers["authorization"] == "Bearer test-chat-secret"
    assert wire.headers.get("openai-organization", "") == "" and wire.headers.get("openai-project", "") == ""
    assert "x-ambient" not in wire.headers


def test_canary_never_reaches_receipts_errors_logs_or_repr(caplog):
    key = "unit-canary-secret-987654321"
    url = "https://canary-user:canary-password@provider.invalid/v1?token=canary-query"
    cfg = config()
    cfg = replace(cfg, chat=replace(cfg.chat, api_key=key, base_url=url))
    def handler(wire):
        return httpx.Response(400, json={"error": {"message": "Bearer " + key + " " + url, "type": "invalid_request_error"}})
    caplog.set_level(logging.DEBUG)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with create_providers(cfg, http_client=client) as providers:
            req = ChatRequest("planner", key + url, (), {"private": key}, SCHEMA)
            with pytest.raises(ProviderError) as failure:
                providers.chat.complete(req)
    evidence = " ".join([caplog.text, str(failure.value), repr(failure.value),
                         "".join(traceback.format_exception(failure.type, failure.value, failure.tb)),
                         repr(failure.value.receipt), json.dumps(failure.value.receipt.to_record()),
                         repr(req), repr(cfg), json.dumps(cfg.safe_dict())])
    for secret in (key, url, "canary-password", "canary-query"):
        assert secret not in evidence
