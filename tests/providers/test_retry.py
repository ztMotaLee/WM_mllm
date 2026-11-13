"""Retry and spend reservations use in-process MockTransport only."""
import io
import base64
from dataclasses import replace
from email.utils import formatdate

import httpx
import pytest
from PIL import Image

from wmmllm.budget import BudgetController, BudgetExceeded, PriceBoundExceeded
from wmmllm.config import (AppConfig, BudgetConfig, ChatConfig, ImageConfig,
                           PriceConfig, RetryConfig)
from wmmllm.providers import (ChatRequest, ImageEditRequest, ImageInput,
                             ProviderRefusal, TransportExhausted, create_providers)
from wmmllm.providers.retry import RetryPolicy, retry_after_seconds

SCHEMA = {"type": "json_schema", "json_schema": {"name": "decision", "strict": True,
          "schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}}


def config(budget=None):
    return AppConfig(ChatConfig("test-secret", "test-chat", base_url="https://provider.invalid/v1"),
                     ImageConfig("test-secret", "test-image", base_url="https://provider.invalid/v1"),
                     transport=RetryConfig(jitter=0), budget=budget or BudgetConfig())


def request():
    return ChatRequest("planner", "System", (), {"action": "test"}, SCHEMA)


def completion(usage=None):
    return {"id": "test-id", "object": "chat.completion", "created": 0, "model": "test-chat",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "{}"}, "finish_reason": "stop"}], "usage": usage}


def prices(bound=1.0, **rates):
    return {"chat": PriceConfig(upper_bound_usd=bound, **rates), "image": PriceConfig(upper_bound_usd=bound)}


def test_source_backoff_cap_jitter_and_retry_after():
    error = RuntimeError()
    error.retry_after = 0
    policy = RetryPolicy()
    assert [policy.delay(i, error, rng=lambda a, b: 1) for i in range(1, 7)] == [20, 40, 80, 160, 160, 160]
    assert policy.delay(1, error, rng=lambda a, b: a) == 16
    assert policy.delay(1, error, rng=lambda a, b: b) == 24
    error.retry_after = 200
    assert policy.delay(1, error, rng=lambda a, b: 1) == 200
    error.retry_after = formatdate(1020, usegmt=True)
    assert retry_after_seconds(error, now=lambda: 1000) == 20
    nested = RuntimeError()
    nested.body = {"headers": {"Retry-After": "91"}}
    error.__cause__ = nested
    assert retry_after_seconds(error, now=lambda: 1000) == 91


def test_transport_retry_has_no_sdk_retry_and_preserves_attempt_receipts(monkeypatch):
    sleeps, calls = [], []
    monkeypatch.setattr("wmmllm.providers.openai_compatible.time.sleep", sleeps.append)
    def handler(wire):
        calls.append(wire)
        if len(calls) < 3:
            return httpx.Response(429, headers={"retry-after": "25"}, json={"error": {"message": "retry later"}})
        return httpx.Response(200, json=completion())
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with create_providers(config(), http_client=client) as providers:
            result = providers.chat.complete(request())
    assert len(calls) == 3 and sleeps == [25, 40]
    assert [a.status for a in result.receipt.attempts] == ["error", "error", "complete"]
    assert [a.http_status for a in result.receipt.attempts] == [429, 429, None]


def test_five_attempt_exhaustion(monkeypatch):
    sleeps, calls = [], []
    monkeypatch.setattr("wmmllm.providers.openai_compatible.time.sleep", sleeps.append)
    def handler(wire):
        calls.append(wire)
        raise httpx.ReadTimeout("synthetic secret transport body", request=wire)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with create_providers(config(), http_client=client) as providers:
            with pytest.raises(TransportExhausted) as failure:
                providers.chat.complete(request())
    assert len(calls) == 5 and sleeps == [20, 40, 80, 160]
    assert len(failure.value.receipt.attempts) == 5
    assert "synthetic secret" not in str(failure.value)


def test_4xx_is_not_retried(monkeypatch):
    sleeps, calls = [], []
    monkeypatch.setattr("wmmllm.providers.openai_compatible.time.sleep", sleeps.append)
    def handler(wire):
        calls.append(wire)
        return httpx.Response(422, json={"error": {"message": "unsupported effort"}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with create_providers(config(), http_client=client) as providers:
            with pytest.raises(ProviderRefusal):
                providers.chat.complete(request())
    assert len(calls) == 1 and sleeps == []


def test_unknown_usage_is_charged_bound_and_next_call_is_blocked():
    budget = BudgetController(BudgetConfig(max_cost_usd=1, prices=prices()))
    calls = []
    def handler(wire):
        calls.append(wire)
        return httpx.Response(200, json=completion())
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with create_providers(config(), http_client=client, budget=budget) as providers:
            result = providers.chat.complete(request())
            with pytest.raises(BudgetExceeded):
                providers.chat.complete(request())
    assert len(calls) == 1 and result.receipt.attempts[0].charged_usd == 1
    assert budget.spent_usd == 1


def test_cap_blocks_initial_call_before_dispatch():
    budget = BudgetController(BudgetConfig(max_cost_usd=.5, prices=prices()))
    calls = []
    with httpx.Client(transport=httpx.MockTransport(lambda wire: calls.append(wire))) as client:
        with create_providers(config(), http_client=client, budget=budget) as providers:
            with pytest.raises(BudgetExceeded):
                providers.chat.complete(request())
    assert not calls


def test_retry_reservation_block_keeps_already_paid_attempt_receipt(monkeypatch):
    budget = BudgetController(BudgetConfig(max_cost_usd=1, prices=prices()))
    calls, sleeps = [], []
    monkeypatch.setattr("wmmllm.providers.openai_compatible.time.sleep", sleeps.append)
    def handler(wire):
        calls.append(wire)
        return httpx.Response(503, json={"error": {"message": "retry"}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with create_providers(config(), http_client=client, budget=budget) as providers:
            with pytest.raises(BudgetExceeded) as failure:
                providers.chat.complete(request())
    assert len(calls) == 1 and budget.spent_usd == 1
    assert failure.value.receipt.attempts[0].charged_usd == 1
    assert failure.value.receipt.status == "error"


def test_price_bound_exceeded_after_dispatch_keeps_actual_response_receipt():
    budget = BudgetController(BudgetConfig(max_cost_usd=100, prices=prices(.01, input_usd_per_million=1, output_usd_per_million=1)))
    cfg = config()
    cfg = replace(cfg, chat=replace(cfg.chat, max_output_tokens=1))
    def handler(wire):
        return httpx.Response(200, json=completion({"prompt_tokens": 20000, "completion_tokens": 1}))
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with create_providers(cfg, http_client=client, budget=budget) as providers:
            with pytest.raises(PriceBoundExceeded) as failure:
                providers.chat.complete(request())
    receipt = failure.value.receipt
    assert receipt.returned_model == "test-chat" and receipt.usage.input_tokens == 20000
    assert receipt.response_sha256 and receipt.error_class == "PriceBoundExceeded"
    assert receipt.attempts[0].charged_usd == pytest.approx(.020001)


def test_image_transport_recreates_consumed_multipart_streams(tmp_path, monkeypatch):
    path = tmp_path / "source.png"
    Image.new("RGB", (16, 8), "red").save(path)
    stream = io.BytesIO()
    Image.new("RGB", (1792, 1024), "green").save(stream, format="PNG")
    payload = base64.b64encode(stream.getvalue()).decode()
    parts, sleeps = [], []
    monkeypatch.setattr("wmmllm.providers.openai_compatible.time.sleep", sleeps.append)
    def handler(wire):
        parts.append(wire.content)
        if len(parts) == 1:
            return httpx.Response(500, json={"error": {"message": "retry"}})
        return httpx.Response(200, json={"model": "test-image", "quality": "high", "data": [{"b64_json": payload}]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with create_providers(config(), http_client=client) as providers:
            result = providers.image.edit(ImageEditRequest("Edit", (ImageInput(path),), tmp_path / "out"))
    assert len(parts) == 2 and all(b"input_0.png" in body and b"PNG" in body for body in parts)
    assert len(result.receipt.attempts) == 2 and sleeps == [20]
