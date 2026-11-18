"""Retry and spend reservations use in-process MockTransport only."""
import io
import base64
from dataclasses import replace
from email.utils import formatdate

import httpx
import pytest
from PIL import Image

import json

from wmmllm.budget import BudgetController, BudgetError, BudgetExceeded, PriceBoundExceeded
from wmmllm.config import (AppConfig, BudgetConfig, ChatConfig, ImageConfig,
                           PriceConfig, RetryConfig)
from wmmllm.memory.observer import ObserverDeps, observe_state
from wmmllm.providers import (ChatRequest, ImageEditRequest, ImageInput,
                             ProviderRefusal, TransportExhausted, create_providers)
from wmmllm.providers.errors import TerminalQuotaError, TransportError
from wmmllm.providers.retry import (TERMINAL_QUOTA_CODES, RetryPolicy, classify, raise_if_fatal_provider_error,
                                    retry_after_seconds, safe_provider_error)
from wmmllm.session.io import file_ref

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
        providers = create_providers(config(), http_client=client)
        result = providers.chat.complete(request())
    assert len(calls) == 3 and sleeps == [25, 40]
    assert [a.status for a in result.receipt.attempts] == ["error", "error", "complete"]
    assert [a.http_status for a in result.receipt.attempts] == [429, 429, None]


@pytest.mark.parametrize("failure", ["timeout", "422"])
def test_transport_exhaustion_and_4xx_is_not_retried(monkeypatch, failure):
    sleeps, calls = [], []
    monkeypatch.setattr("wmmllm.providers.openai_compatible.time.sleep", sleeps.append)
    def handler(wire):
        calls.append(wire)
        if failure == "422":
            return httpx.Response(422, json={"error": {"message": "unsupported effort"}})
        raise httpx.ReadTimeout("synthetic secret transport body", request=wire)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        providers = create_providers(config(), http_client=client)
        with pytest.raises(ProviderRefusal if failure == "422" else TransportExhausted) as caught:
            providers.chat.complete(request())
    if failure == "422":
        assert len(calls) == 1 and sleeps == []
        return
    assert len(calls) == 5 and sleeps == [20, 40, 80, 160] and len(caught.value.receipt.attempts) == 5
    assert "synthetic secret" not in str(caught.value)


@pytest.mark.parametrize("cap,status", [(1, 200), (1, 503), (.5, 200)])
def test_budget_gates_every_dispatch_and_keeps_paid_receipts(monkeypatch, cap, status):
    budget = BudgetController(BudgetConfig(max_cost_usd=cap, prices=prices()))
    calls, sleeps = [], []
    monkeypatch.setattr("wmmllm.providers.openai_compatible.time.sleep", sleeps.append)
    def handler(wire):
        calls.append(wire)
        return httpx.Response(status, json=completion() if status == 200 else {"error": {"message": "retry"}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        providers = create_providers(config(), http_client=client, budget=budget)
        if status == 200 and cap == 1:
            assert providers.chat.complete(request()).receipt.attempts[0].charged_usd == 1
        with pytest.raises(BudgetExceeded) as failure:
            providers.chat.complete(request())
    assert len(calls) == (0 if cap < 1 else 1) and budget.snapshot()["accounted_cost_usd"] == (0 if cap < 1 else 1)
    if status == 503:
        assert failure.value.receipt.attempts[0].charged_usd == 1 and failure.value.receipt.status == "error"


def test_price_bound_exceeded_after_dispatch_keeps_actual_response_receipt():
    budget = BudgetController(BudgetConfig(max_cost_usd=100, prices=prices(.01, input_usd_per_million=1, output_usd_per_million=1)))
    cfg = config()
    cfg = replace(cfg, chat=replace(cfg.chat, max_output_tokens=1))
    def handler(wire):
        return httpx.Response(200, json=completion({"prompt_tokens": 20000, "completion_tokens": 1}))
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        providers = create_providers(cfg, http_client=client, budget=budget)
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
        providers = create_providers(config(), http_client=client)
        result = providers.image.edit(ImageEditRequest("Edit", (ImageInput(path),), tmp_path / "out"))
    assert len(parts) == 2 and all(b"input_0.png" in body and b"PNG" in body for body in parts)
    assert len(result.receipt.attempts) == 2 and sleeps == [20]


@pytest.mark.parametrize("code", sorted(TERMINAL_QUOTA_CODES))
def test_terminal_quota_429_is_dispatched_once_and_receipt_is_sanitized(monkeypatch, code):
    calls, sleeps = [], []
    monkeypatch.setattr("wmmllm.providers.openai_compatible.time.sleep", sleeps.append)
    def handler(wire):
        calls.append(wire)
        return httpx.Response(429, json={"error": {"code": code, "message": "procedural-private-error-canary", "type": "quota"}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        providers = create_providers(config(), http_client=client)
        with pytest.raises(TerminalQuotaError) as error:
            providers.chat.complete(request())
    assert len(calls) == 1 and sleeps == [] and error.value.error_class == code
    assert len(error.value.receipt.attempts) == 1
    assert "procedural-private-error-canary" not in json.dumps(error.value.receipt.to_record())


def test_fatal_guard_keeps_budget_authorization_and_quota_but_handles_other_400(tmp_path):
    for status in (401, 403, 404):
        error = ProviderRefusal("SyntheticFailure", http_status=status)
        with pytest.raises(ProviderRefusal) as caught:
            raise_if_fatal_provider_error(error)
        assert caught.value is error
    with pytest.raises(BudgetError):
        raise_if_fatal_provider_error(BudgetError("Local bound"))
    quota = TransportError("insufficient_quota", http_status=429)
    assert isinstance(classify(quota), TerminalQuotaError)
    with pytest.raises(TransportError):
        raise_if_fatal_provider_error(quota)
    assert raise_if_fatal_provider_error(ProviderRefusal("OptionalRepairFailure", http_status=400)) is None
    assert safe_provider_error(ProviderRefusal("private text https://provider.invalid/")) == {"class": "ProviderError"}
    # The observer does not retry a terminal quota transport error.
    source = tmp_path / "source.png"
    Image.new("RGB", (4, 2), "blue").save(source)
    reference = file_ref(source, run_root=tmp_path)
    class Chat:
        route, calls = "test", 0
        def complete(self, request):
            self.calls += 1
            raise quota
    chat = Chat()
    deps = ObserverDeps(chat, tmp_path)
    with pytest.raises(TransportError):
        observe_state(task_id="test", segment=1, caption="Observe", start=reference, endpoint=reference,
                      known_subjects={}, deps=deps, out_dir=tmp_path / "observation")
    assert chat.calls == 1
