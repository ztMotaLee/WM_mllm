"""Exercise real SDK streaming and logging boundaries with isolated test inputs."""
from __future__ import annotations

from dataclasses import dataclass
import io
import json
import logging
from types import SimpleNamespace

import httpx
from PIL import Image
import pytest

from wmmllm.agent import bootstrap, executor
from wmmllm.config import AppConfig, ChatConfig, ImageConfig, RetryConfig
from wmmllm.logging_utils import protect_provider_logs
from wmmllm.memory.observer import ObserverDeps, observe_state
from wmmllm.providers import ChatRequest, create_providers
from wmmllm.providers.errors import ProviderError, TerminalQuotaError
from wmmllm.providers.retry import TERMINAL_QUOTA_CODES, raise_if_fatal_provider_error
from wmmllm.session.io import file_ref, write_once


SCHEMA = {"type": "json_schema", "json_schema": {"name": "decision", "strict": True,
    "schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}}
SECRET = "opaque-extra-procedural-credential"
REMOTE_DETAIL = "private-stream-detail-canary"


def _config():
    return AppConfig(ChatConfig(SECRET, "test-chat", base_url="https://provider.invalid/v1",
                               transport="responses_stream"),
                     ImageConfig(SECRET, "test-image", base_url="https://provider.invalid/v1"),
                     transport=RetryConfig(max_attempts=3, base_delay_seconds=.001,
                                           max_delay_seconds=.001, jitter=0))


def _wire(event, code):
    error = {"code": code, "message": REMOTE_DETAIL + " " + SECRET}
    packet = ({"type": "error", **error, "param": None} if event == "error" else
              {"type": "response.failed", "response": {"id": "test-response", "model": "test-chat",
               "status": "failed", "error": error, "output": [],
               "usage": {"input_tokens": 5, "output_tokens": 0, "total_tokens": 5,
                         "input_tokens_details": {"cached_tokens": 0},
                         "output_tokens_details": {"reasoning_tokens": 0}}}})
    return ("event: " + event + "\ndata: " + json.dumps(packet) + "\n\n").encode()


def _call_site(site, providers, config, tmp_path, monkeypatch):
    def complete(*args, **kwargs):
        return providers.chat.complete(ChatRequest("test", "Unchanged system", (), {}, SCHEMA))

    if site == "direct":
        return complete()
    seed = tmp_path / "seed.png"
    Image.new("RGB", (16, 8)).save(seed)
    if site == "observer":
        ref = file_ref(seed, run_root=tmp_path)
        deps = ObserverDeps(providers.chat, {"chat": 2}, 1, (0, 1), ("source", "endpoint"),
                            "Unchanged system", SCHEMA, lambda: 0., tmp_path)
        return observe_state(task_id="test", segment=1, caption="Observe", start=ref,
                             endpoint=ref, known_subjects={}, deps=deps,
                             out_dir=tmp_path / "observation")
    if site.startswith("bootstrap"):
        deps = SimpleNamespace(config=config, run_root=tmp_path)
        row = {"key": "fixture", "task_id": "test", "initial_image": seed,
               "segment": 1, "current_caption": "Look left."}
        monkeypatch.setattr(bootstrap, "camera_path", lambda *a, **kw: {"camera_path": []})
        monkeypatch.setattr(bootstrap, "_initial", lambda *a, **kw: {"context": {}, "panorama_state": {}})
        monkeypatch.setattr(bootstrap.geo, "load_panorama_state", lambda *a, **kw: {})
        gates = []

        def gate(*args, **kwargs):
            gates.append(True)
            if site == "bootstrap_final_gate" and len(gates) > 1:
                return complete()
            return False, {"verdict": "FAIL", "detail": "",
                           "repair_views": [{"yaw": 45., "pitch": 0.}]}, {"metrics": []}

        def repair(*args, **kwargs):
            if site == "bootstrap_repair":
                return complete()
            raise ProviderError("BadRequestError", http_status=400)

        monkeypatch.setattr(bootstrap, "gate", gate)
        monkeypatch.setattr(bootstrap, "repair", repair)
        return bootstrap.initialize(row, tmp_path, width=128, deps=deps)
    assert site == "executor_gate"
    write_once(tmp_path / "bootstrap/RESULT.json", {"seam_gate": {"planned_path": {"camera_path": []}}},
               run_root=tmp_path)
    current = {"pose": {"yaw": 0., "pitch": 0.}, "panorama_state": {}, "panorama": {},
               "seam_status": "resolved"}
    monkeypatch.setattr(bootstrap, "gate", complete)
    return executor.check_actual_reachable_pose({"segment": 1}, tmp_path / "turn",
        SimpleNamespace(current_node=current), {}, {"yaw": 90., "pitch": 0.},
        deps=SimpleNamespace(run_root=tmp_path))


@pytest.mark.parametrize("event", ["error", "response.failed"])
@pytest.mark.parametrize("code", sorted(TERMINAL_QUOTA_CODES))
@pytest.mark.parametrize("site", ["direct", "observer", "bootstrap_repair", "bootstrap_final_gate", "executor_gate"])
def test_streamed_terminal_quota_propagates_without_retry_or_optional_commit(
        tmp_path, monkeypatch, event, code, site):
    calls, responses, sleeps = [], [], []
    monkeypatch.setattr("wmmllm.providers.openai_compatible.time.sleep", sleeps.append)

    def handler(request):
        calls.append(request)
        response = httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_wire(event, code))
        responses.append(response)
        return response

    config = _config()
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        with create_providers(config, http_client=http) as providers:
            with pytest.raises(TerminalQuotaError) as caught:
                _call_site(site, providers, config, tmp_path, monkeypatch)
    error = caught.value
    assert error.error_class == code and error.kind == "terminal_quota"
    assert len(calls) == 1 and sleeps == [] and all(response.is_closed for response in responses)
    receipt = error.receipt.to_record()
    assert receipt["status"] == "error" and receipt["error_class"] == code
    assert len(receipt["attempts"]) == 1
    assert SECRET not in json.dumps(receipt) and REMOTE_DETAIL not in json.dumps(receipt)
    assert SECRET not in str(error) and REMOTE_DETAIL not in str(error)
    assert not hasattr(error, "body") and not hasattr(error, "raw")
    if event == "response.failed":
        assert receipt["returned_model"] == "test-chat"
        assert receipt["usage"]["input_tokens"] == 5
        assert receipt["response_sha256"]
    assert not (tmp_path / "fixture/bootstrap/COMMITTED.json").exists()
    assert not (tmp_path / "turn/ACTUAL_REACHABLE_GATE.json").exists()
    assert not (tmp_path / "observation/OBSERVATION.json").exists()


@pytest.mark.parametrize("event", ["error", "response.failed"])
@pytest.mark.parametrize("code", [None, "unknown_provider_code"])
def test_stream_error_prose_and_unknown_codes_do_not_become_terminal_quota(event, code):
    # The known code appears only in prose, which never determines classification.
    wire = _wire(event, code).replace(REMOTE_DETAIL.encode(), b"insufficient_quota")
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=wire))) as http:
        with create_providers(_config(), http_client=http) as providers:
            with pytest.raises(ProviderError) as caught:
                providers.chat.complete(ChatRequest("test", "Unchanged system", (), {}, SCHEMA))
    assert not isinstance(caught.value, TerminalQuotaError)
    assert raise_if_fatal_provider_error(caught.value) is None
    assert "unknown_provider_code" not in str(caught.value)


@pytest.mark.parametrize("code", sorted(TERMINAL_QUOTA_CODES))
@pytest.mark.parametrize("tail", ["read_error", "invalid_event"])
def test_failed_quota_event_is_fatal_before_reading_a_broken_stream_tail(monkeypatch, code, tail):
    calls, consumed_tail, closed, sleeps = [], [], [], []
    monkeypatch.setattr("wmmllm.providers.openai_compatible.time.sleep", sleeps.append)

    class Wire(httpx.SyncByteStream):
        def __iter__(self):
            yield _wire("response.failed", code)
            consumed_tail.append(True)
            if tail == "read_error":
                raise httpx.ReadError(REMOTE_DETAIL)
            yield b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta"}\n\n'

        def close(self):
            closed.append(True)

    def handler(request):
        calls.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Wire())

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        with create_providers(_config(), http_client=http) as providers:
            with pytest.raises(TerminalQuotaError) as caught:
                providers.chat.complete(ChatRequest("test", "Unchanged system", (), {}, SCHEMA))
    assert caught.value.error_class == code
    assert len(calls) == 1 and not sleeps and not consumed_tail and closed
    assert caught.value.receipt.returned_model == "test-chat"
    assert caught.value.receipt.usage.input_tokens == 5
    assert caught.value.receipt.response_sha256


@dataclass
class _Details:
    api_key: str
    nested: object = None


@pytest.mark.parametrize("kind", ["request", "dataclass", "mapping", "exception", "nested", "opaque",
                                 "cyclic_dataclass", "cyclic_mapping", "unprintable", "number_subclass"])
@pytest.mark.parametrize("logger_name", ["openai.extra_regression.transport", "wmmllm.extra_regression"])
def test_later_handlers_receive_sanitized_opaque_extras(kind, logger_name):
    protect_provider_logs((SECRET,))
    request = httpx.Request("GET", "https://provider.invalid/private?token=" + SECRET)

    class Opaque:
        def __str__(self):
            return "opaque=" + SECRET

    class Unprintable:
        def __str__(self):
            raise ValueError(SECRET)

    class Number(int):
        def __str__(self):
            return SECRET

    cyclic = _Details(SECRET)
    cyclic.nested = cyclic
    mapping = {"api_key": SECRET}
    mapping["self"] = mapping
    values = {"request": request, "dataclass": _Details(SECRET, request),
              "mapping": {"api_key": SECRET, "request": request},
              "exception": RuntimeError(SECRET, request),
              "nested": {"objects": [_Details(SECRET), RuntimeError(SECRET, request)]},
              "opaque": Opaque(), "cyclic_dataclass": cyclic, "cyclic_mapping": mapping,
              "unprintable": Unprintable(), "number_subclass": Number(1)}
    root = logging.getLogger()
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(logging.Formatter("%(message)s %(context)s"))
    # Install both the logger and its handler after protection is configured.
    root.addHandler(handler)
    logger = logging.getLogger(logger_name)
    previous_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        logger.warning("Ordinary warning", extra={"context": values[kind]})
    finally:
        logger.setLevel(previous_level)
        root.removeHandler(handler)
        handler.close()
    text = output.getvalue()
    assert text and SECRET not in text and "https://provider.invalid/" not in text
    if kind == "exception":
        assert "RuntimeError" in text and "Request(" not in text
    if kind.startswith("cyclic"):
        assert "[REDACTED_RECURSION]" in text
    if kind == "unprintable":
        assert "[REDACTED_OBJECT]" in text
