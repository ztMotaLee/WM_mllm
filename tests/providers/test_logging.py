"""Synthetic canaries; these values are test data and never contact a service."""
from __future__ import annotations

from dataclasses import dataclass
import io
import json
import logging

from wmmllm.logging_utils import get_logger, redact, safe_error


KEY = "canary-key-4938-not-a-real-credential"
URL = "https://example-user:example-password@example.invalid/v1?token=canary-query"


def test_redaction_covers_explicit_key_url_authorization_and_query():
    raw = f"{KEY} {URL} Authorization: Bearer {KEY}\n?token=canary-query"
    output = redact(raw, (KEY,))
    for forbidden in (KEY, URL, "example-password", "canary-query"):
        assert forbidden not in output


def test_log_formatting_and_exception_traceback_are_filtered():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger = get_logger("wmmllm.canary", (KEY,))
    logger.addHandler(handler)
    old_level, old_propagate = logger.level, logger.propagate
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        try:
            raise ValueError(f"{KEY} {URL}")
        except ValueError:
            logger.exception("provider failed: %s %s", KEY, URL)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        logger.propagate = old_propagate
    output = stream.getvalue()
    assert "ValueError" in output; assert KEY not in output; assert URL not in output
    assert "example-password" not in output


def test_error_description_never_copies_provider_message():
    error = RuntimeError(f"private prompt {KEY} {URL}")
    assert safe_error(error) == "RuntimeError"


def test_cross_provider_credentials_never_enter_diagnostics(caplog):
    import traceback, httpx, pytest
    from wmmllm.config import load_config
    from wmmllm.providers import ChatRequest, RouteMismatch, create_providers
    chat_key, image_key = "canary-llm-only-value-5342", "canary-image-only-value-1950"
    chat_url = "https://canary-user:canary-password@example.invalid/v1?token=canary-chat-query"
    image_url = "https://image-user:image-password@example.invalid/v1?access_token=canary-image-query"
    config = load_config(environ={"WMM_LLM_API_KEY": chat_key, "WMM_LLM_BASE_URL": chat_url, "WMM_LLM_MODEL": "test-chat",
                                  "WMM_IMAGE_API_KEY": image_key, "WMM_IMAGE_BASE_URL": image_url, "WMM_IMAGE_MODEL": "test-image"},
                         overrides={"transport": {"max_attempts": 1}})
    # A provider returning an unexpected model can echo another arm's key.
    answer = {"id": "x", "object": "chat.completion", "created": 1, "model": image_key,
              "choices": [{"index": 0, "message": {"role": "assistant", "content": "{}"}, "finish_reason": "stop"}]}
    schema = {"type": "json_schema", "json_schema": {"name": "test", "strict": True, "schema": {
        "type": "object", "properties": {}, "required": [], "additionalProperties": False}}}
    caplog.set_level(logging.DEBUG)
    with httpx.Client(transport=httpx.MockTransport(lambda wire: httpx.Response(200, json=answer))) as http:
        providers = create_providers(config, http_client=http)
        with pytest.raises(RouteMismatch) as captured:
            providers.chat.complete(ChatRequest("planner", "test", (), {}, schema, reasoning_effort=chat_key))
    error = captured.value
    get_logger("wmmllm.canary.integration").error("canary %s %s %s %s", chat_key, image_key, chat_url, image_url)
    texts = [str(error), repr(error.receipt), json.dumps(error.receipt.to_record()), json.dumps(config.safe_dict()),
             repr(config), caplog.text, "".join(traceback.format_exception(type(error), error, error.__traceback__))]
    for text in texts:
        for value in (chat_key, image_key, chat_url, image_url, "canary-password", "image-password"):
            assert value not in text


@dataclass
class _Details:
    api_key: str
    nested: object = None


def test_late_child_loggers_and_handlers_receive_sanitized_records():
    import httpx
    get_logger(secrets=(KEY,))
    request = httpx.Request("GET", "https://provider.invalid/private?token=" + KEY)

    class Unprintable:
        def __str__(self):
            raise ValueError(KEY)

    cyclic = _Details(KEY)
    cyclic.nested = cyclic
    extras = [request, _Details(KEY, request), {"api_key": KEY, "request": request},
              RuntimeError(KEY, request), cyclic, Unprintable()]
    root, stream = logging.getLogger(), io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s %(custom)s"))
    root.addHandler(handler)
    logger = logging.getLogger("openai.late_child.transport")
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        logger.debug("private-payload-canary %s", KEY, extra={"custom": KEY})
        try:
            raise RuntimeError("private-body-canary " + KEY)
        except RuntimeError:
            logger.exception("failure", extra={"custom": KEY})
        for extra in extras:
            logging.getLogger("wmmllm.late_child").warning("Ordinary warning", extra={"custom": extra})
    finally:
        logger.setLevel(old_level)
        root.removeHandler(handler)
    output = stream.getvalue()
    assert "Provider transport diagnostic" in output and "[REDACTED_RECURSION]" in output
    assert "[REDACTED_OBJECT]" in output and "RuntimeError" in output
    for forbidden in (KEY, "private-payload-canary", "private-body-canary", "https://provider.invalid/"):
        assert forbidden not in output
