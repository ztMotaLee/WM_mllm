"""End-to-end receipt/log canaries with synthetic credentials and mocked HTTP."""
from __future__ import annotations

import json
import logging
import traceback

import httpx
import pytest

from wmmllm.config import load_config
from wmmllm.logging_utils import get_logger
from wmmllm.providers import create_providers
from wmmllm.providers.errors import RouteMismatch
from wmmllm.providers.types import ChatRequest


def test_cross_provider_credentials_never_enter_diagnostics(caplog):
    chat_key = "canary-llm-only-value-5342"
    image_key = "canary-image-only-value-1950"
    chat_url = "https://canary-user:canary-password@example.invalid/v1?token=canary-chat-query"
    image_url = "https://image-user:image-password@example.invalid/v1?access_token=canary-image-query"
    config = load_config(environ={
        "WMM_LLM_API_KEY": chat_key,
        "WMM_LLM_BASE_URL": chat_url,
        "WMM_LLM_MODEL": "test-chat",
        "WMM_IMAGE_API_KEY": image_key,
        "WMM_IMAGE_BASE_URL": image_url,
        "WMM_IMAGE_MODEL": "test-image",
    }, overrides={"transport.max_attempts": 1})

    def respond(request):
        # A provider returning an unexpected model can echo another arm's key.
        return httpx.Response(200, json={
            "id": "test-response", "object": "chat.completion", "created": 1,
            "model": image_key,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "{}"}, "finish_reason": "stop"}],
        })

    request = ChatRequest(role="planner", system="test instructions", images=(), payload={}, reasoning_effort=chat_key,
                          response_format={"type": "json_schema", "json_schema": {
                              "name": "test", "strict": True,
                              "schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
                          }})
    caplog.set_level(logging.DEBUG)
    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        with create_providers(config, http_client=http) as providers:
            with pytest.raises(RouteMismatch) as captured:
                providers.chat.complete(request)
    error = captured.value
    assert error.receipt is not None
    logger = get_logger("wmmllm.canary.integration", (chat_key, image_key, chat_url, image_url))
    logger.error("canary %s %s %s %s", chat_key, image_key, chat_url, image_url)
    texts = [str(error), repr(error.receipt), json.dumps(error.receipt.to_record()),
             json.dumps(config.safe_dict()), repr(config), caplog.text,
             "".join(traceback.format_exception(type(error), error, error.__traceback__))]
    for text in texts:
        for value in (chat_key, image_key, chat_url, image_url, "canary-password", "image-password", "canary-chat-query", "canary-image-query"):
            assert value not in text
