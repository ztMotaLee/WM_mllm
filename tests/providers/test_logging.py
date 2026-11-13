"""Synthetic canaries; these values are test data and never contact a service."""
from __future__ import annotations

from dataclasses import dataclass
import io
import json
import logging

from wmmllm.logging_utils import RedactingFilter, get_logger, redact, safe_error, sanitize


KEY = "canary-key-4938-not-a-real-credential"
URL = "https://example-user:example-password@example.invalid/v1?token=canary-query"


def test_redaction_covers_explicit_key_url_authorization_and_query():
    raw = f"{KEY} {URL} Authorization: Bearer {KEY}\n?token=canary-query"
    output = redact(raw, (KEY,))
    for forbidden in (KEY, URL, "example-password", "canary-query"):
        assert forbidden not in output


def test_recursive_dump_omits_secret_fields_and_urls():
    @dataclass
    class Credentials:
        api_key: str
        base_url: str

    output = json.dumps(sanitize({"auth": Credentials(KEY, URL), "input_tokens": 8}, (KEY,)))
    assert KEY not in output
    assert URL not in output
    assert "example-password" not in output
    assert '"input_tokens": 8' in output


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
    assert "ValueError" in output
    assert KEY not in output
    assert URL not in output
    assert "example-password" not in output


def test_error_description_never_copies_provider_message():
    error = RuntimeError(f"private prompt {KEY} {URL}")
    assert safe_error(error) == "RuntimeError"


def test_sdk_diagnostics_never_copy_payload_text():
    filter_ = RedactingFilter((KEY,), suppress_payloads=True)
    debug = logging.LogRecord("openai.transport", logging.DEBUG, "", 0, "private prompt", (), None)
    assert filter_.filter(debug) is False
    warning = logging.LogRecord("httpx", logging.WARNING, "", 0, f"private prompt {KEY} {URL}", (), None)
    assert filter_.filter(warning)
    assert "private prompt" not in warning.getMessage()
    assert KEY not in warning.getMessage()
    assert URL not in warning.getMessage()
