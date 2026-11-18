"""Small logging boundary that never persists provider payloads or credentials.

Pass only event names, counters, timing, hashes and sanitized receipts to logs.
Provider exception bodies must be replaced by ``safe_error`` before logging.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import fields, is_dataclass
import logging
import re
import threading
from typing import Any
from urllib.parse import quote, quote_plus


_URL = re.compile(r"(?:https?|wss?)://[^\s<>\"']+", re.IGNORECASE)
_BEARER = re.compile(r"\bBearer\s+[^\s,;\"'}\]]+", re.IGNORECASE)
_AUTHORIZATION = re.compile(r"\bAuthorization\s*[:=]\s*[^\r\n]+", re.IGNORECASE)
_ASSIGNMENT = re.compile(
    r"\b(?:[A-Z0-9_]*API[_-]?KEY|ACCESS[_-]?TOKEN|CLIENT[_-]?SECRET|PASSWORD)"
    r"\s*[\"']?\s*[:=]\s*[\"']?[^\s,;\"'}\]]+", re.IGNORECASE
)
_QUERY_TOKEN = re.compile(
    r"([?&](?:api[_-]?key|token|access_token|sig|signature|x-amz-signature)=)[^\s&#\"']+",
    re.IGNORECASE,
)
_KEY_SIGNATURE = re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{12,}\b")
_SECRET_FIELD = re.compile(
    r"^(?:.*api[_-]?key|authorization|access[_-]?token|refresh[_-]?token|client[_-]?secret|password|secret|credentials?)$",
    re.IGNORECASE,
)


def _secret_variants(secrets: Iterable[str]) -> tuple[str, ...]:
    values: set[str] = set()
    for secret in secrets:
        if not isinstance(secret, str) or not secret:
            continue
        values.update((secret, quote(secret, safe=""), quote_plus(secret, safe="")))
    return tuple(sorted(values, key=len, reverse=True))


def redact(text: Any, secrets: Iterable[str] = ()) -> str:
    """Remove supplied secrets, auth values and complete URLs from text."""
    value = str(text)
    for secret in _secret_variants(secrets):
        value = value.replace(secret, "[REDACTED]")
    value = _AUTHORIZATION.sub("Authorization: [REDACTED]", value)
    value = _BEARER.sub("Bearer [REDACTED]", value)
    value = _ASSIGNMENT.sub("[REDACTED_CREDENTIAL]", value)
    value = _URL.sub("[REDACTED_URL]", value)
    value = _QUERY_TOKEN.sub(r"\1[REDACTED]", value)
    return _KEY_SIGNATURE.sub("[REDACTED]", value)


def sanitize(value: Any, secrets: Iterable[str] = ()) -> Any:
    """Return JSON-compatible sanitized data; do not pass raw requests here."""
    secrets = tuple(secrets)
    ancestors: set[int] = set()

    def clean(item: Any, depth: int = 0) -> Any:
        if depth >= 64:
            return "[REDACTED_NESTING]"
        if isinstance(item, BaseException):
            return safe_error(item, secrets)
        dataclass = is_dataclass(item) and not isinstance(item, type)
        if dataclass or isinstance(item, (Mapping, list, tuple)):
            identity = id(item)
            if identity in ancestors:
                return "[REDACTED_RECURSION]"
            ancestors.add(identity)
            try:
                if dataclass:
                    item = {f.name: getattr(item, f.name) for f in fields(item)}
                if isinstance(item, Mapping):
                    return {redact(k, secrets): "[REDACTED]" if _SECRET_FIELD.fullmatch(str(k))
                            else clean(v, depth + 1) for k, v in item.items()}
                return [clean(v, depth + 1) for v in item]
            finally:
                # Dataclasses were expanded above; remove their original ID.
                ancestors.remove(identity)
        if isinstance(item, str):
            return redact(item, secrets)
        if item is None or type(item) in (bool, int, float):
            return item
        return redact(item, secrets)

    try:
        return clean(value)
    except Exception:
        # Opaque extras may have a broken string conversion or mapping access.
        return "[REDACTED_OBJECT]"


def safe_error(error: BaseException, secrets: Iterable[str] = ()) -> str:
    """Describe an error without forwarding a remote body, URL or request text."""
    name = type(error).__name__
    status = getattr(error, "status_code", getattr(error, "http_status", None))
    suffix = f" (HTTP {status})" if isinstance(status, int) and 100 <= status <= 599 else ""
    return redact(name + suffix, secrets)


class RedactingFilter(logging.Filter):
    """Sanitize before any handler sees a record, including captured exceptions."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self.secrets = tuple(secrets)

    def filter(self, record: logging.LogRecord) -> bool:
        # SDK debug records can contain full requests and returned payloads.
        provider_record = record.name in ("openai", "httpx", "httpcore") or record.name.startswith(
            ("openai.", "httpx.", "httpcore.")
        )
        if provider_record:
            if record.levelno < logging.WARNING:
                return False
            record.msg = "Provider transport diagnostic; details omitted"
            record.args = ()
        try:
            message = record.getMessage()
        except Exception:
            message = "Log formatting failed"
        record.msg = redact(message, self.secrets)
        record.args = ()
        if record.exc_info:
            record.exc_text = safe_error(record.exc_info[1], self.secrets)
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = redact(record.exc_text, self.secrets)
        if record.stack_info:
            record.stack_info = None
        for key, value in list(vars(record).items()):
            if key not in ("msg", "args", "exc_info", "exc_text", "stack_info"):
                if _SECRET_FIELD.fullmatch(key):
                    setattr(record, key, "[REDACTED]")
                else:
                    setattr(record, key, sanitize(value, self.secrets))
        return True


_DISPATCH_LOCK = threading.RLock()
_DISPATCH_FILTER = RedactingFilter()
_DISPATCH_INSTALLED = False


def get_logger(name: str = "wmmllm", secrets: Iterable[str] = ()) -> logging.Logger:
    """Return a logger whose records are redacted at dispatch; no handlers or levels change.

    The filter runs in Logger.callHandlers, which also covers SDK, child loggers
    and handlers added later, and suppresses SDK payload diagnostics.
    """
    global _DISPATCH_INSTALLED
    with _DISPATCH_LOCK:
        _DISPATCH_FILTER.secrets = tuple(set(_DISPATCH_FILTER.secrets) | set(secrets))
        if not _DISPATCH_INSTALLED:
            previous = logging.Logger.callHandlers

            def protected_dispatch(logger: logging.Logger, record: logging.LogRecord) -> None:
                if _DISPATCH_FILTER.filter(record):
                    previous(logger, record)

            logging.Logger.callHandlers = protected_dispatch
            _DISPATCH_INSTALLED = True
    return logging.getLogger(name)
