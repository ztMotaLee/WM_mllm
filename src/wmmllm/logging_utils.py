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

    def __init__(self, secrets: Iterable[str] = (), *, suppress_payloads: bool = False) -> None:
        super().__init__()
        self.secrets = tuple(secrets)
        self.suppress_payloads = suppress_payloads

    def filter(self, record: logging.LogRecord) -> bool:
        # SDK debug records can contain full requests and returned payloads.
        provider_record = record.name in ("openai", "httpx", "httpcore") or record.name.startswith(
            ("openai.", "httpx.", "httpcore.")
        )
        if self.suppress_payloads and provider_record:
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


def _attach(logger: logging.Logger, secrets: Iterable[str], *, suppress_payloads: bool = False) -> None:
    secrets = tuple(secrets)
    for current in logger.filters:
        if isinstance(current, RedactingFilter):
            current.secrets = tuple(set(current.secrets) | set(secrets))
            current.suppress_payloads = current.suppress_payloads or suppress_payloads
            return
    logger.addFilter(RedactingFilter(secrets, suppress_payloads=suppress_payloads))


def get_logger(name: str = "wmmllm", secrets: Iterable[str] = ()) -> logging.Logger:
    """Return a filtered logger without installing handlers or changing levels."""
    logger = logging.getLogger(name)
    _attach(logger, secrets)
    return logger


def protect_provider_logs(secrets: Iterable[str] = ()) -> None:
    """Suppress SDK payload diagnostics and redact emitted transport warnings.

    The factory invokes this after importing the SDK, before client construction.
    Filtering runs at dispatch so later child loggers and handlers are covered.
    """
    _protect_log_dispatch(secrets)
    names = {"openai", "httpx", "httpcore"}
    names.update(name for name in logging.Logger.manager.loggerDict
                 if name.startswith(("openai.", "httpx.", "httpcore.")))
    for name in names:
        _attach(logging.getLogger(name), secrets, suppress_payloads=True)
    for handler in logging.getLogger().handlers:
        # Handler filtering also covers later-created SDK child loggers.
        if not any(isinstance(f, RedactingFilter) for f in handler.filters):
            handler.addFilter(RedactingFilter(secrets, suppress_payloads=True))
        else:
            for current in handler.filters:
                if isinstance(current, RedactingFilter):
                    current.secrets = tuple(set(current.secrets) | set(secrets))
                    current.suppress_payloads = True


_DISPATCH_LOCK = threading.RLock()
_DISPATCH_FILTER = RedactingFilter(suppress_payloads=True)
_DISPATCH_INSTALLED = False


def _protect_log_dispatch(secrets: Iterable[str]) -> None:
    """Guard Logger dispatch, which also precedes handlers created later.

    Ancestor logger filters do not see propagated child records. A handler
    filter only covers handlers that already exist. Dispatch filtering sees
    the complete record, including extra fields, before any handler uses it.
    The existing dispatch function and unrelated logging behavior are retained.
    """
    global _DISPATCH_INSTALLED
    with _DISPATCH_LOCK:
        _DISPATCH_FILTER.secrets = tuple(set(_DISPATCH_FILTER.secrets) | set(secrets))
        if _DISPATCH_INSTALLED:
            return
        previous = logging.Logger.callHandlers

        def protected_dispatch(logger: logging.Logger, record: logging.LogRecord) -> None:
            if _DISPATCH_FILTER.filter(record):
                previous(logger, record)

        logging.Logger.callHandlers = protected_dispatch
        _DISPATCH_INSTALLED = True
