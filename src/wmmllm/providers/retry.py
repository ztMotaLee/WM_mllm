"""The five-attempt transport policy; semantic failures are never retried."""
from __future__ import annotations

import math
import random
import re
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Callable, Mapping

from ..budget import BudgetError
from .errors import ProviderError, ProviderRefusal, TerminalQuotaError, TransportError

TRANSPORT_CLASSES = frozenset({"APIConnectionError", "APITimeoutError", "InternalServerError",
    "RateLimitError", "ConnectionError", "ConnectTimeout", "ReadTimeout", "ReadError",
    "RemoteProtocolError", "CallInterrupted"})
TERMINAL_QUOTA_CODES = frozenset({"insufficient_quota", "billing_hard_limit_reached",
    "billing_not_active", "credit_balance_exhausted"})


def _exception_chain(error: BaseException):
    seen = set()
    for _ in range(8):
        if error is None or id(error) in seen:
            break
        seen.add(id(error))
        yield error
        error = error.__cause__ or error.__context__


def terminal_quota_code_from_payload(payload: Mapping) -> str | None:
    """Keep only allowlisted codes from an HTTP or streaming error payload."""
    values = [payload.get(name) for name in ("code", "type")]
    nested = payload.get("error")
    if isinstance(nested, Mapping):
        values.extend(nested.get(name) for name in ("code", "type"))
    return next((value for value in values
                 if isinstance(value, str) and value in TERMINAL_QUOTA_CODES), None)


def terminal_quota_code(error: BaseException) -> str | None:
    """Read only explicit machine codes; never infer quota from remote prose."""
    for current in _exception_chain(error):
        values = [getattr(current, name, None) for name in ("code", "type", "error_class")]
        for body in (getattr(current, "body", None), getattr(current, "raw", None)):
            if isinstance(body, Mapping):
                values.append(terminal_quota_code_from_payload(body))
        for value in values:
            if isinstance(value, str) and value in TERMINAL_QUOTA_CODES:
                return value
    return None


def safe_provider_error(error: BaseException) -> dict[str, str]:
    """Return a bounded local class and an allowlisted code, with no body."""
    name = getattr(error, "error_class", type(error).__name__)
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", name) is None:
        name = "ProviderError"
    result = {"class": name}
    code = terminal_quota_code(error)
    if code is not None:
        result["code"] = code
    return result


def raise_if_fatal_provider_error(error: BaseException) -> None:
    """Keep budget, authorization, missing-model and quota failures fatal."""
    if any(isinstance(current, (BudgetError, TerminalQuotaError)) or http_status(current) in (401, 403, 404)
           for current in _exception_chain(error)) or terminal_quota_code(error) is not None:
        raise error


def http_status(error: BaseException) -> int | None:
    for name in ("http_status", "status_code"):
        status = getattr(error, name, None)
        if type(status) is int:
            return status
    return None


def retry_after_seconds(error: BaseException, now: Callable[[], float] = time.time) -> float:
    """Numeric/date Retry-After, including wrapped SDK errors and gateway bodies."""
    values: list[object] = []
    seen: set[int] = set()
    current: BaseException | None = error
    for _ in range(8):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        values.append(getattr(current, "retry_after", None))
        for data in (getattr(current, "raw", None), getattr(current, "body", None)):
            if isinstance(data, dict):
                values.extend((data.get("retry_after"), data.get("retry-after")))
                for name in ("error", "headers"):
                    nested = data.get(name)
                    if isinstance(nested, dict):
                        values.extend(nested.get(k) for k in ("retry_after", "retry-after", "Retry-After"))
        headers = getattr(getattr(current, "response", None), "headers", None)
        if headers is None:
            headers = getattr(current, "headers", None)
        if headers is not None:
            values.append(headers.get("Retry-After"))
        # Read the message only to recover a numeric retry hint. Never store it.
        values.extend(re.findall(r"(?:retry_after|retry-after)[\"']?\s*[:=]\s*[\"']?([0-9]+(?:\.[0-9]+)?)", str(current), re.I))
        current = current.__cause__ or current.__context__
    delays: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            delay = float(value)
        except (TypeError, ValueError):
            try:
                delay = parsedate_to_datetime(str(value)).timestamp() - now()
            except (TypeError, ValueError, OverflowError):
                continue
        if math.isfinite(delay) and delay >= 0:
            delays.append(delay)
    return max(delays, default=0.0)


def classify(error: BaseException) -> ProviderError:
    quota_code = terminal_quota_code(error)
    if quota_code is not None:
        if isinstance(error, TerminalQuotaError):
            return error
        return TerminalQuotaError(quota_code, http_status=http_status(error),
                                  usage=getattr(error, "usage", None),
                                  receipt=getattr(error, "receipt", None))
    if isinstance(error, ProviderError):
        return error
    status = http_status(error)
    names = {cls.__name__ for cls in type(error).__mro__}
    kind = TransportError if names & TRANSPORT_CLASSES or status == 429 or (status is not None and 500 <= status < 600) else ProviderRefusal if status is not None and 400 <= status < 500 else ProviderError
    result = kind(safe_provider_error(error)["class"], http_status=status)
    result.retry_after = retry_after_seconds(error)
    return result


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    base_delay_seconds: float = 20.0
    max_delay_seconds: float = 160.0
    jitter: float = 0.2

    def delay(self, attempt: int, error: BaseException,
              rng: Callable[[float, float], float] = random.uniform) -> float:
        base = min(self.max_delay_seconds, self.base_delay_seconds * 2 ** (attempt - 1))
        return max(retry_after_seconds(error), base * rng(1 - self.jitter, 1 + self.jitter))
