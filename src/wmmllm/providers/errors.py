"""Safe provider failures; remote messages and bodies are never retained."""
from __future__ import annotations

from .types import CallReceipt, Usage


class ProviderError(RuntimeError):
    kind = "provider"

    def __init__(self, error_class: str, *, http_status: int | None = None,
                 usage: Usage | None = None, receipt: CallReceipt | None = None) -> None:
        # error_class is always a local constant or an exception type name.
        message = "Provider call failed: " + error_class
        if http_status is not None:
            message += " (HTTP %d)" % http_status
        super().__init__(message)
        self.error_class = error_class
        self.http_status = http_status
        self.usage = usage
        self.receipt = receipt
        self.retry_after = 0.0


class TransportError(ProviderError):
    kind = "transport"


class TransportExhausted(TransportError):
    pass


class TokenLimitError(ProviderError):
    kind = "token_limit"


class ProviderRefusal(ProviderError):
    kind = "refusal"


class TerminalQuotaError(ProviderRefusal):
    """Billing or quota exhaustion that cannot be repaired by retrying."""
    kind = "terminal_quota"


class RouteMismatch(ProviderRefusal):
    pass


class LocalCallError(ProviderError):
    kind = "local"
