"""Typed real-provider adapters and the explicit no-generation video backend."""
from .errors import (LocalCallError, ProviderError, ProviderRefusal, RouteMismatch,
                     TerminalQuotaError, TokenLimitError, TransportError, TransportExhausted)
from .factory import Providers, create_providers
from .openai_compatible import OpenAIChatClient, OpenAIImageEditClient
from .types import (AttemptReceipt, CallReceipt, ChatClient, ChatRequest, ChatResponse,
                    ImageEditClient, ImageEditRequest, ImageEditResult, ImageInput,
                    PresentationResult, Usage, VideoBackend, VideoTransition)
from .video import NoneVideoBackend

__all__ = ["AttemptReceipt", "CallReceipt", "ChatClient", "ChatRequest", "ChatResponse",
           "ImageEditClient", "ImageEditRequest", "ImageEditResult", "ImageInput",
           "PresentationResult", "Usage", "VideoBackend", "VideoTransition", "Providers",
           "create_providers", "OpenAIChatClient", "OpenAIImageEditClient", "NoneVideoBackend",
           "ProviderError", "TransportError", "TransportExhausted", "TokenLimitError",
           "ProviderRefusal", "TerminalQuotaError", "RouteMismatch", "LocalCallError"]
