"""Construct only real, explicitly configured providers; construction makes no call."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from ..budget import BudgetController
from ..logging_utils import get_logger
from .openai_compatible import OpenAIChatClient, OpenAIImageEditClient
from .types import ChatClient, ImageEditClient, VideoBackend
from .video import NoneVideoBackend


@dataclass
class Providers:
    chat: ChatClient
    image: ImageEditClient
    video: VideoBackend

    def close(self) -> None:
        self.chat.close()
        self.image.close()


def create_providers(config: Any, *, http_client: httpx.Client | None = None,
                     budget: Any = None, logger: Any = None) -> Providers:
    """Inject an httpx client for tests; its lifecycle remains with its owner.

    The validated AppConfig is the only source of keys, endpoints and models.
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_ORG_ID, OPENAI_PROJECT_ID and
    OPENAI_CUSTOM_HEADERS cannot supply or override these settings. Empty
    organization/project are passed explicitly. SDK max_retries is always zero.
    """
    secrets = config.secrets()
    # Installs the redacting log dispatch filter with every configured secret.
    provider_logger = get_logger("wmmllm.providers", secrets=secrets)
    logger = logger if logger is not None else provider_logger
    budget = budget if budget is not None else BudgetController(config.budget)
    options = {"secrets": secrets, "http_client": http_client, "budget": budget, "logger": logger}
    chat = OpenAIChatClient(config.chat, config.transport, **options)
    try:
        image = OpenAIImageEditClient(config.image, config.transport, **options)
    except Exception:
        chat.close()
        raise
    return Providers(chat, image, NoneVideoBackend())
