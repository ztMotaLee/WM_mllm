"""Construct only real, explicitly configured providers; construction makes no call."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from ..budget import BudgetController
from ..logging_utils import get_logger, protect_provider_logs
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

    def __enter__(self) -> "Providers":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def create_providers(config: Any, *, http_client: httpx.Client | None = None,
                     budget: Any = None, logger: Any = None) -> Providers:
    """Inject an httpx client for tests; its lifecycle remains with its owner.

    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_ORG_ID, OPENAI_PROJECT_ID and
    OPENAI_CUSTOM_HEADERS cannot supply or override these settings. Empty
    organization/project are passed explicitly. SDK max_retries is always zero.
    """
    # Validate both arms before creating either SDK client.
    for name, prefix in (("chat", "WMM_LLM"), ("image", "WMM_IMAGE")):
        settings = getattr(config, name)
        if not settings.api_key:
            raise ValueError("Missing %s_API_KEY for %s provider; no request was sent." % (prefix, name))
        if not settings.model:
            raise ValueError("Missing %s_MODEL; choose a model supported by your %s endpoint; no request was sent." % (prefix, name))
    if config.chat.transport not in ("chat_completions", "responses_stream"):
        raise ValueError("Unknown chat transport; choose chat_completions or responses_stream")
    if config.video.backend != "none":
        raise ValueError("Video backend must be none; generated transition video is unavailable.")
    secrets = tuple(value for settings in (config.chat, config.image)
                    for value in (settings.api_key, settings.base_url, settings.organization, settings.project)
                    if value)
    protect_provider_logs(secrets=secrets)
    logger = logger if logger is not None else get_logger("wmmllm.providers", secrets=secrets)
    budget = budget if budget is not None else BudgetController(config.budget)
    chat = OpenAIChatClient(config.chat, config.transport, http_client=http_client, budget=budget, logger=logger)
    try:
        image = OpenAIImageEditClient(config.image, config.transport, http_client=http_client, budget=budget, logger=logger)
    except Exception:
        chat.close()
        raise
    chat._secrets = image._secrets = secrets
    return Providers(chat, image, NoneVideoBackend())
