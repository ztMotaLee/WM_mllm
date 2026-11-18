"""Explicit, side-effect-free provider configuration.

Credentials come only from WMM_* process variables or an explicitly selected
local env file. Importing this module never reads files or constructs a client.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlsplit

import yaml


class ConfigError(ValueError):
    """A local configuration failure whose message never contains input values."""


@dataclass(frozen=True, repr=False)
class ChatConfig:
    api_key: str = field(repr=False)
    model: str
    base_url: str = field(default="", repr=False)
    organization: str | None = field(default=None, repr=False)
    project: str | None = field(default=None, repr=False)
    transport: str = "chat_completions"
    reasoning_effort: str = "high"
    max_output_tokens: int = 32000
    timeout_seconds: float = 600.0

    def __post_init__(self) -> None:
        _validate_endpoint(self, "chat")
        if self.transport != "chat_completions":
            raise ConfigError("chat.transport must be chat_completions; responses_stream is no longer supported.")
        if self.reasoning_effort not in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
            raise ConfigError("Unsupported chat.reasoning_effort; select an explicit supported setting.")
        _integer(self.max_output_tokens, "chat.max_output_tokens")
        _number(self.timeout_seconds, "chat.timeout_seconds", positive=True)

    def __repr__(self) -> str:
        return "ChatConfig(credentials=<redacted>, model=<configured>)"


@dataclass(frozen=True, repr=False)
class ImageConfig:
    api_key: str = field(repr=False)
    model: str
    base_url: str = field(default="", repr=False)
    organization: str | None = field(default=None, repr=False)
    project: str | None = field(default=None, repr=False)
    quality: str = "high"
    view_request_size: str = "1792x1024"
    panorama_request_size: str = "2048x1024"
    view_output_size: tuple[int, int] = (1344, 768)
    panorama_output_size: tuple[int, int] = (2048, 1024)
    timeout_seconds: float = 420.0

    def __post_init__(self) -> None:
        _validate_endpoint(self, "image")
        if self.quality not in ("low", "medium", "high", "auto"):
            raise ConfigError("Unsupported image.quality.")
        for size in (self.view_request_size, self.panorama_request_size):
            if not isinstance(size, str) or not re.fullmatch(r"[1-9][0-9]*x[1-9][0-9]*", size):
                raise ConfigError("Image request sizes must use WIDTHxHEIGHT.")
        object.__setattr__(self, "view_output_size", _size(self.view_output_size, "image.view_output_size"))
        object.__setattr__(self, "panorama_output_size", _size(self.panorama_output_size, "image.panorama_output_size"))
        _number(self.timeout_seconds, "image.timeout_seconds", positive=True)

    def __repr__(self) -> str:
        return "ImageConfig(credentials=<redacted>, model=<configured>)"


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 5
    base_delay_seconds: float = 20.0
    max_delay_seconds: float = 160.0
    jitter: float = 0.2

    def __post_init__(self) -> None:
        _integer(self.max_attempts, "transport.max_attempts")
        _number(self.base_delay_seconds, "transport.base_delay_seconds", positive=True)
        _number(self.max_delay_seconds, "transport.max_delay_seconds", positive=True)
        if _number(self.jitter, "transport.jitter", positive=False) > 1:
            raise ConfigError("transport.jitter must be between zero and one.")


@dataclass(frozen=True)
class PriceConfig:
    """User-provided prices; the upper bound covers one dispatched attempt."""
    upper_bound_usd: float
    input_usd_per_million: float | None = None
    output_usd_per_million: float | None = None
    text_input_usd_per_million: float | None = None
    image_input_usd_per_million: float | None = None
    image_output_usd_per_million: float | None = None
    cached_input_usd_per_million: float | None = None

    def __post_init__(self) -> None:
        _number(self.upper_bound_usd, "budget price upper_bound_usd", positive=True)
        for name in _PRICE_FIELDS - {"upper_bound_usd"}:
            value = getattr(self, name)
            if value is not None:
                _number(value, "budget token price", positive=False)


@dataclass(frozen=True)
class BudgetConfig:
    max_cost_usd: float | None = None
    prices: Mapping[str, PriceConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_cost_usd is not None:
            _number(self.max_cost_usd, "budget.max_cost_usd", positive=False)
        if any(role not in ("chat", "image") for role in self.prices):
            raise ConfigError("Budget prices must use chat and image entries.")
        if self.max_cost_usd is not None and set(self.prices) != {"chat", "image"}:
            raise ConfigError("budget.max_cost_usd requires user prices and per-attempt upper bounds for chat and image; no request was sent.")
        object.__setattr__(self, "prices", MappingProxyType(dict(self.prices)))


@dataclass(frozen=True)
class VideoConfig:
    backend: str = "none"
    num_frames: int = 124
    fps: float = 24.0
    size: tuple[int, int] = (1344, 768)

    def __post_init__(self) -> None:
        if self.backend != "none":
            raise ConfigError("video.backend must be none; generated video is unavailable in this package.")
        _integer(self.num_frames, "video.num_frames")
        if self.num_frames < 2:
            raise ConfigError("video.num_frames must be at least two.")
        _number(self.fps, "video.fps", positive=True)
        object.__setattr__(self, "size", _size(self.size, "video.size"))


@dataclass(frozen=True, repr=False)
class AppConfig:
    chat: ChatConfig
    image: ImageConfig
    transport: RetryConfig = field(default_factory=RetryConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    video: VideoConfig = field(default_factory=VideoConfig)

    def safe_dict(self) -> dict[str, Any]:
        """Return public settings, omitting credentials, URLs and user identifiers.

        Arbitrary model identifiers are deliberately represented by presence;
        individual call receipts record model attribution after redaction.
        """
        return {
            "chat": {"model_configured": bool(self.chat.model),
                     "transport": self.chat.transport,
                     "reasoning_effort": self.chat.reasoning_effort,
                     "max_output_tokens": self.chat.max_output_tokens,
                     "timeout_seconds": self.chat.timeout_seconds},
            "image": {"model_configured": bool(self.image.model),
                      "quality": self.image.quality,
                      "view_request_size": self.image.view_request_size,
                      "panorama_request_size": self.image.panorama_request_size,
                      "view_output_size": list(self.image.view_output_size),
                      "panorama_output_size": list(self.image.panorama_output_size),
                      "timeout_seconds": self.image.timeout_seconds},
            "transport": {"max_attempts": self.transport.max_attempts,
                          "base_delay_seconds": self.transport.base_delay_seconds,
                          "max_delay_seconds": self.transport.max_delay_seconds,
                          "jitter": self.transport.jitter},
            "budget": {"max_cost_usd": self.budget.max_cost_usd,
                       "priced_roles": sorted(self.budget.prices)},
            "video": {"backend": self.video.backend, "num_frames": self.video.num_frames,
                      "fps": self.video.fps, "size": list(self.video.size)},
        }

    def __repr__(self) -> str:
        return f"AppConfig({self.safe_dict()!r})"

    def secrets(self) -> tuple[str, ...]:
        """Configured credentials and endpoint identities, for redaction only."""
        return tuple(value for settings in (self.chat, self.image)
                     for value in (settings.api_key, settings.base_url, settings.organization, settings.project) if value)


_PRICE_FIELDS = frozenset({"upper_bound_usd", "input_usd_per_million", "output_usd_per_million", "cached_input_usd_per_million",
                          "text_input_usd_per_million", "image_input_usd_per_million", "image_output_usd_per_million"})
_ENDPOINT_FIELDS = frozenset({"base_url", "model", "organization", "project"})
_CHAT_FIELDS = _ENDPOINT_FIELDS | {"transport", "reasoning_effort", "max_output_tokens", "timeout_seconds"}
_IMAGE_FIELDS = _ENDPOINT_FIELDS | {"quality", "view_request_size", "panorama_request_size", "view_output_size", "panorama_output_size", "timeout_seconds"}
_ALLOWED = {"chat": _CHAT_FIELDS, "image": _IMAGE_FIELDS,
            "transport": {"max_attempts", "base_delay_seconds", "max_delay_seconds", "jitter"},
            "budget": {"max_cost_usd", "prices", "price_table_file"},
            "video": {"backend", "num_frames", "fps", "size"}}
_ENV_NAMES = frozenset(f"WMM_{role}_{name}" for role in ("LLM", "IMAGE")
                       for name in ("API_KEY", "BASE_URL", "MODEL", "ORGANIZATION", "PROJECT"))


def _number(value: Any, name: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigError(f"{name} must be a finite number.")
    if value < 0 or (positive and value == 0):
        raise ConfigError(f"{name} must be {'positive' if positive else 'nonnegative'}.")
    return float(value)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer.")
    return value


def _size(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{name} must contain width and height.")
    return (_integer(value[0], name), _integer(value[1], name))



_MISSING_KEY = {"chat": "Missing WMM_LLM_API_KEY for chat provider; set the process environment or an explicitly selected env file; no request was sent.",
                "image": "Missing WMM_IMAGE_API_KEY for image provider; no request was sent."}


def _validate_endpoint(config: ChatConfig | ImageConfig, role: str) -> None:
    if not isinstance(config.api_key, str) or not config.api_key.strip():
        raise ConfigError(_MISSING_KEY[role])
    if not isinstance(config.model, str) or not config.model.strip():
        prefix = "WMM_LLM" if role == "chat" else "WMM_IMAGE"
        raise ConfigError(f"Missing {prefix}_MODEL; choose a model supported by your endpoint; no request was sent.")
    if not isinstance(config.base_url, str) or not config.base_url.strip():
        prefix = "WMM_LLM" if role == "chat" else "WMM_IMAGE"
        raise ConfigError(f"Missing {prefix}_BASE_URL; set the base URL of your own provider endpoint; no request was sent.")
    for name in ("api_key", "base_url", "model", "organization", "project"):
        value = getattr(config, name)
        if value is not None and (not isinstance(value, str) or "\n" in value or "\r" in value):
            raise ConfigError("Endpoint settings must not contain line breaks.")
    if not re.match(r"^https?://", config.base_url, re.IGNORECASE):
        raise ConfigError("Provider base_url must use HTTP or HTTPS.")

def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        raise ConfigError(f"Unable to read {label}; check the explicitly selected file.") from None
    if data is None:
        return {}
    if not isinstance(data, dict) or any(not isinstance(key, str) for key in data):
        raise ConfigError(f"{label} must be a mapping.")
    return data


def _reject_credentials(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(part in normalized for part in ("api_key", "access_token", "secret", "password", "authorization")):
                raise ConfigError("Inline credentials are forbidden in YAML and CLI overrides; use WMM_* environment variables or an explicitly selected env file.")
            _reject_credentials(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_credentials(item)


def _validate_mapping(data: Mapping[str, Any]) -> None:
    _reject_credentials(data)
    if any(name not in _ALLOWED and name != "env_file" for name in data):
        raise ConfigError("Unknown top-level configuration field.")
    for name, allowed in _ALLOWED.items():
        block = data.get(name, {})
        if not isinstance(block, Mapping) or any(key not in allowed for key in block):
            raise ConfigError(f"Unknown or invalid {name} configuration field.")
        if name in ("chat", "image") and isinstance(block.get("base_url"), str):
            try:
                url = urlsplit(block["base_url"])
                query_names = {key.lower().replace("-", "_") for key, _ in parse_qsl(url.query)}
                sensitive = {"api_key", "access_token", "token", "key", "authorization", "password", "secret", "signature", "sig"}
                has_credentials = url.username is not None or url.password is not None or bool(query_names & sensitive)
            except ValueError:
                raise ConfigError("Invalid provider base_url in YAML or CLI overrides.") from None
            if has_credentials:
                raise ConfigError("Credential-bearing base_url values are forbidden in YAML and CLI overrides; use WMM_* environment variables or an explicitly selected env file.")


def _override_mapping(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    if overrides is None:
        return {}
    if not isinstance(overrides, Mapping):
        raise ConfigError("CLI overrides must be a mapping.")
    if any(not isinstance(key, str) for key in overrides):
        raise ConfigError("CLI override names must be strings.")
    result = {key: dict(value) if isinstance(value, Mapping) else value for key, value in overrides.items()}
    _validate_mapping(result)
    return result


def _env_values(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        raise ConfigError("Unable to read the explicitly selected env file.") from None
    values: dict[str, str] = {}
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        name, value = text.split("=", 1)
        name = name.strip()
        if name.startswith("export "):
            name = name[7:].strip()
        if name not in _ENV_NAMES:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in (chr(34), chr(39)):
            value = value[1:-1]
        values[name] = value
    return values


def _explicit_path(value: Any, parent: Path, label: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise ConfigError(f"{label} must be an explicitly selected path.")
    path = Path(value).expanduser()
    return path if path.is_absolute() else parent / path


def load_config(config_path: str | os.PathLike[str] | None = None, *,
                env_file: str | os.PathLike[str] | None = None,
                overrides: Mapping[str, Any] | None = None,
                environ: Mapping[str, str] | None = None) -> AppConfig:
    """Resolve defaults < YAML < WMM environment < CLI nonsecret overrides.

    Process-variable presence precedes the explicitly selected env file,
    including an explicitly empty value. Each empty image endpoint field then inherits its chat
    counterpart. Relative YAML file references use the YAML file's directory;
    explicit ``env_file`` arguments use the current directory. No file is
    searched for automatically, and unrelated OPENAI_* variables are ignored.
    """
    data = {} if config_path is None else _read_yaml(Path(config_path), "configuration YAML")
    _validate_mapping(data)
    cli = _override_mapping(overrides)
    parent = Path.cwd() if config_path is None else Path(config_path).parent
    if env_file is not None:
        env_path = _explicit_path(env_file, Path.cwd(), "env_file")
    else:
        env_path = _explicit_path(cli.get("env_file", data.get("env_file")), parent, "env_file")
    file_values = _env_values(env_path)
    process = os.environ if environ is None else environ
    values = {name: (process[name] if name in process else file_values.get(name, "")).strip() for name in _ENV_NAMES}
    blocks = {name: dict(data.get(name, {})) for name in _ALLOWED}
    for role, prefix in (("chat", "WMM_LLM"), ("image", "WMM_IMAGE")):
        for field_name in _ENDPOINT_FIELDS:
            variable = f"{prefix}_{field_name.upper()}"
            if variable in process or variable in file_values:
                blocks[role][field_name] = values[variable]
    for name in _ALLOWED:
        blocks[name].update(cli.get(name, {}))
    for field_name in _ENDPOINT_FIELDS:
        value = blocks["chat"].get(field_name)
        if value is not None and not isinstance(value, str):
            raise ConfigError("Chat endpoint settings must be strings.")
        blocks["chat"][field_name] = (value or "").strip() or ("" if field_name == "base_url" else None)
        value = blocks["image"].get(field_name)
        if value is not None and not isinstance(value, str):
            raise ConfigError("Image endpoint settings must be strings.")
        blocks["image"][field_name] = (value or "").strip() or blocks["chat"][field_name]
    chat_key = values["WMM_LLM_API_KEY"]
    c = ChatConfig(api_key=chat_key, **blocks["chat"])
    i = ImageConfig(api_key=values["WMM_IMAGE_API_KEY"] or chat_key, **blocks["image"])
    t = RetryConfig(**blocks["transport"])
    b = blocks["budget"]
    prices = b.get("prices", {})
    price_file = _explicit_path(b.get("price_table_file"), parent, "budget.price_table_file")
    if price_file is not None:
        if prices:
            raise ConfigError("Choose budget.prices or budget.price_table_file, not both.")
        prices = _read_yaml(price_file, "user price table")
    if not isinstance(prices, Mapping):
        raise ConfigError("budget.prices must be a mapping.")
    _reject_credentials(prices)
    parsed_prices: dict[str, PriceConfig] = {}
    for role, price in prices.items():
        if role not in ("chat", "image") or not isinstance(price, Mapping) or set(price) - _PRICE_FIELDS or "upper_bound_usd" not in price:
            raise ConfigError("Each chat/image price requires upper_bound_usd and optional token rates.")
        parsed_prices[role] = PriceConfig(**price)
    budget = BudgetConfig(max_cost_usd=b.get("max_cost_usd"), prices=parsed_prices)
    return AppConfig(chat=c, image=i, transport=t, budget=budget, video=VideoConfig(**blocks["video"]))
