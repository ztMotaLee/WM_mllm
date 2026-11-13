"""Provider boundaries. Receipts contain metadata and hashes, never payload text."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class ImageInput:
    path: Path = field(repr=False)
    label: str = ""
    sha256: str | None = None


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int | None = 0
    reasoning_tokens: int = 0
    image_input_tokens: int = 0
    image_output_tokens: int = 0
    text_input_tokens: int = 0

    def to_record(self) -> dict[str, int | None]:
        return asdict(self)


@dataclass(frozen=True)
class AttemptReceipt:
    number: int
    status: Literal["complete", "error"]
    elapsed_seconds: float
    error_class: str | None = None
    http_status: int | None = None

    backoff_seconds: float = 0.0
    charged_usd: float | None = None


@dataclass(frozen=True)
class CallReceipt:
    kind: Literal["chat", "image"]
    role: str
    requested_model: str
    returned_model: str | None
    attribution_status: Literal["known", "unknown"]
    usage: Usage | None
    usage_status: Literal["known", "unknown"]
    status: Literal["complete", "error"]
    transport: str
    elapsed_seconds: float
    request_sha256: str
    response_sha256: str | None
    image_sha256: tuple[str, ...]
    attempts: tuple[AttemptReceipt, ...]
    reasoning_effort: str | None = None
    first_byte_seconds: float | None = None
    max_silence_seconds: float | None = None
    timing_scope: str = "unavailable"
    error_class: str | None = None
    http_status: int | None = None

    image_conversions: tuple[Mapping[str, Any], ...] = ()

    def to_record(self) -> dict[str, Any]:
        """A JSON-compatible record without request/response bodies or file paths."""
        return {"schema": "wmmllm.call_receipt.v1", **asdict(self)}


@dataclass(frozen=True)
class ChatRequest:
    role: str
    system: str = field(repr=False)
    images: tuple[ImageInput, ...] = field(repr=False)
    payload: Mapping[str, Any] = field(repr=False)
    response_format: Mapping[str, Any] = field(repr=False)
    reasoning_effort: str | None = None
    max_output_tokens: int | None = None
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class ChatResponse:
    text: str = field(repr=False)
    finish_reason: str
    model: str | None
    receipt: CallReceipt

    @property
    def usage(self) -> Usage | None:
        return self.receipt.usage


@dataclass(frozen=True)
class ImageEditRequest:
    prompt: str = field(repr=False)
    images: tuple[ImageInput, ...] = field(repr=False)
    output_dir: Path = field(repr=False)
    mask: ImageInput | None = field(default=None, repr=False)
    panorama: bool = False
    request_size: str | None = None
    output_size: tuple[int, int] | None = None
    quality: str | None = None
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class ImageEditResult:
    image: Path = field(repr=False)
    raw_size: tuple[int, int]
    returned_quality: str
    resize: Literal["none", "uniform_lanczos"]
    receipt: CallReceipt

    @property
    def usage(self) -> Usage | None:
        return self.receipt.usage


@runtime_checkable
class ChatClient(Protocol):
    route: str

    def complete(self, request: ChatRequest) -> ChatResponse: ...

    def close(self) -> None: ...


@runtime_checkable
class ImageEditClient(Protocol):
    route: str

    def edit(self, request: ImageEditRequest) -> ImageEditResult: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class VideoTransition:
    transition_id: str
    operation: str
    first_frame: Path
    last_frame: Path
    duration_seconds: float = 0.0
    output_path: Path | None = None


@dataclass(frozen=True)
class PresentationResult:
    status: Literal["rendered", "none", "generated"]
    video_path: Path | None
    backend: str
    reason: str | None = None


@runtime_checkable
class VideoBackend(Protocol):
    def present(self, transition: VideoTransition) -> PresentationResult: ...
