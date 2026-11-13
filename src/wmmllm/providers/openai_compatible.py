"""OpenAI-compatible chat and multipart image editing, with no model fallback.

The SDK retries are disabled. Its custom-header environment merge is cleared
before use; keys, endpoint, organization and project are always explicit.
"""
from __future__ import annotations

import base64
import binascii
from contextlib import contextmanager
import hashlib
import io
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import httpx
from openai import OpenAI
from PIL import Image

from ..image_io import encode_rgb_png, prepare_rgb_image
from ..logging_utils import protect_provider_logs, redact
from .errors import (LocalCallError, ProviderError, ProviderRefusal, RouteMismatch,
                     TerminalQuotaError, TokenLimitError, TransportError, TransportExhausted)
from .retry import RetryPolicy, classify, safe_provider_error, terminal_quota_code_from_payload
from .types import (AttemptReceipt, CallReceipt, ChatRequest, ChatResponse,
                    ImageEditRequest, ImageEditResult, ImageInput, Usage)


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _positive(value: Any, name: str, *, integer: bool = False) -> Any:
    if type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be positive")
    return value


def _image(value: ImageInput, conversions: list[dict[str, Any]] | None = None) -> tuple[Image.Image, str]:
    try:
        raw = Path(value.path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if value.sha256 is not None and value.sha256 != digest:
            raise ValueError("Image hash mismatch")
        with Image.open(io.BytesIO(raw)) as opened:
            image, conversion = prepare_rgb_image(opened)
        if conversions is not None:
            conversions.append({"stage": "input", "index": len(conversions), **conversion})
        return image, digest
    except Exception:
        raise LocalCallError("ImageInputInvalid") from None


def _png(image: Image.Image, name: str) -> io.BytesIO:
    if image.mode == "RGBA":
        # Edit masks carry alpha; reconstruct them independently from RGB inputs.
        clean = Image.frombytes("RGBA", image.size, image.tobytes())
        stream = io.BytesIO()
        clean.save(stream, format="PNG")
        with Image.open(io.BytesIO(stream.getvalue())) as decoded:
            if (decoded.mode != clean.mode or decoded.size != clean.size
                    or decoded.tobytes() != clean.tobytes() or decoded.info):
                raise LocalCallError("MaskEncodingInvalid")
    else:
        encoded, _ = encode_rgb_png(image)
        stream = io.BytesIO(encoded)
    stream.name = name
    stream.seek(0)
    return stream


def build_messages(request: ChatRequest, *, conversions: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Preserve source ordering, RGB PNG conversion, thumbnail and JSON bytes."""
    content: list[dict[str, Any]] = []
    hashes: list[str] = []
    for number, value in enumerate(request.images, 1):
        image, digest = _image(value, conversions)
        hashes.append(digest)
        if max(image.size) > 1024:
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        url = "data:image/png;base64," + base64.b64encode(_png(image, "image.png").getvalue()).decode("ascii")
        content.extend([{"type": "text", "text": "Image %d: %s." % (number, value.label)},
                        {"type": "image_url", "image_url": {"url": url}}])
    content.append({"type": "text", "text": json.dumps(request.payload, ensure_ascii=False)})
    return [{"role": "system", "content": request.system}, {"role": "user", "content": content}], tuple(hashes)


def _strict_format(value: Mapping[str, Any]) -> dict[str, Any]:
    fmt = dict(value)
    if set(fmt) != {"type", "json_schema"} or fmt["type"] != "json_schema":
        raise ValueError("Provider requires a strict JSON schema response_format")
    schema = fmt["json_schema"]
    if not isinstance(schema, Mapping) or not {"name", "schema", "strict"} <= schema.keys() or set(schema) - {"name", "schema", "strict", "description"} or schema["strict"] is not True:
        raise ValueError("Provider requires explicit name, schema and strict=true")
    # Copy without rewriting names, descriptions, schema, or ordering.
    return {"type": "json_schema", "json_schema": dict(schema)}


def map_responses_request(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Map the frozen system/user subset without dropping any request field."""
    required = {"model", "messages", "response_format", "reasoning_effort", "max_completion_tokens", "timeout"}
    if set(kwargs) != required:
        raise ValueError("Unmapped Chat Completions request fields")
    _positive(kwargs["timeout"], "timeout")
    _positive(kwargs["max_completion_tokens"], "max_output_tokens", integer=True)
    messages = kwargs["messages"]
    if not isinstance(messages, list) or len(messages) != 2 or [m.get("role") for m in messages] != ["system", "user"]:
        raise ValueError("Expected the system/user message pair")
    inputs = []
    for message in messages:
        if set(message) != {"role", "content"}:
            raise ValueError("Unmapped message fields")
        content = message["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        parts = []
        for part in content:
            if part.get("type") == "text" and set(part) == {"type", "text"} and isinstance(part["text"], str):
                parts.append({"type": "input_text", "text": part["text"]})
            elif part.get("type") == "image_url" and set(part) == {"type", "image_url"}:
                image = part["image_url"]
                if message["role"] != "user" or not isinstance(image, dict) or "url" not in image or set(image) - {"url", "detail"} or not isinstance(image["url"], str) or image.get("detail", "auto") not in ("auto", "low", "high"):
                    raise ValueError("Unsupported image part")
                converted = {"type": "input_image", "image_url": image["url"]}
                if "detail" in image:
                    converted["detail"] = image["detail"]
                parts.append(converted)
            else:
                raise ValueError("Unsupported message part")
        inputs.append({"role": message["role"], "content": parts})
    fmt = _strict_format(kwargs["response_format"])
    return {"model": kwargs["model"], "input": inputs,
            "reasoning": {"effort": kwargs["reasoning_effort"]},
            "text": {"format": {"type": "json_schema", **fmt["json_schema"]}},
            "max_output_tokens": kwargs["max_completion_tokens"], "timeout": kwargs["timeout"],
            "stream": True, "background": False, "store": False}


def _counts(value: Any, *, responses: bool = False, image: bool = False) -> Usage | None:
    if not isinstance(value, Mapping):
        return None
    a, b = ("input_tokens", "output_tokens") if responses or image else ("prompt_tokens", "completion_tokens")
    input_count, output_count = value.get(a), value.get(b)
    if type(input_count) is not int or type(output_count) is not int or input_count < 0 or output_count < 0:
        return None
    inputs = value.get("input_tokens_details" if responses or image else "prompt_tokens_details") or {}
    outputs = value.get("output_tokens_details" if responses or image else "completion_tokens_details") or {}
    if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
        return None
    if image and any(type(inputs.get(name)) is not int for name in ("image_tokens", "text_tokens")):
        # Aggregate image usage cannot safely price the missing text/image split.
        return None
    if responses and (type(inputs.get("cached_tokens")) is not int or type(outputs.get("reasoning_tokens")) is not int):
        return None
    cached = inputs.get("cached_tokens")
    reasoning = outputs.get("reasoning_tokens") or 0
    image_in, text_in = inputs.get("image_tokens") or 0, inputs.get("text_tokens") or 0
    image_out = outputs.get("image_tokens", output_count) if image else 0
    details = (reasoning, image_in, text_in, image_out)
    if (any(type(v) is not int or v < 0 for v in details)
            or (cached is not None and (type(cached) is not int or cached < 0 or cached > input_count))
            or reasoning > output_count):
        return None
    if image and (image_in + text_in != input_count or image_out != output_count):
        return None
    if responses and value.get("total_tokens", input_count + output_count) != input_count + output_count:
        return None
    return Usage(input_count, output_count, cached, reasoning,
                 image_in if image else 0, image_out, text_in if image else 0)


def _model_matches(requested: str, returned: str | None) -> bool:
    if not returned:
        return True
    wanted, got = requested.rsplit("/", 1)[-1], returned.rsplit("/", 1)[-1]
    return got == wanted or (got.startswith(wanted) and re.fullmatch(r"-\d{4}-\d{2}-\d{2}", got[len(wanted):]) is not None)


class _IsolatedOpenAI(OpenAI):
    """SDK 2.53+ merges OPENAI_CUSTOM_HEADERS even with default_headers={}."""

    def __init__(self, settings: Any, *, http_client: httpx.Client | None = None) -> None:
        super().__init__(api_key=settings.api_key, base_url=settings.base_url,
                         organization=settings.organization or "", project=settings.project or "",
                         admin_api_key="", workload_identity=None, webhook_secret="", provider=None,
                         websocket_base_url="", timeout=settings.timeout_seconds, max_retries=0,
                         default_headers={}, default_query={}, http_client=http_client,
                         _strict_response_validation=False, _enforce_credentials=True)
        # Discard the constructor's unconditional ambient custom-header merge.
        # No request occurs in construction and the process environment is untouched.
        self._custom_headers = {}


@dataclass
class _Outcome:
    value: Any = field(repr=False)
    raw: Mapping[str, Any] = field(repr=False)
    usage: Usage | None = None
    model: str | None = None
    first_byte_seconds: float | None = None
    max_silence_seconds: float | None = None
    timing_scope: str = "unavailable"
    image_conversions: tuple[Mapping[str, Any], ...] = ()


class _Provider:
    def __init__(self, settings: Any, retry: Any, *, http_client: httpx.Client | None = None,
                 budget: Any = None, logger: Any = None) -> None:
        self.settings, self.route = settings, settings.model
        self.policy = RetryPolicy(retry.max_attempts, retry.base_delay_seconds, retry.max_delay_seconds, retry.jitter)
        self.budget, self.logger = budget, logger
        self._owns_client = http_client is None
        self._secrets = tuple(value for value in (settings.api_key, settings.base_url,
                              settings.organization, settings.project) if value)
        protect_provider_logs(secrets=self._secrets)
        try:
            self._client = _IsolatedOpenAI(settings, http_client=http_client)
        except Exception:
            raise LocalCallError("ClientConfigurationInvalid") from None

    @contextmanager
    def _parsed_response(self, raw: Mapping[str, Any], usage: Usage | None):
        """Preserve attribution and a digest even when a parsed answer is rejected."""
        try:
            yield
        except ProviderError as error:
            model = raw.get("model")
            error.returned_model = redact(model, secrets=self._secrets) if isinstance(model, str) and model else None
            error.response_sha256 = _hash(raw)
            error.usage = usage
            raise error from None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _execute(self, kind: str, role: str, request_hash: str, hashes: tuple[str, ...],
                 dispatch: Callable[[], _Outcome], *, effort: str | None = None,
                 max_output_tokens: int | None = None,
                 image_conversions: tuple[Mapping[str, Any], ...] = ()) -> tuple[_Outcome, CallReceipt]:
        started = time.monotonic()
        attempts: list[AttemptReceipt] = []
        outcome = None
        receipt = None
        for number in range(1, self.policy.max_attempts + 1):
            # Budget errors are local and never wrapped/retried or charged again.
            try:
                reservation = self.budget.reserve(kind, model=self.route, max_output_tokens=max_output_tokens) if self.budget is not None else None
            except Exception as budget_error:
                if receipt is not None:
                    budget_error.receipt = receipt
                raise budget_error from None
            attempt_start = time.monotonic()
            error = None
            accounting_error = None
            try:
                outcome = dispatch()
            except Exception as exc:
                error = classify(exc)
            usage = outcome.usage if error is None else error.usage
            if reservation is not None:
                try:
                    reservation.commit(usage=None if usage is None else usage.to_record())
                except Exception as exc:
                    accounting_error = exc
            charged = reservation.charge_usd if reservation is not None else None
            failed = error is not None or accounting_error is not None
            error_class = (safe_provider_error(accounting_error)["class"] if accounting_error is not None
                           else None if error is None else safe_provider_error(error)["class"])
            delay = self.policy.delay(number, error) if accounting_error is None and isinstance(error, TransportError) and number < self.policy.max_attempts else 0.0
            attempts.append(AttemptReceipt(number, "error" if failed else "complete",
                            time.monotonic() - attempt_start,
                            error_class,
                            None if error is None else error.http_status, delay, charged))
            model = outcome.model if error is None else getattr(error, "returned_model", None)
            model = redact(model, secrets=self._secrets) if model else None
            receipt = CallReceipt(kind=kind, role=redact(role, secrets=self._secrets),
                requested_model=redact(self.route, secrets=self._secrets), returned_model=model,
                attribution_status="known" if model else "unknown", usage=usage,
                usage_status="known" if usage is not None else "unknown",
                status="error" if failed else "complete",
                transport=self.settings.transport if kind == "chat" else "images.edit",
                elapsed_seconds=time.monotonic() - started, request_sha256=request_hash,
                response_sha256=_hash(outcome.raw) if error is None else getattr(error, "response_sha256", None), image_sha256=hashes,
                attempts=tuple(attempts), reasoning_effort=redact(effort, secrets=self._secrets) if effort is not None else None,
                first_byte_seconds=outcome.first_byte_seconds if error is None else None,
                max_silence_seconds=outcome.max_silence_seconds if error is None else None,
                timing_scope=outcome.timing_scope if error is None else "unavailable",
                error_class=error_class,
                http_status=None if error is None else error.http_status,
                image_conversions=image_conversions + (outcome.image_conversions if error is None else ()))
            if self.logger is not None:
                self.logger.info("provider kind=%s attempt=%d status=%s", kind, number, receipt.status)
            if accounting_error is not None:
                accounting_error.receipt = receipt
                raise accounting_error from None
            if error is None:
                return outcome, receipt
            error.receipt = receipt
            if not isinstance(error, TransportError):
                raise error from None
            if number == self.policy.max_attempts:
                raise TransportExhausted("TransportExhausted", http_status=error.http_status,
                                         usage=usage, receipt=receipt) from None
            time.sleep(delay)
        raise AssertionError("Retry policy must permit an attempt")


class OpenAIChatClient(_Provider):
    def complete(self, request: ChatRequest) -> ChatResponse:
        s = self.settings
        effort = request.reasoning_effort if request.reasoning_effort is not None else s.reasoning_effort
        if not isinstance(effort, str) or not effort:
            raise ValueError("reasoning_effort must be explicit")
        cap = _positive(request.max_output_tokens if request.max_output_tokens is not None else s.max_output_tokens, "max_output_tokens", integer=True)
        timeout = _positive(request.timeout_seconds if request.timeout_seconds is not None else s.timeout_seconds, "timeout_seconds")
        conversions = []
        messages, hashes = build_messages(request, conversions=conversions)
        kwargs = {"model": self.route, "messages": messages, "response_format": _strict_format(request.response_format),
                  "reasoning_effort": effort, "max_completion_tokens": cap, "timeout": timeout}
        if s.transport not in ("chat_completions", "responses_stream"):
            raise ValueError("Unknown chat transport; choose chat_completions or responses_stream")
        submitted = kwargs if s.transport == "chat_completions" else map_responses_request(kwargs)

        def dispatch() -> _Outcome:
            if s.transport == "responses_stream":
                raw, timings = self._stream(submitted)
                usage = _counts(raw.get("usage"), responses=True)
            else:
                raw = self._client.chat.completions.create(**submitted).model_dump()
                timings = {}
                usage = _counts(raw.get("usage"))
            with self._parsed_response(raw, usage):
                if s.transport == "responses_stream":
                    text, finish = self._response_answer(raw, usage)
                else:
                    try:
                        choice = raw["choices"][0]
                        text, finish = choice["message"]["content"], choice["finish_reason"]
                        refusal = choice["message"].get("refusal")
                    except (KeyError, IndexError, TypeError):
                        raise ProviderError("MalformedResponse", usage=usage) from None
                    if refusal or finish == "content_filter":
                        raise ProviderRefusal("ProviderRefusal", usage=usage)
                    if finish == "length":
                        raise TokenLimitError("TokenLimitError", usage=usage)
                    if finish != "stop" or not isinstance(text, str) or not text:
                        raise ProviderError("IncompleteAnswer", usage=usage)
                model = raw.get("model")
                if model is not None and not isinstance(model, str):
                    raise ProviderError("ReturnedModelInvalid", usage=usage)
                if not _model_matches(self.route, model):
                    raise RouteMismatch("RouteMismatch", usage=usage)
                return _Outcome((text, finish), raw, usage, model, **timings)

        outcome, receipt = self._execute("chat", request.role, _hash(submitted), hashes, dispatch,
                                         effort=effort, max_output_tokens=cap,
                                         image_conversions=tuple(conversions))
        return ChatResponse(outcome.value[0], outcome.value[1], receipt.returned_model, receipt)

    def _response_answer(self, raw: Mapping[str, Any], usage: Usage | None) -> tuple[str, str]:
        if raw.get("status") == "incomplete" and (raw.get("incomplete_details") or {}).get("reason") == "max_output_tokens":
            raise TokenLimitError("TokenLimitError", usage=usage)
        if raw.get("status") != "completed":
            code = terminal_quota_code_from_payload(raw)
            if code is not None:
                raise TerminalQuotaError(code, usage=usage)
            raise ProviderRefusal("ResponsesNotCompleted", usage=usage)
        texts: list[str] = []
        messages = 0
        for item in raw.get("output") or []:
            if item.get("type") == "reasoning":
                continue
            if item.get("type") != "message" or item.get("role") != "assistant" or item.get("phase") not in (None, "final_answer") or item.get("status") not in (None, "completed"):
                raise ProviderError("UnexpectedResponseOutput", usage=usage)
            messages += 1
            for part in item.get("content") or []:
                if part.get("type") == "refusal":
                    raise ProviderRefusal("ProviderRefusal", usage=usage)
                if part.get("type") != "output_text" or not isinstance(part.get("text"), str):
                    raise ProviderError("UnexpectedResponseContent", usage=usage)
                texts.append(part["text"])
        if messages != 1 or not texts or not "".join(texts):
            raise ProviderError("IncompleteAnswer", usage=usage)
        return "".join(texts), "stop"

    def _stream(self, submitted: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        start = time.monotonic()
        first = last = None
        max_gap = 0.0
        native = None
        identity: dict[str, str] = {}
        stream = self._client.responses.create(**submitted)
        response = getattr(stream, "response", None)
        original = getattr(response, "iter_raw", None)
        tapped = callable(original) and not getattr(response, "is_stream_consumed", False)
        if tapped:
            def timed(*args: Any, **kwargs: Any):
                nonlocal first, last, max_gap
                for data in original(*args, **kwargs):
                    if data:
                        now = time.monotonic()
                        if first is None:
                            first = now
                        if last is not None:
                            max_gap = max(max_gap, now - last)
                        last = now
                    yield data
            response.iter_raw = timed
        try:
            for event in stream:
                packet = event.model_dump(exclude_unset=True)
                event_type = packet.get("type")
                if not isinstance(event_type, str) or not event_type:
                    raise ProviderError("MalformedStreamEvent")
                current = packet.get("response")
                if isinstance(current, dict):
                    for key in ("id", "model"):
                        value = current.get(key)
                        if value is not None:
                            if key in identity and identity[key] != value:
                                raise ProviderError("ResponsesIdentityChanged")
                            identity[key] = value
                if native is not None and event_type.endswith(".delta"):
                    raise ProviderError("DeltaAfterTerminal")
                if event_type in ("response.completed", "response.incomplete", "response.failed", "response.cancelled"):
                    if native is not None or not isinstance(current, dict):
                        raise ProviderError("InvalidTerminalEvent")
                    if current.get("status") != event_type.split(".", 1)[1]:
                        raise ProviderError("TerminalStatusMismatch")
                    native = current
                    if event_type == "response.failed":
                        code = terminal_quota_code_from_payload(current)
                        if code is not None:
                            raise TerminalQuotaError(code)
                elif event_type == "error":
                    code = terminal_quota_code_from_payload(packet)
                    if code is not None:
                        raise TerminalQuotaError(code)
                    raise ProviderError("ResponsesStreamError")
            if native is None:
                # A cleanly truncated stream is not a retryable semantic result.
                raise ProviderError("MissingTerminalEvent")
        except ProviderError as error:
            model = identity.get("model")
            error.returned_model = redact(model, secrets=self._secrets) if model else None
            if native is not None:
                error.response_sha256 = _hash(native)
                error.usage = _counts(native.get("usage"), responses=True)
            raise error from None
        finally:
            if tapped:
                response.iter_raw = original
            stream.close()
        finished = time.monotonic()
        silence = max(max_gap, (first if first is not None else finished) - start,
                      0.0 if last is None else finished - last) if tapped else None
        return native, {"first_byte_seconds": None if first is None else first - start,
                        "max_silence_seconds": silence,
                        "timing_scope": "httpx.Response.iter_raw body bytes" if tapped else "unavailable"}


class OpenAIImageEditClient(_Provider):
    def edit(self, request: ImageEditRequest) -> ImageEditResult:
        s = self.settings
        if not request.images:
            raise LocalCallError("ImageInputRequired")
        size = request.request_size or (s.panorama_request_size if request.panorama else s.view_request_size)
        output_size = request.output_size or (s.panorama_output_size if request.panorama else s.view_output_size)
        if not isinstance(size, str) or re.fullmatch(r"[1-9]\d*x[1-9]\d*", size) is None:
            raise ValueError("request_size must be WIDTHxHEIGHT")
        if len(output_size) != 2 or any(type(v) is not int or v <= 0 for v in output_size):
            raise ValueError("output_size must contain two positive integers")
        quality = request.quality if request.quality is not None else s.quality
        if not isinstance(quality, str) or not quality:
            raise ValueError("quality must be explicit")
        timeout = _positive(request.timeout_seconds if request.timeout_seconds is not None else s.timeout_seconds, "timeout_seconds")
        images, hashes, conversions = [], [], []
        for value in request.images:
            image, digest = _image(value, conversions)
            images.append(image)
            hashes.append(digest)
        mask = None
        if request.mask is not None:
            if len(images) != 1:
                raise LocalCallError("MaskedEditRequiresOneImage")
            try:
                raw = Path(request.mask.path).read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                if request.mask.sha256 is not None and request.mask.sha256 != digest:
                    raise ValueError()
                with Image.open(io.BytesIO(raw)) as opened:
                    if opened.mode != "L" or opened.size != images[0].size:
                        raise ValueError()
                    mask = opened.copy()
                values = set(mask.getdata())
                if values != {0, 255}:
                    raise ValueError()
                width, height = (int(v) for v in size.split("x"))
                images[0] = images[0].resize((width, height), Image.Resampling.LANCZOS)
                mask = mask.resize((width, height), Image.Resampling.NEAREST)
                if set(mask.getdata()) != {0, 255}:
                    raise ValueError()
                rgba = Image.new("RGBA", mask.size, (255, 255, 255, 255))
                rgba.putalpha(mask.point(lambda v: 255 - v))
                mask = rgba
                hashes.append(digest)
            except Exception:
                raise LocalCallError("MaskedEditRequestInvalid") from None
        metadata = {"model": self.route, "prompt": request.prompt, "images": hashes,
                    "size": size, "output_size": output_size, "quality": quality,
                    "timeout": timeout, "mask": request.mask is not None}

        def dispatch() -> _Outcome:
            # Recreate streams for every transport attempt; the SDK consumes them.
            files = [_png(image, "input_%d.png" % number) for number, image in enumerate(images)]
            kwargs = {"model": self.route, "prompt": request.prompt,
                      "image": files if len(files) > 1 else files[0], "size": size,
                      "quality": quality, "timeout": timeout}
            if mask is not None:
                kwargs["mask"] = _png(mask, "mask.png")
            try:
                raw = self._client.images.edit(**kwargs).model_dump()
            finally:
                for stream in files:
                    stream.close()
                if mask is not None:
                    kwargs["mask"].close()
            usage = _counts(raw.get("usage"), image=True)
            with self._parsed_response(raw, usage):
                model = raw.get("model")
                if model is not None and not isinstance(model, str):
                    raise ProviderError("ReturnedModelInvalid", usage=usage)
                if not _model_matches(self.route, model):
                    raise RouteMismatch("RouteMismatch", usage=usage)
                data = raw.get("data") or []
                encoded = data[0].get("b64_json") if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict) else None
                if not isinstance(encoded, str) or not encoded:
                    raise ProviderRefusal("ImagePayloadInvalid", usage=usage)
                try:
                    payload = base64.b64decode(encoded, validate=True)
                    with Image.open(io.BytesIO(payload)) as opened:
                        opened.load()
                        raw_size = opened.size
                        endpoint, conversion = prepare_rgb_image(opened)
                except (binascii.Error, OSError, ValueError):
                    raise ProviderRefusal("ImagePayloadInvalid", usage=usage) from None
                if raw.get("quality") != quality:
                    raise ProviderRefusal("ReturnedQualityMismatch", usage=usage)
                width, height = output_size
                if raw_size[0] * height != raw_size[1] * width:
                    raise ProviderRefusal("AspectMismatch", usage=usage)
                resize = "none"
                if raw_size != tuple(output_size):
                    endpoint = endpoint.resize(tuple(output_size), Image.Resampling.LANCZOS)
                    resize = "uniform_lanczos"
                try:
                    out = Path(request.output_dir)
                    out.mkdir(parents=True, exist_ok=True)
                    path = out / "endpoint.png"
                    # Never overwrite an accepted endpoint or follow a symlink.
                    with path.open("xb") as handle:
                        encoded, saved_conversion = encode_rgb_png(endpoint)
                        handle.write(encoded)
                except Exception:
                    raise LocalCallError("ImageOutputWriteFailed", usage=usage) from None
                return _Outcome((path, raw_size, quality, resize), raw, usage, model,
                                image_conversions=({"stage": "provider_endpoint", **conversion,
                                                    "saved_pixel_sha256": saved_conversion["pixel_sha256"]},))

        outcome, receipt = self._execute("image", "editor", _hash(metadata), tuple(hashes), dispatch,
                                        image_conversions=tuple(conversions))
        return ImageEditResult(*outcome.value, receipt)
