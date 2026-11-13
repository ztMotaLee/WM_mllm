"""Release-review regressions use procedural images and in-process transports."""
from __future__ import annotations

import base64
from email.parser import BytesParser
from email.policy import default
import io
import json
import logging
from pathlib import Path
import struct

import httpx
from PIL import Image, ImageCms
import pytest

from wmmllm.budget import BudgetController, BudgetError
from wmmllm.config import (AppConfig, BudgetConfig, ChatConfig, ConfigError,
                           ImageConfig, PriceConfig, RetryConfig, VideoConfig,
                           load_config)
from wmmllm.image_io import encode_rgb_png, load_rgb_image, prepare_rgb_image, save_rgb_png
from wmmllm.logging_utils import protect_provider_logs
from wmmllm.memory.observer import ObserverDeps, observe_state
from wmmllm.providers import ChatRequest, ImageEditRequest, ImageInput, create_providers
from wmmllm.providers.errors import ProviderRefusal, TerminalQuotaError, TransportError
from wmmllm.providers.openai_compatible import _counts, build_messages
from wmmllm.providers.retry import (TERMINAL_QUOTA_CODES, classify,
                                     raise_if_fatal_provider_error, safe_provider_error)
from wmmllm.session.io import file_ref


SCHEMA = {"type": "json_schema", "json_schema": {"name": "decision", "strict": True,
    "schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}}


def _config():
    return AppConfig(ChatConfig("review-test-credential", "test-chat", base_url="https://provider.invalid/v1"),
                     ImageConfig("review-test-credential", "test-image", base_url="https://provider.invalid/v1"),
                     transport=RetryConfig(jitter=0))


def _profile(non_srgb):
    profile = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    if non_srgb:
        # A procedural valid RGB matrix profile with a different transfer curve.
        for index in range(struct.unpack_from(">I", profile, 128)[0]):
            name, offset, size = struct.unpack_from(">4sII", profile, 132 + index * 12)
            if name == b"rTRC":
                assert profile[offset:offset + 4] == b"para" and size >= 16
                struct.pack_into(">i", profile, offset + 12, round(1.8 * 65536))
    return bytes(profile)


def _metadata_image(path, fmt, non_srgb):
    image = Image.frombytes("RGB", (32, 16), bytes((index * 29 % 256 for index in range(32 * 16 * 3))))
    exif = Image.Exif()
    exif[270] = "procedural-private-metadata-canary"
    kwargs = {"icc_profile": _profile(non_srgb), "exif": exif}
    if fmt == "WEBP":
        kwargs["lossless"] = True
    image.save(path, format=fmt, **kwargs)
    with Image.open(path) as opened:
        assert opened.info.get("icc_profile") and len(opened.getexif())
        decoded = opened.convert("RGB")
        if non_srgb:
            expected = ImageCms.profileToProfile(opened,
                ImageCms.ImageCmsProfile(io.BytesIO(opened.info["icc_profile"])),
                ImageCms.createProfile("sRGB"), outputMode="RGB")
            assert expected.tobytes() != decoded.tobytes()
        else:
            expected = decoded
    return expected


def _assert_clean(encoded, expected):
    with Image.open(io.BytesIO(encoded)) as decoded:
        assert decoded.mode == "RGB" and decoded.size == expected.size
        assert decoded.tobytes() == expected.tobytes()
        assert decoded.info == {} and not len(decoded.getexif())
    assert b"procedural-private-metadata-canary" not in encoded


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
@pytest.mark.parametrize("non_srgb", [False, True])
def test_metadata_removed_pixels_preserved_or_converted_once(tmp_path, fmt, non_srgb):
    source = tmp_path / ("source." + fmt.lower())
    expected = _metadata_image(source, fmt, non_srgb)
    clean, record = load_rgb_image(source)
    expected_conversion = "embedded_icc_to_srgb" if non_srgb else "srgb_identity"
    assert record["color_conversion"] == expected_conversion
    assert record["decoded_rgb_pixels_changed"] == non_srgb
    assert clean.info == {} and clean.tobytes() == expected.tobytes()
    encoded, next_record = encode_rgb_png(clean)
    assert next_record["color_conversion"] == "none"
    _assert_clean(encoded, expected)
    saved = tmp_path / "saved.png"
    save_rgb_png(clean, saved)
    _assert_clean(saved.read_bytes(), expected)
    messages, _ = build_messages(ChatRequest("test", "Unchanged system", (ImageInput(source),), {}, SCHEMA))
    uploaded = base64.b64decode(messages[1]["content"][1]["image_url"]["url"].split(",", 1)[1])
    _assert_clean(uploaded, expected)


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
@pytest.mark.parametrize("non_srgb", [False, True])
def test_edit_upload_and_returned_endpoint_strip_metadata_and_record_conversion(tmp_path, fmt, non_srgb):
    source = tmp_path / ("source." + fmt.lower())
    expected = _metadata_image(source, fmt, non_srgb)
    captured = []
    def handler(wire):
        captured.append(wire)
        return httpx.Response(200, json={"model": "test-image", "quality": "high",
            "data": [{"b64_json": base64.b64encode(source.read_bytes()).decode()}]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with create_providers(_config(), http_client=client) as providers:
            result = providers.image.edit(ImageEditRequest("Unchanged edit", (ImageInput(source),),
                tmp_path / "out", request_size="32x16", output_size=(32, 16)))
    message = BytesParser(policy=default).parsebytes(b"Content-Type: " + captured[0].headers["content-type"].encode()
        + b"\r\nMIME-Version: 1.0\r\n\r\n" + captured[0].content)
    uploads = [part.get_payload(decode=True) for part in message.iter_parts() if part.get_filename()]
    assert len(uploads) == 1
    _assert_clean(uploads[0], expected)
    _assert_clean(result.image.read_bytes(), expected)
    records = result.receipt.to_record()["image_conversions"]
    assert [record["stage"] for record in records] == ["input", "provider_endpoint"]
    assert all(record["color_conversion"] == ("embedded_icc_to_srgb" if non_srgb else "srgb_identity")
               for record in records)
    assert "procedural-private-metadata-canary" not in json.dumps(records)


def test_invalid_icc_fails_ingestion_without_copying_profile():
    image = Image.new("RGB", (2, 2))
    image.info["icc_profile"] = b"procedural-private-metadata-canary"
    with pytest.raises(ValueError, match="color profile") as error:
        prepare_rgb_image(image)
    assert "procedural-private" not in str(error.value)


@pytest.mark.parametrize("code", sorted(TERMINAL_QUOTA_CODES))
def test_terminal_quota_429_is_dispatched_once_and_receipt_is_sanitized(monkeypatch, code):
    calls, sleeps = [], []
    monkeypatch.setattr("wmmllm.providers.openai_compatible.time.sleep", sleeps.append)
    def handler(wire):
        calls.append(wire)
        return httpx.Response(429, json={"error": {"code": code,
            "message": "procedural-private-error-canary", "type": "quota"}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with create_providers(_config(), http_client=client) as providers:
            with pytest.raises(TerminalQuotaError) as error:
                providers.chat.complete(ChatRequest("test", "Unchanged system", (), {}, SCHEMA))
    assert len(calls) == 1 and sleeps == []
    assert error.value.error_class == code
    assert len(error.value.receipt.attempts) == 1
    assert "procedural-private-error-canary" not in json.dumps(error.value.receipt.to_record())


@pytest.mark.parametrize("status", [401, 403, 404])
def test_authorization_and_missing_model_errors_are_fatal(status):
    error = ProviderRefusal("SyntheticFailure", http_status=status)
    with pytest.raises(ProviderRefusal) as caught:
        raise_if_fatal_provider_error(error)
    assert caught.value is error


def test_fatal_guard_keeps_budget_and_terminal_quota_but_handles_other_400():
    with pytest.raises(BudgetError):
        raise_if_fatal_provider_error(BudgetError("Local bound"))
    quota = TransportError("insufficient_quota", http_status=429)
    assert isinstance(classify(quota), TerminalQuotaError)
    with pytest.raises(TransportError):
        raise_if_fatal_provider_error(quota)
    assert raise_if_fatal_provider_error(ProviderRefusal("OptionalRepairFailure", http_status=400)) is None
    assert safe_provider_error(ProviderRefusal("private text https://provider.invalid/")) == {"class": "ProviderError"}


def test_observer_does_not_retry_a_terminal_quota_transport_error(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (4, 2), "blue").save(source)
    reference = file_ref(source, run_root=tmp_path)
    class Chat:
        route = "test"
        calls = 0
        def complete(self, request):
            self.calls += 1
            raise TransportError("insufficient_quota", http_status=429)
    chat = Chat()
    deps = ObserverDeps(chat, {"chat": 2}, 1, (0, 1), ("source", "endpoint"), "Unchanged system",
                        SCHEMA, lambda: 0.0, tmp_path)
    with pytest.raises(TransportError):
        observe_state(task_id="test", segment=1, caption="Observe", start=reference,
                      endpoint=reference, known_subjects={}, deps=deps, out_dir=tmp_path / "observation")
    assert chat.calls == 1


def _budget(price):
    return BudgetController(BudgetConfig(prices={"chat": price}))


def test_chat_cached_input_cost_uses_distinct_rate():
    price = PriceConfig(1, input_usd_per_million=10, cached_input_usd_per_million=2,
                       output_usd_per_million=20)
    reservation = _budget(price).reserve("chat")
    assert reservation.commit(usage={"input_tokens": 1000, "cached_input_tokens": 600,
                                     "output_tokens": 100}) == pytest.approx(.0072)
    assert reservation.cost_status == "measured"


@pytest.mark.parametrize("cached", [-1, 1001, True, 1.5, "20"])
def test_invalid_cached_counts_retain_the_full_bound(cached):
    reservation = _budget(PriceConfig(1, input_usd_per_million=10,
        cached_input_usd_per_million=2, output_usd_per_million=20)).reserve("chat")
    assert reservation.commit(usage={"input_tokens": 1000, "cached_input_tokens": cached,
                                     "output_tokens": 100}) == 1
    assert reservation.cost_status == "upper_bound"


def test_missing_cached_rate_charges_full_input_and_missing_count_keeps_bound():
    usage = {"input_tokens": 1000, "cached_input_tokens": 600, "output_tokens": 100}
    full = _budget(PriceConfig(1, input_usd_per_million=10, output_usd_per_million=20)).reserve("chat")
    assert full.commit(usage=usage) == pytest.approx(.012)
    reservation = _budget(PriceConfig(1, input_usd_per_million=10,
        cached_input_usd_per_million=2, output_usd_per_million=20)).reserve("chat")
    del usage["cached_input_tokens"]
    assert reservation.commit(usage=usage) == 1
    assert reservation.cost_status == "upper_bound"
    counts = _counts({"prompt_tokens": 1000, "completion_tokens": 100})
    assert counts.cached_input_tokens is None


@pytest.mark.parametrize("cached", [0, 600])
def test_raw_usage_cached_details_are_priced(cached):
    reservation = _budget(PriceConfig(1, input_usd_per_million=10,
        cached_input_usd_per_million=2, output_usd_per_million=20)).reserve("chat")
    assert reservation.commit(usage={"prompt_tokens": 1000, "completion_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": cached}}) == pytest.approx((1000-cached)*.00001+cached*.000002+.002)


def test_cached_rate_yaml_is_validated(tmp_path):
    path = tmp_path / "prices.yaml"
    path.write_text("budget:\n  prices:\n    chat:\n      upper_bound_usd: 1\n      input_usd_per_million: 10\n      cached_input_usd_per_million: 2\n      output_usd_per_million: 20\n")
    config = load_config(path, environ={"WMM_LLM_API_KEY": "review-test-credential",
                         "WMM_LLM_MODEL": "test-chat", "WMM_IMAGE_MODEL": "test-image"})
    assert config.budget.prices["chat"].cached_input_usd_per_million == 2
    with pytest.raises(ConfigError):
        PriceConfig(1, cached_input_usd_per_million=-1)


def test_late_child_logger_and_root_handler_cannot_bypass_provider_redaction():
    secret = "late-log-procedural-credential"
    protect_provider_logs((secret,))
    root = logging.getLogger()
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s %(custom)s"))
    root.addHandler(handler)
    logger = logging.getLogger("openai.review_late_child.transport")
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        logger.debug("private-payload-canary %s", secret, extra={"custom": secret})
        logger.warning("private-payload-canary %s", secret, extra={"custom": secret})
        try:
            raise RuntimeError("private-body-canary " + secret)
        except RuntimeError:
            logger.exception("failure", extra={"custom": secret})
    finally:
        logger.setLevel(old_level)
        root.removeHandler(handler)
    output = stream.getvalue()
    assert output and "Provider transport diagnostic" in output
    assert secret not in output and "private-payload-canary" not in output
    assert "private-body-canary" not in output


@pytest.mark.parametrize("frames", [-1, 0, 1, True, 2.5])
def test_video_config_rejects_fewer_than_two_frames_and_nonintegers(frames):
    with pytest.raises(ConfigError):
        VideoConfig(num_frames=frames)


def test_video_config_accepts_two_frames():
    assert VideoConfig(num_frames=2).num_frames == 2
