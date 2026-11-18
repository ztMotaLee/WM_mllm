"""Offline configuration tests; all credential values are synthetic."""
from dataclasses import FrozenInstanceError

import pytest
import yaml

from wmmllm.config import ChatConfig, ConfigError, ImageConfig, RetryConfig, load_config


TEST_ENV = {"WMM_LLM_API_KEY": "test-only-config-secret", "WMM_LLM_MODEL": "test-chat",
            "WMM_LLM_BASE_URL": "https://provider.invalid/v1"}


def write_yaml(tmp_path, data):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_defaults_and_fieldwise_image_fallback():
    env = {**TEST_ENV, "WMM_LLM_BASE_URL": "https://example.invalid/v1",
           "WMM_LLM_ORGANIZATION": "chat-org", "WMM_LLM_PROJECT": "chat-project",
           "WMM_IMAGE_MODEL": "test-image", "WMM_IMAGE_PROJECT": "image-project"}
    cfg = load_config(environ=env)
    assert cfg.chat.transport == "chat_completions"; assert cfg.chat.reasoning_effort == "high"
    assert cfg.chat.max_output_tokens == 32000; assert cfg.chat.timeout_seconds == 600
    assert cfg.image.api_key == cfg.chat.api_key; assert cfg.image.base_url == cfg.chat.base_url
    assert cfg.image.model == "test-image"; assert cfg.image.organization == "chat-org"
    assert cfg.image.project == "image-project"; assert cfg.image.view_request_size == "1792x1024"
    assert cfg.image.panorama_request_size == "2048x1024"; assert cfg.image.view_output_size == (1344, 768)
    assert cfg.image.panorama_output_size == (2048, 1024); assert cfg.image.timeout_seconds == 420
    assert cfg.transport.max_attempts == 5; assert cfg.transport.jitter == 0.2; assert cfg.video.backend == "none"
    with pytest.raises(FrozenInstanceError):
        cfg.chat.model = "changed"


@pytest.mark.parametrize("env,variable", [({}, "WMM_LLM_API_KEY"),
    ({"WMM_LLM_API_KEY": "test-only-key"}, "WMM_LLM_MODEL"),
    ({"WMM_IMAGE_API_KEY": "test-only-image-key", "WMM_IMAGE_MODEL": "image"}, "WMM_LLM_API_KEY"),
    ({"WMM_LLM_API_KEY": "test-only-key", "WMM_LLM_MODEL": "chat"}, "WMM_LLM_BASE_URL")])
def test_missing_configuration_fails_locally(env, variable):
    with pytest.raises(ConfigError, match=variable) as exc:
        load_config(environ=env)
    assert "no request was sent" in str(exc.value)
    for value in env.values():
        assert value not in str(exc.value)


def test_only_explicit_env_file_and_process_precedence(tmp_path):
    credentials = tmp_path / "credentials.txt"
    credentials.write_text("export WMM_LLM_API_KEY='test-only-file-key'\nWMM_LLM_MODEL=file-chat\n"
                           "WMM_LLM_ORGANIZATION=file-org\nWMM_IMAGE_MODEL=file-image\n"
                           "WMM_LLM_BASE_URL=https://file.invalid/v1\n"
                           "OPENAI_PROJECT_ID=ignored-project\n", encoding="utf-8")
    cfg = load_config(env_file=credentials, environ={"WMM_LLM_API_KEY": "test-only-process-key", "WMM_LLM_MODEL": "process-chat"})
    assert cfg.chat.api_key == "test-only-process-key"; assert cfg.chat.model == "process-chat"
    assert cfg.chat.organization == "file-org"; assert cfg.image.model == "file-image"
    assert cfg.chat.project is None
    # A nearby credentials file is not consulted without an explicit path.
    with pytest.raises(ConfigError, match="WMM_LLM_API_KEY"):
        load_config(environ={})
    (tmp_path / "second.txt").write_text("WMM_LLM_API_KEY=test-only-second\nWMM_LLM_MODEL=second\n"
                                         "WMM_LLM_BASE_URL=https://second.invalid/v1\n")
    config = write_yaml(tmp_path, {"env_file": "credentials.txt"})
    assert load_config(config, environ={}).chat.model == "file-chat"
    assert load_config(config, env_file=tmp_path / "second.txt", environ={}).chat.model == "second"


def test_nonsecret_cli_environment_yaml_precedence(tmp_path):
    config = write_yaml(tmp_path, {"chat": {"model": "yaml-chat", "timeout_seconds": 90},
                                   "image": {"model": "yaml-image"}})
    cfg = load_config(config, environ=TEST_ENV,
                      overrides={"chat": {"model": "cli-chat", "timeout_seconds": 80}})
    assert cfg.chat.model == "cli-chat"; assert cfg.chat.timeout_seconds == 80
    assert cfg.image.model == "yaml-image"


def test_ambient_openai_settings_do_not_supply_credentials():
    ambient = {"OPENAI_API_KEY": "test-only-ambient", "OPENAI_BASE_URL": "https://ignored.invalid",
               "OPENAI_ORG_ID": "ignored-org", "OPENAI_PROJECT_ID": "ignored-project",
               "OPENAI_CUSTOM_HEADERS": "ignored-headers"}
    with pytest.raises(ConfigError, match="WMM_LLM_API_KEY"):
        load_config(environ=ambient)
    cfg = load_config(environ={**ambient, **TEST_ENV})
    assert cfg.chat.base_url == TEST_ENV["WMM_LLM_BASE_URL"]; assert cfg.chat.organization is None
    assert cfg.chat.project is None


@pytest.mark.parametrize("data", [
    {"chat": {"api_key": "test-only-forbidden"}},
    {"WMM_LLM_API_KEY": "test-only-forbidden"},
    {"image": {"authorization": "test-only-forbidden"}},
    {"budget": {"prices": {"secret": "test-only-forbidden"}}},
])
def test_inline_yaml_and_cli_keys_rejected_without_echo(tmp_path, data):
    for kwargs in ({"config_path": write_yaml(tmp_path, data)}, {"overrides": data}):
        with pytest.raises(ConfigError) as exc:
            load_config(environ=TEST_ENV, **kwargs)
        assert "test-only-forbidden" not in str(exc.value); assert "Inline credentials" in str(exc.value)


@pytest.mark.parametrize("data", [
    {"chat": {"transport": "unsupported"}}, {"chat": {"reasoning_effort": "ultra"}},
    {"chat": {"max_output_tokens": 0}}, {"chat": {"timeout_seconds": float("inf")}},
    {"image": {"view_output_size": [1344]}}, {"image": {"quality": "unsupported"}},
    {"transport": {"jitter": 2}}, {"transport": {"max_attempts": False}},
    {"video": {"backend": "unsupported"}}, {"chat": {"unknown": "test-only-private-value"}},
])
def test_invalid_settings_have_actionable_local_errors(tmp_path, data):
    with pytest.raises(ConfigError) as exc:
        load_config(write_yaml(tmp_path, data), environ=TEST_ENV)
    assert "test-only-private-value" not in str(exc.value)
    (tmp_path / "bad.yaml").write_text("chat: [test-only-private-value\n")
    with pytest.raises(ConfigError) as exc:
        load_config(tmp_path / "bad.yaml", environ=TEST_ENV)
    assert "test-only-private-value" not in str(exc.value)
    with pytest.raises(ConfigError, match="explicitly selected env file"):
        load_config(env_file=tmp_path / "missing.txt", environ=TEST_ENV)


def test_cap_requires_user_prices_and_both_upper_bounds(tmp_path):
    for prices in ({}, {"chat": {"upper_bound_usd": 1}}):
        config = write_yaml(tmp_path, {"budget": {"max_cost_usd": 10, "prices": prices}})
        with pytest.raises(ConfigError, match="requires user prices"):
            load_config(config, environ=TEST_ENV)
    prices = tmp_path / "prices.yaml"
    prices.write_text(yaml.safe_dump({"chat": {"upper_bound_usd": 1}, "image": {"upper_bound_usd": 2}}))
    cfg = load_config(write_yaml(tmp_path, {"budget": {"max_cost_usd": 3, "price_table_file": "prices.yaml"}}), environ=TEST_ENV)
    assert cfg.budget.prices["image"].upper_bound_usd == 2
    with pytest.raises(TypeError):
        cfg.budget.prices["image"] = cfg.budget.prices["chat"]


def test_explicit_empty_process_values_do_not_resurrect_env_file_values(tmp_path):
    credentials = tmp_path / "credentials.txt"
    credentials.write_text("WMM_LLM_API_KEY=test-only-file-key\nWMM_LLM_MODEL=file-chat\n"
                           "WMM_IMAGE_MODEL=file-image\nWMM_IMAGE_ORGANIZATION=file-org\n"
                           "WMM_LLM_BASE_URL=https://file.invalid/v1\n")
    cfg = load_config(env_file=credentials, environ={**TEST_ENV,
        "WMM_LLM_ORGANIZATION": "process-org", "WMM_IMAGE_MODEL": "", "WMM_IMAGE_ORGANIZATION": ""})
    assert cfg.image.model == TEST_ENV["WMM_LLM_MODEL"]; assert cfg.image.organization == "process-org"
    with pytest.raises(ConfigError, match="WMM_LLM_API_KEY"):
        load_config(env_file=credentials, environ={"WMM_LLM_API_KEY": ""})


@pytest.mark.parametrize("url", [
    "https://test-user:test-only-secret@example.invalid/v1",
    "https://example.invalid/v1?api_key=test-only-secret",
    "https://example.invalid/v1?access_token=test-only-secret",
])
def test_credential_bearing_yaml_and_cli_urls_are_forbidden(tmp_path, url):
    for kwargs in ({"config_path": write_yaml(tmp_path, {"chat": {"base_url": url}})},
                   {"overrides": {"chat": {"base_url": url}}}):
        with pytest.raises(ConfigError) as exc:
            load_config(environ=TEST_ENV, **kwargs)
        assert url not in str(exc.value); assert "test-only-secret" not in str(exc.value)


def test_nested_overrides_only():
    cfg = load_config(environ=TEST_ENV, overrides={"chat": {"max_output_tokens": 42, "reasoning_effort": "xhigh"}})
    assert cfg.chat.max_output_tokens == 42 and cfg.chat.reasoning_effort == "xhigh"
    with pytest.raises(ConfigError, match="Unknown top-level"):
        load_config(environ=TEST_ENV, overrides={"chat.model": "first"})
    with pytest.raises(ConfigError, match="responses_stream is no longer supported"):
        load_config(environ=TEST_ENV, overrides={"chat": {"transport": "responses_stream"}})


@pytest.mark.parametrize("constructor,kwargs", [
    (ChatConfig, {"api_key": "", "model": "test-chat"}),
    (ChatConfig, {"api_key": "test-only-key", "model": "test-chat"}),
    (ChatConfig, {"api_key": "test-only-key", "model": ""}),
    (ImageConfig, {"api_key": "test-only-key", "model": "test-image", "quality": "invalid"}),
    (RetryConfig, {"max_attempts": 0}),
])
def test_direct_typed_config_construction_also_fails_fast(constructor, kwargs):
    with pytest.raises(ConfigError):
        constructor(**kwargs)
