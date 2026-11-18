"""Offline budget reservations use synthetic user prices, not provider tariffs."""
from concurrent.futures import ThreadPoolExecutor
import traceback

import pytest

from wmmllm.budget import BudgetCheckpointError, BudgetController, BudgetError, BudgetExceeded, PriceBoundExceeded
from wmmllm.config import BudgetConfig, ConfigError, PriceConfig, load_config


def prices(chat=1.0, image=2.0):
    return {"chat": PriceConfig(chat), "image": PriceConfig(image)}


def cached_chat():
    return BudgetController(BudgetConfig(prices={"chat": PriceConfig(
        1, input_usd_per_million=10, cached_input_usd_per_million=2, output_usd_per_million=20)}))


def test_cap_requires_prices_and_resume_cannot_disable_it():
    with pytest.raises(ConfigError, match="requires user prices"):
        BudgetConfig(max_cost_usd=1)
    for value in (-1, float("inf"), float("nan"), True):
        with pytest.raises(ConfigError):
            PriceConfig(value)
    controller = BudgetController(BudgetConfig(max_cost_usd=1, prices=prices()))
    with pytest.raises(BudgetError, match="Resuming a capped run requires an explicit spend cap"):
        BudgetController.from_snapshot(BudgetConfig(), controller.snapshot())


def test_reserve_stops_before_cap_and_commit_is_idempotent():
    controller = BudgetController(BudgetConfig(max_cost_usd=2, prices=prices()))
    first = controller.reserve("chat")
    assert controller.snapshot()["reserved_cost_usd"] == 1
    with pytest.raises(BudgetExceeded, match="no request was sent"):
        controller.reserve("image")
    assert first.commit() == 1 and first.commit() == 1
    controller.reserve("chat").commit(usage={})
    with pytest.raises(BudgetExceeded):
        controller.reserve("chat")
    assert controller.snapshot()["accounted_cost_usd"] == 2 and controller.snapshot()["upper_bound_calls"] == 2
    uncapped = BudgetController()
    assert uncapped.reserve("chat").commit() is None
    assert uncapped.snapshot()["total_cost_usd"] is None and uncapped.snapshot()["unknown_calls"] == 1


def test_measured_usage_including_cached_rates():
    table = {**prices(), "chat": PriceConfig(1, input_usd_per_million=2, output_usd_per_million=4),
             "image": PriceConfig(2, text_input_usd_per_million=1, image_input_usd_per_million=2,
                                  image_output_usd_per_million=3)}
    controller = BudgetController(BudgetConfig(max_cost_usd=5, prices=table))
    reservation = controller.reserve("chat", max_output_tokens=2000)
    assert reservation.commit(usage={"input_tokens": 1000, "output_tokens": 2000}) == pytest.approx(0.01)
    assert controller.snapshot()["reserved_cost_usd"] == 0
    usage = {"text_input_tokens": 100, "image_input_tokens": 200, "image_output_tokens": 300}
    assert controller.reserve("image").commit(usage=usage) == pytest.approx(0.0014)
    for usage in (None, {"input_tokens": 0}, {"input_tokens": -1, "output_tokens": 1}):
        assert controller.reserve("chat").commit(usage=usage) == 1
    assert BudgetController(BudgetConfig(prices=prices())).reserve("chat").commit(
        usage={"input_tokens": 0, "output_tokens": 0}) == 1
    usage = {"input_tokens": 1000, "cached_input_tokens": 600, "output_tokens": 100}
    assert cached_chat().reserve("chat").commit(usage=usage) == pytest.approx(.0072)
    full = BudgetController(BudgetConfig(prices={"chat": PriceConfig(1, input_usd_per_million=10, output_usd_per_million=20)}))
    assert full.reserve("chat").commit(usage=usage) == pytest.approx(.012)
    for cached in (-1, 1001, True, 1.5, "20", None):
        value = dict(usage, cached_input_tokens=cached) if cached is not None else {"input_tokens": 1000, "output_tokens": 100}
        assert cached_chat().reserve("chat").commit(usage=value) == 1


def test_cached_rate_yaml_is_validated(tmp_path):
    path = tmp_path / "prices.yaml"
    path.write_text("budget:\n  prices:\n    chat:\n      upper_bound_usd: 1\n      input_usd_per_million: 10\n      cached_input_usd_per_million: 2\n      output_usd_per_million: 20\n")
    config = load_config(path, environ={"WMM_LLM_API_KEY": "review-test-credential",
                         "WMM_LLM_MODEL": "test-chat", "WMM_IMAGE_MODEL": "test-image",
                         "WMM_LLM_BASE_URL": "https://provider.invalid/v1"})
    assert config.budget.prices["chat"].cached_input_usd_per_million == 2
    with pytest.raises(ConfigError):
        PriceConfig(1, cached_input_usd_per_million=-1)


def test_bound_violation_and_inconsistent_output_bound_stop_dispatch():
    table = {**prices(), "chat": PriceConfig(1, input_usd_per_million=100000, output_usd_per_million=100)}
    controller = BudgetController(BudgetConfig(max_cost_usd=10, prices=table))
    with pytest.raises(BudgetError, match="no request was sent"):
        controller.reserve("chat", max_output_tokens=20000)
    assert controller.snapshot()["reserved_cost_usd"] == 0
    with pytest.raises(PriceBoundExceeded):
        controller.reserve("chat").commit(usage={"input_tokens": 20, "output_tokens": 0})
    assert controller.snapshot()["accounted_cost_usd"] == 2 and controller.snapshot()["stopped"] is True
    with pytest.raises(BudgetExceeded):
        controller.reserve("chat")


def test_parallel_reservations_share_the_same_cap():
    controller = BudgetController(BudgetConfig(max_cost_usd=3, prices=prices()))
    def attempt(_):
        try:
            return controller.reserve("chat")
        except BudgetExceeded:
            return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        reservations = [value for value in pool.map(attempt, range(8)) if value is not None]
    assert len(reservations) == 3 and controller.snapshot()["reserved_cost_usd"] == 3


def test_resume_charges_pending_bounds_and_keeps_spent():
    config = BudgetConfig(max_cost_usd=3, prices=prices())
    controller = BudgetController(config)
    controller.reserve("chat")
    controller.reserve("image")
    saved = []
    restored = BudgetController.from_snapshot(config, controller.snapshot(), checkpoint=saved.append)
    assert restored.snapshot()["accounted_cost_usd"] == 3 and saved[0]["pending_reservations"] == []
    assert restored.snapshot()["upper_bound_calls"] == restored.snapshot()["completed_calls"] == 2
    with pytest.raises(BudgetExceeded, match="no request was sent"):
        restored.reserve("chat")
    assert BudgetController.from_snapshot(config, restored.snapshot()).snapshot()["completed_calls"] == 2
    precise = BudgetController(BudgetConfig(max_cost_usd=0.3, prices=prices(chat=0.1)))
    for _ in range(3):
        precise.reserve("chat").commit()
    assert precise.snapshot()["accounted_cost_usd_exact"] == "0.3"
    assert BudgetController.from_snapshot(precise.config, precise.snapshot()).snapshot()["accounted_cost_usd"] == 0.3
    unknown = BudgetController()
    unknown.reserve("chat")
    with pytest.raises(BudgetError, match="unknown historical costs"):
        BudgetController.from_snapshot(BudgetConfig(max_cost_usd=10, prices=prices()), unknown.snapshot())


def test_checkpoint_failures_before_and_after_dispatch():
    secret = "test-only-checkpoint-secret"
    def fail(_):
        raise OSError(secret)
    controller = BudgetController(BudgetConfig(max_cost_usd=3, prices=prices()), checkpoint=fail)
    with pytest.raises(BudgetCheckpointError, match="no request was sent") as exc:
        controller.reserve("chat")
    assert controller.snapshot()["accounted_cost_usd"] == 0 and controller.snapshot()["pending_reservations"] == []
    assert controller.snapshot()["stopped"] is True
    assert secret not in "".join(traceback.format_exception(type(exc.value), exc.value, exc.value.__traceback__))
    saved = []
    def once(snapshot):
        if saved:
            raise OSError(secret)
        saved.append(snapshot)
    controller = BudgetController(BudgetConfig(max_cost_usd=3, prices=prices()), checkpoint=once)
    with pytest.raises(BudgetCheckpointError, match="charges remain accounted"):
        controller.reserve("chat").commit()
    assert controller.snapshot()["accounted_cost_usd"] == 1
    with pytest.raises(BudgetCheckpointError):
        controller.reserve("chat")
    assert BudgetController.from_snapshot(controller.config, saved[0]).snapshot()["accounted_cost_usd"] == 1


@pytest.mark.parametrize("mutate", [
    lambda state: state.pop("schema"),
    lambda state: state.update(unknown_calls=1),
    lambda state: state.update(reserved_cost_usd=1, reserved_cost_usd_exact="1"),
    lambda state: state.update(pending_reservations=[{"id": 1}]),
    lambda state: state.update(accounted_cost_usd_exact="test-only-snapshot-secret"),
])
def test_invalid_snapshot_fails_locally_without_echo(mutate):
    snapshot = BudgetController().snapshot()
    mutate(snapshot)
    with pytest.raises(BudgetError, match="Invalid budget snapshot") as exc:
        BudgetController.from_snapshot(BudgetConfig(), snapshot)
    assert "test-only-snapshot-secret" not in str(exc.value)
