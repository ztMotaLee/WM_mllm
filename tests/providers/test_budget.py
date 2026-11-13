"""Offline budget reservations use synthetic user prices, not provider tariffs."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from wmmllm.budget import BudgetCheckpointError, BudgetController, BudgetError, BudgetExceeded, PriceBoundExceeded
from wmmllm.config import BudgetConfig, ConfigError, PriceConfig


def prices(chat=1.0, image=2.0):
    return {"chat": PriceConfig(chat), "image": PriceConfig(image)}


def test_cap_requires_prices_at_startup():
    with pytest.raises(ConfigError, match="requires user prices"):
        BudgetConfig(max_cost_usd=1)


def test_resume_cannot_silently_disable_a_previously_configured_cap():
    controller = BudgetController(BudgetConfig(max_cost_usd=1, prices=prices()))
    with pytest.raises(BudgetError, match="Resuming a capped run requires an explicit spend cap"):
        BudgetController.from_snapshot(BudgetConfig(), controller.snapshot())


def test_reserve_stops_before_the_attempt_would_exceed_cap():
    controller = BudgetController(BudgetConfig(max_cost_usd=2, prices=prices()))
    first = controller.reserve("chat")
    assert controller.reserved_usd == 1
    with pytest.raises(BudgetExceeded, match="no request was sent"):
        controller.reserve("image")
    first.commit()
    assert controller.spent_usd == 1
    second = controller.reserve("chat")
    second.commit()
    with pytest.raises(BudgetExceeded):
        controller.reserve("chat")
    assert controller.snapshot()["upper_bound_calls"] == 2


def test_uncertain_failure_charges_every_retry_attempt():
    controller = BudgetController(BudgetConfig(max_cost_usd=3, prices=prices()))
    for _ in range(3):
        controller.reserve("chat").commit(usage=None)
    with pytest.raises(BudgetExceeded):
        controller.reserve("chat")
    assert controller.snapshot()["accounted_cost_usd"] == 3
    assert controller.snapshot()["completed_calls"] == 3


def test_missing_usage_stays_unknown_when_uncapped_without_prices():
    controller = BudgetController()
    assert controller.reserve("chat").commit() is None
    snapshot = controller.snapshot()
    assert snapshot["total_cost_usd"] is None
    assert snapshot["unknown_calls"] == 1


def test_complete_chat_usage_releases_unused_reservation():
    table = {**prices(), "chat": PriceConfig(1, input_usd_per_million=2, output_usd_per_million=4)}
    controller = BudgetController(BudgetConfig(max_cost_usd=2, prices=table))
    reservation = controller.reserve("chat", max_input_tokens=1000, max_output_tokens=2000)
    assert reservation.commit(input_tokens=1000, output_tokens=2000) == pytest.approx(0.01)
    assert reservation.cost_status == "measured"
    assert controller.spent_usd == pytest.approx(0.01)
    assert controller.reserved_usd == 0
    controller.reserve("chat").release()


@pytest.mark.parametrize("usage", [None, {}, {"input_tokens": 0},
    {"input_tokens": -1, "output_tokens": 1}, {"input_tokens": True, "output_tokens": 0}])
def test_incomplete_or_invalid_usage_retains_bound(usage):
    table = {**prices(), "chat": PriceConfig(1, input_usd_per_million=2, output_usd_per_million=4)}
    controller = BudgetController(BudgetConfig(max_cost_usd=1, prices=table))
    assert controller.reserve("chat").commit(usage=usage) == 1
    assert controller.snapshot()["upper_bound_calls"] == 1


def test_missing_token_prices_never_turn_usage_into_zero_cost():
    controller = BudgetController(BudgetConfig(max_cost_usd=1, prices=prices()))
    assert controller.reserve("chat").commit(input_tokens=0, output_tokens=0) == 1


def test_image_usage_prices_each_modality():
    table = {**prices(), "image": PriceConfig(2, text_input_usd_per_million=1,
             image_input_usd_per_million=2, image_output_usd_per_million=3)}
    controller = BudgetController(BudgetConfig(max_cost_usd=3, prices=table))
    assert controller.reserve("image").commit(usage={"text_input_tokens": 100,
        "image_input_tokens": 200, "image_output_tokens": 300}) == pytest.approx(0.0014)


def test_raw_sdk_image_usage_has_same_modality_cost():
    table = {**prices(), "image": PriceConfig(2, text_input_usd_per_million=1,
             image_input_usd_per_million=2, image_output_usd_per_million=3)}
    controller = BudgetController(BudgetConfig(max_cost_usd=3, prices=table))
    assert controller.reserve("image").commit(usage={"input_tokens_details": {
        "text_tokens": 100, "image_tokens": 200}, "output_tokens": 300}) == pytest.approx(0.0014)


def test_release_is_only_for_an_undispatched_attempt_and_commit_is_idempotent():
    controller = BudgetController(BudgetConfig(max_cost_usd=1, prices=prices()))
    reservation = controller.reserve("chat")
    reservation.release()
    assert controller.reserved_usd == 0
    assert controller.spent_usd == 0
    assert reservation.cost_status == "undispatched"
    assert reservation.commit() == 0
    reservation = controller.reserve("chat")
    assert reservation.commit() == 1
    assert reservation.commit() == 1
    assert controller.spent_usd == 1


def test_user_bound_violation_charges_actual_and_stops_future_calls():
    controller = BudgetController(BudgetConfig(max_cost_usd=10, prices=prices()))
    with pytest.raises(PriceBoundExceeded):
        controller.reserve("chat").commit(cost_usd=2)
    assert controller.spent_usd == 2
    assert controller.snapshot()["stopped"] is True
    with pytest.raises(BudgetExceeded):
        controller.reserve("chat")


def test_inconsistent_user_token_upper_bound_fails_before_call():
    table = {**prices(), "chat": PriceConfig(1, input_usd_per_million=100,
                                             output_usd_per_million=100)}
    controller = BudgetController(BudgetConfig(max_cost_usd=10, prices=table))
    with pytest.raises(BudgetError, match="no request was sent"):
        controller.reserve("chat", max_input_tokens=10000, max_output_tokens=10000)
    assert controller.reserved_usd == 0


def test_parallel_reservations_share_the_same_cap():
    controller = BudgetController(BudgetConfig(max_cost_usd=3, prices=prices()))
    def attempt(_):
        try:
            return controller.reserve("chat")
        except BudgetExceeded:
            return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        reservations = list(pool.map(attempt, range(8)))
    assert len([value for value in reservations if value is not None]) == 3
    assert controller.reserved_usd == 3
    for reservation in reservations:
        if reservation is not None:
            reservation.commit()
    assert controller.spent_usd == 3


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), True])
def test_invalid_user_bounds_are_rejected(value):
    with pytest.raises(ConfigError):
        PriceConfig(value)


def test_known_output_bound_alone_rejects_inconsistent_chat_price():
    table = {**prices(), "chat": PriceConfig(1, output_usd_per_million=100)}
    controller = BudgetController(BudgetConfig(max_cost_usd=10, prices=table))
    with pytest.raises(BudgetError, match="no request was sent"):
        controller.reserve("chat", max_output_tokens=20000)
    assert controller.reserved_usd == 0


def test_unknown_input_bound_retains_full_user_reservation():
    table = {**prices(), "chat": PriceConfig(1, output_usd_per_million=1)}
    controller = BudgetController(BudgetConfig(max_cost_usd=10, prices=table))
    reservation = controller.reserve("chat", max_output_tokens=20000)
    assert reservation.upper_bound_usd == 1
    assert reservation.commit() == 1


def test_resume_retains_spent_and_blocks_the_next_attempt():
    config = BudgetConfig(max_cost_usd=1, prices=prices())
    controller = BudgetController(config)
    controller.reserve("chat").commit()
    restored = BudgetController.from_snapshot(config, controller.snapshot())
    assert restored.spent_usd == 1
    assert restored.snapshot()["completed_calls"] == 1
    with pytest.raises(BudgetExceeded, match="no request was sent"):
        restored.reserve("chat")


def test_resume_charges_pending_bounds_and_never_releases_them():
    config = BudgetConfig(max_cost_usd=3, prices=prices())
    controller = BudgetController(config)
    controller.reserve("chat")
    controller.reserve("image")
    before = controller.snapshot()
    assert len(before["pending_reservations"]) == 2
    restored = BudgetController.from_snapshot(config, before)
    assert restored.spent_usd == 3
    assert restored.reserved_usd == 0
    assert restored.snapshot()["pending_reservations"] == []
    assert restored.snapshot()["upper_bound_calls"] == 2
    assert restored.snapshot()["completed_calls"] == 2
    with pytest.raises(BudgetExceeded):
        restored.reserve("chat")
    # Repeated restarts preserve the settled uncertainty exactly once.
    again = BudgetController.from_snapshot(config, restored.snapshot())
    assert again.spent_usd == 3
    assert again.snapshot()["completed_calls"] == 2


def test_resume_keeps_historical_pending_bound_when_prices_change():
    controller = BudgetController(BudgetConfig(prices=prices(chat=2)))
    controller.reserve("chat")
    restored = BudgetController.from_snapshot(
        BudgetConfig(max_cost_usd=5, prices=prices(chat=0.1)), controller.snapshot())
    assert restored.spent_usd == 2


@pytest.mark.parametrize("pending", [False, True])
def test_resume_unpriced_unknown_attempts_prevent_new_cap(pending):
    controller = BudgetController()
    reservation = controller.reserve("chat")
    if not pending:
        reservation.commit()
    snapshot = controller.snapshot()
    restored = BudgetController.from_snapshot(BudgetConfig(), snapshot)
    assert restored.snapshot()["unknown_calls"] == 1
    assert restored.snapshot()["total_cost_usd"] is None
    with pytest.raises(BudgetError, match="unknown historical costs"):
        BudgetController.from_snapshot(BudgetConfig(max_cost_usd=10, prices=prices()), snapshot)


def test_checkpoint_runs_before_reserve_returns_and_after_settlement():
    saved = []
    controller = BudgetController(BudgetConfig(max_cost_usd=3, prices=prices()), checkpoint=saved.append)
    reservation = controller.reserve("chat")
    assert len(saved) == 1
    assert saved[0]["reserved_cost_usd"] == 1
    assert len(saved[0]["pending_reservations"]) == 1
    reservation.commit()
    assert len(saved) == 2
    assert saved[1]["pending_reservations"] == []
    assert saved[1]["accounted_cost_usd"] == 1
    # Snapshots are detached; later settlements cannot mutate the saved state.
    assert len(saved[0]["pending_reservations"]) == 1


def test_restore_checkpoints_conservatively_settled_state_before_return():
    config = BudgetConfig(max_cost_usd=3, prices=prices())
    controller = BudgetController(config)
    controller.reserve("chat")
    saved = []
    restored = BudgetController.from_snapshot(config, controller.snapshot(), checkpoint=saved.append)
    assert len(saved) == 1
    assert saved[0]["pending_reservations"] == []
    assert saved[0]["accounted_cost_usd"] == restored.spent_usd == 1


def test_checkpoint_failure_before_dispatch_is_unbilled_and_sanitized():
    import traceback
    secret = "test-only-checkpoint-secret"
    url = "https://user:password@example.invalid/?token=test-only-checkpoint-secret"
    def fail(_):
        raise OSError(secret + url)
    controller = BudgetController(BudgetConfig(max_cost_usd=3, prices=prices()), checkpoint=fail)
    with pytest.raises(BudgetCheckpointError, match="no request was sent") as exc:
        controller.reserve("chat")
    assert controller.spent_usd == 0
    assert controller.reserved_usd == 0
    assert controller.snapshot()["pending_reservations"] == []
    assert controller.snapshot()["completed_calls"] == 0
    assert controller.snapshot()["stopped"] is True
    text = "".join(traceback.format_exception(type(exc.value), exc.value, exc.value.__traceback__))
    assert secret not in text
    assert url not in text
    with pytest.raises(BudgetCheckpointError):
        controller.reserve("chat")


def test_checkpoint_failure_after_dispatch_retains_charge_and_stops():
    saved = []
    def checkpoint(snapshot):
        if saved:
            raise OSError("test-only-checkpoint-secret")
        saved.append(snapshot)
    controller = BudgetController(BudgetConfig(max_cost_usd=3, prices=prices()), checkpoint=checkpoint)
    reservation = controller.reserve("chat")
    with pytest.raises(BudgetCheckpointError, match="charges remain accounted"):
        reservation.commit()
    assert controller.spent_usd == 1
    assert controller.reserved_usd == 0
    with pytest.raises(BudgetCheckpointError):
        controller.reserve("chat")
    # The last durable checkpoint records the unresolved attempt, preserving its bound.
    restored = BudgetController.from_snapshot(controller.config, saved[0])
    assert restored.spent_usd == 1


def test_release_checkpoints_only_a_known_undispatched_attempt():
    saved = []
    controller = BudgetController(BudgetConfig(prices=prices()), checkpoint=saved.append)
    reservation = controller.reserve("chat")
    reservation.release()
    assert saved[-1]["pending_reservations"] == []
    assert saved[-1]["completed_calls"] == 0
    assert saved[-1]["accounted_cost_usd"] == 0


@pytest.mark.parametrize("mutate", [
    lambda state: state.pop("schema"),
    lambda state: state.update(accounted_cost_usd=-1),
    lambda state: state.update(unknown_calls=1),
    lambda state: state.update(reserved_cost_usd=1, reserved_cost_usd_exact="1"),
    lambda state: state.update(pending_reservations=[{"id": 1}]),
    lambda state: state.update(accounted_cost_usd_exact="test-only-snapshot-secret"),
    lambda state: state.update(next_reservation_id=False),
])
def test_invalid_snapshot_fails_locally_without_echo(mutate):
    controller = BudgetController()
    snapshot = controller.snapshot()
    mutate(snapshot)
    with pytest.raises(BudgetError, match="Invalid budget snapshot") as exc:
        BudgetController.from_snapshot(BudgetConfig(), snapshot)
    assert "test-only-snapshot-secret" not in str(exc.value)


def test_resume_preserves_decimal_accounting_precision():
    config = BudgetConfig(max_cost_usd=0.3, prices=prices(chat=0.1))
    controller = BudgetController(config)
    for _ in range(3):
        controller.reserve("chat").commit()
    snapshot = controller.snapshot()
    assert snapshot["accounted_cost_usd_exact"] == "0.3"
    restored = BudgetController.from_snapshot(config, snapshot)
    assert restored.spent_usd == 0.3
    with pytest.raises(BudgetExceeded):
        restored.reserve("chat")
