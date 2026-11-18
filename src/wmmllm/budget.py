"""Conservative per-attempt accounting using only user-provided prices.

A reservation is made immediately before each transport attempt. Dispatch
failures and incomplete usage retain the entire configured upper bound. A
resumable session checkpoints every snapshot; ``from_snapshot`` charges
outstanding attempts their full bounds. This module performs no file I/O.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import threading
from typing import Any, Callable, Mapping

from .config import BudgetConfig, PriceConfig


class BudgetError(RuntimeError):
    """Invalid accounting information; messages contain no provider payloads."""


class BudgetExceeded(BudgetError):
    """The next attempt is blocked before dispatch by the configured cap."""


class PriceBoundExceeded(BudgetExceeded):
    """Observed cost exceeded a user bound; further dispatch is blocked."""


class BudgetCheckpointError(BudgetError):
    """Atomic checkpoint failure; the controller stops further dispatch."""


_SNAPSHOT_SCHEMA = "wmmllm.budget.v1"
_SNAPSHOT_FIELDS = {"schema", "max_cost_usd", "max_cost_usd_exact", "accounted_cost_usd",
                    "accounted_cost_usd_exact", "total_cost_usd", "reserved_cost_usd",
                    "reserved_cost_usd_exact", "unknown_calls", "upper_bound_calls",
                    "completed_calls", "stopped", "checkpoint_failed", "pending_reservations",
                    "next_reservation_id"}
_MAX_PENDING_RESERVATIONS = 10000


def _decimal(value: int | float | Decimal) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise BudgetError("Cost must be a finite nonnegative number.")
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise BudgetError("Cost must be a finite nonnegative number.")
    return result


def _tokens(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _usage_cost(role: str, price: PriceConfig, usage: Mapping[str, Any] | None) -> Decimal | None:
    """Price a ``Usage.to_record()`` mapping; None when counts or rates are incomplete."""
    if not usage:
        return None
    if role == "chat":
        counts = (_tokens(usage.get("input_tokens")), _tokens(usage.get("output_tokens")))
        rates = (price.input_usd_per_million, price.output_usd_per_million)
        raw_cached = usage.get("cached_input_tokens")
        cached = _tokens(raw_cached)
        if raw_cached is not None and (cached is None or counts[0] is None or cached > counts[0]):
            return None
        if price.cached_input_usd_per_million is not None:
            # A missing count cannot establish how many tokens used either rate.
            if cached is None:
                return None
            counts = (None if counts[0] is None else counts[0] - cached, cached, counts[1])
            rates = (price.input_usd_per_million, price.cached_input_usd_per_million,
                     price.output_usd_per_million)
        # Without a cached rate, charge all input at the full input rate.
    else:
        counts = (_tokens(usage.get("text_input_tokens")), _tokens(usage.get("image_input_tokens")),
                  _tokens(usage.get("image_output_tokens")))
        rates = (price.text_input_usd_per_million, price.image_input_usd_per_million,
                 price.image_output_usd_per_million)
        if any(count is None for count in counts) or any(rate is None for rate in rates):
            counts = (_tokens(usage.get("input_tokens")), _tokens(usage.get("output_tokens")))
            rates = (price.input_usd_per_million, price.output_usd_per_million)
    if any(count is None for count in counts) or any(rate is None for rate in rates):
        return None
    return sum((Decimal(count) * _decimal(rate) for count, rate in zip(counts, rates)), Decimal(0)) / Decimal(1_000_000)


class Reservation:
    """One dispatched attempt, never a whole retry loop."""

    def __init__(self, controller: BudgetController, role: str, bound: Decimal | None,
                 reservation_id: int) -> None:
        self._controller = controller
        self.reservation_id = reservation_id
        self.role = role
        self._bound = bound
        self._finished = False
        self.charge_usd: float | None = None

    @property
    def upper_bound_usd(self) -> float | None:
        return None if self._bound is None else float(self._bound)

    def commit(self, *, usage: Mapping[str, Any] | None = None) -> float | None:
        """Charge measured usage, or the bound when usage/prices are incomplete.

        Repeated commits return the first charge. Without user prices and a
        cap, unknown cost stays None.
        """
        return self._controller._commit(self, usage)


class BudgetController:
    """Thread-safe accounting shared by chat and image providers in one run.

    The checkpoint callback atomically persists each snapshot under the lock.
    """

    def __init__(self, config: BudgetConfig | None = None, *,
                 checkpoint: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.config = config or BudgetConfig()
        self._checkpoint = checkpoint
        self._cap = None if self.config.max_cost_usd is None else _decimal(self.config.max_cost_usd)
        self._spent = Decimal(0)
        self._reserved = Decimal(0)
        self._unknown_calls = 0
        self._upper_bound_calls = 0
        self._completed_calls = 0
        self._blocked = False
        self._checkpoint_failed = False
        self._pending: dict[int, Reservation] = {}
        self._next_reservation_id = 1
        self._lock = threading.RLock()

    @classmethod
    def from_snapshot(cls, config: BudgetConfig, snapshot: Mapping[str, Any], *,
                      checkpoint: Callable[[dict[str, Any]], None] | None = None) -> BudgetController:
        """Restore spent and pending charges; invalid state fails without echoing input."""
        controller = cls(config, checkpoint=checkpoint)

        def require(condition: Any) -> None:
            if not condition:
                raise BudgetError("Invalid budget snapshot.")

        def exact(state: Mapping[str, Any], name: str) -> Decimal | None:
            numeric, text = state[name], state[f"{name}_exact"]
            if numeric is None or text is None:
                require(numeric is None and text is None)
                return None
            require(isinstance(text, str) and 1 <= len(text) <= 128)
            amount = Decimal(text)
            require(amount.is_finite() and amount >= 0 and _decimal(numeric) == Decimal(str(float(amount))))
            return amount

        try:
            require(isinstance(snapshot, Mapping) and snapshot.get("schema") == _SNAPSHOT_SCHEMA
                    and set(snapshot) == _SNAPSHOT_FIELDS)
            old_cap = exact(snapshot, "max_cost_usd")
            spent, reserved = exact(snapshot, "accounted_cost_usd"), exact(snapshot, "reserved_cost_usd")
            require(spent is not None and reserved is not None)
            unknown, upper, completed = (snapshot[name] for name in ("unknown_calls", "upper_bound_calls", "completed_calls"))
            require(all(_tokens(value) is not None for value in (unknown, upper, completed))
                    and unknown + upper <= completed and not (old_cap is not None and unknown))
            require(isinstance(snapshot["stopped"], bool) and isinstance(snapshot["checkpoint_failed"], bool))
            total = snapshot["total_cost_usd"]
            require(not ((unknown and total is not None) or
                         (not unknown and (total is None or _decimal(total) != Decimal(str(float(spent)))))))
            pending = snapshot["pending_reservations"]
            require(isinstance(pending, list) and len(pending) <= _MAX_PENDING_RESERVATIONS)
            ids: set[int] = set()
            pending_total = Decimal(0)
            unknown_pending = 0
            for entry in pending:
                require(isinstance(entry, Mapping) and set(entry) == {"id", "role", "upper_bound_usd", "upper_bound_usd_exact"})
                number = entry["id"]
                require(_tokens(number) is not None and number != 0 and number not in ids
                        and entry["role"] in ("chat", "image"))
                ids.add(number)
                bound = exact(entry, "upper_bound_usd")
                if bound is None:
                    require(old_cap is None)
                    unknown_pending += 1
                else:
                    require(bound != 0)
                    pending_total += bound
            next_id = snapshot["next_reservation_id"]
            require(_tokens(next_id) is not None and next_id > max(ids, default=0) and next_id != 0
                    and pending_total == reserved)
        except (BudgetError, InvalidOperation, KeyError, TypeError, ValueError, OverflowError):
            raise BudgetError("Invalid budget snapshot; resume stopped before any request.") from None
        controller._spent = spent + pending_total
        controller._unknown_calls = unknown + unknown_pending
        controller._upper_bound_calls = upper + len(pending) - unknown_pending
        controller._completed_calls = completed + len(pending)
        controller._blocked = snapshot["stopped"]
        controller._checkpoint_failed = snapshot["checkpoint_failed"]
        controller._next_reservation_id = next_id
        if old_cap is not None and controller._cap is None:
            raise BudgetError("Resuming a capped run requires an explicit spend cap; no request was sent.")
        if controller._cap is not None and controller._unknown_calls:
            raise BudgetError("Cannot enable a spend cap with unknown historical costs; reconcile them before resuming; no request was sent.")
        if controller._cap is not None and controller._spent > controller._cap:
            raise BudgetExceeded("Restored charges exceed budget.max_cost_usd; no request was sent.")
        with controller._lock:
            controller._save_checkpoint(before_dispatch=True)
        return controller

    def _save_checkpoint(self, *, before_dispatch: bool = False) -> None:
        if self._checkpoint is None:
            return
        try:
            self._checkpoint(self.snapshot())
        except Exception:
            self._checkpoint_failed = True
            message = ("Budget checkpoint failed before dispatch; no request was sent."
                       if before_dispatch else
                       "Budget checkpoint failed after settlement; charges remain accounted and further requests are stopped.")
            raise BudgetCheckpointError(message) from None

    def reserve(self, role: str, *, max_output_tokens: int | None = None) -> Reservation:
        """Reserve the per-attempt bound before dispatch; reject a bound below the output-token cost."""
        if role not in ("chat", "image"):
            raise BudgetError("Budget role must be chat or image.")
        if max_output_tokens is not None and _tokens(max_output_tokens) is None:
            raise BudgetError("Token bounds must be nonnegative integers.")
        price = self.config.prices.get(role)
        bound = None if price is None else _decimal(price.upper_bound_usd)
        if (role == "chat" and price is not None and max_output_tokens is not None
                and price.output_usd_per_million is not None
                and Decimal(max_output_tokens) * _decimal(price.output_usd_per_million) / Decimal(1_000_000) > bound):
            raise BudgetError("User per-attempt bound is below the supplied token upper bounds; no request was sent.")
        with self._lock:
            if self._checkpoint_failed:
                raise BudgetCheckpointError("Budget stopped after checkpoint failure; no request was sent.")
            if self._blocked:
                raise BudgetExceeded("Budget stopped after a user price bound was exceeded; no request was sent.")
            if self._cap is not None and (bound is None or self._spent + self._reserved + bound > self._cap):
                raise BudgetExceeded("The next attempt could exceed budget.max_cost_usd; no request was sent.")
            if len(self._pending) >= _MAX_PENDING_RESERVATIONS:
                raise BudgetError("Too many pending budget reservations; no request was sent.")
            reservation = Reservation(self, role, bound, self._next_reservation_id)
            self._next_reservation_id += 1
            self._pending[reservation.reservation_id] = reservation
            if bound is not None:
                self._reserved += bound
            try:
                self._save_checkpoint(before_dispatch=True)
            except BudgetCheckpointError:
                self._pending.pop(reservation.reservation_id)
                if bound is not None:
                    self._reserved -= bound
                reservation._finished = True
                reservation.charge_usd = 0.0
                raise
            return reservation

    def _commit(self, reservation: Reservation, usage: Mapping[str, Any] | None) -> float | None:
        with self._lock:
            if reservation._finished:
                return reservation.charge_usd
            price = self.config.prices.get(reservation.role)
            measured = _usage_cost(reservation.role, price, usage) if price is not None else None
            charge = reservation._bound if measured is None else measured
            if reservation._bound is not None:
                self._reserved -= reservation._bound
            reservation._finished = True
            self._pending.pop(reservation.reservation_id)
            reservation.charge_usd = None if charge is None else float(charge)
            self._completed_calls += 1
            if charge is None:
                self._unknown_calls += 1
            else:
                self._spent += charge
            if measured is None and charge is not None:
                self._upper_bound_calls += 1
            if measured is not None and reservation._bound is not None and measured > reservation._bound:
                self._blocked = True
                self._save_checkpoint()
                raise PriceBoundExceeded("Observed cost exceeded the user per-attempt bound; accounting retained the cost and further requests are stopped.")
            self._save_checkpoint()
            return reservation.charge_usd

    def snapshot(self) -> dict[str, Any]:
        """Return detached JSON-compatible state, including pending reservations and exact amounts."""
        with self._lock:
            pending = [{"id": item.reservation_id, "role": item.role,
                        "upper_bound_usd": item.upper_bound_usd,
                        "upper_bound_usd_exact": None if item._bound is None else str(item._bound)}
                       for item in self._pending.values()]
            return {"schema": _SNAPSHOT_SCHEMA,
                    "max_cost_usd": None if self._cap is None else float(self._cap),
                    "max_cost_usd_exact": None if self._cap is None else str(self._cap),
                    "accounted_cost_usd": float(self._spent),
                    "accounted_cost_usd_exact": str(self._spent),
                    "total_cost_usd": None if self._unknown_calls else float(self._spent),
                    "reserved_cost_usd": float(self._reserved),
                    "reserved_cost_usd_exact": str(self._reserved),
                    "unknown_calls": self._unknown_calls,
                    "upper_bound_calls": self._upper_bound_calls,
                    "completed_calls": self._completed_calls, "stopped": self._blocked or self._checkpoint_failed,
                    "checkpoint_failed": self._checkpoint_failed,
                    "pending_reservations": pending,
                    "next_reservation_id": self._next_reservation_id}
