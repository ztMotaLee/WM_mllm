"""Conservative per-attempt accounting using only user-provided prices.

A reservation is made immediately before each transport attempt. Dispatch
failures and incomplete usage retain the entire configured upper bound. A
reservation can be released only when the caller knows no request was sent.

For resumable sessions, the caller must supply a checkpoint callback that atomically
stores each snapshot under the run root. The callback runs after reservation
and before dispatch is permitted, and after every settlement. Restore with
BudgetController.from_snapshot; outstanding attempts are charged their full
bounds because their outcomes are uncertain. This module performs no file I/O.
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
    if not isinstance(usage, Mapping):
        return None
    if role == "chat":
        counts = (_tokens(usage.get("input_tokens", usage.get("prompt_tokens"))),
                  _tokens(usage.get("output_tokens", usage.get("completion_tokens"))))
        rates = (price.input_usd_per_million, price.output_usd_per_million)
        details = usage.get("input_tokens_details", usage.get("prompt_tokens_details"))
        if not isinstance(details, Mapping):
            details = {}
        raw_cached = usage.get("cached_input_tokens", details.get("cached_tokens"))
        has_cached = raw_cached is not None
        cached = _tokens(raw_cached)
        if has_cached and (cached is None or counts[0] is None or cached > counts[0]):
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
        input_details = usage.get("input_tokens_details") or usage.get("input_token_details") or {}
        output_details = usage.get("output_tokens_details") or {}
        if not isinstance(input_details, Mapping):
            input_details = {}
        if not isinstance(output_details, Mapping):
            output_details = {}
        counts = (_tokens(usage.get("text_input_tokens", input_details.get("text_tokens"))),
                  _tokens(usage.get("image_input_tokens", input_details.get("image_tokens"))),
                  _tokens(usage.get("image_output_tokens", output_details.get("image_tokens", usage.get("output_tokens")))))
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
        self.cost_status = "reserved"

    @property
    def upper_bound_usd(self) -> float | None:
        return None if self._bound is None else float(self._bound)

    def commit(self, *, usage: Mapping[str, Any] | None = None,
               input_tokens: int | None = None, output_tokens: int | None = None,
               cost_usd: float | None = None) -> float | None:
        """Charge measured usage, or the bound when usage/prices are incomplete.

        Repeated commits return the first charge without double-counting. If
        accounting is uncapped and has no user prices, unknown cost stays None.
        """
        return self._controller._commit(self, usage=usage, input_tokens=input_tokens,
                                        output_tokens=output_tokens, cost_usd=cost_usd)

    def release(self) -> None:
        """Release only a reservation for a provably undispatched request."""
        self._controller._release(self)


class BudgetController:
    """Thread-safe accounting shared by chat and image providers in one run.

    A resumable session must provide checkpoint. It must atomically persist
    the supplied detached snapshot before returning and must not dispatch
    requests or mutate this controller. Callback execution holds the lock.
    Resume the saved run with from_snapshot instead of constructing a new
    controller, which represents a genuinely new run with no historical spend.
    """

    def __init__(self, config: BudgetConfig | None = None, *,
                 checkpoint: Callable[[dict[str, Any]], None] | None = None) -> None:
        if checkpoint is not None and not callable(checkpoint):
            raise BudgetError("Budget checkpoint must be callable.")
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
        """Restore a run without resetting spent or uncertain attempt charges.

        Snapshot storage and atomic replacement belong to the session caller.
        Supplying checkpoint persists the restored, conservatively settled
        state before this method returns. A capped resume refuses any old
        unpriced attempt until its cost is explicitly reconciled by the caller.
        Invalid state produces one sanitized local error without input values.
        """
        controller = cls(config, checkpoint=checkpoint)
        try:
            previous_cap = controller._restore(snapshot)
        except (BudgetError, InvalidOperation, KeyError, TypeError, ValueError, OverflowError):
            raise BudgetError("Invalid budget snapshot; resume stopped before any request.") from None
        if previous_cap is not None and controller._cap is None:
            raise BudgetError("Resuming a capped run requires an explicit spend cap; no request was sent.")
        if controller._cap is not None and controller._unknown_calls:
            raise BudgetError("Cannot enable a spend cap with unknown historical costs; reconcile them before resuming; no request was sent.")
        if controller._cap is not None and controller._spent > controller._cap:
            raise BudgetExceeded("Restored charges exceed budget.max_cost_usd; no request was sent.")
        with controller._lock:
            controller._save_checkpoint(before_dispatch=True)
        return controller

    @staticmethod
    def _restored_cost(state: Mapping[str, Any], name: str) -> Decimal | None:
        numeric, exact = state[name], state[f"{name}_exact"]
        if numeric is None or exact is None:
            if numeric is not None or exact is not None:
                raise BudgetError("Invalid budget snapshot.")
            return None
        if not isinstance(exact, str) or not 1 <= len(exact) <= 128:
            raise BudgetError("Invalid budget snapshot.")
        amount = Decimal(exact)
        if not amount.is_finite() or amount < 0 or _decimal(numeric) != Decimal(str(float(amount))):
            raise BudgetError("Invalid budget snapshot.")
        return amount

    def _restore(self, state: Mapping[str, Any]) -> Decimal | None:
        if not isinstance(state, Mapping) or state.get("schema") != _SNAPSHOT_SCHEMA:
            raise BudgetError("Invalid budget snapshot.")
        required = {"schema", "max_cost_usd", "max_cost_usd_exact", "accounted_cost_usd",
                    "accounted_cost_usd_exact", "total_cost_usd", "reserved_cost_usd",
                    "reserved_cost_usd_exact", "unknown_calls", "upper_bound_calls",
                    "completed_calls", "stopped", "checkpoint_failed", "pending_reservations",
                    "next_reservation_id"}
        if set(state) != required:
            raise BudgetError("Invalid budget snapshot.")
        old_cap = self._restored_cost(state, "max_cost_usd")
        spent = self._restored_cost(state, "accounted_cost_usd")
        reserved = self._restored_cost(state, "reserved_cost_usd")
        if spent is None or reserved is None:
            raise BudgetError("Invalid budget snapshot.")
        counters = [state[name] for name in ("unknown_calls", "upper_bound_calls", "completed_calls")]
        if any(_tokens(value) is None for value in counters) or counters[0] + counters[1] > counters[2]:
            raise BudgetError("Invalid budget snapshot.")
        if old_cap is not None and counters[0]:
            raise BudgetError("Invalid budget snapshot.")
        if not isinstance(state["stopped"], bool) or not isinstance(state["checkpoint_failed"], bool):
            raise BudgetError("Invalid budget snapshot.")
        total = state["total_cost_usd"]
        if (counters[0] and total is not None) or (not counters[0] and (total is None or _decimal(total) != Decimal(str(float(spent))))):
            raise BudgetError("Invalid budget snapshot.")
        pending = state["pending_reservations"]
        if not isinstance(pending, list) or len(pending) > _MAX_PENDING_RESERVATIONS:
            raise BudgetError("Invalid budget snapshot.")
        ids: set[int] = set()
        pending_total = Decimal(0)
        unknown_pending = 0
        for entry in pending:
            if not isinstance(entry, Mapping) or set(entry) != {"id", "role", "upper_bound_usd", "upper_bound_usd_exact"}:
                raise BudgetError("Invalid budget snapshot.")
            number = entry["id"]
            if _tokens(number) is None or number == 0 or number in ids or entry["role"] not in ("chat", "image"):
                raise BudgetError("Invalid budget snapshot.")
            ids.add(number)
            bound = self._restored_cost(entry, "upper_bound_usd")
            if bound is None:
                if old_cap is not None:
                    raise BudgetError("Invalid budget snapshot.")
                unknown_pending += 1
            elif bound == 0:
                raise BudgetError("Invalid budget snapshot.")
            else:
                pending_total += bound
        next_id = state["next_reservation_id"]
        if _tokens(next_id) is None or next_id <= max(ids, default=0) or next_id == 0 or pending_total != reserved:
            raise BudgetError("Invalid budget snapshot.")
        self._spent = spent + pending_total
        self._reserved = Decimal(0)
        self._unknown_calls = counters[0] + unknown_pending
        self._upper_bound_calls = counters[1] + len(pending) - unknown_pending
        self._completed_calls = counters[2] + len(pending)
        self._blocked = state["stopped"]
        self._checkpoint_failed = state["checkpoint_failed"]
        self._next_reservation_id = next_id
        return old_cap

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

    @property
    def spent_usd(self) -> float:
        """Accounted charges; see snapshot unknown_calls for unpriced attempts."""
        with self._lock:
            return float(self._spent)

    @property
    def reserved_usd(self) -> float:
        with self._lock:
            return float(self._reserved)

    def reserve(self, role: str, *, model: str | None = None,
                max_input_tokens: int | None = None,
                max_output_tokens: int | None = None) -> Reservation:
        """Reserve the configured per-attempt bound before any network dispatch.

        role is chat/image, independent of a caller's semantic role. Model is
        accepted for the provider interface and is never persisted. Optional
        token bounds detect an inconsistent user bound before dispatch.
        """
        if role not in ("chat", "image"):
            raise BudgetError("Budget role must be chat or image.")
        for value in (max_input_tokens, max_output_tokens):
            if value is not None and _tokens(value) is None:
                raise BudgetError("Token bounds must be nonnegative integers.")
        price = self.config.prices.get(role)
        bound = None if price is None else _decimal(price.upper_bound_usd)
        if role == "chat" and price is not None:
            known_token_bound = sum((Decimal(count) * _decimal(rate) for count, rate in (
                (max_input_tokens, price.input_usd_per_million),
                (max_output_tokens, price.output_usd_per_million))
                if count is not None and rate is not None), Decimal(0)) / Decimal(1_000_000)
            if known_token_bound > bound:
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
                reservation.cost_status = "undispatched"
                reservation.charge_usd = 0.0
                raise
            return reservation

    def _commit(self, reservation: Reservation, *, usage: Mapping[str, Any] | None,
                input_tokens: int | None, output_tokens: int | None,
                cost_usd: float | None) -> float | None:
        with self._lock:
            if reservation._finished:
                return reservation.charge_usd
            price = self.config.prices.get(reservation.role)
            if usage is None and (input_tokens is not None or output_tokens is not None):
                usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
            measured = _decimal(cost_usd) if cost_usd is not None else (
                _usage_cost(reservation.role, price, usage) if price is not None else None)
            charge = reservation._bound if measured is None else measured
            if reservation._bound is not None:
                self._reserved -= reservation._bound
            reservation._finished = True
            self._pending.pop(reservation.reservation_id)
            reservation.charge_usd = None if charge is None else float(charge)
            reservation.cost_status = "unknown" if charge is None else ("upper_bound" if measured is None else "measured")
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

    def _release(self, reservation: Reservation) -> None:
        with self._lock:
            if reservation._finished:
                return
            if reservation._bound is not None:
                self._reserved -= reservation._bound
            reservation._finished = True
            self._pending.pop(reservation.reservation_id)
            reservation.cost_status = "undispatched"
            reservation.charge_usd = 0.0
            self._save_checkpoint()

    def snapshot(self) -> dict[str, Any]:
        """Return detached JSON-compatible state for atomic run-root storage.

        Do not reconstruct this state from totals alone. Pending reservations
        and exact decimal amounts are required for a conservative resume.
        """
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
