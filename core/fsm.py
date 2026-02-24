"""
core/fsm.py — Strict finite state machine for NeuroWeave Sentinel.

Thread-safe FSM with explicit validated transition map, per-state enter/exit
callbacks, transition history (last 50), and structured logging.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from core.constants import FSMState

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Custom exception
# ──────────────────────────────────────────────────────────────

class InvalidTransitionError(RuntimeError):
    """
    Raised when a requested FSM transition is not in the valid transition map.

    Args:
        from_state: Current state at the time of the illegal attempt.
        to_state: Requested (invalid) target state.
        reason: Caller-supplied reason string.
    """

    def __init__(
        self,
        from_state: FSMState,
        to_state: FSMState,
        reason: str = "",
    ) -> None:
        self.from_state = from_state
        self.to_state = to_state
        self.reason = reason
        super().__init__(
            f"Invalid transition {from_state.value} → {to_state.value}"
            + (f" (reason: {reason})" if reason else "")
        )


# ──────────────────────────────────────────────────────────────
# Valid transition map — single source of truth
# ──────────────────────────────────────────────────────────────

_VALID_TRANSITIONS: dict[FSMState, list[FSMState]] = {
    FSMState.IDLE: [
        FSMState.TOKEN_SELECTION,
        FSMState.EMERGENCY,
    ],
    FSMState.TOKEN_SELECTION: [
        FSMState.CONFIRMATION,
        FSMState.IDLE,
        FSMState.EMERGENCY,
    ],
    FSMState.CONFIRMATION: [
        FSMState.GENERATING,
        FSMState.TOKEN_SELECTION,
        FSMState.EMERGENCY,
    ],
    FSMState.GENERATING: [
        FSMState.VALIDATING,
        FSMState.FALLBACK,
        FSMState.EMERGENCY,
    ],
    FSMState.VALIDATING: [
        FSMState.SPEAKING,
        FSMState.FALLBACK,
        FSMState.CONFIRMATION,
    ],
    FSMState.SPEAKING: [
        FSMState.IDLE,
        FSMState.EMERGENCY,
    ],
    FSMState.FALLBACK: [
        FSMState.SPEAKING,
        FSMState.IDLE,
    ],
    FSMState.EMERGENCY: [
        FSMState.IDLE,  # Only after reset()
    ],
}

# Maximum number of transition records kept in history
_MAX_HISTORY = 50


# ──────────────────────────────────────────────────────────────
# FSM class
# ──────────────────────────────────────────────────────────────

class SentinelFSM:
    """
    Thread-safe finite state machine for NeuroWeave Sentinel.

    Enforces the explicit transition map defined in :data:`_VALID_TRANSITIONS`.
    Illegal transitions raise :class:`InvalidTransitionError` immediately.
    Each transition fires per-state ``on_exit`` and ``on_enter`` callbacks.
    The last 50 transitions are retained in :meth:`get_history`.

    Args:
        on_transition: Optional callback invoked after every successful
            transition with signature ``(from_state, to_state, reason)``.
    """

    def __init__(
        self,
        on_transition: Callable[[FSMState, FSMState, str], None] | None = None,
    ) -> None:
        """Initialise FSM in IDLE state."""
        self._state: FSMState = FSMState.IDLE
        self._lock = threading.Lock()
        self._history: list[dict] = []
        self._external_callback = on_transition
        self._last_transition: dict | None = None

        logger.info("SentinelFSM initialised in state: %s", FSMState.IDLE.value)

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    @property
    def current_state(self) -> FSMState:
        """
        Return the current FSM state (thread-safe read).

        Returns:
            The active :class:`~core.constants.FSMState`.
        """
        with self._lock:
            return self._state

    def transition(self, new_state: FSMState, reason: str = "") -> None:
        """
        Attempt a validated state transition.

        Fires ``_on_exit_<from>`` then ``_on_enter_<to>`` callbacks.
        Records the transition in history. Notifies the external callback.

        Args:
            new_state: Target state to transition to.
            reason: Human-readable reason for the transition (for logs/history).

        Raises:
            InvalidTransitionError: If the transition is not in the valid map.
        """
        with self._lock:
            from_state = self._state
            allowed = _VALID_TRANSITIONS.get(from_state, [])

            if new_state not in allowed:
                raise InvalidTransitionError(from_state, new_state, reason)

            # Fire exit callback (before changing state)
            self._fire_on_exit(from_state)

            self._state = new_state

            # Record transition
            record = {
                "from": from_state.value,
                "to": new_state.value,
                "reason": reason,
                "timestamp": time.time(),
            }
            self._history.append(record)
            if len(self._history) > _MAX_HISTORY:
                self._history.pop(0)
            self._last_transition = record

        logger.info(
            "FSM: %s → %s%s",
            from_state.value,
            new_state.value,
            f" [{reason}]" if reason else "",
        )

        # Fire enter callback (after releasing lock to avoid deadlock)
        self._fire_on_enter(new_state)

        # External callback (also outside lock)
        if self._external_callback is not None:
            try:
                self._external_callback(from_state, new_state, reason)
            except Exception as exc:  # noqa: BLE001
                logger.warning("FSM external callback raised: %s", exc)

    def reset(self) -> None:
        """
        Force the FSM back to IDLE unconditionally.

        Used after an EMERGENCY is resolved or to recover from any state.
        Bypasses the transition map (does NOT raise InvalidTransitionError).
        Logs a warning to distinguish from normal validated transitions.
        """
        with self._lock:
            from_state = self._state
            self._fire_on_exit(from_state)
            self._state = FSMState.IDLE
            record = {
                "from": from_state.value,
                "to": FSMState.IDLE.value,
                "reason": "RESET",
                "timestamp": time.time(),
            }
            self._history.append(record)
            if len(self._history) > _MAX_HISTORY:
                self._history.pop(0)
            self._last_transition = record

        logger.warning(
            "FSM: RESET from %s → IDLE", from_state.value
        )
        self._fire_on_enter(FSMState.IDLE)

    def get_history(self) -> list[dict]:
        """
        Return a copy of the last (up to 50) transition records.

        Each record is a dict with keys:
        - ``from``: Source state name.
        - ``to``: Target state name.
        - ``reason``: Caller-supplied reason string.
        - ``timestamp``: Unix epoch float.

        Returns:
            List of transition record dicts, oldest first.
        """
        with self._lock:
            return list(self._history)

    def can_transition(self, target: FSMState) -> bool:
        """
        Check whether a transition to ``target`` is currently valid.

        Does not acquire the lock before checking (approximate read). Use
        :meth:`transition` for authoritative validation.

        Args:
            target: Candidate target state.

        Returns:
            True if the transition is in the valid map for the current state.
        """
        return target in _VALID_TRANSITIONS.get(self._state, [])

    # ──────────────────────────────────────────
    # on_enter callbacks — override in subclass
    # ──────────────────────────────────────────

    def _on_enter_idle(self) -> None:
        """Called when FSM enters IDLE."""
        logger.debug("FSM enter: IDLE — ready for gaze input")

    def _on_enter_token_selection(self) -> None:
        """Called when FSM enters TOKEN_SELECTION."""
        logger.debug("FSM enter: TOKEN_SELECTION — patient selecting tokens")

    def _on_enter_confirmation(self) -> None:
        """Called when FSM enters CONFIRMATION."""
        logger.debug("FSM enter: CONFIRMATION — awaiting confirm/cancel")

    def _on_enter_generating(self) -> None:
        """Called when FSM enters GENERATING — LLM inference starting."""
        logger.debug("FSM enter: GENERATING — LLM inference in progress")

    def _on_enter_validating(self) -> None:
        """Called when FSM enters VALIDATING — safety checks running."""
        logger.debug("FSM enter: VALIDATING — running safety pipeline")

    def _on_enter_speaking(self) -> None:
        """Called when FSM enters SPEAKING — TTS playback starting."""
        logger.debug("FSM enter: SPEAKING — TTS synthesis active")

    def _on_enter_emergency(self) -> None:
        """Called when FSM enters EMERGENCY — immediate broadcast triggered."""
        logger.critical("FSM enter: EMERGENCY — emergency override active")

    def _on_enter_fallback(self) -> None:
        """Called when FSM enters FALLBACK — using pre-set fallback message."""
        logger.warning("FSM enter: FALLBACK — using safety fallback message")

    # ──────────────────────────────────────────
    # on_exit callbacks — override in subclass
    # ──────────────────────────────────────────

    def _on_exit_idle(self) -> None:
        """Called when FSM exits IDLE."""
        logger.debug("FSM exit: IDLE")

    def _on_exit_token_selection(self) -> None:
        """Called when FSM exits TOKEN_SELECTION."""
        logger.debug("FSM exit: TOKEN_SELECTION")

    def _on_exit_confirmation(self) -> None:
        """Called when FSM exits CONFIRMATION."""
        logger.debug("FSM exit: CONFIRMATION")

    def _on_exit_generating(self) -> None:
        """Called when FSM exits GENERATING."""
        logger.debug("FSM exit: GENERATING")

    def _on_exit_validating(self) -> None:
        """Called when FSM exits VALIDATING.")"""
        logger.debug("FSM exit: VALIDATING")

    def _on_exit_speaking(self) -> None:
        """Called when FSM exits SPEAKING."""
        logger.debug("FSM exit: SPEAKING")

    def _on_exit_emergency(self) -> None:
        """Called when FSM exits EMERGENCY."""
        logger.warning("FSM exit: EMERGENCY — returning to IDLE")

    def _on_exit_fallback(self) -> None:
        """Called when FSM exits FALLBACK."""
        logger.debug("FSM exit: FALLBACK")

    # ──────────────────────────────────────────
    # Internal dispatch helpers
    # ──────────────────────────────────────────

    def _fire_on_enter(self, state: FSMState) -> None:
        """
        Dispatch to the on_enter callback for ``state``.

        Uses a name-based dispatch so subclasses can override individual
        callbacks without touching this method.

        Args:
            state: The state being entered.
        """
        method_name = f"_on_enter_{state.value.lower()}"
        method = getattr(self, method_name, None)
        if callable(method):
            try:
                method()
            except Exception as exc:  # noqa: BLE001
                logger.warning("on_enter callback %r raised: %s", method_name, exc)

    def _fire_on_exit(self, state: FSMState) -> None:
        """
        Dispatch to the on_exit callback for ``state``.

        NOTE: Called while ``self._lock`` is held — callbacks must not
        call :meth:`transition` or :meth:`current_state` to avoid deadlock.

        Args:
            state: The state being exited.
        """
        method_name = f"_on_exit_{state.value.lower()}"
        method = getattr(self, method_name, None)
        if callable(method):
            try:
                method()
            except Exception as exc:  # noqa: BLE001
                logger.warning("on_exit callback %r raised: %s", method_name, exc)

    # ──────────────────────────────────────────
    # Dunder methods
    # ──────────────────────────────────────────

    def __repr__(self) -> str:
        """
        Return a developer-readable representation showing current state and
        the most recent transition.

        Returns:
            String like ``SentinelFSM(state=IDLE, last=TOKEN_SELECTION→IDLE[reason])``.
        """
        with self._lock:
            state_str = self._state.value
            if self._last_transition:
                last = (
                    f"{self._last_transition['from']}"
                    f"→{self._last_transition['to']}"
                    + (
                        f"[{self._last_transition['reason']}]"
                        if self._last_transition["reason"]
                        else ""
                    )
                )
            else:
                last = "none"
        return f"SentinelFSM(state={state_str}, last={last})"
