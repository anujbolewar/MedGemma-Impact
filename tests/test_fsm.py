"""
tests/test_fsm.py — pytest unit tests for core.fsm.SentinelFSM.

All 6 test cases must pass in under 5 seconds total.
No external dependencies beyond the project source.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from typing import List

import pytest

from core.constants import FSMState
from core.fsm import InvalidTransitionError, SentinelFSM


# ──────────────────────────────────────────────────────────────
# Transition map mirror (must stay in sync with core/fsm.py)
# We copy it here so tests are self-documenting and raise clear
# failures if the production map diverges.
# ──────────────────────────────────────────────────────────────

VALID_TRANSITIONS: dict[FSMState, list[FSMState]] = {
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
        FSMState.IDLE,
    ],
}

# All states that list EMERGENCY as a valid target
_STATES_WITH_EMERGENCY: list[FSMState] = [
    s for s, targets in VALID_TRANSITIONS.items()
    if FSMState.EMERGENCY in targets
]


# ──────────────────────────────────────────────────────────────
# Helpers: force FSM into a specific state without going through
# the public transition() API (needed to set up invalid-transition
# and reset tests from arbitrary states).
# ──────────────────────────────────────────────────────────────

def _force_state(fsm: SentinelFSM, target: FSMState) -> None:
    """
    Force ``fsm`` into ``target`` by replaying a minimal valid path from
    IDLE, or by using reset() + replay.  Skips states where EMERGENCY is
    the only allowed predecessor (handled via reset).
    """
    # Always start from a known position
    fsm.reset()  # → IDLE

    paths: dict[FSMState, list[FSMState]] = {
        FSMState.IDLE:            [],
        FSMState.TOKEN_SELECTION: [FSMState.TOKEN_SELECTION],
        FSMState.CONFIRMATION:    [FSMState.TOKEN_SELECTION, FSMState.CONFIRMATION],
        FSMState.GENERATING:      [FSMState.TOKEN_SELECTION, FSMState.CONFIRMATION,
                                   FSMState.GENERATING],
        FSMState.VALIDATING:      [FSMState.TOKEN_SELECTION, FSMState.CONFIRMATION,
                                   FSMState.GENERATING, FSMState.VALIDATING],
        FSMState.SPEAKING:        [FSMState.TOKEN_SELECTION, FSMState.CONFIRMATION,
                                   FSMState.GENERATING, FSMState.VALIDATING,
                                   FSMState.SPEAKING],
        FSMState.FALLBACK:        [FSMState.TOKEN_SELECTION, FSMState.CONFIRMATION,
                                   FSMState.GENERATING, FSMState.FALLBACK],
        FSMState.EMERGENCY:       [FSMState.EMERGENCY],
    }
    for step in paths[target]:
        fsm.transition(step, reason="_force_state")


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture()
def fsm() -> SentinelFSM:
    """Fresh FSM instance in IDLE state."""
    return SentinelFSM()


@pytest.fixture()
def fsm_with_callbacks() -> tuple[SentinelFSM, list[tuple]]:
    """FSM that records every external callback invocation."""
    log: list[tuple[FSMState, FSMState, str]] = []

    def _cb(from_: FSMState, to_: FSMState, reason: str) -> None:
        log.append((from_, to_, reason))

    return SentinelFSM(on_transition=_cb), log


# ──────────────────────────────────────────────────────────────
# Test 1 — Valid transitions
# ──────────────────────────────────────────────────────────────

class TestValidTransitions:
    """Every edge in VALID_TRANSITIONS must succeed without raising."""

    def test_every_valid_edge(self, fsm: SentinelFSM) -> None:
        """
        Walk every (from, to) edge in the transition map and verify
        transition() succeeds (does not raise).  Forces FSM into the
        'from' state before each check.
        """
        tested: list[str] = []
        for from_state, targets in VALID_TRANSITIONS.items():
            for to_state in targets:
                _force_state(fsm, from_state)
                fsm.transition(to_state, reason="test_valid")
                assert fsm.current_state == to_state, (
                    f"Expected {to_state} after {from_state} → {to_state}"
                )
                tested.append(f"{from_state.value}→{to_state.value}")

        # Sanity: all edges counted
        total_edges = sum(len(v) for v in VALID_TRANSITIONS.values())
        assert len(tested) == total_edges, (
            f"Expected {total_edges} edges, tested {len(tested)}"
        )

    def test_initial_state_is_idle(self, fsm: SentinelFSM) -> None:
        assert fsm.current_state is FSMState.IDLE

    def test_full_happy_path(self, fsm: SentinelFSM) -> None:
        """IDLE → TOKEN_SELECTION → CONFIRMATION → GENERATING → VALIDATING → SPEAKING → IDLE."""
        path = [
            FSMState.TOKEN_SELECTION,
            FSMState.CONFIRMATION,
            FSMState.GENERATING,
            FSMState.VALIDATING,
            FSMState.SPEAKING,
            FSMState.IDLE,
        ]
        for step in path:
            fsm.transition(step, reason="happy_path")
            assert fsm.current_state == step

    def test_fallback_path(self, fsm: SentinelFSM) -> None:
        """GENERATING → FALLBACK → SPEAKING → IDLE."""
        _force_state(fsm, FSMState.GENERATING)
        fsm.transition(FSMState.FALLBACK, reason="test")
        fsm.transition(FSMState.SPEAKING, reason="test")
        fsm.transition(FSMState.IDLE, reason="test")
        assert fsm.current_state is FSMState.IDLE

    def test_external_callback_called(
        self, fsm_with_callbacks: tuple[SentinelFSM, list]
    ) -> None:
        fsm, log = fsm_with_callbacks
        fsm.transition(FSMState.TOKEN_SELECTION, reason="cb_test")
        assert len(log) == 1
        from_s, to_s, reason = log[0]
        assert from_s is FSMState.IDLE
        assert to_s is FSMState.TOKEN_SELECTION
        assert reason == "cb_test"

    def test_can_transition_returns_true_for_valid(self, fsm: SentinelFSM) -> None:
        assert fsm.can_transition(FSMState.TOKEN_SELECTION) is True
        assert fsm.can_transition(FSMState.EMERGENCY) is True

    def test_can_transition_returns_false_for_invalid(self, fsm: SentinelFSM) -> None:
        # IDLE cannot go directly to GENERATING
        assert fsm.can_transition(FSMState.GENERATING) is False


# ──────────────────────────────────────────────────────────────
# Test 2 — Invalid transitions
# ──────────────────────────────────────────────────────────────

class TestInvalidTransition:
    """Illegal state transitions must raise InvalidTransitionError."""

    @pytest.mark.parametrize("from_state, to_state", [
        # Direct long-jump: IDLE → GENERATING
        (FSMState.IDLE,            FSMState.GENERATING),
        # IDLE → SPEAKING (skips several states)
        (FSMState.IDLE,            FSMState.SPEAKING),
        # IDLE → VALIDATING
        (FSMState.IDLE,            FSMState.VALIDATING),
        # TOKEN_SELECTION → SPEAKING (skips CONFIRMATION + GENERATING)
        (FSMState.TOKEN_SELECTION, FSMState.SPEAKING),
        # GENERATING → IDLE (must go through VALIDATING / FALLBACK)
        (FSMState.GENERATING,      FSMState.IDLE),
        # EMERGENCY → TOKEN_SELECTION (can only go to IDLE)
        (FSMState.EMERGENCY,       FSMState.TOKEN_SELECTION),
    ])
    def test_raises_invalid_transition_error(
        self,
        fsm: SentinelFSM,
        from_state: FSMState,
        to_state: FSMState,
    ) -> None:
        _force_state(fsm, from_state)
        with pytest.raises(InvalidTransitionError) as exc_info:
            fsm.transition(to_state, reason="should_fail")
        err = exc_info.value
        assert err.from_state == from_state
        assert err.to_state == to_state
        # State must NOT have changed
        assert fsm.current_state == from_state

    def test_state_unchanged_after_invalid(self, fsm: SentinelFSM) -> None:
        """Verify current_state is preserved after a rejected transition."""
        assert fsm.current_state is FSMState.IDLE
        with pytest.raises(InvalidTransitionError):
            fsm.transition(FSMState.FALLBACK, reason="bad")
        assert fsm.current_state is FSMState.IDLE

    def test_invalid_transition_error_message(self, fsm: SentinelFSM) -> None:
        """Error message must mention both state names."""
        with pytest.raises(InvalidTransitionError) as exc_info:
            fsm.transition(FSMState.SPEAKING, reason="jump")
        msg = str(exc_info.value)
        assert "IDLE" in msg
        assert "SPEAKING" in msg


# ──────────────────────────────────────────────────────────────
# Test 3 — reset()
# ──────────────────────────────────────────────────────────────

class TestReset:
    """reset() must return the FSM to IDLE from every possible state."""

    @pytest.mark.parametrize("state", list(FSMState))
    def test_reset_from_every_state(
        self, fsm: SentinelFSM, state: FSMState
    ) -> None:
        _force_state(fsm, state)
        assert fsm.current_state is state, (
            f"Setup failed: expected {state}, got {fsm.current_state}"
        )
        fsm.reset()
        assert fsm.current_state is FSMState.IDLE, (
            f"Expected IDLE after reset() from {state}"
        )

    def test_reset_records_in_history(self, fsm: SentinelFSM) -> None:
        fsm.transition(FSMState.TOKEN_SELECTION, reason="pre_reset")
        fsm.reset()
        history = fsm.get_history()
        # The last record must be from TOKEN_SELECTION → IDLE with reason RESET
        last = history[-1]
        assert last["to"] == FSMState.IDLE.value
        assert last["reason"] == "RESET"

    def test_double_reset_is_idempotent(self, fsm: SentinelFSM) -> None:
        """Calling reset() twice should always leave FSM in IDLE."""
        _force_state(fsm, FSMState.SPEAKING)
        fsm.reset()
        fsm.reset()
        assert fsm.current_state is FSMState.IDLE

    def test_reset_allows_normal_use_afterwards(self, fsm: SentinelFSM) -> None:
        """After reset(), all valid transitions from IDLE must work."""
        _force_state(fsm, FSMState.GENERATING)
        fsm.reset()
        # Normal use: IDLE → TOKEN_SELECTION
        fsm.transition(FSMState.TOKEN_SELECTION, reason="post_reset")
        assert fsm.current_state is FSMState.TOKEN_SELECTION


# ──────────────────────────────────────────────────────────────
# Test 4 — Thread safety
# ──────────────────────────────────────────────────────────────

class TestThreadSafety:
    """
    Verify that concurrent reads of current_state are always consistent
    and that concurrent transitions from the same state converge cleanly.
    """

    def test_concurrent_reads(self, fsm: SentinelFSM) -> None:
        """10 threads simultaneously reading current_state must never see None."""
        results: list[FSMState | None] = [None] * 10
        barrier  = threading.Barrier(10)

        def _read(idx: int) -> None:
            barrier.wait()            # synchronise all threads to start at once
            results[idx] = fsm.current_state

        threads = [threading.Thread(target=_read, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)

        assert all(r is not None for r in results), "Some reads returned None"
        assert all(r is FSMState.IDLE for r in results), (
            f"Expected all IDLE, got: {results}"
        )

    def test_concurrent_transitions_no_deadlock(self, fsm: SentinelFSM) -> None:
        """
        5 threads each attempt TOKEN_SELECTION from IDLE simultaneously.
        Exactly one must succeed; the others will raise InvalidTransitionError
        (because state changed after the first winner).
        Neither raises an unhandled exception nor causes a deadlock.
        """
        successes: list[bool] = []
        barrier = threading.Barrier(5)
        lock    = threading.Lock()

        def _try_transition() -> None:
            barrier.wait()
            try:
                fsm.transition(FSMState.TOKEN_SELECTION, reason="race_test")
                with lock:
                    successes.append(True)
            except InvalidTransitionError:
                with lock:
                    successes.append(False)

        threads = [
            threading.Thread(target=_try_transition) for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)

        assert len(successes) == 5, "Not all threads completed"
        # Exactly one thread should have succeeded (IDLE → TOKEN_SELECTION)
        winners = [s for s in successes if s]
        assert len(winners) == 1, (
            f"Expected exactly 1 winner, got {len(winners)}: {successes}"
        )
        assert fsm.current_state is FSMState.TOKEN_SELECTION

    def test_concurrent_reads_during_transition(self) -> None:
        """
        One writer transitions state in a tight loop; 9 readers poll
        current_state.  All reads must return a valid FSMState (no partial
        writes / torn reads).
        """
        fsm = SentinelFSM()
        valid_states = set(FSMState)
        bad_reads: list = []
        stop_flag = threading.Event()

        def _writer() -> None:
            for _ in range(100):
                fsm.reset()
                try:
                    fsm.transition(FSMState.TOKEN_SELECTION, reason="load_test")
                except InvalidTransitionError:
                    pass

        def _reader() -> None:
            while not stop_flag.is_set():
                s = fsm.current_state
                if s not in valid_states:
                    bad_reads.append(s)

        readers = [threading.Thread(target=_reader) for _ in range(9)]
        writer  = threading.Thread(target=_writer)
        for r in readers:
            r.start()
        writer.start()
        writer.join(timeout=3.0)
        stop_flag.set()
        for r in readers:
            r.join(timeout=1.0)

        assert not bad_reads, f"Torn reads detected: {bad_reads[:5]}"


# ──────────────────────────────────────────────────────────────
# Test 5 — History
# ──────────────────────────────────────────────────────────────

class TestHistory:
    """get_history() must faithfully record transitions and cap at 50."""

    def test_empty_on_init(self, fsm: SentinelFSM) -> None:
        assert fsm.get_history() == []

    def test_records_single_transition(self, fsm: SentinelFSM) -> None:
        fsm.transition(FSMState.TOKEN_SELECTION, reason="hist_test")
        history = fsm.get_history()
        assert len(history) == 1
        record = history[0]
        assert record["from"] == FSMState.IDLE.value
        assert record["to"]   == FSMState.TOKEN_SELECTION.value
        assert record["reason"] == "hist_test"
        assert isinstance(record["timestamp"], float)
        assert record["timestamp"] > 0

    def test_returns_copy(self, fsm: SentinelFSM) -> None:
        """Mutating the returned list must not alter internal history."""
        fsm.transition(FSMState.TOKEN_SELECTION, reason="copy_test")
        history = fsm.get_history()
        history.clear()
        assert len(fsm.get_history()) == 1

    def test_ordered_oldest_first(self, fsm: SentinelFSM) -> None:
        """Transitions must appear in chronological order."""
        path = [
            FSMState.TOKEN_SELECTION,
            FSMState.CONFIRMATION,
            FSMState.GENERATING,
            FSMState.VALIDATING,
            FSMState.SPEAKING,
            FSMState.IDLE,
        ]
        for step in path:
            fsm.transition(step)
        history = fsm.get_history()
        tos = [r["to"] for r in history]
        expected = [s.value for s in path]
        assert tos == expected, f"Order mismatch:\n  got:      {tos}\n  expected: {expected}"

    def test_caps_at_50_records(self, fsm: SentinelFSM) -> None:
        """History must never exceed 50 records — oldest are dropped first."""
        # Generate 60 transitions by toggling TOKEN_SELECTION ↔ IDLE
        for i in range(60):
            if fsm.current_state is FSMState.IDLE:
                fsm.transition(FSMState.TOKEN_SELECTION, reason=f"toggle_{i}")
            else:
                fsm.transition(FSMState.IDLE, reason=f"toggle_{i}")

        history = fsm.get_history()
        assert len(history) <= 50, f"History exceeded cap: {len(history)}"

    def test_history_after_reset(self, fsm: SentinelFSM) -> None:
        """reset() adds a record with reason='RESET' to history."""
        fsm.transition(FSMState.TOKEN_SELECTION)
        pre_len = len(fsm.get_history())
        fsm.reset()
        history = fsm.get_history()
        assert len(history) == pre_len + 1
        assert history[-1]["reason"] == "RESET"
        assert history[-1]["to"] == FSMState.IDLE.value

    def test_timestamps_monotone(self, fsm: SentinelFSM) -> None:
        """Transition timestamps must be non-decreasing."""
        for step in [FSMState.TOKEN_SELECTION, FSMState.IDLE,
                     FSMState.TOKEN_SELECTION, FSMState.IDLE]:
            fsm.transition(step)
            time.sleep(0.001)   # ensure clock advances
        timestamps = [r["timestamp"] for r in fsm.get_history()]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1], (
                f"Non-monotone timestamps at index {i}: "
                f"{timestamps[i-1]} > {timestamps[i]}"
            )


# ──────────────────────────────────────────────────────────────
# Test 6 — EMERGENCY reachable from all states
# ──────────────────────────────────────────────────────────────

class TestEmergencyFromAnyState:
    """
    EMERGENCY must be directly reachable (one hop) from every state that
    allows it in the transition map.  States that omit EMERGENCY (VALIDATING,
    FALLBACK, EMERGENCY itself) are guarded by the transition map — attempting
    to go there should raise InvalidTransitionError.
    """

    @pytest.mark.parametrize("from_state", _STATES_WITH_EMERGENCY)
    def test_emergency_reachable_in_one_hop(
        self, fsm: SentinelFSM, from_state: FSMState
    ) -> None:
        _force_state(fsm, from_state)
        fsm.transition(FSMState.EMERGENCY, reason="emergency_test")
        assert fsm.current_state is FSMState.EMERGENCY

    @pytest.mark.parametrize("from_state", [
        FSMState.VALIDATING,
        FSMState.FALLBACK,
        FSMState.EMERGENCY,
    ])
    def test_emergency_not_in_map_raises(
        self, fsm: SentinelFSM, from_state: FSMState
    ) -> None:
        """
        VALIDATING and FALLBACK must NOT allow direct EMERGENCY transition
        (confirmed from the production transition map).
        """
        _force_state(fsm, from_state)
        with pytest.raises(InvalidTransitionError):
            fsm.transition(FSMState.EMERGENCY, reason="should_fail")
        # State must be unchanged
        assert fsm.current_state is from_state

    def test_emergency_to_idle_only(self, fsm: SentinelFSM) -> None:
        """After EMERGENCY, only IDLE is a valid next state."""
        _force_state(fsm, FSMState.EMERGENCY)
        # Attempt all non-IDLE states
        non_idle = [s for s in FSMState if s is not FSMState.IDLE]
        for bad_target in non_idle:
            with pytest.raises(InvalidTransitionError):
                fsm.transition(bad_target, reason="not_allowed")
        # Only IDLE must succeed
        fsm.transition(FSMState.IDLE, reason="emergency_cleared")
        assert fsm.current_state is FSMState.IDLE

    def test_emergency_states_cover_map(self) -> None:
        """
        All states that list EMERGENCY as a target in the live map must
        match what is listed in our local VALID_TRANSITIONS mirror.
        """
        from core.fsm import _VALID_TRANSITIONS as LIVE_MAP  # noqa: PLC0415
        live_emergency_sources = {
            s for s, targets in LIVE_MAP.items()
            if FSMState.EMERGENCY in targets
        }
        mirror_emergency_sources = set(_STATES_WITH_EMERGENCY)
        assert live_emergency_sources == mirror_emergency_sources, (
            "Live map and test mirror disagree on which states can reach EMERGENCY.\n"
            f"  Live:   {sorted(s.value for s in live_emergency_sources)}\n"
            f"  Mirror: {sorted(s.value for s in mirror_emergency_sources)}"
        )
