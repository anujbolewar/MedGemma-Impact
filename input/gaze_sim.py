"""
input/gaze_sim.py — Simulated gaze input for demo and testing without a webcam.

Provides :class:`SimulatedGazeTracker`, which is a drop-in replacement for
:class:`~input.gaze_webcam.WebcamGazeTracker` via duck typing.  Three modes
are supported through the :class:`SimulationMode` enum:

SCRIPTED
    Takes a list of ``(direction, duration_ms, confidence)`` tuples and
    plays them back in real time.  The special direction ``'BLINK'`` emits
    exactly one frame with ``blink=True`` then immediately advances.
    After the last step the tracker holds ``'CENTER'`` until stopped.

RANDOM
    Picks a random direction at each step according to a configurable
    probability distribution.  Each direction is held for a random duration
    (0.5 – 2.5 s).  Spontaneous blinks occur with a low probability per
    frame (~1 per 10 s at 30 fps).

INTERACTIVE
    Reads key input via :meth:`inject_key` and emits frames at the target
    FPS based on the current key state.  Arrow keys set direction; ``space``
    injects a blink; ``'e'`` forces a sustained UP direction (emergency
    override trigger); ``'r'`` resets to CENTER.

Direction strings and their canonical ``GazeFrame.direction`` mapping
----------------------------------------------------------------------
Five canonical directions (``'CENTER'``, ``'LEFT'``, ``'RIGHT'``, ``'UP'``,
``'DOWN'``) are exposed in emitted frames.  Demo scripts may also use four
extended diagonals (``'UP_LEFT'``, ``'UP_RIGHT'``, ``'DOWN_LEFT'``,
``'DOWN_RIGHT'``) which hit corner cells of the 3 × 3 symbol grid; these are
folded back to the nearest vertical canonical direction in the frame.

Symbol-board cell mapping (3 × 3 grid, vectors in ``[-1, 1]`` space)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    UP_LEFT(-0.6,-0.6)=r0c0   UP(0,-0.6)=r0c1    UP_RIGHT(0.6,-0.6)=r0c2
    LEFT(-0.6,0)=r1c0         CENTER(0,0)=r1c1    RIGHT(0.6,0)=r1c2
    DOWN_LEFT(-0.6,0.6)=r2c0  DOWN(0,0.6)=r2c1   DOWN_RIGHT(0.6,0.6)=r2c2

Pre-built demo scripts
----------------------
Three class-level constants are provided:

``DEMO_PAIN_SCRIPT``
    Navigates the symbol board to select Chest → Pain → Right Now, driving
    the LLM toward *"I have pain in my chest"*.

``DEMO_WATER_SCRIPT``
    Selects Head/Face → Discomfort → Right Now to communicate thirst,
    driving the LLM toward *"I need water please"*.

``DEMO_EMERGENCY_SCRIPT``
    Holds UP gaze for 3 500 ms — exceeding ``C.EMERGENCY_HOLD_MS`` (3 000 ms)
    to trigger the emergency override.

All logging goes through :func:`core.logger.get_logger`.
"""

from __future__ import annotations

import enum
import random
import threading
import time
from typing import Dict, List, Optional, Tuple

from input.gaze_webcam import GazeFrame
from core.logger import get_logger
from core.constants import C  # C.GAZE_DWELL_MS, C.EMERGENCY_HOLD_MS


# ── Direction → canonical gaze_vector map ([-1, 1] space) ────────────────────

#: Maps every supported direction string to a representative ``(x, y)``
#: gaze vector.  Diagonal entries are for demo-script use only.
_DIRECTION_VECTORS: Dict[str, Tuple[float, float]] = {
    "CENTER":      (0.0,   0.0),
    "LEFT":        (-0.6,  0.0),
    "RIGHT":       (0.6,   0.0),
    "UP":          (0.0,  -0.6),
    "DOWN":        (0.0,   0.6),
    # Extended — demo scripts / SCRIPTED mode only
    "UP_LEFT":     (-0.6, -0.6),
    "UP_RIGHT":    (0.6,  -0.6),
    "DOWN_LEFT":   (-0.6,  0.6),
    "DOWN_RIGHT":  (0.6,   0.6),
    # Special — one blink frame; vector stays at centre
    "BLINK":       (0.0,   0.0),
}

#: Fold every direction (including diagonals) to one of the five canonical
#: ``GazeFrame.direction`` values.
_TO_CANONICAL: Dict[str, str] = {
    "CENTER": "CENTER", "LEFT": "LEFT", "RIGHT": "RIGHT",
    "UP": "UP",         "DOWN": "DOWN",
    "UP_LEFT": "UP",    "UP_RIGHT": "UP",
    "DOWN_LEFT": "DOWN","DOWN_RIGHT": "DOWN",
    "BLINK": "CENTER",
}

# ── Type alias for a single script step ──────────────────────────────────────

#: ``(direction_string, hold_duration_ms, confidence)``
ScriptStep = Tuple[str, float, float]


# ── Simulation mode enum ──────────────────────────────────────────────────────

class SimulationMode(enum.Enum):
    """Operating mode for :class:`SimulatedGazeTracker`."""

    SCRIPTED    = "SCRIPTED"
    RANDOM      = "RANDOM"
    INTERACTIVE = "INTERACTIVE"


# ── Simulator ─────────────────────────────────────────────────────────────────

class SimulatedGazeTracker:
    """
    Simulated gaze input for demo and testing without a webcam.

    Provides the same duck-typed interface as
    :class:`~input.gaze_webcam.WebcamGazeTracker`:
    :meth:`start`, :meth:`stop`, :meth:`get_latest`.

    The background thread runs at *fps* and updates :meth:`get_latest` with
    the appropriate :class:`~input.gaze_webcam.GazeFrame` for the current
    mode.

    Args:
        mode: Operating mode (``SCRIPTED``, ``RANDOM``, or ``INTERACTIVE``).
        camera_id: Accepted for interface parity with
            :class:`~input.gaze_webcam.WebcamGazeTracker`; silently ignored.
        fps: Target emission frame-rate.
        script: Steps for ``SCRIPTED`` mode.  Defaults to
            :attr:`DEMO_PAIN_SCRIPT` when ``None``.
        direction_probs: Direction → weight mapping for ``RANDOM`` mode.
            Weights need not sum to 1.  Defaults to a centre-biased uniform
            distribution over the five canonical directions.
    """

    # ── Pre-built demo scripts ─────────────────────────────────────────────
    #
    # Cell layout reference (Page 0 = body parts, Page 1 = sensations,
    #                         Page 2 = urgency/intensity):
    #
    #   UP_LEFT=r0c0  UP=r0c1    UP_RIGHT=r0c2
    #   LEFT=r1c0     CENTER=r1c1  RIGHT=r1c2
    #   DOWN_LEFT=r2c0 DOWN=r2c1  DOWN_RIGHT=r2c2
    #
    # Each step: (direction, duration_ms, confidence)
    # GAZE_DWELL_MS = 1 500 ms → steps held for 2 000 ms trigger dwell.

    DEMO_PAIN_SCRIPT: List[ScriptStep] = [
        # ── Phase 1: body-part selection (Page 0) ─────────────────────────
        ("CENTER",    800.0, 1.0),  # Initial fixation — pipeline ready
        ("UP",       2000.0, 1.0),  # Dwell on Chest   (p0 r0c1) ≥ GAZE_DWELL_MS
        ("BLINK",     200.0, 1.0),  # Confirm → Chest selected ✓
        ("CENTER",    600.0, 1.0),  # Rest while pipeline transitions to Page 1
        # ── Phase 2: sensation selection (Page 1) ─────────────────────────
        ("UP_LEFT",  2000.0, 1.0),  # Dwell on Pain    (p1 r0c0)
        ("BLINK",     200.0, 1.0),  # Confirm → Pain selected ✓
        ("CENTER",    600.0, 1.0),  # Rest while pipeline transitions to Page 2
        # ── Phase 3: urgency selection (Page 2) ───────────────────────────
        ("UP_LEFT",  2000.0, 1.0),  # Dwell on Right Now (p2 r0c0)
        ("BLINK",     200.0, 1.0),  # Confirm → Right Now selected ✓
        # ── LLM generates: "I have pain in my chest" ──────────────────────
        ("CENTER",   1500.0, 0.9),  # Rest while LLM reconstructs sentence
    ]

    DEMO_WATER_SCRIPT: List[ScriptStep] = [
        # ── Phase 1: body-part selection (Page 0) ─────────────────────────
        ("CENTER",    800.0, 1.0),  # Initial fixation
        ("UP_LEFT",  2000.0, 1.0),  # Dwell on Head/Face (p0 r0c0) — implies mouth/throat
        ("BLINK",     200.0, 1.0),  # Confirm → Head/Face selected ✓
        ("CENTER",    600.0, 1.0),  # Rest while pipeline transitions to Page 1
        # ── Phase 2: sensation selection (Page 1) ─────────────────────────
        ("DOWN",     2000.0, 1.0),  # Dwell on Discomfort (p1 r2c1) — implies thirst
        ("BLINK",     200.0, 1.0),  # Confirm → Discomfort selected ✓
        ("CENTER",    600.0, 1.0),  # Rest while pipeline transitions to Page 2
        # ── Phase 3: urgency selection (Page 2) ───────────────────────────
        ("UP_LEFT",  2000.0, 1.0),  # Dwell on Right Now (p2 r0c0)
        ("BLINK",     200.0, 1.0),  # Confirm → Right Now selected ✓
        # ── LLM generates: "I need water please" ──────────────────────────
        ("CENTER",   1500.0, 0.9),  # Rest while LLM reconstructs sentence
    ]

    DEMO_EMERGENCY_SCRIPT: List[ScriptStep] = [
        # Sustained UP gaze > C.EMERGENCY_HOLD_MS (3 000 ms) triggers override
        ("CENTER",    500.0, 1.0),  # Brief centre before emergency
        ("UP",       3500.0, 1.0),  # Hold UP — fires at 3 000 ms → EMERGENCY ✓
        ("CENTER",   1000.0, 1.0),  # Return to centre after trigger
    ]

    # ── Constructor ────────────────────────────────────────────────────────

    def __init__(
        self,
        mode: SimulationMode = SimulationMode.SCRIPTED,
        camera_id: int = 0,
        fps: int = 30,
        script: Optional[List[ScriptStep]] = None,
        direction_probs: Optional[Dict[str, float]] = None,
    ) -> None:
        self._mode = mode
        self._fps = fps
        self._script: List[ScriptStep] = (
            script if script is not None else list(self.DEMO_PAIN_SCRIPT)
        )

        # Default RANDOM-mode direction weights (centre-biased)
        self._direction_probs: Dict[str, float] = direction_probs or {
            "CENTER": 0.50,
            "LEFT":   0.10,
            "RIGHT":  0.10,
            "UP":     0.15,
            "DOWN":   0.15,
        }

        # Shared mutable state
        self._running: bool = False
        self._lock = threading.Lock()
        self._latest: Optional[GazeFrame] = None
        self._thread: Optional[threading.Thread] = None

        # INTERACTIVE-mode per-key state (written by inject_key, read by loop)
        self._i_direction: str = "CENTER"
        self._i_blink: bool = False     # consumed after one frame
        self._i_emergency: bool = False

        log = get_logger()
        log.info(
            "gaze",
            "sim_init",
            {"mode": mode.value, "fps": fps, "dwell_ms": C.GAZE_DWELL_MS},
        )

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Launch the background simulation thread.

        Safe to call only once; calling again after :meth:`stop` requires
        creating a new instance.
        """
        log = get_logger()
        _target = {
            SimulationMode.SCRIPTED:    self._run_scripted,
            SimulationMode.RANDOM:      self._run_random,
            SimulationMode.INTERACTIVE: self._run_interactive,
        }[self._mode]

        self._running = True
        self._thread = threading.Thread(
            target=_target, name="gaze-sim", daemon=True
        )
        self._thread.start()
        log.info("gaze", "sim_started", {"mode": self._mode.value, "fps": self._fps})

    def stop(self) -> None:
        """
        Signal the background thread to stop and wait up to 3 s for it to join.

        Safe to call multiple times or before :meth:`start`.
        """
        log = get_logger()
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        log.info("gaze", "sim_stopped", {"mode": self._mode.value})

    def get_latest(self) -> Optional[GazeFrame]:
        """
        Return the most recent simulated :class:`~input.gaze_webcam.GazeFrame`.

        Identical contract to :meth:`~input.gaze_webcam.WebcamGazeTracker.get_latest`:
        returns ``None`` before the first frame has been emitted.

        Thread-safe.
        """
        with self._lock:
            return self._latest

    # ── Key injection (INTERACTIVE mode) ──────────────────────────────────

    def inject_key(self, key: str) -> None:
        """
        Drive gaze state from a keyboard event (INTERACTIVE mode).

        Thread-safe; safe to call in any mode (no-op for SCRIPTED/RANDOM).

        Supported keys
        --------------
        ``'Left'`` / ``'Right'`` / ``'Up'`` / ``'Down'``
            Set the current gaze direction.
        ``'space'`` or ``' '``
            Inject a single blink frame (consumed after next :meth:`get_latest` poll).
        ``'e'`` / ``'E'``
            Force sustained ``'UP'`` direction — emergency override trigger.
            Hold key (or call repeatedly) for ``> C.EMERGENCY_HOLD_MS`` ms.
        ``'r'`` / ``'R'``
            Reset direction to ``'CENTER'`` and clear emergency state.

        Args:
            key: Key identifier string (Tkinter keysym or plain character).
        """
        log = get_logger()
        with self._lock:
            if key in ("Left", "left"):
                self._i_direction = "LEFT"
            elif key in ("Right", "right"):
                self._i_direction = "RIGHT"
            elif key in ("Up", "up"):
                self._i_direction = "UP"
            elif key in ("Down", "down"):
                self._i_direction = "DOWN"
            elif key in ("space", " "):
                self._i_blink = True
            elif key in ("e", "E"):
                self._i_direction = "UP"
                self._i_emergency = True
            elif key in ("r", "R"):
                self._i_direction = "CENTER"
                self._i_emergency = False

        if key not in ("space", " "):
            log.info("gaze", "sim_key_injected", {"key": key, "direction": self._i_direction})

    # ── Private: shared frame factory ─────────────────────────────────────

    @staticmethod
    def _make_frame(
        direction: str,
        confidence: float,
        blink: bool = False,
    ) -> GazeFrame:
        """
        Build a :class:`~input.gaze_webcam.GazeFrame` from a direction string.

        Args:
            direction: Any key in :data:`_DIRECTION_VECTORS` (including
                diagonals and ``'BLINK'``).
            confidence: Simulated detection confidence in ``[0, 1]``.
            blink: If ``True``, set ``GazeFrame.blink = True``.

        Returns:
            A frozen :class:`~input.gaze_webcam.GazeFrame`.
        """
        vec = _DIRECTION_VECTORS.get(direction, (0.0, 0.0))
        canonical = _TO_CANONICAL.get(direction, "CENTER")
        return GazeFrame(
            gaze_vector=vec,
            direction=canonical,
            blink=blink,
            confidence=confidence,
            timestamp_ms=time.monotonic() * 1000.0,
        )

    # ── Private: SCRIPTED loop ─────────────────────────────────────────────

    def _run_scripted(self) -> None:
        """
        Play back :attr:`_script` in real time.

        For each step:

        * **Non-BLINK steps** — hold the direction for *duration_ms*,
          emitting one frame per FPS tick.
        * **BLINK steps** — emit a single ``blink=True`` frame then sleep
          for *duration_ms* to represent the blink window before moving on.

        After the final step, holds ``'CENTER'`` at full confidence until
        :meth:`stop` is called.
        """
        log = get_logger()
        interval = 1.0 / self._fps

        for step_idx, (direction, duration_ms, confidence) in enumerate(self._script):
            if not self._running:
                break

            log.info(
                "gaze",
                "sim_scripted_step",
                {
                    "step": step_idx,
                    "direction": direction,
                    "duration_ms": duration_ms,
                    "confidence": confidence,
                },
            )

            if direction == "BLINK":
                # One blink frame, then a short pause for the blink duration.
                frame = self._make_frame("CENTER", confidence, blink=True)
                with self._lock:
                    self._latest = frame
                _sleep_ms(duration_ms)
                continue

            # Hold direction for the specified duration.
            deadline = time.monotonic() + duration_ms / 1000.0
            while self._running and time.monotonic() < deadline:
                t0 = time.monotonic()
                frame = self._make_frame(direction, confidence)
                with self._lock:
                    self._latest = frame
                _sleep_remaining(t0, interval)

        log.info("gaze", "sim_scripted_complete", {"steps": len(self._script)})

        # Park at CENTER until stopped.
        while self._running:
            t0 = time.monotonic()
            frame = self._make_frame("CENTER", 1.0)
            with self._lock:
                self._latest = frame
            _sleep_remaining(t0, interval)

    # ── Private: RANDOM loop ───────────────────────────────────────────────

    def _run_random(self) -> None:
        """
        Emit random gaze frames according to :attr:`_direction_probs`.

        Each randomly chosen direction is held for a uniformly sampled
        duration in ``[0.5, 2.5]`` seconds.  Spontaneous blinks fire with
        ~0.3 % probability per frame (≈ 1 blink per 10 s at 30 fps).
        """
        log = get_logger()
        interval = 1.0 / self._fps

        directions = list(self._direction_probs.keys())
        weights = [self._direction_probs[d] for d in directions]

        current_dir = "CENTER"
        frames_remaining = 0

        while self._running:
            t0 = time.monotonic()

            if frames_remaining <= 0:
                current_dir = random.choices(directions, weights=weights, k=1)[0]
                hold_s = random.uniform(0.5, 2.5)
                frames_remaining = max(1, int(hold_s * self._fps))
                log.info(
                    "gaze",
                    "sim_random_direction",
                    {"direction": current_dir, "hold_s": round(hold_s, 2)},
                )

            # ~0.3 % per frame blink chance (normalised for actual FPS)
            blink_prob = 0.003 / max(self._fps / 30.0, 1e-3)
            blink = random.random() < blink_prob
            confidence = random.uniform(0.85, 1.0)

            frame = self._make_frame(current_dir, confidence, blink=blink)
            with self._lock:
                self._latest = frame

            frames_remaining -= 1
            _sleep_remaining(t0, interval)

    # ── Private: INTERACTIVE loop ──────────────────────────────────────────

    def _run_interactive(self) -> None:
        """
        Emit frames at target FPS driven by :meth:`inject_key` state.

        The blink flag is consumed atomically after being set by a ``space``
        keypress so it fires on exactly one frame.
        """
        interval = 1.0 / self._fps

        while self._running:
            t0 = time.monotonic()

            # Snapshot + consume interactive state under a single lock
            with self._lock:
                direction = self._i_direction
                blink = self._i_blink
                self._i_blink = False  # consume blink flag

            frame = self._make_frame(direction, 1.0, blink=blink)
            with self._lock:
                self._latest = frame

            _sleep_remaining(t0, interval)


# ── Module-level helpers ──────────────────────────────────────────────────────

def _sleep_ms(duration_ms: float) -> None:
    """Sleep for *duration_ms* milliseconds, clipping negatives to zero."""
    secs = duration_ms / 1000.0
    if secs > 0.0:
        time.sleep(secs)


def _sleep_remaining(t0: float, interval: float) -> None:
    """Sleep for whatever fraction of *interval* has not yet elapsed since *t0*."""
    remaining = interval - (time.monotonic() - t0)
    if remaining > 0.0:
        time.sleep(remaining)
