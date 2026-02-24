"""
sentinel/gaze/simulator.py — Keyboard-simulated gaze input for demo/testing.

Provides the same interface as GazeTracker so the rest of the pipeline is
completely unaware of the input source. Arrow keys control gaze position,
Space triggers a blink, and long-hold Space triggers an emergency.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

from sentinel.core.config import GazeConfig
from sentinel.gaze.blink_detector import BlinkEvent
from sentinel.gaze.tracker import GazeFrame

logger = logging.getLogger(__name__)

# Gaze step size per arrow key press (normalised [0, 1])
_STEP: float = 1.0 / 3.0  # Jump one full grid cell

# Long-hold threshold for emergency via keyboard (seconds)
_EMERGENCY_HOLD_SECONDS: float = 3.0


@dataclass
class SimulatedGazeFrame(GazeFrame):
    """
    A GazeFrame produced by the keyboard simulator.

    Extends GazeFrame with an optional pre-computed blink event so that the
    pipeline can skip EAR-based detection (which requires real landmark data).

    Attributes:
        blink_event: If not None, a blink to inject into the pipeline directly.
    """

    blink_event: Optional[BlinkEvent] = None


class KeyboardSimulator:
    """
    Keyboard-driven gaze simulator for demo and testing without a webcam.

    Implements the same interface as :class:`~sentinel.gaze.tracker.GazeTracker`
    so it can be drop-in substituted in the pipeline.

    Key bindings:
    - Arrow Left / Right / Up / Down: Move gaze one grid cell in that direction.
    - Space: Short blink (symbol selection).
    - 'b': Long blink (page turn).
    - 'e': Emergency override trigger.
    - 'r': Reset gaze to centre.

    The simulator does NOT capture real keyboard events from the OS; it exposes
    :meth:`inject_key` for the UI to call on ``<KeyPress>`` events.

    Args:
        config: Gaze configuration (used for EAR parameters in injected frames).
    """

    def __init__(self, config: GazeConfig) -> None:
        """Initialise simulator at centre of screen."""
        self._cfg = config
        self._x: float = 0.5
        self._y: float = 0.5
        self._running: bool = False
        self._lock = threading.Lock()

        # Pending blink to inject on next get_frame() call
        self._pending_blink: Optional[BlinkEvent] = None

        logger.info("KeyboardSimulator initialised — use arrow keys / space / e")

    # ──────────────────────────────────────────
    # Lifecycle (matches GazeTracker interface)
    # ──────────────────────────────────────────

    def start(self) -> None:
        """
        Mark the simulator as running.

        No background thread is needed — the UI thread calls inject_key.
        """
        self._running = True
        logger.info("KeyboardSimulator started")

    def stop(self) -> None:
        """Mark the simulator as stopped."""
        self._running = False
        logger.info("KeyboardSimulator stopped")

    # ──────────────────────────────────────────
    # Consumer API (matches GazeTracker interface)
    # ──────────────────────────────────────────

    def get_frame(self) -> SimulatedGazeFrame:
        """
        Return the current simulated gaze frame.

        Consumes any pending blink event (clearing it after return so it only
        fires once).

        Returns:
            A :class:`SimulatedGazeFrame` with current position and any blink.
        """
        with self._lock:
            blink = self._pending_blink
            self._pending_blink = None
            x, y = self._x, self._y

        return SimulatedGazeFrame(
            x=x,
            y=y,
            timestamp=time.monotonic(),
            confidence=1.0,
            left_ear=self._cfg.blink_ear_threshold + 0.1,  # Healthy EAR (no blink)
            right_ear=self._cfg.blink_ear_threshold + 0.1,
            raw_frame=None,
            blink_event=blink,
        )

    # ──────────────────────────────────────────
    # Key injection (called by UI event handler)
    # ──────────────────────────────────────────

    def inject_key(self, key: str) -> None:
        """
        Process a key press and update simulator state accordingly.

        This method is thread-safe and should be called from the Tkinter
        key-press event callback.

        Supported keys:
        - ``'Left'``, ``'Right'``, ``'Up'``, ``'Down'``: Move gaze.
        - ``'space'`` or ``' '``: Short blink (select).
        - ``'b'``: Long blink (page turn).
        - ``'e'``: Emergency trigger.
        - ``'r'``: Reset gaze to centre.

        Args:
            key: Tkinter keysym string (e.g. ``'Left'``, ``'space'``, ``'e'``).
        """
        with self._lock:
            if key == "Left":
                self._x = max(0.0, self._x - _STEP)
            elif key == "Right":
                self._x = min(1.0, self._x + _STEP)
            elif key == "Up":
                self._y = max(0.0, self._y - _STEP)
            elif key == "Down":
                self._y = min(1.0, self._y + _STEP)
            elif key in ("space", " "):
                self._pending_blink = BlinkEvent(
                    is_long=False,
                    duration_frames=4,
                    timestamp=time.monotonic(),
                )
                logger.debug("Simulator: short blink injected")
            elif key == "b":
                self._pending_blink = BlinkEvent(
                    is_long=True,
                    duration_frames=15,
                    timestamp=time.monotonic(),
                )
                logger.debug("Simulator: long blink injected (page turn)")
            elif key == "r":
                self._x = 0.5
                self._y = 0.5
                logger.debug("Simulator: gaze reset to centre")
            # 'e' is handled directly in the pipeline / UI for emergency

    @property
    def position(self) -> tuple[float, float]:
        """
        Return the current simulated gaze position (thread-safe).

        Returns:
            Tuple of (x, y) normalised to [0, 1].
        """
        with self._lock:
            return (self._x, self._y)
