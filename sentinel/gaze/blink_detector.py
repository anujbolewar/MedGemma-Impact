"""
sentinel/gaze/blink_detector.py — EAR-based blink detection and dwell tracking.

Detects single blinks, long blinks, and gaze dwell events from a stream of
GazeFrames. These events drive symbol selection on the AAC board.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from sentinel.core.config import GazeConfig
from sentinel.gaze.tracker import GazeFrame

logger = logging.getLogger(__name__)


@dataclass
class BlinkEvent:
    """
    A confirmed blink detected from consecutive low-EAR frames.

    Attributes:
        is_long: True if this was a sustained long blink (page navigation).
        duration_frames: Number of frames the blink lasted.
        timestamp: Monotonic time of detection.
    """

    is_long: bool
    duration_frames: int
    timestamp: float


@dataclass
class DwellEvent:
    """
    A completed dwell — gaze held at a single region for the threshold duration.

    Attributes:
        region_id: String identifier of the screen region dwelt upon.
        x: Normalised gaze X at start of dwell.
        y: Normalised gaze Y at start of dwell.
        duration_ms: Total dwell duration in milliseconds.
        timestamp: Monotonic time of dwell completion.
    """

    region_id: str
    x: float
    y: float
    duration_ms: float
    timestamp: float


class BlinkDetector:
    """
    Stateful blink detector using Eye Aspect Ratio (EAR) thresholding.

    Counts consecutive frames where EAR falls below a threshold to
    distinguish single blinks (selection) from long blinks (page turn).

    Args:
        config: Gaze configuration holding threshold and frame count values.
    """

    def __init__(self, config: GazeConfig) -> None:
        """Initialise detector with configuration parameters."""
        self._ear_threshold = config.blink_ear_threshold
        self._consec_frames = config.blink_consec_frames
        self._long_frames = config.long_blink_frames

        self._consecutive_below: int = 0
        self._in_blink: bool = False

    def detect(self, frame: GazeFrame) -> Optional[BlinkEvent]:
        """
        Process a single gaze frame and return a BlinkEvent if one completed.

        A blink is only emitted on the *rising edge* (when EAR recovers above
        threshold) to avoid repeated triggers for a single sustained blink.

        Args:
            frame: The latest gaze frame containing left_ear/right_ear.

        Returns:
            A :class:`BlinkEvent` if a blink just completed, else None.
        """
        # Use the average of both eyes
        avg_ear = (frame.left_ear + frame.right_ear) / 2.0

        if avg_ear < self._ear_threshold:
            self._consecutive_below += 1
            self._in_blink = True
            return None  # Still in blink — wait for rise

        # EAR recovered above threshold
        if self._in_blink and self._consecutive_below >= self._consec_frames:
            duration = self._consecutive_below
            is_long = duration >= self._long_frames
            event = BlinkEvent(
                is_long=is_long,
                duration_frames=duration,
                timestamp=time.monotonic(),
            )
            logger.debug(
                "Blink detected: duration=%d frames, long=%s", duration, is_long
            )
            self._consecutive_below = 0
            self._in_blink = False
            return event

        self._consecutive_below = 0
        self._in_blink = False
        return None


class DwellTracker:
    """
    Tracks how long gaze dwells on a single screen region.

    The screen is implicitly divided into a grid; regions are identified by
    a string computed from the quantised (x, y) position. A :class:`DwellEvent`
    fires once the gaze has stayed in a region for at least ``dwell_threshold_ms``.

    Args:
        dwell_threshold_ms: Milliseconds of continuous gaze required to trigger.
        grid_cols: Number of horizontal grid cells for region quantisation.
        grid_rows: Number of vertical grid cells for region quantisation.
    """

    def __init__(
        self,
        dwell_threshold_ms: int = 800,
        grid_cols: int = 3,
        grid_rows: int = 3,
    ) -> None:
        """Initialise dwell tracker."""
        self._threshold_ms = dwell_threshold_ms
        self._grid_cols = grid_cols
        self._grid_rows = grid_rows

        self._current_region: Optional[str] = None
        self._dwell_start: Optional[float] = None
        self._dwell_x: float = 0.5
        self._dwell_y: float = 0.5
        self._fired: bool = False  # Prevent re-firing for same dwell

    def update(self, x: float, y: float) -> Optional[DwellEvent]:
        """
        Update tracker with the current gaze position.

        Returns a :class:`DwellEvent` if the dwell threshold was just reached,
        None otherwise. After firing, the tracker resets and requires the gaze
        to leave the region before it can fire again.

        Args:
            x: Normalised gaze X in [0, 1].
            y: Normalised gaze Y in [0, 1].

        Returns:
            A :class:`DwellEvent` if dwell completed, else None.
        """
        region = self._quantise(x, y)
        now = time.monotonic()

        if region != self._current_region:
            # Gaze moved to a new region — reset timer
            self._current_region = region
            self._dwell_start = now
            self._dwell_x = x
            self._dwell_y = y
            self._fired = False
            return None

        if self._fired:
            return None  # Already fired for this dwell — wait for move

        if self._dwell_start is None:
            self._dwell_start = now
            return None

        elapsed_ms = (now - self._dwell_start) * 1000.0

        if elapsed_ms >= self._threshold_ms:
            self._fired = True
            event = DwellEvent(
                region_id=region,
                x=self._dwell_x,
                y=self._dwell_y,
                duration_ms=elapsed_ms,
                timestamp=now,
            )
            logger.debug("Dwell on region=%s (%.0fms)", region, elapsed_ms)
            return event

        return None

    def get_dwell_progress(self, x: float, y: float) -> float:
        """
        Return dwell progress [0.0–1.0] for the current gaze position.

        Used by the UI to render a progress ring around the hovered symbol.

        Args:
            x: Current normalised gaze X.
            y: Current normalised gaze Y.

        Returns:
            Float in [0, 1] representing dwell completion progress.
        """
        region = self._quantise(x, y)
        if region != self._current_region or self._dwell_start is None or self._fired:
            return 0.0

        elapsed_ms = (time.monotonic() - self._dwell_start) * 1000.0
        return min(elapsed_ms / self._threshold_ms, 1.0)

    def reset(self) -> None:
        """
        Reset all dwell state.

        Call this after an explicit blink-selection to prevent the dwell
        tracker from firing immediately on the same region.
        """
        self._current_region = None
        self._dwell_start = None
        self._fired = False

    def _quantise(self, x: float, y: float) -> str:
        """
        Convert a continuous (x, y) position to a discrete region ID string.

        Args:
            x: Normalised x in [0, 1].
            y: Normalised y in [0, 1].

        Returns:
            Region string like ``'r1c2'`` (row 1, column 2).
        """
        col = int(min(x * self._grid_cols, self._grid_cols - 1))
        row = int(min(y * self._grid_rows, self._grid_rows - 1))
        return f"r{row}c{col}"
