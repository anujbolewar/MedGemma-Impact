"""
input/signal_fuser.py — Multi-modal signal fusion for NeuroWeave Sentinel.

Fuses gaze (primary), optional EMG, and optional EEG signals into a single
FusedSignalFrame. Missing channels are handled via proportional weight
redistribution so composite_confidence always reflects available data quality.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from core.constants import SentinelConstants as C
from core.logger import get_logger

_log = get_logger()

# ──────────────────────────────────────────────────────────────
# Source protocols — duck-typed, no ABC inheritance required
# ──────────────────────────────────────────────────────────────

@runtime_checkable
class GazeSource(Protocol):
    """Minimal interface required of any gaze provider."""

    def get_frame(self) -> Any:
        """Return the latest gaze frame object."""
        ...


@runtime_checkable
class EMGSource(Protocol):
    """Minimal interface required of any EMG provider."""

    def get_amplitude(self) -> float:
        """Return the latest normalised EMG amplitude in [0.0, 1.0]."""
        ...


@runtime_checkable
class EEGSource(Protocol):
    """Minimal interface required of any EEG provider."""

    def get_alpha_power(self) -> float:
        """Return the latest normalised alpha-band power in [0.0, 1.0]."""
        ...


# ──────────────────────────────────────────────────────────────
# Output dataclass
# ──────────────────────────────────────────────────────────────

@dataclass
class FusedSignalFrame:
    """
    A single fused multi-modal signal observation.

    Attributes:
        primary_direction: Gaze direction string (e.g. ``'UP'``, ``'CENTRE'``).
        gaze_confidence: Raw gaze detection confidence in [0, 1].
        emg_amplitude: Normalised EMG amplitude; 0.0 if channel absent.
        eeg_alpha_power: Normalised EEG alpha-band power; 0.0 if channel absent.
        composite_confidence: Weighted fusion confidence in [0, 1].
        timestamp_ms: Milliseconds since epoch at time of fusion.
        active_channels: List of channel names that contributed to this frame.
    """

    primary_direction: str
    gaze_confidence: float
    emg_amplitude: float
    eeg_alpha_power: float
    composite_confidence: float
    timestamp_ms: float
    active_channels: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# Nominal channel weights (must sum to 1.0)
# ──────────────────────────────────────────────────────────────
_W_GAZE: float = 0.70
_W_EMG:  float = 0.20
_W_EEG:  float = 0.10

# Gaze direction is inferred from the normalised (x, y) position.
# Divide the screen into a 3×3 zone grid centred on CENTRE.
_DIRECTION_MAP: dict[tuple[int, int], str] = {
    (0, 0): "UP-LEFT",   (1, 0): "UP",     (2, 0): "UP-RIGHT",
    (0, 1): "LEFT",      (1, 1): "CENTRE", (2, 1): "RIGHT",
    (0, 2): "DOWN-LEFT", (1, 2): "DOWN",   (2, 2): "DOWN-RIGHT",
}


def _xy_to_direction(x: float, y: float) -> str:
    """
    Map normalised gaze coordinates to a named direction.

    Args:
        x: Horizontal gaze position in [0, 1] (0 = left).
        y: Vertical gaze position in [0, 1] (0 = top).

    Returns:
        Direction string from :data:`_DIRECTION_MAP`.
    """
    col = int(min(x * 3, 2))
    row = int(min(y * 3, 2))
    return _DIRECTION_MAP.get((col, row), "CENTRE")


# ──────────────────────────────────────────────────────────────
# SignalFuser
# ──────────────────────────────────────────────────────────────

class SignalFuser:
    """
    Window-based multi-modal signal fuser.

    Collects up to :data:`~core.constants.SentinelConstants.SIGNAL_WINDOW_MS`
    of gaze frames and (if connected) EMG and EEG samples, then returns a
    single :class:`FusedSignalFrame` representing the best estimate over
    that window (median direction, mean confidences).

    Missing channels are detected at init and their weights distributed
    proportionally among active channels so composite_confidence stays
    meaningful even with partial hardware.

    Args:
        gaze_source: Required gaze provider implementing :class:`GazeSource`.
        emg_source: Optional EMG provider implementing :class:`EMGSource`.
        eeg_source: Optional EEG provider implementing :class:`EEGSource`.
    """

    def __init__(
        self,
        gaze_source: GazeSource,
        emg_source: Optional[EMGSource] = None,
        eeg_source: Optional[EEGSource] = None,
    ) -> None:
        """Initialise fuser, compute effective weights, and log channel status."""
        self._gaze = gaze_source
        self._emg = emg_source
        self._eeg = eeg_source

        # Determine active channels and redistribute weights
        self._active_channels: list[str] = ["gaze"]
        total_w = _W_GAZE
        if emg_source is not None:
            self._active_channels.append("emg")
            total_w += _W_EMG
        if eeg_source is not None:
            self._active_channels.append("eeg")
            total_w += _W_EEG

        # Normalise weights so they always sum to 1.0
        self._eff_w_gaze = _W_GAZE / total_w
        self._eff_w_emg  = (_W_EMG / total_w) if emg_source else 0.0
        self._eff_w_eeg  = (_W_EEG / total_w) if eeg_source else 0.0

        # Window-based sample buffers (rolling, capped by time)
        self._window_ms = C.SIGNAL_WINDOW_MS
        self._gaze_buf: deque[tuple[float, float, float, float]] = deque()
        # (x, y, confidence, timestamp_ms)
        self._emg_buf: deque[tuple[float, float]] = deque()
        # (amplitude, timestamp_ms)
        self._eeg_buf: deque[tuple[float, float]] = deque()
        # (alpha_power, timestamp_ms)

        _log.info("signal_fuser", "init", {
            "active_channels": self._active_channels,
            "weights": {
                "gaze": round(self._eff_w_gaze, 3),
                "emg":  round(self._eff_w_emg, 3),
                "eeg":  round(self._eff_w_eeg, 3),
            },
            "window_ms": self._window_ms,
        })

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def get_fused_frame(self) -> FusedSignalFrame:
        """
        Poll all connected sources and return a fused signal frame.

        Polls the gaze source (and EMG/EEG if connected) once, appends
        the reading to the rolling window buffer, then computes the
        weighted fusion over all samples within :attr:`SIGNAL_WINDOW_MS`.

        Returns:
            A :class:`FusedSignalFrame` with direction, confidence values,
            channel list, and a composite confidence score.
        """
        now_ms = time.monotonic() * 1000.0

        # ── Poll gaze ────────────────────────────────────────
        gaze_frame = self._gaze.get_frame()
        gaze_x = float(getattr(gaze_frame, "x", 0.5))
        gaze_y = float(getattr(gaze_frame, "y", 0.5))
        gaze_conf = float(getattr(gaze_frame, "confidence", 1.0))
        self._gaze_buf.append((gaze_x, gaze_y, gaze_conf, now_ms))

        # ── Poll EMG (optional) ───────────────────────────────
        emg_amp = 0.0
        if self._emg is not None:
            try:
                emg_amp = float(self._emg.get_amplitude())
                emg_amp = max(0.0, min(1.0, emg_amp))
            except Exception as exc:  # noqa: BLE001
                _log.warn("signal_fuser", "emg_read_error", {"error": str(exc)})
            self._emg_buf.append((emg_amp, now_ms))

        # ── Poll EEG (optional) ───────────────────────────────
        eeg_alpha = 0.0
        if self._eeg is not None:
            try:
                eeg_alpha = float(self._eeg.get_alpha_power())
                eeg_alpha = max(0.0, min(1.0, eeg_alpha))
            except Exception as exc:  # noqa: BLE001
                _log.warn("signal_fuser", "eeg_read_error", {"error": str(exc)})
            self._eeg_buf.append((eeg_alpha, now_ms))

        # ── Evict stale samples ───────────────────────────────
        cutoff = now_ms - self._window_ms
        self._evict(self._gaze_buf, cutoff)
        self._evict(self._emg_buf, cutoff)
        self._evict(self._eeg_buf, cutoff)

        # ── Aggregate window ──────────────────────────────────
        direction, mean_gaze_conf = self._aggregate_gaze()
        mean_emg_amp    = self._mean(self._emg_buf, col=0) if self._emg else 0.0
        mean_eeg_alpha  = self._mean(self._eeg_buf, col=0) if self._eeg else 0.0

        # ── Composite confidence ──────────────────────────────
        # Gaze-only path (no EMG/EEG) short-circuits to raw gaze confidence
        if not self._emg and not self._eeg:
            composite = mean_gaze_conf
        else:
            composite = (
                self._eff_w_gaze * mean_gaze_conf
                + self._eff_w_emg  * (mean_emg_amp if self._emg else 0.0)
                + self._eff_w_eeg  * (mean_eeg_alpha if self._eeg else 0.0)
            )
            composite = round(min(max(composite, 0.0), 1.0), 4)

        return FusedSignalFrame(
            primary_direction=direction,
            gaze_confidence=round(mean_gaze_conf, 4),
            emg_amplitude=round(mean_emg_amp, 4),
            eeg_alpha_power=round(mean_eeg_alpha, 4),
            composite_confidence=composite,
            timestamp_ms=round(now_ms, 2),
            active_channels=list(self._active_channels),
        )

    @property
    def active_channels(self) -> list[str]:
        """Return the list of channel names contributing to fusion."""
        return list(self._active_channels)

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _aggregate_gaze(self) -> tuple[str, float]:
        """
        Compute the modal direction and mean confidence from the gaze window.

        Modal direction uses a simple frequency count over all buffered
        gaze samples. Mean confidence is the arithmetic mean.

        Returns:
            Tuple of ``(direction_string, mean_confidence)``.
        """
        if not self._gaze_buf:
            return "CENTRE", 0.0

        direction_counts: dict[str, int] = {}
        confidences: list[float] = []
        for x, y, conf, _ in self._gaze_buf:
            d = _xy_to_direction(x, y)
            direction_counts[d] = direction_counts.get(d, 0) + 1
            confidences.append(conf)

        modal_dir = max(direction_counts, key=lambda k: direction_counts[k])
        mean_conf = sum(confidences) / len(confidences)
        return modal_dir, round(mean_conf, 4)

    @staticmethod
    def _mean(buf: deque, col: int) -> float:
        """
        Compute the arithmetic mean of column ``col`` in a deque of tuples.

        Args:
            buf: Deque of tuples.
            col: Which tuple index to average across.

        Returns:
            Mean value, or 0.0 if buffer is empty.
        """
        if not buf:
            return 0.0
        return sum(row[col] for row in buf) / len(buf)

    @staticmethod
    def _evict(buf: deque, cutoff_ms: float) -> None:
        """
        Remove samples with timestamp (last column) older than ``cutoff_ms``.

        Args:
            buf: Sample deque where the last element of each tuple is timestamp_ms.
            cutoff_ms: Samples older than this are discarded.
        """
        while buf and buf[0][-1] < cutoff_ms:
            buf.popleft()
