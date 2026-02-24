"""
input/gaze_webcam.py — Real-time gaze tracking using MediaPipe Face Mesh.

Captures webcam frames in a daemon background thread, extracts iris
landmark positions with MediaPipe FaceMesh (iris refinement enabled),
and emits :class:`GazeFrame` objects with a 2-D normalised gaze vector,
classified direction, blink flag, confidence score, and millisecond
timestamp.

Key design decisions
--------------------
* Gaze vector is in **[-1, 1] × [-1, 1]** with (0, 0) = straight ahead.
  Raw MediaPipe iris coordinates (0–1 normalised) are shifted by -0.5 and
  doubled; EMA smoothing reduces jitter before direction classification.
* EAR blink detection uses the standard Soukupova formula across 6
  landmarks per eye.  A blink is reported when the average EAR of both
  eyes drops below :data:`_EAR_BLINK_THRESHOLD`.
* Direction classification uses a configurable dead-zone
  (:data:`_DIRECTION_THRESHOLD`) so short noise excursions do not produce
  spurious direction changes.
* :class:`DwellDetector` is an inner class of :class:`WebcamGazeTracker`
  that uses ``C.GAZE_DWELL_MS`` for the hold time.
* When the camera cannot be opened, :meth:`~WebcamGazeTracker.start`
  logs the error and sets an internal flag; all subsequent calls to
  :meth:`~WebcamGazeTracker.get_latest` return ``None`` rather than
  raising.
* Every :data:`_CONFIDENCE_LOG_INTERVAL` frames the average confidence
  over that window is written to the NWSLogger JSONL log.

All logging goes through :func:`core.logger.get_logger`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as _mp_tasks
from mediapipe.tasks.python import vision as _mp_vision
import numpy as np

from core.logger import get_logger
from core.constants import C  # C.GAZE_DWELL_MS, C.CONFIDENCE_MIN

# ── Tuning constants (mirrored from sentinel.yaml gaze section) ───────────────
_EAR_BLINK_THRESHOLD: float = 0.21   # EAR below this → blink
_GAZE_MIN_CONFIDENCE: float = 0.6    # MediaPipe detection / tracking confidence
_GAZE_EMA_ALPHA: float = 0.4         # EMA factor for gaze smoothing (higher = less smooth)
_DIRECTION_THRESHOLD: float = 0.25   # Dead-zone radius for CENTER classification

# How many frames between rolling-average confidence log entries.
_CONFIDENCE_LOG_INTERVAL: int = 100

# Path to the MediaPipe FaceLandmarker .task model (required by Tasks API ≥ 0.10)
_FACE_LANDMARKER_TASK: str = str(
    Path(__file__).parent.parent / "models" / "face_landmarker.task"
)

# ── MediaPipe landmark indices ────────────────────────────────────────────────
_LEFT_IRIS: int = 468    # Left iris centre
_RIGHT_IRIS: int = 473   # Right iris centre

# 6-point EAR indices per eye (Soukupova formula)
_LEFT_EAR_IDX: tuple[int, ...] = (33, 160, 158, 133, 153, 144)
_RIGHT_EAR_IDX: tuple[int, ...] = (362, 385, 387, 263, 373, 380)


# ── Public data container ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class GazeFrame:
    """
    A single processed gaze observation produced by :class:`WebcamGazeTracker`.

    Attributes:
        gaze_vector: ``(x, y)`` gaze offset, each component in ``[-1, 1]``.
            ``(0, 0)`` means straight ahead; negative-x is LEFT, negative-y
            is UP (MediaPipe y-axis is top-down, so we invert it so that
            looking *up* yields a negative y in screen space).
        direction: Dominant gaze direction string — one of
            ``'LEFT'``, ``'RIGHT'``, ``'UP'``, ``'DOWN'``, ``'CENTER'``.
        blink: ``True`` when average EAR across both eyes is below
            :data:`_EAR_BLINK_THRESHOLD`.
        confidence: Detection confidence in ``[0, 1]``.  ``0.0`` means no
            face was detected on this frame.
        timestamp_ms: Monotonic timestamp of the frame in **milliseconds**
            (``time.monotonic() * 1000``).
    """

    gaze_vector: Tuple[float, float]
    direction: str
    blink: bool
    confidence: float
    timestamp_ms: float


# ── Module-level private helpers ──────────────────────────────────────────────

def _compute_ear(landmarks: object, indices: tuple[int, ...]) -> float:
    """
    Compute Eye Aspect Ratio (EAR) from 6 landmark points.

    Formula: ``EAR = (||p2−p6|| + ||p3−p5||) / (2 × ||p1−p4||)``

    Typical values: ~0.30 for open eye, < 0.21 for blink.

    Args:
        landmarks: MediaPipe ``NormalizedLandmarkList`` (supports index access).
        indices: 6-element tuple ``(p1, p2, p3, p4, p5, p6)`` matching the
            Soukupova EAR definition.

    Returns:
        EAR as a float in ``[0, ~0.5]``; returns ``0.0`` if horizontal
        distance is degenerate (< 1e-6).
    """
    pts = [(landmarks[i].x, landmarks[i].y) for i in indices]  # type: ignore[index]

    def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    v1 = _dist(pts[1], pts[5])
    v2 = _dist(pts[2], pts[4])
    h = _dist(pts[0], pts[3])
    return (v1 + v2) / (2.0 * h) if h > 1e-6 else 0.0


def _classify_direction(
    gaze_x: float,
    gaze_y: float,
    threshold: float = _DIRECTION_THRESHOLD,
) -> str:
    """
    Map a smoothed ``(gaze_x, gaze_y)`` vector to a direction string.

    The dead-zone is a square of half-side *threshold* centred at the
    origin.  Outside the dead-zone the dominant axis wins; ties go to
    the vertical axis.

    Args:
        gaze_x: Horizontal gaze component in ``[-1, 1]``.
        gaze_y: Vertical gaze component in ``[-1, 1]`` (negative = UP).
        threshold: Dead-zone half-size.

    Returns:
        One of ``'CENTER'``, ``'LEFT'``, ``'RIGHT'``, ``'UP'``, ``'DOWN'``.
    """
    if abs(gaze_x) < threshold and abs(gaze_y) < threshold:
        return "CENTER"
    if abs(gaze_y) >= abs(gaze_x):
        return "UP" if gaze_y < 0.0 else "DOWN"
    return "LEFT" if gaze_x < 0.0 else "RIGHT"


# ── Main tracker ──────────────────────────────────────────────────────────────

class WebcamGazeTracker:
    """
    Real-time gaze tracker using MediaPipe FaceMesh with iris landmark
    refinement.

    Captures webcam frames in a daemon background thread.  Each frame is
    processed to extract iris position, blink state, and gaze direction.
    Consumers call :meth:`get_latest` from any thread to retrieve the most
    recent :class:`GazeFrame`.

    Args:
        camera_id: OpenCV camera index (``0`` = default webcam).
        fps: Target capture frame-rate.  The loop sleeps to honour this
            budget; actual FPS depends on USB bus and processing time.

    Example::

        tracker = WebcamGazeTracker(camera_id=0, fps=30)
        tracker.start()
        frame = tracker.get_latest()
        if frame:
            print(frame.direction, frame.gaze_vector)
        tracker.stop()
    """

    # ── Inner class: DwellDetector ────────────────────────────────────────

    class DwellDetector:
        """
        Detects when the gaze direction is held steady long enough to count
        as a deliberate dwell selection.

        The detector maintains a single *current direction* and the monotonic
        time at which that direction was first seen.  Calling
        :meth:`check_dwell` with the same direction repeatedly will return
        ``True`` once the elapsed time exceeds *dwell_ms*.  Passing a
        *different* direction resets the internal clock.

        Args:
            dwell_ms: Required hold time in milliseconds.  Defaults to
                ``C.GAZE_DWELL_MS`` (1 500 ms).

        Example::

            dwell = WebcamGazeTracker.DwellDetector()
            while True:
                frame = tracker.get_latest()
                if dwell.check_dwell(frame.direction):
                    print("Selected:", frame.direction)
                    dwell.reset_dwell()
        """

        def __init__(self, dwell_ms: float = C.GAZE_DWELL_MS) -> None:
            """Initialise with hold threshold *dwell_ms* milliseconds."""
            self._dwell_ms: float = dwell_ms
            self._current_direction: Optional[str] = None
            self._start_ms: float = 0.0

        def check_dwell(self, direction: str) -> bool:
            """
            Update the dwell state and return whether dwell time has been met.

            If *direction* differs from the previous call the timer resets.
            Consecutive calls with the same *direction* accumulate time until
            the threshold is reached.

            Args:
                direction: Current gaze direction string (e.g. ``'LEFT'``).

            Returns:
                ``True`` on the **first** call after the dwell threshold is
                met (and on all subsequent calls until :meth:`reset_dwell` is
                called).
            """
            now_ms = time.monotonic() * 1000.0
            if direction != self._current_direction:
                self._current_direction = direction
                self._start_ms = now_ms
                return False
            return (now_ms - self._start_ms) >= self._dwell_ms

        def reset_dwell(self) -> None:
            """Reset the dwell timer and clear the remembered direction."""
            self._current_direction = None
            self._start_ms = 0.0

    # ── Constructor ───────────────────────────────────────────────────────

    def __init__(self, camera_id: int = 0, fps: int = 30) -> None:
        """
        Initialise resources but do **not** open the camera yet.

        Call :meth:`start` to begin capture.

        Args:
            camera_id: OpenCV camera device index.
            fps: Desired capture frame-rate.
        """
        self._camera_id: int = camera_id
        self._fps: int = fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._running: bool = False
        self._camera_error: bool = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Optional[GazeFrame] = None

        # EMA state — initialised to (0, 0) = straight ahead in [-1,1] space
        self._smooth_x: float = 0.0
        self._smooth_y: float = 0.0

        # MediaPipe FaceLandmarker — Tasks API (mediapipe ≥ 0.10, replaces solutions.face_mesh)
        _base = _mp_tasks.BaseOptions(model_asset_path=_FACE_LANDMARKER_TASK)
        _opts = _mp_vision.FaceLandmarkerOptions(
            base_options=_base,
            running_mode=_mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=_GAZE_MIN_CONFIDENCE,
            min_face_presence_confidence=_GAZE_MIN_CONFIDENCE,
            min_tracking_confidence=_GAZE_MIN_CONFIDENCE,
        )
        self._face_landmarker = _mp_vision.FaceLandmarker.create_from_options(_opts)

        log = get_logger()
        log.info(
            "gaze",
            "tracker_init",
            {"camera_id": camera_id, "fps": fps, "dwell_ms": C.GAZE_DWELL_MS},
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Open the webcam and launch the background capture thread.

        If the camera cannot be opened an error is logged and an internal
        flag is set so that :meth:`get_latest` returns ``None`` gracefully.
        No exception is raised so the rest of the application can continue
        in a degraded / simulated-input mode.
        """
        log = get_logger()
        self._cap = cv2.VideoCapture(self._camera_id)

        if not self._cap.isOpened():
            self._camera_error = True
            self._cap = None
            log.error(
                "gaze",
                "camera_open_failed",
                {
                    "camera_id": self._camera_id,
                    "hint": "check device index or use gaze_sim.py",
                },
            )
            return

        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            name="gaze-capture",
            daemon=True,
        )
        self._thread.start()
        log.info(
            "gaze",
            "tracker_started",
            {"camera_id": self._camera_id, "target_fps": self._fps},
        )

    def stop(self) -> None:
        """
        Signal the background thread to stop and release all resources.

        Blocks up to 3 seconds for the thread to join.  Safe to call
        multiple times or when the tracker was never started.
        """
        log = get_logger()
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        try:
            self._face_landmarker.close()
        except Exception:  # noqa: BLE001
            pass

        log.info("gaze", "tracker_stopped", {"camera_id": self._camera_id})

    # ── Consumer API ──────────────────────────────────────────────────────

    def get_latest(self) -> Optional[GazeFrame]:
        """
        Return the most recent :class:`GazeFrame` (thread-safe).

        Returns:
            The latest :class:`GazeFrame`, or ``None`` if the camera failed
            to open or no frame has been produced yet.
        """
        if self._camera_error:
            return None
        with self._lock:
            return self._latest

    # ── Background thread ─────────────────────────────────────────────────

    def _loop(self) -> None:
        """
        Background capture loop — runs until :meth:`stop` is called.

        Throttles to the target FPS via ``time.sleep``.  Logs the rolling
        average detection confidence every :data:`_CONFIDENCE_LOG_INTERVAL`
        frames.
        """
        log = get_logger()
        interval: float = 1.0 / self._fps

        # Rolling confidence accumulators (local to this thread — no lock needed)
        conf_sum: float = 0.0
        frame_count: int = 0

        while self._running:
            t0 = time.monotonic()

            # Defensive guard — cap may be None if stop() races with us
            if self._cap is None or not self._cap.isOpened():
                time.sleep(interval)
                continue

            ret, bgr = self._cap.read()
            if not ret:
                log.warn("gaze", "frame_read_failed", {"camera_id": self._camera_id})
                time.sleep(interval)
                continue

            frame = self._process(bgr)

            with self._lock:
                self._latest = frame

            # Rolling confidence tracking
            conf_sum += frame.confidence
            frame_count += 1

            if frame_count >= _CONFIDENCE_LOG_INTERVAL:
                avg_conf = conf_sum / frame_count
                log.info(
                    "gaze",
                    "confidence_avg",
                    {
                        "avg_confidence": round(avg_conf, 4),
                        "frames_sampled": frame_count,
                        "camera_id": self._camera_id,
                    },
                )
                conf_sum = 0.0
                frame_count = 0

            # FPS throttle
            elapsed = time.monotonic() - t0
            sleep_remaining = interval - elapsed
            if sleep_remaining > 0.0:
                time.sleep(sleep_remaining)

    # ── Frame processing ──────────────────────────────────────────────────

    def _process(self, bgr: np.ndarray) -> GazeFrame:
        """
        Extract a :class:`GazeFrame` from a raw BGR camera frame.

        Processing steps:

        1. Convert BGR → RGB and pass to MediaPipe FaceMesh.
        2. If no face detected: return a zero-vector CENTER frame with
           ``confidence=0.0`` (keeps EMA state unchanged).
        3. Average left (468) and right (473) iris centres.
        4. Convert ``[0, 1]`` MediaPipe coords to ``[-1, 1]``:
           ``gaze_x = (raw_x − 0.5) × 2``.  Y is negated so that
           looking *up* yields a **negative** y value.
        5. Apply EMA smoothing.
        6. Compute per-eye EAR; average EAR < threshold → blink.
        7. Classify direction via :func:`_classify_direction`.

        Args:
            bgr: Raw OpenCV BGR frame from the webcam.

        Returns:
            A frozen :class:`GazeFrame`.
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        timestamp_ms = time.monotonic() * 1000.0

        # Tasks API: wrap frame as mp.Image and call detect_for_video
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._face_landmarker.detect_for_video(mp_image, int(timestamp_ms))

        if not results.face_landmarks:
            # No face — hold last smoothed position, zero confidence
            return GazeFrame(
                gaze_vector=(self._smooth_x, self._smooth_y),
                direction=_classify_direction(self._smooth_x, self._smooth_y),
                blink=False,
                confidence=0.0,
                timestamp_ms=timestamp_ms,
            )

        lm = results.face_landmarks[0]  # list of NormalizedLandmark (same .x/.y interface)

        # ── Iris position ─────────────────────────────────────────────────
        raw_x = float((lm[_LEFT_IRIS].x + lm[_RIGHT_IRIS].x) / 2.0)
        raw_y = float((lm[_LEFT_IRIS].y + lm[_RIGHT_IRIS].y) / 2.0)

        # Remap [0, 1] → [-1, 1]; negate y so UP = negative
        norm_x = float(np.clip((raw_x - 0.5) * 2.0, -1.0, 1.0))
        norm_y = float(np.clip((raw_y - 0.5) * -2.0, -1.0, 1.0))

        # ── EMA smoothing ─────────────────────────────────────────────────
        α = _GAZE_EMA_ALPHA
        self._smooth_x = α * norm_x + (1.0 - α) * self._smooth_x
        self._smooth_y = α * norm_y + (1.0 - α) * self._smooth_y

        # ── Blink via EAR ─────────────────────────────────────────────────
        left_ear = _compute_ear(lm, _LEFT_EAR_IDX)
        right_ear = _compute_ear(lm, _RIGHT_EAR_IDX)
        blink = (left_ear + right_ear) / 2.0 < _EAR_BLINK_THRESHOLD

        # ── Direction classification ──────────────────────────────────────
        direction = _classify_direction(self._smooth_x, self._smooth_y)

        return GazeFrame(
            gaze_vector=(self._smooth_x, self._smooth_y),
            direction=direction,
            blink=blink,
            confidence=1.0,
            timestamp_ms=timestamp_ms,
        )
