"""
sentinel/gaze/tracker.py — MediaPipe Iris eye-gaze tracker.

Captures webcam frames, extracts iris landmarks via MediaPipe FaceMesh,
and computes a normalised (x, y) gaze point in [0, 1] screen coordinates.
Falls back to centre (0.5, 0.5) if no face is detected.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from sentinel.core.config import CameraConfig, GazeConfig

logger = logging.getLogger(__name__)

# MediaPipe FaceMesh iris landmark indices
# Left iris centre: 468, Right iris centre: 473
_LEFT_IRIS_IDX: int = 468
_RIGHT_IRIS_IDX: int = 473

# Left and right eye corner indices for reference normalisation
_LEFT_EYE_INNER: int = 133
_LEFT_EYE_OUTER: int = 33
_RIGHT_EYE_INNER: int = 362
_RIGHT_EYE_OUTER: int = 263

# EAR landmark indices (per eye, 6 points each)
_LEFT_EAR_IDX: tuple[int, ...] = (33, 160, 158, 133, 153, 144)
_RIGHT_EAR_IDX: tuple[int, ...] = (362, 385, 387, 263, 373, 380)


@dataclass
class GazeFrame:
    """
    A single gaze observation from the tracker.

    Attributes:
        x: Normalised horizontal gaze position [0=left, 1=right].
        y: Normalised vertical gaze position [0=top, 1=bottom].
        timestamp: Monotonic time of capture.
        confidence: Detection confidence in [0, 1]; 0 = fallback centre.
        left_ear: Eye aspect ratio for the left eye (for blink detection).
        right_ear: Eye aspect ratio for the right eye (for blink detection).
        raw_frame: Optional BGR camera frame (for UI preview display).
    """

    x: float
    y: float
    timestamp: float
    confidence: float
    left_ear: float
    right_ear: float
    raw_frame: Optional[np.ndarray] = None


class GazeTracker:
    """
    Webcam-based gaze tracker using MediaPipe FaceMesh with iris refinement.

    Runs capture in a background thread and exposes the most recent
    :class:`GazeFrame` via :meth:`get_frame`. Applies exponential moving
    average to smooth raw gaze jitter.

    Args:
        config: Camera hardware configuration.
        gaze_config: Gaze processing parameters.
    """

    def __init__(self, config: CameraConfig, gaze_config: GazeConfig) -> None:
        """Initialise tracker resources but do not open the camera yet."""
        self._cfg = config
        self._gaze_cfg = gaze_config
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Smoothed gaze position (EMA)
        self._smooth_x: float = 0.5
        self._smooth_y: float = 0.5

        # Latest frame available to consumers
        self._latest_frame = GazeFrame(
            x=0.5, y=0.5, timestamp=time.monotonic(),
            confidence=0.0, left_ear=0.3, right_ear=0.3,
        )

        # MediaPipe FaceMesh (with iris landmarks enabled)
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(  # type: ignore[attr-defined]
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # Required for iris landmarks 468–477
            min_detection_confidence=gaze_config.confidence_min,
            min_tracking_confidence=gaze_config.confidence_min,
        )

        logger.info(
            "GazeTracker initialised (camera=%d, %dx%d @ %dfps)",
            config.index, config.width, config.height, config.fps,
        )

    # ──────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────

    def start(self) -> None:
        """
        Open the webcam and start the background capture thread.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        self._cap = cv2.VideoCapture(self._cfg.index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self._cfg.index}. "
                "Check that a webcam is connected or use --mode simulator."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
        self._cap.set(cv2.CAP_PROP_FPS, self._cfg.fps)

        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, name="gaze-capture", daemon=True
        )
        self._thread.start()
        logger.info("GazeTracker started")

    def stop(self) -> None:
        """Stop the capture thread and release camera resources."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()
        self._face_mesh.close()
        logger.info("GazeTracker stopped")

    # ──────────────────────────────────────────
    # Consumer API
    # ──────────────────────────────────────────

    def get_frame(self) -> GazeFrame:
        """
        Return the most recent gaze frame (thread-safe).

        Returns:
            The latest :class:`GazeFrame` processed by the capture thread.
        """
        with self._lock:
            return self._latest_frame

    # ──────────────────────────────────────────
    # Capture loop (background thread)
    # ──────────────────────────────────────────

    def _capture_loop(self) -> None:
        """
        Continuously read camera frames and update the latest gaze frame.

        This method runs in the background thread started by :meth:`start`.
        """
        assert self._cap is not None
        frame_interval = 1.0 / self._cfg.fps

        while self._running:
            t0 = time.monotonic()
            ret, bgr = self._cap.read()
            if not ret:
                logger.warning("Camera read failed — using previous frame")
                time.sleep(frame_interval)
                continue

            gaze_frame = self._process_frame(bgr)

            with self._lock:
                self._latest_frame = gaze_frame

            elapsed = time.monotonic() - t0
            sleep_remaining = frame_interval - elapsed
            if sleep_remaining > 0:
                time.sleep(sleep_remaining)

    def _process_frame(self, bgr: np.ndarray) -> GazeFrame:
        """
        Process a raw BGR camera frame into a GazeFrame.

        Converts to RGB, runs MediaPipe inference, extracts iris position
        and EAR values, and applies EMA smoothing.

        Args:
            bgr: Raw BGR frame from OpenCV capture.

        Returns:
            A :class:`GazeFrame` with extracted gaze position and EAR values.
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)
        rgb.flags.writeable = True

        timestamp = time.monotonic()

        if not results.multi_face_landmarks:
            # No face detected — return centre with zero confidence
            return GazeFrame(
                x=0.5, y=0.5,
                timestamp=timestamp,
                confidence=0.0,
                left_ear=self._gaze_cfg.blink_ear_threshold + 0.1,
                right_ear=self._gaze_cfg.blink_ear_threshold + 0.1,
                raw_frame=bgr,
            )

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = bgr.shape[:2]

        # Extract iris centre gaze position
        left_iris = landmarks[_LEFT_IRIS_IDX]
        right_iris = landmarks[_RIGHT_IRIS_IDX]

        # Average both irises for a single gaze point
        raw_x = (left_iris.x + right_iris.x) / 2.0
        raw_y = (left_iris.y + right_iris.y) / 2.0

        # Clamp to [0, 1]
        raw_x = float(np.clip(raw_x, 0.0, 1.0))
        raw_y = float(np.clip(raw_y, 0.0, 1.0))

        # EMA smoothing
        alpha = self._gaze_cfg.gaze_smoothing_alpha
        self._smooth_x = alpha * raw_x + (1.0 - alpha) * self._smooth_x
        self._smooth_y = alpha * raw_y + (1.0 - alpha) * self._smooth_y

        # Compute EAR for both eyes
        left_ear = _compute_ear(landmarks, _LEFT_EAR_IDX)
        right_ear = _compute_ear(landmarks, _RIGHT_EAR_IDX)

        return GazeFrame(
            x=self._smooth_x,
            y=self._smooth_y,
            timestamp=timestamp,
            confidence=1.0,
            left_ear=left_ear,
            right_ear=right_ear,
            raw_frame=bgr,
        )


def _compute_ear(landmarks: object, indices: tuple[int, ...]) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) from 6 facial landmark points.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Where p1..p6 are the 6 eye landmark points in order.

    Args:
        landmarks: MediaPipe face mesh landmark list.
        indices: Tuple of 6 landmark indices (corner, top×2, corner, bottom×2).

    Returns:
        EAR float value; typically ~0.3 open, <0.21 blink.
    """
    pts = [(landmarks[i].x, landmarks[i].y) for i in indices]

    def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        """Euclidean distance between two 2D points."""
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    vertical_1 = dist(pts[1], pts[5])
    vertical_2 = dist(pts[2], pts[4])
    horizontal = dist(pts[0], pts[3])

    if horizontal < 1e-6:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)
