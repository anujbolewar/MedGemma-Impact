"""
tests/test_gaze.py — Unit tests for gaze tracking, blink detection, and keyboard simulator.

Tests use synthetic landmark data — no camera or MediaPipe required.
"""

from __future__ import annotations

import time
import unittest

from sentinel.core.config import GazeConfig
from sentinel.gaze.blink_detector import BlinkDetector, DwellTracker, BlinkEvent
from sentinel.gaze.simulator import KeyboardSimulator
from sentinel.gaze.tracker import GazeFrame, _compute_ear


class TestComputeEAR(unittest.TestCase):
    """Tests for the Eye Aspect Ratio computation function."""

    def _make_landmark(self, x: float, y: float) -> object:
        """
        Create a minimal mock landmark object.

        Args:
            x: Landmark x coordinate.
            y: Landmark y coordinate.

        Returns:
            A simple namespace object with x and y attributes.
        """
        class _Landmark:
            pass
        lm = _Landmark()
        lm.x = x  # type: ignore[attr-defined]
        lm.y = y  # type: ignore[attr-defined]
        return lm

    def test_open_eye_ear_above_threshold(self) -> None:
        """EAR for a wide-open eye should be well above 0.21."""
        # Horizontal span: 0.1, vertical span: 0.05 each → EAR ≈ 0.5
        landmarks = {
            33: self._make_landmark(0.0, 0.5),   # outer corner
            160: self._make_landmark(0.033, 0.55), # top-left
            158: self._make_landmark(0.067, 0.55), # top-right
            133: self._make_landmark(0.1, 0.5),   # inner corner
            153: self._make_landmark(0.033, 0.45), # bottom-left
            144: self._make_landmark(0.067, 0.45), # bottom-right
        }
        indices = (33, 160, 158, 133, 153, 144)

        class _FakeLandmarks:
            def __getitem__(self, i: int) -> object:
                return landmarks[i]

        ear = _compute_ear(_FakeLandmarks(), indices)
        self.assertGreater(ear, 0.21, "Open eye EAR should be above blink threshold")

    def test_closed_eye_ear_below_threshold(self) -> None:
        """EAR for a nearly closed eye should be below 0.21."""
        # All vertical landmark pairs fixed at same x to isolate vertical distance.
        # p2=(0,0.501) vs p6=(0,0.499): dist=0.002
        # p3=(0,0.501) vs p5=(0,0.499): dist=0.002
        # horizontal: p1=(0,0.5) vs p4=(0.2,0.5): dist=0.2
        # EAR = (0.002+0.002)/(2*0.2) = 0.01 — well below 0.21
        landmarks = {
            33:  self._make_landmark(0.0, 0.5),
            160: self._make_landmark(0.0, 0.501),
            158: self._make_landmark(0.0, 0.501),
            133: self._make_landmark(0.2, 0.5),
            153: self._make_landmark(0.0, 0.499),
            144: self._make_landmark(0.0, 0.499),
        }
        indices = (33, 160, 158, 133, 153, 144)

        class _FakeLandmarks:
            def __getitem__(self, i: int) -> object:
                return landmarks[i]

        ear = _compute_ear(_FakeLandmarks(), indices)
        self.assertLess(ear, 0.21, "Closed eye EAR should be below blink threshold")

    def test_zero_horizontal_span_does_not_raise(self) -> None:
        """EAR computation with zero horizontal span should return 0.0 safely."""
        landmarks = {i: self._make_landmark(0.5, 0.5) for i in (33, 160, 158, 133, 153, 144)}
        indices = (33, 160, 158, 133, 153, 144)

        class _FakeLandmarks:
            def __getitem__(self, i: int) -> object:
                return landmarks[i]

        ear = _compute_ear(_FakeLandmarks(), indices)
        self.assertEqual(ear, 0.0)


class TestBlinkDetector(unittest.TestCase):
    """Tests for the BlinkDetector state machine."""

    def _make_frame(self, ear: float) -> GazeFrame:
        """
        Create a GazeFrame with the given EAR for both eyes.

        Args:
            ear: Eye Aspect Ratio value for both eyes.

        Returns:
            A synthetic GazeFrame.
        """
        return GazeFrame(
            x=0.5, y=0.5,
            timestamp=time.monotonic(),
            confidence=1.0,
            left_ear=ear,
            right_ear=ear,
        )

    def setUp(self) -> None:
        """Set up a BlinkDetector with standard test config."""
        self.cfg = GazeConfig(
            blink_ear_threshold=0.21,
            blink_consec_frames=3,
            long_blink_frames=12,
        )
        self.detector = BlinkDetector(self.cfg)

    def test_no_blink_when_eye_open(self) -> None:
        """No BlinkEvent should fire when EAR stays well above threshold."""
        for _ in range(10):
            result = self.detector.detect(self._make_frame(0.35))
        self.assertIsNone(result)

    def test_short_blink_fires_on_rise(self) -> None:
        """A short blink (3–11 frames) should fire exactly once on EAR recovery."""
        # 5 frames below threshold
        for _ in range(5):
            event = self.detector.detect(self._make_frame(0.10))
            self.assertIsNone(event, msg="No event should fire during blink")

        # EAR recovers
        event = self.detector.detect(self._make_frame(0.35))
        self.assertIsNotNone(event, msg="Event should fire on EAR recovery")
        assert event is not None
        self.assertFalse(event.is_long)
        self.assertEqual(event.duration_frames, 5)

    def test_long_blink_fires_on_rise(self) -> None:
        """A long blink (≥12 frames) should fire with is_long=True."""
        for _ in range(15):
            self.detector.detect(self._make_frame(0.08))
        event = self.detector.detect(self._make_frame(0.35))
        self.assertIsNotNone(event)
        assert event is not None
        self.assertTrue(event.is_long)

    def test_below_consec_threshold_does_not_fire(self) -> None:
        """Fewer than consec_frames below threshold should not produce a blink."""
        for _ in range(2):  # Less than consec_frames=3
            self.detector.detect(self._make_frame(0.10))
        event = self.detector.detect(self._make_frame(0.35))
        self.assertIsNone(event)

    def test_second_blink_detectable_after_recovery(self) -> None:
        """The detector should be able to detect a second blink after recovery."""
        for _ in range(4):
            self.detector.detect(self._make_frame(0.10))
        self.detector.detect(self._make_frame(0.35))  # First blink

        for _ in range(4):
            self.detector.detect(self._make_frame(0.10))
        event = self.detector.detect(self._make_frame(0.35))  # Second blink
        self.assertIsNotNone(event)


class TestDwellTracker(unittest.TestCase):
    """Tests for DwellTracker region quantisation and threshold firing."""

    def setUp(self) -> None:
        """Create a DwellTracker with a short threshold for fast tests."""
        self.tracker = DwellTracker(
            dwell_threshold_ms=100,  # 100ms for speed
            grid_cols=3,
            grid_rows=3,
        )

    def test_no_event_before_threshold(self) -> None:
        """No DwellEvent should fire immediately after gaze enters a region."""
        event = self.tracker.update(0.1, 0.1)
        self.assertIsNone(event)

    def test_event_fires_after_threshold(self) -> None:
        """DwellEvent should fire once the threshold duration elapses."""
        self.tracker.update(0.1, 0.1)
        time.sleep(0.12)  # Wait past 100ms
        event = self.tracker.update(0.1, 0.1)
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.region_id, "r0c0")

    def test_event_does_not_re_fire(self) -> None:
        """Dwell event must not fire repeatedly for the same region."""
        self.tracker.update(0.1, 0.1)
        time.sleep(0.12)
        self.tracker.update(0.1, 0.1)  # First fire
        event = self.tracker.update(0.1, 0.1)  # Should not fire again
        self.assertIsNone(event)

    def test_region_change_resets_timer(self) -> None:
        """Moving gaze to a new region should reset the dwell timer."""
        self.tracker.update(0.1, 0.1)
        time.sleep(0.06)
        # Move to new region — timer resets
        self.tracker.update(0.5, 0.5)
        time.sleep(0.06)
        # Only 60ms in new region — should not fire
        event = self.tracker.update(0.5, 0.5)
        self.assertIsNone(event)

    def test_progress_increases_over_time(self) -> None:
        """Dwell progress should monotonically increase while in the same region."""
        self.tracker.update(0.1, 0.1)
        time.sleep(0.04)
        p1 = self.tracker.get_dwell_progress(0.1, 0.1)
        time.sleep(0.04)
        p2 = self.tracker.get_dwell_progress(0.1, 0.1)
        self.assertGreater(p2, p1)

    def test_reset_clears_state(self) -> None:
        """After reset, no event fires until threshold is reached again."""
        self.tracker.update(0.1, 0.1)
        time.sleep(0.12)
        self.tracker.reset()
        event = self.tracker.update(0.1, 0.1)
        self.assertIsNone(event)


class TestKeyboardSimulator(unittest.TestCase):
    """Tests for KeyboardSimulator gaze position and blink injection."""

    def setUp(self) -> None:
        """Create a simulator with default config."""
        self.cfg = GazeConfig()
        self.sim = KeyboardSimulator(self.cfg)
        self.sim.start()

    def tearDown(self) -> None:
        """Stop the simulator after each test."""
        self.sim.stop()

    def test_initial_position_is_centre(self) -> None:
        """Simulator should initialise at (0.5, 0.5)."""
        x, y = self.sim.position
        self.assertAlmostEqual(x, 0.5)
        self.assertAlmostEqual(y, 0.5)

    def test_right_key_moves_gaze_right(self) -> None:
        """Pressing Right should increase x coordinate."""
        self.sim.inject_key("Right")
        x, _ = self.sim.position
        self.assertGreater(x, 0.5)

    def test_left_key_moves_gaze_left(self) -> None:
        """Pressing Left should decrease x coordinate."""
        self.sim.inject_key("Left")
        x, _ = self.sim.position
        self.assertLess(x, 0.5)

    def test_up_key_moves_gaze_up(self) -> None:
        """Pressing Up should decrease y coordinate."""
        self.sim.inject_key("Up")
        _, y = self.sim.position
        self.assertLess(y, 0.5)

    def test_down_key_moves_gaze_down(self) -> None:
        """Pressing Down should increase y coordinate."""
        self.sim.inject_key("Down")
        _, y = self.sim.position
        self.assertGreater(y, 0.5)

    def test_space_injects_short_blink(self) -> None:
        """Pressing Space should inject a short BlinkEvent on next get_frame()."""
        self.sim.inject_key("space")
        frame = self.sim.get_frame()
        self.assertIsNotNone(frame.blink_event)
        assert frame.blink_event is not None
        self.assertFalse(frame.blink_event.is_long)

    def test_b_key_injects_long_blink(self) -> None:
        """Pressing 'b' should inject a long BlinkEvent."""
        self.sim.inject_key("b")
        frame = self.sim.get_frame()
        self.assertIsNotNone(frame.blink_event)
        assert frame.blink_event is not None
        self.assertTrue(frame.blink_event.is_long)

    def test_blink_consumed_once(self) -> None:
        """A blink event should only be returned once from get_frame()."""
        self.sim.inject_key("space")
        self.sim.get_frame()  # Consume
        frame2 = self.sim.get_frame()
        self.assertIsNone(frame2.blink_event)

    def test_reset_key_returns_to_centre(self) -> None:
        """Pressing 'r' should reset gaze to (0.5, 0.5)."""
        self.sim.inject_key("Right")
        self.sim.inject_key("Down")
        self.sim.inject_key("r")
        x, y = self.sim.position
        self.assertAlmostEqual(x, 0.5)
        self.assertAlmostEqual(y, 0.5)

    def test_x_clamps_to_bounds(self) -> None:
        """Gaze should not exceed [0, 1] bounds regardless of key presses."""
        for _ in range(10):
            self.sim.inject_key("Right")
        x, _ = self.sim.position
        self.assertLessEqual(x, 1.0)

        for _ in range(10):
            self.sim.inject_key("Left")
        x, _ = self.sim.position
        self.assertGreaterEqual(x, 0.0)


if __name__ == "__main__":
    unittest.main()
