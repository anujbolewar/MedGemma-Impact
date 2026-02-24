"""
tests/test_pipeline.py — Integration tests for the full Sentinel pipeline.

Uses KeyboardSimulator + mocked MedGemmaEngine to test the complete
symbol-select → intent → infer → TTS cycle without requiring a real model or camera.
"""

from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from sentinel.core.config import load_config
from sentinel.core.pipeline import (
    PipelineEvent,
    PipelineMode,
    PipelineState,
    SentinelPipeline,
)
from sentinel.llm.engine import InferenceResult
from sentinel.intent.symbol_board import Symbol


class TestPipelineInit(unittest.TestCase):
    """Tests for SentinelPipeline initialisation in simulator mode."""

    def test_pipeline_initialises_in_simulator_mode(self) -> None:
        """Pipeline should initialise without errors in simulator mode."""
        config = load_config()
        pipeline = SentinelPipeline(config=config, mode=PipelineMode.SIMULATOR)
        self.assertIsNotNone(pipeline)
        self.assertEqual(pipeline.state, PipelineState.IDLE)

    def test_pipeline_exposes_symbol_board(self) -> None:
        """Pipeline should expose a SymbolBoard with at least one page."""
        config = load_config()
        pipeline = SentinelPipeline(config=config, mode=PipelineMode.SIMULATOR)
        self.assertGreater(pipeline.symbol_board.page_count, 0)
        self.assertGreater(len(pipeline.symbol_board.current_symbols), 0)

    def test_factory_from_config_file(self) -> None:
        """from_config_file() should produce a functional pipeline."""
        pipeline = SentinelPipeline.from_config_file(mode=PipelineMode.SIMULATOR)
        self.assertIsNotNone(pipeline)


class TestPipelineIntentAccumulation(unittest.TestCase):
    """
    Integration tests: gaze symbols → intent bundle → LLM trigger.

    The LLM engine is mocked so no model download is required.
    """

    def setUp(self) -> None:
        """Create pipeline with mocked LLM and TTS."""
        self.config = load_config()
        self.events: list[PipelineEvent] = []

        self.pipeline = SentinelPipeline(
            config=self.config,
            mode=PipelineMode.SIMULATOR,
            on_event=self._capture_event,
        )

        # Mock the LLM to return immediately
        mock_result = InferenceResult(
            text="I have moderate pressure in my chest right now.",
            confidence=0.85,
            latency_ms=350.0,
        )
        self.pipeline._llm.load = MagicMock()
        self.pipeline._llm.infer = MagicMock(return_value=mock_result)

        # Mock TTS to do nothing
        self.pipeline._tts.speak = MagicMock()

    def _capture_event(self, event: PipelineEvent) -> None:
        """Capture pipeline events for assertion."""
        self.events.append(event)

    def _select_symbol(self, category: str, value: str) -> None:
        """
        Directly call _process_selection with a synthetic symbol.

        Args:
            category: Symbol category.
            value: Symbol value.
        """
        symbol = Symbol(
            label=value.title(),
            category=category,
            value=value,
            page=0, row=0, col=0,
        )
        self.pipeline._process_selection(symbol)

    def test_four_selections_triggers_inference(self) -> None:
        """Making four symbol selections should trigger LLM inference."""
        self._select_symbol("BODY_PART", "chest")
        self._select_symbol("SENSATION", "pressure")
        self._select_symbol("URGENCY", "right now")
        self._select_symbol("INTENSITY", "moderate")

        # Allow inference thread to complete
        time.sleep(0.5)

        self.pipeline._llm.infer.assert_called_once()  # type: ignore[union-attr]

    def test_sentence_ready_event_emitted(self) -> None:
        """After four selections, a 'sentence_ready' event must be emitted."""
        self._select_symbol("BODY_PART", "chest")
        self._select_symbol("SENSATION", "pressure")
        self._select_symbol("URGENCY", "right now")
        self._select_symbol("INTENSITY", "moderate")
        time.sleep(0.5)

        kinds = [e.kind for e in self.events]
        self.assertIn("sentence_ready", kinds)

    def test_sentence_content_matches_mock(self) -> None:
        """The sentence in the 'sentence_ready' event should match the mock result."""
        self._select_symbol("BODY_PART", "chest")
        self._select_symbol("SENSATION", "pressure")
        self._select_symbol("URGENCY", "right now")
        self._select_symbol("INTENSITY", "moderate")
        time.sleep(0.5)

        sentence_events = [e for e in self.events if e.kind == "sentence_ready"]
        self.assertGreater(len(sentence_events), 0)
        result: InferenceResult = sentence_events[0].payload  # type: ignore[assignment]
        self.assertIn("chest", result.text)

    def test_tts_speak_called_after_inference(self) -> None:
        """TTS.speak() should be called with the reconstructed sentence."""
        self._select_symbol("BODY_PART", "chest")
        self._select_symbol("SENSATION", "pressure")
        self._select_symbol("URGENCY", "right now")
        self._select_symbol("INTENSITY", "moderate")
        time.sleep(0.5)

        self.pipeline._tts.speak.assert_called_once()  # type: ignore[union-attr]

    def test_intent_reset_after_inference(self) -> None:
        """Intent bundle should be reset after a completed inference cycle."""
        self._select_symbol("BODY_PART", "chest")
        self._select_symbol("SENSATION", "pressure")
        self._select_symbol("URGENCY", "right now")
        self._select_symbol("INTENSITY", "moderate")
        time.sleep(0.5)

        self.assertFalse(self.pipeline._intent.is_complete())


class TestEmergencyOverrideIntegration(unittest.TestCase):
    """Integration tests for the emergency override system."""

    def setUp(self) -> None:
        """Create pipeline with mocked TTS for emergency tests."""
        config = load_config()
        self.pipeline = SentinelPipeline(
            config=config,
            mode=PipelineMode.SIMULATOR,
        )
        self.pipeline._tts.speak_emergency = MagicMock()
        self.pipeline._llm.load = MagicMock()

    def test_emergency_trigger_bypasses_llm(self) -> None:
        """Emergency trigger must not call LLM infer."""
        self.pipeline._llm.infer = MagicMock()
        triggered = self.pipeline._emergency.trigger(source="test")
        # Allow any threads to complete
        time.sleep(0.3)
        self.assertTrue(triggered)
        self.pipeline._llm.infer.assert_not_called()  # type: ignore[union-attr]

    def test_emergency_cooldown_prevents_rapid_retrigger(self) -> None:
        """Second emergency trigger within cooldown should be rejected."""
        # Temporarily reduce cooldown for test speed
        from sentinel.core.config import EmergencyConfig
        config = load_config()
        # Override cooldown to 5s to test rejection
        triggered1 = self.pipeline._emergency.trigger(source="test1")
        triggered2 = self.pipeline._emergency.trigger(source="test2")
        self.assertTrue(triggered1)
        self.assertFalse(triggered2)

    def test_emergency_logs_event(self) -> None:
        """Emergency trigger should create an audit log entry."""
        self.pipeline._emergency.trigger(source="unit_test")
        log = self.pipeline._emergency.session_log
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0].trigger_source, "unit_test")


class TestPipelineEventCallback(unittest.TestCase):
    """Tests that pipeline events are dispatched to the callback."""

    def test_gaze_events_emitted_after_start(self) -> None:
        """After start(), gaze events should be emitted on each tick."""
        config = load_config()
        events: list[PipelineEvent] = []

        pipeline = SentinelPipeline(
            config=config,
            mode=PipelineMode.SIMULATOR,
            on_event=lambda e: events.append(e),
        )
        pipeline._llm.load = MagicMock()
        pipeline._llm.infer = MagicMock()

        pipeline._gaze_source.start()
        pipeline._running = True

        # Run a few ticks manually (no background thread needed)
        for _ in range(3):
            pipeline._tick()

        pipeline._gaze_source.stop()

        gaze_events = [e for e in events if e.kind == "gaze"]
        self.assertGreater(len(gaze_events), 0)

    def test_state_change_events_emitted(self) -> None:
        """State transitions should emit 'state_change' events."""
        config = load_config()
        events: list[PipelineEvent] = []

        pipeline = SentinelPipeline(
            config=config,
            mode=PipelineMode.SIMULATOR,
            on_event=lambda e: events.append(e),
        )
        pipeline._set_state(PipelineState.SELECTING)
        pipeline._set_state(PipelineState.GENERATING)

        kinds = [e.kind for e in events]
        self.assertIn("state_change", kinds)


if __name__ == "__main__":
    unittest.main()
