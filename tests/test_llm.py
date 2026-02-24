"""
tests/test_llm.py — Unit tests for PromptBuilder and MedGemmaEngine.

No real model is loaded — transformers is mocked throughout.
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from sentinel.core.config import LLMConfig
from sentinel.intent.classifier import IntentBundle
from sentinel.llm.prompt_builder import PromptBuilder, PromptTokens
from sentinel.llm.engine import MedGemmaEngine, InferenceResult, ModelNotFoundError


class TestPromptBuilder(unittest.TestCase):
    """Tests for PromptBuilder template rendering and validation."""

    def setUp(self) -> None:
        """Create a PromptBuilder for each test."""
        self.builder = PromptBuilder()

    def _complete_bundle(self) -> IntentBundle:
        """
        Create a fully populated IntentBundle for tests.

        Returns:
            A complete IntentBundle with all four fields set.
        """
        return IntentBundle(
            body_part="chest",
            sensation="pressure",
            urgency="right now",
            intensity="moderate",
        )

    def test_build_returns_non_empty_string(self) -> None:
        """PromptBuilder.build() must return a non-empty string."""
        prompt = self.builder.build(self._complete_bundle())
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 50)

    def test_build_contains_all_tokens(self) -> None:
        """The built prompt must contain all four intent token values."""
        bundle = self._complete_bundle()
        prompt = self.builder.build(bundle)
        self.assertIn("chest", prompt)
        self.assertIn("pressure", prompt)
        self.assertIn("right now", prompt)
        self.assertIn("moderate", prompt)

    def test_build_contains_safety_instruction(self) -> None:
        """The built prompt must always contain the safety instruction."""
        prompt = self.builder.build(self._complete_bundle())
        self.assertIn("Do not provide any diagnosis", prompt)

    def test_build_chat_messages_returns_two_messages(self) -> None:
        """build_chat_messages() should return exactly two messages."""
        messages = self.builder.build_chat_messages(self._complete_bundle())
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")

    def test_system_prompt_in_chat_messages(self) -> None:
        """System message must contain the ALS/AAC system context."""
        messages = self.builder.build_chat_messages(self._complete_bundle())
        self.assertIn("ALS", messages[0]["content"])
        self.assertIn("assistive communication", messages[0]["content"])

    def test_unknown_fallback_for_partial_bundle(self) -> None:
        """Partial bundles should fall back to 'unknown' in prompt tokens."""
        partial = IntentBundle(body_part="head")
        prompt = self.builder.build(partial)
        self.assertIn("unknown", prompt)

    def test_estimate_input_tokens_positive(self) -> None:
        """Token estimate should be a positive integer."""
        estimate = self.builder.estimate_input_tokens(self._complete_bundle())
        self.assertGreater(estimate, 0)


class TestPromptTokensValidation(unittest.TestCase):
    """Tests for Pydantic validation of PromptTokens."""

    def test_valid_tokens_accepted(self) -> None:
        """Valid non-empty token values should pass validation."""
        tokens = PromptTokens(
            body_part="chest",
            sensation="pressure",
            urgency="right now",
            intensity="moderate",
        )
        self.assertEqual(tokens.body_part, "chest")

    def test_empty_token_raises_validation_error(self) -> None:
        """Empty token values should raise a pydantic ValidationError."""
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            PromptTokens(
                body_part="",
                sensation="pressure",
                urgency="right now",
                intensity="moderate",
            )

    def test_whitespace_only_token_raises_validation_error(self) -> None:
        """Whitespace-only token values should raise a pydantic ValidationError."""
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            PromptTokens(
                body_part="  ",
                sensation="pressure",
                urgency="right now",
                intensity="moderate",
            )

    def test_token_values_are_lowercased(self) -> None:
        """Token values should be normalised to lowercase."""
        tokens = PromptTokens(
            body_part="Chest",
            sensation="Pressure",
            urgency="Right Now",
            intensity="Moderate",
        )
        self.assertEqual(tokens.body_part, "chest")


class TestMedGemmaEngineInit(unittest.TestCase):
    """Tests for MedGemmaEngine initialisation (no model loading)."""

    def test_engine_not_loaded_on_init(self) -> None:
        """Engine should not load model on __init__."""
        config = LLMConfig()
        engine = MedGemmaEngine(config)
        self.assertFalse(engine._loaded)

    def test_model_not_found_error_when_cache_missing(self) -> None:
        """Engine.load() should raise ModelNotFoundError if cache dir doesn't exist."""
        config = LLMConfig(cache_dir="/nonexistent/path/to/cache")
        engine = MedGemmaEngine(config)
        with self.assertRaises(ModelNotFoundError):
            engine.load()


class TestMedGemmaEngineInferMocked(unittest.TestCase):
    """Tests for MedGemmaEngine inference logic using mocked transformers."""

    def _make_engine(self) -> MedGemmaEngine:
        """
        Create an engine with a short latency budget for timeout testing.

        Returns:
            A MedGemmaEngine with latency_budget_ms=200.
        """
        config = LLMConfig(
            latency_budget_ms=200,
            max_new_tokens=20,
        )
        return MedGemmaEngine(config)

    def test_infer_returns_fallback_on_timeout(self) -> None:
        """
        infer() should return a fallback result if generation exceeds budget.

        Simulates a slow generation by sleeping longer than the budget.
        """
        engine = self._make_engine()
        engine._loaded = True  # Bypass load()

        # Mock model that sleeps longer than the budget
        slow_model = MagicMock()
        slow_tokenizer = MagicMock()

        def slow_generate(**kwargs: object) -> MagicMock:
            time.sleep(0.5)  # 500ms > 200ms budget
            return MagicMock()

        slow_model.generate = slow_generate
        slow_model.parameters.return_value = iter([MagicMock(device="cpu")])
        # Mock tokenizer returns a MagicMock that supports .to(device)
        mock_input = MagicMock()
        mock_input.__getitem__ = MagicMock(side_effect=lambda k: MagicMock(shape=(1, 10)))
        slow_tokenizer.return_value = mock_input
        slow_tokenizer.eos_token_id = 1
        slow_tokenizer.decode.return_value = ""

        engine._model = slow_model
        engine._tokenizer = slow_tokenizer

        result = engine.infer("Any prompt")
        self.assertTrue(result.truncated)
        self.assertGreater(len(result.text), 0)  # Fallback message present
        self.assertEqual(result.confidence, 0.0)

    def test_unload_clears_model(self) -> None:
        """unload() should set _model and _tokenizer to None."""
        engine = self._make_engine()
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()
        engine._loaded = True

        with patch("torch.cuda.is_available", return_value=False):
            engine.unload()

        self.assertIsNone(engine._model)
        self.assertIsNone(engine._tokenizer)
        self.assertFalse(engine._loaded)

    def test_load_raises_model_not_found_no_cache(self) -> None:
        """load() must raise ModelNotFoundError if no model cache exists."""
        config = LLMConfig(cache_dir="/tmp/nonexistent_sentinel_cache_xyz")
        engine = MedGemmaEngine(config)
        with self.assertRaises(ModelNotFoundError):
            engine.load()


class TestInferenceResult(unittest.TestCase):
    """Tests for InferenceResult dataclass."""

    def test_default_not_truncated(self) -> None:
        """Default InferenceResult should have truncated=False."""
        r = InferenceResult(text="Hello", confidence=0.9, latency_ms=500.0)
        self.assertFalse(r.truncated)

    def test_truncated_flag_set(self) -> None:
        """Explicit truncated=True should be preserved."""
        r = InferenceResult(
            text="partial", confidence=0.0, latency_ms=2400.0, truncated=True
        )
        self.assertTrue(r.truncated)


if __name__ == "__main__":
    unittest.main()
