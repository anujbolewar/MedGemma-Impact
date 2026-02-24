"""
tests/test_intent.py — Unit tests for intent classification and symbol board.

No LLM or camera required — pure logic tests.
"""

from __future__ import annotations

import unittest

from sentinel.core.config import SymbolBoardConfig
from sentinel.intent.classifier import IntentClassifier, IntentBundle
from sentinel.intent.symbol_board import Symbol, SymbolBoard


class TestSymbolBoard(unittest.TestCase):
    """Tests for the SymbolBoard page navigation and cell lookup."""

    def setUp(self) -> None:
        """Build a SymbolBoard with default config."""
        self.board = SymbolBoard(SymbolBoardConfig(columns=3, rows=3, pages=3))

    def test_initial_page_is_zero(self) -> None:
        """Board should start on page 0."""
        self.assertEqual(self.board.current_page, 0)

    def test_current_symbols_nonempty(self) -> None:
        """Each page should have at least one symbol."""
        for page in range(self.board.page_count):
            self.board.set_page(page)
            self.assertGreater(len(self.board.current_symbols), 0)

    def test_next_page_wraps_around(self) -> None:
        """next_page() from last page should wrap to page 0."""
        for _ in range(self.board.page_count):
            self.board.next_page()
        self.assertEqual(self.board.current_page, 0)

    def test_set_page_out_of_range_raises(self) -> None:
        """set_page() with invalid index should raise ValueError."""
        with self.assertRaises(ValueError):
            self.board.set_page(99)

    def test_select_top_left_returns_symbol(self) -> None:
        """Gaze at (0.1, 0.1) should return the top-left cell symbol."""
        symbol = self.board.select(0.1, 0.1)
        self.assertIsNotNone(symbol)
        assert symbol is not None
        self.assertEqual(symbol.row, 0)
        self.assertEqual(symbol.col, 0)

    def test_select_bottom_right_returns_symbol(self) -> None:
        """Gaze at (0.95, 0.95) should return the bottom-right cell symbol."""
        symbol = self.board.select(0.95, 0.95)
        self.assertIsNotNone(symbol)
        assert symbol is not None
        self.assertEqual(symbol.row, 2)
        self.assertEqual(symbol.col, 2)

    def test_get_symbol_by_region_id(self) -> None:
        """Symbols should be retrievable by their region_id string."""
        symbols = self.board.current_symbols
        for s in symbols:
            found = self.board.get_symbol_by_region(s.region_id)
            self.assertIsNotNone(found)
            assert found is not None
            self.assertEqual(found.label, s.label)

    def test_get_symbol_unknown_region_returns_none(self) -> None:
        """Querying a non-existent region ID should return None."""
        result = self.board.get_symbol_by_region("r9c9")
        self.assertIsNone(result)

    def test_page_0_has_body_part_category(self) -> None:
        """Page 0 should contain BODY_PART symbols."""
        self.board.set_page(0)
        categories = {s.category for s in self.board.current_symbols}
        self.assertIn("BODY_PART", categories)

    def test_page_1_has_sensation_category(self) -> None:
        """Page 1 should contain SENSATION symbols."""
        self.board.set_page(1)
        categories = {s.category for s in self.board.current_symbols}
        self.assertIn("SENSATION", categories)

    def test_page_2_has_urgency_and_intensity(self) -> None:
        """Page 2 should contain URGENCY and INTENSITY symbols."""
        self.board.set_page(2)
        categories = {s.category for s in self.board.current_symbols}
        self.assertIn("URGENCY", categories)
        self.assertIn("INTENSITY", categories)


class TestIntentClassifier(unittest.TestCase):
    """Tests for the IntentClassifier stateful bundle accumulation."""

    def _make_symbol(self, category: str, value: str) -> Symbol:
        """
        Create a minimal Symbol for testing.

        Args:
            category: Symbol category (BODY_PART, SENSATION, etc.).
            value: Token value.

        Returns:
            A Symbol object at row=0, col=0, page=0.
        """
        return Symbol(
            label=value.title(),
            category=category,
            value=value,
            page=0, row=0, col=0,
        )

    def setUp(self) -> None:
        """Fresh classifier for each test."""
        self.clf = IntentClassifier()

    def test_initial_bundle_is_incomplete(self) -> None:
        """A fresh classifier should have an incomplete bundle."""
        self.assertFalse(self.clf.is_complete())

    def test_update_adds_body_part(self) -> None:
        """Selecting a BODY_PART symbol should fill body_part in the bundle."""
        self.clf.update(self._make_symbol("BODY_PART", "chest"))
        self.assertEqual(self.clf.current_bundle.body_part, "chest")

    def test_update_adds_sensation(self) -> None:
        """Selecting a SENSATION symbol should fill sensation in the bundle."""
        self.clf.update(self._make_symbol("SENSATION", "pressure"))
        self.assertEqual(self.clf.current_bundle.sensation, "pressure")

    def test_complete_after_four_selections(self) -> None:
        """Bundle should be complete after selecting all four required categories."""
        self.clf.update(self._make_symbol("BODY_PART", "chest"))
        self.clf.update(self._make_symbol("SENSATION", "pressure"))
        self.clf.update(self._make_symbol("URGENCY", "right now"))
        self.clf.update(self._make_symbol("INTENSITY", "moderate"))
        self.assertTrue(self.clf.is_complete())

    def test_overwrite_replaces_category(self) -> None:
        """Selecting a second BODY_PART should overwrite the first (correction)."""
        self.clf.update(self._make_symbol("BODY_PART", "chest"))
        self.clf.update(self._make_symbol("BODY_PART", "back"))
        self.assertEqual(self.clf.current_bundle.body_part, "back")

    def test_missing_categories_reduces_as_filled(self) -> None:
        """missing_categories() should decrease as selections are made."""
        m0 = self.clf.missing_categories()
        self.clf.update(self._make_symbol("BODY_PART", "chest"))
        m1 = self.clf.missing_categories()
        self.assertLess(len(m1), len(m0))
        self.assertNotIn("BODY_PART", m1)

    def test_to_prompt_tokens_returns_all_keys(self) -> None:
        """to_prompt_tokens() should always return all four keys."""
        tokens = self.clf.current_bundle.to_prompt_tokens()
        for key in ("body_part", "sensation", "urgency", "intensity"):
            self.assertIn(key, tokens)

    def test_to_prompt_tokens_fallback_to_unknown(self) -> None:
        """Missing fields should default to 'unknown' in prompt tokens."""
        tokens = self.clf.current_bundle.to_prompt_tokens()
        self.assertEqual(tokens["body_part"], "unknown")

    def test_reset_clears_bundle(self) -> None:
        """reset() should produce a fresh incomplete bundle."""
        self.clf.update(self._make_symbol("BODY_PART", "chest"))
        self.clf.reset()
        self.assertFalse(self.clf.is_complete())
        self.assertIsNone(self.clf.current_bundle.body_part)

    def test_history_tracks_all_selections(self) -> None:
        """History list should accumulate all selections including overwrites."""
        self.clf.update(self._make_symbol("BODY_PART", "chest"))
        self.clf.update(self._make_symbol("BODY_PART", "back"))
        self.assertEqual(len(self.clf.current_bundle.history), 2)


class TestIntentBundle(unittest.TestCase):
    """Tests for IntentBundle data class methods."""

    def test_is_complete_requires_all_four_fields(self) -> None:
        """is_complete() should return False unless all four fields are set."""
        b = IntentBundle(body_part="chest", sensation="pain", urgency="now")
        self.assertFalse(b.is_complete())

    def test_is_complete_with_all_fields(self) -> None:
        """is_complete() should return True when all four fields are set."""
        b = IntentBundle(
            body_part="chest", sensation="pain",
            urgency="now", intensity="severe"
        )
        self.assertTrue(b.is_complete())

    def test_missing_categories_correct(self) -> None:
        """missing_categories() should list only the unset fields."""
        b = IntentBundle(body_part="chest", sensation="pressure")
        missing = b.missing_categories()
        self.assertIn("URGENCY", missing)
        self.assertIn("INTENSITY", missing)
        self.assertNotIn("BODY_PART", missing)
        self.assertNotIn("SENSATION", missing)


if __name__ == "__main__":
    unittest.main()
