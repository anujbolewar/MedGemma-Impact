"""
sentinel/intent/symbol_board.py — AAC symbol board with grid-based gaze selection.

Defines a multi-page, grid-based board of medical communication symbols.
The board maps normalised (x, y) gaze positions to symbols via cell lookup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from sentinel.core.config import SymbolBoardConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Symbol vocabulary — clinical AAC set
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class Symbol:
    """
    A single AAC communication symbol on the board.

    Attributes:
        label: Human-readable label shown on the button.
        category: Semantic category (BODY_PART, SENSATION, URGENCY, INTENSITY, ACTION).
        value: Normalised token value sent to the intent classifier.
        page: Board page index (0-indexed) this symbol appears on.
        row: Grid row (0-indexed).
        col: Grid column (0-indexed).
    """

    label: str
    category: str
    value: str
    page: int
    row: int
    col: int

    @property
    def region_id(self) -> str:
        """Region identifier matching DwellTracker grid quantisation."""
        return f"r{self.row}c{self.col}"


# Clinical AAC vocabulary — Page 0: Body Parts
_PAGE_0_SYMBOLS: list[dict] = [
    {"label": "Head / Face",   "category": "BODY_PART", "value": "head",    "row": 0, "col": 0},
    {"label": "Chest",         "category": "BODY_PART", "value": "chest",   "row": 0, "col": 1},
    {"label": "Abdomen",       "category": "BODY_PART", "value": "abdomen", "row": 0, "col": 2},
    {"label": "Back",          "category": "BODY_PART", "value": "back",    "row": 1, "col": 0},
    {"label": "Left Arm",      "category": "BODY_PART", "value": "left arm","row": 1, "col": 1},
    {"label": "Right Arm",     "category": "BODY_PART", "value": "right arm","row": 1, "col": 2},
    {"label": "Left Leg",      "category": "BODY_PART", "value": "left leg","row": 2, "col": 0},
    {"label": "Right Leg",     "category": "BODY_PART", "value": "right leg","row": 2, "col": 1},
    {"label": "Whole Body",    "category": "BODY_PART", "value": "whole body","row": 2, "col": 2},
]

# Page 1: Sensations
_PAGE_1_SYMBOLS: list[dict] = [
    {"label": "Pain",          "category": "SENSATION", "value": "pain",       "row": 0, "col": 0},
    {"label": "Pressure",      "category": "SENSATION", "value": "pressure",   "row": 0, "col": 1},
    {"label": "Tingling",      "category": "SENSATION", "value": "tingling",   "row": 0, "col": 2},
    {"label": "Nausea",        "category": "SENSATION", "value": "nausea",     "row": 1, "col": 0},
    {"label": "Shortness of\nBreath", "category": "SENSATION", "value": "shortness of breath", "row": 1, "col": 1},
    {"label": "Cold / Chills", "category": "SENSATION", "value": "cold",       "row": 1, "col": 2},
    {"label": "Hot / Fever",   "category": "SENSATION", "value": "hot",        "row": 2, "col": 0},
    {"label": "Discomfort",    "category": "SENSATION", "value": "discomfort", "row": 2, "col": 1},
    {"label": "Burning",       "category": "SENSATION", "value": "burning",    "row": 2, "col": 2},
]

# Page 2: Urgency (row 0–1) + Intensity (row 2)
_PAGE_2_SYMBOLS: list[dict] = [
    {"label": "Right Now",     "category": "URGENCY", "value": "right now",     "row": 0, "col": 0},
    {"label": "Getting\nWorse","category": "URGENCY", "value": "getting worse", "row": 0, "col": 1},
    {"label": "Came On\nSuddenly","category": "URGENCY","value": "sudden",      "row": 0, "col": 2},
    {"label": "Been Here\nA While","category": "URGENCY","value": "ongoing",    "row": 1, "col": 0},
    {"label": "Comes and\nGoes", "category": "URGENCY","value": "intermittent","row": 1, "col": 1},
    {"label": "Getting\nBetter","category": "URGENCY", "value": "improving",    "row": 1, "col": 2},
    {"label": "Mild",          "category": "INTENSITY","value": "mild",         "row": 2, "col": 0},
    {"label": "Moderate",      "category": "INTENSITY","value": "moderate",     "row": 2, "col": 1},
    {"label": "Severe",        "category": "INTENSITY","value": "severe",       "row": 2, "col": 2},
]

# All symbols indexed by page
_ALL_PAGES: list[list[dict]] = [_PAGE_0_SYMBOLS, _PAGE_1_SYMBOLS, _PAGE_2_SYMBOLS]


class SymbolBoard:
    """
    Multi-page grid symbol board for AAC medical communication.

    Provides gaze-coordinate → Symbol lookup, page navigation, and the
    ability to retrieve a symbol by its region ID string.

    Args:
        config: Symbol board layout configuration.
    """

    def __init__(self, config: SymbolBoardConfig) -> None:
        """Build the symbol grid from the vocabulary definition."""
        self._cfg = config
        self._current_page: int = 0

        # Build Symbol objects for all pages
        self._symbols_by_page: list[list[Symbol]] = []
        for page_idx, page_defs in enumerate(_ALL_PAGES):
            page_symbols: list[Symbol] = [
                Symbol(
                    label=d["label"],
                    category=d["category"],
                    value=d["value"],
                    page=page_idx,
                    row=d["row"],
                    col=d["col"],
                )
                for d in page_defs
            ]
            self._symbols_by_page.append(page_symbols)

        logger.info(
            "SymbolBoard ready: %d pages, %dx%d grid",
            len(self._symbols_by_page), config.rows, config.columns,
        )

    @property
    def current_page(self) -> int:
        """Return the index of the currently displayed page."""
        return self._current_page

    @property
    def current_symbols(self) -> list[Symbol]:
        """Return all symbols on the current page."""
        return self._symbols_by_page[self._current_page]

    @property
    def page_count(self) -> int:
        """Return total number of pages."""
        return len(self._symbols_by_page)

    def next_page(self) -> int:
        """
        Advance to the next page (wraps around).

        Returns:
            The new page index.
        """
        self._current_page = (self._current_page + 1) % len(self._symbols_by_page)
        logger.info("SymbolBoard: advanced to page %d", self._current_page)
        return self._current_page

    def set_page(self, page: int) -> None:
        """
        Jump directly to a specific page.

        Args:
            page: Target page index.

        Raises:
            ValueError: If page index is out of range.
        """
        if not (0 <= page < len(self._symbols_by_page)):
            raise ValueError(
                f"Page {page} out of range [0, {len(self._symbols_by_page) - 1}]"
            )
        self._current_page = page

    def select(self, x: float, y: float) -> Optional[Symbol]:
        """
        Return the symbol at normalised gaze position (x, y) on the current page.

        The board is divided into a uniform grid. Each gaze coordinate is
        mapped to the cell it falls in.

        Args:
            x: Normalised horizontal position [0, 1].
            y: Normalised vertical position [0, 1].

        Returns:
            The :class:`Symbol` at that position, or None if outside grid.
        """
        col = int(min(x * self._cfg.columns, self._cfg.columns - 1))
        row = int(min(y * self._cfg.rows, self._cfg.rows - 1))
        return self._get_symbol_at(row, col)

    def get_symbol_by_region(self, region_id: str) -> Optional[Symbol]:
        """
        Look up a symbol by its region ID string (e.g. ``'r1c2'``).

        Args:
            region_id: Region string produced by :class:`~sentinel.gaze.blink_detector.DwellTracker`.

        Returns:
            Matching :class:`Symbol` or None if not found.
        """
        for symbol in self.current_symbols:
            if symbol.region_id == region_id:
                return symbol
        return None

    def get_all_pages(self) -> list[list[Symbol]]:
        """
        Return all symbols grouped by page.

        Returns:
            List of pages, each a list of :class:`Symbol` objects.
        """
        return self._symbols_by_page

    def _get_symbol_at(self, row: int, col: int) -> Optional[Symbol]:
        """
        Return the symbol at a specific (row, col) on the current page.

        Args:
            row: Grid row index.
            col: Grid column index.

        Returns:
            Matching :class:`Symbol` or None.
        """
        for symbol in self.current_symbols:
            if symbol.row == row and symbol.col == col:
                return symbol
        return None
