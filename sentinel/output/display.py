"""
sentinel/output/display.py — On-screen display manager for reconstructed sentences.

Provides thread-safe methods to update the Tkinter UI with the reconstructed
sentence, confidence badge, dwell progress rings, and status messages.
All updates are dispatched via Tkinter's `after()` to ensure thread safety.
"""

from __future__ import annotations

import logging
import tkinter as tk
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Colour palette (dark theme)
_BG_DARK = "#0d0d0d"
_BG_PANEL = "#1a1a2e"
_ACCENT_BLUE = "#4a90e2"
_ACCENT_GREEN = "#27ae60"
_ACCENT_AMBER = "#f39c12"
_ACCENT_RED = "#e74c3c"
_TEXT_PRIMARY = "#ecf0f1"
_TEXT_SECONDARY = "#95a5a6"
_CONFIDENCE_LOW = "#e74c3c"
_CONFIDENCE_MED = "#f39c12"
_CONFIDENCE_HIGH = "#27ae60"


def _confidence_colour(score: float) -> str:
    """
    Map a confidence score in [0, 1] to a traffic-light colour.

    Args:
        score: Confidence value in [0.0, 1.0].

    Returns:
        Hex colour string.
    """
    if score >= 0.75:
        return _CONFIDENCE_HIGH
    elif score >= 0.45:
        return _CONFIDENCE_MED
    return _CONFIDENCE_LOW


class DisplayManager:
    """
    Manages all dynamic UI text regions in the Sentinel application.

    All public methods are safe to call from any thread — they schedule
    Tkinter widget updates via :meth:`tk.Tk.after` to execute on the
    main GUI thread.

    Args:
        root: The root Tkinter window.
        font_family: Font family for all rendered text.
        sentence_font_size: Point size for the main sentence display.
        board_font_size: Point size for symbol board labels.
        status_font_size: Point size for status bar messages.
    """

    def __init__(
        self,
        root: tk.Tk,
        font_family: str = "Arial",
        sentence_font_size: int = 36,
        board_font_size: int = 18,
        status_font_size: int = 12,
    ) -> None:
        """Build all display widget containers (called from GUI thread only)."""
        self._root = root
        self._font_family = font_family
        self._sentence_font_size = sentence_font_size
        self._board_font_size = board_font_size
        self._status_font_size = status_font_size

        # References to managed labels/widgets set by the UI builder
        self._sentence_var: Optional[tk.StringVar] = None
        self._confidence_var: Optional[tk.StringVar] = None
        self._status_var: Optional[tk.StringVar] = None
        self._latency_var: Optional[tk.StringVar] = None
        self._confidence_label: Optional[tk.Label] = None

        # Emergency overlay frame
        self._emergency_overlay: Optional[tk.Frame] = None

    # ──────────────────────────────────────────
    # Widget registration
    # ──────────────────────────────────────────

    def register_sentence_var(self, var: tk.StringVar) -> None:
        """
        Register the StringVar driving the main sentence label.

        Args:
            var: Tkinter StringVar bound to the sentence display label.
        """
        self._sentence_var = var

    def register_confidence_var(
        self, var: tk.StringVar, label: Optional[tk.Label] = None
    ) -> None:
        """
        Register the StringVar and optional Label for the confidence badge.

        Args:
            var: StringVar bound to the confidence text.
            label: Optional Label widget for colour updates.
        """
        self._confidence_var = var
        self._confidence_label = label

    def register_status_var(self, var: tk.StringVar) -> None:
        """
        Register the StringVar for the bottom status bar.

        Args:
            var: StringVar bound to the status label.
        """
        self._status_var = var

    def register_latency_var(self, var: tk.StringVar) -> None:
        """
        Register the StringVar for the latency indicator.

        Args:
            var: StringVar bound to the latency display label.
        """
        self._latency_var = var

    def register_emergency_overlay(self, frame: tk.Frame) -> None:
        """
        Register the full-screen emergency overlay frame.

        The frame is hidden by default and shown on emergency trigger.

        Args:
            frame: A Tkinter Frame that covers the full window.
        """
        self._emergency_overlay = frame

    # ──────────────────────────────────────────
    # Update API (thread-safe)
    # ──────────────────────────────────────────

    def show_sentence(self, text: str, confidence: float) -> None:
        """
        Update the main sentence display with newly reconstructed text.

        Thread-safe — schedules the update on the Tkinter main loop.

        Args:
            text: The reconstructed medical sentence.
            confidence: Confidence score in [0, 1] for the colour badge.
        """
        def _update() -> None:
            if self._sentence_var:
                self._sentence_var.set(text)
            if self._confidence_var:
                pct = int(confidence * 100)
                self._confidence_var.set(f"Confidence: {pct}%")
            if self._confidence_label:
                colour = _confidence_colour(confidence)
                self._confidence_label.configure(foreground=colour)

        self._root.after(0, _update)

    def show_status(self, message: str, colour: str = _TEXT_SECONDARY) -> None:
        """
        Update the status bar with a short status message.

        Thread-safe.

        Args:
            message: Status text (e.g. "Selecting body part…").
            colour: Hex colour for the message (optional, defaults to muted).
        """
        def _update() -> None:
            if self._status_var:
                self._status_var.set(message)

        self._root.after(0, _update)

    def show_latency(self, latency_ms: float, truncated: bool = False) -> None:
        """
        Update the latency indicator with the last inference time.

        Colours the indicator amber if latency exceeds 2000ms, red if truncated.

        Args:
            latency_ms: Inference round-trip time in milliseconds.
            truncated: True if generation was cut short by the budget.
        """
        def _update() -> None:
            if not self._latency_var:
                return
            label = f"{latency_ms:.0f}ms"
            if truncated:
                label += " ⚠ TRUNCATED"
            self._latency_var.set(label)

        self._root.after(0, _update)

    def show_emergency(self, message: str) -> None:
        """
        Display the full-screen emergency overlay.

        Raises the overlay frame to cover all other widgets and displays
        the emergency message prominently. Dismisses after 10 seconds.

        Args:
            message: The emergency message to display.
        """
        def _show() -> None:
            if not self._emergency_overlay:
                logger.warning("Emergency overlay not registered")
                return
            self._emergency_overlay.lift()
            self._emergency_overlay.place(relx=0, rely=0, relwidth=1, relheight=1)

        def _hide() -> None:
            if self._emergency_overlay:
                self._emergency_overlay.place_forget()

        self._root.after(0, _show)
        self._root.after(10_000, _hide)

    def hide_emergency(self) -> None:
        """Immediately hide the emergency overlay."""
        def _update() -> None:
            if self._emergency_overlay:
                self._emergency_overlay.place_forget()

        self._root.after(0, _update)

    def clear_sentence(self) -> None:
        """Clear the sentence display (called on intent reset)."""
        def _update() -> None:
            if self._sentence_var:
                self._sentence_var.set("")
            if self._confidence_var:
                self._confidence_var.set("")

        self._root.after(0, _update)
