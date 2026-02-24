"""
ui/widgets.py — Custom Tkinter widget components for NeuroWeave Sentinel.

Provides accessible, high-contrast widgets tuned for gaze-based navigation:
- GazeProgressRing: Circular dwell progress indicator
- SymbolCell: Symbol board button with hover and selection states
- ConfidenceBadge: Colour-coded confidence indicator label
- StatusBar: Bottom status bar with message history
"""

from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from typing import Callable, Optional

# ──────────────────────────────────────────────────────────────
# Colour palette (WCAG AAA dark theme)
# ──────────────────────────────────────────────────────────────
BG_MAIN = "#0d0d0d"
BG_PANEL = "#1a1a2e"
BG_CELL = "#16213e"
BG_CELL_HOVER = "#0f3460"
BG_CELL_SELECTED = "#533483"
ACCENT_BLUE = "#4a90e2"
ACCENT_GREEN = "#27ae60"
ACCENT_AMBER = "#f39c12"
ACCENT_RED = "#e74c3c"
TEXT_PRIMARY = "#ecf0f1"
TEXT_SECONDARY = "#95a5a6"


class GazeProgressRing(tk.Canvas):
    """
    Circular arc progress indicator showing dwell completion.

    Animates from 0° to 360° as the patient's gaze dwells on the parent
    widget. At full circle, a selection is triggered.

    Args:
        parent: Parent Tkinter widget.
        size: Diameter of the ring in pixels.
        ring_width: Thickness of the arc stroke.
        color: Arc colour.
        bg: Background colour.
    """

    def __init__(
        self,
        parent: tk.Widget,
        size: int = 48,
        ring_width: int = 5,
        color: str = ACCENT_BLUE,
        bg: str = BG_PANEL,
    ) -> None:
        """Create the canvas and draw initial empty ring."""
        super().__init__(
            parent, width=size, height=size,
            bg=bg, highlightthickness=0,
        )
        self._size = size
        self._ring_width = ring_width
        self._color = color
        self._progress: float = 0.0
        self._arc_id: Optional[int] = None
        self._draw(0.0)

    def set_progress(self, progress: float) -> None:
        """
        Update the ring fill to ``progress`` (0.0 = empty, 1.0 = full circle).

        Args:
            progress: Dwell completion ratio in [0.0, 1.0].
        """
        self._progress = max(0.0, min(1.0, progress))
        self._draw(self._progress)

    def reset(self) -> None:
        """Clear the ring back to 0% progress."""
        self._progress = 0.0
        self._draw(0.0)

    def _draw(self, progress: float) -> None:
        """
        Redraw the arc on the canvas.

        Args:
            progress: Completion ratio in [0, 1].
        """
        self.delete("all")
        pad = self._ring_width
        extent = progress * 359.9  # 360 causes arc to disappear
        if extent > 0:
            self.create_arc(
                pad, pad,
                self._size - pad, self._size - pad,
                start=90, extent=-extent,
                outline=self._color,
                width=self._ring_width,
                style=tk.ARC,
            )
        # Background ring
        self.create_arc(
            pad, pad,
            self._size - pad, self._size - pad,
            start=0, extent=359.9,
            outline=TEXT_SECONDARY,
            width=1,
            style=tk.ARC,
        )


class SymbolCell(tk.Frame):
    """
    A single symbol board cell combining a button, icon, and progress ring.

    Supports three visual states: normal, hover (gaze hovering), and
    selected (dwell or blink triggered).

    Args:
        parent: Parent Tkinter widget.
        label: Text to display on the cell.
        icon: Optional emoji prefix.
        on_click: Callback called when the cell is clicked.
        font_size: Font size for the label.
        width: Cell width in pixels.
        height: Cell height in pixels.
    """

    def __init__(
        self,
        parent: tk.Widget,
        label: str,
        icon: str = "",
        on_click: Optional[Callable[[], None]] = None,
        font_size: int = 14,
        width: int = 120,
        height: int = 80,
    ) -> None:
        """Build the cell frame with button and progress ring."""
        super().__init__(parent, bg=BG_CELL, relief=tk.FLAT, bd=2)

        self._on_click = on_click
        self._normal_bg = BG_CELL
        self._hover_bg = BG_CELL_HOVER
        self._selected_bg = BG_CELL_SELECTED

        display_text = f"{icon}  {label}" if icon else label
        cell_font = tkfont.Font(family="Arial", size=font_size, weight="bold")

        self._btn = tk.Label(
            self, text=display_text,
            bg=BG_CELL, fg=TEXT_PRIMARY,
            font=cell_font,
            width=width // 10, height=height // 24,
            cursor="hand2",
            wraplength=width - 8,
        )
        self._btn.pack(expand=True, fill=tk.BOTH, padx=4, pady=4)

        self._ring = GazeProgressRing(self, size=32, bg=BG_CELL)
        self._ring.place(relx=1.0, rely=0.0, anchor=tk.NE, x=-2, y=2)

        self._btn.bind("<Button-1>", lambda _: self._clicked())
        self.bind("<Button-1>", lambda _: self._clicked())

    def set_progress(self, progress: float) -> None:
        """
        Update the dwell progress ring.

        Args:
            progress: Completion ratio [0.0, 1.0].
        """
        self._ring.set_progress(progress)

    def set_hover(self, hovering: bool) -> None:
        """
        Toggle hover state (gaze is over this cell).

        Args:
            hovering: True if gaze is currently on this cell.
        """
        bg = self._hover_bg if hovering else self._normal_bg
        self._btn.configure(bg=bg)
        self.configure(bg=bg)
        self._ring.configure(bg=bg)

    def set_selected(self, selected: bool) -> None:
        """
        Toggle selected state (dwell or blink completed on this cell).

        Args:
            selected: True to show selected appearance.
        """
        bg = self._selected_bg if selected else self._normal_bg
        self._btn.configure(bg=bg)
        self.configure(bg=bg)
        self._ring.reset()

    def reset(self) -> None:
        """Return cell to normal unselected state."""
        self._btn.configure(bg=self._normal_bg)
        self.configure(bg=self._normal_bg)
        self._ring.reset()

    def _clicked(self) -> None:
        """Handle mouse click: set selected state and invoke callback."""
        self.set_selected(True)
        if self._on_click:
            self._on_click()


class ConfidenceBadge(tk.Label):
    """
    Colour-coded confidence label.

    Displays a confidence percentage with a traffic-light colour:
    green (≥75%), amber (45–74%), red (<45%).

    Args:
        parent: Parent Tkinter widget.
        font_size: Label font size.
    """

    def __init__(self, parent: tk.Widget, font_size: int = 13) -> None:
        """Create an initially empty confidence label."""
        super().__init__(
            parent, text="",
            bg=BG_PANEL, fg=ACCENT_GREEN,
            font=tkfont.Font(family="Arial", size=font_size),
        )

    def update(self, confidence: float) -> None:
        """
        Update the badge with a new confidence score.

        Args:
            confidence: Score in [0.0, 1.0].
        """
        pct = int(confidence * 100)
        if confidence >= 0.75:
            colour = ACCENT_GREEN
        elif confidence >= 0.45:
            colour = ACCENT_AMBER
        else:
            colour = ACCENT_RED
        self.configure(text=f"Confidence: {pct}%", fg=colour)

    def clear(self) -> None:
        """Clear the badge text."""
        self.configure(text="")


class StatusBar(tk.Frame):
    """
    Bottom status bar with a current message and last-message history.

    Args:
        parent: Parent Tkinter widget.
        height: Frame height in pixels.
    """

    def __init__(self, parent: tk.Widget, height: int = 32) -> None:
        """Build the status bar frame."""
        super().__init__(parent, bg=BG_PANEL, height=height)
        self.pack_propagate(False)

        self._msg_var = tk.StringVar(value="Ready")
        self._status_label = tk.Label(
            self, textvariable=self._msg_var,
            bg=BG_PANEL, fg=TEXT_SECONDARY,
            font=tkfont.Font(family="Arial", size=11),
        )
        self._status_label.pack(side=tk.LEFT, padx=12, pady=4)

    def set_message(self, message: str, colour: str = TEXT_SECONDARY) -> None:
        """
        Update the status bar message.

        Args:
            message: Status text to display.
            colour: Optional hex colour for the message.
        """
        self._msg_var.set(message)
        self._status_label.configure(fg=colour)
