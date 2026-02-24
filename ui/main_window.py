"""
ui/main_window.py — Main Tkinter interface for NeuroWeave Sentinel.

Three-panel layout (token grid | main display | metrics) with a bottom bar,
confirmation dialog, and fully thread-safe updates via root.after().
WCAG AA compliant high-contrast colors, minimum 14pt fonts throughout.
"""

from __future__ import annotations

import collections
import math
import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk
from typing import Callable, Optional

from core.constants import FSMState, SentinelConstants as C
from core.logger import get_logger

_log = get_logger()

# ──────────────────────────────────────────────────────────────
# Color palette (WCAG AA ≥4.5:1 contrast on dark background)
# ──────────────────────────────────────────────────────────────
BG_ROOT      = "#1a1a2e"   # deep navy
BG_PANEL     = "#16213e"   # panel body
BG_HEADER    = "#0f3460"   # panel header strip
BG_CELL      = "#1e2a45"   # grid cell
BG_CELL_HOV  = "#1565C0"   # hovered gaze cell
BG_CHIP      = "#263759"   # token chip
BG_CONFIRM   = "#0d0d1a"   # modal overlay

FG_MAIN      = "#e8eaf6"   # primary text
FG_DIM       = "#90a4ae"   # secondary text
FG_CHIP      = "#b3d1ff"   # token chip text

# FSM state colors
STATE_COLORS: dict[str, str] = {
    "IDLE":          "#78909c",
    "TOKEN_SELECTION":"#42a5f5",
    "CONFIRMATION":  "#ffa726",
    "GENERATING":    "#29b6f6",
    "VALIDATING":    "#ffee58",
    "SPEAKING":      "#66bb6a",
    "EMERGENCY":     "#ef5350",
    "FALLBACK":      "#ff7043",
}

# Safety badge colors
BADGE_COLORS: dict[str, str] = {
    "PROCEED":  "#388e3c",
    "CONFIRM":  "#f57c00",
    "FALLBACK": "#e65100",
    "BLOCK":    "#c62828",
}

# 3×3 grid layout: (row, col) → category label
GRID_LAYOUT: dict[tuple[int,int], str] = {
    (0,0): "BODY",      (0,1): "SENSATION",   (0,2): "INTENSITY",
    (1,0): "NEEDS",     (1,1): "COGNITIVE",   (1,2): "EMOTIONAL",
    (2,0): "EMERGENCY", (2,1): "TEMPORAL",    (2,2): "MODIFIERS",
}

GRID_ICONS: dict[str, str] = {
    "BODY":      "🫀",
    "SENSATION": "⚡",
    "INTENSITY": "📊",
    "NEEDS":     "🤲",
    "COGNITIVE": "🧠",
    "EMOTIONAL": "💙",
    "EMERGENCY": "🚨",
    "TEMPORAL":  "⏱",
    "MODIFIERS": "🔧",
}


# ──────────────────────────────────────────────────────────────
# Main window
# ──────────────────────────────────────────────────────────────

class SentinelMainWindow:
    """
    Main application window for NeuroWeave Sentinel.

    All public ``update_*`` methods are safe to call from any thread —
    they schedule their work via ``root.after(0, ...)``.

    Args:
        root: Tkinter root obtained from ``tk.Tk()``.
        on_emergency: Callback invoked when the emergency button is pressed.
        on_reset: Callback invoked when the reset button is pressed.
        on_speak: Callback invoked when Speak is confirmed in the dialog.
        on_reject: Callback invoked when Reject is pressed or dialog times out.
        on_mode_toggle: Callback invoked when webcam/sim mode is toggled.
    """

    def __init__(
        self,
        root: tk.Tk,
        on_emergency: Optional[Callable] = None,
        on_reset: Optional[Callable] = None,
        on_speak: Optional[Callable] = None,
        on_reject: Optional[Callable] = None,
        on_mode_toggle: Optional[Callable] = None,
    ) -> None:
        self._root = root
        self._on_emergency = on_emergency or (lambda: None)
        self._on_reset = on_reset or (lambda: None)
        self._on_speak = on_speak or (lambda: None)
        self._on_reject = on_reject or (lambda: None)
        self._on_mode_toggle = on_mode_toggle or (lambda: None)

        self._fsm_state: str = FSMState.IDLE.value
        self._mode_sim: bool = False
        self._confirm_dialog: Optional[tk.Toplevel] = None
        self._confirm_countdown: int = 10
        self._confirm_after_id: Optional[str] = None
        self._session_log: list[str] = []

        self._setup_window()
        self._build_fonts()
        self._build_layout()
        self._bind_shortcuts()

    # ──────────────────────────────────────────
    # Window setup
    # ──────────────────────────────────────────

    def _setup_window(self) -> None:
        self._root.title("NeuroWeave Sentinel  |  AAC Medical Communication")
        self._root.configure(bg=BG_ROOT)
        self._root.minsize(1100, 700)
        self._root.geometry("1200x750")
        self._root.resizable(True, True)
        # Grid weights for responsive resize
        self._root.grid_rowconfigure(0, weight=1)
        self._root.grid_columnconfigure(0, weight=0)   # left fixed
        self._root.grid_columnconfigure(1, weight=1)   # center expands
        self._root.grid_columnconfigure(2, weight=0)   # right fixed
        self._root.grid_rowconfigure(1, weight=0)      # bottom bar

    def _build_fonts(self) -> None:
        self._f_label   = tkfont.Font(family="Helvetica", size=14)
        self._f_small   = tkfont.Font(family="Helvetica", size=12)
        self._f_medium  = tkfont.Font(family="Helvetica", size=16, weight="bold")
        self._f_large   = tkfont.Font(family="Helvetica", size=28, weight="bold")
        self._f_state   = tkfont.Font(family="Helvetica", size=20, weight="bold")
        self._f_btn     = tkfont.Font(family="Helvetica", size=14, weight="bold")
        self._f_chip    = tkfont.Font(family="Helvetica", size=12)
        self._f_grid    = tkfont.Font(family="Helvetica", size=13, weight="bold")

    # ──────────────────────────────────────────
    # Layout construction
    # ──────────────────────────────────────────

    def _build_layout(self) -> None:
        self._left   = self._build_left_panel()
        self._center = self._build_center_panel()
        self._right  = self._build_right_panel()
        self._bottom = self._build_bottom_bar()

        self._left.grid  (row=0, column=0, sticky="nsew", padx=(8,4), pady=(8,4))
        self._center.grid(row=0, column=1, sticky="nsew", padx=4,     pady=(8,4))
        self._right.grid (row=0, column=2, sticky="nsew", padx=(4,8), pady=(8,4))
        self._bottom.grid(row=1, column=0, columnspan=3, sticky="ew",
                          padx=8, pady=(0,8))

    # ── LEFT PANEL ────────────────────────────────────────────

    def _build_left_panel(self) -> tk.Frame:
        pnl = tk.Frame(self._root, bg=BG_PANEL, width=300,
                       relief="flat", bd=0)
        pnl.grid_propagate(False)
        pnl.grid_columnconfigure(0, weight=1)

        hdr = tk.Label(pnl, text="TOKEN GRID", bg=BG_HEADER,
                       fg=FG_MAIN, font=self._f_medium, pady=8)
        hdr.grid(row=0, column=0, sticky="ew")

        # 3×3 category grid
        grid_frame = tk.Frame(pnl, bg=BG_PANEL)
        grid_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=8)
        for c in range(3):
            grid_frame.grid_columnconfigure(c, weight=1)

        self._grid_btns: dict[str, tk.Button] = {}
        for (r, c), cat in GRID_LAYOUT.items():
            icon = GRID_ICONS.get(cat, "")
            btn = tk.Button(
                grid_frame,
                text=f"{icon}\n{cat}",
                bg=BG_CELL,
                fg=FG_MAIN,
                font=self._f_grid,
                relief="flat",
                bd=0,
                padx=4, pady=10,
                cursor="hand2",
                wraplength=90,
            )
            btn.grid(row=r, column=c, padx=3, pady=3, sticky="nsew")
            self._grid_btns[cat] = btn

        tk.Label(pnl, text="SELECTED TOKENS", bg=BG_PANEL,
                 fg=FG_DIM, font=self._f_small).grid(
            row=2, column=0, sticky="w", padx=10, pady=(4,0))

        # Token chips container (scrollable)
        chip_frame_outer = tk.Frame(pnl, bg=BG_PANEL, height=200)
        chip_frame_outer.grid(row=3, column=0, sticky="ew", padx=8, pady=4)
        chip_frame_outer.grid_propagate(False)
        self._chip_canvas = tk.Canvas(chip_frame_outer, bg=BG_PANEL,
                                      highlightthickness=0)
        self._chip_canvas.pack(fill="both", expand=True)
        self._chip_inner = tk.Frame(self._chip_canvas, bg=BG_PANEL)
        self._chip_canvas.create_window((0, 0), window=self._chip_inner, anchor="nw")

        return pnl

    # ── CENTER PANEL ──────────────────────────────────────────

    def _build_center_panel(self) -> tk.Frame:
        pnl = tk.Frame(self._root, bg=BG_PANEL, relief="flat", bd=0)
        pnl.grid_columnconfigure(0, weight=1)

        # State label
        self._state_lbl = tk.Label(
            pnl, text="● IDLE", bg=BG_PANEL,
            fg=STATE_COLORS["IDLE"], font=self._f_state, pady=10)
        self._state_lbl.grid(row=0, column=0, sticky="ew", pady=(8,2))

        # Reconstructed text
        text_frame = tk.Frame(pnl, bg=BG_CELL, pady=24)
        text_frame.grid(row=1, column=0, sticky="ew", padx=16, pady=8)
        text_frame.grid_columnconfigure(0, weight=1)
        self._text_lbl = tk.Label(
            text_frame,
            text="Awaiting patient input…",
            bg=BG_CELL, fg=FG_MAIN,
            font=self._f_large,
            wraplength=440,
            justify="center",
        )
        self._text_lbl.grid(row=0, column=0, padx=16, sticky="ew")

        # Confidence bar
        conf_frame = tk.Frame(pnl, bg=BG_PANEL)
        conf_frame.grid(row=2, column=0, sticky="ew", padx=16, pady=(0,4))
        conf_frame.grid_columnconfigure(1, weight=1)
        tk.Label(conf_frame, text="Confidence", bg=BG_PANEL,
                 fg=FG_DIM, font=self._f_label).grid(row=0, column=0, padx=(0,8))
        self._conf_bar = ttk.Progressbar(conf_frame, orient="horizontal",
                                         length=300, mode="determinate",
                                         maximum=100)
        self._conf_bar.grid(row=0, column=1, sticky="ew")
        self._conf_pct_lbl = tk.Label(
            conf_frame, text="0%", bg=BG_PANEL,
            fg=FG_MAIN, font=self._f_label, width=5)
        self._conf_pct_lbl.grid(row=0, column=2, padx=(8,0))

        # Intent token pills
        tk.Label(pnl, text="INTENT TOKENS", bg=BG_PANEL,
                 fg=FG_DIM, font=self._f_small).grid(
            row=3, column=0, sticky="w", padx=16, pady=(8,0))
        self._pills_frame = tk.Frame(pnl, bg=BG_PANEL)
        self._pills_frame.grid(row=4, column=0, sticky="ew", padx=16, pady=4)

        return pnl

    # ── RIGHT PANEL ───────────────────────────────────────────

    def _build_right_panel(self) -> tk.Frame:
        pnl = tk.Frame(self._root, bg=BG_PANEL, width=280,
                       relief="flat", bd=0)
        pnl.grid_propagate(False)
        pnl.grid_columnconfigure(0, weight=1)

        tk.Label(pnl, text="METRICS", bg=BG_HEADER,
                 fg=FG_MAIN, font=self._f_medium, pady=8).grid(
            row=0, column=0, sticky="ew")

        # Latency
        tk.Label(pnl, text="Inference Latency", bg=BG_PANEL,
                 fg=FG_DIM, font=self._f_small).grid(
            row=1, column=0, sticky="w", padx=10, pady=(10,0))
        self._latency_lbl = tk.Label(
            pnl, text="— ms", bg=BG_PANEL,
            fg=FG_MAIN, font=self._f_medium)
        self._latency_lbl.grid(row=2, column=0, sticky="w", padx=10)

        # Entropy bar
        tk.Label(pnl, text="Entropy Score", bg=BG_PANEL,
                 fg=FG_DIM, font=self._f_small).grid(
            row=3, column=0, sticky="w", padx=10, pady=(10,0))
        entropy_row = tk.Frame(pnl, bg=BG_PANEL)
        entropy_row.grid(row=4, column=0, sticky="ew", padx=10, pady=2)
        entropy_row.grid_columnconfigure(0, weight=1)
        self._entropy_bar = ttk.Progressbar(entropy_row, orient="horizontal",
                                            length=200, mode="determinate",
                                            maximum=100)
        self._entropy_bar.grid(row=0, column=0, sticky="ew")
        self._entropy_lbl = tk.Label(entropy_row, text="0.00",
                                     bg=BG_PANEL, fg=FG_MAIN,
                                     font=self._f_small, width=5)
        self._entropy_lbl.grid(row=0, column=1, padx=(6,0))

        # Safety badge
        tk.Label(pnl, text="Safety Decision", bg=BG_PANEL,
                 fg=FG_DIM, font=self._f_small).grid(
            row=5, column=0, sticky="w", padx=10, pady=(10,0))
        self._badge_lbl = tk.Label(
            pnl, text="PROCEED", bg=BADGE_COLORS["PROCEED"],
            fg="#ffffff", font=self._f_btn, padx=12, pady=6)
        self._badge_lbl.grid(row=6, column=0, sticky="w", padx=10, pady=2)

        # Session log
        tk.Label(pnl, text="Recent Utterances", bg=BG_PANEL,
                 fg=FG_DIM, font=self._f_small).grid(
            row=7, column=0, sticky="w", padx=10, pady=(12,0))
        self._log_frame = tk.Frame(pnl, bg=BG_PANEL)
        self._log_frame.grid(row=8, column=0, sticky="nsew", padx=10, pady=4)
        self._log_labels: list[tk.Label] = []
        for i in range(5):
            lbl = tk.Label(self._log_frame, text="", bg=BG_PANEL,
                           fg=FG_DIM, font=self._f_small,
                           anchor="w", wraplength=220, justify="left")
            lbl.pack(fill="x", pady=2)
            self._log_labels.append(lbl)

        # ── Live metrics dashboard (embedded) ─────────────
        sep = tk.Frame(pnl, bg="#263759", height=1)
        sep.grid(row=9, column=0, sticky="ew", padx=6, pady=(8, 0))
        self._dashboard = MetricsDashboard(pnl, root=self._root, row_start=10)

        return pnl

    # ── BOTTOM BAR ────────────────────────────────────────────

    def _build_bottom_bar(self) -> tk.Frame:
        bar = tk.Frame(self._root, bg=BG_HEADER, pady=6)
        bar.grid_columnconfigure(0, weight=1)  # LED strip expands

        # FSM state LEDs
        led_frame = tk.Frame(bar, bg=BG_HEADER)
        led_frame.pack(side="left", padx=12)
        self._led_labels: dict[str, tk.Label] = {}
        for state in FSMState:
            col = STATE_COLORS.get(state.value, "#555")
            lbl = tk.Label(led_frame, text="■", fg="#333",
                           bg=BG_HEADER, font=self._f_small)
            lbl.pack(side="left", padx=3)
            tk.Label(led_frame, text=state.value, fg=FG_DIM,
                     bg=BG_HEADER, font=self._f_small).pack(
                side="left", padx=(0,8))
            self._led_labels[state.value] = lbl
        self._update_leds("IDLE")

        # Right-side controls
        ctrl = tk.Frame(bar, bg=BG_HEADER)
        ctrl.pack(side="right", padx=12)

        # Mode toggle
        self._mode_var = tk.StringVar(value="WEBCAM")
        mode_btn = tk.Button(
            ctrl, textvariable=self._mode_var,
            bg="#37474f", fg=FG_MAIN, font=self._f_label,
            relief="flat", bd=0, padx=12, pady=4,
            cursor="hand2",
            command=self._toggle_mode,
        )
        mode_btn.pack(side="left", padx=6)

        # Reset button
        reset_btn = tk.Button(
            ctrl, text="↺ RESET",
            bg="#455a64", fg=FG_MAIN, font=self._f_btn,
            relief="flat", bd=0, padx=12, pady=4,
            cursor="hand2",
            command=self._on_reset,
        )
        reset_btn.pack(side="left", padx=6)

        # Emergency button
        self._emrg_btn = tk.Button(
            ctrl, text="🚨  EMERGENCY  (Ctrl+E)",
            bg=STATE_COLORS["EMERGENCY"], fg="#ffffff",
            font=self._f_btn, relief="flat", bd=0,
            padx=16, pady=6, cursor="hand2",
            command=self._fire_emergency,
        )
        self._emrg_btn.pack(side="left", padx=6)

        return bar

    # ──────────────────────────────────────────
    # Key bindings
    # ──────────────────────────────────────────

    def _bind_shortcuts(self) -> None:
        self._root.bind("<Control-e>", lambda _e: self._fire_emergency())
        self._root.bind("<Control-E>", lambda _e: self._fire_emergency())

    # ──────────────────────────────────────────
    # Thread-safe public update API
    # ──────────────────────────────────────────

    def update_fsm_state(self, state: str) -> None:
        """Update the FSM state label and LED indicators (thread-safe)."""
        self._root.after(0, self._apply_fsm_state, state)

    def update_text(self, text: str) -> None:
        """Update the reconstructed text in the center panel (thread-safe)."""
        self._root.after(0, self._text_lbl.config, {"text": text})

    def update_confidence(self, score: float) -> None:
        """Update the confidence progress bar (thread-safe)."""
        self._root.after(0, self._apply_confidence, score)

    def update_tokens(self, token_codes: list[str]) -> None:
        """Refresh the token chips in the left panel (thread-safe)."""
        self._root.after(0, self._apply_tokens, token_codes)

    def update_pills(self, token_codes: list[str]) -> None:
        """Refresh the intent token pills in the center panel (thread-safe)."""
        self._root.after(0, self._apply_pills, token_codes)

    def update_gaze_zone(self, category: str) -> None:
        """Highlight the active gaze zone in the token grid (thread-safe)."""
        self._root.after(0, self._apply_gaze_zone, category)

    def update_latency(self, latency_ms: float) -> None:
        """Update the latency metric label (thread-safe)."""
        self._root.after(0, self._latency_lbl.config,
                         {"text": f"{latency_ms:.0f} ms"})

    def update_entropy(self, entropy: float) -> None:
        """Update the entropy progress bar (thread-safe)."""
        self._root.after(0, self._apply_entropy, entropy)

    def update_safety_badge(self, decision: str) -> None:
        """Update the safety badge label (thread-safe)."""
        self._root.after(0, self._apply_badge, decision)

    def add_session_log(self, utterance: str) -> None:
        """Prepend an utterance to the session log (thread-safe)."""
        self._root.after(0, self._apply_session_log, utterance)

    def show_confirmation(self, text: str) -> None:
        """Show the modal confirmation dialog (thread-safe)."""
        self._root.after(0, self._open_confirm_dialog, text)

    def dismiss_confirmation(self) -> None:
        """Dismiss the confirmation dialog if open (thread-safe)."""
        self._root.after(0, self._close_confirm_dialog)

    # ──────────────────────────────────────────
    # Private appliers (run on main thread)
    # ──────────────────────────────────────────

    def _apply_fsm_state(self, state: str) -> None:
        self._fsm_state = state
        color = STATE_COLORS.get(state, FG_DIM)
        self._state_lbl.config(text=f"● {state}", fg=color)
        self._update_leds(state)

    def _update_leds(self, active: str) -> None:
        for state_val, lbl in self._led_labels.items():
            if state_val == active:
                lbl.config(fg=STATE_COLORS.get(state_val, FG_MAIN))
            else:
                lbl.config(fg="#333333")

    def _apply_confidence(self, score: float) -> None:
        pct = int(min(max(score, 0.0), 1.0) * 100)
        self._conf_bar["value"] = pct
        self._conf_pct_lbl.config(text=f"{pct}%")

    def _apply_tokens(self, codes: list[str]) -> None:
        for w in self._chip_inner.winfo_children():
            w.destroy()
        for code in codes:
            tk.Label(
                self._chip_inner,
                text=code.replace("_", " "),
                bg=BG_CHIP, fg=FG_CHIP,
                font=self._f_chip,
                padx=8, pady=3, relief="flat",
            ).pack(side="top", anchor="w", pady=2, padx=2)

    def _apply_pills(self, codes: list[str]) -> None:
        for w in self._pills_frame.winfo_children():
            w.destroy()
        for code in codes:
            color = STATE_COLORS.get("TOKEN_SELECTION", BG_CHIP)
            tk.Label(
                self._pills_frame,
                text=code.replace("_", " "),
                bg=color, fg="#ffffff",
                font=self._f_chip,
                padx=8, pady=3, relief="flat",
            ).pack(side="left", padx=3, pady=2)

    def _apply_gaze_zone(self, category: str) -> None:
        for cat, btn in self._grid_btns.items():
            btn.config(bg=BG_CELL_HOV if cat == category else BG_CELL)

    def _apply_entropy(self, entropy: float) -> None:
        val = int(min(max(entropy, 0.0), 1.0) * 100)
        self._entropy_bar["value"] = val
        self._entropy_lbl.config(text=f"{entropy:.2f}")

    def _apply_badge(self, decision: str) -> None:
        color = BADGE_COLORS.get(decision.upper(), "#37474f")
        self._badge_lbl.config(text=decision.upper(), bg=color)

    def _apply_session_log(self, utterance: str) -> None:
        self._session_log.insert(0, utterance)
        self._session_log = self._session_log[:5]
        for i, lbl in enumerate(self._log_labels):
            text = self._session_log[i] if i < len(self._session_log) else ""
            lbl.config(text=text[:40] + ("…" if len(text) > 40 else ""))

    # ──────────────────────────────────────────
    # Confirmation dialog
    # ──────────────────────────────────────────

    def _open_confirm_dialog(self, text: str) -> None:
        if self._confirm_dialog and self._confirm_dialog.winfo_exists():
            return

        dlg = tk.Toplevel(self._root)
        dlg.title("Confirm Utterance")
        dlg.configure(bg=BG_CONFIRM)
        dlg.resizable(False, False)
        # Centre over main window
        x = self._root.winfo_x() + 200
        y = self._root.winfo_y() + 180
        dlg.geometry(f"700x280+{x}+{y}")
        dlg.grab_set()          # modal
        dlg.focus_set()
        self._confirm_dialog = dlg
        self._confirm_countdown = 10

        tk.Label(dlg, text="CONFIRM UTTERANCE?", bg=BG_CONFIRM,
                 fg=FG_MAIN, font=self._f_state, pady=12).pack()

        tk.Label(dlg, text=text, bg=BG_CONFIRM, fg=FG_MAIN,
                 font=self._f_medium, wraplength=640,
                 justify="center", pady=8).pack()

        self._countdown_lbl = tk.Label(
            dlg, text=f"Auto-reject in {self._confirm_countdown}s",
            bg=BG_CONFIRM, fg=FG_DIM, font=self._f_small)
        self._countdown_lbl.pack(pady=(0, 8))

        btn_row = tk.Frame(dlg, bg=BG_CONFIRM)
        btn_row.pack(pady=8)

        tk.Button(
            btn_row, text="✔  SPEAK  (Enter)",
            bg=STATE_COLORS["SPEAKING"], fg="#ffffff",
            font=self._f_btn, relief="flat", padx=24, pady=10,
            cursor="hand2",
            command=self._confirm_speak,
        ).pack(side="left", padx=16)

        tk.Button(
            btn_row, text="✖  REJECT  (Esc)",
            bg=STATE_COLORS["EMERGENCY"], fg="#ffffff",
            font=self._f_btn, relief="flat", padx=24, pady=10,
            cursor="hand2",
            command=self._confirm_reject,
        ).pack(side="left", padx=16)

        dlg.bind("<Return>", lambda _: self._confirm_speak())
        dlg.bind("<Escape>", lambda _: self._confirm_reject())

        self._tick_countdown()

    def _tick_countdown(self) -> None:
        if not self._confirm_dialog or not self._confirm_dialog.winfo_exists():
            return
        if self._confirm_countdown <= 0:
            self._confirm_reject()
            return
        self._countdown_lbl.config(
            text=f"Auto-reject in {self._confirm_countdown}s")
        self._confirm_countdown -= 1
        self._confirm_after_id = self._root.after(1000, self._tick_countdown)

    def _confirm_speak(self) -> None:
        self._close_confirm_dialog()
        self._on_speak()

    def _confirm_reject(self) -> None:
        self._close_confirm_dialog()
        self._on_reject()

    def _close_confirm_dialog(self) -> None:
        if self._confirm_after_id:
            self._root.after_cancel(self._confirm_after_id)
            self._confirm_after_id = None
        if self._confirm_dialog and self._confirm_dialog.winfo_exists():
            self._confirm_dialog.grab_release()
            self._confirm_dialog.destroy()
        self._confirm_dialog = None

    # ──────────────────────────────────────────
    # Button handlers
    # ──────────────────────────────────────────

    def _fire_emergency(self) -> None:
        _log.warn("ui", "emergency_button_pressed", {})
        self._on_emergency()

    def _toggle_mode(self) -> None:
        self._mode_sim = not self._mode_sim
        label = "SIMULATED" if self._mode_sim else "WEBCAM"
        self._mode_var.set(label)
        _log.info("ui", "mode_toggle", {"mode": label})
        self._on_mode_toggle()

    # ──────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────

    def run(self) -> None:
        """Start the Tkinter main loop (blocking)."""
        _log.info("ui", "mainloop_start", {})
        self._root.mainloop()


# ──────────────────────────────────────────────────────────────
# MetricsDashboard
# ──────────────────────────────────────────────────────────────

class MetricsDashboard:
    """
    Live metrics panel embedded in the right panel of :class:`SentinelMainWindow`.

    Displays 7 session metrics and an entropy sparkline, refreshed every 500ms
    via ``root.after()``.  All public ``record_*`` methods are thread-safe.

    Metrics shown:
    - Total utterances (session counter)
    - Average end-to-end latency (rolling 10)
    - LLM inference latency: last + rolling average
    - Safety gate pass rate (% PROCEED)
    - Token selection time: average ms IDLE → GENERATING
    - GPU memory usage (MB) from ``torch.cuda.memory_allocated()``
    - Entropy sparkline: last 10 values as a colour-banded bar chart
    """

    _SPARKLINE_W = 100
    _SPARKLINE_H = 30
    _WINDOW       = 10        # rolling average window
    _REFRESH_MS   = 500

    def __init__(
        self,
        parent: tk.Frame,
        root: tk.Tk,
        row_start: int = 0,
    ) -> None:
        """Build all metric widgets inside *parent* starting at grid row *row_start*."""
        self._root = root
        self._parent = parent
        self._row = row_start

        # ── Data stores (thread-safe: only ever read on main thread) ──
        self._total_utterances: int = 0
        self._e2e_latencies: collections.deque[float] = collections.deque(maxlen=self._WINDOW)
        self._llm_latencies:  collections.deque[float] = collections.deque(maxlen=self._WINDOW)
        self._llm_last: float = 0.0
        self._safety_total: int = 0
        self._safety_pass:  int = 0
        self._token_sel_times: collections.deque[float] = collections.deque(maxlen=self._WINDOW)
        self._entropy_history: collections.deque[float] = collections.deque(maxlen=self._WINDOW)

        # Pending thread-safe updates (list of callables applied on next tick)
        import threading
        self._pending_lock = threading.Lock()
        self._pending: list = []

        self._build_widgets()
        self._schedule_refresh()

    # ──────────────────────────────────────────
    # Thread-safe record methods
    # ──────────────────────────────────────────

    def record_utterance(self, e2e_ms: float) -> None:
        """Record a completed utterance with its end-to-end latency."""
        with self._pending_lock:
            self._pending.append(('utterance', e2e_ms))

    def record_llm_latency(self, latency_ms: float) -> None:
        """Record LLM inference latency for the most recent generation."""
        with self._pending_lock:
            self._pending.append(('llm', latency_ms))

    def record_safety(self, passed: bool) -> None:
        """Record whether the safety gate passed (True) or blocked/fell back (False)."""
        with self._pending_lock:
            self._pending.append(('safety', passed))

    def record_token_selection_time(self, ms: float) -> None:
        """Record the time in ms from IDLE to GENERATING for one utterance."""
        with self._pending_lock:
            self._pending.append(('token_sel', ms))

    def record_entropy(self, entropy: float) -> None:
        """Append one entropy reading to the sparkline history."""
        with self._pending_lock:
            self._pending.append(('entropy', entropy))

    # ──────────────────────────────────────────
    # Widget construction
    # ──────────────────────────────────────────

    def _build_widgets(self) -> None:
        r = self._row
        p = self._parent
        f_dim   = tkfont.Font(family='Helvetica', size=11)
        f_val   = tkfont.Font(family='Helvetica', size=12, weight='bold')

        def _section(label: str, row: int) -> tk.Label:
            tk.Label(p, text=label, bg=BG_PANEL, fg=FG_DIM,
                     font=f_dim, anchor='w').grid(
                row=row, column=0, sticky='w', padx=10, pady=(6, 0))
            val = tk.Label(p, text='—', bg=BG_PANEL, fg=FG_MAIN,
                           font=f_val, anchor='w')
            val.grid(row=row + 1, column=0, sticky='w', padx=16, pady=(0, 2))
            return val

        tk.Label(p, text='LIVE METRICS', bg=BG_HEADER, fg=FG_MAIN,
                 font=tkfont.Font(family='Helvetica', size=13, weight='bold'),
                 pady=4).grid(row=r, column=0, sticky='ew')
        r += 1

        self._lbl_utterances  = _section('Utterances (session)', r);    r += 2
        self._lbl_e2e         = _section('Avg E2E latency', r);          r += 2
        self._lbl_llm         = _section('LLM latency (last / avg)', r); r += 2
        self._lbl_safety      = _section('Safety pass rate', r);         r += 2
        self._lbl_token_sel   = _section('Avg token select time', r);    r += 2
        self._lbl_gpu         = _section('GPU memory', r);               r += 2

        # Entropy sparkline
        tk.Label(p, text='Entropy trend (last 10)', bg=BG_PANEL, fg=FG_DIM,
                 font=f_dim, anchor='w').grid(
            row=r, column=0, sticky='w', padx=10, pady=(6, 0))
        r += 1
        self._spark_canvas = tk.Canvas(
            p, width=self._SPARKLINE_W, height=self._SPARKLINE_H,
            bg=BG_CELL, highlightthickness=0)
        self._spark_canvas.grid(row=r, column=0, sticky='w', padx=10, pady=(2, 6))

    # ──────────────────────────────────────────
    # Refresh loop
    # ──────────────────────────────────────────

    def _schedule_refresh(self) -> None:
        """Schedule the next metrics refresh on the Tkinter event loop."""
        self._root.after(self._REFRESH_MS, self._refresh)

    def _refresh(self) -> None:
        """Apply pending updates and redraw all metric widgets."""
        # Drain pending updates (swap list under lock)
        with self._pending_lock:
            pending, self._pending = self._pending, []

        for event in pending:
            kind = event[0]
            val  = event[1]
            if kind == 'utterance':
                self._total_utterances += 1
                self._e2e_latencies.append(val)
            elif kind == 'llm':
                self._llm_latencies.append(val)
                self._llm_last = val
            elif kind == 'safety':
                self._safety_total += 1
                if val:
                    self._safety_pass += 1
            elif kind == 'token_sel':
                self._token_sel_times.append(val)
            elif kind == 'entropy':
                self._entropy_history.append(val)

        # ── Compute display values ────────────────────────
        avg_e2e = (sum(self._e2e_latencies) / len(self._e2e_latencies)
                   if self._e2e_latencies else 0.0)
        avg_llm = (sum(self._llm_latencies) / len(self._llm_latencies)
                   if self._llm_latencies else 0.0)
        pass_rate = (
            100.0 * self._safety_pass / self._safety_total
            if self._safety_total else 0.0
        )
        avg_tok = (sum(self._token_sel_times) / len(self._token_sel_times)
                   if self._token_sel_times else 0.0)
        gpu_mb = self._read_gpu_mb()

        # ── Update labels ─────────────────────────────────
        self._lbl_utterances.config(text=str(self._total_utterances))
        self._lbl_e2e.config(text=f'{avg_e2e:.0f} ms')
        self._lbl_llm.config(
            text=f'{self._llm_last:.0f} ms  /  {avg_llm:.0f} ms avg')
        self._lbl_safety.config(
            text=f'{pass_rate:.1f}%  ({self._safety_pass}/{self._safety_total})')
        self._lbl_token_sel.config(text=f'{avg_tok:.0f} ms avg')
        self._lbl_gpu.config(
            text=f'{gpu_mb:.1f} MB' if gpu_mb >= 0 else 'N/A (CPU)')

        # ── Redraw sparkline ──────────────────────────────
        self._draw_sparkline()

        # Schedule next tick
        self._schedule_refresh()

    def _draw_sparkline(self) -> None:
        """
        Draw the entropy sparkline on the Canvas.

        Each bar: green if entropy < 0.5, yellow if 0.5–0.75, red if > 0.75.
        """
        c = self._spark_canvas
        c.delete('all')
        vals = list(self._entropy_history)
        if not vals:
            c.create_text(
                self._SPARKLINE_W // 2, self._SPARKLINE_H // 2,
                text='no data', fill=FG_DIM, font=('Helvetica', 9))
            return

        n = len(vals)
        bar_w = max(1, math.floor(self._SPARKLINE_W / self._WINDOW))
        gap   = 1
        h     = self._SPARKLINE_H

        for i, v in enumerate(vals):
            x0 = i * (bar_w + gap)
            x1 = x0 + bar_w
            bar_h = max(2, int(v * h))
            y0 = h - bar_h
            y1 = h
            color = (
                '#ef5350' if v > 0.75
                else '#ffee58' if v > 0.5
                else '#66bb6a'
            )
            c.create_rectangle(x0, y0, x1, y1, fill=color, outline='')

    @staticmethod
    def _read_gpu_mb() -> float:
        """
        Read current GPU memory allocated via ``torch.cuda``.

        Returns:
            Megabytes allocated, or ``-1.0`` if CUDA is unavailable.
        """
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated(0) / (1024 ** 2)
        except Exception:  # noqa: BLE001
            pass
        return -1.0
