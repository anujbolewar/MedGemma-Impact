"""
ui/app.py — NeuroWeave Sentinel main Tkinter application.

Full-screen accessible UI for ALS / locked-in patients.
Layout: camera preview (left) | symbol board (centre) | sentence output (right).
Emergency button always visible. Supports webcam and keyboard simulator modes.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import tkinter as tk
from tkinter import font as tkfont
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from sentinel.core.config import SentinelConfig, load_config
from sentinel.core.pipeline import (
    PipelineEvent,
    PipelineMode,
    PipelineState,
    SentinelPipeline,
)
from sentinel.gaze.blink_detector import DwellTracker
from sentinel.gaze.simulator import KeyboardSimulator
from sentinel.intent.symbol_board import Symbol
from sentinel.llm.engine import InferenceResult
from sentinel.output.display import DisplayManager

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Colour constants (dark, high-contrast, WCAG AAA)
# ──────────────────────────────────────────────
BG_MAIN = "#0d0d0d"
BG_PANEL = "#1a1a2e"
BG_BOARD_CELL = "#16213e"
BG_BOARD_CELL_HOVER = "#0f3460"
BG_BOARD_CELL_SELECTED = "#533483"
ACCENT_BLUE = "#4a90e2"
ACCENT_GREEN = "#27ae60"
ACCENT_AMBER = "#f39c12"
ACCENT_RED = "#e74c3c"
EMERGENCY_RED = "#c0392b"
TEXT_PRIMARY = "#ecf0f1"
TEXT_SECONDARY = "#95a5a6"
TEXT_EMERGENCY = "#ffffff"


def _setup_logging(level: str = "INFO") -> None:
    """
    Configure root logger for the application.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
    """
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


class SentinelApp(tk.Tk):
    """
    Main Tkinter application for NeuroWeave Sentinel.

    Orchestrates the full UI: camera preview, symbol board, sentence output
    panel, status bar, and emergency overlay. Provides both webcam and
    simulator input modes.

    Args:
        config: Validated :class:`SentinelConfig`.
        mode: Input mode (webcam or keyboard simulator).
    """

    def __init__(self, config: SentinelConfig, mode: PipelineMode) -> None:
        """Build and configure the application window."""
        super().__init__()

        self._cfg = config
        self._mode = mode
        self._pipeline: Optional[SentinelPipeline] = None

        # Font setup
        self._font_sentence = tkfont.Font(
            family=config.ui.font_family,
            size=config.ui.sentence_font_size,
            weight="bold",
        )
        self._font_board = tkfont.Font(
            family=config.ui.font_family,
            size=config.ui.board_font_size,
        )
        self._font_status = tkfont.Font(
            family=config.ui.font_family,
            size=config.ui.status_font_size,
        )
        self._font_emergency = tkfont.Font(
            family=config.ui.font_family,
            size=42,
            weight="bold",
        )

        # StringVars for data binding
        self._sentence_var = tk.StringVar(value="")
        self._confidence_var = tk.StringVar(value="")
        self._status_var = tk.StringVar(value="Ready — select a body part to begin")
        self._latency_var = tk.StringVar(value="")
        self._mode_var = tk.StringVar(
            value=f"Mode: {'Simulator' if mode == PipelineMode.SIMULATOR else 'Webcam'}"
        )

        # Board cell button references: {region_id → tk.Button}
        self._board_buttons: dict[str, tk.Button] = {}
        self._board_progress_vars: dict[str, tk.DoubleVar] = {}

        # Camera preview image reference (prevent GC)
        self._camera_photo: Optional[ImageTk.PhotoImage] = None

        # Display manager (set up after widget creation)
        self._display: Optional[DisplayManager] = None

        # Dwell tracker for UI progress rings
        self._dwell_tracker = DwellTracker(
            dwell_threshold_ms=config.gaze.dwell_threshold_ms,
            grid_cols=config.symbol_board.columns,
            grid_rows=config.symbol_board.rows,
        )

        self._build_window()
        self._build_layout()

        # Bind keyboard events
        self.bind("<KeyPress>", self._on_keypress)
        self.bind("<Escape>", lambda _: self._reset_intent())
        self.bind("<F1>", lambda _: self._trigger_emergency_ui())
        self.bind("<F2>", lambda _: self._toggle_mode())

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ──────────────────────────────────────────
    # Window & layout construction
    # ──────────────────────────────────────────

    def _build_window(self) -> None:
        """Configure root window properties."""
        self.title("NeuroWeave Sentinel — Assistive Communication")
        self.configure(bg=BG_MAIN)
        self.resizable(True, True)
        if self._cfg.ui.fullscreen:
            self.attributes("-fullscreen", True)
        else:
            self.geometry("1200x700")
        self.minsize(900, 600)

    def _build_layout(self) -> None:
        """Construct the three-panel layout and all child widgets."""
        # ── Top bar ────────────────────────────
        top_bar = tk.Frame(self, bg=BG_PANEL, height=50)
        top_bar.pack(fill=tk.X, side=tk.TOP)
        top_bar.pack_propagate(False)

        tk.Label(
            top_bar, text="⬛ NeuroWeave Sentinel",
            bg=BG_PANEL, fg=ACCENT_BLUE,
            font=tkfont.Font(family=self._cfg.ui.font_family, size=16, weight="bold"),
        ).pack(side=tk.LEFT, padx=16, pady=10)

        tk.Label(
            top_bar, textvariable=self._mode_var,
            bg=BG_PANEL, fg=TEXT_SECONDARY,
            font=self._font_status,
        ).pack(side=tk.LEFT, padx=8)

        # Emergency button — always visible in top bar
        self._emergency_btn = tk.Button(
            top_bar,
            text="🚨  EMERGENCY",
            bg=EMERGENCY_RED,
            fg=TEXT_EMERGENCY,
            font=tkfont.Font(family=self._cfg.ui.font_family, size=13, weight="bold"),
            relief=tk.RAISED,
            bd=3,
            padx=18,
            command=self._trigger_emergency_ui,
            cursor="hand2",
        )
        self._emergency_btn.pack(side=tk.RIGHT, padx=16, pady=8)

        # Latency indicator
        if self._cfg.ui.show_latency_indicator:
            tk.Label(
                top_bar, textvariable=self._latency_var,
                bg=BG_PANEL, fg=TEXT_SECONDARY,
                font=self._font_status,
            ).pack(side=tk.RIGHT, padx=8)

        # ── Main body ──────────────────────────
        body = tk.Frame(self, bg=BG_MAIN)
        body.pack(fill=tk.BOTH, expand=True)

        # Left: Camera preview
        if self._cfg.ui.show_camera_feed and self._mode == PipelineMode.WEBCAM:
            self._camera_frame = tk.Frame(
                body, bg=BG_PANEL,
                width=self._cfg.ui.camera_preview_width + 16,
            )
            self._camera_frame.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
            self._camera_frame.pack_propagate(False)

            tk.Label(
                self._camera_frame, text="Eye Tracking",
                bg=BG_PANEL, fg=TEXT_SECONDARY, font=self._font_status,
            ).pack(pady=(8, 4))

            self._camera_label = tk.Label(
                self._camera_frame, bg=BG_PANEL,
                width=self._cfg.ui.camera_preview_width,
                height=self._cfg.ui.camera_preview_height,
            )
            self._camera_label.pack(pady=4)
        else:
            self._camera_label = None  # type: ignore[assignment]

        # Centre: Symbol board
        board_frame = tk.Frame(body, bg=BG_MAIN)
        board_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._build_symbol_board(board_frame)

        # Right: Sentence output
        output_frame = tk.Frame(body, bg=BG_PANEL, width=380)
        output_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)
        output_frame.pack_propagate(False)
        self._build_output_panel(output_frame)

        # ── Bottom status bar ──────────────────
        status_bar = tk.Frame(self, bg=BG_PANEL, height=32)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        status_bar.pack_propagate(False)

        tk.Label(
            status_bar, textvariable=self._status_var,
            bg=BG_PANEL, fg=TEXT_SECONDARY, font=self._font_status,
        ).pack(side=tk.LEFT, padx=12, pady=6)

        # ── Emergency overlay ──────────────────
        self._emergency_overlay_frame = tk.Frame(self, bg=EMERGENCY_RED)
        self._emergency_overlay_frame.place_forget()

        tk.Label(
            self._emergency_overlay_frame,
            text="🚨 EMERGENCY 🚨",
            bg=EMERGENCY_RED, fg=TEXT_EMERGENCY,
            font=self._font_emergency,
        ).pack(expand=True)

        self._emergency_msg_label = tk.Label(
            self._emergency_overlay_frame,
            text="",
            bg=EMERGENCY_RED, fg=TEXT_EMERGENCY,
            font=tkfont.Font(family=self._cfg.ui.font_family, size=28),
            wraplength=800,
        )
        self._emergency_msg_label.pack(expand=True)

        tk.Button(
            self._emergency_overlay_frame,
            text="Dismiss",
            bg=BG_MAIN, fg=TEXT_PRIMARY,
            font=self._font_status,
            command=lambda: self._emergency_overlay_frame.place_forget(),
        ).pack(pady=20)

        # ── Register with DisplayManager ───────
        self._display = DisplayManager(
            root=self,
            font_family=self._cfg.ui.font_family,
            sentence_font_size=self._cfg.ui.sentence_font_size,
            board_font_size=self._cfg.ui.board_font_size,
            status_font_size=self._cfg.ui.status_font_size,
        )
        self._display.register_sentence_var(self._sentence_var)
        self._display.register_confidence_var(self._confidence_var, self._confidence_label)
        self._display.register_status_var(self._status_var)
        self._display.register_latency_var(self._latency_var)
        self._display.register_emergency_overlay(self._emergency_overlay_frame)

    def _build_symbol_board(self, parent: tk.Frame) -> None:
        """
        Build the paged symbol board grid inside the given parent frame.

        Creates a grid of buttons for the current page, with page navigation
        controls at the bottom.

        Args:
            parent: Parent frame to build the board inside.
        """
        tk.Label(
            parent, text="Symbol Board — Gaze to select",
            bg=BG_MAIN, fg=TEXT_SECONDARY, font=self._font_status,
        ).pack(pady=(0, 6))

        self._board_grid_frame = tk.Frame(parent, bg=BG_MAIN)
        self._board_grid_frame.pack(fill=tk.BOTH, expand=True)

        # Page navigation controls
        nav_frame = tk.Frame(parent, bg=BG_MAIN)
        nav_frame.pack(fill=tk.X, pady=6)

        tk.Button(
            nav_frame, text="◀ Prev",
            bg=BG_PANEL, fg=TEXT_PRIMARY, font=self._font_status,
            command=self._prev_page, cursor="hand2",
        ).pack(side=tk.LEFT, padx=8)

        self._page_label = tk.Label(
            nav_frame, text="Page 1 / 3 — Body Parts",
            bg=BG_MAIN, fg=TEXT_SECONDARY, font=self._font_status,
        )
        self._page_label.pack(side=tk.LEFT, expand=True)

        tk.Button(
            nav_frame, text="Next ▶",
            bg=BG_PANEL, fg=TEXT_PRIMARY, font=self._font_status,
            command=self._next_page, cursor="hand2",
        ).pack(side=tk.RIGHT, padx=8)

        # Reset button
        tk.Button(
            nav_frame, text="↺ Reset",
            bg=BG_PANEL, fg=ACCENT_AMBER, font=self._font_status,
            command=self._reset_intent, cursor="hand2",
        ).pack(side=tk.RIGHT, padx=8)

    def _render_board_page(self, symbols: list[Symbol]) -> None:
        """
        Render symbol buttons for the given page onto the board grid.

        Clears previous buttons before drawing new ones.

        Args:
            symbols: List of :class:`Symbol` objects to render.
        """
        for widget in self._board_grid_frame.winfo_children():
            widget.destroy()
        self._board_buttons.clear()

        rows = self._cfg.symbol_board.rows
        cols = self._cfg.symbol_board.columns

        for r in range(rows):
            self._board_grid_frame.rowconfigure(r, weight=1)
        for c in range(cols):
            self._board_grid_frame.columnconfigure(c, weight=1)

        for symbol in symbols:
            btn = tk.Button(
                self._board_grid_frame,
                text=symbol.label,
                bg=BG_BOARD_CELL,
                fg=TEXT_PRIMARY,
                activebackground=BG_BOARD_CELL_HOVER,
                activeforeground=TEXT_PRIMARY,
                font=self._font_board,
                relief=tk.FLAT,
                bd=2,
                wraplength=180,
                padx=8, pady=12,
                cursor="hand2",
                command=lambda s=symbol: self._on_symbol_click(s),
            )
            btn.grid(
                row=symbol.row, column=symbol.col,
                sticky="nsew", padx=4, pady=4,
            )
            self._board_buttons[symbol.region_id] = btn

    def _build_output_panel(self, parent: tk.Frame) -> None:
        """
        Build the right-hand output panel: sentence display, confidence badge,
        and intent progress indicator.

        Args:
            parent: Parent frame for the output panel.
        """
        tk.Label(
            parent, text="Reconstructed Sentence",
            bg=BG_PANEL, fg=TEXT_SECONDARY, font=self._font_status,
        ).pack(pady=(12, 4), padx=12)

        sentence_label = tk.Label(
            parent,
            textvariable=self._sentence_var,
            bg=BG_PANEL, fg=TEXT_PRIMARY,
            font=self._font_sentence,
            wraplength=340,
            justify=tk.LEFT,
        )
        sentence_label.pack(fill=tk.X, padx=16, pady=8)

        # Confidence badge
        self._confidence_label = tk.Label(
            parent,
            textvariable=self._confidence_var,
            bg=BG_PANEL, fg=ACCENT_GREEN,
            font=tkfont.Font(family=self._cfg.ui.font_family, size=14),
        )
        self._confidence_label.pack(pady=4)

        tk.Frame(parent, bg=TEXT_SECONDARY, height=1).pack(fill=tk.X, padx=16, pady=8)

        # Intent progress checklist
        tk.Label(
            parent, text="Intent Progress",
            bg=BG_PANEL, fg=TEXT_SECONDARY, font=self._font_status,
        ).pack(pady=(0, 4))

        self._intent_vars: dict[str, tk.StringVar] = {}
        for category in ["BODY_PART", "SENSATION", "URGENCY", "INTENSITY"]:
            var = tk.StringVar(value=f"○ {category.replace('_', ' ').title()}")
            self._intent_vars[category] = var
            tk.Label(
                parent, textvariable=var,
                bg=BG_PANEL, fg=TEXT_SECONDARY,
                font=self._font_status,
                anchor=tk.W,
            ).pack(fill=tk.X, padx=20, pady=2)

        # Speak again button
        tk.Button(
            parent, text="🔊 Speak Again",
            bg=BG_PANEL, fg=ACCENT_BLUE,
            font=self._font_status,
            command=self._speak_again, cursor="hand2",
        ).pack(pady=16)

    # ──────────────────────────────────────────
    # Pipeline lifecycle
    # ──────────────────────────────────────────

    def start_pipeline(self) -> None:
        """
        Create, load model, and start the Sentinel pipeline.

        Model loading happens in a background thread to keep the UI responsive.
        Displays "Loading model…" status during initialisation.
        """
        if self._display:
            self._display.show_status("Loading MedGemma model… (30–60s on first run)", "#f39c12")

        self._pipeline = SentinelPipeline(
            config=self._cfg,
            mode=self._mode,
            on_event=self._on_pipeline_event,
        )

        # Refresh board UI with the pipeline's symbol board
        page_symbols = self._pipeline.symbol_board.current_symbols
        self.after(0, lambda: self._render_board_page(page_symbols))

        def _load_and_start() -> None:
            """Load model in background, then start the event loop."""
            try:
                self._pipeline.load_model()  # type: ignore[union-attr]
                self._pipeline.start()  # type: ignore[union-attr]
                self.after(0, lambda: self._status_var.set(
                    "Ready — select a body part to begin"
                ))
            except Exception as exc:  # noqa: BLE001
                logger.error("Pipeline startup failed: %s", exc, exc_info=True)
                self.after(0, lambda: self._status_var.set(
                    f"⚠ Error: {exc}"
                ))

        threading.Thread(target=_load_and_start, daemon=True, name="model-loader").start()

    # ──────────────────────────────────────────
    # Event handlers
    # ──────────────────────────────────────────

    def _on_pipeline_event(self, event: PipelineEvent) -> None:
        """
        Handle events emitted by the Sentinel pipeline (called from pipeline thread).

        Dispatches UI updates to the Tkinter main loop via after().

        Args:
            event: The pipeline event to handle.
        """
        if event.kind == "sentence_ready":
            result: InferenceResult = event.payload  # type: ignore[assignment]
            self.after(0, lambda r=result: self._on_sentence_ready(r))

        elif event.kind == "symbol_selected":
            symbol: Symbol = event.payload  # type: ignore[assignment]
            self.after(0, lambda s=symbol: self._on_symbol_selected_ui(s))

        elif event.kind == "state_change":
            state: PipelineState = event.payload  # type: ignore[assignment]
            self.after(0, lambda s=state: self._on_state_change(s))

        elif event.kind == "gaze" and self._camera_label:
            gaze_frame = event.payload
            if gaze_frame and gaze_frame.raw_frame is not None:  # type: ignore[union-attr]
                self.after(0, lambda f=gaze_frame: self._update_camera_preview(f))

        elif event.kind == "emergency":
            msg: str = event.payload  # type: ignore[assignment]
            self.after(0, lambda m=msg: self._show_emergency_overlay(m))

        elif event.kind == "latency_warn":
            payload: dict = event.payload  # type: ignore[assignment]
            self.after(0, lambda p=payload: self._latency_var.set(
                f"⚠ {p['elapsed_ms']:.0f}ms"
            ))

    def _on_sentence_ready(self, result: InferenceResult) -> None:
        """
        Update UI with a newly reconstructed sentence.

        Args:
            result: The :class:`InferenceResult` from the LLM engine.
        """
        if self._display:
            self._display.show_sentence(result.text, result.confidence)
            self._display.show_latency(result.latency_ms, result.truncated)
        self._last_sentence = result.text
        self._reset_intent_progress()

    def _on_symbol_selected_ui(self, symbol: Symbol) -> None:
        """
        Visually highlight the selected symbol button and update intent progress.

        Args:
            symbol: The selected :class:`Symbol`.
        """
        # Highlight button
        btn = self._board_buttons.get(symbol.region_id)
        if btn:
            btn.configure(bg=BG_BOARD_CELL_SELECTED)

        # Update intent progress checklist
        var = self._intent_vars.get(symbol.category)
        if var:
            friendly = symbol.category.replace("_", " ").title()
            var.set(f"✅ {friendly}: {symbol.value}")

        if self._display:
            missing = [
                k.replace("_", " ").title()
                for k, v in self._intent_vars.items()
                if not v.get().startswith("✅")
            ]
            if missing:
                self._display.show_status(f"Now select: {', '.join(missing)}")
            else:
                self._display.show_status("Intent complete — generating sentence…")

    def _on_state_change(self, state: PipelineState) -> None:
        """
        Update status bar and button states based on pipeline state.

        Args:
            state: The new :class:`PipelineState`.
        """
        state_messages = {
            PipelineState.IDLE: "Ready — select a body part to begin",
            PipelineState.SELECTING: "Selecting…",
            PipelineState.GENERATING: "⏳ Generating sentence — please wait…",
            PipelineState.SPEAKING: "🔊 Speaking…",
            PipelineState.EMERGENCY: "🚨 EMERGENCY ACTIVE",
            PipelineState.SHUTDOWN: "System shutting down",
        }
        msg = state_messages.get(state, state.name)
        if self._display:
            self._display.show_status(msg)

    def _on_keypress(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """
        Handle key press events for simulator mode.

        Forwards navigation keys to the keyboard simulator if active.
        Direct emergency key (F1) and reset (Escape) are bound separately.

        Args:
            event: Tkinter key event.
        """
        if self._pipeline is None:
            return

        # Only inject into simulator
        if (
            self._mode == PipelineMode.SIMULATOR
            and hasattr(self._pipeline, "_gaze_source")
            and isinstance(self._pipeline._gaze_source, KeyboardSimulator)
        ):
            key = event.keysym
            if key in ("Left", "Right", "Up", "Down", "space", "b", "r"):
                self._pipeline._gaze_source.inject_key(key)
            elif key == "e":
                self._trigger_emergency_ui()

    def _on_symbol_click(self, symbol: Symbol) -> None:
        """
        Handle a direct mouse click on a symbol button (demo convenience).

        Args:
            symbol: The symbol that was clicked.
        """
        if self._pipeline:
            self._pipeline._board.set_page(symbol.page)
            # Directly process the selection via the intent classifier
            self._pipeline._process_selection(symbol)

    def _trigger_emergency_ui(self) -> None:
        """Trigger emergency override from UI button or F1 key."""
        if self._pipeline:
            threading.Thread(
                target=lambda: self._pipeline.trigger_emergency(),  # type: ignore[union-attr]
                daemon=True,
            ).start()
        else:
            # Pipeline not started — use TTS directly
            self._show_emergency_overlay("EMERGENCY — I need immediate help now")

    def _show_emergency_overlay(self, message: str) -> None:
        """
        Show the full-screen red emergency overlay with the given message.

        Args:
            message: The emergency message string to display.
        """
        self._emergency_msg_label.configure(text=message)
        self._emergency_overlay_frame.lift()
        self._emergency_overlay_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.after(10_000, self._emergency_overlay_frame.place_forget)

    def _next_page(self) -> None:
        """Navigate the symbol board to the next page."""
        if self._pipeline:
            self._pipeline.symbol_board.next_page()
            page = self._pipeline.symbol_board.current_page
            symbols = self._pipeline.symbol_board.current_symbols
            self._render_board_page(symbols)
            page_names = ["Body Parts", "Sensations", "Urgency & Intensity"]
            name = page_names[page] if page < len(page_names) else f"Page {page + 1}"
            self._page_label.configure(text=f"Page {page + 1} / 3 — {name}")

    def _prev_page(self) -> None:
        """Navigate the symbol board to the previous page."""
        if self._pipeline:
            board = self._pipeline.symbol_board
            board.set_page((board.current_page - 1) % board.page_count)
            page = board.current_page
            symbols = board.current_symbols
            self._render_board_page(symbols)
            page_names = ["Body Parts", "Sensations", "Urgency & Intensity"]
            name = page_names[page] if page < len(page_names) else f"Page {page + 1}"
            self._page_label.configure(text=f"Page {page + 1} / 3 — {name}")

    def _reset_intent(self) -> None:
        """Reset intent classifier and restore board button colours."""
        if self._pipeline:
            self._pipeline._intent.reset()
        for btn in self._board_buttons.values():
            btn.configure(bg=BG_BOARD_CELL)
        self._reset_intent_progress()
        if self._display:
            self._display.show_status("Intent reset — select a body part to begin")
            self._display.clear_sentence()

    def _reset_intent_progress(self) -> None:
        """Reset the intent progress checklist to unchecked state."""
        for category, var in self._intent_vars.items():
            friendly = category.replace("_", " ").title()
            var.set(f"○ {friendly}")
        for btn in self._board_buttons.values():
            btn.configure(bg=BG_BOARD_CELL)

    def _speak_again(self) -> None:
        """Re-speak the last reconstructed sentence."""
        text = getattr(self, "_last_sentence", "")
        if text and self._pipeline:
            self._pipeline._tts.speak(text)

    def _toggle_mode(self) -> None:
        """Toggle between webcam and simulator mode (F2) — requires restart."""
        if self._display:
            self._display.show_status(
                "Mode toggle requires restart. Use --mode webcam / --mode simulator"
            )

    def _update_camera_preview(self, gaze_frame: object) -> None:
        """
        Update the camera preview label with the latest frame and gaze crosshair.

        Args:
            gaze_frame: A :class:`~sentinel.gaze.tracker.GazeFrame` with raw_frame.
        """
        if self._camera_label is None:
            return
        try:
            raw: np.ndarray = gaze_frame.raw_frame  # type: ignore[attr-defined]
            x_px = int(gaze_frame.x * raw.shape[1])  # type: ignore[attr-defined]
            y_px = int(gaze_frame.y * raw.shape[0])  # type: ignore[attr-defined]

            # Draw crosshair
            overlay = raw.copy()
            cv2.circle(overlay, (x_px, y_px), 12, (74, 144, 226), 2)
            cv2.line(overlay, (x_px - 18, y_px), (x_px + 18, y_px), (74, 144, 226), 1)
            cv2.line(overlay, (x_px, y_px - 18), (x_px, y_px + 18), (74, 144, 226), 1)

            # Resize for preview
            pw = self._cfg.ui.camera_preview_width
            ph = self._cfg.ui.camera_preview_height
            resized = cv2.resize(overlay, (pw, ph))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(image=img)
            self._camera_photo = photo
            self._camera_label.configure(image=photo)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Camera preview update error: %s", exc)

    def _on_close(self) -> None:
        """Handle window close — stop pipeline gracefully."""
        logger.info("Window closing — stopping pipeline")
        if self._pipeline:
            threading.Thread(target=self._pipeline.stop, daemon=True).start()
        self.after(500, self.destroy)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the Sentinel application.

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description="NeuroWeave Sentinel — Offline AAC for ALS/locked-in patients",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["webcam", "simulator"],
        default="simulator",
        help="Input mode: 'webcam' (MediaPipe iris) or 'simulator' (keyboard). Default: simulator",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to sentinel.yaml config file. Auto-discovers if not specified.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. Default: INFO",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the NeuroWeave Sentinel application.

    Parses arguments, loads configuration, builds the Tkinter app,
    starts the pipeline in a background thread, and runs the GUI event loop.
    """
    args = _parse_args()
    _setup_logging(args.log_level)

    logger.info("Starting NeuroWeave Sentinel (mode=%s)", args.mode)

    config: SentinelConfig = load_config(args.config)

    mode = (
        PipelineMode.SIMULATOR
        if args.mode == "simulator"
        else PipelineMode.WEBCAM
    )

    app = SentinelApp(config=config, mode=mode)
    # Start pipeline after the event loop is running
    app.after(200, app.start_pipeline)

    try:
        app.mainloop()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — exiting")
        sys.exit(0)


if __name__ == "__main__":
    main()
