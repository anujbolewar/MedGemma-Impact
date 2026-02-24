"""
sentinel/core/pipeline.py — Main orchestration pipeline for NeuroWeave Sentinel.

Ties together gaze tracking, intent classification, LLM inference, TTS output,
and emergency safety. Runs the core event loop and enforces the 2.5s latency budget.
"""

from __future__ import annotations

import logging
import signal
import time
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from sentinel.core.config import SentinelConfig, load_config
from sentinel.gaze.tracker import GazeFrame, GazeTracker
from sentinel.gaze.blink_detector import BlinkDetector, DwellTracker, BlinkEvent, DwellEvent
from sentinel.gaze.simulator import KeyboardSimulator
from sentinel.intent.symbol_board import SymbolBoard, Symbol
from sentinel.intent.classifier import IntentClassifier, IntentBundle
from sentinel.llm.engine import MedGemmaEngine, InferenceResult
from sentinel.llm.prompt_builder import PromptBuilder
from sentinel.output.tts import TTSEngine
from sentinel.safety.emergency import EmergencyOverride

logger = logging.getLogger(__name__)

LATENCY_BUDGET_MS: int = 2500  # Hard constraint — never exceed


class PipelineMode(Enum):
    """Input mode for the pipeline."""

    WEBCAM = 1
    SIMULATOR = 2


class PipelineState(Enum):
    """Lifecycle state of the Sentinel pipeline."""

    IDLE = 1
    SELECTING = 2
    GENERATING = 3
    SPEAKING = 4
    EMERGENCY = 5
    SHUTDOWN = 6


@dataclass
class PipelineEvent:
    """
    An event emitted by the pipeline that the UI can observe.

    Attributes:
        kind: One of 'gaze', 'symbol_selected', 'intent_ready', 'sentence_ready',
              'speaking', 'emergency', 'state_change', 'latency_warn'.
        payload: Arbitrary data associated with the event.
        timestamp: Monotonic time of event creation.
    """

    kind: str
    payload: object = None
    timestamp: float = field(default_factory=time.monotonic)


# Type alias for UI callbacks
EventCallback = Callable[[PipelineEvent], None]


class SentinelPipeline:
    """
    Main orchestration pipeline for NeuroWeave Sentinel.

    Manages the event loop, coordinates all subsystems, enforces latency budgets,
    and exposes a callback interface for the UI layer.

    Args:
        config: Validated :class:`SentinelConfig` instance.
        mode: Input mode — webcam or keyboard simulator.
        on_event: Optional callback invoked on each :class:`PipelineEvent`.
    """

    def __init__(
        self,
        config: SentinelConfig,
        mode: PipelineMode = PipelineMode.WEBCAM,
        on_event: Optional[EventCallback] = None,
    ) -> None:
        """Initialise all subsystems without starting the event loop."""
        self._config = config
        self._mode = mode
        self._on_event = on_event
        self._state = PipelineState.IDLE
        self._running = False
        self._loop_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        logger.info("Initialising SentinelPipeline (mode=%s)", mode.name)

        # Gaze input
        if mode == PipelineMode.SIMULATOR:
            self._gaze_source: GazeTracker | KeyboardSimulator = KeyboardSimulator(
                config=config.gaze
            )
        else:
            self._gaze_source = GazeTracker(config=config.camera, gaze_config=config.gaze)

        # Blink / dwell
        self._blink_detector = BlinkDetector(config=config.gaze)
        self._dwell_tracker = DwellTracker(
            dwell_threshold_ms=config.gaze.dwell_threshold_ms
        )

        # Symbol board & intent
        self._board = SymbolBoard(config=config.symbol_board)
        self._intent = IntentClassifier()

        # LLM
        self._prompt_builder = PromptBuilder()
        self._llm = MedGemmaEngine(config=config.llm)

        # Output
        self._tts = TTSEngine(config=config.tts)

        # Safety
        self._emergency = EmergencyOverride(
            config=config.emergency,
            tts=self._tts,
        )

        logger.info("SentinelPipeline initialised successfully")

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    @classmethod
    def from_config_file(
        cls,
        config_path: str | None = None,
        mode: PipelineMode = PipelineMode.WEBCAM,
        on_event: Optional[EventCallback] = None,
    ) -> "SentinelPipeline":
        """
        Convenience factory: load config from file and build pipeline.

        Args:
            config_path: Optional path to sentinel.yaml; auto-discovers if None.
            mode: Input mode to use.
            on_event: UI event callback.

        Returns:
            A ready-to-start :class:`SentinelPipeline`.
        """
        config = load_config(config_path)
        return cls(config=config, mode=mode, on_event=on_event)

    def load_model(self) -> None:
        """
        Pre-load MedGemma into memory.

        This is a blocking call that should be invoked before starting the
        event loop. Surfaces a clear error if the model is not cached.
        """
        logger.info("Loading MedGemma model — this may take 20–60 seconds on first run")
        self._llm.load()
        logger.info("Model loaded successfully")

    def start(self) -> None:
        """
        Start the pipeline event loop in a background thread.

        The loop reads gaze frames, processes dwell events, triggers LLM
        inference when intent is complete, and dispatches output.
        Safe to call only once; raises RuntimeError if already running.
        """
        if self._running:
            raise RuntimeError("Pipeline is already running")

        self._running = True
        self._gaze_source.start()

        # Register SIGINT / SIGTERM for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        self._loop_thread = threading.Thread(
            target=self._event_loop,
            name="sentinel-pipeline",
            daemon=True,
        )
        self._loop_thread.start()
        logger.info("Pipeline event loop started")

    def stop(self) -> None:
        """
        Stop the pipeline event loop and release all resources.

        Blocks until the background thread terminates (max 5s).
        """
        logger.info("Stopping pipeline...")
        self._running = False
        self._gaze_source.stop()

        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5.0)

        self._llm.unload()
        self._tts.shutdown()
        self._set_state(PipelineState.SHUTDOWN)
        logger.info("Pipeline stopped")

    def trigger_emergency(self) -> None:
        """
        Immediately activate the emergency override from any thread.

        Bypasses all LLM calls and outputs the pre-configured alert.
        """
        self._set_state(PipelineState.EMERGENCY)
        self._emergency.trigger()
        self._emit(PipelineEvent("emergency", payload=self._emergency.last_message))
        time.sleep(self._config.emergency.cooldown_seconds)
        self._set_state(PipelineState.IDLE)

    @property
    def state(self) -> PipelineState:
        """Return the current pipeline state (thread-safe)."""
        with self._lock:
            return self._state

    @property
    def symbol_board(self) -> SymbolBoard:
        """Return the SymbolBoard for UI rendering."""
        return self._board

    # ──────────────────────────────────────────
    # Event loop
    # ──────────────────────────────────────────

    def _event_loop(self) -> None:
        """
        Core processing loop (runs in background thread).

        On each iteration:
        1. Read a gaze frame from the input source.
        2. Detect blinks / dwell events.
        3. Map dwell to symbol board selection.
        4. Update intent bundle.
        5. If intent is complete, trigger LLM inference.
        6. Dispatch TTS and display output.
        """
        logger.debug("Event loop running")
        while self._running:
            try:
                self._tick()
            except Exception as exc:  # noqa: BLE001
                logger.error("Unhandled error in event loop: %s", exc, exc_info=True)
                # Continue running — system must not crash for patient safety
            time.sleep(0.01)  # ~100 Hz poll; gaze source has its own rate limiting

    def _tick(self) -> None:
        """
        Process one iteration of the event loop.

        Reads one gaze frame, detects input events, and progresses the
        state machine accordingly.
        """
        frame: GazeFrame = self._gaze_source.get_frame()
        self._emit(PipelineEvent("gaze", payload=frame))

        if self.state in (PipelineState.GENERATING, PipelineState.EMERGENCY):
            return  # Do not process new input during inference or emergency

        # Keyboard simulator surfaces blink events directly
        if hasattr(frame, "blink_event"):
            blink_event: Optional[BlinkEvent] = frame.blink_event  # type: ignore[attr-defined]
        else:
            blink_event = self._blink_detector.detect(frame)

        # Blink acts as explicit selection confirmation
        if blink_event is not None:
            self._handle_blink(blink_event, frame)
            return

        # Dwell acts as hover → auto-select
        dwell_event: Optional[DwellEvent] = self._dwell_tracker.update(
            x=frame.x, y=frame.y
        )
        if dwell_event is not None:
            self._handle_dwell(dwell_event)

    def _handle_blink(self, blink: BlinkEvent, frame: GazeFrame) -> None:
        """
        Handle a confirmed blink event.

        A long blink navigates the symbol board page.
        A short blink selects the currently gazed-at symbol.

        Args:
            blink: The detected blink event.
            frame: The current gaze frame providing position.
        """
        if blink.is_long:
            self._board.next_page()
            logger.debug("Page advanced to %d", self._board.current_page)
            return

        symbol = self._board.select(frame.x, frame.y)
        if symbol is not None:
            self._process_selection(symbol)

    def _handle_dwell(self, dwell: DwellEvent) -> None:
        """
        Handle a completed dwell event (gaze held on a region).

        Routes to emergency trigger if the dwell region is the emergency zone,
        otherwise treats it as a symbol selection.

        Args:
            dwell: The completed dwell event with region ID.
        """
        if dwell.region_id == "emergency":
            threading.Thread(target=self.trigger_emergency, daemon=True).start()
            return

        symbol = self._board.get_symbol_by_region(dwell.region_id)
        if symbol is not None:
            self._process_selection(symbol)

    def _process_selection(self, symbol: Symbol) -> None:
        """
        Update intent with a selected symbol and trigger inference if complete.

        Args:
            symbol: The symbol the patient selected.
        """
        self._set_state(PipelineState.SELECTING)
        bundle: IntentBundle = self._intent.update(symbol)
        self._emit(PipelineEvent("symbol_selected", payload=symbol))
        logger.info("Symbol selected: %s (%s)", symbol.label, symbol.category)

        if self._intent.is_complete():
            self._emit(PipelineEvent("intent_ready", payload=bundle))
            threading.Thread(
                target=self._run_inference,
                args=(bundle,),
                daemon=True,
                name="sentinel-infer",
            ).start()

    def _run_inference(self, bundle: IntentBundle) -> None:
        """
        Run MedGemma inference for a completed intent bundle.

        Enforces the 2.5s end-to-end latency budget. Dispatches TTS and
        emits 'sentence_ready' event. Resets intent state after completion.

        Args:
            bundle: The completed intent bundle to reconstruct.
        """
        self._set_state(PipelineState.GENERATING)
        t_start = time.monotonic()

        prompt = self._prompt_builder.build(bundle)

        try:
            result: InferenceResult = self._llm.infer(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM inference failed: %s", exc, exc_info=True)
            self._set_state(PipelineState.IDLE)
            return

        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        if elapsed_ms > LATENCY_BUDGET_MS:
            logger.warning(
                "End-to-end latency %.0fms exceeded budget (%dms)",
                elapsed_ms,
                LATENCY_BUDGET_MS,
            )
            self._emit(
                PipelineEvent("latency_warn", payload={"elapsed_ms": elapsed_ms})
            )

        logger.info(
            "Inference complete: %.0fms | text=%r | confidence=%.2f",
            result.latency_ms,
            result.text,
            result.confidence,
        )

        self._emit(PipelineEvent("sentence_ready", payload=result))
        self._intent.reset()

        # Speak output
        self._set_state(PipelineState.SPEAKING)
        self._tts.speak(result.text)
        self._emit(PipelineEvent("speaking", payload=result.text))
        self._set_state(PipelineState.IDLE)

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    def _set_state(self, new_state: PipelineState) -> None:
        """
        Transition to a new pipeline state and emit a state_change event.

        Args:
            new_state: The state to transition to.
        """
        with self._lock:
            old = self._state
            self._state = new_state
        if old != new_state:
            logger.debug("State: %s → %s", old.name, new_state.name)
            self._emit(PipelineEvent("state_change", payload=new_state))

    def _emit(self, event: PipelineEvent) -> None:
        """
        Invoke the UI callback with an event (non-blocking, swallows errors).

        Args:
            event: The event to dispatch to registered listeners.
        """
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Event callback raised: %s", exc)

    def _handle_signal(self, signum: int, frame: object) -> None:
        """
        Handle OS signals (SIGINT/SIGTERM) for graceful shutdown.

        Args:
            signum: Signal number received.
            frame: Current stack frame (unused).
        """
        logger.info("Received signal %d — stopping pipeline", signum)
        self.stop()
