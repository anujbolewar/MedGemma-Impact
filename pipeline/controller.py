"""
pipeline/controller.py — NWSController: main system orchestrator for NeuroWeave Sentinel.

Wires together all subsystems in strict initialisation order and drives the FSM
state machine through the full communication pipeline::

    GazeSource ─► SignalFuser ─► IntentEncoder ─► LLM ─► SafetyGate ─► TTS

An internal EventBus lets UI components subscribe to pipeline events without
holding direct references to internal modules.  Long-running steps (LLM
inference, TTS playback) are dispatched to background daemon threads so the
30 Hz main loop always returns promptly and emergency detection stays active.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from core.constants import C, FSMState
from core.fsm import InvalidTransitionError, SentinelFSM
from core.logger import get_logger
from encoder.intent_encoder import IntentEncoder
from encoder.intent_encoder import IntentPacket as EncoderPacket
from encoder.token_vocab import VOCAB
from input.signal_fuser import FusedSignalFrame, SignalFuser
from llm.model_loader import load_model, warm_up
from llm.reconstructor import (
    IntentPacket as LLMPacket,
    ReconstructionResult,
    SentenceReconstructor,
)
from output.emergency import EmergencyOverride
from output.tts_engine import TTSEngine
from safety.confidence import SafetyAction, SafetyGate
from safety.entropy_guard import EntropyGuard
from safety.grammar_check import GrammarChecker

_log = get_logger()

# ── EventBus event-name constants ─────────────────────────────────────────────

ON_TOKEN_SELECTED      = "ON_TOKEN_SELECTED"
"""Fired when the intent encoder finalises a confirmed token packet."""

ON_RECONSTRUCTION      = "ON_RECONSTRUCTION"
"""Fired after LLM inference completes with the reconstructed sentence."""

ON_SAFETY_DECISION     = "ON_SAFETY_DECISION"
"""Fired after the safety gate evaluates the reconstructed sentence."""

ON_SPEAKING            = "ON_SPEAKING"
"""Fired just before TTS begins synthesising a sentence."""

ON_EMERGENCY           = "ON_EMERGENCY"
"""Fired when the emergency override is triggered."""

ON_FALLBACK            = "ON_FALLBACK"
"""Fired when a template fallback sentence is substituted for LLM output."""

ON_CONFIRMATION_NEEDED = "ON_CONFIRMATION_NEEDED"
"""Fired when the safety gate requests patient confirmation before voicing."""


# ── Gaze adapter ──────────────────────────────────────────────────────────────

@dataclass
class _GazePoint:
    """
    Duck-typed gaze frame satisfying :class:`~input.signal_fuser.SignalFuser`'s
    ``GazeSource`` protocol (``get_frame()`` → object with ``.x``, ``.y``,
    ``.confidence``).
    """

    x: float
    y: float
    confidence: float


class _GazeAdapter:
    """
    Bridges ``WebcamGazeTracker`` / ``SimulatedGazeTracker`` to
    :class:`~input.signal_fuser.SignalFuser`.

    Both tracker types expose ``get_latest() → Optional[GazeFrame]`` where
    ``GazeFrame.gaze_vector`` is ``(x, y)`` in ``[−1, 1]²``.
    :class:`~input.signal_fuser.SignalFuser` expects ``get_frame()`` returning
    an object with ``.x``, ``.y`` in ``[0, 1]²`` and ``.confidence``.

    Args:
        tracker: Any gaze tracker with a ``get_latest()`` method.
    """

    _NULL = _GazePoint(x=0.5, y=0.5, confidence=0.0)

    def __init__(self, tracker: Any) -> None:
        self._tracker = tracker

    def get_frame(self) -> _GazePoint:
        """Return the latest gaze point remapped to ``[0, 1]²``, or a centred null frame."""
        frame = self._tracker.get_latest()
        if frame is None:
            return self._NULL
        gx = (float(frame.gaze_vector[0]) + 1.0) / 2.0
        gy = (float(frame.gaze_vector[1]) + 1.0) / 2.0
        return _GazePoint(x=gx, y=gy, confidence=float(frame.confidence))


# ── NWSController ─────────────────────────────────────────────────────────────

class NWSController:
    """
    Main system orchestrator for NeuroWeave Sentinel.

    Initialises all subsystems in strict dependency order, runs the 30 Hz gaze
    processing loop, and routes reconstructed sentences through the FSM and
    safety pipeline.

    Subsystem initialisation order:

    1.  :func:`~core.logger.get_logger` (singleton)
    2.  :class:`~core.fsm.SentinelFSM`
    3.  Gaze source (:class:`~input.gaze_webcam.WebcamGazeTracker` or
        :class:`~input.gaze_sim.SimulatedGazeTracker`, chosen by ``config['use_sim']``)
    4.  :class:`~input.signal_fuser.SignalFuser`
    5.  :data:`~encoder.token_vocab.VOCAB` — validated at init
    6.  :class:`~encoder.intent_encoder.IntentEncoder`
    7.  :func:`~llm.model_loader.load_model` + :func:`~llm.model_loader.warm_up`
    8.  :class:`~llm.reconstructor.SentenceReconstructor`
    9.  :class:`~safety.entropy_guard.EntropyGuard`,
        :class:`~safety.grammar_check.GrammarChecker`,
        :class:`~safety.confidence.SafetyGate`
    10. :class:`~output.tts_engine.TTSEngine`
    11. :class:`~output.emergency.EmergencyOverride`

    Args:
        config: Configuration dict.  Recognised keys:

            ==================  =======  ================================================
            Key                 Type     Description
            ==================  =======  ================================================
            ``use_sim``         bool     ``True`` → simulator; ``False`` → webcam.
            ``sim_mode``        str      ``'SCRIPTED'`` | ``'RANDOM'`` | ``'INTERACTIVE'``
            ``camera_id``       int      Webcam device index (webcam mode only).
            ``model_path``      str      HuggingFace model ID or local directory path.
            ==================  =======  ================================================

    Example::

        ctrl = NWSController({"use_sim": True, "model_path": C.MODEL_ID})
        ctrl.subscribe(ON_SPEAKING, lambda d: print("Speaking:", d["text"]))
        t = threading.Thread(target=ctrl.run, daemon=True, name="nws-main")
        t.start()
        ...
        ctrl.shutdown()
        t.join(timeout=5.0)
    """

    #: Seconds to wait for patient confirmation before auto-rejecting.
    _CONFIRM_TIMEOUT_S: float = 30.0

    #: Milliseconds to wait for TTS playback to finish before returning to IDLE.
    _SPEAK_SYNC_TIMEOUT_MS: float = 15_000.0

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise all modules in dependency order, logging each step's latency."""

        # ── 1. Core logger ────────────────────────────────────────────────
        _t = time.perf_counter()
        self._log = get_logger()
        self._log.perf("pipeline", "init_logger",
                       (time.perf_counter() - _t) * 1_000.0, {})

        # ── 2. FSM ────────────────────────────────────────────────────────
        _t = time.perf_counter()
        self._fsm = SentinelFSM(on_transition=self._on_fsm_transition)
        self._log.perf("pipeline", "init_fsm",
                       (time.perf_counter() - _t) * 1_000.0, {})

        # ── 3. Gaze source ────────────────────────────────────────────────
        _t = time.perf_counter()
        _use_sim: bool = bool(config.get("use_sim", True))
        if _use_sim:
            from input.gaze_sim import SimulatedGazeTracker, SimulationMode  # noqa: PLC0415
            _raw_mode = config.get("sim_mode", "SCRIPTED")
            _mode = (
                SimulationMode[_raw_mode] if isinstance(_raw_mode, str) else _raw_mode
            )
            self._gaze_tracker: Any = SimulatedGazeTracker(mode=_mode)
        else:
            from input.gaze_webcam import WebcamGazeTracker  # noqa: PLC0415
            self._gaze_tracker = WebcamGazeTracker(
                camera_id=int(config.get("camera_id", 0))
            )
        self._gaze_adapter = _GazeAdapter(self._gaze_tracker)
        self._log.perf("pipeline", "init_gaze",
                       (time.perf_counter() - _t) * 1_000.0, {"use_sim": _use_sim})

        # ── 4. Signal fuser ───────────────────────────────────────────────
        _t = time.perf_counter()
        self._fuser = SignalFuser(gaze_source=self._gaze_adapter)
        self._log.perf("pipeline", "init_signal_fuser",
                       (time.perf_counter() - _t) * 1_000.0, {})

        # ── 5. Token vocab (module-level constant — verify it loaded) ─────
        _t = time.perf_counter()
        _vocab_size = len(VOCAB)
        self._log.perf("pipeline", "init_token_vocab",
                       (time.perf_counter() - _t) * 1_000.0, {"vocab_size": _vocab_size})

        # ── 6. Intent encoder ─────────────────────────────────────────────
        _t = time.perf_counter()
        self._intent_encoder = IntentEncoder()
        self._log.perf("pipeline", "init_intent_encoder",
                       (time.perf_counter() - _t) * 1_000.0, {})

        # ── 7. LLM model loader + warm-up ─────────────────────────────────
        _t = time.perf_counter()
        _model_path: str = str(config.get("model_path", C.MODEL_ID))
        _model, _tokenizer = load_model(_model_path)
        warm_up(_model, _tokenizer)
        self._log.perf("pipeline", "init_model_loader",
                       (time.perf_counter() - _t) * 1_000.0, {"model_path": _model_path})

        # ── 8. Sentence reconstructor ─────────────────────────────────────
        _t = time.perf_counter()
        self._reconstructor = SentenceReconstructor(
            model=_model, tokenizer=_tokenizer
        )
        self._log.perf("pipeline", "init_reconstructor",
                       (time.perf_counter() - _t) * 1_000.0, {})

        # ── 9. Safety pipeline ────────────────────────────────────────────
        _t = time.perf_counter()
        self._entropy_guard = EntropyGuard()
        self._grammar_checker = GrammarChecker()
        self._safety_gate = SafetyGate(
            entropy_guard=self._entropy_guard,
            grammar_checker=self._grammar_checker,
        )
        self._log.perf("pipeline", "init_safety",
                       (time.perf_counter() - _t) * 1_000.0, {})

        # ── 10. TTS engine ────────────────────────────────────────────────
        _t = time.perf_counter()
        self._tts = TTSEngine()
        self._log.perf("pipeline", "init_tts_engine",
                       (time.perf_counter() - _t) * 1_000.0, {})

        # ── 11. Emergency override ────────────────────────────────────────
        # EmergencyOverride holds a direct FSM reference; check_trigger()
        # automatically fires trigger() (FSM → EMERGENCY + audio) when the
        # patient dwells on the emergency gaze zone.
        _t = time.perf_counter()
        self._emergency_override = EmergencyOverride(
            tts_engine=self._tts, fsm=self._fsm
        )
        self._log.perf("pipeline", "init_emergency",
                       (time.perf_counter() - _t) * 1_000.0, {})

        # ── EventBus ──────────────────────────────────────────────────────
        self._subscribers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = (
            defaultdict(list)
        )

        # ── Runtime state ─────────────────────────────────────────────────
        self._running: bool = False

        # Confirmation handshake: inference thread blocks; UI sets via confirm_output()
        self._confirm_event = threading.Event()
        self._confirm_approved: Optional[bool] = None

        self._log.info("pipeline", "controller_ready", {
            "use_sim": _use_sim,
            "model_path": _model_path,
            "vocab_size": _vocab_size,
        })

    # ── EventBus ──────────────────────────────────────────────────────────────

    def subscribe(self, event: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register *callback* to receive payloads whenever *event* is published.

        Multiple callbacks per event are supported. Each callback is invoked
        synchronously in registration order; exceptions are caught and logged so
        a failing callback never disrupts remaining subscribers.

        Args:
            event:    One of the ``ON_*`` module-level constants.
            callback: Callable ``(data: dict) → None``.
        """
        self._subscribers[event].append(callback)
        _log.info("pipeline", "event_subscribed", {"event": event})

    def publish(self, event: str, data: Dict[str, Any]) -> None:
        """
        Dispatch *event* to all registered callbacks with payload *data*.

        Args:
            event: Event name string (one of the ``ON_*`` constants).
            data:  JSON-safe dict payload passed verbatim to each callback.
        """
        for cb in self._subscribers.get(event, []):
            try:
                cb(data)
            except Exception as exc:  # noqa: BLE001
                _log.error("pipeline", "event_callback_error", {
                    "event": event,
                    "error": str(exc),
                })

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Start the gaze tracker and enter the 30 Hz main processing loop.

        **Blocking** — returns only after :meth:`shutdown` sets ``_running``
        to ``False``.  Typical usage runs this in a background thread so that
        the UI main loop remains on the main thread::

            t = threading.Thread(target=ctrl.run, daemon=True, name="nws-main")
            t.start()
            ...
            ctrl.shutdown()
            t.join(timeout=5.0)
        """
        self._gaze_tracker.start()
        self._running = True
        _log.info("pipeline", "run_start", {})
        self._main_loop()

    def shutdown(self) -> None:
        """
        Gracefully stop the main loop and all subsystems.

        Sets ``_running`` to ``False``, unblocks any waiting confirmation,
        stops the gaze tracker, terminates TTS, and flushes the JSONL log.
        Safe to call from any thread.
        """
        _log.info("pipeline", "shutdown_requested", {})
        self._running = False
        # Unblock any inference thread blocked in confirm_output() wait
        self._confirm_approved = False
        self._confirm_event.set()
        self._gaze_tracker.stop()
        self._tts.shutdown()
        _log.info("pipeline", "controller_shutdown", {})
        _log.flush()

    def confirm_output(self, approved: bool) -> None:
        """
        Resolve a pending patient-confirmation request from the UI.

        Must be called while the FSM is in the ``CONFIRMATION`` state (i.e.
        after an :data:`ON_CONFIRMATION_NEEDED` event has been published).
        Calling from any other state is a no-op (logged as a warning).

        Args:
            approved: ``True`` → voice the pending sentence immediately.
                      ``False`` → discard and return to token selection.
        """
        if self._fsm.current_state != FSMState.CONFIRMATION:
            _log.warn("pipeline", "confirm_output_wrong_state", {
                "current_state": self._fsm.current_state.value,
                "approved": approved,
            })
            return
        self._confirm_approved = approved
        self._confirm_event.set()
        _log.info("pipeline", "confirm_output_received", {"approved": approved})

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _main_loop(self) -> None:
        """Drive the 30 Hz gaze processing loop until ``_running`` is ``False``."""
        interval_s = 1.0 / 30.0
        while self._running:
            _t0 = time.monotonic()
            try:
                self._tick()
            except Exception as exc:  # noqa: BLE001
                _log.error("pipeline", "tick_unhandled_error", {"error": str(exc)})
            _remainder = interval_s - (time.monotonic() - _t0)
            if _remainder > 0.0:
                time.sleep(_remainder)

    def _tick(self) -> None:
        """
        Process one gaze frame at ~30 Hz.

        Polling order:

        1. Fuse gaze signals into a :class:`~input.signal_fuser.FusedSignalFrame`.
        2. Check emergency dwell via :meth:`~output.emergency.EmergencyOverride.check_trigger`;
           if triggered, publish :data:`ON_EMERGENCY` and return.
        3. Skip frames when FSM is not in ``IDLE`` or ``TOKEN_SELECTION``.
        4. Feed the frame to :class:`~encoder.intent_encoder.IntentEncoder`.
        5. When the encoder returns a confirmed packet, advance the FSM and
           dispatch LLM inference to a background thread.
        """
        frame: FusedSignalFrame = self._fuser.get_fused_frame()

        # ── Emergency dwell check ─────────────────────────────────────────
        # EmergencyOverride.check_trigger() handles dwell timing, FSM
        # transition, and audio broadcast internally; returns True once fired.
        if self._emergency_override.check_trigger(frame):
            _log.critical("pipeline", "emergency_triggered", {
                "direction": frame.primary_direction,
                "composite_confidence": frame.composite_confidence,
            })
            self.publish(ON_EMERGENCY, {
                "source": "gaze_dwell",
                "direction": frame.primary_direction,
            })
            return

        # ── Guard: only process selection frames in active states ─────────
        _state = self._fsm.current_state
        if _state not in (FSMState.IDLE, FSMState.TOKEN_SELECTION):
            return

        # ── IDLE → TOKEN_SELECTION on first non-centre gaze ───────────────
        if _state is FSMState.IDLE and frame.primary_direction != "CENTRE":
            try:
                self._fsm.transition(FSMState.TOKEN_SELECTION, reason="gaze_active")
            except InvalidTransitionError:
                pass  # Tolerate concurrent transition attempts

        # ── Feed frame to stateful intent encoder ─────────────────────────
        enc_packet: Optional[EncoderPacket] = self._intent_encoder.encode([frame])

        if enc_packet is None:
            return  # Still collecting — nothing more to do this tick

        # ── Dwell-confirm: patient finished token selection ───────────────
        self.publish(ON_TOKEN_SELECTED, enc_packet.to_dict())
        _log.info("pipeline", "selection_confirmed", enc_packet.to_dict())

        # Advance FSM: TOKEN_SELECTION → CONFIRMATION → GENERATING
        # CONFIRMATION is a transient "selection committed" gate; we
        # auto-advance immediately to GENERATING without blocking.
        try:
            if self._fsm.current_state is FSMState.TOKEN_SELECTION:
                self._fsm.transition(
                    FSMState.CONFIRMATION, reason="selection_complete"
                )
            self._fsm.transition(FSMState.GENERATING, reason="auto_confirm")
        except InvalidTransitionError as exc:
            _log.error("pipeline", "fsm_selection_advance_error", {
                "error": str(exc),
                "state": self._fsm.current_state.value,
            })
            self._fsm.reset()
            self._intent_encoder.reset()
            return

        # Bridge encoder packet → LLM packet (token codes + mean confidence)
        llm_packet = LLMPacket(
            tokens=enc_packet.token_codes,
            confidence=enc_packet.confidence,
        )

        # Dispatch inference to a background thread (keeps tick() non-blocking)
        threading.Thread(
            target=self._run_inference,
            args=(llm_packet,),
            daemon=True,
            name="nws-inference",
        ).start()

    # ── Inference + safety thread ──────────────────────────────────────────────

    def _run_inference(self, packet: LLMPacket) -> None:
        """
        Full LLM inference + safety pipeline.  Runs in a background thread.

        Pipeline::

            GENERATING
              └─► [LLM inference]
                    └─► VALIDATING
                          └─► [SafetyGate.evaluate]
                                ├─ PROCEED  → SPEAKING → IDLE
                                ├─ CONFIRM  → CONFIRMATION ─► (approved) → SPEAKING → IDLE
                                │                           └► (rejected) → TOKEN_SELECTION
                                ├─ FALLBACK → FALLBACK → SPEAKING → IDLE
                                └─ BLOCK    → IDLE (forced reset, no audio)

        Args:
            packet: :class:`~llm.reconstructor.IntentPacket` to reconstruct.
        """
        # ── LLM inference ─────────────────────────────────────────────────
        _t_inf = time.perf_counter()
        try:
            result: ReconstructionResult = self._reconstructor.reconstruct(packet)
        except Exception as exc:  # noqa: BLE001
            _log.error("pipeline", "inference_exception", {"error": str(exc)})
            self._fsm.reset()
            self._intent_encoder.reset()
            return

        _inf_ms = (time.perf_counter() - _t_inf) * 1_000.0
        _log.perf("pipeline", "inference_complete", _inf_ms, {
            "text": result.text,
            "is_timeout": result.is_timeout,
            "is_fallback": result.is_fallback,
        })
        self.publish(ON_RECONSTRUCTION, {
            "text": result.text,
            "latency_ms": round(_inf_ms, 2),
            "is_timeout": result.is_timeout,
            "is_fallback": result.is_fallback,
        })

        # ── GENERATING → VALIDATING ───────────────────────────────────────
        try:
            self._fsm.transition(FSMState.VALIDATING, reason="inference_done")
        except InvalidTransitionError as exc:
            _log.error("pipeline", "fsm_to_validating_error", {
                "error": str(exc), "state": self._fsm.current_state.value,
            })
            self._fsm.reset()
            return

        # ── Safety gate ───────────────────────────────────────────────────
        _decision = self._safety_gate.evaluate(result, packet)
        self.publish(ON_SAFETY_DECISION, {
            "action": _decision.action.value,
            "composite_confidence": _decision.composite_confidence,
            "reasons": _decision.reasons,
            "latency_ms": _decision.latency_ms,
        })

        # ── Route by safety action ────────────────────────────────────────
        _action = _decision.action

        if _action is SafetyAction.PROCEED:
            self._do_speak(result.text, reason="safety_proceed")

        elif _action is SafetyAction.CONFIRM:
            self._do_confirm_flow(result, packet, _decision.composite_confidence)

        elif _action is SafetyAction.FALLBACK:
            self._do_fallback(packet, reason="safety_fallback")

        else:  # SafetyAction.BLOCK
            _log.warn("pipeline", "output_blocked", {
                "text": result.text,
                "composite_confidence": _decision.composite_confidence,
                "reasons": _decision.reasons,
            })
            # VALIDATING → IDLE (forced; VALIDATING→IDLE not in transition map)
            self._fsm.reset()
            self._intent_encoder.reset()

    # ── Pipeline step helpers ──────────────────────────────────────────────────

    def _do_speak(self, text: str, reason: str = "speak") -> None:
        """
        Transition current state → ``SPEAKING``, synthesise *text*, then → ``IDLE``.

        Valid source states: ``VALIDATING`` (PROCEED and post-confirm paths)
        or ``FALLBACK`` (template fallback path).

        Blocks the calling thread via :meth:`~output.tts_engine.TTSEngine.speak_sync`
        until playback completes or :attr:`_SPEAK_SYNC_TIMEOUT_MS` elapses.

        Args:
            text:   Sentence to synthesise and voice.
            reason: FSM transition reason label used for log entries.
        """
        try:
            self._fsm.transition(FSMState.SPEAKING, reason=reason)
        except InvalidTransitionError as exc:
            _log.error("pipeline", "fsm_to_speaking_error", {
                "error": str(exc), "state": self._fsm.current_state.value,
            })
            self._fsm.reset()
            return

        self.publish(ON_SPEAKING, {"text": text})
        _ok = self._tts.speak_sync(text, timeout_ms=self._SPEAK_SYNC_TIMEOUT_MS)
        if not _ok:
            _log.warn("pipeline", "speak_sync_incomplete", {"text": text})

        try:
            self._fsm.transition(FSMState.IDLE, reason="speaking_done")
        except InvalidTransitionError:
            self._fsm.reset()

    def _do_confirm_flow(
        self,
        result: ReconstructionResult,
        packet: LLMPacket,
        composite_confidence: float,
    ) -> None:
        """
        Enter ``CONFIRMATION`` state and block until the patient responds.

        Publishes :data:`ON_CONFIRMATION_NEEDED`, then waits up to
        :attr:`_CONFIRM_TIMEOUT_S` for :meth:`confirm_output` to be called by
        the UI thread.

        - **Approved** → ``CONFIRMATION → GENERATING → VALIDATING → SPEAKING → IDLE``
        - **Rejected / timeout** → ``CONFIRMATION → TOKEN_SELECTION``

        Args:
            result:               LLM reconstruction pending confirmation.
            packet:               Original intent packet (kept for reference).
            composite_confidence: Safety gate composite score for UI display.
        """
        self._confirm_event.clear()
        self._confirm_approved = None

        try:
            self._fsm.transition(FSMState.CONFIRMATION, reason="safety_confirm")
        except InvalidTransitionError as exc:
            _log.error("pipeline", "fsm_to_confirmation_error", {
                "error": str(exc), "state": self._fsm.current_state.value,
            })
            self._fsm.reset()
            return

        self.publish(ON_CONFIRMATION_NEEDED, {
            "text": result.text,
            "composite_confidence": round(composite_confidence, 4),
        })

        # Block inference thread — UI thread calls confirm_output()
        _responded = self._confirm_event.wait(timeout=self._CONFIRM_TIMEOUT_S)

        if _responded and self._confirm_approved:
            # Patient approved — re-use the stored result (skip LLM re-inference)
            # CONFIRMATION → GENERATING → VALIDATING → SPEAKING
            try:
                self._fsm.transition(FSMState.GENERATING, reason="user_confirmed")
                self._fsm.transition(FSMState.VALIDATING, reason="confirmed_skip_llm")
            except InvalidTransitionError as exc:
                _log.error("pipeline", "fsm_confirm_advance_error", {
                    "error": str(exc), "state": self._fsm.current_state.value,
                })
                self._fsm.reset()
                return
            self._do_speak(result.text, reason="user_confirmed_speak")
        else:
            _decline_reason = "user_rejected" if _responded else "confirm_timeout"
            _log.info("pipeline", "confirmation_declined", {"reason": _decline_reason})
            try:
                self._fsm.transition(
                    FSMState.TOKEN_SELECTION, reason=_decline_reason
                )
            except InvalidTransitionError:
                self._fsm.reset()
            self._intent_encoder.reset()

    def _do_fallback(self, packet: LLMPacket, reason: str = "fallback") -> None:
        """
        Transition to ``FALLBACK``, reconstruct from template, then speak.

        Publishes :data:`ON_FALLBACK` with the template sentence text before
        voicing it.

        Args:
            packet: Intent packet for :meth:`~llm.reconstructor.SentenceReconstructor.reconstruct_from_template`.
            reason: FSM transition reason label.
        """
        try:
            self._fsm.transition(FSMState.FALLBACK, reason=reason)
        except InvalidTransitionError as exc:
            _log.error("pipeline", "fsm_to_fallback_error", {
                "error": str(exc), "state": self._fsm.current_state.value,
            })
            self._fsm.reset()
            return

        _fallback_result = self._reconstructor.reconstruct_from_template(packet)
        _log.info("pipeline", "fallback_sentence", {"text": _fallback_result.text})
        self.publish(ON_FALLBACK, {"text": _fallback_result.text})

        # FALLBACK → SPEAKING → IDLE
        self._do_speak(_fallback_result.text, reason="fallback_speak")

    # ── FSM transition callback ───────────────────────────────────────────────

    def _on_fsm_transition(
        self,
        from_state: FSMState,
        to_state: FSMState,
        reason: str,
    ) -> None:
        """
        Wired to :class:`~core.fsm.SentinelFSM` as ``on_transition`` callback.

        Logs every state transition and schedules automatic EMERGENCY recovery
        when the FSM enters the ``EMERGENCY`` state.

        Args:
            from_state: Previous FSM state.
            to_state:   New FSM state.
            reason:     Caller-supplied reason string.
        """
        _log.info("pipeline", "fsm_transition", {
            "from": from_state.value,
            "to": to_state.value,
            "reason": reason,
        })

        if to_state is FSMState.EMERGENCY:
            # Schedule auto-recovery after emergency audio and cooldown finish
            threading.Thread(
                target=self._emergency_recovery,
                daemon=True,
                name="nws-emerg-recovery",
            ).start()

    def _emergency_recovery(self) -> None:
        """
        Background thread: wait for emergency audio to finish, then reset to IDLE.

        Sleeps 5 seconds (long enough for emergency TTS to complete), clears the
        ``EmergencyOverride`` detection state so it can re-trigger, then
        transitions the FSM back to ``IDLE``.
        """
        time.sleep(5.0)
        self._emergency_override.reset()
        try:
            self._fsm.transition(FSMState.IDLE, reason="emergency_cleared")
        except InvalidTransitionError:
            self._fsm.reset()
        self._intent_encoder.reset()
        _log.info("pipeline", "emergency_recovery_complete", {})

