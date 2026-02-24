"""
output/emergency.py — Emergency override system for NeuroWeave Sentinel.

CRITICAL: This module NEVER calls the LLM. All responses are pre-recorded
or pre-configured. The entire trigger() path must complete in < 500ms.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from typing import Optional

from core.constants import FSMState, SentinelConstants as C
from core.logger import get_logger
from input.signal_fuser import FusedSignalFrame
from output.tts_engine import TTSEngine

_log = get_logger()

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

# Gaze direction that triggers the emergency check
_TRIGGER_DIR: str = C.EMERGENCY_GAZE_DIRECTION           # "UP"

# Seconds of sustained gaze required for standard trigger
_STANDARD_HOLD_S: float = 3.0

# Seconds for the unconditional fast-path trigger (ignores confidence)
_FAST_HOLD_S: float = 5.0

# Minimum confidence for standard path (keeps noise out)
_MIN_CONFIDENCE: float = 0.5

# Fallback message if WAV file is missing
_FALLBACK_TEXT: str = "EMERGENCY. Please call for help immediately."

# Pre-synthesised emergency text (used to generate the WAV once at startup)
_EMERGENCY_TEXT: str = (
    "EMERGENCY. Please call for immediate help. "
    "This patient needs attention now."
)


# ──────────────────────────────────────────────────────────────
# EmergencyAudioGenerator
# ──────────────────────────────────────────────────────────────

class EmergencyAudioGenerator:
    """
    Generates the emergency WAV file at application startup.

    Tries, in order:
    1. Coqui TTS (if installed)
    2. espeak subprocess
    3. Silent fallback (creates an empty placeholder WAV)

    The WAV is only re-generated if the file does not already exist.
    """

    @staticmethod
    def generate_emergency_wav(output_path: str) -> None:
        """
        Synthesise the emergency phrase to ``output_path`` if not present.

        Args:
            output_path: Destination WAV file path.
        """
        if os.path.exists(output_path):
            _log.info("emergency_audio", "wav_exists", {"path": output_path})
            return

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Try Coqui TTS
        if EmergencyAudioGenerator._try_coqui(output_path):
            return

        # Try espeak
        if EmergencyAudioGenerator._try_espeak(output_path):
            return

        # Silent placeholder — prevents missing-file errors at runtime
        EmergencyAudioGenerator._write_silent_wav(output_path)

    @staticmethod
    def _try_coqui(output_path: str) -> bool:
        """Attempt synthesis with Coqui TTS. Returns True on success."""
        try:
            from TTS.api import TTS  # type: ignore
            tts = TTS("tts_models/en/ljspeech/vits", progress_bar=False)
            tts.tts_to_file(text=_EMERGENCY_TEXT, file_path=output_path)
            _log.info("emergency_audio", "generated_coqui", {"path": output_path})
            return True
        except Exception as exc:  # noqa: BLE001
            _log.warn("emergency_audio", "coqui_failed", {"error": str(exc)})
            return False

    @staticmethod
    def _try_espeak(output_path: str) -> bool:
        """Attempt synthesis with espeak WAV output. Returns True on success."""
        try:
            result = subprocess.run(
                ["espeak", "-s", "130", "-v", "en",
                 "--stdout", _EMERGENCY_TEXT],
                capture_output=True,
                timeout=10.0,
            )
            if result.returncode == 0 and result.stdout:
                with open(output_path, "wb") as f:
                    f.write(result.stdout)
                _log.info("emergency_audio", "generated_espeak", {"path": output_path})
                return True
            return False
        except Exception as exc:  # noqa: BLE001
            _log.warn("emergency_audio", "espeak_failed", {"error": str(exc)})
            return False

    @staticmethod
    def _write_silent_wav(output_path: str) -> None:
        """Write a minimal silent WAV file as a placeholder."""
        # 44-byte WAV header for 0 PCM samples (valid, zero-length audio)
        _riff: bytes = (
            b"RIFF" + (36).to_bytes(4, "little") +
            b"WAVE"
            b"fmt " + (16).to_bytes(4, "little") +
            (1).to_bytes(2, "little") +      # PCM
            (1).to_bytes(2, "little") +      # mono
            (22050).to_bytes(4, "little") +  # sample rate
            (44100).to_bytes(4, "little") +  # byte rate
            (2).to_bytes(2, "little") +      # block align
            (16).to_bytes(2, "little") +     # bits/sample
            b"data" + (0).to_bytes(4, "little")
        )
        with open(output_path, "wb") as f:
            f.write(_riff)
        _log.warn("emergency_audio", "silent_placeholder", {"path": output_path})


# ──────────────────────────────────────────────────────────────
# EmergencyOverride
# ──────────────────────────────────────────────────────────────

class EmergencyOverride:
    """
    Emergency override detector and broadcaster.

    Monitors each :class:`~input.signal_fuser.FusedSignalFrame` for
    sustained gaze in the emergency direction (``EMERGENCY_GAZE_DIRECTION``).
    When the trigger condition is met, calls :meth:`trigger` which fires
    synchronously in < 500ms without touching the LLM.

    Args:
        tts_engine: Running :class:`~output.tts_engine.TTSEngine` instance.
        fsm: The pipeline FSM (any object with ``.current_state`` property
             and ``.transition(FSMState)`` method).
    """

    def __init__(self, tts_engine: TTSEngine, fsm: object) -> None:
        """Initialise detector state and pre-generate the emergency WAV."""
        self._tts = tts_engine
        self._fsm = fsm
        self._audio_path: str = C.EMERGENCY_AUDIO_PATH

        # Detection state
        self._hold_start: Optional[float] = None    # monotonic time when hold began
        self._triggered: bool = False
        self._lock = threading.Lock()

        # Pre-generate WAV at startup (non-blocking)
        threading.Thread(
            target=EmergencyAudioGenerator.generate_emergency_wav,
            args=(self._audio_path,),
            daemon=True,
            name="emergency-wav-gen",
        ).start()

        _log.info("emergency_override", "init", {
            "trigger_dir": _TRIGGER_DIR,
            "standard_hold_s": _STANDARD_HOLD_S,
            "fast_hold_s": _FAST_HOLD_S,
            "min_confidence": _MIN_CONFIDENCE,
            "audio_path": self._audio_path,
        })

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def check_trigger(self, fused_frame: FusedSignalFrame) -> bool:
        """
        Evaluate one gaze frame for emergency trigger conditions.

        Trigger fires when **all** of these are met:
        - Direction == ``EMERGENCY_GAZE_DIRECTION`` for ≥ 3 seconds AND confidence ≥ 0.5
        - FSM is not already in EMERGENCY state

        Fast path: if direction held for ≥ 5 seconds, triggers regardless of confidence.

        Args:
            fused_frame: Latest fused signal frame from :class:`~input.signal_fuser.SignalFuser`.

        Returns:
            True if the emergency was triggered this call, False otherwise.
        """
        with self._lock:
            if self._triggered:
                return False

            direction = fused_frame.primary_direction
            confidence = fused_frame.composite_confidence

            # Reset hold timer if gaze left the trigger zone
            if direction != _TRIGGER_DIR:
                self._hold_start = None
                return False

            # Already in EMERGENCY state — don't re-trigger
            fsm_state = getattr(self._fsm, "current_state", None)
            if fsm_state == FSMState.EMERGENCY:
                self._hold_start = None
                return False

            now = time.monotonic()
            if self._hold_start is None:
                self._hold_start = now
                return False

            hold_s = now - self._hold_start

            # Fast-path: unconditional trigger at 5s
            if hold_s >= _FAST_HOLD_S:
                _log.warn("emergency_override", "fast_path_trigger", {
                    "hold_s": round(hold_s, 2),
                    "confidence": round(confidence, 3),
                })
                self._triggered = True

            # Standard path: 3s + sufficient confidence
            elif hold_s >= _STANDARD_HOLD_S and confidence >= _MIN_CONFIDENCE:
                _log.warn("emergency_override", "standard_trigger", {
                    "hold_s": round(hold_s, 2),
                    "confidence": round(confidence, 3),
                })
                self._triggered = True

            if self._triggered:
                # Trigger outside the lock to avoid deadlock in trigger()
                pass
            else:
                return False

        # Fire trigger (outside lock)
        self.trigger()
        return True

    def trigger(self) -> None:
        """
        Execute the emergency override sequence.

        1. Transitions FSM to EMERGENCY state.
        2. Plays the pre-recorded emergency WAV (or fallback TTS).
        3. Logs the complete event with latency.

        NEVER calls the LLM. Must complete in < 500ms.
        """
        t0 = time.monotonic()

        _log.critical("emergency_override", "trigger_start", {
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

        # ── 1. FSM transition ─────────────────────────────────
        try:
            self._fsm.transition(FSMState.EMERGENCY, "emergency_gaze_trigger")
        except Exception as exc:  # noqa: BLE001
            # Do not abort — audio must still fire
            _log.warn("emergency_override", "fsm_transition_failed", {
                "error": str(exc)
            })

        # ── 2. Audio broadcast ────────────────────────────────
        if os.path.exists(self._audio_path):
            self._tts.speak_file(self._audio_path)
        else:
            _log.warn("emergency_override", "wav_not_found", {
                "path": self._audio_path,
                "fallback": "tts_speak",
            })
            self._tts.speak(_FALLBACK_TEXT, priority=1)

        # ── 3. Latency measurement ────────────────────────────
        latency_ms = (time.monotonic() - t0) * 1000.0
        _log.critical("emergency_override", "trigger_complete", {
            "latency_ms": round(latency_ms, 2),
            "within_budget": latency_ms < 500.0,
        })

        if latency_ms >= 500.0:
            _log.warn("emergency_override", "trigger_over_budget", {
                "latency_ms": round(latency_ms, 2),
                "budget_ms": 500.0,
            })

    def reset(self) -> None:
        """
        Clear all detection state, allowing the emergency to re-trigger.

        Call this after the clinical situation has been resolved and the
        FSM has been reset to IDLE.
        """
        with self._lock:
            self._hold_start = None
            self._triggered = False
        _log.info("emergency_override", "reset", {})
