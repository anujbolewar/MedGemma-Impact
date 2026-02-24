"""
sentinel/safety/emergency.py — Emergency override system.

Provides an immediate bypass of the LLM pipeline to deliver pre-configured
emergency messages via TTS. Designed for situations where the patient cannot
wait for gaze-based symbol selection.

Safety guarantees:
- No LLM calls — deterministic and instantaneous
- 10s cooldown prevents accidental re-trigger
- Session logging with timestamp for clinical audit trail
- Thread-safe trigger mechanism
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sentinel.core.config import EmergencyConfig

logger = logging.getLogger(__name__)


@dataclass
class EmergencyEvent:
    """
    Record of a single emergency trigger event for audit logging.

    Attributes:
        message: The emergency message that was broadcast.
        timestamp: Unix timestamp of the trigger.
        trigger_source: How the emergency was triggered (ui_button, keyboard, dwell).
        acknowledged: Set to True by clinical staff acknowledgement (future feature).
    """

    message: str
    timestamp: float = field(default_factory=time.time)
    trigger_source: str = "unknown"
    acknowledged: bool = False


class EmergencyOverride:
    """
    Immediate emergency communication override system.

    Bypasses all LLM calls and directly produces audible emergency alerts
    using pre-configured messages. Enforces a cooldown to prevent accidental
    rapid re-triggers.

    Args:
        config: Emergency configuration with messages and cooldown settings.
        tts: The :class:`~sentinel.output.tts.TTSEngine` instance for audio output.
    """

    # Imported lazily to avoid circular imports
    _TTSEngineType = None

    def __init__(
        self,
        config: EmergencyConfig,
        tts: object,  # TTSEngine — typed loosely to avoid circular import
    ) -> None:
        """Initialise emergency override with config and TTS engine."""
        self._cfg = config
        self._tts = tts
        self._lock = threading.Lock()
        self._last_trigger: float = 0.0
        self._active: bool = False
        self._session_log: list[EmergencyEvent] = []
        self._message_index: int = 0  # Cycles through configured messages
        logger.info(
            "EmergencyOverride ready: %d messages, cooldown=%ds",
            len(config.messages),
            config.cooldown_seconds,
        )

    @property
    def last_message(self) -> Optional[str]:
        """Return the most recently broadcast emergency message."""
        if self._session_log:
            return self._session_log[-1].message
        return None

    @property
    def is_active(self) -> bool:
        """Return True if emergency mode is currently active."""
        with self._lock:
            return self._active

    @property
    def session_log(self) -> list[EmergencyEvent]:
        """Return an immutable copy of all emergency events in this session."""
        with self._lock:
            return list(self._session_log)

    def trigger(self, source: str = "unknown") -> bool:
        """
        Activate the emergency override, broadcasting the current message via TTS.

        This method is thread-safe. If a cooldown is active, the call is
        silently rejected (returns False) to prevent accidental re-triggers
        while still being in the emergency state.

        Args:
            source: Description of what triggered the emergency (for audit log).

        Returns:
            True if the emergency was activated, False if in cooldown.
        """
        with self._lock:
            now = time.time()
            cooldown_remaining = self._cfg.cooldown_seconds - (now - self._last_trigger)

            if cooldown_remaining > 0 and self._last_trigger > 0:
                logger.warning(
                    "Emergency trigger rejected: cooling down (%.1fs remaining)",
                    cooldown_remaining,
                )
                return False

            self._active = True
            self._last_trigger = now

            # Select the current message and advance the index for next time
            message = self._cfg.messages[self._message_index % len(self._cfg.messages)]
            self._message_index += 1

            event = EmergencyEvent(
                message=message,
                timestamp=now,
                trigger_source=source,
            )
            self._session_log.append(event)

        logger.critical(
            "EMERGENCY TRIGGERED [source=%s]: %r", source, message
        )

        # Broadcast immediately — no LLM, no pipeline, direct TTS
        self._broadcast(message)
        return True

    def trigger_specific(self, message_index: int, source: str = "unknown") -> bool:
        """
        Trigger a specific emergency message by index.

        Useful when the UI exposes multiple emergency buttons (e.g.
        "I'm in pain" vs "Call nurse").

        Args:
            message_index: Index into the configured messages list.
            source: Trigger source for audit logging.

        Returns:
            True if triggered, False if in cooldown or index out of range.
        """
        if not (0 <= message_index < len(self._cfg.messages)):
            logger.error(
                "Emergency message index %d out of range [0, %d]",
                message_index,
                len(self._cfg.messages) - 1,
            )
            return False

        with self._lock:
            now = time.time()
            cooldown_remaining = self._cfg.cooldown_seconds - (now - self._last_trigger)
            if cooldown_remaining > 0 and self._last_trigger > 0:
                return False

            self._active = True
            self._last_trigger = now
            message = self._cfg.messages[message_index]
            self._session_log.append(
                EmergencyEvent(message=message, timestamp=now, trigger_source=source)
            )

        logger.critical("EMERGENCY [idx=%d, source=%s]: %r", message_index, source, message)
        self._broadcast(message)
        return True

    def reset(self) -> None:
        """
        Clear the active emergency state.

        The session log is preserved for clinical audit. This should be called
        once the cooldown has expired and the system returns to normal operation.
        """
        with self._lock:
            self._active = False
        logger.info("EmergencyOverride: reset to normal operation")

    def cooldown_remaining(self) -> float:
        """
        Return remaining cooldown seconds (0.0 if ready to trigger again).

        Returns:
            Seconds until next trigger is allowed; 0.0 if cooldown has elapsed.
        """
        with self._lock:
            if self._last_trigger == 0.0:
                return 0.0
            elapsed = time.time() - self._last_trigger
            remaining = self._cfg.cooldown_seconds - elapsed
            return max(0.0, remaining)

    def _broadcast(self, message: str) -> None:
        """
        Deliver the emergency message via TTS.

        Attempts WAV playback first if an audio file exists. Falls back to
        pyttsx3 synthesis if WAV is unavailable or playback fails.

        Args:
            message: The emergency text to broadcast.
        """
        # Try pre-recorded WAV first (higher quality, pre-rendered)
        if hasattr(self._tts, "play_wav"):
            wav_path = self._cfg.resolved_audio_dir / f"emergency_{self._message_index - 1}.wav"
            if wav_path.exists():
                threading.Thread(
                    target=self._tts.play_wav,
                    args=(wav_path,),
                    daemon=True,
                    name="emergency-audio",
                ).start()
                return

        # Fallback: TTS synthesis
        if self._cfg.use_tts_fallback and hasattr(self._tts, "speak_emergency"):
            threading.Thread(
                target=self._tts.speak_emergency,
                args=(message,),
                daemon=True,
                name="emergency-tts",
            ).start()
        else:
            logger.error(
                "Emergency broadcast failed: no audio output available. Message: %r",
                message,
            )
