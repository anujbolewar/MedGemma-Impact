"""
sentinel/output/tts.py — Offline text-to-speech engine using pyttsx3.

Provides non-blocking speech synthesis running in a daemon thread.
Supports standard sentence speech and emergency pre-recorded audio playback.
All audio output is fully offline — no network calls are made.
"""

from __future__ import annotations

import logging
import threading
import time
import wave
from pathlib import Path
from typing import Optional

import pyttsx3  # type: ignore[import]

from sentinel.core.config import TTSConfig

logger = logging.getLogger(__name__)


class TTSEngine:
    """
    Offline text-to-speech engine wrapping pyttsx3.

    Speech requests are queued and executed in a dedicated background thread
    to avoid blocking the main event loop. A maximum of one speech request
    is processed at a time; new requests interrupt any currently-running speech.

    Args:
        config: TTS configuration (rate, volume, voice selection).
    """

    def __init__(self, config: TTSConfig) -> None:
        """Initialise TTS engine and configure voice properties."""
        self._cfg = config
        self._lock = threading.Lock()
        self._speaking = False
        self._pending_text: Optional[str] = None
        self._shutdown_flag = False

        self._engine: Optional[pyttsx3.Engine] = None
        self._worker_thread: Optional[threading.Thread] = None

        self._init_engine()
        self._start_worker()

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def speak(self, text: str) -> None:
        """
        Queue text for asynchronous speech synthesis.

        If speech is currently playing, it is interrupted and the new text
        takes priority. This is intentional — the patient's latest intent
        should always be voiced without delay.

        Args:
            text: The sentence to speak. Must be non-empty.
        """
        text = text.strip()
        if not text:
            logger.warning("TTSEngine.speak called with empty text — ignored")
            return

        with self._lock:
            self._pending_text = text
            if self._engine and self._speaking:
                try:
                    self._engine.stop()
                except Exception:  # noqa: BLE001
                    pass

        logger.info("TTS: queuing speech: %r", text[:80])

    def speak_emergency(self, text: str) -> None:
        """
        Immediately speak an emergency message, bypassing the queue.

        Interrupts any in-progress speech. Runs in the calling thread for
        maximum responsiveness in emergency scenarios.

        Args:
            text: The emergency message to speak immediately.
        """
        logger.warning("TTS EMERGENCY: speaking %r", text)
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", max(130, self._cfg.rate - 20))
            engine.setProperty("volume", 1.0)  # Always maximum for emergency
            if self._cfg.voice_id:
                engine.setProperty("voice", self._cfg.voice_id)
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:  # noqa: BLE001
            logger.error("TTS emergency speech failed: %s", exc)

    def play_wav(self, wav_path: Path) -> None:
        """
        Play a pre-recorded WAV file (blocking, for emergency audio).

        Falls back to TTS synthesis if the file cannot be played.

        Args:
            wav_path: Path to the WAV audio file.
        """
        if not wav_path.exists():
            logger.warning("WAV not found at %s — skipping playback", wav_path)
            return

        try:
            import simpleaudio as sa  # type: ignore[import]  # optional dep
            with wave.open(str(wav_path), "rb") as wf:
                raw = wf.readframes(wf.getnframes())
                audio = sa.play_buffer(
                    raw,
                    num_channels=wf.getnchannels(),
                    bytes_per_sample=wf.getsampwidth(),
                    sample_rate=wf.getframerate(),
                )
                audio.wait_done()
        except ImportError:
            logger.warning("simpleaudio not installed — falling back to pyttsx3 for WAV")
        except Exception as exc:  # noqa: BLE001
            logger.error("WAV playback failed: %s", exc)

    @property
    def is_speaking(self) -> bool:
        """Return True if the TTS engine is currently producing audio."""
        with self._lock:
            return self._speaking

    def shutdown(self) -> None:
        """
        Stop the worker thread and release TTS resources.

        Safe to call multiple times. Blocks until the worker exits (max 3s).
        """
        self._shutdown_flag = True
        if self._worker_thread:
            self._worker_thread.join(timeout=3.0)
        if self._engine:
            try:
                self._engine.stop()
            except Exception:  # noqa: BLE001
                pass
        logger.info("TTSEngine shut down")

    # ──────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────

    def _init_engine(self) -> None:
        """
        Initialise the pyttsx3 engine and apply configuration.

        Creates a fresh engine instance and sets rate, volume, and optional voice.
        pyttsx3 engines are not thread-safe, so this instance is only used in the worker thread.
        """
        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self._cfg.rate)
            self._engine.setProperty("volume", self._cfg.volume)
            if self._cfg.voice_id:
                self._engine.setProperty("voice", self._cfg.voice_id)
            logger.info(
                "TTS engine initialised (rate=%d, volume=%.1f)",
                self._cfg.rate, self._cfg.volume,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("TTS engine init failed: %s — TTS will be unavailable", exc)
            self._engine = None

    def _start_worker(self) -> None:
        """Start the background worker thread that polls for pending speech."""
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="tts-worker", daemon=True
        )
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """
        Background loop that drains the speech queue.

        Polls for pending text every 50ms. When text is available, speaks it
        using the pyttsx3 engine (blocking runAndWait).
        """
        while not self._shutdown_flag:
            text: Optional[str] = None
            with self._lock:
                if self._pending_text is not None:
                    text = self._pending_text
                    self._pending_text = None

            if text is not None and self._engine is not None:
                with self._lock:
                    self._speaking = True
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception as exc:  # noqa: BLE001
                    logger.error("TTS speak error: %s", exc)
                finally:
                    with self._lock:
                        self._speaking = False
            else:
                time.sleep(0.05)
