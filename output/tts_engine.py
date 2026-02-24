"""
output/tts_engine.py — Thread-safe offline text-to-speech engine.

Primary: Coqui TTS ('tts_models/en/ljspeech/vits') via temp WAV + pygame.
Fallback: subprocess espeak when Coqui fails to load.

All synthesis runs in a daemon worker thread; the public API is non-blocking.
"""

from __future__ import annotations

import os
import queue
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from core.constants import SentinelConstants as C
from core.logger import get_logger

_log = get_logger()

# Sentinel value to signal the worker to exit
_STOP_SENTINEL = object()

# espeak fallback command template
_ESPEAK_CMD: list[str] = ["espeak", "-s", "150", "-v", "en"]


# ──────────────────────────────────────────────────────────────
# Internal queue item
# ──────────────────────────────────────────────────────────────

@dataclass
class _SpeechJob:
    """A single speech synthesis job enqueued for the worker thread."""

    job_id: str
    text: str
    priority: int = 0            # 0 = normal, 1 = urgent
    done_event: threading.Event = field(default_factory=threading.Event)
    success: bool = False


# ──────────────────────────────────────────────────────────────
# TTSEngine
# ──────────────────────────────────────────────────────────────

class TTSEngine:
    """
    Non-blocking offline TTS engine with priority queue and worker thread.

    On init, attempts to load the Coqui TTS model
    ``tts_models/en/ljspeech/vits`` and initialise ``pygame.mixer``.
    If Coqui fails, falls back to ``espeak`` via subprocess.

    Usage::

        engine = TTSEngine()
        job_id = engine.speak("I need help.", priority=0)
        ok = engine.speak_sync("Emergency!", timeout_ms=2000)
        engine.speak_file("assets/emergency.wav")
        engine.stop()
        engine.shutdown()
    """

    def __init__(self) -> None:
        """Initialise audio backend and start the worker thread."""
        self._tts: Optional[object] = None         # Coqui TTS instance
        self._espeak_fallback: bool = False
        self._pygame_ready: bool = False

        # Job queue (maxsize=3 — drop oldest normal jobs when full)
        self._queue: queue.Queue[object] = queue.Queue(maxsize=3)
        self._current_job: Optional[_SpeechJob] = None
        self._lock = threading.Lock()
        self._running = False

        self._init_audio_backend()
        self._start_worker()

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def speak(self, text: str, priority: int = 0) -> str:
        """
        Enqueue a speech request and return immediately.

        If priority is 1 (urgent), any waiting jobs are drained before
        inserting so the urgent message plays next.

        Args:
            text: Text to synthesise.
            priority: 0 = normal (queued), 1 = urgent (jumps queue).

        Returns:
            A unique job ID string.
        """
        if not text or not text.strip():
            return ""

        job_id = str(uuid.uuid4())[:8]
        job = _SpeechJob(job_id=job_id, text=text.strip(), priority=priority)

        if priority >= 1:
            # Drain normal jobs to make room for urgent
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

        try:
            self._queue.put_nowait(job)
        except queue.Full:
            # Drop oldest normal job if queue is full
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(job)

        _log.info("tts_engine", "enqueued", {
            "job_id": job_id,
            "priority": priority,
            "text_len": len(text),
        })
        return job_id

    def speak_sync(self, text: str, timeout_ms: float = C.TTS_MAX_MS) -> bool:
        """
        Synthesise and play text, blocking until complete or timeout.

        Args:
            text: Text to synthesise.
            timeout_ms: Maximum milliseconds to wait.

        Returns:
            True if synthesis completed before timeout, False otherwise.
        """
        if not text or not text.strip():
            return False

        job_id = str(uuid.uuid4())[:8]
        job = _SpeechJob(job_id=job_id, text=text.strip(), priority=1)

        # Drain queue and put urgent job at front
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put_nowait(job)

        completed = job.done_event.wait(timeout=timeout_ms / 1000.0)
        if not completed:
            _log.warn("tts_engine", "speak_sync_timeout", {
                "job_id": job_id,
                "timeout_ms": timeout_ms,
            })
        return completed and job.success

    def speak_file(self, wav_path: str) -> None:
        """
        Play a pre-recorded WAV file directly via pygame (non-blocking).

        Bypasses TTS synthesis — used for emergency pre-recorded audio.

        Args:
            wav_path: Absolute or relative path to the WAV file.
        """
        if not self._pygame_ready:
            _log.warn("tts_engine", "speak_file_no_pygame", {"path": wav_path})
            return

        def _play() -> None:
            try:
                import pygame  # type: ignore
                sound = pygame.mixer.Sound(wav_path)
                sound.play()
                _log.info("tts_engine", "speak_file_played", {"path": wav_path})
            except Exception as exc:  # noqa: BLE001
                _log.error("tts_engine", "speak_file_error", {
                    "path": wav_path, "error": str(exc)
                })

        threading.Thread(target=_play, daemon=True, name="tts-wav").start()

    def stop(self) -> None:
        """
        Stop the current playback immediately and clear the queue.

        If pygame is available, stops the mixer channel. Drains pending jobs.
        """
        # Drain queue
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        if self._pygame_ready:
            try:
                import pygame  # type: ignore
                pygame.mixer.stop()
            except Exception:  # noqa: BLE001
                pass

        _log.info("tts_engine", "stopped", {})

    def shutdown(self) -> None:
        """
        Stop the worker thread and release audio resources.

        Blocks until the worker thread exits (up to 3 seconds). Should be
        called once on application exit.
        """
        self._running = False
        self._queue.put(_STOP_SENTINEL)

        if hasattr(self, "_worker") and self._worker.is_alive():
            self._worker.join(timeout=3.0)

        if self._pygame_ready:
            try:
                import pygame  # type: ignore
                pygame.mixer.quit()
            except Exception:  # noqa: BLE001
                pass

        _log.info("tts_engine", "shutdown", {})

    # ──────────────────────────────────────────
    # Initialisation helpers
    # ──────────────────────────────────────────

    def _init_audio_backend(self) -> None:
        """
        Attempt to load Coqui TTS and pygame. Falls back to espeak on failure.
        """
        # ── pygame mixer ─────────────────────────────────────
        try:
            import pygame  # type: ignore
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            self._pygame_ready = True
            _log.info("tts_engine", "pygame_ready", {
                "freq": 22050, "channels": 1
            })
        except Exception as exc:  # noqa: BLE001
            _log.warn("tts_engine", "pygame_init_failed", {"error": str(exc)})

        # ── Coqui TTS ─────────────────────────────────────────
        try:
            from TTS.api import TTS  # type: ignore
            self._tts = TTS("tts_models/en/ljspeech/vits", progress_bar=False)
            _log.info("tts_engine", "coqui_tts_loaded", {
                "model": "tts_models/en/ljspeech/vits"
            })
        except Exception as exc:  # noqa: BLE001
            _log.warn("tts_engine", "coqui_tts_unavailable", {
                "error": str(exc),
                "fallback": "espeak",
            })
            self._espeak_fallback = True

    def _start_worker(self) -> None:
        """Start the daemon worker thread."""
        self._running = True
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="tts-worker",
            daemon=True,
        )
        self._worker.start()
        _log.info("tts_engine", "worker_started", {})

    # ──────────────────────────────────────────
    # Worker thread
    # ──────────────────────────────────────────

    def _worker_loop(self) -> None:
        """
        Drain the job queue: synthesise → play WAV → delete temp file.

        Runs until :meth:`shutdown` is called. Exceptions are caught and
        logged — the loop never crashes the worker thread.
        """
        while self._running:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is _STOP_SENTINEL:
                break

            job: _SpeechJob = item  # type: ignore[assignment]
            with self._lock:
                self._current_job = job

            try:
                self._synthesise_and_play(job)
            except Exception as exc:  # noqa: BLE001
                _log.error("tts_engine", "worker_error", {
                    "job_id": job.job_id,
                    "error": str(exc),
                })
            finally:
                job.done_event.set()
                with self._lock:
                    self._current_job = None

    def _synthesise_and_play(self, job: _SpeechJob) -> None:
        """
        Synthesise ``job.text`` to a temp WAV, play it, then delete the file.

        Logs synthesis latency. If synthesis exceeds TTS_MAX_MS, logs a
        warning and skips playback.

        Args:
            job: The speech job to process.
        """
        t0 = time.monotonic()

        # ── Synthesis ────────────────────────────────────────
        tmp_path: Optional[str] = None
        try:
            if not self._espeak_fallback and self._tts is not None:
                # Coqui TTS → temp WAV
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp:
                    tmp_path = tmp.name

                self._tts.tts_to_file(  # type: ignore[attr-defined]
                    text=job.text,
                    file_path=tmp_path,
                )
            else:
                # espeak → no WAV output (plays directly via subprocess)
                result = subprocess.run(
                    _ESPEAK_CMD + [job.text],
                    timeout=C.TTS_MAX_MS / 1000.0,
                    capture_output=True,
                )
                latency_ms = (time.monotonic() - t0) * 1000.0
                _log.perf("tts_engine", "espeak_done", latency_ms, {
                    "job_id": job.job_id,
                    "returncode": result.returncode,
                })
                job.success = result.returncode == 0
                return
        except subprocess.TimeoutExpired:
            _log.warn("tts_engine", "espeak_timeout", {
                "job_id": job.job_id, "timeout_ms": C.TTS_MAX_MS
            })
            return
        except Exception as exc:  # noqa: BLE001
            _log.error("tts_engine", "synthesis_error", {
                "job_id": job.job_id, "error": str(exc)
            })
            return

        latency_ms = (time.monotonic() - t0) * 1000.0

        # ── Latency check ─────────────────────────────────────
        if latency_ms > C.TTS_MAX_MS:
            _log.warn("tts_engine", "synthesis_too_slow", {
                "job_id": job.job_id,
                "latency_ms": round(latency_ms, 1),
                "limit_ms": C.TTS_MAX_MS,
            })
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return

        _log.perf("tts_engine", "synthesis_done", latency_ms, {
            "job_id": job.job_id,
            "text_len": len(job.text),
        })

        # ── Playback ──────────────────────────────────────────
        try:
            if tmp_path and self._pygame_ready:
                import pygame  # type: ignore
                sound = pygame.mixer.Sound(tmp_path)
                channel = sound.play()
                if channel:
                    while channel.get_busy():
                        time.sleep(0.02)
            job.success = True
        except Exception as exc:  # noqa: BLE001
            _log.error("tts_engine", "playback_error", {
                "job_id": job.job_id, "error": str(exc)
            })
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
