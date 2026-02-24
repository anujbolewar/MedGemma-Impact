"""
core/constants.py — All system constants for NeuroWeave Sentinel.

Single frozen dataclass with typed constant groups: FSM states (Enum),
timing budgets, safety thresholds, hardware limits, and emergency settings.
Call ``SentinelConstants.validate()`` on startup to check RAM/VRAM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# FSM States
# ──────────────────────────────────────────────────────────────

class FSMState(Enum):
    """All valid states for the NeuroWeave Sentinel finite state machine."""

    IDLE = "IDLE"
    TOKEN_SELECTION = "TOKEN_SELECTION"
    CONFIRMATION = "CONFIRMATION"
    GENERATING = "GENERATING"
    VALIDATING = "VALIDATING"
    SPEAKING = "SPEAKING"
    EMERGENCY = "EMERGENCY"
    FALLBACK = "FALLBACK"


# ──────────────────────────────────────────────────────────────
# Frozen constants dataclass
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SentinelConstants:
    """
    Frozen dataclass holding all NeuroWeave Sentinel system constants.

    All fields are typed. Use the class attributes directly — do not
    instantiate this class. Call :meth:`validate` on application startup
    to log warnings if hardware limits would be exceeded.

    Example::

        from core.constants import SentinelConstants as C, FSMState

        print(C.MAX_INFERENCE_MS)   # 2000
        print(FSMState.EMERGENCY)   # FSMState.EMERGENCY
        C.validate()
    """

    # ── Timing (milliseconds) ─────────────────────────────────
    GAZE_DWELL_MS: ClassVar[float] = 1500.0
    """ms gaze must hold on a cell to confirm a selection."""

    EMERGENCY_HOLD_MS: ClassVar[float] = 3000.0
    """ms gaze must hold on the emergency zone to trigger override."""

    MAX_INFERENCE_MS: ClassVar[float] = 2000.0
    """Hard cutoff for MedGemma inference; generation aborted after this."""

    TTS_MAX_MS: ClassVar[float] = 800.0
    """Maximum allowed TTS synthesis time before fallback audio is used."""

    SIGNAL_WINDOW_MS: ClassVar[float] = 3000.0
    """Multi-modal signal fusion window width in milliseconds."""

    # ── Safety thresholds ─────────────────────────────────────
    ENTROPY_THRESHOLD: ClassVar[float] = 0.75
    """Suppress output if mean token entropy exceeds this (nats)."""

    CONFIDENCE_MIN: ClassVar[float] = 0.60
    """Minimum composite confidence score required to voice a sentence."""

    GRAMMAR_MAX_ERRORS: ClassVar[int] = 2
    """Maximum grammar errors allowed before output is rejected."""

    MAX_SENTENCE_TOKENS: ClassVar[int] = 30
    """Maximum number of tokens to generate per inference call."""

    MIN_SENTENCE_TOKENS: ClassVar[int] = 3
    """Minimum generated tokens; fewer indicates a likely empty output."""

    # ── Hardware limits ───────────────────────────────────────
    MAX_RAM_GB: ClassVar[float] = 8.0
    """System RAM budget in gigabytes."""

    MAX_VRAM_GB: ClassVar[float] = 4.0
    """GPU VRAM budget in gigabytes."""

    MODEL_ID: ClassVar[str] = "google/medgemma-4b-it"
    """HuggingFace model identifier for MedGemma."""

    QUANTIZATION: ClassVar[str] = "nf4"
    """Quantization mode: 'nf4' | 'int8' | 'none'."""

    # ── Emergency ─────────────────────────────────────────────
    EMERGENCY_GAZE_DIRECTION: ClassVar[str] = "UP"
    """Gaze direction that, held for EMERGENCY_HOLD_MS, triggers the override."""

    EMERGENCY_AUDIO_PATH: ClassVar[str] = "assets/emergency.wav"
    """Path to the pre-recorded emergency WAV file (relative to project root)."""

    # ── FSM states reference ──────────────────────────────────
    States: ClassVar[type[FSMState]] = FSMState
    """Convenience reference to :class:`FSMState` — use ``C.States.IDLE``."""

    # ─────────────────────────────────────────────────────────
    @classmethod
    def validate(cls) -> None:
        """
        Check current RAM and VRAM against system limits and log warnings.

        Call once on application startup. Does not raise — warnings are
        logged so the system can still attempt to run with degraded capacity.

        RAM is checked via :mod:`psutil`. VRAM is checked via ``torch.cuda``
        if a CUDA device is available. Both are silently skipped if the
        respective libraries are not installed.
        """
        cls._check_ram()
        cls._check_vram()

    @classmethod
    def _check_ram(cls) -> None:
        """Log a warning if available system RAM is below MAX_RAM_GB."""
        try:
            import psutil  # type: ignore
            available_gb = psutil.virtual_memory().available / (1024 ** 3)
            total_gb = psutil.virtual_memory().total / (1024 ** 3)
            if available_gb < cls.MAX_RAM_GB:
                logger.warning(
                    "RAM warning: %.1f GB available / %.1f GB total — "
                    "system limit is %.1f GB. Performance may be degraded.",
                    available_gb, total_gb, cls.MAX_RAM_GB,
                )
            else:
                logger.info(
                    "RAM OK: %.1f GB available (limit %.1f GB)",
                    available_gb, cls.MAX_RAM_GB,
                )
        except ImportError:
            logger.debug("psutil not installed — skipping RAM check")
        except Exception as exc:  # noqa: BLE001
            logger.debug("RAM check failed: %s", exc)

    @classmethod
    def _check_vram(cls) -> None:
        """Log a warning if GPU VRAM is below MAX_VRAM_GB (if CUDA available)."""
        try:
            import torch  # type: ignore
            if not torch.cuda.is_available():
                logger.info("VRAM check: no CUDA device — running on CPU")
                return
            free_bytes, total_bytes = torch.cuda.mem_get_info(device=0)
            free_gb = free_bytes / (1024 ** 3)
            total_gb = total_bytes / (1024 ** 3)
            if free_gb < cls.MAX_VRAM_GB:
                logger.warning(
                    "VRAM warning: %.1f GB free / %.1f GB total — "
                    "limit is %.1f GB. Consider closing other GPU processes.",
                    free_gb, total_gb, cls.MAX_VRAM_GB,
                )
            else:
                logger.info(
                    "VRAM OK: %.1f GB free / %.1f GB total (limit %.1f GB)",
                    free_gb, total_gb, cls.MAX_VRAM_GB,
                )
        except ImportError:
            logger.debug("torch not installed — skipping VRAM check")
        except Exception as exc:  # noqa: BLE001
            logger.debug("VRAM check failed: %s", exc)


# ──────────────────────────────────────────────────────────────
# Module-level convenience alias
# ──────────────────────────────────────────────────────────────

#: Convenience alias — ``from core.constants import C``
C = SentinelConstants

# ── Emergency module-level constants (consumed by output.emergency) ───────────

EMERGENCY_COOLDOWN_S: int = 10
"""Minimum seconds between successive emergency broadcast triggers."""

EMERGENCY_MESSAGES: tuple[str, ...] = (
    "Emergency! I need immediate help!",
    "Please call for help right now!",
    "I am in distress, please help me!",
    "Call a nurse immediately!",
    "I need urgent medical attention!",
)
"""Pre-recorded emergency message strings cycled round-robin on each trigger."""
