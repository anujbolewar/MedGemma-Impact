"""
encoder/intent_encoder.py — Gaze sequence → IntentToken packet encoder.

Maps successive FusedSignalFrame readings through a 3×3 directional grid
to token categories, cycles through tokens on blinks, and finalises the
selection when the patient holds centre gaze for a confirm dwell.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from core.constants import SentinelConstants as C
from core.logger import get_logger
from encoder.token_vocab import (
    VOCAB,
    IntentToken,
    get_by_category,
    tokens_to_prompt_string,
    validate_sequence,
)
from input.signal_fuser import FusedSignalFrame

_log = get_logger()

# ──────────────────────────────────────────────────────────────
# Grid layout: direction → token category
# ──────────────────────────────────────────────────────────────
#
#  UP-LEFT    │   UP      │ UP-RIGHT
#   BODY      │ SENSATION │ INTENSITY
# ────────────┼───────────┼──────────
#   LEFT      │  CENTRE   │  RIGHT
#   NEEDS     │ COGNITIVE │ EMOTIONAL
# ────────────┼───────────┼──────────
#  DOWN-LEFT  │   DOWN    │ DOWN-RIGHT
#  EMERGENCY  │ TEMPORAL  │ MODIFIERS

_DIRECTION_TO_CATEGORY: dict[str, str] = {
    "UP-LEFT":    "BODY",
    "UP":         "SENSATION",
    "UP-RIGHT":   "INTENSITY",
    "LEFT":       "NEEDS",
    "CENTRE":     "COGNITIVE",
    "RIGHT":      "EMOTIONAL",
    "DOWN-LEFT":  "EMERGENCY",
    "DOWN":       "TEMPORAL",
    "DOWN-RIGHT": "MODIFIERS",
}

# Category tokens in stable display order (for cycling via blinks)
_CATEGORY_TOKENS: dict[str, list[IntentToken]] = {
    cat: get_by_category(cat) for cat in _DIRECTION_TO_CATEGORY.values()
}

# Direction strings used in confirm-dwell detection
_CENTRE_DIRECTION = "CENTRE"

# Minimum number of consecutive centre frames to count as a confirm dwell
# GAZE_DWELL_MS / assumed tick rate (≈50ms): 1500/50 = 30 frames minimum
_CONFIRM_DWELL_FRAMES: int = max(1, int(C.GAZE_DWELL_MS / 50))


# ──────────────────────────────────────────────────────────────
# Output dataclass
# ──────────────────────────────────────────────────────────────

@dataclass
class IntentPacket:
    """
    Finalised output of an encoding session.

    Attributes:
        tokens: Ordered list of selected :class:`~encoder.token_vocab.IntentToken`.
        token_codes: List of token code strings (mirrors tokens).
        prompt_string: Comma-separated natural hints ready for LLM injection.
        selection_duration_ms: Wall time from first frame to finalisation.
        confidence: Mean composite_confidence from all input frames.
        timestamp_ms: Milliseconds since epoch at moment of finalisation.
    """

    tokens: list[IntentToken]
    token_codes: list[str]
    prompt_string: str
    selection_duration_ms: float
    confidence: float
    timestamp_ms: float

    def to_dict(self) -> dict:
        """
        Serialise to a JSON-safe dict for structured logging.

        Returns:
            Dict with all fields; ``tokens`` is replaced by ``token_codes``.
        """
        return {
            "token_codes": self.token_codes,
            "prompt_string": self.prompt_string,
            "selection_duration_ms": round(self.selection_duration_ms, 2),
            "confidence": round(self.confidence, 4),
            "timestamp_ms": round(self.timestamp_ms, 2),
            "token_count": len(self.tokens),
        }


# ──────────────────────────────────────────────────────────────
# IntentEncoder
# ──────────────────────────────────────────────────────────────

class IntentEncoder:
    """
    Stateful encoder that accumulates gaze frames into an :class:`IntentPacket`.

    Interaction model
    -----------------
    1. Patient gazes at a grid zone → that zone's category is *hovered*.
    2. A short blink cycles to the next token within that category.
       Then selects the currently highlighted token into the pending list.
    3. Gazing at CENTRE for ``GAZE_DWELL_MS`` confirms the full selection,
       producing an :class:`IntentPacket`.
    4. Calling :meth:`reset` clears state without emitting a packet.

    Args:
        vocab: The full token code → :class:`IntentToken` dict (from
               :data:`~encoder.token_vocab.VOCAB`). Defaults to the module
               constant if not provided.
    """

    def __init__(self, vocab: Optional[dict[str, IntentToken]] = None) -> None:
        """Initialise to empty selection state."""
        self._vocab = vocab if vocab is not None else VOCAB
        self._selected: list[IntentToken] = []
        self._category_cursors: dict[str, int] = {
            cat: 0 for cat in _CATEGORY_TOKENS
        }
        self._start_ms: float = time.monotonic() * 1000.0
        self._centre_frame_count: int = 0
        self._last_direction: str = ""
        self._confidences: list[float] = []

        _log.info("intent_encoder", "init", {
            "vocab_size": len(self._vocab),
            "confirm_dwell_frames": _CONFIRM_DWELL_FRAMES,
        })

    # ──────────────────────────────────────────
    # Primary API
    # ──────────────────────────────────────────

    def encode(self, fused_frames: list[FusedSignalFrame]) -> Optional[IntentPacket]:
        """
        Process a batch of :class:`~input.signal_fuser.FusedSignalFrame` readings.

        This is the main method — call it on every tick with the latest batch
        of fused frames. Returns an :class:`IntentPacket` only when a confirm
        dwell is detected; returns ``None`` otherwise.

        Args:
            fused_frames: One or more fused signal frames from the current tick.

        Returns:
            A finalised :class:`IntentPacket` if the session was confirmed,
            or ``None`` if still in progress.
        """
        if not fused_frames:
            return None

        for frame in fused_frames:
            self._confidences.append(frame.composite_confidence)
            packet = self._process_frame(frame)
            if packet is not None:
                return packet

        return None

    def reset(self) -> None:
        """
        Clear the current selection and restart the encoding session.

        Does not emit an :class:`IntentPacket`. Call this after a confirmed
        packet has been handed to the pipeline, or on user cancel.
        """
        self._selected.clear()
        self._category_cursors = {cat: 0 for cat in _CATEGORY_TOKENS}
        self._start_ms = time.monotonic() * 1000.0
        self._centre_frame_count = 0
        self._last_direction = ""
        self._confidences.clear()
        _log.info("intent_encoder", "reset", {})

    def get_current_selection(self) -> list[IntentToken]:
        """
        Return a copy of the currently accumulated (unconfirmed) token list.

        Returns:
            List of :class:`~encoder.token_vocab.IntentToken` objects selected
            so far in the current session.
        """
        return list(self._selected)

    def current_category(self, direction: str) -> Optional[str]:
        """
        Return the category name for a given gaze direction.

        Args:
            direction: Direction string from a :class:`~input.signal_fuser.FusedSignalFrame`.

        Returns:
            Category string, or ``None`` if direction is unmapped.
        """
        return _DIRECTION_TO_CATEGORY.get(direction)

    def highlighted_token(self, direction: str) -> Optional[IntentToken]:
        """
        Return the token currently highlighted (under cursor) in a zone.

        Args:
            direction: Current gaze direction.

        Returns:
            The :class:`~encoder.token_vocab.IntentToken` at the cursor
            position for that direction's category, or ``None``.
        """
        cat = _DIRECTION_TO_CATEGORY.get(direction)
        if cat is None:
            return None
        tokens = _CATEGORY_TOKENS[cat]
        cursor = self._category_cursors.get(cat, 0)
        return tokens[cursor % len(tokens)] if tokens else None

    # ──────────────────────────────────────────
    # Internal frame processor
    # ──────────────────────────────────────────

    def _process_frame(self, frame: FusedSignalFrame) -> Optional[IntentPacket]:
        """
        Handle a single frame: update hover state, detect blinks, detect confirm.

        Args:
            frame: A single :class:`~input.signal_fuser.FusedSignalFrame`.

        Returns:
            :class:`IntentPacket` if confirmed this frame, else ``None``.
        """
        direction = frame.primary_direction

        # ── Confirm dwell detection ───────────────────────────
        if direction == _CENTRE_DIRECTION:
            self._centre_frame_count += 1
            if self._centre_frame_count >= _CONFIRM_DWELL_FRAMES:
                return self._finalise(frame)
        else:
            self._centre_frame_count = 0

        # ── Blink detection (EAR-based via gaze_confidence drop) ─
        # A short, very low confidence reading (< 0.3) signals a blink.
        # The blink selects the currently highlighted token in the
        # hovered category, then advances that category's cursor.
        if self._is_blink(frame) and direction != _CENTRE_DIRECTION:
            token = self.highlighted_token(direction)
            if token:
                self._select_token(token)
                cat = _DIRECTION_TO_CATEGORY[direction]
                tokens = _CATEGORY_TOKENS[cat]
                self._category_cursors[cat] = (
                    self._category_cursors[cat] + 1
                ) % max(len(tokens), 1)

        self._last_direction = direction
        return None

    def _is_blink(self, frame: FusedSignalFrame) -> bool:
        """
        Infer a blink from a sudden gaze confidence drop.

        A confidence value below 0.30 (eye partially/fully closed) on a
        frame where the previous direction was the same zone is treated as
        a deliberate gaze blink selection gesture.

        Args:
            frame: Current gaze frame.

        Returns:
            True if this frame represents a blink selection event.
        """
        return (
            frame.gaze_confidence < 0.30
            and frame.primary_direction == self._last_direction
        )

    def _select_token(self, token: IntentToken) -> None:
        """
        Add a token to the current selection and log the event.

        If the same token is already the most recent selection, it is
        not duplicated (prevents double-blink artefacts).

        Args:
            token: Token to append.
        """
        if self._selected and self._selected[-1].code == token.code:
            return  # Deduplicate consecutive identical selections
        self._selected.append(token)
        _log.info("intent_encoder", "token_selected", {
            "code": token.code,
            "category": token.category,
            "natural_hint": token.natural_hint,
            "selection_count": len(self._selected),
        })

    def _finalise(self, frame: FusedSignalFrame) -> Optional[IntentPacket]:
        """
        Finalise the session into an :class:`IntentPacket` on confirm dwell.

        Validates the sequence, builds the packet, resets internal state,
        and logs the full packet summary.

        Args:
            frame: The frame that triggered the confirm dwell.

        Returns:
            :class:`IntentPacket` if the sequence is valid, ``None`` otherwise.
        """
        if not self._selected:
            _log.warn("intent_encoder", "confirm_empty", {
                "reason": "No tokens selected — ignoring confirm dwell"
            })
            self._centre_frame_count = 0
            return None

        if not validate_sequence(self._selected):
            _log.warn("intent_encoder", "confirm_invalid", {
                "token_codes": [t.code for t in self._selected],
                "reason": "Only modifier/cognitive tokens — sequence rejected",
            })
            self._centre_frame_count = 0
            return None

        now_ms = time.monotonic() * 1000.0
        duration_ms = now_ms - self._start_ms
        mean_conf = (
            sum(self._confidences) / len(self._confidences)
            if self._confidences else frame.composite_confidence
        )

        packet = IntentPacket(
            tokens=list(self._selected),
            token_codes=[t.code for t in self._selected],
            prompt_string=tokens_to_prompt_string(self._selected),
            selection_duration_ms=round(duration_ms, 2),
            confidence=round(mean_conf, 4),
            timestamp_ms=round(now_ms, 2),
        )

        _log.info("intent_encoder", "packet_finalised", packet.to_dict())

        # Reset for next utterance
        self.reset()
        return packet
