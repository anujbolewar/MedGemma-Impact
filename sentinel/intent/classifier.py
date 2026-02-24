"""
sentinel/intent/classifier.py — Stateful intent bundle accumulator.

Maps symbol selections into a structured IntentBundle with four clinical
dimensions: BODY_PART, SENSATION, URGENCY, INTENSITY. Emits a complete bundle
when all required fields are filled, ready for LLM sentence reconstruction.
"""

from __future__ import annotations

import logging
import time
from copy import copy
from dataclasses import dataclass, field
from typing import Optional

from sentinel.intent.symbol_board import Symbol

logger = logging.getLogger(__name__)

# Required categories to form a complete intent
_REQUIRED_CATEGORIES: frozenset[str] = frozenset(
    {"BODY_PART", "SENSATION", "URGENCY", "INTENSITY"}
)


@dataclass
class IntentBundle:
    """
    A structured representation of a patient's communicative intent.

    All fields are optional until the bundle is complete. The classifier
    accumulates these values as the patient selects symbols.

    Attributes:
        body_part: Body location (e.g. "chest").
        sensation: Feeling type (e.g. "pressure").
        urgency: Temporal urgency (e.g. "right now").
        intensity: Severity level (e.g. "moderate").
        timestamp: Monotonic time when the bundle was last updated.
        history: Ordered list of (category, value) selections in this session.
    """

    body_part: Optional[str] = None
    sensation: Optional[str] = None
    urgency: Optional[str] = None
    intensity: Optional[str] = None
    timestamp: float = field(default_factory=time.monotonic)
    history: list[tuple[str, str]] = field(default_factory=list)

    def is_complete(self) -> bool:
        """
        Return True if all four required intent dimensions are filled.

        Returns:
            True when body_part, sensation, urgency, and intensity are all set.
        """
        return all(
            [
                self.body_part is not None,
                self.sensation is not None,
                self.urgency is not None,
                self.intensity is not None,
            ]
        )

    def to_prompt_tokens(self) -> dict[str, str]:
        """
        Serialise the bundle into a flat dict of token keys and values.

        Used by :class:`~sentinel.llm.prompt_builder.PromptBuilder` to fill
        the prompt template. Falls back to "unknown" for any missing field
        so prompting is always safe even with partial bundles.

        Returns:
            Dict with keys: body_part, sensation, urgency, intensity.
        """
        return {
            "body_part": self.body_part or "unknown",
            "sensation": self.sensation or "unknown",
            "urgency": self.urgency or "unknown",
            "intensity": self.intensity or "unknown",
        }

    def missing_categories(self) -> list[str]:
        """
        Return the list of category names still needed to complete the bundle.

        Returns:
            List of category strings (e.g. ['URGENCY', 'INTENSITY']).
        """
        missing = []
        if self.body_part is None:
            missing.append("BODY_PART")
        if self.sensation is None:
            missing.append("SENSATION")
        if self.urgency is None:
            missing.append("URGENCY")
        if self.intensity is None:
            missing.append("INTENSITY")
        return missing


class IntentClassifier:
    """
    Stateful accumulator that builds an :class:`IntentBundle` from symbol selections.

    Each call to :meth:`update` adds one symbol value to the appropriate
    bundle dimension. When :meth:`is_complete` returns True, the bundle is
    ready to be sent to the LLM. Calling :meth:`reset` starts a fresh session.
    """

    def __init__(self) -> None:
        """Initialise with an empty intent bundle."""
        self._bundle = IntentBundle()
        logger.debug("IntentClassifier initialised")

    def update(self, symbol: Symbol) -> IntentBundle:
        """
        Incorporate a symbol selection into the current intent bundle.

        If the symbol's category is already filled, the new value overwrites
        the previous one (patient can correct their selection).

        Args:
            symbol: The :class:`~sentinel.intent.symbol_board.Symbol` selected.

        Returns:
            A copy of the updated :class:`IntentBundle`.
        """
        category = symbol.category
        value = symbol.value

        # Record selection history regardless of category
        self._bundle.history.append((category, value))
        self._bundle = IntentBundle(
            body_part=value if category == "BODY_PART" else self._bundle.body_part,
            sensation=value if category == "SENSATION" else self._bundle.sensation,
            urgency=value if category == "URGENCY" else self._bundle.urgency,
            intensity=value if category == "INTENSITY" else self._bundle.intensity,
            timestamp=time.monotonic(),
            history=list(self._bundle.history),
        )

        logger.info(
            "Intent updated: %s=%r | complete=%s | missing=%s",
            category,
            value,
            self._bundle.is_complete(),
            self._bundle.missing_categories(),
        )
        return copy(self._bundle)

    def is_complete(self) -> bool:
        """
        Return True if the current bundle has all required intent dimensions.

        Returns:
            True when all of BODY_PART, SENSATION, URGENCY, INTENSITY are set.
        """
        return self._bundle.is_complete()

    def reset(self) -> None:
        """
        Clear the current bundle to start a fresh communication session.

        Call this after the bundle has been sent to the LLM and the sentence
        has been spoken.
        """
        logger.info("IntentClassifier: bundle reset")
        self._bundle = IntentBundle()

    @property
    def current_bundle(self) -> IntentBundle:
        """
        Return a copy of the current (possibly incomplete) intent bundle.

        Returns:
            A copy of the current :class:`IntentBundle`.
        """
        return copy(self._bundle)

    def missing_categories(self) -> list[str]:
        """
        Return which categories still need to be selected.

        Returns:
            List of category name strings that are not yet filled.
        """
        return self._bundle.missing_categories()
