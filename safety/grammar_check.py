"""
safety/grammar_check.py — Grammar validation for NeuroWeave Sentinel LLM outputs.

Uses language-tool-python in offline mode (local JRE server) to validate
reconstructed patient communication sentences. Falls back to structural checks
if LanguageTool is unavailable. Lazily initialised on first check() call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from core.constants import SentinelConstants as C
from core.logger import get_logger

_log = get_logger()

# ──────────────────────────────────────────────────────────────
# Medical / care plausibility vocabulary (200 words)
# ──────────────────────────────────────────────────────────────
# A sentence must contain at least 2 words from this set to pass
# the plausibility heuristic. Words are lowercase for fast lookup.

MEDICAL_VOCAB: frozenset[str] = frozenset({
    # Body regions
    "head", "neck", "chest", "back", "abdomen", "stomach", "arm", "leg",
    "foot", "feet", "hand", "shoulder", "hip", "knee", "ankle", "wrist",
    "spine", "throat", "mouth", "eye", "eyes", "ear", "ears", "skin",
    "body", "torso", "pelvis", "groin", "buttocks", "tailbone", "rib",
    "ribs", "collarbone", "jaw", "tongue", "teeth", "scalp", "temple",
    "forehead", "cheek", "finger", "fingers", "toe", "toes", "elbow",
    # Sensations / symptoms
    "pain", "ache", "aching", "hurting", "hurt", "sore", "soreness",
    "pressure", "tight", "tightness", "burning", "tingling", "numb",
    "numbness", "nausea", "nauseated", "dizzy", "dizziness", "weak",
    "weakness", "fatigue", "tired", "exhausted", "itching", "itch",
    "swollen", "swelling", "cramp", "cramping", "stiff", "stiffness",
    "spasm", "throbbing", "pulsing", "stabbing", "twisting", "bloated",
    "bloating", "tenderness", "sensitive", "raw", "hot", "cold",
    "shaking", "trembling", "twitching",
    # Intensity / descriptors
    "mild", "moderate", "severe", "unbearable", "intense", "sharp",
    "dull", "constant", "intermittent", "sudden", "gradual", "extreme",
    "chronic", "acute", "significant",
    # Temporal
    "now", "today", "recently", "ongoing", "worsening", "started",
    "since", "worse", "better", "improving", "persistent", "beginning",
    "worsened", "increased", "decreased",
    # Needs / requests
    "water", "thirsty", "food", "hungry", "toilet", "bathroom",
    "medication", "medicine", "nurse", "doctor", "help", "rest",
    "sleep", "reposition", "uncomfortable", "family", "call", "urgent",
    "emergency", "immediately", "carer", "caregiver", "pillow", "blanket",
    # Cognitive / communicative
    "yes", "no", "understand", "confused", "unsure", "repeat", "please",
    "thank", "could", "would", "cannot", "agree", "disagree", "correct",
    # Emotional
    "anxious", "calm", "distressed", "okay", "worried", "afraid",
    "sad", "happy", "scared", "frustrated", "relieved", "upset",
    "depressed", "hopeful",
    # Clinical / physiological
    "breathing", "breath", "breathless", "swallowing", "speaking",
    "coughing", "bleeding", "catheter", "tube", "oxygen", "monitor",
    "temperature", "fever", "blood", "pressure", "pulse", "heartbeat",
    "heartrate", "circulation", "infection", "wound", "scar", "bruise",
    "rash", "allergy", "seizure", "stroke",
    # Linking / grammatical (clinical context)
    "feel", "feeling", "have", "having", "need", "want",
    "am", "is", "are", "was", "seems", "appears", "experiencing",
    "experience", "getting", "become", "becoming",
    # Location qualifiers
    "left", "right", "side", "lower", "upper", "whole", "general",
    "area", "part", "inner", "outer", "deep", "surface",
})

# Minimum matches required for plausibility
_MIN_VOCAB_MATCHES: int = 2

# Punctuation pattern for sentence-final check
_TERMINAL_PUNCT = re.compile(r"[.!?]$")

# Capital letter at start
_LEADING_CAPITAL = re.compile(r"^[A-Z]")


# ──────────────────────────────────────────────────────────────
# Output dataclass
# ──────────────────────────────────────────────────────────────

@dataclass
class GrammarReport:
    """
    Result of a grammar validation pass.

    Attributes:
        error_count: Number of grammar errors (WHITESPACE errors excluded).
        errors: Human-readable error message strings.
        word_count: Number of whitespace-separated words.
        passes: True when error_count ≤ GRAMMAR_MAX_ERRORS and 2 ≤ word_count ≤ 30.
        plausibility_score: Float in [0, 1] reflecting medical vocabulary coverage.
        has_terminal_punct: True if the sentence ends with ``.``, ``!``, or ``?``.
        starts_with_capital: True if the sentence starts with an uppercase letter.
        cannot_evaluate: True if grammar checking failed entirely.
    """

    error_count: int = 0
    errors: list[str] = field(default_factory=list)
    word_count: int = 0
    passes: bool = False
    plausibility_score: float = 0.0
    has_terminal_punct: bool = False
    starts_with_capital: bool = False
    cannot_evaluate: bool = False

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict for structured logging."""
        return {
            "error_count": self.error_count,
            "word_count": self.word_count,
            "errors": self.errors[:5],   # cap for log readability
            "passes": self.passes,
            "plausibility_score": round(self.plausibility_score, 3),
            "has_terminal_punct": self.has_terminal_punct,
            "starts_with_capital": self.starts_with_capital,
            "cannot_evaluate": self.cannot_evaluate,
        }


# ──────────────────────────────────────────────────────────────
# GrammarChecker
# ──────────────────────────────────────────────────────────────

class GrammarChecker:
    """
    Grammar validator for NeuroWeave Sentinel reconstructed sentences.

    Lazily initialises a LanguageTool server (offline, local JRE) on the
    first :meth:`check` call. If LanguageTool cannot be started (missing JRE
    or library), the checker falls back to a structural-only validation.

    Args:
        max_errors: Maximum permitted grammar errors (default from constants).
        min_words: Minimum word count for a valid sentence.
        max_words: Maximum word count for a valid sentence.
    """

    def __init__(
        self,
        max_errors: int = C.GRAMMAR_MAX_ERRORS,
        min_words: int = 2,
        max_words: int = C.MAX_SENTENCE_TOKENS,
    ) -> None:
        """Initialise configuration without starting LanguageTool yet."""
        self._max_errors = max_errors
        self._min_words = min_words
        self._max_words = max_words
        self._lt: Optional[object] = None
        self._lt_failed = False     # If True, stop retrying LanguageTool

        _log.info("grammar_check", "init", {
            "max_errors": max_errors,
            "min_words": min_words,
            "max_words": max_words,
            "lt_mode": "lazy",
        })

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def check(self, text: str) -> GrammarReport:
        """
        Validate the given sentence and return a :class:`GrammarReport`.

        Checks performed:
        1. Word count within [``min_words``, ``max_words``].
        2. Starts with a capital letter.
        3. Ends with terminal punctuation.
        4. LanguageTool grammar check (WHITESPACE errors excluded).
        5. Medical vocabulary plausibility heuristic.

        Args:
            text: Reconstructed patient communication sentence.

        Returns:
            A fully populated :class:`GrammarReport`.
        """
        if not text or not text.strip():
            return GrammarReport(cannot_evaluate=True, passes=False)

        text = text.strip()
        words = text.split()
        word_count = len(words)

        has_terminal = bool(_TERMINAL_PUNCT.search(text))
        has_capital = bool(_LEADING_CAPITAL.match(text))
        plausibility = self._plausibility(words)

        # ── LanguageTool grammar check ────────────────────────
        errors: list[str] = []
        error_count = 0
        lt_available = self._ensure_lt()

        if lt_available and self._lt is not None:
            try:
                matches = self._lt.check(text)  # type: ignore[attr-defined]
                for m in matches:
                    # Skip whitespace-only errors
                    category = (
                        getattr(m, "category", None)
                        or getattr(m, "ruleIssueType", "")
                    )
                    if isinstance(category, str) and category.upper() == "WHITESPACE":
                        continue
                    # Safely extract fields across library versions
                    rule_id = getattr(m, "ruleId", None) or getattr(m, "rule_id", "RULE")
                    message = getattr(m, "message", None) or str(m)
                    offset  = getattr(m, "offset", "?")
                    err_len = getattr(m, "errorLength", None) or getattr(m, "matchedlength", "?")
                    errors.append(
                        f"[{rule_id}] {message} "
                        f"(offset {offset}, len {err_len})"
                    )
                error_count = len(errors)
            except Exception as exc:  # noqa: BLE001
                _log.warn("grammar_check", "lt_check_failed", {"error": str(exc)})
                # Fall through with 0 errors — structural checks still apply

        # ── Structural error heuristics (always applied) ──────
        structural_errors = self._structural_checks(text, words)
        # Structural errors count separately, not added to LT count
        all_error_msgs = errors + structural_errors
        total_errors = error_count + len(structural_errors)

        passes = (
            total_errors <= self._max_errors
            and self._min_words <= word_count <= self._max_words
        )

        report = GrammarReport(
            error_count=total_errors,
            errors=all_error_msgs,
            word_count=word_count,
            passes=passes,
            plausibility_score=round(plausibility, 3),
            has_terminal_punct=has_terminal,
            starts_with_capital=has_capital,
            cannot_evaluate=False,
        )

        level = "info" if passes else "warn"
        getattr(_log, level)("grammar_check", "evaluated", report.to_dict())
        return report

    def close(self) -> None:
        """
        Close the LanguageTool server process and release resources.

        Safe to call multiple times. Should be called on application shutdown
        to avoid orphaned JRE processes.
        """
        if self._lt is not None:
            try:
                self._lt.close()  # type: ignore[attr-defined]
                _log.info("grammar_check", "lt_closed", {})
            except Exception as exc:  # noqa: BLE001
                _log.warn("grammar_check", "lt_close_error", {"error": str(exc)})
            finally:
                self._lt = None

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _ensure_lt(self) -> bool:
        """
        Lazily initialise the LanguageTool server on first call.

        Sets ``self._lt_failed = True`` if initialisation fails so we
        only attempt startup once.

        Returns:
            True if LanguageTool is ready, False if unavailable.
        """
        if self._lt is not None:
            return True
        if self._lt_failed:
            return False

        try:
            import language_tool_python  # type: ignore
            self._lt = language_tool_python.LanguageTool(
                "en-US",
                remote_server=None,  # offline / local JRE only
            )
            _log.info("grammar_check", "lt_started", {"lang": "en-US"})
            return True
        except Exception as exc:  # noqa: BLE001
            self._lt_failed = True
            _log.warn("grammar_check", "lt_unavailable", {
                "error": str(exc),
                "fallback": "structural checks only",
            })
            return False

    def _structural_checks(self, text: str, words: list[str]) -> list[str]:
        """
        Lightweight structural error heuristics that run regardless of LT.

        Checks:
        - Missing terminal punctuation
        - Missing leading capital letter
        - Excessive word repetition (3+ consecutive identical words)

        Args:
            text: Full sentence string.
            words: Pre-split word list.

        Returns:
            List of error description strings.
        """
        errs: list[str] = []
        if not _TERMINAL_PUNCT.search(text):
            errs.append("Missing terminal punctuation (. ! ?)")
        if not _LEADING_CAPITAL.match(text):
            errs.append("Sentence does not start with a capital letter")
        # Check for 3+ consecutive identical words
        lw = [w.lower().strip(".,!?") for w in words]
        for i in range(len(lw) - 2):
            if lw[i] == lw[i + 1] == lw[i + 2]:
                errs.append(f"Repeated word: '{lw[i]}' (possible hallucination)")
                break
        return errs

    def _plausibility(self, words: list[str]) -> float:
        """
        Compute a medical vocabulary plausibility score.

        Normalises the number of recognised medical/care vocabulary matches
        against :data:`_MIN_VOCAB_MATCHES` as the expected baseline.
        Capped at 1.0.

        Args:
            words: Whitespace-split word tokens from the sentence.

        Returns:
            Float in ``[0.0, 1.0]`` — 1.0 means fully plausible.
        """
        if not words:
            return 0.0
        clean = [w.lower().strip(".,!?;:'\"") for w in words]
        matches = sum(1 for w in clean if w in MEDICAL_VOCAB)
        # Score: 0.0 = no matches, 1.0 = at least _MIN_VOCAB_MATCHES
        return min(1.0, matches / max(_MIN_VOCAB_MATCHES, 1))
