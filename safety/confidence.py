"""
safety/confidence.py — Master safety gate for NeuroWeave Sentinel.

Combines token-level entropy, grammar quality, semantic plausibility, and
gaze-input confidence into a composite score, then maps that score to a
:class:`SafetyAction` decision that the pipeline uses to determine whether to
voice, confirm, fall back, or block the reconstructed sentence.

Decision ladder::

    composite ≥ 0.75            → PROCEED  (voice immediately)
    0.60 ≤ composite < 0.75     → CONFIRM  (ask patient to confirm)
    0.40 ≤ composite < 0.60     → FALLBACK (use template sentence)
    composite < 0.40            → BLOCK    (suppress output entirely)

Hard-block overrides (checked before composite score):
    - ``result.is_timeout``             → FALLBACK
    - ``word_count < MIN_SENTENCE_TOKENS`` → BLOCK
    - ``word_count > MAX_SENTENCE_TOKENS`` → BLOCK
"""

from __future__ import annotations

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List

from core.constants import C
from core.logger import get_logger
from llm.reconstructor import IntentPacket, ReconstructionResult
from safety.entropy_guard import EntropyGuard, EntropyReport
from safety.grammar_check import GrammarChecker, GrammarReport

log = get_logger()

# ── Composite-score weights (must sum to 1.0) ─────────────────────────────────
_W_ENTROPY = 0.35
_W_GRAMMAR = 0.25
_W_PLAUSIBILITY = 0.20
_W_INPUT = 0.20


# ── SafetyAction ──────────────────────────────────────────────────────────────

class SafetyAction(Enum):
    """
    Outcome of a :class:`SafetyGate` evaluation.

    Values:
        PROCEED:  Composite confidence ≥ 0.75 — voice the sentence immediately.
        CONFIRM:  0.60 ≤ composite < 0.75 — ask the patient to confirm before voicing.
        FALLBACK: 0.40 ≤ composite < 0.60, or timeout — replace with template sentence.
        BLOCK:    composite < 0.40, or sentence too short/long — suppress output entirely.
    """

    PROCEED = "PROCEED"
    CONFIRM = "CONFIRM"
    FALLBACK = "FALLBACK"
    BLOCK = "BLOCK"


# ── SafetyDecision ────────────────────────────────────────────────────────────

@dataclass
class SafetyDecision:
    """
    Complete result of a :class:`SafetyGate` evaluation.

    Attributes:
        action:               The recommended pipeline action.
        composite_confidence: Weighted composite score in ``[0, 1]``.
        entropy_report:       Full :class:`~safety.entropy_guard.EntropyReport`.
        grammar_report:       Full :class:`~safety.grammar_check.GrammarReport`.
        reasons:              Human-readable list of reasons for the decision.
        latency_ms:           Wall-clock time consumed by :meth:`SafetyGate.evaluate`.
    """

    action: SafetyAction
    composite_confidence: float
    entropy_report: EntropyReport
    grammar_report: GrammarReport
    reasons: List[str]
    latency_ms: float


# ── SafetyGate ────────────────────────────────────────────────────────────────

class SafetyGate:
    """
    Master safety gate for the NeuroWeave Sentinel output pipeline.

    Accepts a :class:`~llm.reconstructor.ReconstructionResult` and the
    originating :class:`~llm.reconstructor.IntentPacket`, runs entropy and
    grammar checks, computes a four-component composite confidence score, and
    returns a :class:`SafetyDecision`.

    Composite formula::

        entropy_score     = 1.0 − entropy_report.mean_entropy   (0.5 if cannot_evaluate)
        grammar_score     = clamp(1.0 − error_count / 5.0, 0, 1)
        plausibility      = grammar_report.plausibility_score
        input_confidence  = packet.confidence

        composite = (entropy_score     × 0.35
                   + grammar_score     × 0.25
                   + plausibility      × 0.20
                   + input_confidence  × 0.20)

    Hard-block overrides (evaluated before composite score):
        - ``result.is_timeout``                        → FALLBACK
        - ``word_count < C.MIN_SENTENCE_TOKENS``       → BLOCK
        - ``word_count > C.MAX_SENTENCE_TOKENS``       → BLOCK

    Args:
        entropy_guard:   Configured :class:`~safety.entropy_guard.EntropyGuard`.
        grammar_checker: Configured :class:`~safety.grammar_check.GrammarChecker`.

    Example::

        gate = SafetyGate(EntropyGuard(), GrammarChecker())
        decision = gate.evaluate(reconstruction_result, intent_packet)
        if decision.action is SafetyAction.PROCEED:
            tts.speak(reconstruction_result.text)
    """

    def __init__(
        self,
        entropy_guard: EntropyGuard,
        grammar_checker: GrammarChecker,
    ) -> None:
        self._entropy_guard = entropy_guard
        self._grammar_checker = grammar_checker

    # ── Public API ────────────────────────────────────────────────────────

    def evaluate(
        self,
        result: ReconstructionResult,
        packet: IntentPacket,
    ) -> SafetyDecision:
        """
        Run all safety checks and return a :class:`SafetyDecision`.

        The full decision (action, composite score, sub-reports, reasons) is
        logged as JSON via :func:`~core.logger.get_logger` on every call.

        Args:
            result: Output of :class:`~llm.reconstructor.SentenceReconstructor`.
            packet: The intent packet that produced *result*.

        Returns:
            A :class:`SafetyDecision` with all component details.
        """
        t0 = time.perf_counter()
        reasons: List[str] = []

        # ── 1. Run sub-checks ─────────────────────────────────────────────
        entropy_report: EntropyReport = self._entropy_guard.evaluate(result)
        grammar_report: GrammarReport = self._grammar_checker.check(result.text)

        # ── 2. Compute component scores ───────────────────────────────────
        if entropy_report.cannot_evaluate:
            entropy_score = 0.5
            reasons.append("entropy_cannot_evaluate: defaulting entropy_score to 0.5")
        else:
            entropy_score = max(0.0, min(1.0, 1.0 - entropy_report.mean_entropy))

        grammar_score = max(0.0, min(1.0, 1.0 - grammar_report.error_count / 5.0))
        plausibility = grammar_report.plausibility_score
        input_confidence = packet.confidence

        composite = (
            entropy_score    * _W_ENTROPY
            + grammar_score  * _W_GRAMMAR
            + plausibility   * _W_PLAUSIBILITY
            + input_confidence * _W_INPUT
        )
        composite = round(min(max(composite, 0.0), 1.0), 4)

        # ── 3. Hard-block overrides ───────────────────────────────────────
        if result.is_timeout:
            reasons.append("result_is_timeout: inference exceeded latency budget")
            action = SafetyAction.FALLBACK
        elif grammar_report.word_count < C.MIN_SENTENCE_TOKENS:
            reasons.append(
                f"word_count_too_low: {grammar_report.word_count} < {C.MIN_SENTENCE_TOKENS}"
            )
            action = SafetyAction.BLOCK
        elif grammar_report.word_count > C.MAX_SENTENCE_TOKENS:
            reasons.append(
                f"word_count_too_high: {grammar_report.word_count} > {C.MAX_SENTENCE_TOKENS}"
            )
            action = SafetyAction.BLOCK

        # ── 4. Composite-score ladder ─────────────────────────────────────
        else:
            if composite >= 0.75:
                action = SafetyAction.PROCEED
            elif composite >= 0.60:
                action = SafetyAction.CONFIRM
                reasons.append(
                    f"composite_confidence {composite:.3f} in CONFIRM band [0.60, 0.75)"
                )
            elif composite >= 0.40:
                action = SafetyAction.FALLBACK
                reasons.append(
                    f"composite_confidence {composite:.3f} in FALLBACK band [0.40, 0.60)"
                )
            else:
                action = SafetyAction.BLOCK
                reasons.append(
                    f"composite_confidence {composite:.3f} below BLOCK threshold 0.40"
                )

        # ── 5. Populate sub-check reasons ────────────────────────────────
        if not entropy_report.passes and not entropy_report.cannot_evaluate:
            reasons.append(
                f"entropy_failed: flagged_fraction={entropy_report.flagged_fraction:.3f}"
                f" (threshold={entropy_report.threshold:.3f})"
            )
        if not grammar_report.passes:
            reasons.append(
                f"grammar_failed: error_count={grammar_report.error_count}"
                f" (max={C.GRAMMAR_MAX_ERRORS})"
            )
        if result.is_fallback:
            reasons.append("result_is_template_fallback: LLM inference was not used")

        latency_ms = (time.perf_counter() - t0) * 1000.0

        decision = SafetyDecision(
            action=action,
            composite_confidence=composite,
            entropy_report=entropy_report,
            grammar_report=grammar_report,
            reasons=reasons,
            latency_ms=round(latency_ms, 2),
        )

        # ── 6. Structured log ─────────────────────────────────────────────
        log.perf(
            "safety",
            "gate_evaluate",
            latency_ms,
            {
                "action": action.value,
                "composite": composite,
                "entropy_score": round(entropy_score, 4),
                "grammar_score": round(grammar_score, 4),
                "plausibility": round(plausibility, 4),
                "input_confidence": round(input_confidence, 4),
                "entropy_report": entropy_report.to_dict(),
                "grammar_report": grammar_report.to_dict(),
                "is_timeout": result.is_timeout,
                "is_fallback": result.is_fallback,
                "reasons": reasons,
            },
        )

        return decision
