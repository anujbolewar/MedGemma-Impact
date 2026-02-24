"""
tests/test_safety.py — Pytest unit tests for the safety gate pipeline.

Tests cover EntropyGuard, GrammarChecker (mocked), and SafetyGate together.
LanguageTool is never started — the GrammarChecker is patched with a fixture
that returns controlled GrammarReport objects.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from core.constants import C
from llm.reconstructor import IntentPacket, ReconstructionResult
from safety.confidence import SafetyAction, SafetyDecision, SafetyGate
from safety.entropy_guard import EntropyGuard, EntropyReport
from safety.grammar_check import GrammarChecker, GrammarReport

# ──────────────────────────────────────────────────────────────
# Constants (mirror of confidence.py for readability)
# ──────────────────────────────────────────────────────────────
_W_ENTROPY      = 0.35
_W_GRAMMAR      = 0.25
_W_PLAUSIBILITY = 0.20
_W_INPUT        = 0.20

# Thresholds from constants.py
_MIN_WORDS = C.MIN_SENTENCE_TOKENS   # 3
_MAX_WORDS = C.MAX_SENTENCE_TOKENS   # 30


# ──────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────

def _make_result(
    text: str = "I feel chest pain now.",
    logits: torch.Tensor | None = None,
    is_timeout: bool = False,
    is_fallback: bool = False,
    latency_ms: float = 400.0,
) -> ReconstructionResult:
    """Create a minimal :class:`ReconstructionResult` for testing."""
    return ReconstructionResult(
        text=text,
        token_ids=[],
        logits=logits,
        latency_ms=latency_ms,
        is_timeout=is_timeout,
        is_fallback=is_fallback,
    )


def _make_packet(confidence: float = 0.85) -> IntentPacket:
    """Create a mock :class:`IntentPacket` with the given confidence."""
    return IntentPacket(tokens=["SENS_PAIN"], confidence=confidence)


def _make_grammar_report(
    error_count: int = 0,
    word_count: int = 6,
    plausibility: float = 1.0,
    passes: bool = True,
) -> GrammarReport:
    """Build a controlled GrammarReport without starting LanguageTool."""
    return GrammarReport(
        error_count=error_count,
        errors=["err"] * error_count,
        word_count=word_count,
        passes=passes,
        plausibility_score=plausibility,
        has_terminal_punct=True,
        starts_with_capital=True,
        cannot_evaluate=False,
    )


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture()
def real_entropy_guard() -> EntropyGuard:
    """Real EntropyGuard with default threshold (0.75)."""
    return EntropyGuard()


@pytest.fixture()
def mock_grammar_checker() -> MagicMock:
    """
    Patched GrammarChecker whose check() return value is set per-test.
    Starts with a clean OK report so tests opt-in to specific failures.
    """
    checker = MagicMock(spec=GrammarChecker)
    checker.check.return_value = _make_grammar_report()
    return checker


@pytest.fixture()
def gate(real_entropy_guard: EntropyGuard, mock_grammar_checker: MagicMock) -> SafetyGate:
    """SafetyGate wired with a real EntropyGuard + mocked GrammarChecker."""
    return SafetyGate(
        entropy_guard=real_entropy_guard,
        grammar_checker=mock_grammar_checker,
    )


# ──────────────────────────────────────────────────────────────
# Test 1 — High entropy → BLOCK or FALLBACK
# ──────────────────────────────────────────────────────────────

class TestEntropyHighBlock:
    """
    When every token position has near-uniform logits (entropy ≈ 1.0),
    the composite confidence drops low enough to BLOCK or FALLBACK.
    """

    def test_uniform_logits_give_low_action(
        self,
        gate: SafetyGate,
        mock_grammar_checker: MagicMock,
    ) -> None:
        """10-token uniform distribution → entropy ≈ 1.0 → low composite."""
        # Uniform over 100-token vocab → max entropy after normalisation
        logits = torch.zeros(8, 100)   # all logits equal → softmax uniform

        # Grammar clean, good word count, decent input confidence
        mock_grammar_checker.check.return_value = _make_grammar_report(
            error_count=0, word_count=6, plausibility=0.9, passes=True
        )
        result = _make_result(
            text="I feel severe chest pain today.",
            logits=logits,
        )
        packet = _make_packet(confidence=0.80)

        decision = gate.evaluate(result, packet)

        assert decision.action in (SafetyAction.BLOCK, SafetyAction.FALLBACK), (
            f"Expected BLOCK or FALLBACK with uniform logits, got {decision.action}"
        )
        # Entropy report must show very high entropy
        assert decision.entropy_report.mean_entropy > 0.9, (
            f"Expected mean_entropy > 0.9, got {decision.entropy_report.mean_entropy}"
        )

    def test_high_entropy_composite_below_threshold(
        self,
        gate: SafetyGate,
        mock_grammar_checker: MagicMock,
    ) -> None:
        """Composite must be < 0.60 when entropy ≈ 1.0."""
        logits = torch.zeros(5, 50)
        mock_grammar_checker.check.return_value = _make_grammar_report(
            word_count=5, plausibility=0.8
        )
        decision = gate.evaluate(
            _make_result(logits=logits),
            _make_packet(confidence=0.70),
        )
        # entropy_score ≈ 0.0, so composite  = 0.0×0.35 + 1.0×0.25 + 0.8×0.20 + 0.7×0.20
        #                                    = 0.0 + 0.25 + 0.16 + 0.14 = 0.55
        assert decision.composite_confidence < 0.60, (
            f"Expected composite < 0.60 with high entropy, got {decision.composite_confidence}"
        )

    def test_entropy_report_attached_to_decision(
        self,
        gate: SafetyGate,
        mock_grammar_checker: MagicMock,
    ) -> None:
        """SafetyDecision must carry the populated EntropyReport."""
        logits = torch.zeros(3, 10)
        mock_grammar_checker.check.return_value = _make_grammar_report()
        decision = gate.evaluate(_make_result(logits=logits), _make_packet())
        assert isinstance(decision.entropy_report, EntropyReport)
        assert not decision.entropy_report.cannot_evaluate


# ──────────────────────────────────────────────────────────────
# Test 2 — Low entropy → PROCEED
# ──────────────────────────────────────────────────────────────

class TestEntropyLowProceed:
    """
    When one logit per position is very large (near-certain prediction),
    normalised entropy ≈ 0 and composite confidence is high → PROCEED.
    """

    def _confident_logits(self, n_tokens: int = 6, vocab: int = 1000) -> torch.Tensor:
        """Return logits where position-0 token is 1000x more likely than any other."""
        t = torch.zeros(n_tokens, vocab)
        t[:, 0] = 20.0    # very high logit → softmax ≈ 1.0 at position 0
        return t

    def test_confident_logits_give_proceed(
        self,
        gate: SafetyGate,
        mock_grammar_checker: MagicMock,
    ) -> None:
        """Entropy ≈ 0, good grammar, plausibility = 1.0, confidence = 0.90 → PROCEED."""
        logits = self._confident_logits()
        mock_grammar_checker.check.return_value = _make_grammar_report(
            error_count=0, word_count=6, plausibility=1.0, passes=True
        )
        decision = gate.evaluate(
            _make_result(text="I feel severe chest pain now.", logits=logits),
            _make_packet(confidence=0.90),
        )
        assert decision.action is SafetyAction.PROCEED, (
            f"Expected PROCEED, got {decision.action} "
            f"(composite={decision.composite_confidence})"
        )
        assert decision.composite_confidence >= 0.75

    def test_entropy_near_zero(
        self,
        gate: SafetyGate,
        mock_grammar_checker: MagicMock,
    ) -> None:
        """Entropy report values must be close to 0 for confident logits."""
        logits = self._confident_logits(n_tokens=5, vocab=32000)
        mock_grammar_checker.check.return_value = _make_grammar_report()
        decision = gate.evaluate(_make_result(logits=logits), _make_packet())
        assert decision.entropy_report.mean_entropy < 0.05, (
            f"Expected mean_entropy < 0.05 for confident logits, "
            f"got {decision.entropy_report.mean_entropy}"
        )


# ──────────────────────────────────────────────────────────────
# Test 3 — Short sentence → BLOCK
# ──────────────────────────────────────────────────────────────

class TestGrammarShortSentence:
    """word_count < MIN_SENTENCE_TOKENS (3) must be hard-blocked."""

    @pytest.mark.parametrize("word_count", [0, 1, 2])
    def test_short_sentence_hard_block(
        self,
        gate: SafetyGate,
        mock_grammar_checker: MagicMock,
        word_count: int,
    ) -> None:
        mock_grammar_checker.check.return_value = _make_grammar_report(
            word_count=word_count,
            passes=False,
        )
        # Use confident logits so entropy alone would not block
        t = torch.zeros(3, 100); t[:, 0] = 20.0
        decision = gate.evaluate(
            _make_result(text="Pain.", logits=t),
            _make_packet(confidence=0.95),
        )
        assert decision.action is SafetyAction.BLOCK, (
            f"word_count={word_count}: expected BLOCK, got {decision.action}"
        )
        assert any("word_count_too_low" in r for r in decision.reasons), (
            f"Expected word_count_too_low in reasons: {decision.reasons}"
        )

    def test_min_boundary_does_not_block(
        self,
        gate: SafetyGate,
        mock_grammar_checker: MagicMock,
    ) -> None:
        """word_count == MIN_SENTENCE_TOKENS should NOT trigger the short-sentence block."""
        mock_grammar_checker.check.return_value = _make_grammar_report(
            word_count=_MIN_WORDS,     # exactly 3
            error_count=0,
            plausibility=1.0,
            passes=True,
        )
        t = torch.zeros(3, 100); t[:, 0] = 20.0
        decision = gate.evaluate(
            _make_result(logits=t),
            _make_packet(confidence=0.95),
        )
        assert decision.action is not SafetyAction.BLOCK or (
            "word_count_too_low" not in " ".join(decision.reasons)
        ), f"Boundary word_count={_MIN_WORDS} should not trigger short-sentence BLOCK"


# ──────────────────────────────────────────────────────────────
# Test 4 — Long sentence → BLOCK
# ──────────────────────────────────────────────────────────────

class TestGrammarLongSentence:
    """word_count > MAX_SENTENCE_TOKENS (30) must be hard-blocked."""

    @pytest.mark.parametrize("word_count", [31, 35, 50, 100])
    def test_long_sentence_hard_block(
        self,
        gate: SafetyGate,
        mock_grammar_checker: MagicMock,
        word_count: int,
    ) -> None:
        mock_grammar_checker.check.return_value = _make_grammar_report(
            word_count=word_count,
            passes=False,
        )
        t = torch.zeros(4, 100); t[:, 0] = 20.0
        decision = gate.evaluate(
            _make_result(text=" ".join(["word"] * word_count), logits=t),
            _make_packet(confidence=0.95),
        )
        assert decision.action is SafetyAction.BLOCK, (
            f"word_count={word_count}: expected BLOCK, got {decision.action}"
        )
        assert any("word_count_too_high" in r for r in decision.reasons), (
            f"Expected word_count_too_high in reasons: {decision.reasons}"
        )

    def test_max_boundary_does_not_block(
        self,
        gate: SafetyGate,
        mock_grammar_checker: MagicMock,
    ) -> None:
        """word_count == MAX_SENTENCE_TOKENS should NOT trigger the long-sentence block."""
        mock_grammar_checker.check.return_value = _make_grammar_report(
            word_count=_MAX_WORDS,   # exactly 30
            error_count=0,
            plausibility=1.0,
            passes=True,
        )
        t = torch.zeros(4, 100); t[:, 0] = 20.0
        decision = gate.evaluate(_make_result(logits=t), _make_packet(confidence=0.95))
        assert "word_count_too_high" not in " ".join(decision.reasons), (
            f"Boundary word_count={_MAX_WORDS} should not trigger long-sentence BLOCK"
        )


# ──────────────────────────────────────────────────────────────
# Test 5 — Composite threshold boundaries
# ──────────────────────────────────────────────────────────────

class TestConfidenceThresholdBoundary:
    """
    Exact composite values at the band boundaries must map to the
    correct SafetyAction.

    compositor formula (weights):
        entropy_score  × 0.35  +
        grammar_score  × 0.25  +
        plausibility   × 0.20  +
        input_conf     × 0.20

    We drive composite to specific values by setting:
        - logits=None  → entropy_cannot_evaluate → entropy_score=0.5
        - grammar: 0 errors → grammar_score=1.0
        - plausibility from mock GrammarReport
        - input_confidence from IntentPacket

    With logits=None, entropy_score=0.5:
        composite = 0.5×0.35 + 1.0×0.25 + plaus×0.20 + conf×0.20
                  = 0.175 + 0.25 + plaus×0.20 + conf×0.20

    We need plaus×0.20 + conf×0.20 to hit exact composites.
    We set plaus=1.0 and vary conf:
        composite = 0.175 + 0.25 + 0.20 + conf×0.20
                  = 0.625 + conf×0.20

    So:
        conf=0.875 → composite = 0.625 + 0.175 = 0.800  → PROCEED  (≥0.75)
        conf=0.625 → composite = 0.625 + 0.125 = 0.750  → PROCEED  (≥0.75 exactly)
        conf=0.500 → composite = 0.625 + 0.100 = 0.725  → CONFIRM  (0.60–0.75)
        conf=0.125 → composite = 0.625 + 0.025 = 0.650  → CONFIRM
        conf=-0.125 → needs clamp; use conf=0.0 → 0.625 → CONFIRM
        Get FALLBACK: need composite ∈ [0.40, 0.60)
            → conf×0.20 = target - 0.625; negative → unworkable.
    
    For FALLBACK/BLOCK we switch to plaus=0.0:
        composite = 0.175 + 0.25 + 0.0 + conf×0.20
                  = 0.425 + conf×0.20
        conf=0.875 → 0.425 + 0.175 = 0.600 → CONFIRM (boundary)
        conf=0.750 → 0.425 + 0.150 = 0.575 → FALLBACK (≥0.40)
        conf=0.000 → 0.425 + 0.0   = 0.425 → FALLBACK
        conf=-1.0  → use conf=0.0 clamped
    For BLOCK: need < 0.40 with plaus=0.0:
        composite = 0.425 + conf×0.20
        conf must give 0.425 + conf×0.20 < 0.40 → conf < -0.125 → unworkable.
    Use plaus=0.0 AND slightly lower entropy_score by using near-uniform logits:
        real logits: uniform → entropy≈1.0 → entropy_score≈0.0
        composite ≈ 0 + 0.25 + 0 + conf×0.20 = 0.25 + conf×0.20
        conf=0.0 → composite=0.25  → BLOCK  (< 0.40)
        conf=0.5 → composite=0.35  → BLOCK
    """

    def _gate_no_lt(
        self,
        grammar_report: GrammarReport,
        logits: torch.Tensor | None,
        confidence: float,
    ) -> SafetyDecision:
        """Build gate with mocked grammar and evaluate once."""
        checker = MagicMock(spec=GrammarChecker)
        checker.check.return_value = grammar_report
        g = SafetyGate(EntropyGuard(), checker)
        return g.evaluate(_make_result(logits=logits), _make_packet(confidence=confidence))

    def test_composite_0_75_gives_proceed(self) -> None:
        """Exact boundary: composite = 0.750 → PROCEED."""
        gr = _make_grammar_report(error_count=0, word_count=6,
                                  plausibility=1.0, passes=True)
        # logits=None → entropy_score=0.5
        # composite = 0.175 + 0.25 + 0.20 + 0.625×0.20 = 0.625 + 0.125 = 0.750
        d = self._gate_no_lt(gr, logits=None, confidence=0.625)
        assert abs(d.composite_confidence - 0.750) < 0.01, (
            f"Expected composite ≈ 0.750, got {d.composite_confidence}"
        )
        assert d.action is SafetyAction.PROCEED, (
            f"composite={d.composite_confidence}: expected PROCEED, got {d.action}"
        )

    def test_composite_0_60_gives_confirm(self) -> None:
        """Composite in [0.60, 0.75) range → CONFIRM."""
        # plaus=0.0, uniform logits → entropy_score≈0
        # composite ≈ 0 + 0.25 + 0.0 + 0.75×0.20 = 0.25 + 0.15 = 0.40 → barely FALLBACK
        # Use plaus=1.0, logits=None, conf=0.125
        # composite = 0.175 + 0.25 + 0.20 + 0.125×0.20 = 0.625 + 0.025 = 0.650 → CONFIRM
        gr = _make_grammar_report(error_count=0, word_count=6,
                                  plausibility=1.0, passes=True)
        d = self._gate_no_lt(gr, logits=None, confidence=0.125)
        assert 0.60 <= d.composite_confidence < 0.75, (
            f"Expected composite in [0.60,0.75), got {d.composite_confidence}"
        )
        assert d.action is SafetyAction.CONFIRM, (
            f"composite={d.composite_confidence}: expected CONFIRM, got {d.action}"
        )

    def test_composite_0_40_gives_fallback(self) -> None:
        """Composite in [0.40, 0.60) range → FALLBACK."""
        # plaus=0.0 (no medical words), logits=None, conf=0.75
        # composite = 0.175 + 0.25 + 0.0 + 0.75×0.20 = 0.425 + 0.15 = 0.575 → FALLBACK
        gr = _make_grammar_report(error_count=0, word_count=6,
                                  plausibility=0.0, passes=True)
        d = self._gate_no_lt(gr, logits=None, confidence=0.75)
        assert 0.40 <= d.composite_confidence < 0.60, (
            f"Expected composite in [0.40,0.60), got {d.composite_confidence}"
        )
        assert d.action is SafetyAction.FALLBACK, (
            f"composite={d.composite_confidence}: expected FALLBACK, got {d.action}"
        )

    def test_composite_below_0_40_gives_block(self) -> None:
        """Composite < 0.40 → BLOCK."""
        # Uniform logits → entropy_score≈0; plaus=0.0, conf=0.0
        # composite ≈ 0 + 0.25 + 0.0 + 0.0 = 0.25  → BLOCK
        uniform_logits = torch.zeros(5, 100)   # all equal → uniform softmax
        gr = _make_grammar_report(error_count=0, word_count=6,
                                  plausibility=0.0, passes=True)
        d = self._gate_no_lt(gr, logits=uniform_logits, confidence=0.0)
        assert d.composite_confidence < 0.40, (
            f"Expected composite < 0.40, got {d.composite_confidence}"
        )
        assert d.action is SafetyAction.BLOCK, (
            f"composite={d.composite_confidence}: expected BLOCK, got {d.action}"
        )


# ──────────────────────────────────────────────────────────────
# Test 6 — Timeout → FALLBACK (hard override)
# ──────────────────────────────────────────────────────────────

class TestTimeoutRoutesToFallback:
    """
    ``ReconstructionResult.is_timeout=True`` must always route to FALLBACK,
    regardless of logit quality, grammar, or word count.
    """

    def test_timeout_overrides_high_confidence(
        self,
        gate: SafetyGate,
        mock_grammar_checker: MagicMock,
    ) -> None:
        """Even with perfect entropy + grammar, timeout → FALLBACK."""
        # Confident logits → entropy_score ≈ 1.0 (best case)
        t = torch.zeros(6, 100); t[:, 0] = 20.0
        mock_grammar_checker.check.return_value = _make_grammar_report(
            error_count=0, word_count=6, plausibility=1.0, passes=True
        )
        result = _make_result(logits=t, is_timeout=True)
        decision = gate.evaluate(result, _make_packet(confidence=1.0))
        assert decision.action is SafetyAction.FALLBACK, (
            f"Timeout should force FALLBACK, got {decision.action}"
        )

    def test_timeout_reason_logged(
        self,
        gate: SafetyGate,
        mock_grammar_checker: MagicMock,
    ) -> None:
        """The reason list must mention is_timeout."""
        mock_grammar_checker.check.return_value = _make_grammar_report()
        t = torch.zeros(4, 100); t[:, 0] = 20.0
        decision = gate.evaluate(
            _make_result(logits=t, is_timeout=True),
            _make_packet(),
        )
        assert any("timeout" in r.lower() for r in decision.reasons), (
            f"Expected 'timeout' in reasons, got: {decision.reasons}"
        )

    def test_non_timeout_is_not_forced_to_fallback(
        self,
        gate: SafetyGate,
        mock_grammar_checker: MagicMock,
    ) -> None:
        """is_timeout=False must NOT force FALLBACK when composite is high."""
        t = torch.zeros(6, 100); t[:, 0] = 20.0
        mock_grammar_checker.check.return_value = _make_grammar_report(
            error_count=0, word_count=6, plausibility=1.0, passes=True
        )
        decision = gate.evaluate(
            _make_result(logits=t, is_timeout=False),
            _make_packet(confidence=0.95),
        )
        # Should be PROCEED (high confidence) — must NOT be FALLBACK due to timeout
        assert decision.action is SafetyAction.PROCEED, (
            f"Expected PROCEED with is_timeout=False, got {decision.action}"
        )

    @pytest.mark.parametrize("word_count", [1, 6, 35])
    def test_timeout_overrides_any_word_count(
        self,
        gate: SafetyGate,
        mock_grammar_checker: MagicMock,
        word_count: int,
    ) -> None:
        """
        Timeout check is first in confidence.py; word-count checks come after.
        So with is_timeout=True, action must always be FALLBACK, not BLOCK.
        """
        mock_grammar_checker.check.return_value = _make_grammar_report(
            word_count=word_count, passes=(2 < word_count <= 30)
        )
        decision = gate.evaluate(
            _make_result(is_timeout=True),
            _make_packet(),
        )
        assert decision.action is SafetyAction.FALLBACK, (
            f"word_count={word_count}, is_timeout=True: expected FALLBACK, "
            f"got {decision.action}"
        )


# ──────────────────────────────────────────────────────────────
# Misc integration smoke-tests
# ──────────────────────────────────────────────────────────────

class TestSafetyDecisionFields:
    """Verify SafetyDecision is fully populated on every code path."""

    def _decision(
        self,
        gate: SafetyGate,
        mock_grammar_checker: MagicMock,
        **result_kwargs,
    ) -> SafetyDecision:
        mock_grammar_checker.check.return_value = _make_grammar_report()
        t = torch.zeros(4, 100); t[:, 0] = 20.0
        return gate.evaluate(
            _make_result(logits=t, **result_kwargs),
            _make_packet(),
        )

    def test_latency_ms_positive(self, gate, mock_grammar_checker) -> None:
        d = self._decision(gate, mock_grammar_checker)
        assert d.latency_ms > 0.0

    def test_grammar_report_attached(self, gate, mock_grammar_checker) -> None:
        d = self._decision(gate, mock_grammar_checker)
        assert isinstance(d.grammar_report, GrammarReport)

    def test_entropy_report_attached(self, gate, mock_grammar_checker) -> None:
        d = self._decision(gate, mock_grammar_checker)
        assert isinstance(d.entropy_report, EntropyReport)

    def test_composite_in_unit_interval(self, gate, mock_grammar_checker) -> None:
        d = self._decision(gate, mock_grammar_checker)
        assert 0.0 <= d.composite_confidence <= 1.0

    def test_fallback_result_adds_reason(self, gate, mock_grammar_checker) -> None:
        """is_fallback=True on result must appear in reasons."""
        mock_grammar_checker.check.return_value = _make_grammar_report()
        t = torch.zeros(4, 100); t[:, 0] = 20.0
        d = gate.evaluate(
            _make_result(logits=t, is_fallback=True),
            _make_packet(confidence=1.0),
        )
        assert any("fallback" in r.lower() for r in d.reasons), (
            f"Expected fallback reason, got: {d.reasons}"
        )
