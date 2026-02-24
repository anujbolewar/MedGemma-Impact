"""
safety/entropy_guard.py — Per-token logit entropy analysis for NeuroWeave Sentinel.

Computes Shannon entropy at each token position from the LLM's raw logits,
normalises to [0, 1], and decides whether the output is reliable enough
to be voiced. High entropy indicates the model was uncertain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from core.constants import SentinelConstants as C
from core.logger import get_logger

_log = get_logger()

# Lazy import torch so the module can be imported even without CUDA
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # noqa: BLE001
    _TORCH_AVAILABLE = False

# Maximum flagged-position fraction before the output fails
_MAX_FLAGGED_FRACTION: float = 0.20


# ──────────────────────────────────────────────────────────────
# Output dataclass
# ──────────────────────────────────────────────────────────────

@dataclass
class EntropyReport:
    """
    Result of an entropy analysis pass over a single LLM generation.

    Attributes:
        mean_entropy: Arithmetic mean of per-token normalised entropy values.
        max_entropy:  Maximum normalised entropy across all token positions.
        flagged_positions: Indices of tokens whose entropy > ``threshold``.
        flagged_fraction:  Ratio of flagged tokens to total generated tokens.
        passes: True when mean_entropy ≤ threshold AND flagged_fraction < 0.20.
        threshold: The threshold value used for this evaluation.
        cannot_evaluate: True when logits were absent or malformed.
        per_token_entropy: Full list of per-token entropy values (may be empty).
    """

    mean_entropy: float = 0.0
    max_entropy: float = 0.0
    flagged_positions: list[int] = field(default_factory=list)
    flagged_fraction: float = 0.0
    passes: bool = False
    threshold: float = C.ENTROPY_THRESHOLD
    cannot_evaluate: bool = False
    per_token_entropy: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict for structured logging."""
        return {
            "mean_entropy": round(self.mean_entropy, 4),
            "max_entropy": round(self.max_entropy, 4),
            "flagged_positions": self.flagged_positions,
            "flagged_fraction": round(self.flagged_fraction, 4),
            "passes": self.passes,
            "threshold": self.threshold,
            "cannot_evaluate": self.cannot_evaluate,
        }


# ──────────────────────────────────────────────────────────────
# EntropyGuard
# ──────────────────────────────────────────────────────────────

class EntropyGuard:
    """
    Logit-entropy safety gate for NeuroWeave Sentinel LLM outputs.

    Computes per-token Shannon entropy ``H = -Σ p·log(p)`` from the
    model's raw logit distributions and normalises by ``log(vocab_size)``
    so that the result is in **[0, 1]**: 0 = perfectly certain, 1 = uniform.

    Args:
        threshold: Maximum acceptable normalised entropy (default from
                   :data:`~core.constants.SentinelConstants.ENTROPY_THRESHOLD`).
    """

    def __init__(self, threshold: float = C.ENTROPY_THRESHOLD) -> None:
        """Initialise guard with the given acceptance threshold."""
        self._threshold = threshold
        _log.info("entropy_guard", "init", {
            "threshold": threshold,
            "torch_available": _TORCH_AVAILABLE,
        })

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def compute_token_entropy(self, logits: "torch.Tensor") -> list[float]:
        """
        Compute normalised Shannon entropy for each token position.

        H(t) = -Σ_{v} p_v · log(p_v) / log(vocab_size)

        Args:
            logits: Float tensor of shape ``(num_tokens, vocab_size)`` containing
                    the raw (un-normalised) logit distributions produced by the
                    language model for each generated token position.

        Returns:
            List of per-token normalised entropy values in ``[0, 1]``,
            one entry per token position.

        Raises:
            Nothing — malformed input is handled with a warning and returns ``[]``.
        """
        if not _TORCH_AVAILABLE:
            _log.warn("entropy_guard", "torch_unavailable", {
                "action": "returning empty entropy list"
            })
            return []

        if logits is None or not hasattr(logits, "shape"):
            _log.warn("entropy_guard", "bad_logits", {
                "reason": "logits is None or not a tensor"
            })
            return []

        if logits.ndim != 2:
            _log.warn("entropy_guard", "bad_logits_shape", {
                "shape": list(logits.shape),
                "expected_ndim": 2,
                "reason": "expected (num_tokens, vocab_size)",
            })
            return []

        num_tokens, vocab_size = logits.shape
        if vocab_size < 2:
            _log.warn("entropy_guard", "bad_vocab_size", {
                "vocab_size": vocab_size
            })
            return []

        try:
            import torch
            import torch.nn.functional as F  # noqa: N812

            # Stable softmax + log-prob
            log_probs = F.log_softmax(logits.float(), dim=-1)   # (T, V)
            probs = log_probs.exp()                               # (T, V)

            # H = -Σ p * log(p)  — numerically safe: 0 * -inf → 0 via nansum
            raw_entropy = -(probs * log_probs).nansum(dim=-1)    # (T,)

            # Normalise to [0, 1]
            log_v = math.log(vocab_size)
            normalised = (raw_entropy / log_v).clamp(0.0, 1.0)   # (T,)

            return normalised.tolist()

        except Exception as exc:  # noqa: BLE001
            _log.warn("entropy_guard", "compute_error", {"error": str(exc)})
            return []

    def evaluate(self, result: object) -> EntropyReport:
        """
        Evaluate the entropy of an LLM generation result.

        Accepts any object with a ``logits`` attribute (typically a
        :class:`~llm.reconstructor.ReconstructionResult`). If ``logits``
        is ``None`` or malformed, returns an :class:`EntropyReport` with
        ``cannot_evaluate=True``.

        Args:
            result: Object with a ``logits`` attribute (``torch.Tensor`` or
                    ``None``).

        Returns:
            A fully populated :class:`EntropyReport`.
        """
        logits = getattr(result, "logits", None)

        # ── Cannot evaluate guard ────────────────────────────
        if logits is None:
            _log.info("entropy_guard", "cannot_evaluate", {
                "reason": "result.logits is None",
            })
            return EntropyReport(
                cannot_evaluate=True,
                threshold=self._threshold,
                passes=True,  # Don't block output when logits unavailable
            )

        per_token = self.compute_token_entropy(logits)

        if not per_token:
            # compute_token_entropy already logged the cause
            return EntropyReport(
                cannot_evaluate=True,
                threshold=self._threshold,
                passes=True,
            )

        # ── Aggregate statistics ─────────────────────────────
        n = len(per_token)
        mean_entropy = sum(per_token) / n
        max_entropy = max(per_token)
        flagged = [i for i, h in enumerate(per_token) if h > self._threshold]
        flagged_fraction = len(flagged) / n

        passes = (
            mean_entropy <= self._threshold
            and flagged_fraction < _MAX_FLAGGED_FRACTION
        )

        report = EntropyReport(
            mean_entropy=round(mean_entropy, 6),
            max_entropy=round(max_entropy, 6),
            flagged_positions=flagged,
            flagged_fraction=round(flagged_fraction, 4),
            passes=passes,
            threshold=self._threshold,
            cannot_evaluate=False,
            per_token_entropy=[round(h, 6) for h in per_token],
        )

        level = "info" if passes else "warn"
        getattr(_log, level)("entropy_guard", "evaluated", report.to_dict())
        return report
