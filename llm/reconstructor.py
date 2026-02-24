"""
llm/reconstructor.py — Intent-to-sentence reconstruction using MedGemma.

Accepts an :class:`IntentPacket` (ordered list of vocabulary token codes),
builds a constrained clinical prompt, runs deterministic MedGemma inference
inside a :class:`concurrent.futures.ThreadPoolExecutor` with a hard latency
budget, and returns a :class:`ReconstructionResult`.

Design notes
------------
* **Prompt contract** — the system and user prompts are fixed strings
  (see :data:`_SYSTEM_PROMPT` / :data:`_USER_TEMPLATE`).  Do not modify
  them without re-validating clinical output quality.
* **Deterministic inference** — ``do_sample=False`` is mandatory.  The
  ``temperature=1.0`` is set explicitly but is effectively ignored by
  greedy decoding; it is here to satisfy model config validation.
* **Timeout** — hard cutoff is ``C.MAX_INFERENCE_MS / 1000`` seconds via
  ``Future.result(timeout=...)``.  The background thread is marked daemon
  so it does not block process exit if abandoned.
* **CPU fallback** — if GPU inference raises any exception, the module
  re-attempts with inputs on CPU and ``max_new_tokens=20``.  If the model
  is already on CPU (because VRAM was below the 3.5 GB threshold) this is
  a no-op; otherwise the model is migrated to CPU in-place with a warning.
* **Logits** — collected via ``output_scores=True, return_dict_in_generate=True``
  and stacked to shape ``(num_new_tokens, 1, vocab_size)`` for downstream
  entropy calculation by ``safety/entropy_guard.py``.
* **Template fallback** — :meth:`SentenceReconstructor.reconstruct_from_template`
  provides a rule-based path that requires no GPU and fires when LLM
  reconstruction fails completely.

All logging goes through :func:`core.logger.get_logger` (NWSLogger JSONL).
"""

from __future__ import annotations

import concurrent.futures
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, List, NamedTuple, Optional, Tuple

from core.logger import get_logger
from core.constants import C  # C.MAX_INFERENCE_MS, C.MAX_SENTENCE_TOKENS, …

# Lazy import of vocab to avoid circular import at collection time
# (encoder.token_vocab has no dependency on llm.*)
from encoder.token_vocab import VOCAB  # dict[str, IntentToken]

if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from encoder.intent_encoder import IntentBundle


# ── Prompt constants (system contract — do not modify) ───────────────────────

_SYSTEM_PROMPT: str = (
    "You are an assistive communication system for patients with severe motor impairment. "
    "Your task is to reconstruct a single, natural, first-person spoken sentence from "
    "a list of communication intent tokens. "
    "Rules: Output ONLY the sentence. No explanation. No prefix. "
    "Maximum 25 words. Must be grammatically correct. "
    "Must sound like a patient speaking to a nurse or caregiver."
)

_USER_TEMPLATE: str = "Intent tokens: {token_string}\nSpoken sentence:"

# ── Inference hyper-parameters (deterministic — do not change) ───────────────
_DO_SAMPLE: bool = False
_TEMPERATURE: float = 1.0          # ignored when do_sample=False; explicit per spec
_MAX_NEW_TOKENS: int = 30
_MIN_NEW_TOKENS: int = 3
_REPETITION_PENALTY: float = 1.2

# ── Safe fallback sentences ───────────────────────────────────────────────────
_FB_TIMEOUT  = "I need to communicate — please check with me."
_FB_ERROR    = "I need assistance — could you help me please."
_FB_GENERIC  = "I would like to tell you something important."

# ── Minimum decoded characters before a result is considered non-empty ────────
_MIN_CHARS: int = 3

# ── Prefixes the model sometimes echoes back; strip them ──────────────────────
_ECHO_PREFIXES = re.compile(
    r"^(spoken sentence\s*:?\s*|intent tokens\s*:.*?\n|patient\s*:?\s*)",
    re.IGNORECASE,
)


# ── IntentPacket ──────────────────────────────────────────────────────────────

@dataclass
class IntentPacket:
    """
    An ordered sequence of vocabulary token codes representing a patient's
    communication intent.

    Each element of :attr:`tokens` must be a key in
    :data:`~encoder.token_vocab.VOCAB` (e.g. ``'BODY_CHEST'``, ``'SENS_PAIN'``,
    ``'INT_SEVERE'``, ``'TIME_NOW'``).

    Attributes:
        tokens: Ordered list of token codes.  Order reflects selection order
            on the symbol board (body part first, then sensation, etc.).

    Example::

        pkt = IntentPacket(tokens=['BODY_CHEST', 'SENS_PAIN', 'INT_SEVERE', 'TIME_NOW'])
        reconstructor.reconstruct(pkt)
    """

    tokens: List[str] = field(default_factory=list)
    confidence: float = 1.0  # Aggregated gaze / input confidence [0, 1]

    # ── Query helpers ─────────────────────────────────────────────────────

    def has(self, code: str) -> bool:
        """Return ``True`` if *code* is present in :attr:`tokens`."""
        return code in self.tokens

    def any_of(self, *codes: str) -> bool:
        """Return ``True`` if **any** of *codes* are present."""
        return any(c in self.tokens for c in codes)

    def first_of(self, prefix: str) -> Optional[str]:
        """
        Return the first token code that starts with *prefix*, or ``None``.

        Example: ``packet.first_of('BODY_')`` → ``'BODY_CHEST'``.
        """
        return next((t for t in self.tokens if t.startswith(prefix)), None)

    def natural_string(self) -> str:
        """
        Build a comma-separated natural-language hint string from the tokens,
        suitable for injection into the LLM prompt.

        Unknown codes are included verbatim (lower-cased) so the prompt never
        contains placeholders.

        Example::

            pkt = IntentPacket(['BODY_CHEST', 'SENS_PAIN', 'INT_SEVERE'])
            pkt.natural_string()
            # → 'chest or upper body, pain or hurting, severe'
        """
        parts = []
        for code in self.tokens:
            tok = VOCAB.get(code)
            parts.append(tok.natural_hint if tok else code.lower().replace("_", " "))
        return ", ".join(parts)

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_bundle(cls, bundle: "IntentBundle") -> "IntentPacket":
        """
        Convert an :class:`~encoder.intent_encoder.IntentBundle` to an
        :class:`IntentPacket` by reverse-looking up natural hint values in
        :data:`~encoder.token_vocab.VOCAB`.

        The mapping is best-effort: if a bundle field value cannot be matched
        to a VOCAB code the field is silently skipped.  Callers that
        construct tokens directly should prefer ``IntentPacket(tokens=[...])``.

        Args:
            bundle: A completed ``IntentBundle`` from the intent encoder.

        Returns:
            An :class:`IntentPacket` with resolved token codes.
        """
        # Build a reverse map: natural_hint → code for each expected prefix
        _hint_map = {tok.natural_hint: tok.code for tok in VOCAB.values()}
        # Also map by the token's description (some bundles store description)
        _desc_map = {tok.description.lower(): tok.code for tok in VOCAB.values()}

        def _resolve(value: Optional[str], prefix: str) -> Optional[str]:
            if not value:
                return None
            # Exact natural_hint match
            if value in _hint_map and _hint_map[value].startswith(prefix):
                return _hint_map[value]
            # Case-insensitive description match
            vl = value.lower()
            if vl in _desc_map and _desc_map[vl].startswith(prefix):
                return _desc_map[vl]
            # Substring match within the target category
            for code, tok in VOCAB.items():
                if code.startswith(prefix) and (
                    value.lower() in tok.natural_hint.lower()
                    or value.lower() in tok.description.lower()
                ):
                    return code
            return None

        codes: List[str] = []
        for val, prefix in (
            (bundle.body_part, "BODY_"),   # type: ignore[union-attr]
            (bundle.sensation, "SENS_"),
            (bundle.urgency,   "TIME_"),
            (bundle.intensity, "INT_"),
        ):
            code = _resolve(val, prefix)
            if code:
                codes.append(code)

        return cls(tokens=codes)


# ── ReconstructionResult ──────────────────────────────────────────────────────

@dataclass
class ReconstructionResult:
    """
    Output from a single :meth:`SentenceReconstructor.reconstruct` call.

    Attributes:
        text: The reconstructed patient communication sentence.
        token_ids: Raw output token IDs from the model (empty on fallback).
        logits: Stacked generation logits, shape
            ``(num_new_tokens, 1, vocab_size)``, for entropy calculation.
            ``None`` on timeout, CPU fallback, or template paths.
        latency_ms: Wall-clock time from call start to result ready (ms).
        is_timeout: ``True`` if the inference exceeded ``MAX_INFERENCE_MS``.
        is_fallback: ``True`` if the result came from a template or fallback
            string rather than from LLM inference.
    """

    text: str
    token_ids: List[int]
    logits: Optional["torch.Tensor"]
    latency_ms: float
    is_timeout: bool
    is_fallback: bool


# ── Template rules ────────────────────────────────────────────────────────────

class _TemplateRule(NamedTuple):
    """
    A single rule in the template-based fallback path.

    Attributes:
        name: Short identifier used in log entries.
        match: Called with the :class:`IntentPacket`; returns ``True`` if this
            rule applies.
        render: Called with the :class:`IntentPacket`; returns the sentence.
    """

    name: str
    match: Callable[[IntentPacket], bool]
    render: Callable[[IntentPacket], str]


def _intensity_word(p: IntentPacket) -> str:
    """Return the intensity adjective for the packet, or empty string."""
    code = p.first_of("INT_")
    return {
        "INT_MILD":     "mild",
        "INT_MODERATE": "moderate",
        "INT_SEVERE":   "severe",
        "INT_CRITICAL": "critical",
    }.get(code or "", "")


def _body_phrase(p: IntentPacket) -> str:
    """Return the natural-hint phrase for the body-part token."""
    code = p.first_of("BODY_")
    if code and code in VOCAB:
        return VOCAB[code].natural_hint
    return "my body"


def _int_pfx(p: IntentPacket) -> str:
    """Return 'mild ', 'severe ', etc. with trailing space, or empty."""
    w = _intensity_word(p)
    return f"{w} " if w else ""


# Priority-ordered rules — first match wins.
_TEMPLATE_RULES: List[_TemplateRule] = [
    # 1 ── Severe/critical emergency pain ────────────────────────────────────
    _TemplateRule(
        name="emrg_severe_pain",
        match=lambda p: p.any_of("EMRG_PAIN_SEVERE")
                        or (p.has("SENS_PAIN") and p.has("INT_CRITICAL")),
        render=lambda p: "I am in severe, unbearable pain — please help me right now.",
    ),
    # 2 ── Breathing difficulty ───────────────────────────────────────────────
    _TemplateRule(
        name="emrg_breathing",
        match=lambda p: p.has("EMRG_BREATHING"),
        render=lambda p: "I am having difficulty breathing, please come immediately.",
    ),
    # 3 ── General emergency / call nurse ────────────────────────────────────
    _TemplateRule(
        name="emrg_urgent_nurse",
        match=lambda p: p.any_of("EMRG_URGENT", "EMRG_CALL_NURSE"),
        render=lambda p: "Please call the nurse, this is urgent.",
    ),
    # 4 ── Chest pain specifically ────────────────────────────────────────────
    _TemplateRule(
        name="chest_pain",
        match=lambda p: p.has("SENS_PAIN") and p.has("BODY_CHEST"),
        render=lambda p: f"I have {_int_pfx(p)}pain in my chest.",
    ),
    # 5 ── Pain at any body part ──────────────────────────────────────────────
    _TemplateRule(
        name="pain_body",
        match=lambda p: p.has("SENS_PAIN"),
        render=lambda p: f"I have {_int_pfx(p)}pain in my {_body_phrase(p)}.",
    ),
    # 6 ── Chest pressure ─────────────────────────────────────────────────────
    _TemplateRule(
        name="chest_pressure",
        match=lambda p: p.has("SENS_PRESSURE") and p.has("BODY_CHEST"),
        render=lambda p: f"I am feeling {_int_pfx(p)}pressure in my chest.",
    ),
    # 7 ── Head pressure ──────────────────────────────────────────────────────
    _TemplateRule(
        name="head_pressure",
        match=lambda p: p.has("SENS_PRESSURE") and p.has("BODY_HEAD"),
        render=lambda p: f"I have {_int_pfx(p)}pressure in my head.",
    ),
    # 8 ── General pressure (any body part) ───────────────────────────────────
    _TemplateRule(
        name="pressure_body",
        match=lambda p: p.has("SENS_PRESSURE"),
        render=lambda p: f"I am feeling {_int_pfx(p)}pressure in my {_body_phrase(p)}.",
    ),
    # 9 ── Nausea ─────────────────────────────────────────────────────────────
    _TemplateRule(
        name="nausea",
        match=lambda p: p.has("SENS_NAUSEA"),
        render=lambda p: "I feel nauseous and unwell.",
    ),
    # 10 ── Need water ────────────────────────────────────────────────────────
    _TemplateRule(
        name="need_water",
        match=lambda p: p.has("NEED_WATER"),
        render=lambda p: "I am thirsty, I need water please.",
    ),
    # 11 ── Need food ─────────────────────────────────────────────────────────
    _TemplateRule(
        name="need_food",
        match=lambda p: p.has("NEED_FOOD"),
        render=lambda p: "I am hungry, could I have something to eat please.",
    ),
    # 12 ── Need toilet ───────────────────────────────────────────────────────
    _TemplateRule(
        name="need_toilet",
        match=lambda p: p.has("NEED_TOILET"),
        render=lambda p: "I need to use the bathroom, could you help me please.",
    ),
    # 13 ── Need medication ───────────────────────────────────────────────────
    _TemplateRule(
        name="need_medication",
        match=lambda p: p.has("NEED_MEDICATION"),
        render=lambda p: "I need pain medication, please.",
    ),
    # 14 ── Need nurse ────────────────────────────────────────────────────────
    _TemplateRule(
        name="need_nurse",
        match=lambda p: p.has("NEED_NURSE"),
        render=lambda p: "Could you please call the nurse.",
    ),
    # 15 ── Need repositioning ────────────────────────────────────────────────
    _TemplateRule(
        name="need_reposition",
        match=lambda p: p.has("NEED_REPOSITION"),
        render=lambda p: "I need to change my position, could you help me please.",
    ),
    # 16 ── Need rest ─────────────────────────────────────────────────────────
    _TemplateRule(
        name="need_rest",
        match=lambda p: p.has("NEED_REST"),
        render=lambda p: "I am tired and I need to rest.",
    ),
    # 17 ── Need family ───────────────────────────────────────────────────────
    _TemplateRule(
        name="need_family",
        match=lambda p: p.has("NEED_FAMILY"),
        render=lambda p: "I would like my family to be with me.",
    ),
    # 18 ── General discomfort ────────────────────────────────────────────────
    _TemplateRule(
        name="discomfort_body",
        match=lambda p: p.has("SENS_DISCOMFORT"),
        render=lambda p: f"I am experiencing {_int_pfx(p)}discomfort in my {_body_phrase(p)}.",
    ),
    # 19 ── Itching ───────────────────────────────────────────────────────────
    _TemplateRule(
        name="itch_body",
        match=lambda p: p.has("SENS_ITCH"),
        render=lambda p: f"I have {_int_pfx(p)}itching in my {_body_phrase(p)}.",
    ),
    # 20 ── Worsening (modifier on any sensation) ─────────────────────────────
    _TemplateRule(
        name="worsening",
        match=lambda p: p.has("TIME_WORSENING") and p.first_of("SENS_") is not None,
        render=lambda p: (
            f"I have {_int_pfx(p)}pain in my {_body_phrase(p)} "
            "and it is getting worse."
        ),
    ),
    # 21 ── Cognitive YES ─────────────────────────────────────────────────────
    _TemplateRule(
        name="cog_yes",
        match=lambda p: p.has("COG_YES"),
        render=lambda p: "Yes, that is correct.",
    ),
    # 22 ── Cognitive NO ──────────────────────────────────────────────────────
    _TemplateRule(
        name="cog_no",
        match=lambda p: p.has("COG_NO"),
        render=lambda p: "No, that is not right.",
    ),
]


# ── SentenceReconstructor ─────────────────────────────────────────────────────

class SentenceReconstructor:
    """
    MedGemma-based intent-to-sentence reconstructor.

    Accepts an :class:`IntentPacket`, formats it into the fixed clinical prompt,
    and runs deterministic greedy decoding within the latency budget.  Falls
    back through three degradation levels:

    1. GPU inference (primary path).
    2. CPU inference on ``max_new_tokens=20`` if GPU raises.
    3. :meth:`reconstruct_from_template` if CPU also fails.

    Args:
        model: A loaded ``AutoModelForCausalLM`` instance (from
            :func:`~llm.model_loader.load_model`).
        tokenizer: The matching ``AutoTokenizer``.

    Example::

        model, tokenizer = load_model("google/medgemma-4b-it")
        warm_up(model, tokenizer)
        recon = SentenceReconstructor(model, tokenizer)
        result = recon.reconstruct(
            IntentPacket(tokens=["BODY_CHEST", "SENS_PAIN", "INT_SEVERE", "TIME_NOW"])
        )
        print(result.text)
        # → "I have severe pain in my chest."
    """

    def __init__(
        self,
        model: "AutoModelForCausalLM",
        tokenizer: "AutoTokenizer",
    ) -> None:
        """Bind to a loaded model and tokenizer; do not load here."""
        self._model = model
        self._tokenizer = tokenizer
        log = get_logger()
        log.info("llm", "reconstructor_init", {})

    # ── Public API ────────────────────────────────────────────────────────

    def reconstruct(self, packet: IntentPacket) -> ReconstructionResult:
        """
        Reconstruct a spoken sentence from *packet* using MedGemma.

        The inference runs in a daemon thread managed by a single-worker
        :class:`concurrent.futures.ThreadPoolExecutor`.  If it has not
        completed within ``C.MAX_INFERENCE_MS / 1000`` seconds the future
        is abandoned and a timeout result is returned.

        On any GPU exception the method automatically retries on CPU.  If
        CPU also fails, :meth:`reconstruct_from_template` is called as a
        final fallback.

        Args:
            packet: Intent tokens to reconstruct from.

        Returns:
            A :class:`ReconstructionResult` with all fields populated.
        """
        log = get_logger()
        t0 = time.monotonic()
        timeout_s = C.MAX_INFERENCE_MS / 1000.0

        prompt_text, input_len = self._build_prompt_text(packet)
        log.info(
            "llm",
            "reconstruct_start",
            {
                "tokens": packet.tokens,
                "prompt_chars": len(prompt_text),
                "timeout_s": timeout_s,
            },
        )

        # ── GPU inference via thread pool ─────────────────────────────────
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="llm-infer"
        ) as pool:
            future = pool.submit(self._run_inference, prompt_text, input_len)
            try:
                text, token_ids, logits = future.result(timeout=timeout_s)
                latency_ms = (time.monotonic() - t0) * 1000.0
                text = self._post_process(text)
                result = ReconstructionResult(
                    text=text if text else _FB_ERROR,
                    token_ids=token_ids,
                    logits=logits,
                    latency_ms=latency_ms,
                    is_timeout=False,
                    is_fallback=(not text),
                )
                log.perf(
                    "llm",
                    "reconstruct_gpu_done",
                    latency_ms=latency_ms,
                    data={"tokens_generated": len(token_ids)},
                )
                return result

            except concurrent.futures.TimeoutError:
                latency_ms = (time.monotonic() - t0) * 1000.0
                log.warn(
                    "llm",
                    "reconstruct_timeout",
                    {"latency_ms": round(latency_ms, 1), "timeout_s": timeout_s},
                )
                return ReconstructionResult(
                    text=_FB_TIMEOUT,
                    token_ids=[],
                    logits=None,
                    latency_ms=latency_ms,
                    is_timeout=True,
                    is_fallback=True,
                )

            except Exception as exc:  # noqa: BLE001
                log.warn(
                    "llm",
                    "gpu_inference_failed",
                    {"error": str(exc), "fallback": "cpu"},
                )

        # ── CPU fallback ──────────────────────────────────────────────────
        try:
            cpu_t0 = time.monotonic()
            text, token_ids, _ = self._run_cpu_fallback(prompt_text, input_len)
            latency_ms = (time.monotonic() - t0) * 1000.0
            text = self._post_process(text)
            log.perf(
                "llm",
                "reconstruct_cpu_done",
                latency_ms=(time.monotonic() - cpu_t0) * 1000.0,
                data={"tokens_generated": len(token_ids)},
            )
            return ReconstructionResult(
                text=text if text else _FB_ERROR,
                token_ids=token_ids,
                logits=None,       # skipped on CPU fallback to save time
                latency_ms=latency_ms,
                is_timeout=False,
                is_fallback=(not text),
            )

        except Exception as cpu_exc:  # noqa: BLE001
            log.error(
                "llm",
                "cpu_fallback_failed",
                {"error": str(cpu_exc), "fallback": "template"},
            )
            return self.reconstruct_from_template(packet)

    def reconstruct_from_template(self, packet: IntentPacket) -> ReconstructionResult:
        """
        Rule-based sentence reconstruction without the LLM.

        Evaluates :data:`_TEMPLATE_RULES` in priority order and returns the
        first match.  Falls back to :data:`_FB_GENERIC` if no rule matches.

        This path is entirely CPU-bound and completes in microseconds.  It is
        used as the final fallback when both GPU and CPU inference fail, and
        can also be called directly for testing or latency-critical contexts.

        Args:
            packet: Intent tokens to reconstruct from.

        Returns:
            A :class:`ReconstructionResult` with ``is_fallback=True`` and
            ``logits=None``.
        """
        log = get_logger()
        t0 = time.monotonic()

        text = _FB_GENERIC
        matched_rule = "generic_fallback"

        for rule in _TEMPLATE_RULES:
            if rule.match(packet):
                try:
                    text = rule.render(packet)
                    matched_rule = rule.name
                except Exception as exc:  # noqa: BLE001
                    log.warn(
                        "llm",
                        "template_render_error",
                        {"rule": rule.name, "error": str(exc)},
                    )
                break

        latency_ms = (time.monotonic() - t0) * 1000.0
        log.perf(
            "llm",
            "reconstruct_template_done",
            latency_ms=latency_ms,
            data={"rule": matched_rule, "tokens": packet.tokens},
        )
        return ReconstructionResult(
            text=text,
            token_ids=[],
            logits=None,
            latency_ms=latency_ms,
            is_timeout=False,
            is_fallback=True,
        )

    # ── Private helpers ───────────────────────────────────────────────────

    def _build_prompt_text(self, packet: IntentPacket) -> Tuple[str, int]:
        """
        Apply the chat template and tokenize, returning ``(prompt_text, input_len)``.

        Tries ``apply_chat_template`` with a separate system role first.
        If the tokenizer raises (some models do not support the system role),
        falls back to prepending the system prompt to the user message.

        Args:
            packet: Source intent packet.

        Returns:
            ``(prompt_text, input_len)`` where *prompt_text* is the
            fully-formatted string ready to be tokenized and *input_len* is
            the number of input token IDs (needed to slice output IDs).
        """
        token_string = packet.natural_string() or "unspecified"
        user_content = _USER_TEMPLATE.format(token_string=token_string)

        # Attempt structured system + user chat template
        try:
            prompt_text: str = self._tokenizer.apply_chat_template(  # type: ignore[union-attr]
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:  # noqa: BLE001
            # Tokenizer does not support system role — merge into user turn
            prompt_text = self._tokenizer.apply_chat_template(  # type: ignore[union-attr]
                [{"role": "user", "content": f"{_SYSTEM_PROMPT}\n\n{user_content}"}],
                tokenize=False,
                add_generation_prompt=True,
            )

        input_ids = self._tokenizer(  # type: ignore[union-attr]
            prompt_text, return_tensors="pt"
        )["input_ids"]
        input_len: int = input_ids.shape[1]
        return prompt_text, input_len

    def _run_inference(
        self,
        prompt_text: str,
        input_len: int,
    ) -> Tuple[str, List[int], Optional["torch.Tensor"]]:
        """
        Run MedGemma inference on the GPU (or whichever device the model is on).

        Called inside a thread-pool worker.  Returns raw decoded text, new
        token IDs, and stacked logits.

        Args:
            prompt_text: Fully-formatted prompt string.
            input_len: Number of prompt input token IDs to slice off output.

        Returns:
            ``(decoded_text, new_token_ids, logits_tensor)``
        """
        import torch  # type: ignore

        try:
            device = next(self._model.parameters()).device  # type: ignore[union-attr]
        except StopIteration:
            device = torch.device("cpu")

        inputs = self._tokenizer(  # type: ignore[union-attr]
            prompt_text, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = self._model.generate(  # type: ignore[union-attr]
                **inputs,
                max_new_tokens=_MAX_NEW_TOKENS,
                min_new_tokens=_MIN_NEW_TOKENS,
                do_sample=_DO_SAMPLE,
                temperature=_TEMPERATURE,
                repetition_penalty=_REPETITION_PENALTY,
                pad_token_id=self._tokenizer.eos_token_id,  # type: ignore[union-attr]
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Slice out only the newly generated token IDs
        new_ids: List[int] = outputs.sequences[0][input_len:].tolist()

        # Stack per-step score tensors → (num_new_tokens, 1, vocab_size)
        logits: Optional["torch.Tensor"] = None
        if outputs.scores:
            logits = torch.stack(outputs.scores, dim=0)  # type: ignore[call-overload]

        decoded = self._tokenizer.decode(  # type: ignore[union-attr]
            new_ids, skip_special_tokens=True
        ).strip()

        return decoded, new_ids, logits

    def _run_cpu_fallback(
        self,
        prompt_text: str,
        input_len: int,
    ) -> Tuple[str, List[int], None]:
        """
        Retry inference on CPU with a reduced ``max_new_tokens=20`` budget.

        If the model is already on CPU (e.g. VRAM was insufficient) this is
        identical to normal inference.  If the model is on GPU it is migrated
        to CPU in-place (permanent for this session) with a warning.

        Args:
            prompt_text: Fully-formatted prompt string.
            input_len: Number of prompt input token IDs to slice off output.

        Returns:
            ``(decoded_text, new_token_ids, None)``  — logits are not
            collected on the CPU fallback path to save time.
        """
        import torch  # type: ignore

        log = get_logger()

        try:
            device = next(self._model.parameters()).device  # type: ignore[union-attr]
        except StopIteration:
            device = torch.device("cpu")

        if device.type != "cpu":
            log.warn(
                "llm",
                "cpu_fallback_migrate_model",
                {"from_device": str(device)},
            )
            try:
                self._model = self._model.to("cpu")  # type: ignore[union-attr]
            except Exception as migrate_exc:  # noqa: BLE001
                log.error(
                    "llm", "cpu_migrate_failed", {"error": str(migrate_exc)}
                )
                raise

        inputs = self._tokenizer(  # type: ignore[union-attr]
            prompt_text, return_tensors="pt"
        )  # already on CPU (default)

        with torch.no_grad():
            output_ids = self._model.generate(  # type: ignore[union-attr]
                **inputs,
                max_new_tokens=20,
                min_new_tokens=_MIN_NEW_TOKENS,
                do_sample=_DO_SAMPLE,
                temperature=_TEMPERATURE,
                repetition_penalty=_REPETITION_PENALTY,
                pad_token_id=self._tokenizer.eos_token_id,  # type: ignore[union-attr]
            )

        new_ids: List[int] = output_ids[0][input_len:].tolist()
        decoded = self._tokenizer.decode(  # type: ignore[union-attr]
            new_ids, skip_special_tokens=True
        ).strip()

        return decoded, new_ids, None

    @staticmethod
    def _post_process(text: str) -> str:
        """
        Clean raw model output before storing in :class:`ReconstructionResult`.

        Steps applied (in order):

        1. Strip leading/trailing whitespace.
        2. Remove any echoed prompt prefix (``Spoken sentence:``, etc.).
        3. Strip wrapping quotation marks.
        4. Truncate to 25 words maximum, re-appending a terminal period.
        5. Capitalise the first character.
        6. Ensure the sentence ends with ``.``, ``!``, or ``?``.

        Args:
            text: Raw decoded string from the tokenizer.

        Returns:
            Cleaned sentence string.  Returns an empty string if the result
            is shorter than :data:`_MIN_CHARS` after cleaning (signals the
            caller to use a fallback).
        """
        text = text.strip()
        if not text:
            return ""

        # Strip echoed prompt prefixes
        text = _ECHO_PREFIXES.sub("", text).strip()

        # Strip wrapping quotes (single or double)
        text = text.strip("\"'")

        # Truncate to 25 words
        words = text.split()
        if len(words) > 25:
            text = " ".join(words[:25])
            # Try to re-close at the last sentence boundary
            for punct in (".", ",", ";"):
                idx = text.rfind(punct)
                if idx > len(text) // 2:
                    text = text[:idx + 1]
                    break
            else:
                text = text.rstrip(",;") + "."

        # Capitalise first character
        if text:
            text = text[0].upper() + text[1:]

        # Ensure terminal punctuation
        if text and text[-1] not in ".!?":
            text += "."

        return text if len(text) >= _MIN_CHARS else ""
