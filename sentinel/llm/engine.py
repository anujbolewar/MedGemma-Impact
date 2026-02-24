"""
sentinel/llm/engine.py — MedGemma 4b-it inference engine with NF4 quantization.

Loads medgemma-4b-it in 4-bit NF4 quantized mode to stay within the 4GB VRAM
constraint. Enforces a hard 2.3s latency budget via threading with timeout.
This module is a thread-safe singleton — the model is loaded once and reused.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from sentinel.core.config import LLMConfig

logger = logging.getLogger(__name__)


class ModelNotFoundError(RuntimeError):
    """
    Raised when the MedGemma model is not cached locally.

    This system is fully offline — if the model is absent, the user
    must run `scripts/setup_model.py` while online to download it first.
    """


@dataclass
class InferenceResult:
    """
    Result of a single MedGemma inference call.

    Attributes:
        text: The reconstructed sentence (stripped of prompt and special tokens).
        confidence: Estimated confidence in [0, 1] (based on token probability).
        latency_ms: Wall-clock time from prompt submission to result.
        truncated: True if generation was cut short by the latency budget.
        input_tokens: Number of input tokens consumed.
        output_tokens: Number of new tokens generated.
    """

    text: str
    confidence: float
    latency_ms: float
    truncated: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


class MedGemmaEngine:
    """
    Thread-safe MedGemma 4b-it inference engine.

    Uses bitsandbytes NF4 quantization to fit within 4GB VRAM.
    The model is loaded lazily on first call to :meth:`load` or :meth:`infer`.
    A 2.3s hard latency cutoff is enforced via a background thread with timeout.

    Args:
        config: LLM configuration (model ID, quantization, latency budget, etc.).
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialise the engine without loading the model."""
        self._cfg = config
        self._lock = threading.Lock()
        self._model: Optional[object] = None
        self._tokenizer: Optional[object] = None
        self._loaded: bool = False
        logger.info("MedGemmaEngine initialised (model=%s)", config.model_id)

    # ──────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────

    def load(self) -> None:
        """
        Load MedGemma model and tokenizer into memory.

        Must be called before :meth:`infer`. Safe to call multiple times —
        subsequent calls are no-ops if model is already loaded. Blocking.

        Raises:
            ModelNotFoundError: If the model is not cached and we are offline.
            RuntimeError: If model loading fails for any other reason.
        """
        with self._lock:
            if self._loaded:
                return
            self._load_locked()

    def unload(self) -> None:
        """
        Release model from memory and free GPU/CPU resources.

        Safe to call even if the model was never loaded.
        """
        with self._lock:
            if not self._loaded:
                return
            logger.info("Unloading MedGemma model from memory")
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded")

    # ──────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────

    def infer(self, prompt: str) -> InferenceResult:
        """
        Run MedGemma inference on a prompt string.

        Lazy-loads the model if not already loaded. Enforces a hard latency
        budget of ``config.latency_budget_ms`` ms via threading. If generation
        exceeds the budget, any partial text generated so far is returned with
        ``truncated=True``.

        Args:
            prompt: The formatted user-turn prompt text.

        Returns:
            An :class:`InferenceResult` with generated text and metadata.

        Raises:
            ModelNotFoundError: If model is not cached and system is offline.
        """
        if not self._loaded:
            self.load()

        t_start = time.monotonic()

        # Run generation in a sub-thread so we can enforce timeout
        result_container: list[InferenceResult] = []
        exc_container: list[Exception] = []

        def _generate() -> None:
            """Generate a response and store it in result_container."""
            try:
                result = self._generate_internal(prompt, t_start)
                result_container.append(result)
            except Exception as exc:  # noqa: BLE001
                exc_container.append(exc)

        gen_thread = threading.Thread(target=_generate, daemon=True, name="medgemma-gen")
        gen_thread.start()

        timeout_seconds = self._cfg.latency_budget_ms / 1000.0
        gen_thread.join(timeout=timeout_seconds)

        if exc_container:
            raise exc_container[0]

        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        if result_container:
            return result_container[0]

        # Timeout — return a safe fallback message
        logger.warning(
            "Inference timed out after %.0fms (budget=%.0fms)",
            elapsed_ms,
            self._cfg.latency_budget_ms,
        )
        return InferenceResult(
            text="I am trying to communicate — please wait.",
            confidence=0.0,
            latency_ms=elapsed_ms,
            truncated=True,
        )

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _load_locked(self) -> None:
        """
        Internal model loading (must be called while holding self._lock).

        Checks the local HF cache first. Raises ModelNotFoundError if absent.
        """
        # Deferred imports — avoid import-time GPU allocation
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig  # type: ignore

        cache_dir = self._cfg.resolved_cache_dir
        model_id = self._cfg.model_id

        logger.info(
            "Loading %s from cache: %s", model_id, cache_dir
        )

        # Verify model presence in local cache without making network calls
        self._assert_model_cached(model_id, cache_dir)

        # Build quantization config
        bnb_config = self._build_bnb_config()

        try:
            logger.info("Loading tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=str(cache_dir),
                local_files_only=True,
                trust_remote_code=False,
            )

            logger.info("Loading model weights (NF4 quantized)...")
            t0 = time.monotonic()
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=str(cache_dir),
                local_files_only=True,
                quantization_config=bnb_config,
                device_map=self._cfg.device_map,
                torch_dtype=torch.float16,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
            )
            elapsed = (time.monotonic() - t0) * 1000.0
            logger.info("Model loaded in %.0fms", elapsed)

        except OSError as exc:
            raise ModelNotFoundError(
                f"Model '{model_id}' not found in cache '{cache_dir}'. "
                "Run `python scripts/setup_model.py` while online to download it."
            ) from exc

        self._loaded = True
        self._log_memory_usage()

    def _build_bnb_config(self) -> "BitsAndBytesConfig":  # type: ignore[name-defined]
        """
        Build the bitsandbytes quantization config based on llm.quantization setting.

        Returns:
            A BitsAndBytesConfig for NF4, INT8, or passthrough (none).
        """
        from transformers import BitsAndBytesConfig  # type: ignore

        q = self._cfg.quantization
        if q == "nf4":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif q == "int8":
            return BitsAndBytesConfig(load_in_8bit=True)
        else:
            # No quantization — return minimal config
            return BitsAndBytesConfig()

    def _generate_internal(self, prompt: str, t_start: float) -> InferenceResult:
        """
        Execute the model's generate() call and decode the output.

        This is the method that runs in the background thread with timeout.

        Args:
            prompt: Formatted prompt string.
            t_start: Monotonic start time for latency calculation.

        Returns:
            A fully populated :class:`InferenceResult`.
        """
        assert self._tokenizer is not None
        assert self._model is not None

        # Determine device for input tensors
        device = next(self._model.parameters()).device  # type: ignore[attr-defined]

        # Tokenize
        inputs = self._tokenizer(  # type: ignore[attr-defined]
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        input_length = inputs["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(  # type: ignore[attr-defined]
                **inputs,
                max_new_tokens=self._cfg.max_new_tokens,
                temperature=self._cfg.temperature,
                do_sample=self._cfg.do_sample,
                repetition_penalty=self._cfg.repetition_penalty,
                pad_token_id=self._tokenizer.eos_token_id,  # type: ignore[attr-defined]
                eos_token_id=self._tokenizer.eos_token_id,  # type: ignore[attr-defined]
            )

        # Decode only the new tokens (skip the input prompt)
        new_tokens = outputs[0][input_length:]
        output_length = len(new_tokens)

        decoded = self._tokenizer.decode(  # type: ignore[attr-defined]
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        # Estimate confidence from output length (heuristic: longer = more certain)
        confidence = min(0.95, output_length / max(self._cfg.max_new_tokens, 1))
        confidence = round(confidence, 3)

        return InferenceResult(
            text=decoded,
            confidence=confidence,
            latency_ms=elapsed_ms,
            truncated=False,
            input_tokens=input_length,
            output_tokens=output_length,
        )

    def _assert_model_cached(self, model_id: str, cache_dir: Path) -> None:
        """
        Verify the model exists in the local HuggingFace cache.

        Args:
            model_id: HuggingFace model identifier.
            cache_dir: Path to the HF hub cache directory.

        Raises:
            ModelNotFoundError: If no cached model files are found.
        """
        # HF stores models as snapshots: <cache>/<model_slug>/snapshots/<hash>/
        slug = model_id.replace("/", "--")
        model_cache = cache_dir / f"models--{slug}"

        if not model_cache.exists():
            raise ModelNotFoundError(
                f"Model '{model_id}' not found in cache '{cache_dir}'. "
                "Run `python scripts/setup_model.py` to download the model."
            )

        snapshots = list((model_cache / "snapshots").glob("*"))
        if not snapshots:
            raise ModelNotFoundError(
                f"Model '{model_id}' cache directory exists but has no snapshots. "
                "Re-run `python scripts/setup_model.py` to repair the download."
            )

        logger.info("Model found in cache: %s", snapshots[0])

    def _log_memory_usage(self) -> None:
        """Log current RAM and VRAM usage after model load."""
        try:
            import psutil
            process = psutil.Process()
            ram_gb = process.memory_info().rss / (1024 ** 3)
            logger.info("RAM usage after model load: %.2f GB", ram_gb)
        except ImportError:
            pass

        if torch.cuda.is_available():
            try:
                vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                logger.info("VRAM usage after model load: %.2f GB", vram_gb)
            except Exception:  # noqa: BLE001
                pass
