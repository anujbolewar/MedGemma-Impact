"""
llm/model_loader.py — MedGemma 4-bit NF4 quantized model loader.

Provides a functional API for loading, caching, and unloading
``google/medgemma-4b-it`` with 4-bit NF4 quantization via bitsandbytes.
Gracefully degrades to 8-bit quantization, then fp16, if bitsandbytes is
unavailable or does not support 4-bit.

VRAM is checked before loading: if free GPU memory is below 3.5 GB the
loader warns and falls back to ``device_map='cpu'``.  Available system RAM
is also checked and logged.  Memory footprint is measured and logged after
the model is resident.  A warm-up pass JIT-compiles CUDA kernels so the
first real inference is fast.

All logging goes through :func:`core.logger.get_logger` (NWSLogger JSONL
structured logger).  No internet access is required after initial model
setup — ``local_files_only`` is *not* forced here to allow first-time
downloads via ``scripts/setup_model.py``.

Public API
----------
``load_model(model_path)``
    Load + quantize the model; populate the module-level cache; return
    ``(AutoModelForCausalLM, AutoTokenizer)``.

``get_model()``
    Return the cached singleton or call ``load_model(C.MODEL_ID)``.

``unload_model()``
    Free all GPU / CPU memory and reset the module-level cache.

``warm_up(model, tokenizer)``
    One dummy forward pass (prompt ``'Hello.'``) to JIT-compile kernels;
    warm-up latency is logged via :meth:`~core.logger.NWSLogger.perf`.
"""

from __future__ import annotations

import gc
import sys
import time
from typing import TYPE_CHECKING, Optional, Tuple

from core.logger import get_logger
from core.constants import C  # SentinelConstants — C.MODEL_ID, C.MAX_RAM_GB, …

if TYPE_CHECKING:  # imported only by type-checkers, never at runtime
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

# ── Module-level singleton ────────────────────────────────────────────────────
_model: Optional[object] = None
_tokenizer: Optional[object] = None
_loaded_path: Optional[str] = None

# Minimum free VRAM (GB) required to attempt GPU loading.
_MIN_VRAM_GB: float = 3.5

# Fixed warm-up prompt — must not be changed (system contract).
_WARM_UP_PROMPT: str = "Hello."


# ── Public API ────────────────────────────────────────────────────────────────

def load_model(
    model_path: str,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load MedGemma from *model_path* with 4-bit NF4 quantization.

    The function populates the module-level cache so subsequent calls to
    :func:`get_model` return the already-loaded instance without re-loading.

    Parameters
    ----------
    model_path:
        HuggingFace model identifier or absolute local path, e.g.
        ``'google/medgemma-4b-it'``.

    Returns
    -------
    tuple[AutoModelForCausalLM, AutoTokenizer]
        The loaded model set to ``eval()`` mode and the matching tokenizer.

    Notes
    -----
    * If free VRAM < 3.5 GB a **warning** is logged and ``device_map='cpu'``
      is used instead of ``'auto'``.
    * bitsandbytes :class:`ImportError` is handled gracefully:

      1. NF4 4-bit  (preferred)
      2. 8-bit       (bitsandbytes installed but 4-bit creation fails)
      3. fp16        (bitsandbytes not installed at all)
    """
    global _model, _tokenizer, _loaded_path
    log = get_logger()

    # ── Return cached instance without re-loading ─────────────────────────
    if _model is not None and _tokenizer is not None:
        log.info("llm", "model_cache_hit", {"model_path": model_path})
        return _model, _tokenizer  # type: ignore[return-value]

    log.info("llm", "load_model_start", {"model_path": model_path})

    # ── Hardware pre-flight ───────────────────────────────────────────────
    device_map = _check_hardware(log)

    # ── Quantization config ───────────────────────────────────────────────
    bnb_config, quant_label = _build_quant_config(log)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    log.info("llm", "tokenizer_loading", {"model_path": model_path})
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        padding_side="left",
    )

    # ── Model ─────────────────────────────────────────────────────────────
    load_kwargs: dict = {
        "device_map": device_map,
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
    }
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config

    log.info(
        "llm",
        "model_loading",
        {"model_path": model_path, "quant": quant_label, "device_map": device_map},
    )
    t0 = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()
    elapsed_ms = (time.monotonic() - t0) * 1000.0

    log.perf(
        "llm",
        "model_loaded",
        latency_ms=elapsed_ms,
        data={"model_path": model_path, "quant": quant_label},
    )

    # ── Memory footprint ──────────────────────────────────────────────────
    _log_memory_footprint(model, log)

    # ── Populate cache ────────────────────────────────────────────────────
    _model = model
    _tokenizer = tokenizer
    _loaded_path = model_path

    return model, tokenizer  # type: ignore[return-value]


def get_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Return the cached ``(model, tokenizer)`` singleton.

    Calls :func:`load_model` with ``core.constants.C.MODEL_ID`` on first
    access.

    Returns
    -------
    tuple[AutoModelForCausalLM, AutoTokenizer]
    """
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer  # type: ignore[return-value]
    return load_model(C.MODEL_ID)


def unload_model() -> None:
    """
    Delete the cached model / tokenizer and release all memory.

    * Sets module-level globals to ``None``.
    * Calls ``torch.cuda.empty_cache()`` if CUDA is available.
    * Runs ``gc.collect()`` to reclaim CPU tensors immediately.

    Safe to call multiple times — a no-op when nothing is loaded.
    """
    global _model, _tokenizer, _loaded_path
    log = get_logger()

    if _model is None:
        log.info("llm", "unload_noop", {"reason": "model_not_loaded"})
        return

    log.info("llm", "unload_start", {"loaded_path": _loaded_path})
    del _model
    del _tokenizer
    _model = None
    _tokenizer = None
    _loaded_path = None

    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log.info("llm", "cuda_cache_cleared", {})
    except ImportError:
        pass

    gc.collect()
    log.info("llm", "unload_complete", {})


def warm_up(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> None:
    """
    Run one dummy forward pass to JIT-compile CUDA kernels.

    Uses the fixed prompt ``'Hello.'`` as specified by the system contract.
    Warm-up latency is logged via :meth:`~core.logger.NWSLogger.perf`.

    Parameters
    ----------
    model:
        A loaded ``AutoModelForCausalLM`` instance.
    tokenizer:
        The matching ``AutoTokenizer`` instance.
    """
    log = get_logger()
    log.info("llm", "warm_up_start", {"prompt": _WARM_UP_PROMPT})

    try:
        import torch  # type: ignore

        inputs = tokenizer(_WARM_UP_PROMPT, return_tensors="pt")

        # Resolve the device from the model's first parameter tensor.
        try:
            device = next(model.parameters()).device  # type: ignore[union-attr]
        except StopIteration:
            device = torch.device("cpu")

        inputs = {k: v.to(device) for k, v in inputs.items()}

        t0 = time.monotonic()
        with torch.no_grad():
            model.generate(  # type: ignore[union-attr]
                **inputs,
                max_new_tokens=1,
                do_sample=False,
            )
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        log.perf(
            "llm",
            "warm_up_done",
            latency_ms=elapsed_ms,
            data={"prompt": _WARM_UP_PROMPT},
        )

    except Exception as exc:  # noqa: BLE001
        log.error("llm", "warm_up_failed", {"error": str(exc)})


# ── Private helpers ───────────────────────────────────────────────────────────

def _check_hardware(log) -> str:  # type: ignore[no-untyped-def]
    """
    Inspect free VRAM and available RAM; return the appropriate ``device_map``.

    VRAM policy
    ~~~~~~~~~~~
    * No CUDA device present          → ``'cpu'``
    * Free VRAM < 3.5 GB              → warn + ``'cpu'``
    * Free VRAM ≥ 3.5 GB              → ``'auto'``

    RAM policy
    ~~~~~~~~~~
    Checked and logged against ``C.MAX_RAM_GB``; does **not** change
    ``device_map`` — the OS will page if RAM is tight.

    Returns
    -------
    str
        ``'auto'`` or ``'cpu'``.
    """
    device_map = "auto"

    # ── VRAM check ────────────────────────────────────────────────────────
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            log.info("llm", "vram_check", {"cuda": False, "device_map": "cpu"})
            device_map = "cpu"
        else:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device=0)
            free_gb = free_bytes / (1024 ** 3)
            total_gb = total_bytes / (1024 ** 3)

            if free_gb < _MIN_VRAM_GB:
                log.warn(
                    "llm",
                    "vram_insufficient",
                    {
                        "free_gb": round(free_gb, 2),
                        "total_gb": round(total_gb, 2),
                        "threshold_gb": _MIN_VRAM_GB,
                        "fallback": "device_map=cpu",
                    },
                )
                device_map = "cpu"
            else:
                log.info(
                    "llm",
                    "vram_ok",
                    {
                        "free_gb": round(free_gb, 2),
                        "total_gb": round(total_gb, 2),
                        "device_map": "auto",
                    },
                )

    except ImportError:
        log.warn("llm", "torch_unavailable", {"fallback": "device_map=cpu"})
        device_map = "cpu"

    # ── RAM check ─────────────────────────────────────────────────────────
    try:
        import psutil  # type: ignore

        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)

        if available_gb < C.MAX_RAM_GB:
            log.warn(
                "llm",
                "ram_low",
                {
                    "available_gb": round(available_gb, 2),
                    "total_gb": round(total_gb, 2),
                    "threshold_gb": C.MAX_RAM_GB,
                },
            )
        else:
            log.info(
                "llm",
                "ram_ok",
                {
                    "available_gb": round(available_gb, 2),
                    "total_gb": round(total_gb, 2),
                },
            )

    except ImportError:
        log.info("llm", "ram_check_skipped", {"reason": "psutil_not_installed"})

    return device_map


def _build_quant_config(log) -> Tuple[Optional[object], str]:  # type: ignore[no-untyped-def]
    """
    Build a ``BitsAndBytesConfig`` for NF4 4-bit quantization.

    Graceful fallback order
    ~~~~~~~~~~~~~~~~~~~~~~~
    1. **NF4 4-bit** — ``load_in_4bit=True``, ``bnb_4bit_quant_type='nf4'``,
       ``bnb_4bit_compute_dtype=torch.float16``,
       ``bnb_4bit_use_double_quant=True``.
    2. **8-bit** — bitsandbytes installed but 4-bit config creation raises.
    3. **fp16 / no quantization** — bitsandbytes not installed at all.

    Returns
    -------
    tuple[BitsAndBytesConfig | None, str]
        ``(config, label)`` where *label* is one of
        ``'nf4_4bit'``, ``'int8'``, or ``'fp16_no_quant'``.
    """
    try:
        import bitsandbytes  # noqa: F401 — validate availability first
        import torch  # type: ignore
        from transformers import BitsAndBytesConfig  # type: ignore

        try:
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            log.info(
                "llm",
                "quant_config",
                {
                    "mode": "nf4_4bit",
                    "compute_dtype": "float16",
                    "double_quant": True,
                },
            )
            return config, "nf4_4bit"

        except Exception as inner_exc:  # noqa: BLE001
            # bitsandbytes present but 4-bit creation failed — try 8-bit.
            log.warn(
                "llm",
                "quant_4bit_failed",
                {"error": str(inner_exc), "fallback": "int8"},
            )
            config = BitsAndBytesConfig(load_in_8bit=True)
            log.info("llm", "quant_config", {"mode": "int8_fallback"})
            return config, "int8"

    except ImportError as exc:
        log.warn(
            "llm",
            "bitsandbytes_unavailable",
            {"error": str(exc), "fallback": "fp16_no_quant"},
        )
        return None, "fp16_no_quant"


def _log_memory_footprint(model: object, log) -> None:  # type: ignore[no-untyped-def]
    """
    Log the model's memory footprint in GB after loading.

    Resolution order
    ~~~~~~~~~~~~~~~~
    1. ``model.get_memory_footprint()``  (transformers ≥ 4.35)
    2. ``torch.cuda.memory_allocated()`` (GPU VRAM allocated)
    3. ``sys.getsizeof(model)``          (very rough CPU approximation)
    """
    # Preferred: transformers built-in helper (returns bytes).
    try:
        footprint_bytes: int = model.get_memory_footprint()  # type: ignore[union-attr]
        footprint_gb = footprint_bytes / (1024 ** 3)
        log.info(
            "llm",
            "model_footprint",
            {"footprint_gb": round(footprint_gb, 3), "source": "get_memory_footprint"},
        )
        return
    except (AttributeError, TypeError):
        pass

    # Fallback 1: CUDA allocated bytes.
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            allocated_bytes = torch.cuda.memory_allocated()
            footprint_gb = allocated_bytes / (1024 ** 3)
            log.info(
                "llm",
                "model_footprint",
                {"footprint_gb": round(footprint_gb, 3), "source": "cuda_allocated"},
            )
            return
    except ImportError:
        pass

    # Fallback 2: sys.getsizeof (very approximate for CPU tensors).
    approx_gb = sys.getsizeof(model) / (1024 ** 3)
    log.info(
        "llm",
        "model_footprint",
        {"footprint_gb": round(approx_gb, 6), "source": "sys_getsizeof_approx"},
    )
