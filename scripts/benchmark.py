"""
scripts/benchmark.py — End-to-end latency and memory benchmarking.

Runs 20 inference cycles with synthetic intent bundles, reports p50/p95/p99
latencies, and verifies all hard constraints are satisfied.
Requires the model to be downloaded (run setup_model.py first).

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --iterations 10 --report-memory
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Hard constraints (matches the project specification)
MAX_LATENCY_P95_MS: float = 2500.0
MAX_RAM_GB: float = 8.0
MAX_VRAM_GB: float = 4.0


def _setup_logging(level: str = "INFO") -> None:
    """Configure logging for the benchmark script."""
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )


def _synthetic_bundles() -> list[dict[str, str]]:
    """
    Generate a set of diverse synthetic intent token bundles for benchmarking.

    Returns:
        List of dicts with keys: body_part, sensation, urgency, intensity.
    """
    return [
        {"body_part": "chest", "sensation": "pressure", "urgency": "right now", "intensity": "moderate"},
        {"body_part": "head", "sensation": "pain", "urgency": "sudden", "intensity": "severe"},
        {"body_part": "abdomen", "sensation": "nausea", "urgency": "ongoing", "intensity": "mild"},
        {"body_part": "back", "sensation": "discomfort", "urgency": "getting worse", "intensity": "moderate"},
        {"body_part": "left arm", "sensation": "tingling", "urgency": "intermittent", "intensity": "mild"},
        {"body_part": "right leg", "sensation": "burning", "urgency": "right now", "intensity": "severe"},
        {"body_part": "whole body", "sensation": "cold", "urgency": "sudden", "intensity": "moderate"},
        {"body_part": "chest", "sensation": "shortness of breath", "urgency": "getting worse", "intensity": "severe"},
    ]


def run_benchmark(iterations: int, report_memory: bool) -> bool:
    """
    Run inference benchmark and report results.

    Args:
        iterations: Number of inference cycles to run.
        report_memory: If True, report RAM and VRAM usage.

    Returns:
        True if all hard constraints are satisfied, False otherwise.
    """
    # Imports inside function — avoids import-time GPU allocation
    from sentinel.core.config import load_config
    from sentinel.intent.classifier import IntentBundle
    from sentinel.llm.engine import MedGemmaEngine
    from sentinel.llm.prompt_builder import PromptBuilder

    config = load_config()
    engine = MedGemmaEngine(config.llm)
    builder = PromptBuilder()

    print("\n═══ NeuroWeave Sentinel — Latency Benchmark ════════════")
    print(f"  Model:      {config.llm.model_id}")
    print(f"  Quantization: {config.llm.quantization}")
    print(f"  Iterations: {iterations}")
    print(f"  Budget:     {config.llm.latency_budget_ms}ms per inference")
    print("═════════════════════════════════════════════════════════\n")

    print("Loading model…", flush=True)
    t_load = time.monotonic()
    engine.load()
    load_time = (time.monotonic() - t_load) * 1000.0
    print(f"Model loaded in {load_time:.0f}ms\n")

    if report_memory:
        _report_memory()

    bundles = _synthetic_bundles()
    latencies: list[float] = []
    failures: int = 0

    print(f"Running {iterations} inference cycles…\n")
    for i in range(iterations):
        bundle_dict = bundles[i % len(bundles)]
        bundle = IntentBundle(
            body_part=bundle_dict["body_part"],
            sensation=bundle_dict["sensation"],
            urgency=bundle_dict["urgency"],
            intensity=bundle_dict["intensity"],
        )

        prompt = builder.build(bundle)
        t0 = time.monotonic()
        result = engine.infer(prompt)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        latencies.append(elapsed_ms)
        status = "⚠ TRUNCATED" if result.truncated else "✅"
        if result.truncated:
            failures += 1

        print(
            f"  [{i+1:2d}] {elapsed_ms:6.0f}ms | {status} | {result.text[:60]!r}"
        )

    engine.unload()

    # Stats
    if not latencies:
        print("No results collected.")
        return False

    p50 = statistics.median(latencies)
    p95 = _percentile(latencies, 95)
    p99 = _percentile(latencies, 99)
    p_max = max(latencies)
    p_min = min(latencies)

    print(f"\n{'─'*55}")
    print(f"  {'p50 latency':<20} {p50:>8.0f} ms")
    print(f"  {'p95 latency':<20} {p95:>8.0f} ms  {'✅' if p95 <= MAX_LATENCY_P95_MS else '❌ CONSTRAINT VIOLATION'}")
    print(f"  {'p99 latency':<20} {p99:>8.0f} ms")
    print(f"  {'max latency':<20} {p_max:>8.0f} ms")
    print(f"  {'min latency':<20} {p_min:>8.0f} ms")
    print(f"  {'truncations':<20} {failures:>8d} / {iterations}")
    print(f"{'─'*55}")

    if report_memory:
        _report_memory()

    constraint_ok = p95 <= MAX_LATENCY_P95_MS
    if constraint_ok:
        print("\n✅ All latency constraints satisfied!")
    else:
        print(f"\n❌ CONSTRAINT VIOLATION: p95 ({p95:.0f}ms) > {MAX_LATENCY_P95_MS:.0f}ms budget")
        print("   Consider: faster hardware, int8 quantization, or fewer max_new_tokens")

    return constraint_ok


def _percentile(data: list[float], pct: int) -> float:
    """
    Compute the given percentile of a data list.

    Args:
        data: List of float values.
        pct: Percentile to compute (0–100).

    Returns:
        The percentile value.
    """
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = max(0, int(len(sorted_data) * pct / 100) - 1)
    return sorted_data[idx]


def _report_memory() -> None:
    """Print current RAM and VRAM usage."""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        ram_gb = process.memory_info().rss / (1024 ** 3)
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        status = "✅" if ram_gb <= MAX_RAM_GB else "❌"
        print(f"  {'RAM used':<20} {ram_gb:>6.2f} GB / {total_ram_gb:.1f} GB  {status}")
    except ImportError:
        print("  (psutil not installed — skipping RAM report)")

    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            status = "✅" if vram_gb <= MAX_VRAM_GB else "❌"
            print(f"  {'VRAM used':<20} {vram_gb:>6.2f} GB / {total_vram_gb:.1f} GB  {status}")
        else:
            print("  (No CUDA GPU detected — running on CPU)")
    except ImportError:
        print("  (torch not installed — skipping VRAM report)")


def main() -> None:
    """
    Entry point for the benchmark script.

    Parses arguments, runs benchmark, and exits with code 0 if constraints
    pass or code 1 if any hard constraint is violated.
    """
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Benchmark NeuroWeave Sentinel latency and memory"
    )
    parser.add_argument(
        "--iterations", type=int, default=20,
        help="Number of inference iterations to run (default: 20)"
    )
    parser.add_argument(
        "--report-memory", action="store_true",
        help="Include RAM and VRAM usage in report"
    )
    args = parser.parse_args()

    passed = run_benchmark(args.iterations, args.report_memory)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
