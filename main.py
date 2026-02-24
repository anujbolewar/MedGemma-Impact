"""
main.py — NeuroWeave Sentinel application entry point.

Parses CLI args, runs system pre-flight checks, and wires the pipeline
controller with the Tkinter UI (or runs headless for testing).
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import traceback

# ──────────────────────────────────────────────────────────────
# ASCII banner
# ──────────────────────────────────────────────────────────────

_BANNER = r"""
  _   _                      _    _
 | \ | | ___ _   _ _ __ ___ | |  | | __ _ _____ __
 |  \| |/ _ \ | | | '__/ _ \| |  | |/ _` |_  / '_ \
 | |\  |  __/ |_| | | | (_) | |__| | (_| |/ /| | | |
 |_| \_|\___|\__,_|_|  \___/ \____/ \__,_/___|_| |_|
  ____            _   _             _
 / ___|___ _ __ | |_(_)_ __   ___| |
 \___ / _ | '_ \| __| | '_ \ / _ | |
  ___  __/| | | | |_| | | | |  __| |
 |____\___|_| |_|\__|_|_| |_|\___|_|

             NeuroWeave Sentinel  v1.0
     AAC Medical Communication System
"""


# ──────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sentinel",
        description="NeuroWeave Sentinel — gaze-driven AAC medical communication",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["webcam", "sim"],
        default="sim",
        help="Input mode: 'webcam' for live camera, 'sim' for keyboard simulator",
    )
    p.add_argument(
        "--demo",
        choices=["pain", "water", "emergency", "full"],
        default=None,
        help="Pre-scripted demo sequence (sim mode only)",
    )
    p.add_argument(
        "--no-gui",
        action="store_true",
        help="Run headless — no Tkinter window (useful for testing)",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARN"],
        default="INFO",
        help="Minimum log level for stderr output",
    )
    p.add_argument(
        "--model-path",
        default="./models/medgemma-4b",
        help="Path to locally downloaded MedGemma-4B weights",
    )
    p.add_argument(
        "--web",
        action="store_true",
        help="Start the FastAPI web UI server instead of Tkinter GUI",
    )
    p.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for the web UI server (default 7860)",
    )
    return p


# ──────────────────────────────────────────────────────────────
# Pre-flight checks
# ──────────────────────────────────────────────────────────────

def _check_python() -> None:
    """Abort if Python version is below 3.11."""
    if sys.version_info < (3, 11):
        print(
            f"[ERROR] Python 3.11+ required; running {sys.version}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[OK] Python {sys.version.split()[0]}")


def _check_resources(model_path: str) -> None:
    """Warn if RAM / VRAM are below limits and check model path."""
    from core.constants import SentinelConstants as C

    # RAM + VRAM — uses SentinelConstants.validate() which logs warnings
    C.validate()

    # Model path
    if os.path.isdir(model_path):
        print(f"[OK] Model path found: {model_path}")
    else:
        print(
            f"[WARN] Model path not found: {model_path!r}\n"
            "       Run: huggingface-cli download google/medgemma-4b-it "
            f"--local-dir {model_path}",
            file=sys.stderr,
        )


# ──────────────────────────────────────────────────────────────
# Controller stub (used when pipeline/controller is not yet wired)
# ──────────────────────────────────────────────────────────────

class _ControllerConfig:
    """Simple config bag passed to the controller."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.mode       = args.mode
        self.demo       = args.demo
        self.no_gui     = args.no_gui
        self.log_level  = args.log_level
        self.model_path = args.model_path


def _load_controller(cfg: _ControllerConfig):
    """Import and instantiate the pipeline controller."""
    try:
        from pipeline.controller import NWSController  # type: ignore
        # Convert _ControllerConfig object to dict for NWSController
        config_dict = {
            "use_sim":    cfg.mode == "sim",
            "sim_mode":   "SCRIPTED" if cfg.demo else "RANDOM",
            "demo_name":  cfg.demo,
            "no_gui":     cfg.no_gui,
            "model_path": cfg.model_path,
            "log_level":  cfg.log_level,
        }
        return NWSController(config_dict)
    except ImportError as exc:
        # Controller may not be built yet — return a minimal stub
        print(f"[WARN] pipeline.controller not available: {exc} — using stub")

        class _Stub:
            def run(self): pass
            def shutdown(self): pass
            def get_window_hooks(self): return {}

        return _Stub()


# ──────────────────────────────────────────────────────────────
# GUI entry point
# ──────────────────────────────────────────────────────────────

def _run_with_gui(controller) -> int:
    """
    Start controller in a background thread and Tkinter in the main thread.

    Returns exit code.
    """
    import tkinter as tk
    from ui.main_window import SentinelMainWindow

    root = tk.Tk()
    root.withdraw()   # hide until fully built

    hooks = {}
    if hasattr(controller, "get_window_hooks"):
        hooks = controller.get_window_hooks()

    window = SentinelMainWindow(
        root,
        on_emergency  = hooks.get("on_emergency"),
        on_reset      = hooks.get("on_reset"),
        on_speak      = hooks.get("on_speak"),
        on_reject     = hooks.get("on_reject"),
        on_mode_toggle= hooks.get("on_mode_toggle"),
    )

    # Wire controller → window update callbacks if controller supports it
    if hasattr(controller, "set_ui"):
        controller.set_ui(window)

    # Start controller in background daemon thread
    ctrl_thread = threading.Thread(
        target=controller.run,
        name="nws-controller",
        daemon=True,
    )
    ctrl_thread.start()

    def _on_close() -> None:
        from core.logger import get_logger
        get_logger().info("main", "window_close", {})
        controller.shutdown()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.deiconify()

    try:
        window.run()   # blocks until window is closed
    except KeyboardInterrupt:
        _on_close()

    return 0


# ──────────────────────────────────────────────────────────────
# Headless entry point
# ──────────────────────────────────────────────────────────────

def _run_headless(controller) -> int:
    """Run the controller in the main thread (no UI). Returns exit code."""
    try:
        controller.run()
    except KeyboardInterrupt:
        pass
    finally:
        controller.shutdown()
    return 0


# ──────────────────────────────────────────────────────────────
# Web UI entry point
# ──────────────────────────────────────────────────────────────

def _run_web(controller, port: int = 7860) -> int:
    """
    Start the controller in a background thread and the FastAPI web server
    in the main thread.

    Open http://localhost:<port>/ in a browser to see the live dashboard.
    """
    from ui.web_app import start_web_server

    ctrl_thread = threading.Thread(
        target=controller.run,
        name="nws-controller",
        daemon=True,
    )
    ctrl_thread.start()

    print(f"[INFO] Web UI → http://localhost:{port}/")
    print("       Press Ctrl-C to stop.")

    try:
        start_web_server(controller, host="0.0.0.0", port=port)
    except KeyboardInterrupt:
        pass
    finally:
        controller.shutdown()
    return 0


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> int:
    """Application entry point. Returns process exit code."""
    print(_BANNER)

    parser = _build_parser()
    args = parser.parse_args()

    # 1. Python version check
    _check_python()

    # 2. Set up stdlib logging level (for NWSLogger stderr mirroring)
    import logging
    level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARN": logging.WARNING}
    logging.basicConfig(level=level_map.get(args.log_level, logging.INFO))

    # 3. Import core and log startup
    from core.logger import get_logger
    from core.constants import SentinelConstants as C
    log = get_logger()
    log.info("main", "args_parsed", {
        "mode": args.mode,
        "demo": args.demo,
        "no_gui": args.no_gui,
        "log_level": args.log_level,
        "model_path": args.model_path,
    })

    # 4. RAM / VRAM / model path checks
    _check_resources(args.model_path)

    # 5. Initialise controller
    cfg = _ControllerConfig(args)
    controller = _load_controller(cfg)
    log.info("main", "controller_ready", {"class": type(controller).__name__})

    # 6/7. Launch
    exit_code = 0
    try:
        if getattr(args, 'web', False):
            print(f"[INFO] Starting web UI — mode={args.mode} port={args.port}")
            exit_code = _run_web(controller, port=args.port)
        elif args.no_gui:
            print("[INFO] Running headless (--no-gui)")
            exit_code = _run_headless(controller)
        else:
            print(f"[INFO] Starting GUI — mode={args.mode}")
            exit_code = _run_with_gui(controller)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted — shutting down…")
        controller.shutdown()
    except Exception:                              # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        try:
            log.critical("main", "unhandled_exception", {"traceback": tb})
        except Exception:                          # noqa: BLE001
            pass
        exit_code = 1
    finally:
        try:
            from core.logger import get_logger as _gl
            _gl().flush()
        except Exception:                          # noqa: BLE001
            pass

    print(f"[INFO] Sentinel exited with code {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
