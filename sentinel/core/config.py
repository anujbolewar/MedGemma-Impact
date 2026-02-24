"""
sentinel/core/config.py — Typed configuration loader for NeuroWeave Sentinel.

Loads config/sentinel.yaml and validates all values into typed dataclasses.
All downstream modules import from this module; never read YAML directly.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Dataclass hierarchy — mirrors sentinel.yaml
# ──────────────────────────────────────────────


@dataclass(frozen=True)
class CameraConfig:
    """Configuration for the webcam capture device."""

    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass(frozen=True)
class GazeConfig:
    """Gaze tracking and blink detection tuning parameters."""

    dwell_threshold_ms: int = 800
    blink_ear_threshold: float = 0.21
    blink_consec_frames: int = 3
    long_blink_frames: int = 12
    gaze_smoothing_alpha: float = 0.4
    confidence_min: float = 0.6


@dataclass(frozen=True)
class SymbolBoardConfig:
    """Symbol board grid layout configuration."""

    columns: int = 3
    rows: int = 3
    pages: int = 3
    cell_padding_px: int = 8


@dataclass(frozen=True)
class LLMConfig:
    """MedGemma inference engine configuration."""

    model_id: str = "google/medgemma-4b-it"
    cache_dir: str = "~/.cache/huggingface/hub"
    quantization: str = "nf4"
    max_new_tokens: int = 60
    temperature: float = 0.3
    do_sample: bool = True
    repetition_penalty: float = 1.1
    latency_budget_ms: int = 2300
    device_map: str = "auto"

    @property
    def resolved_cache_dir(self) -> Path:
        """Return the cache directory as an absolute Path, expanding ~ if needed."""
        return Path(os.path.expanduser(self.cache_dir))


@dataclass(frozen=True)
class TTSConfig:
    """Text-to-speech engine configuration."""

    rate: int = 150
    volume: float = 1.0
    voice_id: Optional[str] = None


@dataclass(frozen=True)
class EmergencyConfig:
    """Emergency override configuration."""

    cooldown_seconds: int = 10
    messages: tuple[str, ...] = (
        "EMERGENCY — I need immediate help now",
        "I am in pain — please come immediately",
        "Please call the nurse station",
        "I cannot breathe properly",
    )
    audio_dir: str = "assets/audio"
    use_tts_fallback: bool = True

    @property
    def resolved_audio_dir(self) -> Path:
        """Return audio directory as an absolute Path."""
        return Path(self.audio_dir)


@dataclass(frozen=True)
class UIConfig:
    """UI display configuration."""

    fullscreen: bool = False
    font_family: str = "Arial"
    sentence_font_size: int = 36
    board_font_size: int = 18
    status_font_size: int = 12
    theme: str = "dark"
    show_camera_feed: bool = True
    show_latency_indicator: bool = True
    camera_preview_width: int = 240
    camera_preview_height: int = 180


@dataclass(frozen=True)
class LoggingConfig:
    """Logging and session recording configuration."""

    level: str = "INFO"
    log_file: str = "logs/sentinel.log"
    max_bytes: int = 10_485_760
    backup_count: int = 3
    log_sessions: bool = True


@dataclass(frozen=True)
class SentinelConfig:
    """Root configuration object — single source of truth for all settings."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    symbol_board: SymbolBoardConfig = field(default_factory=SymbolBoardConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    emergency: EmergencyConfig = field(default_factory=EmergencyConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ──────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────


def _merge(defaults: dict, overrides: dict) -> dict:
    """
    Deep-merge *overrides* into *defaults*, returning a new dict.

    Nested dicts are merged recursively; scalar values in overrides win.

    Args:
        defaults: Base dictionary of default values.
        overrides: Override values loaded from YAML.

    Returns:
        A new dict with overrides applied on top of defaults.
    """
    result: dict = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Path | str | None = None) -> SentinelConfig:
    """
    Load, validate, and return a SentinelConfig from a YAML file.

    The search order for the config file is:
    1. *config_path* argument (if provided)
    2. SENTINEL_CONFIG environment variable
    3. ``config/sentinel.yaml`` relative to this file's project root
    4. Built-in defaults (no file required)

    Args:
        config_path: Optional path to a ``sentinel.yaml`` file.

    Returns:
        A fully populated and frozen :class:`SentinelConfig` instance.

    Raises:
        ValueError: If a required YAML field has an invalid type or value.
        FileNotFoundError: If *config_path* is explicitly given but does not exist.
    """
    # Resolve path
    resolved_path: Path | None = None

    if config_path is not None:
        resolved_path = Path(config_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Config file not found: {resolved_path}")
    elif "SENTINEL_CONFIG" in os.environ:
        resolved_path = Path(os.environ["SENTINEL_CONFIG"])
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"SENTINEL_CONFIG points to missing file: {resolved_path}"
            )
    else:
        # Auto-discover: walk up from this file to find config/sentinel.yaml
        here = Path(__file__).resolve()
        for parent in [here.parent.parent.parent, here.parent.parent]:
            candidate = parent / "config" / "sentinel.yaml"
            if candidate.exists():
                resolved_path = candidate
                break

    raw: dict = {}
    if resolved_path is not None:
        logger.info("Loading config from: %s", resolved_path)
        with resolved_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file must be a YAML mapping, got: {type(loaded)}")
        raw = loaded
    else:
        logger.info("No config file found — using built-in defaults")

    # Build sub-configs from raw dict, falling back to defaults for missing keys
    try:
        camera_cfg = CameraConfig(**raw.get("camera", {}))
        gaze_cfg = GazeConfig(**raw.get("gaze", {}))
        board_cfg = SymbolBoardConfig(**raw.get("symbol_board", {}))
        llm_cfg = LLMConfig(**raw.get("llm", {}))
        tts_cfg = TTSConfig(**raw.get("tts", {}))

        # EmergencyConfig needs special handling: YAML lists → tuple
        emg_raw = dict(raw.get("emergency", {}))
        if "messages" in emg_raw and isinstance(emg_raw["messages"], list):
            emg_raw["messages"] = tuple(emg_raw["messages"])
        emg_cfg = EmergencyConfig(**emg_raw)

        ui_cfg = UIConfig(**raw.get("ui", {}))
        log_cfg = LoggingConfig(**raw.get("logging", {}))

    except TypeError as exc:
        raise ValueError(f"Invalid config value: {exc}") from exc

    # Validate ranges
    _validate_config(camera_cfg, gaze_cfg, llm_cfg, tts_cfg)

    config = SentinelConfig(
        camera=camera_cfg,
        gaze=gaze_cfg,
        symbol_board=board_cfg,
        llm=llm_cfg,
        tts=tts_cfg,
        emergency=emg_cfg,
        ui=ui_cfg,
        logging=log_cfg,
    )
    logger.debug("Config loaded: %s", config)
    return config


def _validate_config(
    camera: CameraConfig,
    gaze: GazeConfig,
    llm: LLMConfig,
    tts: TTSConfig,
) -> None:
    """
    Validate cross-field constraints on the loaded configuration.

    Raises:
        ValueError: If any configured value violates a hard constraint.
    """
    if not (0.0 < gaze.blink_ear_threshold < 1.0):
        raise ValueError(
            f"gaze.blink_ear_threshold must be in (0, 1), got {gaze.blink_ear_threshold}"
        )
    if not (0.0 < gaze.gaze_smoothing_alpha <= 1.0):
        raise ValueError(
            f"gaze.gaze_smoothing_alpha must be in (0, 1], got {gaze.gaze_smoothing_alpha}"
        )
    if llm.latency_budget_ms > 2500:
        raise ValueError(
            f"llm.latency_budget_ms must be ≤2500ms (hard constraint), got {llm.latency_budget_ms}"
        )
    if llm.max_new_tokens > 200:
        raise ValueError(
            f"llm.max_new_tokens capped at 200 for latency constraint, got {llm.max_new_tokens}"
        )
    if not (0.0 <= tts.volume <= 1.0):
        raise ValueError(f"tts.volume must be in [0, 1], got {tts.volume}")
    if llm.quantization not in {"nf4", "int8", "none"}:
        raise ValueError(
            f"llm.quantization must be 'nf4', 'int8', or 'none', got '{llm.quantization}'"
        )
    if camera.fps <= 0:
        raise ValueError(f"camera.fps must be positive, got {camera.fps}")
