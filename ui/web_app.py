"""
ui/web_app.py — FastAPI web server for NeuroWeave Sentinel.

Serves the single-page patient dashboard at http://localhost:<port>/
and streams live pipeline events to the browser over a WebSocket at /ws.

REST endpoints
--------------
GET  /           HTML dashboard
GET  /health     JSON health check
GET  /state      Current FSM state + last event snapshot
POST /confirm    Patient confirmed output  {"approved": true|false}
POST /emergency  Manual emergency trigger  {}

WebSocket
---------
ws://<host>:<port>/ws

Messages pushed by server (JSON):
  {"type": "fsm_state",    "state": "SPEAKING", ...}
  {"type": "gaze",         "direction": "UP", "confidence": 0.92}
  {"type": "token",        "codes": [...], "prompt": "..."}
  {"type": "reconstruction","text": "...", "latency_ms": 980}
  {"type": "safety",       "action": "PROCEED", "confidence": 0.81, "reasons": []}
  {"type": "speaking",     "text": "..."}
  {"type": "fallback",     "text": "..."}
  {"type": "emergency",    "source": "gaze_dwell"}
  {"type": "confirmation", "text": "...", "confidence": 0.72}
  {"type": "tick",         "timestamp_ms": ...}   ← heartbeat every second
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core.logger import get_logger
from pipeline.controller import (
    ON_CONFIRMATION_NEEDED,
    ON_EMERGENCY,
    ON_FALLBACK,
    ON_RECONSTRUCTION,
    ON_SAFETY_DECISION,
    ON_SPEAKING,
    ON_TOKEN_SELECTED,
    NWSController,
)

_log = get_logger()

# ── Static file path ──────────────────────────────────────────────────────────
_STATIC_DIR = Path(__file__).parent / "static"

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="NeuroWeave Sentinel", version="1.0")
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ── Shared state ──────────────────────────────────────────────────────────────
_controller: Optional[NWSController] = None
_connected_clients: Set[WebSocket] = set()
_clients_lock = threading.Lock()

# Snapshot of the last known UI state (for /state endpoint + new WS connects)
_snapshot: Dict[str, Any] = {
    "fsm_state":    "IDLE",
    "gaze":         {"direction": "CENTRE", "confidence": 0.0},
    "tokens":       [],
    "sentence":     "",
    "safety_action": "",
    "safety_confidence": 0.0,
    "pending_confirmation": None,
    "emergency_active": False,
}

# asyncio event loop running in the uvicorn thread
_loop: Optional[asyncio.AbstractEventLoop] = None


# ── WebSocket helpers ─────────────────────────────────────────────────────────

def _push(msg: Dict[str, Any]) -> None:
    """Thread-safe push of a JSON message to every connected WebSocket client."""
    if _loop is None:
        return
    asyncio.run_coroutine_threadsafe(_broadcast(msg), _loop)


async def _broadcast(msg: Dict[str, Any]) -> None:
    text = json.dumps(msg)
    with _clients_lock:
        dead: List[WebSocket] = []
        for ws in _connected_clients:
            try:
                await ws.send_text(text)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            _connected_clients.discard(ws)


# ── EventBus → WebSocket bridge ───────────────────────────────────────────────

def _wire_controller(ctrl: NWSController) -> None:
    """Register all EventBus callbacks so the controller feeds the WS stream."""
    global _controller
    _controller = ctrl

    # FSM transition hook (controller calls _on_fsm_transition internally,
    # but we tap into it via the published events below for UI-relevant changes)

    ctrl.subscribe(ON_TOKEN_SELECTED, _on_token_selected)
    ctrl.subscribe(ON_RECONSTRUCTION, _on_reconstruction)
    ctrl.subscribe(ON_SAFETY_DECISION, _on_safety_decision)
    ctrl.subscribe(ON_SPEAKING, _on_speaking)
    ctrl.subscribe(ON_FALLBACK, _on_fallback)
    ctrl.subscribe(ON_EMERGENCY, _on_emergency)
    ctrl.subscribe(ON_CONFIRMATION_NEEDED, _on_confirmation_needed)

    # Patch controller's _on_fsm_transition to also push state changes to WS
    _original_fsm_cb = ctrl._on_fsm_transition

    def _patched_fsm_cb(from_state, to_state, reason):
        _original_fsm_cb(from_state, to_state, reason)
        _snapshot["fsm_state"] = to_state.value
        if to_state.value == "IDLE":
            _snapshot["pending_confirmation"] = None
        _push({"type": "fsm_state", "state": to_state.value,
               "from": from_state.value, "reason": reason})

    ctrl._fsm._external_callback = _patched_fsm_cb

    # Patch signal fuser to stream gaze ticks at ~5 Hz (every 6th frame)
    _original_fused = ctrl._fuser.get_fused_frame
    _tick_counter = [0]

    def _instrumented_get_fused_frame():
        frame = _original_fused()
        _tick_counter[0] += 1
        if _tick_counter[0] % 6 == 0:
            gaze_data = {
                "direction": frame.primary_direction,
                "confidence": round(frame.composite_confidence, 3),
            }
            _snapshot["gaze"] = gaze_data
            _push({"type": "gaze", **gaze_data})
        return frame

    ctrl._fuser.get_fused_frame = _instrumented_get_fused_frame

    _log.info("web_app", "controller_wired", {})


def _on_token_selected(data: Dict[str, Any]) -> None:
    _snapshot["tokens"] = data.get("token_codes", [])
    _push({"type": "token",
           "codes": data.get("token_codes", []),
           "prompt": data.get("prompt_string", ""),
           "confidence": data.get("confidence", 0)})


def _on_reconstruction(data: Dict[str, Any]) -> None:
    _snapshot["sentence"] = data.get("text", "")
    _push({"type": "reconstruction", **data})


def _on_safety_decision(data: Dict[str, Any]) -> None:
    _snapshot["safety_action"] = data.get("action", "")
    _snapshot["safety_confidence"] = data.get("composite_confidence", 0)
    _push({"type": "safety", **data})


def _on_speaking(data: Dict[str, Any]) -> None:
    _push({"type": "speaking", "text": data.get("text", "")})


def _on_fallback(data: Dict[str, Any]) -> None:
    _snapshot["sentence"] = data.get("text", "")
    _push({"type": "fallback", "text": data.get("text", "")})


def _on_emergency(data: Dict[str, Any]) -> None:
    _snapshot["emergency_active"] = True
    _push({"type": "emergency", **data})


def _on_confirmation_needed(data: Dict[str, Any]) -> None:
    _snapshot["pending_confirmation"] = data
    _push({"type": "confirmation", **data})


# ── Heartbeat ─────────────────────────────────────────────────────────────────

async def _heartbeat() -> None:
    """Push a tick message every second so the client can detect disconnects."""
    while True:
        await asyncio.sleep(1.0)
        _push({"type": "tick", "timestamp_ms": round(time.time() * 1000),
               "fsm_state": _snapshot["fsm_state"]})


# ── App lifecycle ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _on_startup() -> None:
    global _loop
    _loop = asyncio.get_running_loop()
    asyncio.create_task(_heartbeat())
    _log.info("web_app", "startup", {})


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the single-page dashboard."""
    html_path = _STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health() -> JSONResponse:
    ctrl_ok = _controller is not None
    return JSONResponse({
        "status": "ok" if ctrl_ok else "controller_not_ready",
        "fsm_state": _snapshot["fsm_state"],
        "clients": len(_connected_clients),
    })


@app.get("/state")
async def state() -> JSONResponse:
    return JSONResponse(_snapshot)


@app.post("/confirm")
async def confirm(body: Dict[str, Any] = {}) -> JSONResponse:
    if _controller is None:
        return JSONResponse({"error": "controller not ready"}, status_code=503)
    approved = bool(body.get("approved", True))
    threading.Thread(
        target=_controller.confirm_output,
        args=(approved,),
        daemon=True,
    ).start()
    return JSONResponse({"ok": True, "approved": approved})


@app.post("/emergency")
async def emergency() -> JSONResponse:
    if _controller is None:
        return JSONResponse({"error": "controller not ready"}, status_code=503)
    from core.constants import FSMState
    from core.fsm import InvalidTransitionError
    from output.emergency import EmergencyOverride
    override: EmergencyOverride = _controller._emergency_override
    threading.Thread(target=override.trigger, daemon=True).start()
    return JSONResponse({"ok": True})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    with _clients_lock:
        _connected_clients.add(ws)

    # Send current snapshot on connect
    await ws.send_text(json.dumps({
        "type": "snapshot",
        **_snapshot,
    }))
    _log.info("web_app", "ws_connected", {"total": len(_connected_clients)})

    try:
        while True:
            # Keep connection alive; client messages are handled below
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
                await _handle_client_msg(data, ws)
            except Exception:  # noqa: BLE001
                pass
    except WebSocketDisconnect:
        pass
    finally:
        with _clients_lock:
            _connected_clients.discard(ws)
        _log.info("web_app", "ws_disconnected", {"total": len(_connected_clients)})


async def _handle_client_msg(data: Dict[str, Any], ws: WebSocket) -> None:
    """Handle incoming WS messages from browser (confirm/cancel/emergency)."""
    action = data.get("action")
    if action == "confirm" and _controller:
        _controller.confirm_output(True)
    elif action == "cancel" and _controller:
        _controller.confirm_output(False)
    elif action == "emergency" and _controller:
        override = _controller._emergency_override
        threading.Thread(target=override.trigger, daemon=True).start()


# ── Public launcher ───────────────────────────────────────────────────────────

def start_web_server(
    controller: NWSController,
    host: str = "0.0.0.0",
    port: int = 7860,
) -> None:
    """
    Wire *controller* to the WS bridge and start uvicorn in the current thread.

    Blocking — call from a dedicated thread if the controller is already running.

    Args:
        controller: Fully initialised :class:`~pipeline.controller.NWSController`.
        host:       Bind address (default ``0.0.0.0`` — all interfaces).
        port:       TCP port (default ``7860``).
    """
    _wire_controller(controller)

    import uvicorn  # type: ignore
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    _log.info("web_app", "server_start", {"host": host, "port": port})
    server.run()
