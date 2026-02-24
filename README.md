# NeuroWeave Sentinel

> Offline assistive communication system for ALS / locked-in syndrome patients.  
> Eye-gaze input → Intent tokens → MedGemma sentence reconstruction → TTS speech.  
> **Kaggle MedGemma Impact Challenge 2025.**

---

## Architecture

```
Eye Gaze (MediaPipe Iris) ──┐
                            ├─→ Signal Fuser ─→ Intent Encoder ─→ MedGemma 4b-it ─→ Safety Pipeline ─→ TTS
Keyboard Simulator ─────────┘                    50-token vocab     NF4 4-bit quant    entropy+grammar    pyttsx3
                                                                    <2.3s cutoff        confidence score   offline
```

### Key constraints
| Constraint | Limit |
|------------|-------|
| End-to-end latency | ≤ 2.5 s |
| RAM | ≤ 8 GB |
| VRAM | ≤ 4 GB |
| Internet at runtime | ❌ None |

---

## Folder Structure

```
neuroweave-sentinel/
├── core/
│   ├── __init__.py
│   ├── fsm.py            # Finite state machine (9 states, validated transitions)
│   ├── constants.py      # All system constants — single source of truth
│   └── logger.py         # JSONL structured session logger
├── input/
│   ├── __init__.py
│   ├── gaze_webcam.py    # MediaPipe Iris webcam tracker (EMA smoothed)
│   ├── gaze_sim.py       # Keyboard simulator (identical interface)
│   └── signal_fuser.py   # BlinkDetector + DwellDetector → FusedSignal
├── encoder/
│   ├── __init__.py
│   ├── token_vocab.py    # 50-token AAC vocabulary (3 pages × 4×4 grid)
│   └── intent_encoder.py # Stateful token accumulator → IntentBundle
├── llm/
│   ├── __init__.py
│   ├── model_loader.py   # MedGemma NF4 loader with cache validation
│   └── reconstructor.py  # Chat-format prompt + threaded inference + timeout
├── safety/
│   ├── __init__.py
│   ├── entropy_guard.py  # Per-token softmax entropy check
│   ├── grammar_check.py  # LanguageTool + regex fallback grammar validation
│   └── confidence.py     # Composite 4-component confidence scorer
├── output/
│   ├── __init__.py
│   ├── tts_engine.py     # Non-blocking pyttsx3 worker thread
│   └── emergency.py      # Zero-LLM emergency broadcaster with audit log
├── ui/
│   ├── __init__.py
│   ├── main_window.py    # Tkinter 3-panel main window (thread-safe via after())
│   └── widgets.py        # GazeProgressRing, SymbolCell, ConfidenceBadge, StatusBar
├── pipeline/
│   ├── __init__.py
│   └── controller.py     # Full pipeline orchestrator + FSM + UI event binding
├── logs/                 # Runtime JSONL session logs
├── assets/               # Audio files, icons
├── tests/
│   └── __init__.py
├── requirements.txt
├── main.py               # CLI entry point
└── README.md
```

---

## Quick Start

### 1. Install dependencies (Python 3.11+)

```bash
pip install -r requirements.txt
```

### 2. Download MedGemma model (one-time, internet required)

```bash
# Accept terms at https://huggingface.co/google/medgemma-4b-it first
HUGGINGFACE_TOKEN=hf_your_token python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('google/medgemma-4b-it', token='hf_your_token')
AutoModelForCausalLM.from_pretrained('google/medgemma-4b-it', token='hf_your_token')
print('Model downloaded.')
"
```

### 3. Run in simulator mode (no webcam needed)

```bash
python main.py --mode simulator
```

### 4. Run with webcam

```bash
python main.py --mode webcam
```

### 5. Fullscreen (clinical deployment)

```bash
python main.py --mode webcam --fs
```

---

## Simulator Controls

| Key | Action |
|-----|--------|
| Arrow keys | Move gaze cursor |
| Space | Short blink → select highlighted cell |
| `b` | Long blink → next symbol board page |
| F1 | Emergency override |
| Escape | Reset intent bundle |
| `r` | Reset gaze to centre |

---

## AAC Token Vocabulary

| Page | Category | Tokens |
|------|----------|--------|
| 0 | BODY_PART | head, chest, abdomen, back, left/right arm/leg, throat, mouth, eyes, ears, skin, whole body, bladder |
| 1 | SENSATION | pain, pressure, burning, tingling, nausea, shortness of breath, cramping, cold, hot, itching, numbness, dizziness, thirst, hunger, discomfort |
| 2 | URGENCY | right now, sudden, getting worse, ongoing, intermittent, just started, better soon, unsure |
| 2 | INTENSITY | mild, moderate, severe, unbearable, very mild, extreme, improving, unknown |

Complete selection (one from each category) triggers MedGemma inference.

---

## Safety Pipeline

Every LLM output passes through three gates before voicing:

1. **Entropy Guard** — Rejects high-entropy (uncertain) outputs
2. **Grammar Check** — LanguageTool (offline JRE) + regex structural check
3. **Confidence Scorer** — Composite of entropy (35%) + grammar (30%) + length (20%) + latency (15%)

Outputs below the 0.45 confidence threshold are replaced with a safe fallback message.

---

## Emergency Override

- Activated by: Emergency button · F1 key · Long dwell on emergency symbol
- **Bypasses the LLM entirely** — immediate TTS playback
- 10-second cooldown prevents accidental re-triggering
- All events logged to `logs/session.jsonl` with timestamp and source

---

## License

Apache 2.0 — see LICENSE for details.

> ⚠️ **Clinical Disclaimer**: This system is a research prototype for the Kaggle MedGemma Impact Challenge. It is not a certified medical device and must not be used as a substitute for professional clinical communication tools.
# MedGemma-Impact
