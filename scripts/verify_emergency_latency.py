
import time
from unittest.mock import MagicMock
from output.emergency import EmergencyOverride
from output.tts_engine import TTSEngine
from core.fsm import SentinelFSM, FSMState

def verify_latency():
    # Setup mocks
    mock_tts = MagicMock(spec=TTSEngine)
    mock_fsm = MagicMock(spec=SentinelFSM)
    mock_fsm.current_state = FSMState.IDLE
    
    # Initialize
    emergency = EmergencyOverride(tts_engine=mock_tts, fsm=mock_fsm)
    
    print("Testing emergency trigger latency...")
    start_time = time.monotonic()
    emergency.trigger()
    end_time = time.monotonic()
    
    latency_ms = (end_time - start_time) * 1000.0
    print(f"Trigger call returned in {latency_ms:.2f}ms")
    
    # Check if fsm.transition was called
    mock_fsm.transition.assert_called_once_with(FSMState.EMERGENCY, "emergency_gaze_trigger")
    print("FSM transition verified.")
    
    # Check if TTS was called (either speak_file or speak)
    assert mock_tts.speak.called or mock_tts.speak_file.called
    print("TTS broadcast verified.")
    
    if latency_ms < 500:
        print("LATENCY CHECK PASSED (< 500ms)")
    else:
        print("LATENCY CHECK FAILED (>= 500ms)")

if __name__ == "__main__":
    verify_latency()
