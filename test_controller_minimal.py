#!/usr/bin/env python3
"""
Minimal controller test — verifies NWSController initializes without errors.
Skips the long model load step by mocking it.
"""

import sys
import unittest.mock as mock

from pipeline.controller import NWSController

def test_controller_init():
    """Test that NWSController can be instantiated."""
    config = {
        "use_sim": True,
        "sim_mode": "RANDOM",
        "model_path": "mock/path",
    }
    
    # Mock the expensive model load
    mock_model = mock.Mock()
    mock_tokenizer = mock.Mock()
    mock_tokenizer.eos_token_id = 0
    
    with mock.patch('llm.model_loader.load_model', return_value=(mock_model, mock_tokenizer)):
        with mock.patch('llm.model_loader.warm_up'):
            ctrl = NWSController(config)
            print("✓ Controller initialized")
            print(f"✓ FSM state: {ctrl._fsm.current_state.value}")
            print(f"✓ Gaze tracker type: {type(ctrl._gaze_tracker).__name__}")
            print(f"✓ Signal fuser active channels: {ctrl._fuser.active_channels}")
            ctrl.shutdown()
            print("✓ Controller shutdown clean")
    
    return 0

if __name__ == "__main__":
    sys.exit(test_controller_init())
