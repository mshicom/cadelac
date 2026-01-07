#!/usr/bin/env python3
"""
Simple integration test for TCN encoder support in CaDeLaC.
Tests that the models can be created and perform forward passes.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add cadelac to path
sys.path.insert(0, str(Path(__file__).parent))

from cadelac.learning.models.context_aware_delan import ContextAwareDeLaN, LSTMModel, TCNModel

def test_tcn_model_creation():
    """Test that TCN model can be created and perform forward pass."""
    print("Testing TCN model creation...")
    
    input_size = 21  # 7 DOF * 3 (qp, qv, diff_tau)
    output_size = 10
    hist_length = 15
    batch_size = 4
    
    # Create TCN model
    tcn = TCNModel(input_size, output_size)
    
    # Create dummy input
    x = torch.randn(batch_size, hist_length, input_size)
    
    # Forward pass
    output = tcn(x)
    
    assert output.shape == (batch_size, output_size), f"Expected shape {(batch_size, output_size)}, got {output.shape}"
    print("✓ TCN model creation and forward pass successful")
    return True

def test_lstm_model_creation():
    """Test that LSTM model still works."""
    print("Testing LSTM model creation...")
    
    input_size = 21
    hidden_size = 10
    output_size = 10
    num_layers = 5
    hist_length = 15
    batch_size = 4
    
    # Create LSTM model
    lstm = LSTMModel(input_size, hidden_size, output_size, num_layers)
    
    # Create dummy input
    x = torch.randn(batch_size, hist_length, input_size)
    
    # Forward pass
    output = lstm(x)
    
    assert output.shape == (batch_size, output_size), f"Expected shape {(batch_size, output_size)}, got {output.shape}"
    print("✓ LSTM model creation and forward pass successful")
    return True

def test_context_aware_delan_with_lstm():
    """Test ContextAwareDeLaN with LSTM encoder."""
    print("Testing ContextAwareDeLaN with LSTM encoder...")
    
    n_dof = 7
    hist_length = 15
    n_lstm_input = n_dof * 3
    n_lstm_hidden = 10
    n_enc_input = 10
    n_lstm_depth = 5
    
    hyper = {
        'diagonal_epsilon': 0.1,
        'activation': 'Tanh',
        'net_arch_inertia': [30, 20],
        'net_arch_pot': [30, 20],
        'n_enc_input': n_enc_input,
        'n_lstm_hidden': n_lstm_hidden,
        'n_lstm_input': n_lstm_input,
        'n_lstm_depth': n_lstm_depth,
        'hist_length': hist_length,
        'history_encoder': 'lstm',
        'act_ld': 'Softplus',
    }
    
    # Create model
    model = ContextAwareDeLaN(n_dof, **hyper)
    
    # Create dummy inputs
    batch_size = 4
    q = torch.randn(batch_size, n_dof)
    qd = torch.randn(batch_size, n_dof)
    qdd = torch.randn(batch_size, n_dof)
    lstm_input = torch.randn(batch_size, hist_length, n_lstm_input)
    
    # Forward pass
    tau_pred, dEdt = model(q, qd, qdd, lstm_input)
    
    assert tau_pred.shape == (batch_size, n_dof), f"Expected tau shape {(batch_size, n_dof)}, got {tau_pred.shape}"
    assert dEdt.shape == (batch_size,), f"Expected dEdt shape {(batch_size,)}, got {dEdt.shape}"
    print("✓ ContextAwareDeLaN with LSTM encoder successful")
    return True

def test_context_aware_delan_with_tcn():
    """Test ContextAwareDeLaN with TCN encoder."""
    print("Testing ContextAwareDeLaN with TCN encoder...")
    
    n_dof = 7
    hist_length = 15
    n_lstm_input = n_dof * 3  # Still using this name for compatibility
    n_enc_input = 10
    
    hyper = {
        'diagonal_epsilon': 0.1,
        'activation': 'Tanh',
        'net_arch_inertia': [30, 20],
        'net_arch_pot': [30, 20],
        'n_enc_input': n_enc_input,
        'n_lstm_input': n_lstm_input,
        'hist_length': hist_length,
        'history_encoder': 'tcn',
        'tcn_channels': [32, 32, 16, 16],
        'tcn_kernel_size': 3,
        'tcn_dropout': 0.1,
        'act_ld': 'Softplus',
    }
    
    # Create model
    model = ContextAwareDeLaN(n_dof, **hyper)
    
    # Create dummy inputs
    batch_size = 4
    q = torch.randn(batch_size, n_dof)
    qd = torch.randn(batch_size, n_dof)
    qdd = torch.randn(batch_size, n_dof)
    hist_input = torch.randn(batch_size, hist_length, n_lstm_input)
    
    # Forward pass
    tau_pred, dEdt = model(q, qd, qdd, hist_input)
    
    assert tau_pred.shape == (batch_size, n_dof), f"Expected tau shape {(batch_size, n_dof)}, got {tau_pred.shape}"
    assert dEdt.shape == (batch_size,), f"Expected dEdt shape {(batch_size,)}, got {dEdt.shape}"
    print("✓ ContextAwareDeLaN with TCN encoder successful")
    return True

def test_context_aware_delan_backward_compatibility():
    """Test that models without history_encoder specified default to LSTM."""
    print("Testing backward compatibility (no history_encoder specified)...")
    
    n_dof = 7
    hist_length = 15
    n_lstm_input = n_dof * 3
    n_lstm_hidden = 10
    n_enc_input = 10
    n_lstm_depth = 5
    
    # Create hyper dict WITHOUT history_encoder (should default to LSTM)
    hyper = {
        'diagonal_epsilon': 0.1,
        'activation': 'Tanh',
        'net_arch_inertia': [30, 20],
        'net_arch_pot': [30, 20],
        'n_enc_input': n_enc_input,
        'n_lstm_hidden': n_lstm_hidden,
        'n_lstm_input': n_lstm_input,
        'n_lstm_depth': n_lstm_depth,
        'hist_length': hist_length,
        'act_ld': 'Softplus',
    }
    
    # Create model - should default to LSTM
    model = ContextAwareDeLaN(n_dof, **hyper)
    
    # Verify it's using LSTM
    assert model.history_encoder == 'lstm', f"Expected default encoder to be 'lstm', got {model.history_encoder}"
    assert model.lstm is not None, "Expected LSTM to be initialized"
    assert isinstance(model.lstm, LSTMModel), f"Expected LSTMModel, got {type(model.lstm)}"
    
    print("✓ Backward compatibility test successful (defaults to LSTM)")
    return True

def test_no_history_case():
    """Test that hist_length=0 works correctly."""
    print("Testing hist_length=0 case...")
    
    n_dof = 7
    
    hyper = {
        'diagonal_epsilon': 0.1,
        'activation': 'Tanh',
        'net_arch_inertia': [30, 20],
        'net_arch_pot': [30, 20],
        'n_enc_input': 1,
        'hist_length': 0,
        'act_ld': 'Softplus',
    }
    
    # Create model
    model = ContextAwareDeLaN(n_dof, **hyper)
    
    # Create dummy inputs
    batch_size = 4
    q = torch.randn(batch_size, n_dof)
    qd = torch.randn(batch_size, n_dof)
    qdd = torch.randn(batch_size, n_dof)
    
    # Forward pass (no history input)
    tau_pred, dEdt = model(q, qd, qdd)
    
    assert tau_pred.shape == (batch_size, n_dof), f"Expected tau shape {(batch_size, n_dof)}, got {tau_pred.shape}"
    assert dEdt.shape == (batch_size,), f"Expected dEdt shape {(batch_size,)}, got {dEdt.shape}"
    print("✓ No history (hist_length=0) test successful")
    return True

def run_all_tests():
    """Run all integration tests."""
    print("="*60)
    print("Running TCN Integration Tests")
    print("="*60)
    
    tests = [
        test_tcn_model_creation,
        test_lstm_model_creation,
        test_context_aware_delan_with_lstm,
        test_context_aware_delan_with_tcn,
        test_context_aware_delan_backward_compatibility,
        test_no_history_case,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print()
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print()
    print("="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
