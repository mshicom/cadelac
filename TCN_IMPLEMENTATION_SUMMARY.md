# TCN Encoder Support Implementation Summary

This document summarizes the implementation of TCN (Temporal Convolutional Network) encoder support for the CaDeLaC pipeline, allowing side-by-side comparison between LSTM and TCN encoders.

## Changes Made

### 1. Dependency Addition (`cadelac_env.yml`)
- Added `git+https://github.com/paul-krug/pytorch-tcn.git` to the pip dependencies
- Users can install it with: `pip install git+https://github.com/paul-krug/pytorch-tcn.git`

### 2. Model Implementation (`cadelac/learning/models/context_aware_delan.py`)

#### New TCNModel Class
- Implemented `TCNModel` class similar to existing `LSTMModel`
- Uses `pytorch_tcn.tcn.TCN` as the backbone
- Default architecture: `[32, 32, 16, 16]` channels
- Configurable kernel size (default: 3) and dropout (default: 0.1)
- Handles input shape transformation from `(batch, seq_len, features)` to `(batch, features, seq_len)`

#### ContextAwareDeLaN Updates
- Added `history_encoder` parameter: `'lstm'`, `'tcn'`, or `'none'`
- Maintains backward compatibility: defaults to `'lstm'` when not specified
- Encoder selection logic in `__init__()`:
  - `'lstm'`: Creates `LSTMModel` (existing behavior)
  - `'tcn'`: Creates `TCNModel` with configurable parameters
  - `'none'`: No encoder, uses zeros for encoding
- Fixed device initialization issue by setting `self.device` from `self._eye.device`
- Updated `forward()` and `inv_dyn()` to handle `None` encoder case

### 3. Training Script Updates (`cadelac/learning/train_panda.py`)

#### CLI Arguments
- Added `--encoder` or `-e` argument:
  - Default: `'lstm'`
  - Choices: `'lstm'`, `'tcn'`, `'none'`
  - Example: `python -m cadelac.learning.train_panda -e tcn`

#### Hyperparameters
- `history_encoder` field now stored in checkpoint's `hyper` dict
- Enables automatic encoder type detection when loading models

#### Model Naming
- Model names now include encoder type when `hist_length > 0`
- Format: `epochs_{N}_noise__enc_{encoder}_{dataset}.torch`
- Example: `epochs_1000_noise__enc_tcn_panda_mj_101_rand_envs_20_runs_50Hz_lqr.torch`

### 4. Control Wrapper Updates (`cadelac/control/l4c_context_aware_delan.py`)

#### Import Updates
- Added `TCNModel` to imports

#### L4CContextAwareDeLaN Updates
- Added `history_encoder` attribute (defaults to `'lstm'` for backward compatibility)
- Encoder loading logic:
  - Detects encoder type from checkpoint's `hyper` dict
  - Loads appropriate encoder class (LSTM or TCN)
  - Handles `'none'` case
- Maintains `self.lstm` attribute for backward compatibility

#### New Methods
- `eval_hist_encoder_np(input_np)`: Unified method for evaluating any history encoder
  - Works with LSTM, TCN, or no encoder
  - Handles device placement (CPU/CUDA)
  - Returns numpy array
- `eval_lstm_np(input_np)`: Backward compatibility wrapper, calls `eval_hist_encoder_np()`

### 5. MPC Simulation Updates (`cadelac/control/main_mpc.py`)

#### CLI Interface
Added argparse support for comparison mode:
```bash
# Regular mode (unchanged)
python -m cadelac.control.main_mpc

# Comparison mode
python -m cadelac.control.main_mpc --compare \
    --lstm-model path/to/lstm_model.torch \
    --tcn-model path/to/tcn_model.torch \
    --seed 42 \
    --sim-time 10.0
```

#### Comparison Features
- Loads both LSTM and TCN models
- Runs simulations with identical conditions:
  - Same random seed
  - Same initial state
  - Same simulation time
- Generates separate logs with suffixes: `_lstm_comparison` and `_tcn_comparison`
- Prints comprehensive comparison summary:
  - Total simulation time
  - Average step time
  - Average solver time
  - RMS joint tracking error per joint
  - Mean RMS error
  - Differences between models

#### Example Output
```
COMPARISON SUMMARY
============================================================

Simulation Time: 10.0s
Random Seed: 42

LSTM Model:
  Total Time: 15.234s
  Avg Step Time: 0.0305s
  Avg Solver Time: 0.0120s
  RMS Joint Tracking Error: [0.0012, 0.0015, ...]
  Mean RMS Error: 0.001345 rad

TCN Model:
  Total Time: 14.987s
  Avg Step Time: 0.0300s
  Avg Solver Time: 0.0115s
  RMS Joint Tracking Error: [0.0011, 0.0014, ...]
  Mean RMS Error: 0.001289 rad

Difference (TCN - LSTM):
  Mean RMS Error Diff: -0.000056 rad
  Solver Time Diff: -0.0005s
```

## Usage Examples

### Training with TCN Encoder
```bash
# Train a new model with TCN encoder
python -m cadelac.learning.train_panda -l 0 -e tcn

# Evaluate a TCN model
python -m cadelac.learning.train_panda -l 1 -e tcn
```

### Training with LSTM Encoder (Default)
```bash
# Train with LSTM (explicit)
python -m cadelac.learning.train_panda -l 0 -e lstm

# Train with LSTM (implicit, backward compatible)
python -m cadelac.learning.train_panda -l 0
```

### Side-by-Side Comparison
```bash
python -m cadelac.control.main_mpc --compare \
    --lstm-model learning/trained_models/res_model/panda/ContextAware/epochs_1000_noise__enc_lstm_panda_mj_101_rand_envs_20_runs_50Hz_lqr.torch \
    --tcn-model learning/trained_models/res_model/panda/ContextAware/epochs_1000_noise__enc_tcn_panda_mj_101_rand_envs_20_runs_50Hz_lqr.torch \
    --seed 42 \
    --sim-time 20.0
```

## Backward Compatibility

All changes maintain backward compatibility:

1. **Existing LSTM models**: Work without modification
2. **Training script**: Default behavior unchanged (uses LSTM)
3. **Control wrapper**: Automatically detects encoder type from checkpoint
4. **Main MPC**: Can run in original mode without comparison

## Testing

All functionality has been validated with integration tests:
- ✓ TCN model creation and forward pass
- ✓ LSTM model creation and forward pass  
- ✓ ContextAwareDeLaN with LSTM encoder
- ✓ ContextAwareDeLaN with TCN encoder
- ✓ Backward compatibility (defaults to LSTM)
- ✓ No history case (hist_length=0)

## Technical Details

### TCN Architecture
- Default channels: `[32, 32, 16, 16]`
- Kernel size: 3
- Dropout: 0.1
- Input: `(batch, seq_len, features)`
- Output: `(batch, n_enc_input)`

### Device Handling
- Proper device initialization in ContextAwareDeLaN
- Consistent device placement in encoder evaluation
- CUDA support maintained

### Model Checkpoint Format
Checkpoints now include:
```python
{
    'epoch': int,
    'hyper': {
        'history_encoder': 'lstm' | 'tcn' | 'none',
        'n_enc_input': int,
        'hist_length': int,
        'n_lstm_input': int,
        'n_lstm_hidden': int,  # LSTM only
        'n_lstm_depth': int,    # LSTM only
        'tcn_channels': list,   # TCN only
        'tcn_kernel_size': int, # TCN only
        'tcn_dropout': float,   # TCN only
        ...
    },
    'state_dict': OrderedDict(...)
}
```

## Files Modified

1. `cadelac_env.yml` - Added pytorch-tcn dependency
2. `cadelac/learning/models/context_aware_delan.py` - TCN model and encoder selection
3. `cadelac/learning/train_panda.py` - CLI args and encoder type storage
4. `cadelac/control/l4c_context_aware_delan.py` - Unified encoder interface
5. `cadelac/control/main_mpc.py` - Comparison mode and CLI interface

## Notes

- TCN import is done lazily to avoid dependency issues when TCN is not needed
- The `lstm` attribute name is kept for backward compatibility even when using TCN
- All encoder evaluation happens in Python (not CasADi) as specified in requirements
