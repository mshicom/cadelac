# Quick Reference: TCN Encoder Support

## Training Commands

### Train with TCN encoder
```bash
python -m cadelac.learning.train_panda -l 0 -e tcn
```

### Train with LSTM encoder (default)
```bash
python -m cadelac.learning.train_panda -l 0
# or explicitly:
python -m cadelac.learning.train_panda -l 0 -e lstm
```

### Evaluate a trained model
```bash
python -m cadelac.learning.train_panda -l 1 -e tcn
```

## Side-by-Side Comparison

### Basic comparison
```bash
python -m cadelac.control.main_mpc --compare \
    --lstm-model learning/trained_models/res_model/panda/ContextAware/lstm_model.torch \
    --tcn-model learning/trained_models/res_model/panda/ContextAware/tcn_model.torch
```

### With custom parameters
```bash
python -m cadelac.control.main_mpc --compare \
    --lstm-model path/to/lstm_model.torch \
    --tcn-model path/to/tcn_model.torch \
    --seed 42 \
    --sim-time 20.0 \
    --use-viewer  # Optional: enable MuJoCo viewer
```

## Regular Simulation (No Comparison)

```bash
python -m cadelac.control.main_mpc --sim-time 10.0
```

## Model Checkpoint Format

Models now include encoder type in hyperparameters:
```python
checkpoint = {
    'epoch': 1000,
    'hyper': {
        'history_encoder': 'tcn',  # or 'lstm' or 'none'
        'hist_length': 15,
        'n_enc_input': 10,
        # ... other hyperparameters
    },
    'state_dict': { ... }
}
```

## Programmatic Usage

### Load and use TCN model
```python
import torch
from cadelac.control.l4c_context_aware_delan import L4CContextAwareDeLaN

# Load checkpoint
checkpoint = torch.load('model.torch', weights_only=False)

# Create wrapper (automatically detects encoder type)
delan = L4CContextAwareDeLaN(checkpoint, n_dof=7, 
                             n_enc_input=checkpoint['hyper']['n_enc_input'])

# Evaluate encoder
import numpy as np
hist_input = np.random.randn(15, 21)  # (hist_length, n_features)
encoding = delan.eval_hist_encoder_np(hist_input)
```

### Create model with specific encoder
```python
from cadelac.learning.models.context_aware_delan import ContextAwareDeLaN

hyper = {
    'history_encoder': 'tcn',  # or 'lstm' or 'none'
    'hist_length': 15,
    'n_enc_input': 10,
    'n_lstm_input': 21,
    # ... other parameters
}

model = ContextAwareDeLaN(n_dof=7, **hyper)
```

## Encoder Parameters

### LSTM (existing)
- `n_lstm_hidden`: Hidden size
- `n_lstm_depth`: Number of layers
- `n_lstm_input`: Input feature size

### TCN (new)
- `tcn_channels`: List of channel sizes (default: [32, 32, 16, 16])
- `tcn_kernel_size`: Convolution kernel size (default: 3)
- `tcn_dropout`: Dropout rate (default: 0.1)
- `n_lstm_input`: Input feature size (reused for consistency)

## Backward Compatibility

- Old checkpoints without `history_encoder` field automatically use LSTM
- The `eval_lstm_np()` method still works (calls `eval_hist_encoder_np()`)
- Training without `-e` flag defaults to LSTM
- All existing workflows continue to work unchanged
