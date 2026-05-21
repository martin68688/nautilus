# Training Stability Trifecta: Label Smoothing + Gradient Clipping + AMP

## Finding
Three techniques work synergistically: label_smoothing=0.1, clip_grad_norm_=1.0, and AMP.

## Evidence
- ALL top solutions across ALL runs use all three simultaneously

## Mechanism
- Label Smoothing (0.1): softens targets [1,0,0]→[0.9,0.05,0.05], prevents overconfidence
- Gradient Clipping (1.0): prevents large gradient spikes, especially important with MSD
- AMP: 2x speed on A100 + implicit regularization from reduced precision + 30% memory savings

## Condition
Transformer fine-tuning on small datasets. Use all three together.
