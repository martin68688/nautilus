# Proven Hyperparameters for Small-Data Transformer Fine-tuning

Concrete values from a successful DeBERTa-v3-large fine-tuning run on a
short-text classification task (final validation log loss: 0.0776).

## Model Configuration

- **Backbone**: DeBERTa-v3-large
- **Unfrozen layers**: Last 8 layers
- **Classification head**: Linear, on [CLS] token output

## Optimizer and Learning Rates

- **Backbone learning rate**: 2e-5
- **Head learning rate**: 5e-5
- **Gradient clipping**: Enabled
- **LR schedule**: Cosine annealing with warm restarts

## Regularization and Training Control

- **Label smoothing**: Enabled
- **Early stopping**: Patience 5 epochs, monitor validation log loss
- **Max epochs observed**: 22 (converged well within this budget)

## Validation Strategy

- **Split**: First fold of a stratified 5-fold split
