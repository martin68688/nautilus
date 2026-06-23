# Attention-Weighted Multi-Layer Pooling

## Overview

Replacing single-layer CLS pooling with attention-weighted pooling over the
last few hidden layers consistently improves classification performance on
small text-classification tasks.

## Implementation

```python
class AttentionMultiLayerPooling(nn.Module):
    def __init__(self, hidden_size, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states_list, attention_mask):
        pooled = []
        for i in range(self.num_layers):
            mask = attention_mask.unsqueeze(-1).float()
            layer_hidden = hidden_states_list[-(i + 1)] * mask
            mean_pooled = layer_hidden.sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled.append(mean_pooled)
        stacked = torch.stack(pooled, dim=1)
        weights = torch.softmax(self.layer_weights, dim=0)
        weighted = (stacked * weights.unsqueeze(0).unsqueeze(-1)).sum(1)
        return self.classifier(self.dropout(weighted))
```

## Usage Notes

- Configure the transformer to output all hidden states (`output_hidden_states=True`).
- The layer weights are learned jointly with the rest of the model.
- Masked mean pooling correctly handles variable-length sequences.
