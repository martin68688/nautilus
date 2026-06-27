# Architecture Patterns for Small-Data Fine-tuning

## Table of Contents
1. [Mean Pooling](#mean-pooling)
2. [BiLSTM Head](#bilstm-head)
3. [Multi-Scale Convolutional Features](#multi-scale-convolutional-features)
4. [Attention Pooling](#attention-pooling)
5. [Gated Fusion](#gated-fusion)
6. [Combined Label Smoothing + Focal Loss](#combined-loss)

## Mean Pooling

Replace CLS-only pooling with mean pooling over all token embeddings:
```python
mask = attention_mask.unsqueeze(-1).float()
pooled = (last_hidden_state * mask).sum(1) / mask.sum(1)
```

## BiLSTM Head

1-layer BiLSTM (hidden_dim=128) with residual connection and layer norm:
```python
class BiLSTMHead(nn.Module):
    def __init__(self, hidden_dim, num_classes, lstm_hidden=128):
        super().__init__()
        self.bilstm = nn.LSTM(hidden_dim, lstm_hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim, 2 * lstm_hidden)
        self.norm = nn.LayerNorm(2 * lstm_hidden)
        self.classifier = nn.Linear(2 * lstm_hidden, num_classes)

    def forward(self, pooled):
        x = pooled.unsqueeze(1)
        lstm_out, _ = self.bilstm(x)
        out = self.norm(lstm_out.squeeze(1) + self.proj(pooled))
        return self.classifier(out)
```

## Multi-Scale Convolutional Features

Kernel sizes 2, 3, 5, 7 with adaptive max pooling for local n-gram sensitivity.

## Attention Pooling

```python
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

    def forward(self, hidden_states, attention_mask=None):
        attn_out, _ = self.attn(hidden_states, hidden_states, hidden_states,
                                key_padding_mask=attention_mask)
        return attn_out.mean(dim=1)
```

## Gated Fusion

For tasks where hand-crafted features complement transformer embeddings:
```python
# gate dynamically weights transformer vs crafted features per sample
fused = gate * transformer_proj + (1 - gate) * crafted_proj
```

Use multi-head cross-attention with sigmoid gating and residual connection for advanced fusion.

## Combined Label Smoothing + Focal Loss

```python
class CombinedLoss(nn.Module):
    def __init__(self, label_smoothing=0.05, focal_gamma=2.0):
        self.ls = LabelSmoothingCrossEntropy(eps=label_smoothing)
        self.fl = FocalLoss(gamma=focal_gamma)
    def forward(self, logits, targets):
        return self.ls(logits, targets) + self.fl(logits, targets)
```