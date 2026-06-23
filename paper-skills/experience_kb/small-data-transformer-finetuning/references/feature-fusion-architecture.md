# Feature Fusion Architecture

## Overview

A pretrained transformer backbone processes the input text while a parallel branch
extracts and projects handcrafted numerical features. The two representations are
concatenated before the final classifier.

## Feature Extraction

Extract features such as:
- **TF-IDF n-grams** (word and character level)
- **Text statistics**: sentence length, word length distributions, punctuation frequency
- **POS tag patterns**: distribution of part-of-speech tags
- **Sentiment scores**: polarity and subjectivity

## Projection Network

```python
class FeatureProjection(nn.Module):
    def __init__(self, input_dim=1047, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )
```

## Fusion Model

```python
class FusionModel(nn.Module):
    def __init__(self, transformer, feat_dim, num_classes):
        super().__init__()
        self.transformer = transformer
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1))
        self.head = nn.Sequential(
            nn.Linear(transformer.config.hidden_size + 256, 512),
            nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, num_classes))

    def forward(self, input_ids, attention_mask, features):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        feat = self.feat_proj(features)
        return self.head(torch.cat([pooled, feat], dim=-1))
```

## Training Configuration

- **Backbone LR**: 1e-5
- **Head + Feature Branch LR**: 1e-4
- **Label smoothing**: 0.05
- **SWA**: start at 75% of total epochs
- **Early stopping**: patience 8 on validation log loss
