# Proven Configurations for Small-Data Transformer Fine-tuning

Detailed configurations distilled from successful fine-tuning runs on small
NLP text-classification datasets where validation log loss is the objective.

## Table of Contents
1. [Hybrid Model Architecture](#hybrid-model-architecture)
2. [Confidence-Penalty Loss](#confidence-penalty-loss)
3. [Leakage-Free Feature Pipeline](#leakage-free-feature-pipeline)
4. [Multi-Sample Dropout Head](#multi-sample-dropout-head)
5. [OneCycleLR Configuration](#onecyclelr-configuration)
6. [Stochastic Weight Averaging](#stochastic-weight-averaging)
7. [On-the-fly WordNet Augmentation](#on-the-fly-wordnet-augmentation)
8. [Offline Back-Translation Augmentation](#offline-back-translation-augmentation)
9. [Cross-Fold Prediction Averaging](#cross-fold-prediction-averaging)
10. [Regularization Defaults](#regularization-defaults)

## Hybrid Model Architecture

**When to use:** The task has strong stylistic, lexical, or domain-specific
signals (e.g., authorship attribution, sentiment) that a transformer may
underweight on small datasets.

Combine a pre-trained transformer's [CLS] embedding with handcrafted features:

```python
import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self, transformer, num_features, num_classes, proj_dim=64):
        super().__init__()
        self.transformer = transformer
        self.feature_proj = nn.Sequential(
            nn.Linear(num_features, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        hidden = transformer.config.hidden_size + proj_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask, features):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        feat_proj = self.feature_proj(features)
        combined = torch.cat([cls_embedding, feat_proj], dim=-1)
        return self.classifier(combined)
```

## Confidence-Penalty Loss

```python
import torch
import torch.nn.functional as F

def confidence_penalty_loss(logits, labels, penalty_weight=0.5):
    """Cross-entropy + KL divergence vs. uniform to improve calibration."""
    ce_loss = F.cross_entropy(logits, labels)
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    uniform = torch.ones_like(probs) / probs.size(-1)
    kl = (uniform * (uniform.clamp(min=1e-8).log() - log_probs)).sum(dim=-1).mean()
    return ce_loss + penalty_weight * kl
```

## Leakage-Free Feature Pipeline

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

def extract_and_standardize_features(train_texts, test_texts, feature_fn):
    """Extract features independently, standardize test with train statistics."""
    train_features = np.array([feature_fn(text) for text in train_texts])
    test_features = np.array([feature_fn(text) for text in test_texts])

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)  # fit on train only
    test_features = scaler.transform(test_features)         # apply to test
    return train_features, test_features
```

## Multi-Sample Dropout Head

```python
class MultiSampleDropoutHead(nn.Module):
    def __init__(self, hidden_dim, num_classes, dropout_rates=(0.1, 0.15, 0.2, 0.25)):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_rates])

    def forward(self, x):
        x = self.proj(x)
        logits = torch.mean(torch.stack([self.classifier(d(x)) for d in self.dropouts]), dim=0)
        return logits
```

## OneCycleLR Configuration

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[2e-5, 5e-5],
    epochs=num_epochs,
    steps_per_epoch=1,
    pct_start=0.1,
    anneal_strategy='cos',
    final_div_factor=1000
)
scheduler.step()
```

**Regularization settings that broke a plateau at 0.6379 → 0.5133 log loss**:
- Dropout: 0.2 (reduced from 0.3)
- Label smoothing: 0.06 (reduced from 0.15)

## Stochastic Weight Averaging

```python
from torch.optim.swa_utils import AveragedModel, SWALR

swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=1e-6)

for epoch in range(num_epochs):
    if epoch >= 15 and epoch % 5 == 0:
        swa_model.update_parameters(model)
        swa_scheduler.step()
    else:
        normal_scheduler.step()

torch.optim.swa_utils.update_bn(train_loader, swa_model)
```

## On-the-fly WordNet Augmentation

```python
from nltk.corpus import wordnet
import random

def wordnet_augment(tokens, replace_prob=0.5, target_fraction=0.15, stopwords=set()):
    non_stop = [i for i, t in enumerate(tokens) if t.lower() not in stopwords]
    n_to_replace = max(1, int(len(non_stop) * target_fraction))
    indices = random.sample(non_stop, min(n_to_replace, len(non_stop)))
    for i in indices:
        if random.random() < replace_prob:
            syns = wordnet.synsets(tokens[i])
            lemmas = {l.name().replace('_', ' ') for s in syns for l in s.lemmas()
                      if l.name().lower() != tokens[i].lower()}
            if lemmas:
                tokens[i] = random.choice(list(lemmas))
    return tokens
```

## Offline Back-Translation Augmentation

1. Load Helsinki-NLP OpusMT models for round-trip translation.
2. For each training sentence, generate round-trip translations through each pair.
3. Apply augmentation with ~15% probability per sample.
4. Store augmented dataset offline.
5. During training, load from the pre-computed file.

## Cross-Fold Prediction Averaging

```python
test_preds = np.zeros((len(test_texts), num_classes))
for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, train_labels)):
    fold_preds = predict(model, test_texts)
    test_preds += fold_preds / skf.n_splits
```

## Regularization Defaults

| Technique           | Configuration                          |
|---------------------|----------------------------------------|
| Label smoothing     | 0.05 (default); 0.06 if plateau occurs |
| Early stopping      | Patience 3–5, monitor validation log loss|
| Cross-validation    | 5-fold StratifiedKFold                 |
| Dropout             | 0.2 (reduce from >0.3 if plateau)      |
| Weight decay        | 0.01                                   |
| Gradient clipping   | max_norm=1.0                           |