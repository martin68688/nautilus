# Advanced Fine-tuning Techniques

## EMA Re-registration After Unfreezing

When using EMA with staged fine-tuning (freeze backbone → train head → unfreeze backbone),
re-register EMA after unfreezing so all trainable parameters are tracked:

```python
# Stage 2: unfreeze backbone, re-register EMA
for param in model.backbone.parameters():
    param.requires_grad = True
ema = EMA(model)
ema.register()  # now captures all trainable params
```

## Nelder-Mead Optimization for Ensemble Weighting

```python
from scipy.optimize import minimize

def ensemble_log_loss(weights, preds_list, y_true):
    weighted = sum(w * p for w, p in zip(weights, preds_list))
    return log_loss(y_true, weighted)

result = minimize(ensemble_log_loss, x0=[1.0/len(preds_list)]*len(preds_list),
    args=(preds_list, y_val), method='Nelder-Mead')
```

## Diversify Model Architectures for Ensembles

| Architecture | Signal Captured |
|---|---|
| Transformer (DeBERTa-v3) | Deep contextual semantic representations |
| XGBoost on TF-IDF + stylistic features | Surface-level lexical patterns |
| CharCNN-LSTM | Character-level morphological patterns |

**Observed result:** RoBERTa/DeBERTa/ALBERT ensemble (0.4/0.3/0.3 weights)
achieved 0.340 log loss vs. best individual model at 0.541.

## Cross-Attention Fusion for Multi-Modal Features

When combining embeddings from different feature spaces:
- One feature type serves as the **query**
- The other serves as **keys/values**
- Add a **gated residual connection** on top of the attention output

## Multi-Scale Hidden-Layer Aggregation

```python
outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                     output_hidden_states=True)
selected = [outputs.hidden_states[i] for i in [4, 8, 12, -1]]
weights = torch.softmax(self.layer_weights, dim=0)
cls_repr = sum(w * h[:, 0, :] for w, h in zip(weights, selected))
```

## Gradient Noise Injection

```python
noise_sigma = 0.01 * (batch_size * accumulation_steps) ** (-0.5)
for param in model.parameters():
    if param.grad is not None:
        noise = torch.randn_like(param.grad) * noise_sigma
        param.grad.add_(noise)
```

## Progressive Unfreezing — Layer-Count Guard

```python
layers = getattr(model, attr, None)
if layers is None or n_layers_to_unfreeze > len(layers):
    return  # skip unfreezing; do not raise
```

ALBERT has a single shared layer group — a naive unfreezing function will throw index errors.

## OOM Resolution Strategy

1. **Remove AWP** — doubles peak memory.
2. **Remove auxiliary losses** — each adds a full extra forward pass.
3. **Downgrade model size** — large → base.
4. Only then reduce batch size or sequence length.

**Validated result**: Removing AWP + auxiliary losses + switching to base model
resolved OOM while achieving 0.3322 validation log loss.

## Stratified K-Fold Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(train_df), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
    model = build_model()
    train_loop(model, train_df.iloc[train_idx])
    oof_preds[val_idx] = predict(model, train_df.iloc[val_idx])

oof_logloss = log_loss(train_df['label'], oof_preds)
print(f"OOF LogLoss: {oof_logloss:.6f}")
```

## Word-Dropout Augmentation

```python
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512, word_dropout_p=0.1):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.word_dropout_p = word_dropout_p

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.word_dropout_p > 0 and self.training:
            tokens = text.split()
            tokens = [
                tokenizer.unk_token if random.random() < self.word_dropout_p else tok
                for tok in tokens
            ]
            text = ' '.join(tokens)
        enc = self.tokenizer(text, truncation=True, max_length=self.max_len,
            padding='max_length', return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'label': torch.tensor(self.labels[idx], dtype=torch.long)}
```
