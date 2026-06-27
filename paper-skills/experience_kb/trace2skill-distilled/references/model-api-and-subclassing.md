# Model API and Subclassing Patterns

## Table of Contents
1. [Parameter Compatibility Reference](#parameter-compatibility-reference)
2. [Safe Loading Patterns](#safe-loading-patterns)
3. [Model Layer Access](#model-layer-access)
4. [Subclassing Guidance](#subclassing-guidance)
5. [Pre-Training Sanity Check](#pre-training-sanity-check)

## Parameter Compatibility Reference

| Architecture | `hidden_dropout_prob` | `attention_probs_dropout_prob` | Notes |
|---|---|---|---|
| BERT | ✅ | ✅ | Standard config keys |
| RoBERTa | ✅ | ✅ | Standard config keys |
| DeBERTa-v3 | ✅ | ✅ | Standard config keys |
| ModernBERT | ❌ | ❌ | Uses `dropout` in config only |

### Parameter Placement

| Parameter | Goes Into | Does NOT Go Into |
|---|---|---|
| `label_smoothing_factor` | `TrainingArguments` | `from_pretrained()` |
| `weight_decay` | optimizer | `from_pretrained()` |
| `learning_rate` | optimizer | `from_pretrained()` |
| `num_labels` | `from_pretrained()` | `TrainingArguments` |
| `drop_path_rate` | config object | `from_pretrained()` kwargs |

## Safe Loading Patterns

### Config-first loading (recommended)
```python
config = AutoConfig.from_pretrained(model_name, num_labels=n_classes)
if hasattr(config, 'hidden_dropout_prob'):
    config.hidden_dropout_prob = 0.1
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
```

### Defensive try/except
```python
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_classes, **config_kwargs)
except TypeError:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_classes)
```

## Model Layer Access

Never hardcode attribute paths. Inspect first:
```python
for name, _ in model.named_children():
    print(name)
```

Use dynamic matching:
```python
for name, param in model.named_parameters():
    if 'encoder.layer' in name:
        layer_num = int(name.split('layer.')[-1].split('.')[0])
        if layer_num >= num_layers - freeze_top_n:
            param.requires_grad = False
```

## Subclassing Guidance

Different architectures accept different `forward` arguments:

| Argument | BERT | DeBERTa-v2/v3 | RoBERTa |
|----------|------|---------------|---------|
| `head_mask` | Yes | **No** | Yes |
| `token_type_ids` | Yes | Yes | No* |

Minimal custom head pattern:
```python
class CustomModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled)
```

## Pre-Training Sanity Check

```python
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
model.eval()
with torch.no_grad():
    dummy = tokenizer("test sentence", return_tensors="pt")
    out = model(**dummy)
    print(f"Logits shape: {out.logits.shape}")  # expect [1, num_classes]
```