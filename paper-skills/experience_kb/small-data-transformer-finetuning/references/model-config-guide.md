# Model Configuration Parameter Guide

## General Principles

- **Never assume parameter names transfer across model families.**
- Use `AutoModelForSequenceClassification` with `trust_remote_code=False`.
- After instantiation, inspect `model.config` to see available attributes.

## Architecture-Specific Parameters

### BERT / RoBERTa
```python
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)
# Config: model.config.hidden_dropout_prob, model.config.attention_probs_dropout_prob
```

### DeBERTa-v3
```python
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=num_classes)
# Config: model.config.hidden_dropout_prob, model.config.attention_dropout
```

### ModernBERT
```python
model = AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=num_classes)
# NOTE: ModernBERT does NOT accept hidden_dropout_prob as a kwarg.
```

## Safe Initialization Pattern

```python
config = AutoConfig.from_pretrained(model_name, num_labels=num_classes)
if hasattr(config, 'hidden_dropout_prob'):
    config.hidden_dropout_prob = 0.2
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
```
