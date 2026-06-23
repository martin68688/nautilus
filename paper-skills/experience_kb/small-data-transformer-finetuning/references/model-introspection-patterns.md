# Model Introspection Patterns

## Dynamic Base-Model Discovery

```python
def get_base_model(model):
    for attr in ('bert', 'roberta', 'deberta', 'electra', 'distilbert', 'model'):
        base = getattr(model, attr, None)
        if base is not None:
            return base
    raise AttributeError(f"Cannot find base model. Inspect: {[a for a in dir(model) if not a.startswith('_')]}")
```

## Layer Selection by Parameter Names

```python
for name, param in model.named_parameters():
    if 'encoder.layer.' in name:
        layer_idx = int(re.search(r'encoder\.layer\.(\d+)\.', name).group(1))
        param.requires_grad = layer_idx >= (max_layer - 3)
```

## Common Architectures Quick Reference

| Architecture | Base attribute | Encoder path |
|---|---|---|
| BERT | `.bert` | `.bert.encoder.layer.{N}` |
| RoBERTa | `.roberta` | `.roberta.encoder.layer.{N}` |
| DeBERTa-v3 | `.deberta` | `.deberta.encoder.layer.{N}` |
| DistilBERT | `.distilbert` | `.distilbert.transformer.layer.{N}` |
| ModernBERT | `.model` | `.model.layers.{N}` |

**Key takeaway**: Always verify with `model.named_parameters()` before writing
layer-specific logic.
