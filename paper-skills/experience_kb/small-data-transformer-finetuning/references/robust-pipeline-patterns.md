# Robust Pipeline Patterns

## Standard Library Alternatives

```python
# Instead of nltk.word_tokenize(text.lower())
words = text.lower().split()
```

## Runtime NLTK Provisioning

```python
import nltk
for pkg in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger_eng']:
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)
```

## Graceful Degradation for Optional Features

```python
def extract_features```json
(texts):
    features = []
    for t in texts:
        try:
            feats = stylometric_features(t)
        except Exception:
            feats = fallback_features(t)
        features.append(feats)
    return features
```

## Pre-Flight Smoke Test

```python
assert len(train_texts) > 0, 