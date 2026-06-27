# Dependency and Feature Safety Patterns

## Table of Contents
1. [NLTK Resource Downloads](#nltk-resource-downloads)
2. [General Defensive Patterns](#general-defensive-patterns)
3. [Regex-Safe Feature Engineering](#regex-safe-feature-engineering)
4. [Custom Feature Extraction Checklist](#custom-feature-extraction-checklist)
5. [External Library Outputs](#external-library-outputs)
6. [Intermediate Data Handling](#intermediate-data-handling)

## NLTK Resource Downloads

```python
import nltk
for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass
```

### Fallback Tokenization
```python
def safe_sent_tokenize(text):
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except (LookupError, ImportError):
        import re
        return re.split(r'(?<=[.!?])\s+', text)
```

## General Defensive Patterns

- Check before use: verify resources/data files exist.
- Try-except around external calls.
- Provide fallbacks for missing resources.

## Regex-Safe Feature Engineering

Pandas `.str.count()` compiles argument as regex. Escape metacharacters:
```python
# BROKEN: re.error on unescaped metacharacters
text.str.count("(")

# SAFE options
text.apply(lambda s: s.count("("))           # Python str.count
text.str.count(re.escape("("))                # re.escape
text.str.replace("(", "<LPAREN>", regex=False)  # regex=False
```

## Custom Feature Extraction Checklist

- Return 2D array of shape `(n_samples, n_features)`. Use `np.vstack` or `np.column_stack`.
- Add shape assertion: `assert features.ndim == 2`
- Test in isolation on small sample (`.head(5)`) before full pipeline.
- Validate before passing to `StandardScaler.fit_transform()`.

## External Library Outputs

```python
# GOOD: Explicitly cast to numpy
embeddings = np.array(sbert_model.encode(texts, convert_to_numpy=True)).astype(np.float32)

# BAD: Returns Python list, crashes on .shape
embeddings = sbert_model.encode(texts, convert_to_numpy=False)
```

## Intermediate Data Handling

When loading `.npy` files with non-numeric data, pass `allow_pickle=True`. Prefer CSV/pickle for string data. Wrap artifact loading in try-except.