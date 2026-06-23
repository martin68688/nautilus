# Feature Engineering Patterns

Concrete patterns for building safe, self-contained feature extraction
functions that avoid cross-function column dependency errors.

## Self-Contained Function Pattern

```python
def extract_sentence_complexity_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    features["word_count"] = df["text"].str.split().str.len()
    features["avg_word_per_sentence"] = features["word_count"] / (
        df["text"].str.count(r"[.!?]") + 1
    )
    return features
```

## Explicit DataFrame Passing Pattern

```python
def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    stylometric = extract_stylometric_features(df)
    complexity = extract_sentence_complexity_features(df)
    all_features = pd.concat([stylometric, complexity], axis=1)
    for col in expected_cols:
        assert col in all_features.columns, f"Missing column: {col}"
    return all_features
```

## Incremental Validation Checklist

```python
print(f"Columns after step {step_name}: {df.columns.tolist()}")
assert "word_count" in df.columns, "word_count must be created before this step"
```
