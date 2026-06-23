# Feature Engineering for Small-Data NLP Classification

## Feature Families

### 1. Lexical / Statistical Features
- Text length (characters, words, sentences)
- Punctuation ratios
- Average word length, sentence length
- Capitalization ratios

### 2. TF-IDF Features
- **Word-level TF-IDF** (1–2 grams, max 10k features)
- **Character-level TF-IDF** (2–4 grams, max 10k features)

### 3. Stylometric Features
- Sentence complexity
- Rare word ratio
- POS tag distribution
- Function word frequencies
- Vocabulary richness (type-token ratio)

## Integration

### Late Fusion (Recommended)

Train transformer and GBT independently, then ensemble via probability averaging or stacking.

### Early Fusion (Alternative)

```python
combined = torch.cat([transformer_embedding, feature_vector], dim=-1)
logits = classifier(self.dropout(combined))
```

**Warning**: Early fusion increases overfitting risk on very small datasets.

## Task-Specific Recommendations

| Task Type | Key Features |
|---|---|
| Authorship attribution | Stylometric + char TF-IDF |
| Sentiment classification | Word TF-IDF + punctuation ratios |
| Topic classification | Word TF-IDF only |

### Dimensionality Rule of Thumb

Keep total handcrafted feature dimensions below `n_samples / 50`.
