# Punctuation Sequence Features for Authorship Attribution

## Finding
Punctuation-only sequences as character n-grams provide unique stylistic signal difficult to consciously alter.

## Evidence
- Run4 LR branch includes punctuation n-gram features (max_features=500)
- Punctuation habits are the most consistent author fingerprints

## Mechanism
- Strip all alphanumeric characters, keep only punctuation
- CountVectorizer(analyzer=char, ngram_range=(2,4)) captures punctuation rhythms
- Reflects unconscious habits: comma splicing, semicolon usage, dash style

## Important
Must fit on training data only (Run4 original had leakage here).

## Condition
Authorship attribution / style verification. Not useful for semantic classification.
