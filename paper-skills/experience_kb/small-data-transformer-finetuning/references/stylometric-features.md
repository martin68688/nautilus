# Stylometric Features for Authorship Attribution

Detailed feature engineering approach proven effective on authorship identification tasks.

## Feature Categories

### Character-Level Statistics
- Average word length
- Character count per sentence
- Ratio of uppercase to lowercase characters
- Punctuation frequency (commas, semicolons, dashes, etc.)
- Distribution of sentence lengths (mean, std, min, max)

### Readability Metrics
- Approximate Flesch reading ease score
- Average syllables per word (approximation)
- Percentage of complex words (>2 syllables)

### POS Distributions
- Approximate part-of-speech tag ratios
- Noun/verb/adjective/adverb ratios
- Pronoun usage patterns

### Word-Category Ratios
- **Function words**: articles, prepositions, pronouns, auxiliary verbs
- **Archaic words**: thee, thou, hath, dost, etc.
- **Emotional/affective words**: positive/negative sentiment word ratios
- **Content vs. function word ratio**

## Usage Notes

- Combine all features into a single numeric vector per document.
- Feed the vector to XGBoost (or similar gradient-boosted tree model).
- These features capture stylistic patterns that transformers may underweight.
