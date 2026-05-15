---
title: Domain-Specific Pre-training Significantly Improves AA Performance
confidence: HIGH
papers: [frontiers, japanese_bert, sbert_ieee2025]
---

# Domain-Specific Pre-training Significantly Improves AA Performance

Using BERT models pre-trained on domain-relevant corpora (rather than general Wikipedia/BookCorpus) substantially improves authorship attribution accuracy, especially when the target texts share the same domain, language, or text type as the pre-training data.

**Specific evidence from three studies:**

**Frontiers (Japanese literary AA):** Directly compared BERT models pre-trained on different corpora:
- **AozoraBERT** (pre-trained on Japanese literary works from Aozora Bunko): F1=0.969 on Corpus A (literary works overlapping with pre-training data)
- **TohokuBERT** (pre-trained only on Japanese Wikipedia): F1=0.642 on Corpus A — **32.7 points lower**
- **AozoraWikiBERT** (pre-trained on both Wikipedia + Aozora Bunko): F1=0.970 on Corpus A — highest individual score
- **StockMarkBERT** (pre-trained on Japanese news articles): F1=0.600 on Corpus A — lowest individual score
- "The F1 scores of both models [AozoraBERT, AozoraWikiBERT] were more than 30 points higher than that of Model T, which was trained using only Wikipedia."
- However, on Corpus B (not in any pre-training data), DeBERTa achieved the highest F1 (0.823), suggesting that when domain match is absent, architecture matters more.

**Japanese BERT (Learner corpus AA):** Used `cl-tohoku/bert-large-japanese-v2`, a Japanese-specific BERT model pre-trained on Japanese Wikipedia and CC-100, rather than a multilingual BERT. This enabled effective fine-tuning for Japanese text, achieving 84% accuracy for 5 authors and 97% for binary native-speaker classification.

**sbert_ieee2025 (Multi-domain AA):** Used Sentence-BERT (SBERT), which is pre-trained on a diverse corpus of sentence pairs, and further fine-tuned it on a multi-domain corpus (Gutenberg + Twitter + Stack Overflow). The inclusion of contemporary digital texts (Twitter, Stack Overflow) alongside classical literature improved generalizability: "This inclusion enabled the model to capture informal, conversational, and modern digital language patterns, thereby improving its generalizability."

## Papers & Evidence
- `frontiers` (Frontiers in AI: Authorship Attribution Methods): "For Corpus A, Model AW, pre-trained on Wikipedia and Aozora Bunko, achieved the highest score... The F1 scores of both models were more than 30 points higher than that of Model T, which was trained using only Wikipedia." — Direct head-to-head comparison showing 30+ point F1 gain from domain-matched pre-training.
- `japanese_bert` (BERT Fine-tuning for Japanese Authorship Attribution): "We utilized the Japanese pre-trained BERT model available from the Transformer library, named 'cltohoku/bert-large-japanese-v2'" — Chose language-specific BERT over multilingual alternatives.
- `sbert_ieee2025` (Transformer-Based Authorship Attribution): "This inclusion enabled the model to capture informal, conversational, and modern digital language patterns, thereby improving its generalizability." — Shows domain augmentation of training data improves cross-domain AA.

## Actionable Guidance
Select a BERT model pre-trained on data matching your target domain. For literary AA, use a model pre-trained on literary corpora (e.g., AozoraBERT for Japanese literature). For language-specific AA, use a monolingual BERT (e.g., cl-tohoku/bert-large-japanese-v2 for Japanese). For multi-domain AA, augment training data with diverse text types. Expect 20-30+ point F1 improvements from domain-matched pre-training when the target domain is well-covered.

**Condition**: When the target text domain is specific (literary, news, social media) and a domain-matched pre-trained BERT is available.
**Confidence**: HIGH — Frontiers provides direct head-to-head comparisons with 30+ point deltas; Japanese BERT and sbert_ieee2025 provide supporting evidence.
