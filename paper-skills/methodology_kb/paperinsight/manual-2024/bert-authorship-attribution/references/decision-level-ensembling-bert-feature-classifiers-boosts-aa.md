---
title: Decision-Level Ensembling of BERT and Feature-Based Classifiers Boosts AA Performance
confidence: MEDIUM
papers: [bertaa, frontiers]
---

# Decision-Level Ensembling of BERT and Feature-Based Classifiers Boosts AA Performance

Two studies successfully improved authorship attribution accuracy by combining BERT models with feature-based classifiers at the *decision level* (ensembling probability outputs), rather than integrating features into the BERT architecture. This approach is particularly effective when the feature-based models capture complementary stylistic signals that BERT may miss.

**Specific architectures and results:**

**BertAA** (Fabien et al.): Added stylometric features (text length, word count, punctuation frequencies) and hybrid features (character-level bigram and trigram frequencies) via logistic regression classifiers. The probability outputs from BERT and the LR classifiers were concatenated for final classification. Result: "+2.7% relative macro-averaged F1-score on average." The improvement was most notable when BERT alone had low per-author accuracy on some authors.

**Frontiers** (Japanese literary AA): Used soft voting (unweighted averaging of probability vectors) to combine:
- Multiple BERT models pre-trained on different corpora (AozoraBERT, TohokuBERT, AozoraWikiBERT, StockMarkBERT, DeBERTa)
- Feature-based classifiers: AdaBoost + character bigrams, Random Forest + character bigrams, AdaBoost + phrase patterns, RF + phrase patterns
- Results: Integrated ensemble improved F1 from 0.823 (best individual BERT) to 0.96 on Corpus B, and from 0.970 to 1.000 on Corpus A.
- Notably, low-performing individual models (e.g., StockMarkBERT at F1=0.600, phrase pattern classifiers) contributed to top ensembles because they captured different stylistic aspects.

**Key design choice — unweighted soft voting outperformed weighted:**
Frontiers explicitly tested weighted vs. unweighted ensembles: "The weighted ensembles did not show any improvement in score compared to the unweighted ensembles in either corpus." This suggests that simple averaging is robust and avoids overfitting to individual model strengths.

**Contrast with failed feature integration:**
The Japanese BERT study attempted to concatenate stylometric features into BERT's architecture and observed drastic degradation (84% → 53%). The success of BertAA and Frontiers suggests that keeping BERT and feature-based models separate, then combining at the decision level, is the correct approach.

## Papers & Evidence
- `bertaa` (BertAA: BERT for Authorship Attribution): "Improving the macro-averaged F1-Score by 2.7% (relative) on average." — Decision-level combination of BERT probabilities with LR probabilities from stylometric/hybrid features.
- `frontiers` (Frontiers in AI: Authorship Attribution Methods): "The integrated ensemble significantly outperformed the best individual model on Corpus B... improving the F1 score from 0.823 to 0.96." — Soft voting ensemble of BERTs + feature classifiers, with unweighted averaging outperforming weighted.

## Actionable Guidance
To combine BERT with handcrafted features: (1) Train BERT and feature-based classifiers (LR, AdaBoost, RF) independently, (2) Combine their probability outputs via unweighted soft voting (averaging), (3) Include diverse models even if they have low individual scores — they may capture complementary signals. Do NOT concatenate features into BERT's architecture.

**Condition**: When you have both BERT-based and feature-based models available, and the feature-based models capture complementary signals (e.g., phrase patterns, character n-grams) that BERT may underweight.
**Confidence**: MEDIUM — Only 2 papers support this (BertAA, Frontiers), and the Japanese BERT study provides a negative control showing the wrong approach fails. More replication needed.
