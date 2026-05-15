---
title: Retaining Function Words and Stopwords Preserves Stylistic Signal for AA
confidence: MEDIUM
papers: [arxiv2510, sbert_ieee2025]
---

# Retaining Function Words and Stopwords Preserves Stylistic Signal for AA

Two studies provide converging evidence that function words and stopwords carry important stylistic signals for authorship attribution, contrary to standard NLP preprocessing that removes them. Retaining these elements preserves author-specific syntactic and grammatical patterns.

**Specific evidence:**

**arxiv2510** (GPT-2 Predictive Comparison): Conducted a systematic ablation study comparing intact texts vs. content-word-only, function-word-only, and POS-tag-only corpora for authorship attribution using GPT-2 language models:
- **Intact texts**: Perfect (100%) classification accuracy across 8 classic English authors
- **Content-word-only** (function words replaced with `<FUNC>`): Reliably learned author patterns for 6/8 authors, but significantly less effective than intact texts (t(11.77)=3.21, p=7.68e-3)
- **Function-word-only** (content words replaced with `<CONTENT>`): Reliably learned author patterns for 5/8 authors, also significantly less effective than intact texts (t(8.36)=4.82, p=1.15e-3)
- **POS-tag-only**: Reliably learned patterns for only 3/8 authors; not reliably above chance (t(9)=1.616, p=0.141)
- Key finding: Both content words AND function words carry significant stylistic signal. Neither alone is sufficient — the combination is necessary for best performance.

**sbert_ieee2025** (S-BERT + MLP): Explicitly chose NOT to remove stopwords or infrequent tokens during preprocessing: "Unlike traditional text classification tasks, we did not remove stopwords or infrequent tokens, as these elements may contain stylistic information important for identifying an author." This approach contributed to the 90% accuracy achieved by the S-BERT classifier.

**Synthesis:** The arxiv2510 ablation study provides rigorous statistical evidence that function words carry author-specific patterns (5/8 authors distinguishable from function words alone), while sbert_ieee2025 demonstrates the practical benefit of retaining all tokens including stopwords. Together, they suggest that standard stopword removal is harmful for AA.

## Papers & Evidence
- `arxiv2510` (Deep Learning for Authorship Attribution): "Models trained on function-word-only corpora reliably learned author-specific patterns for 5 of the 8 authors... These models were also significantly less effective at distinguishing authors than models trained on the intact texts (t(8.36) = 4.82, p = 1.15 × 10−3)." — Rigorous ablation showing function words carry significant but incomplete stylistic signal.
- `sbert_ieee2025` (Transformer-Based Authorship Attribution): "Unlike traditional text classification tasks, we did not remove stopwords or infrequent tokens, as these elements may contain stylistic information important for identifying an author." — Practical application of retaining all tokens for AA.

## Actionable Guidance
Do NOT remove stopwords or function words when preprocessing text for authorship attribution. These tokens carry author-specific syntactic and grammatical patterns. For BERT-based models, use the full raw text. For feature-based models, include function word frequencies as features. The combination of content words (vocabulary choice) and function words (syntactic patterns) provides the strongest stylistic signal.

**Condition**: When preprocessing text for any authorship attribution task (BERT-based or feature-based).
**Confidence**: MEDIUM — Only 2 papers directly support this, though the arxiv2510 ablation provides strong statistical evidence. More replication across different languages and domains would strengthen confidence.
