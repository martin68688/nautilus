---
title: BERT Alone (Without Handcrafted Features) is Often Sufficient or Superior for AA
confidence: HIGH
papers: [bertaa, japanese_bert, sbert_ieee2025]
---

# BERT Alone (Without Handcrafted Features) is Often Sufficient or Superior for AA

Three studies independently found that using BERT with only raw text input (no handcrafted stylometric features) achieves strong or superior performance for authorship attribution. Adding handcrafted features can actually degrade performance, particularly when features are concatenated into the BERT architecture rather than combined at the decision level.

**Specific evidence:**

**BertAA** (Fabien et al.): The core BertAA model uses only fine-tuned BERT with a dense layer + softmax, without any stylometric or hybrid features. This BERT-only approach "outperforms the TF-IDF and LR benchmark on all experiments, with an average relative accuracy gain of 14.3%." The stylometric and hybrid features were added only as an optional ensemble extension (not integrated into BERT itself).

**Japanese BERT** (Learner corpus): Directly compared BERT-only (text input) vs. BERT with concatenated stylometric features (content word rates, POS bigrams, particle bigrams, comma placement). Results:
- BERT-only: 84% accuracy for 5 authors
- BERT + stylometric features: 53% accuracy for 25 authors — **drastic degradation**
- The paper states: "Directly fine-tuning BERT with stylometric features might have been one reason for the low accuracy... extremely low accuracy was consistently observed across all patterns."

**sbert_ieee2025** (S-BERT + MLP): Used only Sentence-BERT embeddings fed into an MLP classifier, with no handcrafted features. Achieved 90% accuracy. The paper explicitly notes: "Unlike traditional text classification tasks, we did not remove stopwords or infrequent tokens, as these elements may contain stylistic information important for identifying an author." — The BERT model itself captures stylistic information from raw text.

**Contrast with successful feature integration:** BertAA and Frontiers both successfully combined BERT with handcrafted features, but crucially at the *decision level* (ensembling probability outputs), not by concatenating features into the BERT architecture. The Japanese BERT study's failure suggests that feature concatenation disrupts BERT's learned representations.

## Papers & Evidence
- `bertaa` (BertAA: BERT for Authorship Attribution): "BertAA outperforms the TF-IDF and LR benchmark on all experiments, with an average relative accuracy gain of 14.3%." — BERT-only (with dense+softmax) achieves SOTA without handcrafted features.
- `japanese_bert` (BERT Fine-tuning for Japanese Authorship Attribution): "the addition of stylistic features for a set of 25 authors resulted in reduced accuracy (53%)" — Direct evidence that concatenating stylometric features into BERT degrades performance.
- `sbert_ieee2025` (Transformer-Based Authorship Attribution): "The S-BERT-based classifier achieved a peak accuracy of 90%, significantly outperforming classical baselines." — Achieved with S-BERT embeddings + MLP, no handcrafted features.

## Actionable Guidance
Start with a BERT-only approach (fine-tuned BERT + classification head) before attempting to add handcrafted features. If features are needed, combine them at the decision level (ensemble of probability outputs) rather than concatenating them into the BERT architecture. Feature concatenation into BERT's representation layer is likely to degrade performance.

**Condition**: When sufficient training data is available (at least ~10 texts per author) and texts are of moderate length (up to 512 tokens).
**Confidence**: HIGH — Supported by 3 papers with Japanese BERT providing direct A/B comparison showing degradation from feature concatenation.
