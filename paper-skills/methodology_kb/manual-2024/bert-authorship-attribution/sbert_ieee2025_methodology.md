# Transformer-Based Authorship Attribution: Fine-Tuning BERT and S-BERT

**Source**: IEEE ICACT 2025

## [POSITIVE] Transformer-based embeddings (BERT/S-BERT)
Using pre-trained BERT and Sentence-BERT models to generate dense, contextual sentence embeddings instead of handcrafted features.

**Delta**: 90% accuracy, 0.84 F1; outperforms traditional ML (max 64%)
**Condition**: Applied to authorship attribution on a diverse corpus of 15,000 samples from Gutenberg, Twitter, and Stack Overflow.

**Evidence**: "The S-BERT-based classifier achieved a peak accuracy of 90%, significantly outperforming classical baselines."

## [POSITIVE] Fine-tuning BERT/S-BERT with MLP classifier
Fine-tuning pre-trained transformer models jointly with a lightweight Multi-Layer Perceptron (MLP) classifier for author prediction.

**Delta**: 90% accuracy, 0.84 F1
**Condition**: Training over 60 epochs with Adam optimizer, learning rate 2e-5, batch size 1024, dropout 0.1.

**Evidence**: "The core of our classification pipeline integrated transformer-based sentence embeddings with a lightweight Multi-Layer Perceptron (MLP) classifier."

## [POSITIVE] K-means clustering on sentence embeddings
Applying K-means clustering (K=100) on S-BERT embeddings to group texts by authorial style.

**Delta**: Silhouette score of 0.75
**Condition**: Used for interpretability and validation of embedding quality; optimal K determined via silhouette analysis.

**Evidence**: "The clustering evaluation achieved a silhouette score of 0.75, indicating that the embedded texts formed well-separated, dense clusters."

## [POSITIVE] t-SNE visualization of embeddings
Using t-Distributed Stochastic Neighbor Embedding to reduce high-dimensional embeddings to 2D for visual inspection of author clusters.

**Delta**: Qualitative: clear separable clusters
**Condition**: Applied after embedding extraction; parameters tuned including perplexity and learning rate.

**Evidence**: "t-SNE projections revealed that author-specific clusters in the embedding space were visually distinguishable, supporting the model’s learned representations."

## [POSITIVE] No stopword removal
Retaining stopwords and infrequent tokens during preprocessing, as they may carry stylistic information.

**Delta**: Outperforms traditional approaches that remove stopwords
**Condition**: Applied during text preprocessing for authorship attribution.

**Evidence**: "Unlike traditional text classification tasks, we did not remove stopwords or infrequent tokens, as these elements may contain stylistic information important for identifying an author."

## [POSITIVE] Lemmatization instead of stemming
Reducing words to their base dictionary forms using lemmatization to preserve linguistic patterns.

**Delta**: Contributes to overall 90% accuracy
**Condition**: Applied during text normalization step.

**Evidence**: "Lemmatization was applied to reduce words to their base forms. This helps the model recognize recurring linguistic patterns."

## [POSITIVE] Dataset augmentation with contemporary sources
Supplementing Gutenberg literary texts with modern texts from Twitter and Stack Overflow to improve generalizability.

**Delta**: Improves generalizability across classical and contemporary domains
**Condition**: Applied to dataset compilation; corpus includes both classical and modern digital texts.

**Evidence**: "This inclusion enabled the model to capture informal, conversational, and modern digital language patterns, thereby improving its generalizability."

## [POSITIVE] Stratified train/validation/test split
Partitioning the dataset via stratified sampling to maintain proportional author representation across splits.

**Delta**: Ensures unbiased evaluation
**Condition**: Applied during data preparation for all experiments.

**Evidence**: "The complete dataset was stratified and partitioned into training (60%), validation (20%), and testing (20%) subsets, maintaining proportional author representation."

## [POSITIVE] Early stopping based on validation accuracy
Halting training when validation accuracy stops improving to prevent overfitting.

**Delta**: Minimal overfitting observed
**Condition**: Applied during training of the MLP classifier.

**Evidence**: "We used cross-entropy loss for optimization and employed early stopping based on validation accuracy."

## [NEGATIVE] Word2Vec baseline (averaged embeddings)
Using Word2Vec word vectors averaged to form sentence embeddings as a baseline method.

**Delta**: Underperforms transformer-based methods
**Condition**: Used as a baseline comparison; replaced by S-BERT in final model.

**Evidence**: "While effective in capturing basic semantics, Word2Vec lacks contextual sensitivity and fails to model sentence-level meaning adequately."

## [NEGATIVE] Traditional ML baselines (SVM, Decision Trees, Random Forest, Logistic Regression)
Using handcrafted lexical, syntactic, and statistical features with classical classifiers.

**Delta**: Max 64% accuracy vs. 90% for BERT
**Condition**: Used as initial baseline; limited by shallow feature engineering.

**Evidence**: "The best-performing traditional model achieved a maximum accuracy of only 64%."
