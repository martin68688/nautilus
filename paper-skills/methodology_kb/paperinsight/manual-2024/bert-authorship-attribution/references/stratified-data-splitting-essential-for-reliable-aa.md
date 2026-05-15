---
title: Stratified Data Splitting is Essential for Reliable AA Evaluation
confidence: HIGH
papers: [acl2023, bertaa, frontiers, sbert_ieee2025]
---

# Stratified Data Splitting is Essential for Reliable AA Evaluation

Four independent studies independently adopted stratified data splitting to maintain proportional author representation across training, validation, and test sets. This is particularly critical for authorship attribution where class imbalance is common (authors have varying numbers of texts) and where small sample sizes amplify the risk of biased splits.

**Specific splitting strategies used:**
- **ACL2023** (GAN-BERT): 80% train, 10% test, 10% validation, stratified by author IDs. "The stratified split in the train-test-validation ensured a uniform distribution of novels per author, and the test data are distinct from the training data."
- **BertAA** (Fabien et al.): 80% train, 20% test, stratified to maintain equal class proportions. "kept 20% of test data using a stratified approach, meaning that the proportions of each class are kept equal in the training and testing set."
- **Frontiers** (Japanese literary): Stratified 5-fold cross-validation due to small sample size (200 works, 20 per author). "Owing to the small sample size of our data, random division may bias the balance of class labels. Therefore, in this experiment, we used a stratified sampling method to divide the dataset into five folds."
- **sbert_ieee2025** (S-BERT): 60% train, 20% validation, 20% test, stratified. "The complete dataset was stratified and partitioned into training (60%), validation (20%), and testing (20%) subsets, maintaining proportional author representation."

**Why stratification matters for AA:**
Authorship attribution datasets frequently have imbalanced author representation (e.g., Japanese BERT dataset: mean ~16 texts/author, max 68, min 5). Without stratification, a random split could accidentally place all texts from a minority author into the test set, making evaluation unreliable. The ACL2023 study explicitly notes that manual author selection introduces bias, and random sampling of author combinations (50-100 samples per n-author case) was used to mitigate this.

## Papers & Evidence
- `acl2023` (Authorship Attribution with Transformers): "We split train-test-validation (80:10:10) sets, stratified by author ids, for each sample set considered for the experiments...The stratified split in the train-test-validation ensured a uniform distribution of novels per author, and the test data are distinct from the training data." — Explicitly links stratification to uniform author distribution and data distinctness.
- `bertaa` (BertAA: BERT for Authorship Attribution): "kept 20% of test data using a stratified approach, meaning that the proportions of each class are kept equal in the training and testing set." — Uses stratification to maintain class balance.
- `frontiers` (Frontiers in AI: Authorship Attribution Methods): "Owing to the small sample size of our data, random division may bias the balance of class labels. Therefore, in this experiment, we used a stratified sampling method to divide the dataset into five folds." — Explicitly motivates stratification for small-sample AA.
- `sbert_ieee2025` (Transformer-Based Authorship Attribution): "The complete dataset was stratified and partitioned into training (60%), validation (20%), and testing (20%) subsets, maintaining proportional author representation." — Uses stratification for multi-domain corpus.

## Actionable Guidance
Always use stratified splitting (by author ID) when partitioning AA datasets. For standard-sized datasets, use 80:10:10 or 80:20 train-test splits. For small datasets (<500 texts total), use stratified k-fold cross-validation (k=5). This prevents evaluation bias from imbalanced author representation.

**Condition**: All AA experiments with multiple authors where text counts per author vary.
**Confidence**: HIGH — 4 papers independently adopt this practice with explicit motivation.
