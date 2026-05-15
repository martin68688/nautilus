# BertAA: BERT for Authorship Attribution

**Source**: local:bertaa.pdf

## [POSITIVE] BERT fine-tuning with dense layer and softmax
Fine-tuning a pre-trained BERT model with an additional dense layer and softmax activation for authorship classification, without text preprocessing or feature engineering.

**Delta**: up to 5.3% relative accuracy improvement over SOTA
**Condition**: When sufficient training data per author is available, texts are rather short, and there is no large class imbalance.

**Evidence**: "This approach reaches competitive performances on Enron Email, Blog Authorship, and IMDb (and IMDb62) datasets, up to 5.3% (relative) above current state-of-the-art approaches."

## [POSITIVE] Ensemble with stylometric and hybrid features
Adding stylometric features (e.g., text length, word count, punctuation frequencies) and hybrid features (character-level bigram and trigram frequencies) via logistic regression, then concatenating probabilities with BERT output for final classification.

**Delta**: +2.7% relative macro-averaged F1-score on average
**Condition**: When macro-averaged F1-score is prioritized over accuracy; especially useful when BERT alone has low per-author accuracy on some authors.

**Evidence**: "Improving the macro-averaged F1-Score by 2.7% (relative) on average."

## [POSITIVE] Training for multiple epochs (up to 10)
Training the BERT model for more than the typical 2-4 epochs, up to 10 epochs, to improve accuracy on datasets with many authors and less data per author.

**Delta**: Accuracy improved from 88.7% (1 epoch) to 93.0% (10 epochs) on IMDb62
**Condition**: When the number of authors is large (e.g., 62) and training data per author is limited (e.g., 1,000 texts per author).

**Evidence**: "We then increased the number of training epochs and reached 92.3% at 5 epochs, and up to 93.0% at 10 epochs."

## [POSITIVE] Using BERT without additional features (BertAA only)
Using only the fine-tuned BERT model with a dense layer and softmax, without stylometric or hybrid features.

**Delta**: Outperforms TF-IDF baseline by 14.3% relative accuracy on average; SOTA on Blog Authorship (65.4% for 10 authors, 59.7% for 50 authors)
**Condition**: When texts are short, number of authors is moderate, and training data per author is balanced and sufficient.

**Evidence**: "BertAA outperforms the TF-IDF and LR benchmark on all experiments, with an average relative accuracy gain of 14.3%."

## [NEUTRAL] Removing short emails and sender-name utterances (Enron preprocessing)
Dropping emails with fewer than 10 tokens and removing observations containing the sender's name in signatures or forwarded messages.

**Delta**: Not quantified, but enabled fair comparison
**Condition**: When preprocessing the Enron Email corpus to avoid data leakage and ensure consistency with prior work.

**Evidence**: "We also removed all messages of less than 10 tokens to apply the same processing as Ruder et al. (2016). ... we dropped these observations."

## [NEUTRAL] Stratified train-test split
Keeping 20% of test data using a stratified approach to maintain equal class proportions in training and testing sets.

**Delta**: Not quantified, but standard practice
**Condition**: For all experiments to ensure representative evaluation.

**Evidence**: "kept 20% of test data using a stratified approach, meaning that the proportions of each class are kept equal in the training and testing set."

## [NEUTRAL] Using pre-trained BERT (bert-base-cased) without further domain pre-training
Using the standard bert-base-cased model without additional pre-training on the target domain.

**Delta**: Not quantified, but noted as a limitation
**Condition**: When domain adaptation might further improve performance, but was not explored in this work.

**Evidence**: "So far, we have not performed further pre-training of BERT on the target domain, which could also help BertAA."

## [POSITIVE] Training on top N authors with most texts
Selecting the top N authors with the largest number of texts for each dataset to create balanced or near-balanced subsets.

**Delta**: Enables benchmarking; not quantified as a delta
**Condition**: When creating datasets for authorship attribution with controlled number of authors and sufficient data per author.

**Evidence**: "We picked the top N authors with the largest amount of texts, for each of the datasets."
