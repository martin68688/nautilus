# Frontiers in AI: Authorship Attribution Methods

**Source**: local:frontiers.pdf

## [POSITIVE] Integrated Ensemble of BERT and Feature-Based Models
Combines probability vectors from multiple BERT models and multiple feature-based classifiers (using soft voting) into a single ensemble prediction.

**Delta**: F1 improved from 0.823 to 0.96 on Corpus B; from 0.970 to 1.000 on Corpus A
**Condition**: Small-sample authorship attribution tasks (approx. 500 tokens per text, 200 works per corpus, 10 authors).

**Evidence**: "The integrated ensemble significantly outperformed the best individual model on Corpus B—which was not included in the pre-training data—improving the F1 score from 0.823 to 0.96."

## [POSITIVE] Soft Voting Ensemble (Unweighted)
Averages the probability vectors output by each model (BERT or classifier) and selects the class with the highest average probability.

**Delta**: Outperformed weighted ensemble; achieved max F1 of 1.000 on Corpus A and 0.960 on Corpus B
**Condition**: When combining diverse models (BERTs with different pre-training data + feature-based classifiers).

**Evidence**: "The weighted ensembles did not show any improvement in score compared to the unweighted ensembles in either corpus."

## [POSITIVE] Including Low-Performing Individual Models in Ensemble
Incorporating models with low individual F1 scores (e.g., StockMarkBERT, Ada+Phrase, RF+Phrase) into the ensemble.

**Delta**: Ensemble {A, S} achieved F1=0.990 on Corpus A despite S having lowest individual F1 (0.600); top integrated ensembles on Corpus B included low-scoring phrase-pattern classifiers
**Condition**: When the low-performing model captures different stylistic aspects (e.g., phrase patterns suppress content words) or comes from a different domain (e.g., news vs. literature).

**Evidence**: "Model S, pre-trained on news articles, which differ significantly from the literary style... contributed significantly to achieving the highest scores in both their respective ensembles and the integrated ensemble."

## [POSITIVE] Using Phrase Pattern (Bunsetsu Pattern) Features
A feature that masks content words with their POS tags, capturing authorial habits in particle usage and syntactic structure while being robust to content and genre.

**Delta**: Contributed to top integrated ensembles; phrase pattern + AdaBoost/RF were among the lowest individual scores but appeared in top ensembles
**Condition**: When combined with other features (char-bigram, token-unigram) in an ensemble; particularly effective for Japanese literary texts.

**Evidence**: "The phrase pattern can capture individual habits, specifically how distinctive elements of a writer's style are combined and used. Although a phrase pattern never scores high in author estimation, it is robust to text content and genre."

## [POSITIVE] Using BERT Models Pre-trained on Domain-Relevant Data (Aozora Bunko)
Using BERT models pre-trained on Japanese literary works (Aozora Bunko) rather than general Wikipedia or news data.

**Delta**: AozoraBERT F1=0.969 vs TohokuBERT F1=0.642 on Corpus A (which overlaps with pre-training data); AozoraWikiBERT F1=0.970 vs TohokuBERT F1=0.642
**Condition**: When the target text domain (literary works) matches the pre-training domain; especially impactful when target texts are included in pre-training data.

**Evidence**: "For Corpus A, Model AW, pre-trained on Wikipedia and Aozora Bunko, achieved the highest score... The F1 scores of both models were more than 30 points higher than that of Model T, which was trained using only Wikipedia."

## [POSITIVE] Using DeBERTa Architecture
Using DeBERTa, which introduces Disentangled Attention (separately encoding word content and position) and an Enhanced Mask Decoder.

**Delta**: DeBERTa achieved highest individual F1 (0.823) on Corpus B (not in pre-training data), outperforming AozoraWikiBERT (0.820)
**Condition**: When the target text is from a domain not covered by the pre-training data of other BERT models.

**Evidence**: "In Corpus B, Model De achieved the highest F1 score, followed by AW and A. The fact that Model De achieved the highest score for Corpus B suggests that Corpus B was not included in its pre-training data, and that Model De is more generic."

## [POSITIVE] Ensemble of BERTs Trained on Different Pre-training Corpora
Combining BERT models pre-trained on different datasets (Wikipedia, Aozora Bunko, news articles) rather than using a single BERT pre-trained on a combined corpus.

**Delta**: Ensemble {T, A} F1=0.980 on Corpus A vs single model AW (trained on both) F1=0.970; ensemble {T, A} F1=0.856 vs AW F1=0.820 on Corpus B
**Condition**: When individual BERTs are trained on distinct, non-overlapping corpora, introducing diversity in inductive biases.

**Evidence**: "The ensemble scores of Models T and A were higher than those of Model AW. This again suggests that model performance is influenced by the pre-training data and other model-specific properties."

## [POSITIVE] Stratified k-Fold Cross-Validation for Small Samples
Using stratified sampling to divide the small dataset (200 works) into 5 folds, ensuring balanced class labels in training/validation/test sets.

**Delta**: Not quantified directly, but enabled reliable evaluation with only 200 samples
**Condition**: When sample size is small (200 works, 20 per author) and class balance must be maintained.

**Evidence**: "Owing to the small sample size of our data, random division may bias the balance of class labels. Therefore, in this experiment, we used a stratified sampling method to divide the dataset into five folds."

## [NEUTRAL] Using Only First 510 Tokens per Text
Extracting only the first 510 tokens (morphemes) from each literary work to meet BERT's 512-token input limit, applied uniformly to both BERT and feature-based methods.

**Delta**: No direct comparison with longer texts; enabled fair comparison between methods
**Condition**: When comparing BERT-based and feature-based models on the same text segments; necessary due to BERT's input length constraint.

**Evidence**: "In this study, for fairness, only the first 510 tokens were used for all processes, including feature-based methods."

## [NEUTRAL] Fixed Hyperparameters for BERT Fine-Tuning (Batch Size=16, LR=2e-5, Epochs=40)
Using uniform hyperparameters across all BERT models without extensive tuning: mini-batch size 16, learning rate 2e-5, AdamW optimizer, 40 epochs.

**Delta**: No per-model tuning; consistent results across models per preliminary experiments
**Condition**: When comparing multiple BERT models fairly; may not achieve optimal performance for each individual model.

**Evidence**: "While extensive hyperparameter tuning was beyond the scope of this study, preliminary experiments confirmed that the chosen configuration (batch size = 16, learning rate = 2 × 10−5, epochs = 40) yielded consistent results across all models."
