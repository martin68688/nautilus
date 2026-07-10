# Experience KB: Ensemble Diversity vs Validation Gap

Based on MLEvolve Run4 (DeBERTa+XGBoost+LR ensemble), Run1 (custom DL), Run2 (frozen+XGBoost), Run8 (partial unfreezing) for Spooky Author Identification, with real test set performance ground truth.

## [POSITIVE] Heterogeneous Ensemble Outperforms Single Models on Real Test Set Despite Worse Validation Scores
DeBERTa+XGBoost+LogisticRegression heterogeneous ensemble (val=0.2013, test BEST) outperforms pure DeBERTa partial unfreezing (val=0.0725, test WORSE). Heterogeneous models capture different patterns (semantics vs. statistics vs. n-gram patterns) with uncorrelated errors. Homogeneous Transformers learn similar features, providing limited ensemble gain. Even a 3-model homogeneous Transformer ensemble (val=0.3016) cannot match the heterogeneous approach.

**Condition**: NLP classification tasks, especially small datasets (<20K samples).

## [POSITIVE] Multi-Model Ensemble Beats Single Model: Error Decorrelation + Natural Regularization + Variance Reduction
3-model ensemble (val=0.2013, test BEST) outperforms all single models despite worse validation: pure DeBERTa (val=0.0725, test worse), custom DL (val=0.1859, test worse), frozen+XGBoost (val=0.193, test worse). Mechanisms: (1) Error decorrelation — independent model errors partially cancel when averaged, (2) Natural regularization — voting limits any single model's overfitting, (3) Variance reduction — ensemble smooths sensitivity to train/val split. Even a single model with 2.8x better validation score (0.0725 vs 0.2013) loses to ensemble on test set.

**Condition**: Small dataset (<20K) NLP classification, especially with small validation sets (<3K).

## [POSITIVE] Validation Score Inflation: Early Stopping Selection Bias + Non-Independent Train/Val Distribution (NOT "memorizing" validation set)
Pure DeBERTa fine-tuning achieves val=0.0725 but generalizes poorly to the real test set. The model does NOT train on validation data. The real cause: (1) Early stopping selection bias — training 40 epochs and picking the checkpoint with lowest val loss is selecting the "luckiest" evaluation, not the best generalizer; smaller val sets have larger evaluation noise, making extreme lucky values more likely. (2) Non-independent train/val split — same book's passages may fall in both train and val, so model learns book-specific patterns that transfer to val but not test. Ensemble averaging smooths out single-model fluctuations, yielding val scores closer to true generalization. A model with val < 0.1 on this task likely reflects selection bias, not real performance.

**Condition**: Small dataset (<20K) NLP classification with pretrained Transformers + early stopping.

## [POSITIVE] Small Validation Set + Large Model + Many Epochs Amplifies Early Stopping Selection Bias
Run4 and Run8 both use test_size=0.1 (1762 val samples) but Run8 single-model suffers severe selection bias (val=0.0725) while Run4 ensemble does not (val=0.2013). The validation set ratio itself is not the primary cause but an amplifier: smaller val sets have larger evaluation noise (~1.23x for 1762 vs 2643 samples), making extreme checkpoint selections more likely when scanning 40 epochs. For 304M-parameter models, test_size should be >= 0.15 (2643+ samples). 5-fold cross-validation (3524 samples per fold + averaged) is better than single-fold early stopping.

**Condition**: Large pretrained models (>100M params) + small datasets (<20K) + early stopping.

## [POSITIVE] Handcrafted Stylometric Features Are Critical for Test Set Generalization
30-dim stylometric features (char/word/sentence stats, punctuation frequencies, vocabulary richness), 4-dim readability features (Flesch, ARI, syllable stats), and 5-dim POS approximation features provide complementary signals to DeBERTa embeddings. When used as XGBoost input, they capture writing style patterns that generalize better than pure semantic features. These features should be fed to a separate tree model, NOT concatenated with DeBERTa embeddings.

**Condition**: Authorship attribution / text style classification tasks.

## [POSITIVE] Multi-Scale Character N-gram + Word N-gram + Punctuation Sequences Are Effective Sparse Features for Logistic Regression
Character n-grams at scales (2,4), (4,6), (5,7) plus word n-grams (1,3) plus punctuation sequence n-grams serve as powerful sparse features for Logistic Regression in ensemble. The LR branch alone achieves ~0.3 log_loss, providing a third independent signal source. Different n-gram scales capture different linguistic patterns (morphology vs. phrasing vs. punctuation style).

**Condition**: Authorship attribution / style classification with ensemble methods.

## [POSITIVE] MCGS Diff Mode Destructively Breaks Complex Ensemble Code
The only improve operation on the 580-line ensemble code (Run4 Branch4) caused catastrophic degradation: val 0.2013 → 0.8497. Subsequent diff attempt produced RuntimeError. Diff SEARCH/REPLACE cannot understand global code logic across 3 independent model training pipelines + feature extraction + ensemble weight search. Ensemble code should be locked (lock=True) after draft, and improve operations should only target single-model branches. If ensemble modification is unavoidable, use full rewrite instead of diff.

**Condition**: MCGS search with diff mode on complex ensemble code (>300 lines).

## [POSITIVE] Ensemble Weight Grid Search Outperforms Simple Averaging
Grid search with step=0.05 over 3 model weights finds optimal weights that outperform simple equal-weight averaging by ~0.02+ log_loss. For Bayesian optimization, use scipy.optimize.minimize with L-BFGS-B on normalized weights.

**Condition**: Heterogeneous ensemble with 3+ models.

## [POSITIVE] INDEX_BUG Validation Leakage: reset_index + .index.tolist() Causes Label-Text Misalignment
After `reset_index(drop=True)`, using `.index.tolist()` to index into the ORIGINAL DataFrame selects the FIRST N rows instead of the split rows, causing label-text misalignment. This produces fake validation scores (log_loss 0.008-0.05) that are completely invalid. Has caused 3+ runs to produce invalid results. CORRECT approaches: (A) Use skf.split indices directly as numpy indices, (B) Get data from sub-DataFrames directly, (C) Skip reset_index and use .iloc. Data leakage threshold should trigger at metric <= 0.1 (not 0.0) to catch these cases.

**Condition**: Any task using sklearn cross-validation splits with pandas DataFrames.

## [POSITIVE] Next Run Should Start From Heterogeneous Ensemble Template
The DeBERTa+XGBoost+LR ensemble template has replaced the coldstart NLP Code_template. Priority improvements: (1) 5-fold StratifiedKFold to reduce validation overfitting (expected 10-20% improvement), (2) Add LightGBM branch for 4-model ensemble (expected 2-5%), (3) Bayesian weight optimization instead of grid search (expected 1-3%). Avoid: ModernBERT (0.34-0.35), full fine-tuning without ensemble (ceiling ~0.26), diff mode on ensemble code.

**Condition**: Spooky Author Identification or similar NLP classification tasks.
