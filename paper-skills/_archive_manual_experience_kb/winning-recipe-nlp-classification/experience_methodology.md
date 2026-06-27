# Experience KB: Winning Recipe for Small-Data NLP Classification

Based on 12 MLEvolve runs for Spooky Author Identification, with Run 20260516_091845 top4 (val=0.2517) as current best.

## [POSITIVE] Partial Unfreezing + Differentiated Learning Rates is the Optimal Fine-Tuning Strategy for Small Datasets
DeBERTa-v3-large partial unfreezing (last 8/24 layers) with backbone lr=2e-5 and head lr=5e-5 achieves 0.0725 val loss (single model), while full fine-tuning hits a hard ceiling at ~0.26. Freezing the first 16 layers preserves pretrained linguistic knowledge; the 2.5x higher head LR allows the randomly-initialized classifier to converge faster without destabilizing the backbone. Without differentiated LR, backbone weights are corrupted by large gradients from the untrained head in early steps.

**Condition**: Training samples < 50K, model > 100M parameters, pretrained Transformer.

## [POSITIVE] Multi-Sample Dropout (K=4) Provides Free Ensemble Regularization for Classification
Applying K=4 independent dropout masks to the [CLS] embedding and averaging the K logits is equivalent to an implicit ensemble within a single model. In Run 091845, MSD(K=4) improved val from 0.2653 to 0.2517. No additional parameters, no additional inference cost beyond K forward passes through a single linear layer (negligible). The averaged logits are smoother and less prone to overconfident errors. Optimal K: 4 for 3-class tasks, 2-8 range worth scanning.

**Condition**: Classification tasks with dropout-based models. Not applicable to generation tasks.

## [POSITIVE] Multi-Scale Feature Engineering is the Foundation of Heterogeneous Ensemble
Three categories of features, each feeding a different model:
- **Dense features** (→ XGBoost): 26-dim stylometric (after VarianceThreshold 0.001) + 4-dim readability + 5-dim POS approximation + 1024-dim DeBERTa [CLS] embedding = ~1059 dimensions
- **Sparse features** (→ Logistic Regression): char-ngrams(2-4, 4-6, 5-7) + word-ngrams(1-3) + punctuation-ngrams(2-4) = 13500 dimensions → chi2 k=10000
- **Raw text** (→ DeBERTa): tokenized sequences, max_length=512
The key insight is feature-model routing: dense features → tree model, sparse features → linear model, raw text → neural model. This maximizes each model's strength and minimizes error correlation.

**Condition**: NLP classification with ensemble methods. Authorship attribution / style classification tasks.

## [POSITIVE] Chi-Squared Feature Selection on Sparse N-gram Features Reduces Noise and Overfitting
N-gram features are high-dimensional (13,500) and noisy. MaxAbsScaler (preserving non-negativity for chi2) followed by SelectKBest(chi2, k=10000) removes 26% of the least discriminative features. This is critical for Logistic Regression: without chi2, LR overfits to rare n-grams that appear in only 1-2 training samples. With chi2, LR achieves ~0.3 val logloss as a standalone model. The chi2 selector must be fit on training data only.

**Condition**: Sparse n-gram features with Logistic Regression or linear models.

## [POSITIVE] WeightedRandomSampler Handles Class Imbalance Better Than Shuffle
For 3-class authorship with unequal class frequencies, WeightedRandomSampler(weights=1/class_counts, replacement=True) ensures minority classes are sampled proportionally in every epoch. Standard shuffle=True can produce batches with near-zero minority class representation, causing training instability. The sampler weights are computed as 1.0/bincount(y_train), applied per-sample.

**Condition**: Classification with class imbalance ratio > 1.5:1.

## [POSITIVE] Label Smoothing + Gradient Clipping + AMP Form the Training Stability Trifecta
Three techniques that work synergistically: (1) label_smoothing=0.1 softens hard targets, preventing the model from becoming overconfident on training data; (2) clip_grad_norm_(max_norm=1.0) prevents occasional large gradients (especially common with MSD) from corrupting weights; (3) AMP (autocast + GradScaler) provides implicit regularization via mixed-precision computation and 2x training speed on A100. All top solutions use all three simultaneously.

**Condition**: Transformer fine-tuning on small datasets. Always use together.

## [POSITIVE] Train-Only Fit Prevents Feature-Level Data Leakage
All feature transformations (StandardScaler, VarianceThreshold, MaxAbsScaler, TfidfVectorizer, CountVectorizer, SelectKBest) must be fit on training data ONLY. Validation and test sets are only transformed. A common mistake (found in Run4 original code) is fitting the punctuation vectorizer on train+val+test concatenated data, which leaks test-set n-gram vocabulary into training. The fix: fit on train, transform on val/test separately. Check: any vectorizer/scaler/selector that sees test data during fit is a leakage.

**Condition**: Any ML pipeline with feature engineering. Mandatory.

## [POSITIVE] Scheduler with Warmup is Essential, Exact Type is Secondary
Both CosineAnnealingWarmRestarts (Run8 top1, 0.0725) and LinearWarmupDecay (Run 091845 top4, 0.2517) work well. The critical component is the 10% warmup phase: without it, early large gradients destroy pretrained weights, causing 2x worse performance (0.0725 → 0.1457). The decay schedule matters less — cosine restarts help escape local optima, linear decay is simpler and sufficient. Never train without a scheduler.

**Condition**: Fine-tuning pretrained Transformers. Warmup ratio 0.05-0.15.

## [POSITIVE] DeBERTa [CLS] Embedding + Handcrafted Features in XGBoost Outperforms Either Alone
XGBoost with concatenated [DeBERTa CLS (1024-dim) + stylo (26-dim) + readability (4-dim) + POS (5-dim)] achieves better performance than XGBoost with only CLS embeddings or only handcrafted features. The CLS embedding provides high-level semantic understanding; handcrafted features provide interpretable, low-variance stylistic signals. Tree models naturally handle the mixed-scale inputs via feature-wise splits.

**Condition**: Ensemble with tree model branch. Authorship attribution.

## [POSITIVE] 5-Fold Cross-Validation Probability Averaging is the Next Low-Risk Optimization
All current runs use single 90/10 split. 5-fold StratifiedKFold training (each fold 80/20) with softmax probability averaging across 5 models reduces variance without architecture changes. Expected improvement: 5-10% on test set. Each fold uses the same training strategy (partial unfreezing + MSD + WRS + LS 0.1). No hyperparameter changes needed.

**Condition**: When single-model variance is the bottleneck. Compatible with all existing techniques.

## [POSITIVE] Punctuation Sequence Features Provide Unique Signal for Authorship Attribution
Extracting punctuation-only sequences from text (stripping all alphanumeric characters) and treating them as character n-grams captures punctuation habits — the most difficult aspect of writing style to consciously alter. CountVectorizer(analyzer=char, ngram_range=(2,4), max_features=500, min_df=2) on punctuation sequences adds ~500 sparse features. Must be fit on training data only.

**Condition**: Authorship attribution / style verification tasks.

## [POSITIVE] Ensemble Weight Grid Search (Step 0.05) Outperforms Simple Averaging
For 3-model ensemble, grid search over w1 ∈ [0.1, 0.9), w2 ∈ [0.1, 0.9), w3 = 1-w1-w2 with step=0.05 finds weights that outperform equal averaging by ~0.02+ logloss. DeBERTa typically receives the highest weight (~0.55-0.70), XGBoost second (~0.20-0.35), LR third (~0.05-0.15). For future: Bayesian optimization (scipy.optimize with Dirichlet prior) can search continuous space more efficiently.

**Condition**: Heterogeneous ensemble with 3+ models.

## [POSITIVE] K-Fold Cross-Validation Must Fit Transformers Per-Fold to Prevent Leakage
When using StratifiedKFold, ALL feature transformers (StandardScaler, TfidfVectorizer, CountVectorizer, MaxAbsScaler, SelectKBest, VarianceThreshold, LabelEncoder) must be fit on each fold's training portion ONLY. Fitting on the full dataset before splitting causes information from the validation fold to leak into the transformer (e.g., vocabulary, mean/variance, chi2 scores). Only the DeBERTa tokenizer (pretrained, fixed vocabulary) is safe to share across folds. Test set must never participate in any fit/transform step — only final prediction.

**Condition**: Any K-fold cross-validation pipeline with feature engineering. Mandatory.
