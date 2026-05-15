# Experience KB: Small-Data Transformer Fine-Tuning

Based on MLEvolve Run8 (66 steps, 6 branches) and 10 historical runs for Spooky Author Identification.

## [POSITIVE] Partial Unfreezing Beats Full Fine-Tuning on Small Datasets
DeBERTa-v3-large partial unfreezing (last 8 of 24 layers) achieves Log Loss 0.0725, while full fine-tuning hits a hard ceiling at ~0.26 across 8 variants. Freezing the first 16 layers preserves pretrained linguistic knowledge (lexical, syntactic), allowing only the top 8 layers to adapt to the task. This acts as implicit regularization via pretrained knowledge. Optimal range: unfreeze last 6-8 layers. Too few (fully frozen backbone) yields 0.3853; too many (full fine-tuning) yields 0.26~0.34.

**Condition**: When training samples < 50K and model > 100M parameters.

## [POSITIVE] Simple Linear Head Beats Complex Attention/MLP Heads
Single Linear(1024→3) classification head (0.0725) far outperforms multi-component heads (0.1457~0.37). Complex heads (CLS+Mean+Max pooling → projection → MultiheadAttention → 4-layer MLP) introduce large randomly-initialized parameters that produce noisy gradients in early training, interfering with backbone fine-tuning. They also overfit on small datasets (~19K samples). The DeBERTa CLS token already encodes sufficient sentence-level representation; a simple linear mapping to class space is enough.

**Condition**: When backbone provides high-quality sentence representations (DeBERTa/RoBERTa CLS token).

## [POSITIVE] CosineAnnealingWarmRestarts + Linear Warmup is the Best Scheduler for Fine-Tuning
CosineAnnealingWarmRestarts + 10% Linear Warmup (0.0725) vs no scheduler (0.1457) — 2x gap within the same branch. Warmup prevents large early gradients from destroying pretrained weights. Cosine decay enables fine convergence at low LR. Periodic restarts help escape local optima. Simple LinearWarmup+LinearDecay achieves only 0.2631. No scheduler causes unstable training.

**Condition**: When fine-tuning pretrained Transformers, especially with partial unfreezing.

## [POSITIVE] Label Smoothing 0.1 Prevents Overfitting on Small Datasets
All top-8 solutions use label_smoothing=0.1 in CrossEntropyLoss. This softens hard labels [1,0,0] to [0.9, 0.05, 0.05], preventing overconfident predictions and improving generalization. Recommended: 0.1 for 3-class tasks, 0.05 for more classes, never exceed 0.2.

**Condition**: Small dataset (<50K) classification tasks.

## [POSITIVE] DeBERTa-v3-large Significantly Outperforms ModernBERT and Smaller Models
DeBERTa-v3-large (0.0725) >> DeBERTa-v3-base (0.3727) >> DeBERTa-v3-small (0.3741) >> ModernBERT (0.3368~0.3518). Even a 3-model ensemble of smaller Transformers (0.3016) cannot match a single large model with partial unfreezing. The disentangled attention mechanism in DeBERTa is particularly effective for capturing subtle writing style differences in authorship attribution.

**Condition**: Authorship attribution / text style classification tasks with GPU >= 16GB.

## [POSITIVE] Differentiated Learning Rates Protect Pretrained Layers
Backbone unfrozen layers use lr=2e-5, classification head uses lr=5e-5 (2.5x). The randomly-initialized head needs faster learning, while pretrained layers only need gentle adjustment. This prevents large gradients from the head from destabilizing the backbone.

**Condition**: Partial unfreezing fine-tuning.

## [POSITIVE] Heterogeneous Ensemble Beats Homogeneous Ensemble
DeBERTa + XGBoost + LogisticRegression heterogeneous ensemble (0.2013) outperforms DeBERTa-large + DeBERTa-small + DistilBERT homogeneous ensemble (0.3016). Heterogeneous models capture different patterns (semantics vs. statistics) with uncorrelated errors. Homogeneous Transformers learn similar features, providing limited ensemble gain.

**Condition**: When pursuing maximum performance with ensemble methods.

## [POSITIVE] Worst-Node Mutation Can Yield Biggest Breakthrough
In Run8 Branch1, the worst-scoring node (0.4738) mutated to produce 0.1457 — a 69% improvement and the key breakthrough. Low-scoring nodes often use fundamentally different strategies; their mutations are more likely to produce architectural-level changes rather than hyperparameter tweaks. Do not prune low-score branches too aggressively.

**Condition**: Evolutionary search / iterative optimization.

## [POSITIVE] Stylometric Features Can Boost Performance When Correctly Integrated
Features like char_count, word_count, punctuation_ratio, type_token_ratio, hapax_ratio, flesch_reading_ease, honore_statistic, sichel_measure, positive/negative word ratios provide complementary stylistic signal. When integrated via a small feature_proj (Linear(150→64) + LayerNorm + GELU + Dropout) concatenated with CLS embedding, they should further improve performance. WARNING: Run8 top1 had 340 lines of feature extraction code that was NEVER fed to the model due to a data-flow bug (train_set.index used instead of feature columns). Must verify forward() accepts features parameter and Dataset returns features tensor.

**Condition**: Authorship attribution tasks. Must verify data flow from feature extraction → Dataset → model.forward().

## [POSITIVE] 5-Fold StratifiedKFold Training Reduces Variance
Training 5 independent models with the best strategy (partial unfreezing + CosineWarmRestart) and averaging their softmax probabilities reduces variance without architecture changes. Each fold uses early stopping with patience=5.

**Condition**: When single-model performance is already strong and variance is the bottleneck.

## [POSITIVE] Probability Clipping + Row Normalization Prevents Log Loss Explosion
Clip predictions to [1e-15, 1-1e-15] then normalize each row to sum to 1.0 before scoring. This prevents extreme probability values from causing infinite Log Loss.

**Condition**: Any classification task evaluated with Log Loss.
