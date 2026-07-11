# Novel Draft Actual Strategy Routes

Split: `test`; queries: `2`.

Baseline and replay roles are fixed. Only the Novel retrieval condition changes.

| Novel retrieval | Strategy precision@3 | Distinct families@3 | Detail intrusion@3 | Clean expansion@3 | Excluded violations | Gate pass rate | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| tree_only | 0.3333 | 3.00 | 0.6667 | 1.0000 | 2 | 0.0000 | 0 |
| stage_hybrid | 0.3333 | 3.00 | 0.6667 | 1.0000 | 0 | 0.0000 | 0 |
| layered_strategy | 1.0000 | 3.00 | 0.0000 | 1.0000 | 0 | 1.0000 | 0 |

## three-role-draft::d46c5740ac691eaf

### tree_only
1. `roberta_family` via `sop::sg_0183`: Use DistilRoBERTa with 5-fold cross-validation and cosine warmup scheduler for text classification (level=L2_tactic, clean=True, metric=0.4380950529607263)
2. `bert_family` via `sop::sg_0182`: Use XGBoost with handcrafted stylometric features and Sentence-BERT embeddings for text classification (level=L3_repair, clean=True, metric=0.4380950529607263)
3. `modernbert_finetune` via `sop::sg_0088`: Use ModernBERT-large with proper training regularization (level=L1_strategy, clean=True, metric=0.3808756215966935)
### stage_hybrid
1. `general` via `sop::sg_0196`: Use TF-IDF features with n-grams up to 3 (level=L3_repair, clean=True, metric=0.40750749852913043)
2. `vision_transformer_finetune` via `sop::sg_0041`: Use SigLIP2 vision transformer with two-stage fine-tuning for leaf classification (level=L1_strategy, clean=True, metric=0.10133425749879976)
3. `vision_transformer_family` via `sop::sg_0016`: Ensure SigLIP model receives both pixel_values and input_ids for forward pass (level=L3_repair, clean=True, metric=0.9988079524830886)
### layered_strategy
1. `frozen_transformer_tree` via `sop::sg_0118`: Combine multiple frozen transformer embeddings with XGBoost stacking (level=L1_strategy, clean=True, metric=0.454066)
2. `deberta_finetune` via `sop::sg_0213`: Use DeBERTa-v3-large with standard AutoModelForSequenceClassification for strong performance (level=L1_strategy, clean=True, metric=0.296175)
3. `multi_transformer_ensemble` via `sop::sg_0119`: Use weighted ensemble averaging of transformer predictions (level=L1_strategy, clean=True, metric=0.353015)

## three-role-draft::b5519eb09200d292

### tree_only
1. `roberta_family` via `sop::sg_0183`: Use DistilRoBERTa with 5-fold cross-validation and cosine warmup scheduler for text classification (level=L2_tactic, clean=True, metric=0.4380950529607263)
2. `bert_family` via `sop::sg_0182`: Use XGBoost with handcrafted stylometric features and Sentence-BERT embeddings for text classification (level=L3_repair, clean=True, metric=0.4380950529607263)
3. `modernbert_finetune` via `sop::sg_0088`: Use ModernBERT-large with proper training regularization (level=L1_strategy, clean=True, metric=0.3808756215966935)
### stage_hybrid
1. `general` via `sop::sg_0196`: Use TF-IDF features with n-grams up to 3 (level=L3_repair, clean=True, metric=0.40750749852913043)
2. `vision_transformer_finetune` via `sop::sg_0041`: Use SigLIP2 vision transformer with two-stage fine-tuning for leaf classification (level=L1_strategy, clean=True, metric=0.10133425749879976)
3. `vision_transformer_family` via `sop::sg_0016`: Ensure SigLIP model receives both pixel_values and input_ids for forward pass (level=L3_repair, clean=True, metric=0.9988079524830886)
### layered_strategy
1. `frozen_transformer_tree` via `sop::sg_0118`: Combine multiple frozen transformer embeddings with XGBoost stacking (level=L1_strategy, clean=True, metric=0.454066)
2. `deberta_finetune` via `sop::sg_0213`: Use DeBERTa-v3-large with standard AutoModelForSequenceClassification for strong performance (level=L1_strategy, clean=True, metric=0.296175)
3. `multi_transformer_ensemble` via `sop::sg_0119`: Use weighted ensemble averaging of transformer predictions (level=L1_strategy, clean=True, metric=0.353015)

Claim allowed: `False`

This benchmark exposes actual method routes. Downstream superiority still requires a concurrent online control.
