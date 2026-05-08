# HybridBERT - Making BERT Pretraining More Efficient Through Hybrid Mixture of Attention Mechanisms

**Source**: https://aclanthology.org/2024.naacl-srw.30/

## [POSITIVE] Hybrid mixture of self-attention and additive attention
Combining self-attention and additive attention mechanisms in a single BERT-like network, with two variants: HBERTv1 (alternating layers) and HBERTv2 (self-attention in early/late layers, additive in middle).

**Delta**: HBERTv1 reaches 31.59% MLM accuracy after 1 day vs. 15% for vanilla BERT (more than double). Outperforms vanilla BERT by 1% (Massive) and 2.5% (Emotion) after 1 day pretraining.
**Condition**: Limited pretraining budget (1-2 days on a single GPU).

**Evidence**: "From the paper: 'HBERTv1 reaches a pretraining accuracy of 31.59% after just one day of training, compared to 15% for vanilla BERT and AddBERT, which is more than double the accuracy of the baselines.'"

## [POSITIVE] Sub-layer normalization
Applying layer normalization four times instead of twice in each encoder layer.

**Delta**: Contributes to overall performance gains; models with sub-layer norm have slightly more parameters but faster inference.
**Condition**: Used in both HBERTv1 and HBERTv2 architectures.

**Evidence**: "From the paper: 'We also use sub-layer normalization... where layer normalization is applied four times instead of twice in each encoder layer.'"

## [POSITIVE] Removing bias from feed-forward network (FFN) layers
Following Geiping and Goldstein (2022), bias terms are removed from FFN layers.

**Delta**: Not quantified separately, but part of the overall architecture that outperforms baselines.
**Condition**: Applied to both HBERT models.

**Evidence**: "From the paper: 'Following the architectural modifications suggested in (Geiping and Goldstein, 2022), we remove bias from the feed-forward network (FFN) layers.'"

## [POSITIVE] Reduced dropout value
Dropout reduced from 0.1 to 0.005 due to low overfitting risk with limited pretraining.

**Delta**: Not quantified separately, but part of the overall design that improves performance under limited budget.
**Condition**: Limited pretraining budget (1-2 days).

**Evidence**: "From the paper: 'Since the model is only pretrained for a limited time, the chances of over-fitting are extremely low. Consequently, we reduce the dropout value to a very small number.'"

## [POSITIVE] RELU activation instead of GELU
Using RELU activation function instead of GELU to accelerate inference.

**Delta**: Inference time reduced; HBERTv2: 701 ms, HBERTv1: 742 ms vs. BERT-base: 713 ms (competitive or faster).
**Condition**: Applied to all HBERT models.

**Evidence**: "From the paper: 'Following the optimization technique to accelerate inference (Sun et al., 2020), we use the RELU activation function instead of GELU.'"

## [POSITIVE] Weight initialization from pretrained BERT model
Initializing self-attention layers of HBERT from a pretrained BERT model (additive attention and normalization layers remain randomly initialized).

**Delta**: After 1 day: HBERTv1 reaches 40.7% MLM accuracy, HBERTv2 reaches 48.95% (vs. 31.59% without initialization). Downstream: up to 93.05% on Emotion (vs. 88.65% without initialization).
**Condition**: When a pretrained BERT model is available; applicable for researchers modifying existing architectures.

**Evidence**: "From the paper: 'We find that continual pretraining boosts both pretraining and downstream performance. After just one day of pretraining, HBERT reaches a pretraining accuracy of 40.7% (v1) and 48.95% (v2), outperforming all other models, including vanilla BERT.'"

## [NEGATIVE] Additive attention alone (AddBERT baseline)
Using additive attention on all layers instead of a hybrid mixture.

**Delta**: After 5 days, performance gap between AddBERT and BERT-base is more than 35% MLM accuracy. Minimal increase after each training day.
**Condition**: When used without self-attention layers; especially detrimental under limited pretraining budget.

**Evidence**: "From the paper: 'AddBERT shows a minimal increase in accuracy after each training day, showing that the use of additive attention alone seems to be problematic for performance. At the end of day 5, the performance gap between AddBERT and BERT-base is more than 35%.'"

## [POSITIVE] Limited pretraining budget (1-2 days on single GPU)
Restricting pretraining to a single GPU for 1-2 days, as opposed to full pretraining (e.g., 16 TPUs for 4 days).

**Delta**: HBERTv1 achieves 31.59% MLM accuracy after 1 day vs. 15% for vanilla BERT. Outperforms vanilla BERT on downstream tasks after 1 day.
**Condition**: When computational resources are limited; enables pretraining without sophisticated hardware.

**Evidence**: "From the paper: 'We show that HBERT attains twice the pretraining accuracy of a vanilla-BERT baseline.'"

## [POSITIVE] HBERTv2 architecture (self-attention in early/late layers, additive in middle)
Using self-attention in layers 0, 1, 10, 11 and additive attention in intermediate layers (ratio 2:1 additive to self-attention).

**Delta**: After 1 day: 85.15% on Massive, 88.65% on Emotion (vs. 84.59% and 86.11% for vanilla BERT). After 2 days: 85.83% and 89.55%.
**Condition**: Limited pretraining budget; learns slower initially but achieves competitive downstream performance.

**Evidence**: "From the paper: 'HBERTv2... outperforms vanilla BERT by 1% (Massive) and 2.5% (Emotion) after 1 day.'"
