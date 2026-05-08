# The Impact of Depth on Compositional Generalization in Transformer Language Models

**Source**: https://aclanthology.org/2024.naacl-long.402/

## [POSITIVE] Depth-width tradeoff with constant parameter count
To control for the confound between depth and total parameters, models are made deeper by reducing the feed-forward dimension (d_ff) while keeping the total number of parameters constant across a size class.

**Delta**: Deeper models (up to a point) show better compositional generalization and lower perplexity than shallower models of the same size.
**Condition**: Applied when comparing models within a fixed parameter budget (41M, 134M, 374M).

**Evidence**: "To address this confound, we construct classes of models with an equal total number of parameters but differing depths; we do so by reducing the model's feed-forward dimension to compensate for added depth."

## [NEUTRAL] Varying feed-forward ratio (d_model/d_ff) with depth
Instead of keeping the standard ratio of d_model = d_ff/4, the feed-forward dimension is varied relative to the model dimension as depth changes, while keeping attention dimensions constant.

**Delta**: Performance degrades when d_ff becomes smaller than d_model (ratio > 1), but within a range, models can tolerate different ratios.
**Condition**: Applicable when designing transformer architectures with non-standard depth/width proportions.

**Evidence**: "When d_model/d_ff > 1 (red dashed rule), perplexity slowly increases. As models get larger, the range of d_model/d_ff ratios where performance is close-to-optimal expands leftward."

## [POSITIVE] Pretraining on C4 corpus
All models are pretrained as language models on the Colossal Clean Crawled Corpus (C4) for 1M steps with a context size of 1024 tokens.

**Delta**: Deeper models achieve lower perplexity on C4 validation set (e.g., 41M: 28.8 vs 45.7 for 1-layer).
**Condition**: Applies to all models during the pretraining phase.

**Evidence**: "Deeper models have lower perplexity. At the shallow end of the spectrum, increasing depth results in a dramatic improvement in perplexity."

## [POSITIVE] Fine-tuning on compositional generalization tasks
Pretrained models are fine-tuned for 10,000 steps on the training portions of COGS, COGS-vf, GeoQuery, and English Passivization datasets.

**Delta**: Deeper models achieve higher generalization accuracy (e.g., 41M on COGS: 72.3% for 7-layer vs 12.4% for 1-layer).
**Condition**: Applied after pretraining, for evaluating compositional generalization.

**Evidence**: "Deeper models generalize better than shallower models on compositional tasks across datasets and size classes."

## [POSITIVE] Correcting for pretraining perplexity confound
To isolate the effect of depth from better language modeling performance, intermediate checkpoints of deeper models are selected that match the validation perplexity of shallower reference models, then fine-tuned.

**Delta**: Even after equalizing perplexity, deeper models still generalize better (e.g., up through 6 layers on COGS).
**Condition**: Used to control for the indirect effect of depth via language modeling performance.

**Evidence**: "Even when fine-tuning checkpoints with equal validation perplexity, deeper models still generalize better than shallower models do up through six layers."

## [POSITIVE] Correcting for in-distribution loss confound
Generalization accuracy is compared at points during fine-tuning when models have equal loss on the in-distribution portion of the dataset.

**Delta**: Deeper models achieve higher out-of-distribution accuracy even at equal in-distribution loss.
**Condition**: Used to control for the indirect effect of depth via in-distribution task performance.

**Evidence**: "Deeper models generalize better than shallower ones on COGS, even at points during fine-tuning when models have equal loss on the in-distribution portion of the dataset."

## [NEGATIVE] Separate analysis of lexical vs. structural generalization
Performance on COGS and COGS-vf is broken down into lexical generalization (familiar words in new contexts) and structural generalization (novel syntactic structures).

**Delta**: Depth improves lexical generalization but does not meaningfully improve structural generalization (dashed lines flat).
**Condition**: Applicable when analyzing which aspects of compositional generalization benefit from depth.

**Evidence**: "Increasing depth improves lexical generalization in both COGS and COGS-vf, but does not meaningfully improve structural generalization performance."

## [NEGATIVE] Rank analysis of feed-forward transforms
Singular value decomposition is performed on the affine transforms in the feed-forward block to measure how rank-deficient they become as the feed-forward ratio increases.

**Delta**: As d_model/d_ff increases, transforms become increasingly close to rank-deficient, correlating with performance degradation.
**Condition**: Observed when d_ff becomes small relative to d_model (extreme depth-width tradeoffs).

**Evidence**: "As models get deeper and d_model/d_ff ratio gets larger, the input and output projections in the feed-forward block become increasingly close to rank-deficient transforms."

## [NEUTRAL] Using GPT-style causal decoder-only architecture
Instead of encoder-decoder T5 models, decoder-only causal language models are used, with half as many total layers as their encoder-decoder variants.

**Delta**: Not directly compared, but enables controlled depth experiments.
**Condition**: Architectural choice for all experiments.

**Evidence**: "Unlike T5 and the original transformer, we implement GPT-style causal decoder-only language models; following Wang et al. (2022) we consider decoder-only models with half as many total layers as their encoder-decoder variants."

## [POSITIVE] Keeping attention mechanism identical across depths
The model dimension (d_model) and attention dimension (d_attn) are held constant while only the feed-forward dimension varies with depth.

**Delta**: Rules out confound between depth and attention capacity.
**Condition**: Design choice for all models in the study.

**Evidence**: "By keeping the attention mechanism identical across models of varying depths, we rule out the possibility that depth will be confounded with the capacity of the self-attention mechanism."
