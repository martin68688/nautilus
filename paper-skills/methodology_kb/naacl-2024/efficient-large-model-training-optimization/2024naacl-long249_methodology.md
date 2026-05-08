# Memory Augmented Language Models through Mixture of Word Experts

**Source**: https://aclanthology.org/2024.naacl-long.249/

## [POSITIVE] Mixture of Word Experts (MoWE) Architecture
Replaces FFN layers in a subset of Transformer blocks with a sparse layer of word-specific experts, where each token is routed to a single expert based on its token id in a large auxiliary routing vocabulary.

**Delta**: MoWE-Base outperforms T5.1.1-Base by 15.2 EM on TriviaQA (62.8% improvement); MoWE-Large outperforms T5.1.1-Large by 16.6 EM on TriviaQA.
**Condition**: Applied to Transformer-based language models, especially for knowledge-intensive tasks.

**Evidence**: "MoWE-Base and MoWE-Large outperform T5.1.1-Base and T5.1.1-Large, respectively, on all five tasks. There is a significant gain in performance for knowledge intensive tasks – in particular for TriviaQA, WebQuestions and FEVER."

## [POSITIVE] Knowledge-Rich Routing Vocabulary
Routing vocabulary derived from Wikidata entity and relation names, lowercased and split, then filtered by frequency in C4 dataset to select top 1M tokens. This increases the likelihood that vocabulary entries are informative single-word names.

**Delta**: Improvement of ~2 points in F1 on TriviaQA when using routing vocabularies with size above 262K; progressive improvement as vocabulary size increases from 32K to 1M.
**Condition**: Used for routing in MoWE layers; effective when vocabulary size is large (262K to 1M entries).

**Evidence**: "Figure 4 shows that results progressively improve as we increase the routing vocabulary. These improvements are more pronounced when training for longer; see Figure 5."

## [POSITIVE] Fixed Routing via Hash Function
Routing decisions are based on a static hash function mapping routing ids to expert ids, rather than a learned routing function. This avoids training instability and load balancing issues common in learned MoE routing.

**Delta**: No training instability observed; enables scaling to up to 1 million experts.
**Condition**: Applicable to MoE-style models where stable training and large numbers of experts are desired.

**Evidence**: "We did not observe any training instability (e.g. gradient blowup) that are often reported in the pretraining of regular MoE models; we suspect is a helpful artifact of our fixed routing scheme."

## [POSITIVE] Hierarchical Routing with Frequency Bucketing and Expert Blocks
Tokens are first routed to frequency buckets (based on word frequency), then to expert blocks, then to individual experts. This handles the Zipfian distribution of word frequencies and reduces all-to-all communication overhead.

**Delta**: Enabled pretraining of MoWE-Base models with up to 1 million experts using 16 v3 TPUs.
**Condition**: Used when scaling to tens or hundreds of thousands of experts with highly unbalanced token assignments.

**Evidence**: "Our proposed strategy allowed us to pretrain MoWE-Base models with up to 1 million (small) experts using 16 v3 TPUs."

## [POSITIVE] Parameter Sharing Across MoWE Layers
Expert parameters are shared across all MoWE layers (e.g., 4 layers in the main experiments). This keeps the total number of sparse parameters manageable and improves performance.

**Delta**: Empirical results indicated that sharing parameters across the MoWE layers leads to better performance.
**Condition**: When using multiple MoWE layers in the same model.

**Evidence**: "Parameters are shared across all MoWE layers with the following goal: (1) it makes the MoWE layer even more similar to a memory that is accessed at different points of the network; (2) we can keep the overall number of sparse parameters relatively low without the need to decrease the total and the size of experts. Additionally, empirical results indicated that sharing parameters across the MoWE layers leads to better performance."

## [POSITIVE] Freezing MoWE Experts During Finetuning
All MoWE experts are frozen during downstream finetuning to avoid overfitting and catastrophic forgetting of knowledge acquired during pretraining.

**Delta**: MoWE-Base on TriviaQA gets EM of 37.7 when freezing experts vs 33.5 when unfreezing (5 point drop).
**Condition**: Applied during finetuning on downstream tasks, especially knowledge-intensive ones.

**Evidence**: "MoWE-Base on TriviaQA gets EM of 37.7 when freezing the experts during finetuning. When we allow the update of experts during finetuning, EM drops by 5 points to 33.5."

## [POSITIVE] Placing MoWE Layers in Middle of Encoder/Decoder
MoWE layers are placed near the middle of the encoder and decoder (e.g., blocks 5 and 10 in Base, 9 and 17 in Large) to ensure tokens are somewhat contextualized before routing and that later layers can benefit from the MoWE output.

**Delta**: Using 2 MoWE layers in encoder and 2 in decoder gave best results (33.1 EM on TriviaQA vs 31.0 with 1 encoder layer).
**Condition**: When designing MoWE architecture with multiple MoWE layers.

**Evidence**: "We placed MoWE layers towards the middle of the encoder (decoder) because: (1) they receive a representation of the token that is already somewhat contextualized; (2) after the MoWE layer, there are still multiple Transformer Blocks that can benefit from the output of that layer."

## [POSITIVE] Using 32K Experts as a Sweet Spot
Experiments varying number of experts (16K, 32K, 64K) with fixed 1M routing vocabulary showed 32K experts provides the best performance.

**Delta**: 32K experts gave 35.3 F1 on TriviaQA vs 34.8 (16K) and 34.7 (64K).
**Condition**: When using a 1M routing vocabulary; optimal number may vary with vocabulary size.

**Evidence**: "In Fig. 6 we see that 32K experts seems to be a sweet spot in terms of number of experts for MoWE."

## [NEGATIVE] Matching Number of Experts to Vocabulary Size (1M experts) Degrades Performance
Increasing number of experts to match the routing vocabulary size (1M) while keeping total sparse parameters fixed by decreasing expert size leads to progressive degradation.

**Delta**: F1 on TriviaQA drops from ~34.2 (32K experts) to ~32.6 (1M experts).
**Condition**: When expert size is decreased proportionally to maintain fixed total parameters; attributed to sparser training updates and smaller expert capacity.

**Evidence**: "In Fig. 7 we show results for increasing MoWE-baseline for up to 1M experts. We see a progressive degradation in performance when matching the number of experts to the size of the vocabulary."

## [POSITIVE] Salient Span Masking (SSM) Additional Pretraining
Additional 40K pretraining steps using Salient Span Masking data from Guu et al. (2020) to specialize MoWE models on Wikipedia domain.

**Delta**: MoWE-Base + SSM achieves 44.9 EM on TriviaQA vs 39.4 without SSM; MoWE-Large + SSM achieves 50.2 EM.
**Condition**: When applying MoWE to Wikipedia-centric knowledge tasks like TriviaQA.

**Evidence**: "To make MoWE models a little more specialized on Wikipedia domain, which is known to benefit tasks such as TriviaQA, we followed (Roberts et al., 2020) and used the Salient Span Masking (SSM) data from (Guu et al., 2020) to perform an additional number of 40K pretraining steps."

## [POSITIVE] Skipping Top 16 Most Frequent Tokens in Routing
The MoWE layer does not process the top 16 most frequent tokens (e.g., punctuation, non-content words) in the routing vocabulary, speeding up training without hurting performance.

**Delta**: Speeds up training time; no negative impact on downstream performance.
**Condition**: Applied during training and inference of MoWE models.

**Evidence**: "The MoWE layer does not process the top 16 most frequent tokens in the routing vocabulary, i.e. those tokens ids are never routed to an expert. These tokens are punctuation marks and other non-content words and we estimate they can represent up to 28% of the tokens in a batch. This speeds up the training time and does not hurt downstream performance, as these tokens are not content words."
