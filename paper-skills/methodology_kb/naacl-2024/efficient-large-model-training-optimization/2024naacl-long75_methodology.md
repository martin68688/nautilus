# Fine-Tuning Language Models with Reward Learning on Policy

**Source**: https://aclanthology.org/2024.naacl-long.75/

## [POSITIVE] Reward Learning on Policy (RLP)
An unsupervised framework that refines a reward model using policy samples to keep it on-distribution, rather than relying on a fixed reward model trained on offline data.

**Delta**: Outperforms state-of-the-art baselines including PPO-based RLHF
**Condition**: When policy optimization shifts the LLM's data distribution, causing the fixed reward model to be inaccurate off-distribution.

**Evidence**: "Extensive experiments on three benchmark datasets show that RLP consistently outperforms the state-of-the-art."

## [POSITIVE] Unsupervised Multi-View Learning (UML)
Constructs two views for an input by generating two responses from the policy, then optimizes a multi-view information bottleneck loss to learn robust representations of the policy's data distribution.

**Delta**: Outperforms InfoMax (+4.7% win-rate on AlpacaFarm), MVI (+1.0%), and CL (+0.9%)
**Condition**: When retraining the reward model to generalize to the policy's data distribution.

**Evidence**: "Results in Table 4 show that the information bottleneck loss used in RLP-UML performs better than all other variants."

## [POSITIVE] Synthetic Preference Generation (SPG)
Generates a set of outputs for an instruction, quantifies uncertainty via the size of the largest semantic equivalence cluster, and selectively generates pairwise preferences for instructions with low uncertainty.

**Delta**: +3.4% win-rate on AlpacaFarm, +3.0% on LLMBar, +5.0% on Vicuna over PPO
**Condition**: When supplementing human preference data with synthetic data to keep the reward model on-distribution.

**Evidence**: "RLP-SPG brings up from a simulator win-rate of 46.8% to 50.2% in AlpacaFarm, 47.5% to 50.5% in LLMBar, and 57.5% to 62.5% in Vicuna."

## [POSITIVE] Multi-View Information Bottleneck (MIB) Loss
A loss function that retains task-relevant information in representations while discarding superficial information, using symmetrized KL divergence between two views.

**Delta**: Outperforms InfoMax (+4.7% on AlpacaFarm, +2.0% on LLMBar) and MVI (+1.0% on AlpacaFarm, +1.0% on LLMBar)
**Condition**: When learning robust representations of policy samples during reward model retraining.

**Evidence**: "The approach of explicitly removing superficial information in RLP-UML makes it outperform InfoMax and MVI by 4.7% and 1.0% in AlpacaFarm, and 2.0% and 1.0% in LLMBar."

## [POSITIVE] Selective Synthetic Preference Generation with Confidence Threshold
Uses a confidence score based on the size of the largest semantic equivalence cluster, and only generates synthetic preferences for instructions with confidence >= γ (set to 0.5).

**Delta**: Outperforms SelectAll (γ=0) which rejects no data: 50.2% vs 48.7% win-rate on AlpacaFarm
**Condition**: When generating synthetic preference data to ensure high-quality labels and avoid noisy low-confidence instructions.

**Evidence**: "The confidence score based on multiple sampling can be used for selective generation and further improve preference quality."

## [POSITIVE] Sampling a Set of Outputs (n=10) Instead of a Pair
For each instruction, generates n=10 outputs from the policy to construct a set, enabling diversity and uncertainty quantification, rather than just a single pair.

**Delta**: Outperforms pair-based sampling (RLAIF: 46.0% win-rate, Reward: 48.9%) with 50.2% win-rate on AlpacaFarm
**Condition**: When constructing synthetic preference data from policy samples.

**Evidence**: "Sampling a set of outputs rather than a pair for each instruction helps encourage output diversity and leads to high-quality preference generation."

## [POSITIVE] Iterative Retraining of Reward Model and Policy
After initial RLHF (steps 1-3), the reward model is retrained using policy samples (step 4), then the policy is retrained on the updated reward model (step 5).

**Delta**: Outperforms standard serial RLHF (PPO) across all benchmarks
**Condition**: When the policy's data distribution shifts significantly from the initial preference data distribution.

**Evidence**: "Both the two variants of RLP outperform all implemented baselines that do not train reward models using policy samples."

## [POSITIVE] Bidirectional Entailment Clustering for Semantic Equivalence
Clusters policy outputs into groups with a bidirectional entailment algorithm using Deberta-large, to identify semantically equivalent sentences.

**Delta**: Enables accurate confidence estimation and selective generation
**Condition**: When measuring uncertainty and constructing synthetic preference data from multiple policy outputs.

**Evidence**: "We cluster items of y into groups G with a bi-directional entailment algorithm. Sentences from each group are expected to share the same meaning."
