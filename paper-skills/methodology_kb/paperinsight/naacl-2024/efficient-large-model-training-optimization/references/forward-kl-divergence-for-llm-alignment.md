---
title: Forward KL Divergence with On-Policy Sampling Improves LLM Alignment
confidence: MEDIUM
papers: [2024naacl-long.455, 2024naacl-long.75]
---

# Forward KL Divergence with On-Policy Sampling Improves LLM Alignment

Two papers demonstrate that optimizing forward KL divergence (rather than reverse KL as in standard RLHF) combined with on-policy sampling from the target distribution leads to better alignment outcomes.

**ARM** (2024naacl-long.455) optimizes forward KL divergence DKL(π* || πθ) from the target policy to the parametric policy, formulating the target policy as a residual energy-based model: π*(y|x) = (1/Z(x)) * π_sft(y|x) * exp((1/β) * r_φ(x,y)). Self-normalizing importance sampling draws n=32 proposals from π_sft and resamples based on reward scores. ARM with Bradley-Terry (using soft preference labels from the reward model) achieves +7% to +15% over PPO, DPO, and best-of-n. With only 8K preference samples (10% of full dataset), ARM recovers a large proportion of full-data performance.

**RLP** (2024naacl-long.75) keeps the reward model on-policy by retraining it on policy samples. RLP-UML uses unsupervised multi-view learning with a multi-view information bottleneck loss to learn robust representations from policy samples. RLP-SPG generates synthetic preferences from n=10 policy outputs per instruction, using bidirectional entailment clustering for semantic equivalence and selective generation with confidence threshold γ=0.5. RLP achieves +3.4% win-rate on AlpacaFarm, +3.0% on LLMBar, and +5.0% on Vicuna over PPO.

## Papers & Evidence
- `2024naacl-long.455` (ARM): "In this work, we propose to optimize the forward KL, DKL(π* || πθ). ... Our extensive experiments demonstrate its strong performance across multiple datasets, compared to strong baselines like PPO, DPO." — Forward KL with residual EBM and on-policy sampling.
- `2024naacl-long.75` (RLP): "Both the two variants of RLP outperform all implemented baselines that do not train reward models using policy samples." — On-policy reward model retraining.

## Actionable Guidance
For LLM alignment: (1) optimize forward KL divergence (ARM) instead of reverse KL (PPO/DPO) to avoid mode-seeking behavior, (2) use on-policy sampling from the target distribution (self-normalizing importance sampling with n=32 proposals), (3) retrain the reward model on policy samples (RLP) to keep it on-distribution, (4) use soft preference labels from the reward model rather than binary chosen/rejected labels.

**Condition**: When aligning LLMs with human preferences where standard PPO/DPO shows mode collapse or reward overoptimization.
**Confidence**: MEDIUM — Two papers support the on-policy direction but use different formulations (forward KL vs. reward retraining).
