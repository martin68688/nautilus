---
title: Reward Model Overoptimization in Feedback Training Requires On-Policy Correction
confidence: HIGH
papers: [2024naacl-long.75, 2024naacl-long.451]
---

# Reward Model Overoptimization in Feedback Training Requires On-Policy Correction

Multiple papers identify and address the overoptimization problem in reinforcement learning from human feedback (RLHF), where the reward model's score increases while actual quality declines as the policy shifts out of distribution. Both papers propose methods to keep the reward model on-policy.

**Reward Learning on Policy (RLP)** (2024naacl-long.75) identifies that fixed reward models trained on offline data become inaccurate as the policy's data distribution shifts during optimization. RLP refines the reward model using policy samples through two variants: (1) RLP-UML uses unsupervised multi-view learning with a multi-view information bottleneck loss to learn robust representations from policy samples, outperforming InfoMax (+4.7% win-rate on AlpacaFarm); (2) RLP-SPG generates synthetic preferences from policy outputs using semantic equivalence clustering (n=10 outputs per instruction) and selective generation with confidence threshold γ=0.5, achieving +3.4% win-rate on AlpacaFarm over PPO.

**QE-based Feedback Training** (2024naacl-long.451) identifies overoptimization when using Quality Estimation (QE) models as reward models for machine translation: "reward increases while translation quality declines." The paper proposes RAFT+ which uses heuristic penalties for length-ratio errors and off-target translations (setting reward to -∞ for such errors). RAFT+ achieves consistent improvements: average BLEURT +2.7 (LLaMA-2-7B high-resource) and +6.6 (NLLB-200-1.3B low-resource). The paper also notes that even COMET (a reference-based metric) can be overoptimized, recommending BLEURT as a more reliable evaluation metric.

## Papers & Evidence
- `2024naacl-long.75` (RLP): "Extensive experiments on three benchmark datasets show that RLP consistently outperforms the state-of-the-art." — On-policy reward model retraining with multi-view learning.
- `2024naacl-long.451` (QE-based Feedback): "We first identify the overoptimization problem during QE-based feedback training, manifested as an increase in reward while translation quality declines." — Heuristic penalty (RAFT+) to mitigate overoptimization.

## Actionable Guidance
When using reward models for feedback training (RLHF, RAFT, DPO): (1) periodically retrain the reward model on policy samples to keep it on-distribution, (2) use heuristic rules to detect and penalize obvious errors (length-ratio, off-target), (3) generate synthetic preferences from multiple policy outputs (n≥10) with confidence-based filtering, (4) use evaluation metrics that are independent of the reward model to detect overoptimization (e.g., BLEURT instead of COMET when COMET-like models are used as rewards).

**Condition**: When applying RLHF or reward-based feedback training where the policy distribution shifts significantly from the initial preference data.
**Confidence**: HIGH — Two papers independently identify and address the same overoptimization problem with complementary solutions.
