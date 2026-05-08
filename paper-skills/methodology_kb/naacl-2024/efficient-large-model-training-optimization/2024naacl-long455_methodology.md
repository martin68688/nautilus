# ARM: Alignment with Residual Energy-Based Model

**Source**: https://aclanthology.org/2024.naacl-long.455/

## [POSITIVE] Forward KL Divergence Optimization
Instead of minimizing reverse KL divergence (as in RLHF), ARM minimizes forward KL divergence DKL(π* || πθ) from the target policy to the parametric policy.

**Delta**: outperforms baselines
**Condition**: Used as the core optimization objective for aligning LLMs with human preferences.

**Evidence**: "In this work, we propose to optimize the forward KL, DKL(π* || πθ). ... Our extensive experiments demonstrate its strong performance across multiple datasets, compared to strong baselines like PPO, DPO."

## [POSITIVE] Residual Energy-Based Model (EBM) for Target Policy
The target policy π* is formulated as a residual energy-based model: π*(y|x) = (1/Z(x)) * π_sft(y|x) * exp((1/β) * r_φ(x,y)), where r_φ is the learned reward function.

**Delta**: enables sampling from target policy
**Condition**: Used to define the ideal aligned distribution that ARM aims to learn from.

**Evidence**: "Our proposal is driven by the observation that the target distribution π* is a residual energy-based model (EBM)..."

## [POSITIVE] Self-Normalizing Importance Sampling
Samples from the EBM target policy by first drawing n proposals from π_sft, then resampling based on the exponential of the reward score divided by β.

**Delta**: outperforms best-of-n greedy selection
**Condition**: Used to generate training samples from the target policy π*.

**Evidence**: "The difference is that sampling from π* is a probabilistic approach while best-of-n is greedy."

## [POSITIVE] Expert Iteration with MLE
A simpler variant where samples from π* are used to train the policy via maximum likelihood estimation (behavior cloning).

**Delta**: +14% to +30% over SFT
**Condition**: Used as a baseline and intermediate step; works well but is outperformed by full ARM.

**Evidence**: "Simple training method, Expert Iteration, achieves a 14% to 30% improvement over SFT."

## [POSITIVE] ARM with DPO (Bradley-Terry)
Samples two responses from π*, scores them with the reward model to create soft preference labels, then trains the policy using a modified DPO objective with soft labels.

**Delta**: +7% to +15% over PPO, DPO, best-of-n
**Condition**: Primary method; used when pairwise preference data is available.

**Evidence**: "In comparison to the previously top-performing methods, PPO, DPO, and best-of-n, our method also exhibits substantial improvements, ranging from 7% to 15%."

## [POSITIVE] Soft Label Preference Learning
Instead of binary chosen/rejected labels, ARM uses probabilistic preference scores (ρ) from the reward model as soft labels in the DPO loss.

**Delta**: outperforms binary DPO
**Condition**: Used in ARM with Bradley-Terry; leverages continuous reward values.

**Evidence**: "Given a well-learned reward function, our approach has an advantage."

## [POSITIVE] On-Policy Sampling from Target Distribution
ARM samples from the exponentially-tilted SFT policy (π*), making the training more on-policy compared to standard DPO which uses off-policy data.

**Delta**: outperforms off-policy DPO
**Condition**: Used in ARM training; contributes to better alignment.

**Evidence**: "The original DPO learning is off-policy... In contrast, our method is more on-policy since it learns from samples of an exponentially-tilted SFT-policy."

## [NEUTRAL] ARM with Plackett-Luce
Extension of ARM to handle K > 2 responses per prompt, using the Plackett-Luce ranking model and a generalized DPO loss with cross-entropy over rankings.

**Delta**: slightly underperforms Bradley-Terry variant
**Condition**: Used when multiple responses per prompt are available; requires more accurate reward labels.

**Evidence**: "In both datasets, ARM with Plackett-Luce slightly underperforms ARM with Bradley-Terry."

## [POSITIVE] Non-Pairwise Preference Learning
Adapts ARM to settings where only single responses with binary feedback (like/dislike) are available, by training a reward function with a regression loss instead of pairwise loss.

**Delta**: outperforms Expert Iteration and best-of-n
**Condition**: Used when only non-pairwise preference data is available (e.g., real-world chatbot logs).

**Evidence**: "ARM outperforms Expert Iteration and best-of-n. ... our method still yields substantial enhancements over the SFT model."

## [POSITIVE] Low-Resource Training with ARM
Training ARM with limited pairwise preference data (2k or 8k samples instead of full dataset).

**Delta**: recovers large proportion of full-data performance with 8k samples
**Condition**: Used when human preference data is scarce.

**Evidence**: "our method with 8k data (accounting for 10% or less of the full dataset) can recover a large proportion of the performance of the models trained with the full dataset, especially on Anthropic-HH."

## [POSITIVE] Number of Proposal Samples (n=32)
Using 32 proposal samples from π_sft for self-normalizing importance sampling.

**Delta**: 50.1 win-rate vs 47.8 for n=16
**Condition**: Optimal setting for self-normalizing importance sampling in ARM.

**Evidence**: "as n increases from 16 to 32, we observe a clear improvement on win rate. However, further increasing n yields no improvement."
