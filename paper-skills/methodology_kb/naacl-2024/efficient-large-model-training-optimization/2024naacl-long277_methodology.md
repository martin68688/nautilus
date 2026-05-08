# Generalizable and Stable Finetuning of Pretrained Language Models on Low-Resource Texts

**Source**: https://aclanthology.org/2024.naacl-long.277/

## [POSITIVE] Attention-Guided Weight Mixup
Each network weight is represented as a mixup of task-specific weight and pretrained weight, controlled by a learnable attention parameter (alpha) in [0,1]. This is a continuous relaxation of discrete sub-network selection.

**Delta**: Outperforms best baseline by 6.43%, 3.43%, and 1.68% on 300, 500, and 1K training samples respectively
**Condition**: When finetuning PLMs on low-resource datasets (300, 500, 1000 training examples)

**Evidence**: "Our method outperforms the best baseline by 6.43%, 3.43%, and 1.68% on 300, 500, and 1K training samples scenarios, respectively."

## [POSITIVE] Bi-Level Optimization (BLO) Framework
Task weights are learned on one split of training data (lower level) while attention parameters are learned on a separate split (upper level), using 80%-20% split of the training dataset.

**Delta**: Outperforms Joint Training by +1.56 average score (80.42 vs 78.86) with lower std (0.93 vs 1.48)
**Condition**: When combating overfitting and improving generalization on low-resource datasets

**Evidence**: "The results underscore the superior performance of our method across all datasets, yielding a higher average score and a reduced standard deviation."

## [POSITIVE] Low-Rank Approximation of Attention Parameters
The alpha matrix is decomposed into product of two lower-rank matrices (rank r=1) to mitigate overfitting in low-resource settings.

**Delta**: Empirically better performance after rank decomposition
**Condition**: In low-resource domains to mitigate overfitting

**Evidence**: "We use the following weight formulation that proves to have better performance empirically after rank decomposition"

## [POSITIVE] Two-Phase Training (Search + Finetune)
First phase (search) learns optimal attention parameters via BLO; second phase (finetune) further adjusts task weights on entire training dataset with learned attention parameters fixed.

**Delta**: Outperforms vanilla finetuning across all PLMs tested
**Condition**: When finetuning PLMs on low-resource datasets

**Evidence**: "Our method demonstrated its effectiveness across several challenging datasets from the GLUE benchmark, outperforming baselines in low-resource scenarios."

## [POSITIVE] Continuous Relaxation of Discrete Child Network Selection
Replaces FIM-based discrete sub-network selection with continuous optimization using attention parameters, avoiding heuristic-based selection.

**Delta**: Outperforms CHILD-TUNING_D and DPS dense by significant margins
**Condition**: When FIM-based selection is suboptimal due to data scarcity in low-resource settings

**Evidence**: "Our method surpasses all other baselines in terms of average scores, with a particularly notable improvement on the CoLA dataset over baselines"

## [POSITIVE] Weight Decay on Attention Parameters
L2 regularization on alpha parameters encourages values close to 0, promoting higher contribution from pretrained weights.

**Delta**: Part of overall method achieving best results
**Condition**: During the upper-level optimization in BLO framework

**Evidence**: "Weight decay on α encourages the values of α to be close to 0 encouraging a higher contribution from the pretrained weights in the resultant weight estimation."

## [NEUTRAL] Finite Difference Approximation for Hessian-Vector Product
Approximates computationally expensive Hessian-vector product in gradient estimation during BLO optimization.

**Delta**: Enables practical computation, no performance degradation reported
**Condition**: When computing gradients for upper-level optimization in BLO

**Evidence**: "This is approximated using finite-difference approximation (Liu et al., 2018), as described in detail in Appendix A."

## [POSITIVE] 80%-20% Training Split for BLO
Training dataset split into 80% for lower-level task weight learning and 20% for upper-level attention parameter learning.

**Delta**: Better empirical results than 50%-50% split
**Condition**: When implementing the BLO framework for low-resource finetuning

**Evidence**: "50%-50% split setting was explored however 80%-20% split gave better empirical results"

## [NEGATIVE] Joint Training (Ablation - Negative Baseline)
Jointly optimizing W and alpha parameters by minimizing loss on whole training set instead of using BLO.

**Delta**: Performs worse than BLO: 78.86 vs 80.42 average; on MRPC performs worse than vanilla baseline
**Condition**: When compared against BLO approach; not recommended

**Evidence**: "JointTraining performs worse than the vanilla baseline... simultaneous learning of both W and α parameters on the training dataset can lead to overfitting"

## [NEGATIVE] Randomly Fixed Alpha (Ablation - Negative Baseline)
Randomly initializing alpha parameters from Gaussian distribution and keeping them fixed during finetuning.

**Delta**: Performance degrades with higher variance: 79.36 (sigma=0.005) to 69.29 (sigma=0.45) average
**Condition**: When compared against learnable alpha; not recommended

**Evidence**: "Our method with learnable α outperforms the model with randomly initialized α, which underscores the importance of learning the attention parameters"
