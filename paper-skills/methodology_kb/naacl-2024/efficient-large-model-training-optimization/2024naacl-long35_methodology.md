# ALoRA: Allocating Low-Rank Adaptation for Fine-tuning Large Language Models

**Source**: https://aclanthology.org/2024.naacl-long.35/

## [POSITIVE] AB-LoRA (Ablation-based LoRA)
A method to estimate the importance of each LoRA rank by evaluating the performance change when a single rank is zeroed out (M\r) versus when only that rank is kept (Mr), using the negative cross-entropy loss as the metric.

**Delta**: Outperforms variants using architecture weights (ALoRA-DNAS) and sensitivity-based metrics (ALoRA-Sensi) on BoolQ, ReCoRD, and SQUAD tasks.
**Condition**: Used during the pruning and allocation phase of ALoRA; requires a trained super-network and a validation batch.

**Evidence**: "The results show that ALoRA outperforms the two variants, demonstrating that our AB-LoRA method can provide better guidance in allocating LoRA ranks."

## [POSITIVE] Gradual pruning and reallocation of LoRA ranks
LoRA ranks are pruned gradually (nA ranks per iteration) based on importance scores, and the pruned budget is reallocated to un-pruned modules, with priority given to modules with higher average importance scores.

**Delta**: Outperforms SoRA and LoRA across multiple tasks (e.g., +0.9 accuracy on SST-2, +1.3 accuracy on RTE, +0.6 F1 on SQUAD).
**Condition**: Applied after initial training (K1 epochs) and repeated NA times; works under a fixed total rank budget R_target.

**Evidence**: "Our ALoRA method can outperform the recent baselines with comparable tunable parameters."

## [POSITIVE] Uniform initial LoRA rank distribution
Instead of initializing with a large maximum rank (as in AdaLoRA, SoRA), ALoRA initializes each LoRA module with rank = R_target / Nmod, meeting the budget from the start.

**Delta**: Requires less GPU memory than SoRA (18.1 GB vs 18.8 GB) and comparable training speed.
**Condition**: Used at initialization; applicable when the total rank budget is predefined.

**Evidence**: "ALoRA requires less memory costs during training than SoRA since it does not initialize with a larger LoRA rank."

## [NEUTRAL] Gate units for rank control (DNAS-style relaxation)
Gate units α_i (initialized to 1) are injected for each LoRA rank, relaxed to continuous values via sigmoid for differentiability, enabling bi-level optimization for architecture search.

**Delta**: AB-LoRA (which does not rely on these gate values) outperforms the DNAS variant (ALoRA-DNAS) that uses gate values as importance scores.
**Condition**: Used as a formulation for differentiable search; not used as the final importance metric.

**Evidence**: "The architecture weights are not reliable indicators for the final LoRA allocation’s performance."

## [POSITIVE] Importance score based on cross-entropy loss (not accuracy/F1)
The metric S() for importance scoring is set to the negative cross-entropy loss instead of accuracy or F1, because accuracy/F1 may not vary when masking a single rank and is unsuitable for generative tasks.

**Delta**: Enables effective rank pruning across both classification and generation tasks.
**Condition**: Used in AB-LoRA for computing importance scores on a validation batch.

**Evidence**: "We set S() as the negative of the cross-entropy (CE) loss since the widely applied metrics like accuracy or F1: (a) may not vary if the super-network only masks out a single operation, and (b) is not suitable for generative language model fine-tuning."

## [NEUTRAL] Bi-level optimization (DNAS-style) for architecture weights
The gate parameters α' are optimized via bi-level optimization on a split of the training set (D1 for model weights, D2 for architecture weights), following the DNAS framework.

**Delta**: AB-LoRA (which avoids bi-level optimization) achieves better results than the DNAS variant.
**Condition**: Used in the ALoRA-DNAS variant; not used in the final ALoRA method.

**Evidence**: "We freeze the architectural parameters and train only the model parameters on the train set. No bi-level optimization is required, thus saving training time costs."

## [POSITIVE] Post-pruning fine-tuning (K2 epochs)
After each pruning and allocation step, the super-network is further trained for K2 epochs (e.g., 0.25 epoch) to recover performance lost due to pruning.

**Delta**: Enables gradual rank reallocation without significant performance drop.
**Condition**: Applied after each pruning iteration; K2 is set to 0.25 epoch in experiments.

**Evidence**: "After the pruning and adding operations, we tune the altered super-network for K2 > 0 epochs to recover the lost performance."

## [POSITIVE] Priority-based rank allocation to un-pruned modules
When reallocating pruned ranks, modules with higher average importance scores receive more additional ranks (e.g., if nA=8 and three modules remain, the highest-scoring module gets 3, the next gets 3, the lowest gets 2).

**Delta**: Outperforms uniform allocation; leads to better task-specific adaptation.
**Condition**: Used during the allocation step in Algorithm 1.

**Evidence**: "If nA is not divided by the number of un-pruned modules, we allocate the nA ranks as uniformly as possible, with priority given to modules with higher average importance scores."

## [POSITIVE] Using LlaMA-2 7B as backbone with LM head only
Fine-tuning is performed using only the language modeling head (no additional classification heads), with beam search decoding (beam size 5).

**Delta**: Achieves state-of-the-art results on GLUE, SuperGLUE, E2E, and instruction tuning tasks.
**Condition**: Used for all main experiments; applicable to generative LLMs.

**Evidence**: "When fine-tuning LlaMA-2 7B, we only consider the supervised fine-tuning (SFT) setting, that is, all the predictions are generated using the language modeling head (LM head) upon receiving a prompt or an instruction."

## [NEUTRAL] Early stopping with comparable total time cost
Training is stopped early when performance plateaus; ALoRA's total training time (3.81h) is comparable to SoRA (3.63h) and LoRA (2.68h) despite the iterative pruning process.

**Delta**: Time cost is slightly higher than LoRA but comparable to SoRA; memory cost is lower than SoRA.
**Condition**: Used in efficiency comparison experiments on the E2E task.

**Evidence**: "Under early stopping, the total training time cost of ALoRA remains comparable with SoRA and LoRA."
