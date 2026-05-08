# Rehearsal-Free Modular and Compositional Continual Learning for Language Models

**Source**: https://aclanthology.org/2024.naacl-short.39/

## [POSITIVE] Rehearsal-free parameter isolation with PEFT
Allocates task-specific parameters using parameter-efficient fine-tuning (PEFT) modules (prefix-tuning or LoRA) instead of storing data examples. The PLM is frozen, and only PEFT parameters are trained. After training on a task, its module is frozen to prevent forgetting.

**Delta**: Outperforms state-of-the-art methods on near-domain and far-domain benchmarks
**Condition**: All continual learning settings (TIL and CIL) across near-domain and far-domain benchmarks

**Evidence**: "MoCL avoids catastrophic forgetting without storing additional data and facilitates effective knowledge transfer via module composition."

## [POSITIVE] Module composition via task matching weights
During training on a new task, previously learned task-specific modules are composed via a weighted summation. Weights are computed as cosine similarity between input embeddings and trainable task feature vectors.

**Delta**: +7.7 points on AfriSenti compared to ProgPrompt; +7.81 and +4.36 points better than per-task FT on WOS and AfriSenti
**Condition**: All settings, especially effective for near-domain tasks where knowledge transfer is beneficial

**Evidence**: "MoCL composes task modules via task matching, thus avoiding negative interference between tasks while exploiting similarities for knowledge transfer."

## [POSITIVE] Trainable task feature vectors for task matching
Introduces trainable task feature vectors V ∈ R^(N×D) to capture salient features of tasks. Cosine similarity between input embeddings and these vectors determines contribution weights for module composition.

**Delta**: Enables effective knowledge transfer and outperforms baselines
**Condition**: Used during training and inference for both TIL and CIL settings

**Evidence**: "To calculate the contribution weights α_k of each task-specific module, we introduce trainable task feature vectors V ∈ R^(N×D) to capture salient features of tasks in the CL sequence."

## [POSITIVE] Per-instance task matching and module composition
Task matching is performed per-instance (each input sample gets its own composition weights), rather than per-task. Weights are averaged across examples for analysis.

**Delta**: Enables better CIL performance without explicit task identification
**Condition**: Class-incremental learning (CIL) setting where task identities are not provided during testing

**Evidence**: "As MoCL performs per-instance task matching and module composition, we average the weights across all examples from a given task."

## [POSITIVE] Freezing task-specific modules after training
Once training on a task is finished, the corresponding PEFT parameters are frozen to preserve task-specific knowledge in subsequent training, avoiding catastrophic forgetting.

**Delta**: Enables rehearsal-free continual learning without forgetting
**Condition**: All continual learning settings

**Evidence**: "To avoid catastrophic forgetting, the task-specific module is frozen once the training on the respective task is finished."

## [POSITIVE] Joint optimization of PEFT module and task feature vector
Training objective minimizes cross-entropy loss while maximizing cosine similarity between task feature vector and corresponding task input embeddings.

**Delta**: Enables effective task matching and knowledge transfer
**Condition**: During training of each new task

**Evidence**: "The training objective for the n-th task is to find the PEFT module P_n and the task feature vector v_n that minimize the cross-entropy loss of training examples, and, at the same time, maximize the cosine similarity between v_n and the corresponding task input embeddings."

## [POSITIVE] Using task identities during inference (TIL) vs. matching scores (CIL)
In TIL, directly select the task-specific module for inference. In CIL, use matching scores between task inputs and feature vectors for module composition.

**Delta**: Outperforms EPI in CIL setting on most benchmarks
**Condition**: Task-incremental learning (TIL) vs. class-incremental learning (CIL) settings

**Evidence**: "During inference, as the task identities are available in the TIL setting, we directly select the task-specific module for inference. In the CIL setting, we use the matching scores between task inputs and the feature vectors for module composition."

## [NEUTRAL] Sparse weight distribution from task matching
The learned weight distribution is notably sparse, with some modules (e.g., ma, kr, ha) being reused across many tasks while others (e.g., pcm) are task-exclusive.

**Delta**: Observation only; suggests potential for module pruning
**Condition**: Observed on AfriSenti dataset; suggests future work for reducing number of modules

**Evidence**: "Moreover, we observe that there is a pronounced sparsity in the learned weight distributions."
