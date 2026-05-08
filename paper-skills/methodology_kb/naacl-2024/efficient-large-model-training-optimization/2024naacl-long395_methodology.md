# Bridging the Gap between Different Vocabularies for LLM Ensemble

**Source**: https://aclanthology.org/2024.naacl-long.395/

## [POSITIVE] Cross-Model Vocabulary Alignment via Overlapping Tokens
Uses overlapping tokens between LLM vocabularies as supervised labels to learn a mapping matrix that projects token embeddings from different models into a shared space, enabling distribution comparison.

**Delta**: Outperforms individual models and previous ensemble methods across all tasks
**Condition**: Applicable when ensembling LLMs with different vocabularies; requires overlapping tokens between vocabularies.

**Evidence**: "EVA bridges the lexical gap among various LLMs, enabling meticulous ensemble at each generation step."

## [POSITIVE] Noise Reduction via Three-Step Truncation (Top-t, Threshold, Variance)
Reduces noise in the similarity matrix by retaining only top-t similar tokens, discarding low-similarity pairs below a threshold, and removing tokens with low variance in similarity scores.

**Delta**: Produces a sparse mapping matrix of only ~1MB
**Condition**: Applied during vocabulary alignment to filter out meaningless or noisy token alignments.

**Evidence**: "Following these three processes, we obtain a sparse and efficient mapping matrix W_QP, which is only about 1MB."

## [POSITIVE] Filtering Strategy for Unfaithful Tokens
Excludes a model from the ensemble if its top-1 predicted token is not within the top-n tokens predicted by any other model, enforcing consistency among models.

**Delta**: +10.61 accuracy on GSM8K compared to best individual model
**Condition**: Particularly effective for arithmetic reasoning tasks where output tokens are diverse; less impactful for deterministic tasks like machine translation.

**Evidence**: "When we directly average the probability distributions... Upon incorporating the filtering strategy with n=3... we ensemble only Q2 and Q3, resulting in the correct output."

## [POSITIVE] Pivot Model Selection (Largest Vocabulary)
Selects the model with the largest vocabulary as the pivot model to project all other model distributions into its space, avoiding out-of-vocabulary issues and ensuring finer token granularity.

**Delta**: Practical and effective without requiring prior knowledge of model performance
**Condition**: Used when ensembling models with different vocabulary sizes; larger vocabulary provides more specific tokens.

**Evidence**: "We choose the model with the largest vocabulary as the pivot model in our method, rather than selecting the best-performing model."

## [POSITIVE] Top-k Truncation on Output Distributions
Truncates the original output distribution of each candidate model to the top-k tokens to mitigate long-tail noise accumulation.

**Delta**: Robust across various tasks with k=320
**Condition**: Applied at each generation step before ensemble to reduce noise from low-probability tokens.

**Evidence**: "To mitigate the impact of long-tail noise accumulation, we perform top-k truncation on the original output distributions of each candidate model."

## [POSITIVE] Greedy Decoding for All Experiments
Uses greedy decoding (argmax) instead of sampling for generating tokens, as it generally produces higher-quality outputs.

**Delta**: Generally produces higher-quality outputs
**Condition**: Used across all tasks and models in the ensemble.

**Evidence**: "We employ greedy decoding in all experiments since it generally produces higher-quality outputs."

## [POSITIVE] Ensemble via Averaging Mapped Distributions with Filtering
Projects output distributions of all models into the pivot model's vocabulary space using learned mappings, then averages them after applying the filtering strategy to exclude unfaithful models.

**Delta**: Outperforms LLM-Blender (3B fusion model) on 6 out of 8 tasks
**Condition**: Applicable at each generation step for any set of LLMs with different vocabularies.

**Evidence**: "EVA achieves correct answers by performing fine-grained ensemble at each generation step, allowing each token to benefit from the ensemble."

## [NEUTRAL] Zero-Shot and Few-Shot Prompting with Task-Specific Formats
Uses 4-shot in-context learning for machine translation and zero-shot inference for other tasks, with chain-of-thought prompts for arithmetic reasoning, adhering to each model's specific chat format.

**Delta**: Standard prompting approach; no quantitative comparison provided
**Condition**: Applied across all tasks to elicit responses from chat models.

**Evidence**: "For machine translation tasks, we utilize a 4-shot in-context learning setting, whereas for other tasks, we conduct zero-shot inference."
