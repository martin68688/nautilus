# Structured Pruning for Large Language Models Using Coupled Components Elimination and Minor Fine-tuning

**Source**: https://aclanthology.org/2024.findings-naacl.1/

## [POSITIVE] Coupled Components Elimination
Identifies and removes coupled rows/columns (kernels) across different layers simultaneously, preserving dependency relationships within remaining parameters.

**Delta**: 20% parameter reduction while retaining at least 94.4% of original performance metrics
**Condition**: Applicable to Transformer-based LLMs with multi-head attention and FFN layers.

**Evidence**: "It removes coupled components across different layers simultaneously and preserves dependency relationships within remaining parameters, avoiding significant performance degradation."

## [POSITIVE] Minimal Fine-tuning (single stage after pruning)
Instead of iterative prune-finetune cycles, the algorithm performs all pruning first, then a single stage of few-epoch fine-tuning on a small dataset.

**Delta**: Outperforms iterative pruning baselines in generalization; retains 94-96% of original performance
**Condition**: Used after all pruning iterations are complete; fine-tuning on Alpaca dataset for 4 epochs.

**Evidence**: "The intuition of our algorithm is to limit the finetuning operations as few as possible, so that the pruned model will not import too much bias towards specific tasks."

## [POSITIVE] Kernel vs. Feature Partitioning
Rows/columns in parameter matrices are classified as 'kernels' (receive all features) or 'features' (receive specific features), enabling targeted pruning.

**Delta**: Enables structured pruning without specialized libraries
**Condition**: Applied to all parameter matrices in Transformer layers.

**Evidence**: "We divide them into kernels and features based on their functionalities in the inference computation."

## [POSITIVE] Importance Scoring with PLATON-style Sensitivity and Uncertainty
Combines sensitivity (first-order Taylor approximation) and uncertainty (upper confidence bound) to score coupled components and features.

**Delta**: Outperforms L2 weight pruning and random pruning significantly
**Condition**: Used during iterative evaluation-pruning steps; hyperparameters x1=x2=0.5.

**Evidence**: "We refer to the evaluation method proposed by PLATON, which combines the sensitivity of the network to determine the final score for the coupled components."

## [POSITIVE] LoRA Gradient Substitution for Evaluation
Uses LoRA gradients instead of full-scale fine-tuning gradients to reduce computational overhead during importance evaluation.

**Delta**: Reduces computational overhead during pruning
**Condition**: Applied during evaluation phase; LoRA matrices pre-trained for 5 iterations on 10 sentences from C4 dataset.

**Evidence**: "We employ LoRA fine-tuning to restore model performance, and use LoRA gradients instead of full-scale fine-tuning gradients to reduce the computational overhead during pruning."

## [POSITIVE] Uniform Layer-wise Pruning (instead of global pruning)
Prunes the same number of components (heads, kernels, features) from every layer uniformly, avoiding prior knowledge about which layers are important.

**Delta**: Outperforms global pruning which concentrated pruning in early layers and caused severe degradation
**Condition**: Applied to all layers equally; avoids need for prior knowledge about which layers cannot be pruned.

**Evidence**: "We adopt a simpler strategy of uniform pruning for every layer."

## [POSITIVE] Head-level Pruning for Self-Attention (instead of kernel-level)
Removes entire self-attention heads (lowest-scoring) rather than individual kernels within heads, as low-importance kernels concentrate in the same head.

**Delta**: Outperforms kernel-level pruning in BERT, ViT, LLaMA, and Vicuna experiments
**Condition**: Applied to self-attention layers; one head removed per layer per iteration.

**Evidence**: "The latter option performs better when the number of parameters keeps the same. This is because the distribution of importance in the model is not uniform, and low-importance kernels are often concentrated within the same self-attention head."

## [POSITIVE] Feature Pruning Across All Parameter Matrices
Removes the same feature position across all parameter matrices simultaneously, evaluated using only self-attention layers.

**Delta**: Contributes to 20% MACs reduction
**Condition**: Applied to all parameter matrices; 128 features (dk=128) removed per iteration.

**Evidence**: "We only need to group all corresponding features at the same position in the model."

## [POSITIVE] Coupled Component Grouping for Three-Layer FFN (Gate, Up, Down)
Approximates coupled components in LLaMA-style FFN as two sub-components (di*gi and di*ui) and sums their scores.

**Delta**: Enables effective pruning of three-layer FFN structures
**Condition**: Applied to intermediate layers with Gate, Up, Down projections (e.g., LLaMA, ChatGLM3).

**Evidence**: "We approximate the coupled component (di, gi, ui) as two sub-components: digi and diui."

## [POSITIVE] ChatGLM3-specific Head Reordering After Pruning
After pruning Query heads in ChatGLM3 (which has 32 Query heads but only 2 Key/Value heads), remaining Query heads are reordered by parity to maintain correct correspondence.

**Delta**: Retains 94% of original performance at 20% sparsity
**Condition**: Specific to ChatGLM3 architecture with asymmetric Query/Key-Value head counts.

**Evidence**: "We rearrange the Query heads and the parameter matrix O according to their parity as Figure 2."

## [POSITIVE] Proportional Kernel Pruning in Intermediate Layers
Prunes r × dk kernels per intermediate layer, where r = im / (headnum × dk), maintaining constant ratio between FFN and attention dimensions.

**Delta**: Enables consistent pruning across layers without prior knowledge
**Condition**: Applied to FFN layers; ratio r is model-specific (e.g., ~2.7 for LLaMA, 4 for OPT).

**Evidence**: "In each iteration, we prune r × dk kernels for each parameter matrix in the intermediate layers, where r = im/(headnum × dk)."
