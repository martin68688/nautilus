# Shears: Unstructured Sparsity with Neural Low-rank Adapter Search

**Source**: https://aclanthology.org/2024.naacl-industry.34/

## [POSITIVE] Unstructured Sparsification with Wanda
Applying the Wanda algorithm to induce unstructured sparsity in the LLM by zeroing out less important weights based on weight magnitude and input activation norms, before fine-tuning.

**Delta**: Outperforms SparseFT across sparsity levels 0% to 60% on GSM8K with MPT-7B; achieves 50% sparsity with minor accuracy drop (e.g., LLaMA7B average accuracy 45.3% vs LoRA 46.9%)
**Condition**: Applied to LLMs (LLaMA-7B, LLaMA-13B, MPT-7B) before super-adapter training; effective for high sparsity levels (40-50%) while maintaining accuracy.

**Evidence**: "Shears obtains a model with 50% sparsity that contains 1.91× fewer non-zero parameters with minor drops in accuracy."

## [POSITIVE] Neural Low-rank Adapter Search (NLS)
A weight-sharing NAS approach that creates a super-adapter network with elastic low-rank adapters (LoRA) of varying ranks, trained by activating different sub-adapter configurations during forward/backward passes.

**Delta**: At 50% sparsity, Shears (NLS) achieves 45.3% average accuracy vs LoRA's 43.3% on LLaMA7B math reasoning; at 40% sparsity, outperforms all PEFT baselines on LLaMA13B (52.0% average vs LoRA 51.1%)
**Condition**: Applied after sparsification; particularly beneficial when model is sparsified (50% sparsity), but also effective without sparsity (comparable to LoRA).

**Evidence**: "With 50% sparsity, Shears outperforms LoRA significantly, highlighting its efficacy in enhancing model performance under sparsity conditions."

## [POSITIVE] Heuristic Sub-Adapter Configuration Extraction
A zero-cost heuristic to select a sub-adapter configuration approximately at the center of the search space, using the formula c = floor(n/2) for each adapter's elastic width index.

**Delta**: Heuristic achieves 45.3% average accuracy on LLaMA7B math reasoning at 50% sparsity, close to hill-climbing (45.9%) and better than minimal (43.5%)
**Condition**: Used as a fast initial configuration; suitable when computational budget is limited, providing near-optimal performance.

**Evidence**: "The heuristic obtained in O(1) already gives us a reliable indication of the quality of the sub-adapters around the mid-configuration space."

## [POSITIVE] Hill-Climbing Search for Sub-Adapter Optimization
A cost-effective search strategy that starts from the heuristic configuration and explores neighboring configurations to find improved sub-adapter networks.

**Delta**: Hill-climbing achieves 45.9% average accuracy vs heuristic 45.3% and RNSGA-II 45.7% on LLaMA7B math reasoning at 50% sparsity
**Condition**: Applied after heuristic extraction when additional computational budget is available; yields best accuracy among search methods tested.

**Evidence**: "If the user has the budget, a more refined sub-adapter configuration can be searched using a cost-effective hill-climbing strategy that is cheaper than other methods."

## [POSITIVE] Freezing Pruned Weights During Training
After unstructured sparsification, the pruned weight matrix Wp is kept frozen throughout the super-adapter training and sub-adapter search stages.

**Delta**: Enables 50% sparsity with only minor accuracy drop (45.3% vs LoRA 46.9% on LLaMA7B) while reducing trainable parameters significantly
**Condition**: Applied in all Shears experiments; critical for maintaining sparsity and reducing computational cost during fine-tuning.

**Evidence**: "Shears does not make the original model weights W elastic as opposed to the elastic configurations of the adapters."

## [POSITIVE] Zeroth-Order Sparsification (Cost-Effective)
Using Wanda which requires only a single forward pass with a tiny subset of inputs to compute importance scores, instead of iterative weight updates.

**Delta**: Obtaining Wp for a 7B parameter model takes less than five minutes on a single GPU
**Condition**: Applied in the first stage of Shears; suitable for resource-constrained environments where fast sparsification is needed.

**Evidence**: "When using Wanda, only a tiny subset of inputs needs to forward pass to get the unstructured importance measurements instead of more sophisticated approaches that require weight updates and training iterations."

## [NEUTRAL] Elastic Low-Rank Adapter Search Space (Rank [32, 24, 16])
Using a search space of low-rank adapter configurations with ranks 32, 24, and 16 for each target module, enabling different adapter sizes.

**Delta**: Narrow accuracy range across sub-adapter configurations (only ~1 percentage point difference between minimal and maximal)
**Condition**: Applied to all experiments; the specific rank choices [32, 24, 16] were used consistently across models and tasks.

**Evidence**: "Studies indicate a narrow accuracy range, with the difference in accuracy between the minimal and the maximal sub-adapter configuration being only a single accuracy percentage point."

## [POSITIVE] FP16 Precision for Pre-trained Weights
Using FP16 (half-precision) for storing and computing with the frozen pre-trained weights during fine-tuning.

**Delta**: Outperforms SparseFT which uses FP32, while using less memory and compute
**Condition**: Applied during super-adapter training; contributes to efficiency gains compared to full-precision fine-tuning approaches.

**Evidence**: "Shears utilizes FP16 precision for pre-trained weights, and the training process does not involve knowledge distillation. As shown in the figure, our approach, Shears, outperforms SparseFT across sparsity levels from 0% to 60%."

## [POSITIVE] Adapter Target Modules Selection (Q, K, V, Up, Gate, Down)
Applying LoRA adapters to specific linear projection modules in the transformer architecture (query, key, value, up-projection, gate, down-projection).

**Delta**: Shears with NLS outperforms LoRA with same target modules at 50% sparsity (45.3% vs 43.3% on LLaMA7B math)
**Condition**: Applied consistently across all experiments for fair comparison; specific modules may vary by model (e.g., MPT-7B includes 'O' module).

**Evidence**: "For a fair comparison, all ablation experiments with LoRA and NLS tuning applied the same adapter target modules (Q, K, V, Up, and Down)."
