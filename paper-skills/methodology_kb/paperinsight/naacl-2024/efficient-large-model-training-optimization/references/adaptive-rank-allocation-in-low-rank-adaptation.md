---
title: Adaptive Rank Allocation in Low-Rank Adaptation Outperforms Uniform Rank Assignment
confidence: HIGH
papers: [2024naacl-long.35, 2024naacl-long.13, 2024naacl-industry.34]
---

# Adaptive Rank Allocation in Low-Rank Adaptation Outperforms Uniform Rank Assignment

Multiple papers at NAACL 2024 demonstrate that allocating different numbers of LoRA ranks to different layers (adaptive rank allocation) consistently outperforms using the same rank for all layers, with gains of up to +18.48 GLUE points before fine-tuning and +3.19 after fine-tuning.

**ALoRA** (2024naacl-long.35) uses an ablation-based importance scoring method (AB-LoRA) that evaluates the performance change when a single rank is zeroed out versus when only that rank is kept, using negative cross-entropy loss as the metric. Ranks are gradually pruned and reallocated to modules with higher average importance scores. ALoRA outperforms SoRA and standard LoRA across multiple tasks (e.g., +0.9 accuracy on SST-2, +1.3 accuracy on RTE, +0.6 F1 on SQUAD) with comparable or lower memory costs (18.1 GB vs 18.8 GB for SoRA).

**Adaptive Rank Selection (ARS)** (2024naacl-long.13) uses a hypernetwork composed of GRUs and linear layers to generate binary masks over singular values via Gumbel-Sigmoid, learning different ranks per operation. ARS achieves up to +18.48 average GLUE score before fine-tuning (SVD+ARS vs SVD at p=0.48) and +3.19 after fine-tuning. The hypernetwork requires only a small subset of data (e.g., 4000 samples) and negligible computation overhead (0.1 V100 GPU hours for BERT on MNLI).

**Shears** (2024naacl-industry.34) uses Neural Low-rank Adapter Search (NLS) with a weight-sharing NAS approach that creates a super-adapter network with elastic low-rank adapters of varying ranks. At 50% sparsity, Shears achieves 45.3% average accuracy vs LoRA's 43.3% on LLaMA-7B math reasoning. A zero-cost heuristic (c = floor(n/2)) provides near-optimal performance, and hill-climbing search further improves results.

## Papers & Evidence
- `2024naacl-long.35` (ALoRA): "The results show that ALoRA outperforms the two variants, demonstrating that our AB-LoRA method can provide better guidance in allocating LoRA ranks." — Gradual pruning and reallocation of LoRA ranks based on importance scores outperforms uniform allocation.
- `2024naacl-long.13` (Adaptive Rank Selection): "Our method enables adaptive rank selections for individual operations for the model, which creates flexibility to allocate different ranks for different operations... Thus, the overall performance can be largely improved over the uniform rank selection setting." — Hypernetwork-based rank selection with Gumbel-Sigmoid masking.
- `2024naacl-industry.34` (Shears): "With 50% sparsity, Shears outperforms LoRA significantly, highlighting its efficacy in enhancing model performance under sparsity conditions." — Neural low-rank adapter search with elastic ranks.

## Actionable Guidance
Use adaptive rank allocation (via ALoRA's AB-LoRA importance scoring, ARS's hypernetwork-based masking, or Shears' NLS) instead of uniform LoRA ranks. ALoRA is best when you have a fixed total rank budget and want gradual pruning; ARS is best when you can train a hypernetwork on a small data subset; Shears is best when combining with unstructured sparsity.

**Condition**: When fine-tuning LLMs with LoRA where different layers have varying importance for the downstream task.
**Confidence**: HIGH — Three independent papers with different methods all converge on the same conclusion with quantitative evidence.
