---
title: Weight Tying and Parameter Sharing Dramatically Reduce PEFT Parameters Without Performance Loss
confidence: HIGH
papers: [2024naacl-long.481, 2024naacl-long.144, 2024naacl-long.249]
---

# Weight Tying and Parameter Sharing Dramatically Reduce PEFT Parameters Without Performance Loss

Multiple papers demonstrate that sharing parameters across layers or modules in parameter-efficient fine-tuning (PEFT) methods can reduce trainable parameters by 96%+ while maintaining or even improving performance compared to standard per-layer PEFT approaches.

**Tied-LoRA** (2024naacl-long.481) shares the low-rank projection matrices A and B across all layers of the base model. For LLaMA2-7B with rank 8, this reduces trainable parameters from ~4.2M to ~131K (96.875% reduction). The TL6 configuration (vB_tied uA_tied) where A and B are tied and frozen while u and v are trainable and layer-specific, outperforms LoRA on translation (BLEU 41.33 vs 41.30) with only 12.5% of parameters. TL6 maintains high performance across ranks r=2 to 128, unlike other configurations that degrade at higher ranks.

**Emergent Mixture-of-Experts (EMoE)** (2024naacl-long.144) shares expert parameters by constructing them from existing FFN neurons via constrained k-means clustering, without adding extra parameters. The avg-k gating function is constructed by averaging each expert's key vectors, avoiding extra trainable parameters. EMoE achieves +0.74 to +0.84 ID-Avg improvement over vanilla LoRA on BERT-Large and GPT2-XL. After fine-tuning, experts can be merged back into original FFN (EMoE2LoRA) with nearly identical performance.

**Mixture of Word Experts (MoWE)** (2024naacl-long.249) shares expert parameters across all MoWE layers (e.g., 4 layers), keeping the total number of sparse parameters manageable. Empirical results indicate that sharing parameters across MoWE layers leads to better performance. MoWE-Base outperforms T5.1.1-Base by 15.2 EM on TriviaQA (62.8% improvement). Freezing MoWE experts during finetuning prevents overfitting (37.7 EM vs 33.5 when unfrozen).

## Papers & Evidence
- `2024naacl-long.481` (Tied-LoRA): "By merely introducing weight tying across the 32 layers of this model reduces the trainable parameters to ∼131K, which is a 96.875% reduction." — TL6 configuration outperforms LoRA with 12.5% of parameters.
- `2024naacl-long.144` (EMoE): "Cluster top-k exhibits a significant improvement over the standard, random top-k is conversely worse than vanilla fine-tuning." — Parameter-free expert construction via clustering.
- `2024naacl-long.249` (MoWE): "Parameters are shared across all MoWE layers... empirical results indicated that sharing parameters across the MoWE layers leads to better performance." — Shared experts across layers with fixed routing.

## Actionable Guidance
Use weight tying across layers for LoRA adapters (Tied-LoRA TL6 configuration) to reduce parameters by ~96% while maintaining performance. For even greater efficiency, consider constructing experts from existing parameters via clustering (EMoE) or using fixed-routing word-specific experts (MoWE) for knowledge-intensive tasks. Freeze shared parameters during fine-tuning to prevent overfitting.

**Condition**: When fine-tuning large models (7B+) with limited parameter budgets, especially for deployment on memory-constrained devices.
**Confidence**: HIGH — Three papers with different parameter-sharing strategies all show dramatic parameter reduction without performance loss.
