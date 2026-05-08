# Tied-LoRA: Enhancing parameter efficiency of LoRA with Weight Tying

**Source**: https://aclanthology.org/2024.naacl-long.481/

## [POSITIVE] Weight Tying across layers
Sharing the low-rank projection matrices A and B across all layers of the base model, rather than having separate matrices per layer.

**Delta**: 96.875% reduction in trainable parameters (from ~4.2M to ~131K for LLaMA2 7B with rank 8)
**Condition**: Applied to all Tied-LoRA configurations; especially effective for deep models with many layers.

**Evidence**: "By merely introducing weight tying across the 32 layers of this model reduces the trainable parameters to ∼131K, which is a 96.875% reduction."

## [POSITIVE] Selective training of components (A, B, u, v)
Choosing which of the four low-rank components (down projection A, up projection B, scaling vectors u and v) to train or freeze, creating different configurations.

**Delta**: TL6 configuration achieves comparable performance to LoRA within 1-2% while using only 12.5% of parameters in some cases
**Condition**: Best results with TL6 (vB_tied uA_tied) where A and B are tied and frozen, u and v are trainable and layer-specific.

**Evidence**: "Our findings reveal a specific Tied-LoRA configuration that distinguishes itself by showcasing comparable performance to LoRA across multiple tasks while utilizing only a fraction of the parameters employed by the standard LoRA method, particularly at elevated ranks."

## [POSITIVE] Joint QKV projection (shared down projection)
Using a single down projection matrix A (d x r) shared across Q, K, V attention matrices, instead of separate A matrices for each.

**Delta**: Reduces parameters from 6dr (original LoRA) to 4dr
**Condition**: Applied to all Tied-LoRA configurations; reduces parameter count without performance loss.

**Evidence**: "Essentially, the down projection A is shared by Q, K, and V, leading to fewer trainable parameters (4dr) than the original LoRA (6dr)."

## [NEUTRAL] Removing scaling factor alpha (setting α = r)
Setting the scaling factor α equal to the rank r, effectively removing its scaling effect.

**Delta**: No quantitative change reported; simplifies hyperparameter tuning
**Condition**: Applied to all Tied-LoRA configurations; simplifies training without harming performance.

**Evidence**: "Unlike the original LoRA, where α is a hyper-parameter that can be manually set, we simply set α = r, effectively removing its scaling effect."

## [POSITIVE] TL6 configuration (vB_tied uA_tied)
Configuration where A and B are tied across layers and frozen, while u and v are trainable and layer-specific.

**Delta**: Outperforms LoRA on translation task (BLEU 41.33 vs 41.30) with 12.5% of parameters; within 1-2% of LoRA on other tasks
**Condition**: Best overall configuration; particularly effective for tasks leveraging base model strengths (commonsense NLI, extractive QA, summarization).

**Evidence**: "The TL6(vB_tied uA_tied) configuration demonstrates the best overall performance among all Tied-LoRA configurations for both model sizes and is not far behind LoRA. For our translation task with the LLaMA2 7B base model, TL6(vB_tied uA_tied) outperforms LoRA (vBuA) while using 12.5% of the number of parameters."

## [POSITIVE] TL5 configuration (vB_tied uA_tied with frozen u,v=1)
Configuration where A and B are tied across layers, u and v are frozen to 1, and only A and B are trainable.

**Delta**: Comparable to TL6 at typical ranks (r=8,16,32,64,128) with fewer parameters
**Condition**: Effective at lower ranks; slight performance drop at higher ranks compared to TL6.

**Evidence**: "The TL5(vB_tied uA_tied) configuration, while marginally outperformed by TL6(vB_tied uA_tied), is notable for its parameter efficiency. This method achieves comparable performance to TL6(vB_tied uA_tied) for typical ranks r=8,16,32,64,128, but with a reduced parameter count."

## [POSITIVE] Using higher ranks for complex tasks
Selecting higher low-rank dimensions (e.g., r=64 for GSM8K) for tasks requiring more capacity.

**Delta**: GSM8K: LoRA achieves 32.75 EM at r=64 vs lower scores at lower ranks
**Condition**: Task-dependent; mathematical reasoning benefits from higher ranks.

**Evidence**: "For LoRA (vBuA), a rank of 2 suffices for achieving best performance for the SQuAD task, while a higher rank of 64 is optimal for GSM8K."

## [POSITIVE] Tied-LoRA as drop-in replacement for LoRA
Using TL5 or TL6 with the same rank configuration as a pre-tuned LoRA system without re-optimization.

**Delta**: Comparable performance at same rank with significantly fewer parameters
**Condition**: Applicable when LoRA rank is already optimized for a task.

**Evidence**: "Both TL5(vB_tied uA_tied) and TL6(vB_tied uA_tied) show robust performance at the same rank where traditional LoRA (vBuA) is optimized (i.e., performed best). This implies that for systems pretuned for LoRA (vBuA), TL5(vB_tied uA_tied) and TL6(vB_tied uA_tied) can be utilized with the same rank configuration as a 'drop-in' replacement."

## [NEGATIVE] Single-layer LoRA comparison
Applying LoRA to only one transformer layer (instead of all layers) to match Tied-LoRA parameter count.

**Delta**: TL5 significantly outperforms single-layer LoRA at any layer (e.g., GSM8K: 27.07 vs best single-layer 19.56 EM)
**Condition**: Single-layer LoRA is inferior to Tied-LoRA with same parameter budget; lower layers (4 or 8) perform better than higher layers.

**Evidence**: "The TL5(vB_tied uA_tied) configuration was considerably better than any of the layer selection LoRA settings."

## [POSITIVE] Stability across ranks for TL6
TL6 maintains high performance across a wide range of ranks (r=2 to 128), unlike other Tied-LoRA configurations.

**Delta**: TL6 maintains closest performance to LoRA across all ranks; other methods decline at higher ranks
**Condition**: TL6 is the only configuration robust to rank increases; TL3 and TL4 show dramatic drops at higher ranks.

**Evidence**: "TL6(vB_tied uA_tied) maintains high performance across a broad range of ranks and is closest to LoRA (vBuA)."
