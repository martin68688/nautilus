# Unlocking Emergent Modularity in Large Language Models

**Source**: https://aclanthology.org/2024.naacl-long.144/

## [POSITIVE] Emergent Mixture-of-Experts (EMoE)
Transforming standard pre-trained FFN layers into sparse MoE layers by clustering key vectors to form experts and using average key vectors as gating, without adding extra parameters.

**Delta**: +0.74 to +0.84 ID-Avg, OOD rank improved from 4.86 to 4.37 (BERT-Large) and 5.61 to 3.88 (GPT2-XL)
**Condition**: Applied to FFN layers of pre-trained LMs (BERT, GPT2, Llama) during fine-tuning; works with both LoRA and full fine-tuning.

**Evidence**: "Our experiments demonstrate that fine-tuning EMoE effectively improves downstream in-domain and out-of-domain generalization compared with vanilla fine-tuning."

## [POSITIVE] Clustering-based Expert Construction
Partitioning FFN neurons into experts by clustering their key vectors using constrained k-means, ensuring co-activated neurons are grouped together.

**Delta**: +0.92 vs vanilla LoRA (Top-k with cluster); random construction yields -0.11
**Condition**: Used to construct experts from pre-trained FFN layers; requires no additional training data or parameters.

**Evidence**: "Cluster top-k exhibits a significant improvement over the standard, random top-k is conversely worse than vanilla fine-tuning."

## [POSITIVE] Avg-k Gating
Gating function constructed by averaging each expert's key vectors, avoiding extra trainable parameters. Gating scores are the average of activation scores of neurons in that expert.

**Delta**: Outperforms learned gating (EMoE-learn) on average; more stable expert selection during training
**Condition**: Used in EMoE to route inputs to experts; gating weights tied with FFN parameters via Eq. 4.

**Evidence**: "avg-k gating is more stable than learned gating. The expert selection merely changes during fine-tuning."

## [POSITIVE] Top-k Expert Selection
Selecting the k experts with the highest gating scores, activating only a subset of experts per input.

**Delta**: +0.92 vs vanilla LoRA; Bottom-k yields -0.48, Not-top-k yields -0.20
**Condition**: Applied during EMoE fine-tuning; k/N ratio typically 0.25 or 0.5.

**Evidence**: "LoRA tuning results with Bottom-k and Not-top-k expert selections are worse than vanilla LoRA tuning, while Top-K outperforms it."

## [POSITIVE] EMoE2LoRA (Merge after fine-tuning)
Fine-tuning with EMoE architecture, then merging experts back into original FFN for inference, so deployment uses standard architecture.

**Delta**: Performance nearly identical to EMoE, significantly better than vanilla LoRA
**Condition**: Applicable after fine-tuning; allows standard model deployment while benefiting from EM during training.

**Evidence**: "When merging EMoE back into the original FFNs after fine-tuning, the performance remains significantly better than the vanilla LoRA tuning and is almost identical to EMoE."

## [POSITIVE] Masking Neurons with Negative Transfer
EMoE blocks certain activated neurons during training via Top-k expert selection, preventing negative knowledge transfer from those neurons.

**Delta**: Multi-task ID improvement up to +7.56, OOD improvement +1.58
**Condition**: Observed in multi-task learning settings where negative transfer is more pronounced.

**Evidence**: "We hypothesize that the effects of EMoE stem from preventing negative knowledge transfer from blocked neurons."

## [POSITIVE] Sparse Activation via Expert Selection
Only a subset of experts (and thus neurons) are activated per input, reducing interference from irrelevant neurons.

**Delta**: Consistent improvement over dense baseline across various data fractions (e.g., outperforms with <20% data)
**Condition**: Applied during fine-tuning; activation ratio ~0.43 for Top-k (k/N=0.25-0.5).

**Evidence**: "EMoE consistently outperforms the standard across different data factions. EMoE achieves superior results even when using less than 20% of the data."

## [POSITIVE] Constrained Clustering for Balanced Experts
Using balanced k-means clustering to ensure each expert has roughly equal number of neurons, improving load balancing.

**Delta**: Better load balancing than GMoE (which replicates FFNs)
**Condition**: Used during expert construction; ensures no expert is too large or too small.

**Evidence**: "EMoE, with its differently initialized experts parameters, exhibits better load balancing than GMoE (expert 11 in Fig. 5 c is way more frequently selected than other experts)."

## [POSITIVE] Limited Number of EMoE Layers
Only converting a few FFN layers (e.g., last two even layers) into EMoE, rather than all layers.

**Delta**: Performance matches standard when using latter half layers; full conversion slightly underperforms
**Condition**: Applied to a subset of layers (e.g., last 2-4 even layers); full conversion not recommended.

**Evidence**: "If excessive EMoE layers are introduced, performance deteriorates. Taking GPT2-XL (48 layers) as an example, when introducing EMoE every two layers in the latter half, the performance averaged across 5 GLUE tasks (79.36) matches that of the standard model (78.87)."

## [POSITIVE] LoRA Fine-tuning with EMoE
Using Low-Rank Adaptation (LoRA) to fine-tune while EMoE structure is applied to FFN layers (which are frozen).

**Delta**: BERT-Large ID-Avg +0.74, GPT2-XL ID-Avg +0.84; OOD rank improved
**Condition**: Applied when FFN parameters are frozen; LoRA weights added to attention layers.

**Evidence**: "We mainly present the experimental results when employing LoRA to fine-tune the pre-trained language models... EMoE demonstrates enhancements compared to vanilla LoRA-tuning."

## [POSITIVE] Full Fine-tuning with EMoE
Fine-tuning all parameters of the EMoE model (including experts and gating) on downstream tasks.

**Delta**: Llama2-7B MMLU +1.58 (from 46.5 to 48.08)
**Condition**: Applicable to models of various sizes; consistent improvements across tasks.

**Evidence**: "When full-finetuned with Alpaca, EMoE exhibited a notable improvement of 1.58 on MMLU to the standard tuning baseline."

## [NEUTRAL] EMoE-learn (Learned Gating)
Ablation method where gating function is learned (same as GMoE) during fine-tuning, rather than using avg-k gating.

**Delta**: Comparable to EMoE on some tasks but less stable overall; BERT-Large ID-Avg +0.02 vs +0.74 for EMoE
**Condition**: Used as ablation to compare gating methods; requires training gating parameters.

**Evidence**: "While EMoE-learn's results are better than EMoE in several tasks (STSB, QNLI), EMoE exhibits higher stability than EMoE-learn and delivers superior overall results."

## [NEUTRAL] GMoE (Replicated FFN Experts)
Baseline method that replicates entire FFN layers to form MoE experts and trains new gating layers, introducing extra parameters.

**Delta**: BERT-Large ID-Avg +0.18, GPT2-XL +0.73; OOD rank 4.04 vs EMoE 4.37 (BERT-Large)
**Condition**: Used as baseline; requires additional parameters and training; less parameter-efficient than EMoE.

**Evidence**: "EMoE also achieves results comparable to GMoE on BERT-large and outperforms GPT2-XL with much fewer learnable parameters."

## [NEGATIVE] LoRA2EMoE (Inference-only EMoE)
Fine-tuning with standard model, then applying EMoE splitting only during inference.

**Delta**: No improvement over vanilla LoRA; similar or worse performance
**Condition**: Used as ablation to isolate training vs inference effects; not recommended.

**Evidence**: "Doing sparse activation during testing does not contribute to better generalization on average (LoRA2EMoE)."

## [NEGATIVE] Random Expert Construction
Randomly assigning neurons to experts instead of clustering by key vectors.

**Delta**: -0.11 vs vanilla LoRA (Top-k); -0.34 (Bottom-k)
**Condition**: Used as ablation to demonstrate importance of clustering-based construction.

**Evidence**: "Random top-k is conversely worse than vanilla fine-tuning. This suggests that random construction can negatively impact gating."

## [NEGATIVE] Bottom-k Expert Selection
Selecting the k experts with the lowest gating scores (worst matching).

**Delta**: -0.48 vs vanilla LoRA (cluster); -0.34 (random)
**Condition**: Used as ablation to show negative transfer from poorly matched experts.

**Evidence**: "LoRA tuning results with Bottom-k and Not-top-k expert selections are worse than vanilla LoRA tuning."

## [NEGATIVE] Not-top-k Expert Selection
Selecting all experts except the top-k (i.e., N-k experts with lower scores).

**Delta**: -0.20 vs vanilla LoRA
**Condition**: Used as ablation; activates more neurons but performs worse.

**Evidence**: "Not-top-k significantly lags behind Top-k, even though it involves and activates more neurons, suggesting the performance drop is more related to the neurons' property."
