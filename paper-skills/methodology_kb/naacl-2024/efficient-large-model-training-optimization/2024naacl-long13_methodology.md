# Adaptive Rank Selections for Low-Rank Approximation of Language Models

**Source**: https://aclanthology.org/2024.naacl-long.13/

## [POSITIVE] Adaptive Rank Selection (ARS)
Instead of using the same number of ranks for all layers, this method learns a different number of ranks for each operation (layer) using a differentiable binary masking mechanism.

**Delta**: Outperforms uniform rank selection by up to +18.48 average GLUE score before fine-tuning (SVD+ARS vs SVD at p=0.48). After fine-tuning, +3.19 average GLUE score.
**Condition**: Applicable to all linear layers in transformer models (both encoder-only and decoder-only), including large language models like LLaMA.

**Evidence**: "Our method enables adaptive rank selections for individual operations for the model, which creates flexibility to allocate different ranks for different operations, and we can allocate more parameters for more important operations. Thus, the overall performance can be largely improved over the uniform rank selection setting."

## [POSITIVE] Binary Masking with Gumbel-Sigmoid
A binary mask is applied on top of the diagonal singular value matrix S. The mask is made differentiable using straight-through Gumbel-Sigmoid, allowing end-to-end learning of which singular values to keep.

**Delta**: Enables differentiable rank selection; without it, rank selection would be a discrete, non-differentiable optimization problem.
**Condition**: Used during the hypernetwork training phase to learn the mask for each layer.

**Evidence**: "The binary mask m is not differentiable in its plain form; therefore, we incorporate the straight-through Gumbel-Sigmoid operation to make it differentiable."

## [POSITIVE] Hypernetwork (HN) for Rank Generation
A hypernetwork composed of GRUs and linear layers generates the binary masks for all layers. The GRU captures interactions between different operations, and linear layers map outputs to individual operations.

**Delta**: Faster convergence and better solution quality compared to element-wise binary gates. Ablation shows 'w/o hypernetwork' drops performance by -1.28 on MRPC, -0.22 on STSB, -5.09 on COLA.
**Condition**: Used during the rank learning phase; requires a small subset of training data (e.g., 4000 samples for GLUE tasks).

**Evidence**: "It is clear that our method can find a better solution and achieve a much faster convergence rate with HN on MRPC, STSB, and COLA. No doubt, HN largely improves the efficiency when learning the number of ranks."

## [POSITIVE] Singular Value-Aware Masking (Top-k Selection)
Instead of element-wise mask selection, the sum of the binary mask is used to represent the number of ranks, and the top-k singular values are selected based on this sum. A regularization term R_align encourages the mask to align with sorted singular values.

**Delta**: Ablation shows 'w/o R_align' drops performance by -0.50 on MRPC, -0.49 on STSB, -1.51 on COLA. 'w/o Rank Selection' (element-wise) drops by -7.74, -1.50, -10.17 respectively.
**Condition**: Applied during mask learning and final model compression; ensures the mask respects the sorted order of singular values from SVD.

**Evidence**: "This independency can produce a mask that skips some ranks with a high singular value, resulting in a less generalizable selection of ranks. This behavior significantly deteriorated the compressed model... Our ablation study will verify the mentioned insight and prove the effectiveness of R_align."

## [POSITIVE] Weighted SVD (FWSVD / IWSVD)
Instead of vanilla SVD, the weight matrix is re-weighted using Fisher Information (FWSVD) or first-order Taylor expansion importance scores (IWSVD) before decomposition, to incorporate task-specific information.

**Delta**: FWSVD outperforms vanilla SVD by +15.13 average GLUE score before fine-tuning (57.03 vs 41.90 at p=0.48). After fine-tuning, +2.92 (81.34 vs 78.42).
**Condition**: Used as a baseline and combined with ARS; applicable when task-specific data is available for computing importance scores.

**Evidence**: "By using Fisher Information or other importance scores, the compressed model has a much better performance across different tasks since task related information is injected."

## [POSITIVE] Parameter Regularization (R)
A regularization term R(p, T_total) = log(max(p*T_total, T(m)) / T_total) is added to the objective to control the total number of parameters preserved, allowing the user to specify a target compression rate p.

**Delta**: Enables precise control over model size; without it, the model may not reach the desired compression rate.
**Condition**: Applied during hypernetwork training; p is a user-specified parameter (e.g., 0.48, 0.33, 0.24).

**Evidence**: "For a specific task, to maximally preserve the performance given a parameter budget, we minimize the task loss together with the regularization of the number of parameters..."

## [POSITIVE] Freezing Model Weights During Rank Learning
During the optimization of the hypernetwork and masks, the underlying model weights are frozen. Only the hypernetwork parameters are updated.

**Delta**: Makes the rank learning process efficient; requires only a small subset of data (e.g., 4000 samples) and negligible computation overhead (e.g., 0.1 V100 GPU hours for BERT on MNLI).
**Condition**: Used during the rank selection phase before fine-tuning the compressed model.

**Evidence**: "Note that the model weights are frozen during the optimization of Obj. 8; therefore, the learnable parameter is small and can be optimized efficiently."

## [POSITIVE] Fine-tuning After Compression
After the model is compressed using the learned ranks, it is fine-tuned on the downstream task to recover performance.

**Delta**: For FWSVD+ARS on BERT at p=0.48, fine-tuning improves average GLUE from 69.49 to 83.53 (+14.04). For LLaMA-7B, fine-tuning reduces perplexity dramatically (e.g., from 3893.10 to 20.54 on WikiText at 1.7B params).
**Condition**: Applied after compression; duration varies from short (e.g., 0.16B tokens for LLaMA) to full fine-tuning (e.g., 3 epochs on GLUE).

**Evidence**: "After fine-tuning, this gap is 3.19 between SVD and SVD+ARS... After a short fine-tuning, the perplexity of the model can be quickly recovered."

## [POSITIVE] Using a Small Subset for Hypernetwork Training
The hypernetwork is trained on only a small subset of the training data (e.g., 4000 samples for GLUE tasks, or 2000 iterations for language modeling), rather than the full dataset.

**Delta**: Negligible computational overhead (e.g., 0.1 V100 GPU hours for BERT on MNLI). Increasing samples beyond 4000-8000 gives marginal benefit.
**Condition**: Used during the rank learning phase; sufficient for learning good rank configurations due to the smaller search space.

**Evidence**: "It requires only a small subset of the original training data; therefore, the computation overhead to optimize the number of ranks is negligible... the benefit of using too many samples is marginal."

## [POSITIVE] Joint Learning of All Layer Masks
All binary masks (one per layer) are learned jointly in a single pass, allowing rank selection to compete across all layers globally.

**Delta**: Enables important operations to receive more ranks than less important ones, improving overall performance.
**Condition**: Applied during hypernetwork training; global optimization across all layers.

**Evidence**: "Note that all m_l are learned jointly in one pass, and rank selection competes across all layers. In other words, important operations can receive more ranks than the less important ones."
