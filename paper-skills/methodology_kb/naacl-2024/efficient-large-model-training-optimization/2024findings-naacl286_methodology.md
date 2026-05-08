# Personalized Federated Learning for Text Classification with Gradient-Free Prompt Tuning

**Source**: https://aclanthology.org/2024.findings-naacl.286/

## [POSITIVE] Gradient-Free Local Training with Discrete Local Search
Local training of prompt embeddings using discrete local search over natural language tokens instead of gradient-based backpropagation. For each update step, a random prompt token position is selected and replaced with the best candidate from K semantically similar natural language tokens.

**Delta**: Outperforms baselines; achieves highest accuracy on multiple datasets
**Condition**: Applicable when clients have limited memory resources (e.g., edge devices) and cannot afford backpropagation through large PLMs.

**Evidence**: "Experiments show that our gradient-free framework achieves superior performance compared with baselines."

## [POSITIVE] Uploading with Discrete Indices
After local training, the prompt is represented by discrete token indices (16-bit integers) instead of continuous embeddings, reducing upload communication cost by approximately 1000x.

**Delta**: Reduces upload cost from ~16KB per position to 16 bits per position (1000x reduction)
**Condition**: Applicable when upload bandwidth is limited or expensive; requires prompts to be composed of discrete natural language tokens.

**Evidence**: "As the result, we reduce the communication cost by 1000 times (16 Bits vs 16 KB)."

## [POSITIVE] Downloading with Embedding Compression via Linear Word Analogy
Compresses the aggregated global prompt before downloading by approximating each row's residual (change from previous round) using a sparse linear combination of a few pretrained token embeddings, inspired by linear word analogies.

**Delta**: Maintains comparable performance while substantially decreasing download communication cost (e.g., Ours Φ=5 uses 8KB vs 819KB for full download)
**Condition**: Applicable when download bandwidth is limited; works best when prompt embeddings lie near the convex hull of pretrained token embeddings.

**Evidence**: "Ours with Φ = 3 and Φ = 5 can maintain comparable performance for text classification as with Ours (FullDownload), while substantially decreasing download communication cost."

## [POSITIVE] Discrete Local Search with K=5 Candidates
During local training, for each update step, K=5 candidate tokens with highest cosine similarity to the current prompt embedding are evaluated, balancing optimization ability and computational efficiency.

**Delta**: Performance gain from K=2 to K=5 is larger than from K=5 to K=8; K=5 chosen as cost-effective trade-off
**Condition**: Applicable for resource-constrained clients; K=5 provides good balance between search quality and computational cost.

**Evidence**: "The performance gain, i.e., the difference in the optimized loss value, is diminishing when switching from K=2 to K=5 and from K=5 to K=8. ... we take K=5 as a trade-off between the computation efficiency and optimization ability."

## [POSITIVE] Post-Tuning with BBT (Black-Box Tuning)
After federated learning rounds, the compressed prompt is further fine-tuned locally on each client using the gradient-free BBT method (CMA-ES in a low-dimensional subspace) to adapt to the client's specific task/domain without forgetting global knowledge.

**Delta**: Enables personalization without additional communication cost
**Condition**: Applicable after federated training to adapt the global prompt to each client's local data distribution.

**Evidence**: "After the last round of federated learning, we follow (Fallah et al., 2020; Chen et al., 2018) that further fine-tunes p' with a post tuning process for the final p_i (no communication cost)."

## [NEUTRAL] FedAvg Aggregation with Uniform Weighting
The server aggregates uploaded prompts from clients using Federated Averaging (FedAvg) with uniform weighting to produce a global prompt.

**Delta**: Standard aggregation method; no specific quantitative improvement claimed
**Condition**: Standard federated learning aggregation; applicable when clients have equal importance.

**Evidence**: "where we adopt FedAvg (McMahan et al., 2017) and assume uniform weighting."

## [POSITIVE] Using Discrete Tokens Instead of Continuous Prompts
Constraining prompt embeddings to be discrete natural language tokens rather than continuous vectors, which reduces expressiveness but prevents overfitting to client-specific domains and reduces communication cost.

**Delta**: Outperforms continuous prompt tuning baselines (e.g., Prompt Tuning (Fed), Meta Prompt Tuning (Fed)) on multiple datasets
**Condition**: Applicable in heterogeneous federated learning where clients have different domains/tasks; discrete tokens reduce overfitting and improve generalization.

**Evidence**: "Training with continuous prompts may result in the updated p_i being overfit to the domain/task of client i, causing negative knowledge transfer ... our approach can produce better accuracy compared to training with continuous prompt embeddings."

## [POSITIVE] Compressing Residual (Increment) Instead of Full Prompt
During download compression, only the residual (change from previous round's compressed prompt) is compressed and communicated, improving training stability and efficiency.

**Delta**: Enables stable training with reduced download cost
**Condition**: Applicable in iterative federated learning rounds where prompts evolve gradually.

**Evidence**: "for the training stability of p at index t, we only compress its increment (residual) between the previous and current rounds, i.e., R[t] = p[t] - p'_{-1}[t], instead of directly compressing p[t]."

## [POSITIVE] LASSO Regularization for Sparse Projection (α=0.2)
Uses L1 regularization (LASSO) with α=0.2 to encourage sparsity when projecting the residual onto pretrained token embeddings, selecting only a few tokens (Φ) for approximation.

**Delta**: Results with α=0.2 are generally higher than with α=0 (ablation study shows importance of sparsity)
**Condition**: Applicable when compressing prompt embeddings; α=0.2 provides good balance between compression and accuracy.

**Evidence**: "We can find that the results with α = 0 is generally lower than that with α = 0.2, indicating the importance of encouraging sparsity with the lasso loss in (8)."

## [POSITIVE] Local PLM Deployment (No Data Uploading)
The pretrained PLM parameters are downloaded to each client before federated learning starts, so client data never leaves the device, preserving privacy.

**Delta**: Enables privacy-preserving federated learning without data uploading
**Condition**: Applicable when data privacy is paramount; requires clients to have sufficient storage for the PLM.

**Evidence**: "Note that we assume the pretrained parameters of PLMs have been downloaded to each client before the start of federated learning, so that we only need to communicate the prompt parameters during federated learning."
