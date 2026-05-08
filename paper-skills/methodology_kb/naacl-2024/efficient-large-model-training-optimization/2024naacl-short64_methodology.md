# MEMORY-VQ: Compression for Tractable Internet-Scale Memory

**Source**: https://aclanthology.org/2024.naacl-short.64/

## [POSITIVE] VQ-VAE Compression
Using a vector quantization variational autoencoder (VQ-VAE) to compress token representations into discrete codes, reducing storage from 8KB per token to 0.5KB per token.

**Delta**: 16x compression rate, with minimal performance loss (72.66 EM for LUMEN vs 72.42 EM for LUMEN-VQ)
**Condition**: Applied to LUMEN model with 4096-dimensional memories, using g=256 subspaces and C=65536 codes per subspace.

**Evidence**: "LUMEN-VQ achieves a 16x compression rate, meaning we only need 2TB to store memories for all of Wikipedia and 500TB for a 1 trillion token corpus. Moreover, LUMEN-VQ suffers minimal loss in performance on the KILT benchmark."

## [POSITIVE] Product Quantization with VQ-VAE
Splitting each high-dimensional vector into subspaces and independently quantizing each subspace using VQ-VAE, enabling efficient compression and decompression.

**Delta**: Achieves 16x compression with ~0.2 point drop in KILT average exact match
**Condition**: Used for all LUMEN-VQ experiments; number of subspaces g varied to explore quality-compression trade-off.

**Evidence**: "To perform CompressVQ and DecompressVQ we apply product quantization, splitting each vector into subspaces and independently quantizing each subspace using VQ-VAE."

## [POSITIVE] Joint Training of Compression Layer with Model
Training the VQ-VAE compression layer jointly with the rest of the model using straight-through estimator for gradient backpropagation, rather than using a separate compression pipeline.

**Delta**: Outperforms freezing the whole model (71.79 EM) by a significant margin (72.42 EM)
**Condition**: Applied during fine-tuning of LUMEN-VQ; avoids commitment loss which led to model divergence.

**Evidence**: "Freezing the memory encoder during joint training... has little impact on performance. However, updating only VQ-VAE codes while freezing the entire model leads to decreased performance, indicating the model's need to adapt to decompression layer errors."

## [POSITIVE] K-means++ Codebook Initialization
Initializing the VQ-VAE codebooks using a procedure similar to k-means++ initialization to improve codebook quality.

**Delta**: Enables effective training; no quantitative comparison provided but described as part of successful setup
**Condition**: Used during initialization of VQ-VAE codebooks before training.

**Evidence**: "To initialize the codebooks, we use a procedure similar to kmeans++ initialization."

## [POSITIVE] Codebook Reset for Infrequently Used Codes
Re-initializing infrequently used codes using the same k-means++ procedure to prevent codebook collapse and maintain codebook utilization.

**Delta**: Prevents model divergence; no explicit delta but described as part of successful training
**Condition**: Applied during training when codes are rarely used.

**Evidence**: "Additionally, we perform codebook reset using the same procedure to re-initialize infrequently used codes."

## [POSITIVE] Exponential Moving Average (EMA) for Code Updates
Using EMA of memory vectors assigned to each code in each batch to update codebook vectors, with EMA factor of 0.999.

**Delta**: Part of successful VQ-VAE training; no explicit delta
**Condition**: Applied during VQ-VAE training with EMA factor 0.999.

**Evidence**: "Codes are an exponential moving average of memory vectors assigned to the code in each batch."

## [POSITIVE] Omitting Commitment Loss
Avoiding the commitment loss typically used in VQ-VAE training because it led to model divergence.

**Delta**: Prevents model divergence; enables successful training
**Condition**: Applied during LUMEN-VQ training; specific to this model setup.

**Evidence**: "We avoid using the commitment loss in our experiments as it led to model divergence."

## [NEUTRAL] Initializing from Fine-tuned LUMEN vs Training from Scratch
Comparing initialization of VQ-VAE training from a fine-tuned LUMEN model versus training from scratch.

**Delta**: Similar performance: 72.43 EM (from scratch) vs 72.42 EM (from fine-tuned LUMEN)
**Condition**: Both approaches yield comparable results; choice depends on practical considerations.

**Evidence**: "The results in Table 2 show that fine-tuning LUMEN-VQ from scratch achieves similar performance to initializing from a fine-tuned LUMEN model."

## [NEUTRAL] Freezing Memory Encoder During Joint Training
Keeping the memory encoder frozen while training the VQ-VAE and live encoder jointly.

**Delta**: 72.33 EM vs 72.42 EM (full fine-tuning), minimal drop
**Condition**: Applicable when computational budget is limited; memory encoder can be kept frozen with minimal loss.

**Evidence**: "Freezing the memory encoder during joint training... has little impact on performance."

## [NEGATIVE] Smaller Codebook Size (C=4096 vs C=65536)
Using a smaller codebook with 4096 codes per subspace instead of 65536.

**Delta**: Worse quality-compression trade-off at higher compression rates
**Condition**: Only detrimental at higher compression rates; similar performance at lower compression rates.

**Evidence**: "Using a smaller codebook has similar quality-compression trade-offs for lower compression rates but leads to worse trade-offs when we increase the compression rate."

## [NEGATIVE] Scale Down Baseline (LUMEN-XL, LUMEN-Large)
Reducing model dimension d from 4096 to 2048 or 1024 to reduce storage, used as a naive compression baseline.

**Delta**: Significant performance losses as compression rate increases compared to LUMEN-VQ
**Condition**: Used as a baseline comparison; outperformed by LUMEN-VQ at equivalent compression rates.

**Evidence**: "Both baselines exhibit significant performance losses as compression rates increase. In contrast, the LUMEN-VQ measure shows a gradual decline in performance."

## [NEGATIVE] LUMEN-Light Baseline (Truncation)
Retaining memories of only the first K tokens of each passage (inspired by FiD-Light), achieving compression rates of 2 and 4.

**Delta**: Significant performance losses compared to LUMEN-VQ at similar compression rates
**Condition**: Used as a baseline; outperformed by LUMEN-VQ.

**Evidence**: "Both baselines exhibit significant performance losses as compression rates increase. In contrast, the LUMEN-VQ measure shows a gradual decline in performance."

## [NEUTRAL] Additional Compression of Code IDs (gzip/zstd)
Attempting to further compress the integer code IDs using standard compression tools like gzip or zstd.

**Delta**: Modest compression rate of 1.1 for LUMEN-VQ codes; 1.5 for token IDs
**Condition**: Not effective for LUMEN-VQ codes; most subspaces were incompressible.

**Evidence**: "Applying standard compression tools like gzip or zstd to Wikipedia token IDs resulted in a compression factor of around 1.5. However, using the same tools on LUMEN-VQ codes of Wikipedia passages yielded a more modest compression rate of 1.1."
