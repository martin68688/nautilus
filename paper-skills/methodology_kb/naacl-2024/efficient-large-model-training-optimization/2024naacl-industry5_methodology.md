# Efficiently Distilling LLMs for Edge Applications

**Source**: https://aclanthology.org/2024.naacl-industry.5/

## [POSITIVE] Multistage Low-rank Fine-tuning of Super-transformers (MLFS)
A parameter-efficient supernet training method that uses low-rank matrices (LoRA) shared across sub-transformers in multiple stages, with stage-0 fine-tuning the teacher, stage-1 sampling different widths, and stage-2 sampling widths and depths.

**Delta**: outperforms baseline; achieves 1/4 size of teacher with 1/3 latency
**Condition**: Encoder models (BERT-base) on GLUE tasks; decoder models (Santacoder, CodeLlama) with less compression but faster training.

**Evidence**: "Encoder models produced by MLFS are at par or better than much costlier methods. MLFS provides accurate, smaller encoder models at 1/4 the size of the teacher and 1/3 its runtime latency on a single GPU."

## [POSITIVE] Gradient Scaling
A weighted combination of gradients where each sub-transformer gradient is scaled by (n1/nj)^γ, with n1 being the number of trainable parameters in the maxnet and nj in the j-th subnet, to speed up convergence of smaller subnets.

**Delta**: improves minnet convergence (lower loss)
**Condition**: Supernet training with subnets of varying sizes; used in all MLFS experiments.

**Evidence**: "Gradient scaling solves this by speeding up convergence of the smaller subnets to match that of the larger subnets or the maxnet. Fig. 2 shows that gradient scaling improves minnet convergence, indicated by lower minnet loss."

## [POSITIVE] Knowledge Distillation (KD) from Maxnet to Subnets
The largest subnet (maxnet) acts as a teacher for all other subnets, using both output logit KL-divergence loss and feature distillation loss from transformer layers, with shared projection matrices and hyperparameters.

**Delta**: outperforms full fine-tuning without distillation
**Condition**: All subnets during supernet training; distillation factor α=0.9 used in experiments.

**Evidence**: "Fig. 7 shows better convergence of validation loss on the Santacoder 0.7B for MLFS with distillation loss (α > 0). This demonstrates the benefit of MLFS distillation as compared to full MLFS fine tuning of the sliced model."

## [POSITIVE] Feature Distillation with Shared Projection Matrices
Feature vectors from transformer layers are projected to a low-dimensional space using maxnet's projection matrices, which are sliced for subnets, reducing the number of learnable parameters and hyperparameters.

**Delta**: computationally efficient; reduces hyperparameters
**Condition**: All subnets during supernet training; used in MLFS with βl=0.1.

**Evidence**: "To reduce the number of user-chosen hyper-parameters, we propose the following hyperparameter sharing: βj[l] := βgj(l), ∀j,l=1,2,... Thus, apart from setting fewer hyper-parameters, one needs to learn only maxnet's feature projection matrices, making feature distillation in a super-transformer setting computationally efficient."

## [POSITIVE] Low-Rank Adaptation (LoRA) for Supernet Training
Only low-rank matrices A and B (rank r=8) are trained while pre-trained weights are frozen, reducing trainable parameters from d^2 to 6rd, enabling parameter-efficient supernet fine-tuning.

**Delta**: 6rd parameters vs d^2; rank r=8 optimal
**Condition**: All MLFS experiments; rank r=8 used for all reported results.

**Evidence**: "The number of parameters to be learnt in the MLFS approach is 6rd. In contrast, full fine-tuning requires updating d^2 parameters at every iteration. Rank r=8 works well across the GLUE data sets."

## [POSITIVE] Slicing (Weight Projection) for Subnet Extraction
A weight projection operator Π slices the super-transformer's shared weights to obtain weights for any sub-transformer configuration, enabling weight-sharing and dynamic model size selection.

**Delta**: enables palette of models at constant cost
**Condition**: All supernet training; used to generate subnets of varying sizes.

**Evidence**: "This one-time training approach produces a palette of smaller models, helping mitigate the computational cost of fine-tuning a model for each deployment scenario."

## [POSITIVE] Multistage Training (Width then Depth)
Training proceeds in stages: stage-1 samples different widths (same depth), stage-2 samples both widths and depths, with the maxnet always sampled first in each iteration.

**Delta**: outperforms single-stage training
**Condition**: All MLFS experiments; used to progressively increase architectural diversity.

**Evidence**: "In stage-1, we sample sub-transformer models by sampling different widths... In stage-2, we sample sub-transformer models by sampling different widths as well as depths. We always sample the maxnet model from the super-transformer as the 1st sub-transformer model."

## [POSITIVE] Slicing Decoder Models from Larger Teacher
For decoder models, MLFS slices a smaller model from a larger pre-trained teacher (e.g., Santacoder 1.1B to 0.7B) and fine-tunes starting from sliced weights, significantly reducing training time compared to random initialization.

**Delta**: significantly reduces training time; achieves low validation loss much faster
**Condition**: Decoder models (Santacoder, CodeLlama); when a smaller model is needed for edge inference.

**Evidence**: "MLFS slicing of the teacher model can benefit decoder models by reducing substantially the training/fine-tuning time needed compared to a randomly-initialised model, as shown in Fig. 5 on Santacoder sliced from 1.1B to 0.7B."

## [NEGATIVE] Limitation: Decoder Compression Resistance
Decoder-only models are resistant to compression to the same degree as encoders; MLFS retains accuracy at 1/4 size for encoders but only 2/3 size for decoders.

**Delta**: decoder compression limited to 2/3 teacher size vs 1/4 for encoders
**Condition**: Decoder models (Santacoder, CodeLlama); when attempting high compression ratios.

**Evidence**: "Contrary to encoder models, the compression levels that retain sufficient performance of the teacher with decoders is less. While MLFS retains accuracy performance of encoder models at 1/4 the size of the teacher, the decoder models are reduced to at most 2/3 the teacher's size."

## [POSITIVE] Hyperparameter Sharing for Feature Distillation
Feature distillation loss weights βj[l] are shared across subnets by mapping to maxnet layers via gj(l), reducing the number of hyperparameters to set.

**Delta**: fewer hyperparameters; computationally efficient
**Condition**: All MLFS experiments; used with βl=0.1.

**Evidence**: "To reduce the number of user-chosen hyper-parameters, we propose the following hyperparameter sharing: βj[l] := βgj(l), ∀j,l=1,2,... Thus, apart from setting fewer hyper-parameters, one needs to learn only maxnet's feature projection matrices."
