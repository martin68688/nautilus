# PEMA: An Offsite-Tunable Plug-in External Memory Adaptation for Language Models

**Source**: https://aclanthology.org/2024.naacl-long.336/

## [POSITIVE] Plug-in External Memory Adaptation (PEMA)
A PEFT method that uses an external memory to store PLM-generated context representations mapped with target tokens, enabling fine-tuning without full model access.

**Delta**: outperforms other PEFT approaches in memory and latency efficiency
**Condition**: When fine-tuning proprietary PLMs where full weights are not accessible.

**Evidence**: "PEMA outperforms other PEFT approaches in memory and latency efficiency for training, and also excels in maintaining sentence meaning and generating appropriate language and styles."

## [POSITIVE] LoRA-like bottlenecked adapter in final layer
Uses low-rank weight matrices (A, Brct, Bpd) applied on top of the PLM's final layer to learn downstream tasks efficiently.

**Delta**: uses one-tenth of training memory compared to LoRA
**Condition**: When training with limited GPU memory and computational resources.

**Evidence**: "PEMA demonstrates the efficiency by utilizing one-tenth of the training memory consumption compared to LoRA."

## [POSITIVE] External memory construction
Builds a memory of context representations f(ci) paired with target tokens yi from training data, using iterative prompting of the PLM.

**Delta**: enables offsite training without full dataset sharing
**Condition**: When data owner cannot share full dataset with PLM owner.

**Evidence**: "This approach avoids the need for the full dataset from the data owner."

## [POSITIVE] Reconstruction training (Brct)
Trains a decoder Brct to reconstruct the original context representation f(ci) using mean-square error loss, preserving general knowledge from PLM.

**Delta**: improves performance across desired tasks
**Condition**: When training PEMA to retain general knowledge from the pre-trained language model.

**Evidence**: "incorporating the reconstruction decoder improves performance across desired tasks, demonstrating its efficacy in enhancing generation quality."

## [POSITIVE] Joint retraining with token prediction decoder (Bpd)
Trains Bpd to predict target tokens from the compressed representation Af(ci) using cross-entropy loss, while simultaneously reconstructing with Brct.

**Delta**: enhances task performance
**Condition**: When adapting to specific downstream tasks like translation or style transfer.

**Evidence**: "the token prediction decoder enhances task performance."

## [POSITIVE] Gradual Unrolling (GU) interpolation strategy
An interpolation strategy that initially emphasizes PEMA's distribution and gradually shifts to the PLM's context-based predictions as the sentence progresses.

**Delta**: consistently outperforms across λ/λmax values from 0.1 to 0.9
**Condition**: During inference for text generation tasks where context awareness is needed.

**Evidence**: "Using GU consistently outperforms the approach without GU for all λ/λmax values, ranging from 0.1 to 0.9."

## [POSITIVE] Interpolation of PEMA and PLM probabilities
Final next-token probability is computed by interpolating PEMA's output and PLM's output using a tuned parameter λ.

**Delta**: outperforms baselines in sBLEU and COMET
**Condition**: During inference for generating tokens in downstream tasks.

**Evidence**: "PEMA outperforms baselines in sBLEU and COMET."

## [POSITIVE] Offsite-tuning without full model distillation
PEMA avoids the computationally intensive distillation process required by prior offsite-tuning methods like Offsite-Tuning.

**Delta**: faster training latency and lower memory than Offsite-Tuning
**Condition**: When fine-tuning proprietary PLMs without access to full weights.

**Evidence**: "PEMA shows the fastest training latency among all the methods."

## [POSITIVE] Parametric model instead of kNN
Replaces the non-parametric kNN search in kNN-LM with a neural network-based parametric model to mitigate overfitting.

**Delta**: outperforms kNN-LM in sBLEU and COMET
**Condition**: When training data is limited and overfitting is a concern.

**Evidence**: "PEMA outperforms other PEFT approaches... and also excels in maintaining sentence meaning."

## [POSITIVE] Using only LM head and context representation from PLM
PEMA requires only the language model head (Whd) and context representations from the PLM, not the full model weights, during training.

**Delta**: training memory reduced to 478 MB vs 20,082 MB for full fine-tuning
**Condition**: When PLM owner declines to share all weights.

**Evidence**: "PEMA stands out by using only one-tenth of the training memory utilized by LoRA."

## [POSITIVE] Hyperparameter κ for balancing reconstruction and prediction losses
A tunable parameter κ adjusts the emphasis between reconstruction loss and next-token prediction loss during joint retraining.

**Delta**: improves performance when κ is above zero
**Condition**: During training to balance preservation of general knowledge and task-specific adaptation.

**Evidence**: "combining reconstruction loss with next-token prediction loss improves performance over excluding reconstruction loss (red dotted line), as indicated by better results when κ is above zero."

## [POSITIVE] Hyperparameter λmax for Gradual Unrolling
Maximum interpolation proportion for PEMA at the start of generation, which then decreases stepwise.

**Delta**: prevents performance degradation at higher λmax values
**Condition**: During inference to control the influence of PEMA vs PLM over the generated sequence.

**Evidence**: "with GU maintains better performance stability at higher λmax values while achieving noticeable performance improvement over without GU."
