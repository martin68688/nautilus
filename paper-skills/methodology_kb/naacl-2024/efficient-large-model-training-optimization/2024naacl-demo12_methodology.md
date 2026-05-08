# LMFlow: An Extensible Toolkit for Finetuning and Inference of Large Foundation Models

**Source**: https://aclanthology.org/2024.naacl-demo.12/

## [POSITIVE] Continuous Pretraining (Domain Adaptation)
Ongoing pretraining on a large collection of unlabeled, domain-specific data to bridge the gap between pretraining and downstream domains.

**Delta**: outperforms larger general-domain models
**Condition**: When adapting a foundation model to a specialized domain (e.g., medical, law, finance).

**Evidence**: "Lehman et al. (2023) conducted an extensive empirical analysis... and found that specialized clinical models, even smaller in size, significantly outperform larger general-domain models when finetuned on domain-specific data."

## [POSITIVE] Task Tuning (Supervised Finetuning on Domain Data)
Finetuning a model on task-specific data (e.g., medical QA datasets) to improve proficiency in a specific field.

**Delta**: +23.2 average on medical datasets (LLaMA-33B LoRA vs. base LLaMA-33B: 62.9 vs. 39.3 on MMLU medical subsets); +33.3 average on medical QA (LLaMA-33B LoRA vs. base LLaMA-33B: 58.5 vs. 25.2)
**Condition**: When enhancing model performance on a specific task or domain with labeled data.

**Evidence**: "The evaluations on three medical datasets revealed significant enhancements in both in-domain (PubMedQA, MedMCQA) and out-of-domain (MedQA-USMLE) datasets."

## [POSITIVE] Instruction Tuning (Supervised Finetuning on Instruction Data)
Training the model on a small set of task-specific data in prompt-answer format to enhance its ability to follow natural language instructions.

**Delta**: Robin-65B-v2 achieves 65.2 average on Open LLM Leaderboard, ranking top; Robin-7B-v2 scores 51.7, outperforming many larger models.
**Condition**: When improving a model's general instruction-following ability for conversational or task-oriented applications.

**Evidence**: "Robin-7B-v2 scored 51.7 in the OpenLLM standard test, and Robin-13B even reached as high as 59.1, ranking sixth, surpassing many 33B models."

## [POSITIVE] RAFT (Reward rAnked FineTuning) for Alignment
A novel alignment method that uses a reward model to rank outputs of the generative model, then continues training using supervised finetuning-like techniques on the highest-ranked samples.

**Delta**: outperforms SFT and PPO in reward and perplexity; achieves better perplexity (4.031 vs. 4.156 for PPO) and tends to reply with more details.
**Condition**: When aligning a generative model to human preferences (helpfulness, harmlessness, honesty) with greater stability and lower sample complexity than PPO.

**Evidence**: "RAFT achieves a better perplexity and tends to reply with more details, as the response of RAFT is usually longer."

## [POSITIVE] LoRA (Low-Rank Adaptation) for Efficient Tuning
Freezes pretrained model weights and incorporates trainable rank decomposition matrices into each Transformer layer, significantly reducing trainable parameters.

**Delta**: Enables finetuning of 33B model with only ~16 hours on a single 8×A100 server; achieves 62.9 average on MMLU medical subsets.
**Condition**: When computational resources are limited or when finetuning very large models (e.g., 33B, 65B parameters).

**Evidence**: "The LLaMA-33B (LoRA) performance is achieved with only about 16 hours finetuning on the training split of PubMedQA and MedMCQA with a single 8×A100 server."

## [POSITIVE] FlashAttention for Acceleration
A fast and memory-efficient exact attention mechanism with IO-awareness, used for both finetuning and inference acceleration.

**Delta**: Not quantified in the paper, but listed as a key feature for acceleration.
**Condition**: When training or running inference on large models where attention computation is a bottleneck.

**Evidence**: "LMFlow supports... FlashAttention... for finetuning and inference acceleration."

## [POSITIVE] Speculative Decoding for Inference Acceleration
An inference acceleration technique that generates multiple tokens speculatively and then verifies them in parallel, reducing latency.

**Delta**: Not quantified in the paper, but listed as a supported feature for inference acceleration.
**Condition**: When performing inference on large language models where lower latency is desired.

**Evidence**: "Inference Acceleration: Speculative Decoding..."

## [POSITIVE] Gradient Checkpointing for Memory Optimization
A memory optimization technique that trades compute for memory by not storing all intermediate activations during forward pass, recomputing them during backward pass.

**Delta**: Not quantified in the paper, but listed as a key feature for memory optimization.
**Condition**: When training large models with limited GPU memory.

**Evidence**: "Finetuning Acceleration and Memory Optimization: ... Gradient Checkpointing..."

## [POSITIVE] Deepspeed Zero3 for Memory Optimization
A memory optimization technique that partitions model states (optimizer states, gradients, parameters) across multiple GPUs and CPUs, enabling training of very large models.

**Delta**: Not quantified in the paper, but listed as a key feature for memory optimization.
**Condition**: When training very large models that do not fit into a single GPU's memory.

**Evidence**: "Finetuning Acceleration and Memory Optimization: ... Deepspeed Zero3."

## [POSITIVE] Position Interpolation for Long Context Generalization
A technique to extend the context window of LLaMA models by interpolating positional encodings, allowing the model to handle longer sequences than it was originally trained on.

**Delta**: Not quantified in the paper, but listed as a key feature for long context generalization.
**Condition**: When the model needs to process or generate text longer than its original training context length.

**Evidence**: "Long Context Generalization: Position Interpolation for LLaMA (Chen et al., 2023)."

## [POSITIVE] Vocabulary Extension for Model Customization
Extending the model's vocabulary to include new tokens (e.g., domain-specific terms) to improve performance on specialized domains.

**Delta**: Not quantified in the paper, but listed as a key feature for model customization.
**Condition**: When adapting a model to a domain with specialized terminology not well-covered by the original vocabulary.

**Evidence**: "Model Customization: Vocabulary Extension."

## [POSITIVE] Multimodal Finetuning
Finetuning a model to handle multimodal inputs (e.g., text and images) for tasks like reasoning-based object detection.

**Delta**: Not quantified in the paper, but listed as a key feature and demonstrated via a video demo.
**Condition**: When building applications that require understanding and reasoning across multiple modalities (e.g., text and images).

**Evidence**: "Multimodal: Finetuning Multimodal Chatbot for reasoning-based object detection (Pi et al., 2023)."
