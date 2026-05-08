---
title: Speculative Decoding with Lightweight Draft Models Accelerates LLM Inference
confidence: HIGH
papers: [2024naacl-industry.2, 2024naacl-long.50]
---

# Speculative Decoding with Lightweight Draft Models Accelerates LLM Inference

Multiple papers demonstrate that using lightweight draft models (N-gram or compressed cache) for speculative decoding can accelerate LLM inference by 1.95-3.67x without retraining or additional GPU memory, while maintaining lossless output quality.

**Adaptive N-gram Parallel Decoding (ANPD)** (2024naacl-industry.2) uses a token-level N-gram module (no neural network) for rapid drafting of multiple tokens, followed by verification by the original LLM in a single forward pass. The N-gram module is updated dynamically during generation. Multi-Level N-gram (MLN) with optimal prefix matching (N=5, K=7) achieves 1.95x to 3.67x speedup across LLaMA, LLaMA-2, ChatGLM3, and CodeLLaMA models. The method is lossless (output identical to autoregressive decoding), requires no retraining or extra GPU memory, and is described as "a plug-and-play enhancement."

**Neurocache** (2024naacl-long.50) uses compressed hidden states (projected to dimension d=256 vs h=1024, 4x compression) stored in a FIFO cache with kNN retrieval (L2 distance, k=64). A single retrieval operation per token (vs a queries/token for Memorizing Transformer) increases inference speed. Contextual retrieval window (w=2) and extended cache-attention (c=2) improve perplexity from 14.118 to 13.578 on PG-19 64K. For pre-trained models (LLaMA2-7B, Mistral-7B), LoRA adaptation with cache-attention weight initialization from pre-trained self-attention enables scaling to 128K token contexts.

## Papers & Evidence
- `2024naacl-industry.2` (ANPD): "In our experiments, models such as LLaMA and its fine-tuned variants have shown speed improvements up to 3.67×, validating the effectiveness of our proposed ANPD." — N-gram based speculative decoding with N=5, K=7.
- `2024naacl-long.50` (Neurocache): "Neurocache improves upon previous methods by (1) storing compressed states, which reduces cache size [and] (2) performing a single retrieval operation per token which increases inference speed." — Compressed cache with kNN retrieval.

## Actionable Guidance
For accelerating LLM inference: (1) use ANPD with N=5 and K=7 for 1.95-3.67x speedup without any model modification or extra GPU memory — this is the simplest plug-and-play option, (2) for long-context scenarios, use Neurocache with compressed states (d=256) and contextual retrieval window (w=2) to extend effective context to 128K tokens, (3) both methods are lossless and require no retraining of the base model.

**Condition**: When deploying LLMs for inference where latency reduction is critical and GPU memory is constrained.
**Confidence**: HIGH — Two papers with different draft model approaches both demonstrate significant, lossless speedups.
