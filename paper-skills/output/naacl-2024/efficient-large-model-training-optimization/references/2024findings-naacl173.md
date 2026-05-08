---
title: "Compensate Quantization Errors: Make Weights Hierarchical to Compensate Each Other"
source: "https://aclanthology.org/2024.findings-naacl.173/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques']
tags: ['quantization', 'model-compression', 'efficient-inference']
venue: "NAACL 2024"
tldr: "Proposes a hierarchical weight quantization method to compensate for errors and improve LLM efficiency."
---

# Compensate Quantization Errors: Make Weights Hierarchical to Compensate Each Other

**Source**: [https://aclanthology.org/2024.findings-naacl.173/](https://aclanthology.org/2024.findings-naacl.173/)

**TLDR**: Proposes a hierarchical weight quantization method to compensate for errors and improve LLM efficiency.

## Abstract

AbstractEmergent Large Language Models (LLMs) use their extraordinary performance and powerful deduction capacity to discern from traditional language models. However, the expenses of computational resources and storage for these LLMs are stunning, quantization then arises as a trending conversation. To address accuracy decay caused by quantization, two streams of works in post-training quantization methods stand out. One uses other weights to compensate existing quantization error, while the other transfers the quantization difficulty to other parts in the model. Combining both merits, we introduce Learnable Singular value Increment (LSI) as an advanced solution. LSI uses Singular Value Decomposition to extract singular values of the weights and make them learnable to help weights compensate each other conditioned on activation. Incorporating LSI with existing techniques, we achieve state-of-the-art performance in diverse quantization settings, no matter in weight-only, weight-activation or extremely low bit scenarios. By unleashing the potential of LSI, efficient finetuning on quantized model is no longer a prohibitive problem.