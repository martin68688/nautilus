---
title: "Tied-LoRA: Enhancing parameter efficiency of LoRA with Weight Tying"
source: "https://aclanthology.org/2024.naacl-long.481/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques']
tags: ['parameter-efficient', 'lora', 'weight-tying', 'finetuning']
venue: "NAACL 2024"
tldr: "Enhances LoRA's parameter efficiency via weight tying and selective training of parameters."
---

# Tied-LoRA: Enhancing parameter efficiency of LoRA with Weight Tying

**Source**: [https://aclanthology.org/2024.naacl-long.481/](https://aclanthology.org/2024.naacl-long.481/)

**TLDR**: Enhances LoRA's parameter efficiency via weight tying and selective training of parameters.

## Abstract

AbstractWe introduce Tied-LoRA, a novel paradigm leveraging weight tying and selective training to enhance the parameter efficiency of Low-rank Adaptation (LoRA). Our exploration encompasses different plausible combinations of parameter training and freezing, coupled with weight tying, aimed at identifying the optimal trade-off between performance and the count of trainable parameters. Across 5 diverse tasks and two foundational language models with different parameter counts, our experiments provide comprehensive insights into the inherent trade-offs between efficiency and performance.Our findings reveal a specific Tied-LoRA configuration that distinguishes itself by showcasing comparable performance to LoRA across multiple tasks while utilizing only a fraction of the parameters employed by the standard LoRA method, particularly at elevated ranks. This underscores the efficacy of Tied-LoRA in achieving impressive results with significantly reduced model complexity.