---
title: "Gaussian Process Optimization for Adaptable Multi-Objective Text Generation using Linearly-Weighted Language Models"
source: "https://aclanthology.org/2024.findings-naacl.99/"
categories: ['efficient-llm-training-optimization', 'efficient-transformer-acceleration-techniques', 'metaphor-identification-political-framing']
tags: ['text-generation', 'multi-objective-optimization', 'gaussian-process', 'controllable-generation']
venue: "NAACL 2024"
tldr: "Uses Gaussian process optimization for adaptable multi-objective text generation with linearly-weighted language models to handle dynamic weighting schemes."
---

# Gaussian Process Optimization for Adaptable Multi-Objective Text Generation using Linearly-Weighted Language Models

**Source**: [https://aclanthology.org/2024.findings-naacl.99/](https://aclanthology.org/2024.findings-naacl.99/)

**TLDR**: Uses Gaussian process optimization for adaptable multi-objective text generation with linearly-weighted language models to handle dynamic weighting schemes.

## Abstract

AbstractIn multi-objective text generation, we aim to optimize over multiple weighted aspects (e.g., toxicity, semantic preservation, fluency) of the generated text. However, multi-objective weighting schemes may change dynamically in practice according to deployment requirements, evolving business needs, personalization requirements on edge devices, or the availability of new language models and/or objective requirements. Ideally, we need an efficient method to adapt to the dynamic requirements of the overall objective. To address these requirements, we propose a linear combination of objective-specific language models to efficiently adapt the decoding process and optimize for the desired objective without the significant computational overhead of retraining one or more language models. We show empirically that we can leverage Gaussian Process black box optimization to adapt the language model decoder weights to outperform other fixed weighting schemes and standard baselines of the task in only a few iterations of decoding. Overall this approach enables highly efficient adaptation of controllable language models via multi-objective weighting schemes that may evolve dynamically in practical deployment situations.