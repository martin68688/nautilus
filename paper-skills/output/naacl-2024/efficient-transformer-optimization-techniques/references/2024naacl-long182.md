---
title: "DynaMo: Accelerating Language Model Inference with Dynamic Multi-Token Sampling"
source: "https://aclanthology.org/2024.naacl-long.182/"
categories: ['efficient-transformer-optimization-techniques', 'large-language-model-evaluation-augmentation']
tags: ['inference-optimization', 'multi-token', 'decoding']
venue: "NAACL 2024"
tldr: "Proposes DynaMo, a suite of multi-token prediction language models that use dynamic sampling to accelerate inference."
---

# DynaMo: Accelerating Language Model Inference with Dynamic Multi-Token Sampling

**Source**: [https://aclanthology.org/2024.naacl-long.182/](https://aclanthology.org/2024.naacl-long.182/)

**TLDR**: Proposes DynaMo, a suite of multi-token prediction language models that use dynamic sampling to accelerate inference.

## Abstract

AbstractTraditional language models operate autoregressively, i.e., they predict one token at a time. Rapid explosion in model sizes has resulted in high inference times. In this work, we propose DynaMo, a suite of multi-token prediction language models that reduce net inference times. Our models *dynamically* predict multiple tokens based on their confidence in the predicted joint probability distribution. We propose a lightweighttechnique to train these models, leveraging the weights of traditional autoregressive counterparts. Moreover, we propose novel ways to enhance the estimated joint probability to improve text generation quality, namely co-occurrence weighted masking and adaptive thresholding. We also propose systematic qualitative and quantitative methods to rigorously test the quality of generated text for non-autoregressive generation. One of the models in our suite, DynaMo-7.3B-T3, achieves same-quality generated text as the baseline (Pythia-6.9B) while achieving 2.57× speed-up with only 5.87% and 2.67% parameter and training time overheads, respectively.