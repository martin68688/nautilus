---
title: "Lossless Acceleration of Large Language Model via Adaptive N-gram Parallel Decoding"
source: "https://aclanthology.org/2024.naacl-industry.2/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques']
tags: ['efficient-inference', 'parallel-decoding', 'decoding-optimization']
venue: "NAACL 2024"
tldr: "Proposes Adaptive N-gram Parallel Decoding, a lossless method to accelerate LLM inference by leveraging n-gram predictions."
---

# Lossless Acceleration of Large Language Model via Adaptive N-gram Parallel Decoding

**Source**: [https://aclanthology.org/2024.naacl-industry.2/](https://aclanthology.org/2024.naacl-industry.2/)

**TLDR**: Proposes Adaptive N-gram Parallel Decoding, a lossless method to accelerate LLM inference by leveraging n-gram predictions.

## Abstract

AbstractWhile Large Language Models (LLMs) have shown remarkable abilities, they are hindered by significant resource consumption and considerable latency due to autoregressive processing. In this study, we introduce Adaptive N-gram Parallel Decoding (ANPD), an innovative and lossless approach that accelerates inference by allowing the simultaneous generation of multiple tokens. ANPD incorporates a two-stage approach: it begins with a rapid drafting phase that employs an N-gram module, which adapts based on the current interactive context, followed by a verification phase, during which the original LLM assesses and confirms the proposed tokens. Consequently, ANPD preserves the integrity of the LLM’s original output while enhancing processing speed. We further leverage a multi-level architecture for the N-gram module to enhance the precision of the initial draft, consequently reducing inference latency. ANPD eliminates the need for retraining or extra GPU memory, making it an efficient and plug-and-play enhancement. In our experiments, models such as LLaMA and its fine-tuned variants have shown speed improvements up to 3.67x, validating the effectiveness of our proposed ANPD.