---
title: "Investigating Acceleration of LLaMA Inference by Enabling Intermediate Layer Decoding via Instruction Tuning with ‘LITE’"
source: "https://aclanthology.org/2024.findings-naacl.232/"
categories: ['efficient-transformer-optimization-techniques']
tags: ['inference-acceleration', 'instruction-tuning', 'intermediate-layers', 'llama']
venue: "NAACL 2024"
tldr: "Accelerates LLaMA inference by instruction tuning with intermediate layer losses to enable early decoding."
---

# Investigating Acceleration of LLaMA Inference by Enabling Intermediate Layer Decoding via Instruction Tuning with ‘LITE’

**Source**: [https://aclanthology.org/2024.findings-naacl.232/](https://aclanthology.org/2024.findings-naacl.232/)

**TLDR**: Accelerates LLaMA inference by instruction tuning with intermediate layer losses to enable early decoding.

## Abstract

AbstractLarge Language Models (LLMs) have achieved remarkable performance across a wide variety of tasks; however, their large size makes their inference slow and computationally expensive. Focusing on this problem, we study instruction tuning LLMs with additional explicit Losses from the Intermediate layers (LITE) and show that it enables these layers to acquire ‘good’ generation ability without affecting the generation ability of the final layer. We then perform ‘dynamic confidence-based early exiting’ at token level from the intermediate layers which improves the computational efficiency of text generation without sacrificing the quality of the generation. We conduct comprehensive experiments by instruction tuning LLaMA-2 models on the Alpaca dataset and evaluate on four different instruction test sets. We show that dynamic early exiting achieves consistent and considerable inference cost improvements (37.86% for 7B and 46.35% for 13B model) while maintaining the generation quality. We further conduct a thorough analysis of the results and dissect the efficiency improvements which reveals several important findings.