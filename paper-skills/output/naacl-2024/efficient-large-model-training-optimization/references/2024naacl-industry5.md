---
title: "Efficiently Distilling LLMs for Edge Applications"
source: "https://aclanthology.org/2024.naacl-industry.5/"
categories: ['efficient-large-model-training-optimization', 'llm-edge-distillation']
tags: ['distillation', 'edge', 'supernet']
venue: "NAACL 2024"
tldr: "Proposes an efficient multistage low-rank fine-tuning method for distilling LLMs for edge applications."
---

# Efficiently Distilling LLMs for Edge Applications

**Source**: [https://aclanthology.org/2024.naacl-industry.5/](https://aclanthology.org/2024.naacl-industry.5/)

**TLDR**: Proposes an efficient multistage low-rank fine-tuning method for distilling LLMs for edge applications.

## Abstract

AbstractSupernet training of LLMs is of great interest in industrial applications as it confers the ability to produce a palette of smaller models at constant cost, regardless of the number of models (of different size / latency) produced. We propose a new method called Multistage Low-rank Fine-tuning of Super-transformers (MLFS) for parameter-efficient supernet training. We show that it is possible to obtain high-quality encoder models that are suitable for commercial edge applications, and that while decoder-only models are resistant to a comparable degree of compression, decoders can be effectively sliced for a significant reduction in training time.