---
title: "SLiM: Speculative Decoding with Hypothesis Reduction"
source: "https://aclanthology.org/2024.findings-naacl.63/"
categories: ['efficient-llm-training-optimization', 'efficient-transformer-acceleration-techniques', 'zero-shot-multimodal-large-language-models']
tags: ['speculative-decoding', 'inference-optimization', 'llm-acceleration']
venue: "NAACL 2024"
tldr: "Introduces a speculative decoding method with hypothesis reduction to expedite LLM inference while considering computational cost, not just latency."
---

# SLiM: Speculative Decoding with Hypothesis Reduction

**Source**: [https://aclanthology.org/2024.findings-naacl.63/](https://aclanthology.org/2024.findings-naacl.63/)

**TLDR**: Introduces a speculative decoding method with hypothesis reduction to expedite LLM inference while considering computational cost, not just latency.

## Abstract

AbstractSpeculative decoding has emerged as a prominent alternative to autoregressive decoding for expediting inference in large language models (LLMs). However, prevailing assumptions often focus solely on latency reduction, neglecting the computational expenses. In this paper, we present Speculate Less, validate More (SLiM), a speculative decoding enhancement to reduce the speculation set while validating more effective tokens. SLiM is designed to mitigate LLMs’ computation costs associated with the token verification by introducing hypothesis reduction based on a fast posterior estimation. It consistently surpasses counterparts lacking cost reduction across a spectrum from CPU to GPU. Our evaluation with diverse conversational datasets shows that SLiM can achieve a substantial 70% reduction in FLOPs while generating more effective predictions on top of prior arts.