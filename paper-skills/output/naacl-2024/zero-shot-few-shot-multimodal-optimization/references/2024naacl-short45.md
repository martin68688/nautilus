---
title: "Lost in Space: Probing Fine-grained Spatial Understanding in Vision and Language Resamplers"
source: "https://aclanthology.org/2024.naacl-short.45/"
categories: ['zero-shot-few-shot-multimodal-optimization', 'llm-ranking-benchmarking-adaptation']
tags: ['vision-language', 'spatial-reasoning', 'multimodal-llm']
venue: "NAACL 2024"
tldr: "Probes the fine-grained spatial understanding capabilities of vision-and-language resampler modules in multimodal LLMs."
---

# Lost in Space: Probing Fine-grained Spatial Understanding in Vision and Language Resamplers

**Source**: [https://aclanthology.org/2024.naacl-short.45/](https://aclanthology.org/2024.naacl-short.45/)

**TLDR**: Probes the fine-grained spatial understanding capabilities of vision-and-language resampler modules in multimodal LLMs.

## Abstract

AbstractAn effective method for combining frozen large language models (LLM) and visual encoders involves a resampler module that creates a ‘visual prompt’ which is provided to the LLM, along with the textual prompt. While this approach has enabled impressive performance across many coarse-grained tasks like image captioning and visual question answering, more fine-grained tasks that require spatial understanding have not been thoroughly examined. In this paper, we use diagnostic classifiers to measure the extent to which the visual prompt produced by the resampler encodes spatial information. Our results show that this information is largely absent from the resampler output when kept frozen during training of the classifiers. However, when the resampler and classifier are trained jointly, we observe a significant performance boost. This shows that the compression achieved by the resamplers can in principle encode the requisite spatial information, but that more object-aware objectives are needed at the pretraining stage to facilitate this capability.