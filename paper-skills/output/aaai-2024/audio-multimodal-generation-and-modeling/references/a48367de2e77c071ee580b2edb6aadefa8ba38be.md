---
title: "JEN-1 DreamStyler: Customized Musical Concept Learning via Pivotal Parameters Tuning"
source: "https://www.semanticscholar.org/paper/a48367de2e77c071ee580b2edb6aadefa8ba38be"
categories: ['audio-multimodal-generation-and-modeling']
tags: ['audio-generation', 'music', 'large-models', 'parameter-tuning']
venue: "AAAI 2024"
tldr: "JEN-1 DreamStyler customizes musical concept learning in a large text-to-music model by tuning pivotal parameters to match user requirements."
---

# JEN-1 DreamStyler: Customized Musical Concept Learning via Pivotal Parameters Tuning

**Source**: [https://www.semanticscholar.org/paper/a48367de2e77c071ee580b2edb6aadefa8ba38be](https://www.semanticscholar.org/paper/a48367de2e77c071ee580b2edb6aadefa8ba38be)

**TLDR**: JEN-1 DreamStyler customizes musical concept learning in a large text-to-music model by tuning pivotal parameters to match user requirements.

## Abstract

Large models for text-to-music generation have achieved significant progress, facilitating the creation of high-quality and varied musical compositions from provided text prompts. However, input text prompts may not precisely capture user requirements, particularly when the objective is to generate music that embodies a specific concept derived from a designated reference collection. In this paper, we propose a novel method for customized text-to-music generation, which can capture the concept from a two-minute reference music and generate a new piece of music conforming to the concept. We achieve this by fine-tuning a pretrained text-to-music model using the reference music. However, directly fine-tuning all parameters leads to overfitting issues. To address this problem, we propose a Pivotal Parameters Tuning method that enables the model to assimilate the new concept while preserving its original generative capabilities. Additionally, we identify a potential concept conflict when introducing multiple concepts into the pretrained model. We present a concept enhancement strategy to distinguish multiple concepts, enabling the fine-tuned model to generate music incorporating either individual or multiple concepts simultaneously. We also introduce a new dataset and evaluation protocol for this task. Our proposed JEN1-DreamStyler outperforms several baselines in both qualitative and quantitative evaluations.