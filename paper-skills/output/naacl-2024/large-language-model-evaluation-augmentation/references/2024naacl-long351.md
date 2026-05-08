---
title: "Universal Prompt Optimizer for Safe Text-to-Image Generation"
source: "https://aclanthology.org/2024.naacl-long.351/"
categories: ['llm-backdoor-attacks-defense', 'large-language-model-evaluation-augmentation']
tags: ['text-to-image-safety', 'prompt-optimization', 'content-moderation']
venue: "NAACL 2024"
tldr: "Introduces a universal prompt optimizer to make text-to-image generation safer by modifying unsafe input prompts."
---

# Universal Prompt Optimizer for Safe Text-to-Image Generation

**Source**: [https://aclanthology.org/2024.naacl-long.351/](https://aclanthology.org/2024.naacl-long.351/)

**TLDR**: Introduces a universal prompt optimizer to make text-to-image generation safer by modifying unsafe input prompts.

## Abstract

AbstractText-to-Image (T2I) models have shown great performance in generating images based on textual prompts. However, these models are vulnerable to unsafe input to generate unsafe content like sexual, harassment and illegal-activity images. Existing studies based on image checker, model fine-tuning and embedding blocking are impractical in real-world applications. Hence, we propose the first universal **p**rompt **o**ptimizer for **s**afe T2**I** (**POSI**) generation in black-box scenario. We first construct a dataset consisting of toxic-clean prompt pairs by GPT-3.5 Turbo. To guide the optimizer to have the ability of converting toxic prompt to clean prompt while preserving semantic information, we design a novel reward function measuring toxicity and text alignment of generated images and train the optimizer through Proximal Policy Optimization. Experiments show that our approach can effectively reduce the likelihood of various T2I models in generating inappropriate images, with no significant impact on text alignment. It is also flexible to be combined with methods to achieve better performance. Our code is available at [https://github.com/wzongyu/POSI](https://github.com/wzongyu/POSI).