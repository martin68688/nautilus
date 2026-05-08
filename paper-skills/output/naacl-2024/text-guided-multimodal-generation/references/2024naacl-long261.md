---
title: "Empowering Diffusion Models on the Embedding Space for Text Generation"
source: "https://aclanthology.org/2024.naacl-long.261/"
categories: ['zero-shot-few-shot-multimodal-optimization', 'text-guided-multimodal-generation']
tags: ['diffusion-models', 'text-generation', 'embedding-space']
venue: "NAACL 2024"
tldr: "Studies and addresses optimization challenges for diffusion models operating on text embeddings for generation."
---

# Empowering Diffusion Models on the Embedding Space for Text Generation

**Source**: [https://aclanthology.org/2024.naacl-long.261/](https://aclanthology.org/2024.naacl-long.261/)

**TLDR**: Studies and addresses optimization challenges for diffusion models operating on text embeddings for generation.

## Abstract

AbstractDiffusion models have achieved state-of-the-art synthesis quality on both visual and audio tasks, and recent works further adapt them to textual data by diffusing on the embedding space. In this paper, we conduct systematic studies of the optimization challenges encountered with both the embedding space and the denoising model, which have not been carefully explored. Firstly, the data distribution is learnable for embeddings, which may lead to the collapse of the embedding space and unstable training. To alleviate this problem, we propose a new objective called the anchor loss which is more efficient than previous methods. Secondly, we find the noise levels of conventional schedules are insufficient for training a desirable denoising model while introducing varying degrees of degeneration in consequence. To address this challenge, we propose a novel framework called noise rescaling. Based on the above analysis, we propose Difformer, an embedding diffusion model based on Transformer. Experiments on varieties of seminal text generation tasks show the effectiveness of the proposed methods and the superiority of Difformer over previous state-of-the-art embedding diffusion baselines.