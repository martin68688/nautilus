---
title: "LaDiC: Are Diffusion Models Really Inferior to Autoregressive Counterparts for Image-to-Text Generation?"
source: "https://aclanthology.org/2024.naacl-long.373/"
categories: ['text-guided-multimodal-generation']
tags: ['diffusion-models', 'image-captioning', 'image-to-text', 'autoregressive']
venue: "NAACL 2024"
tldr: "This paper re-evaluates diffusion models for image-to-text generation, showing they can outperform autoregressive models in image captioning."
---

# LaDiC: Are Diffusion Models Really Inferior to Autoregressive Counterparts for Image-to-Text Generation?

**Source**: [https://aclanthology.org/2024.naacl-long.373/](https://aclanthology.org/2024.naacl-long.373/)

**TLDR**: This paper re-evaluates diffusion models for image-to-text generation, showing they can outperform autoregressive models in image captioning.

## Abstract

AbstractDiffusion models have exhibited remarkable capabilities in text-to-image generation. However, their performance in image-to-text generation, specifically image captioning, has lagged behind Auto-Regressive (AR) models, casting doubt on their applicability for such tasks. In this work, we revisit diffusion models, highlighting their capacity for holistic context modeling and parallel decoding. With these benefits, diffusion models can alleviate the inherent limitations of AR methods, including their slow inference speed, error propagation, and unidirectional constraints. Furthermore, we identify the prior underperformance of diffusion models stemming from the absence of an effective latent space for image-text alignment, and the discrepancy between continuous diffusion processes and discrete textual data. In response, we introduce a novel architecture, LaDiC, which utilizes a split BERT to create a dedicated latent space for captions and integrates a regularization module to manage varying text lengths. Our framework also includes a diffuser for semantic image-to-text conversion and a Back&Refine technique to enhance token interactivity during inference. LaDiC achieves state-of-the-art performance for diffusion-based methods on the MS COCO dataset with 38.2 BLEU@4 and 126.2 CIDEr, demonstrating exceptional performance without pre-training or ancillary modules. This indicates strong competitiveness with AR models, revealing the previously untapped potential of diffusion models in image-to-text generation.