---
title: "ComCLIP: Training-Free Compositional Image and Text Matching"
source: "https://aclanthology.org/2024.naacl-long.370/"
categories: ['zero-shot-few-shot-multimodal-optimization', 'text-guided-multimodal-generation']
tags: ['compositional', 'vision-language', 'zero-shot']
venue: "NAACL 2024"
tldr: "A training-free method for compositional image-text matching using CLIP."
---

# ComCLIP: Training-Free Compositional Image and Text Matching

**Source**: [https://aclanthology.org/2024.naacl-long.370/](https://aclanthology.org/2024.naacl-long.370/)

**TLDR**: A training-free method for compositional image-text matching using CLIP.

## Abstract

AbstractContrastive Language-Image Pretraining (CLIP) has demonstrated great zero-shot performance for matching images and text. However, it is still challenging to adapt vision-language pretrained models like CLIP to compositional image and text matching — a more challenging image and text matching task requiring the model’s understanding of compositional word concepts and visual components. Towards better compositional generalization in zero-shot image and text matching, in this paper, we study the problem from a causal perspective: the erroneous semantics of individual entities are essentially confounders that cause the matching failure. Therefore, we propose a novel training-free compositional CLIP model (ComCLIP). ComCLIP disentangles input images into subjects, objects, and action subimages and composes CLIP’s vision encoder and text encoder to perform evolving matching over compositional text embedding and subimage embeddings. In this way, ComCLIP can mitigate spurious correlations introduced by the pretrained CLIP models and dynamically evaluate the importance of each component. Experiments on four compositional image-text matching datasets: Winoground, VL-checklist, SVO, and ComVG, and two general image-text retrieval datasets: Flick30K, and MSCOCO demonstrate the effectiveness of our plug-and-play method, which boosts the zero-shot inference ability of CLIP, SLIP, and BLIP2 even without further training or fine-tuning. Our codes can be found at https://github.com/eric-ai-lab/ComCLIP.