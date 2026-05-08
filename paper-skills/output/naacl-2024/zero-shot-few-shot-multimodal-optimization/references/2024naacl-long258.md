---
title: "Extending CLIP’s Image-Text Alignment to Referring Image Segmentation"
source: "https://aclanthology.org/2024.naacl-long.258/"
categories: ['zero-shot-few-shot-multimodal-optimization', 'text-guided-multimodal-generation']
tags: ['multimodal', 'segmentation', 'vision-language']
venue: "NAACL 2024"
tldr: "Extends CLIP's alignment to referring image segmentation by leveraging its pretrained knowledge."
---

# Extending CLIP’s Image-Text Alignment to Referring Image Segmentation

**Source**: [https://aclanthology.org/2024.naacl-long.258/](https://aclanthology.org/2024.naacl-long.258/)

**TLDR**: Extends CLIP's alignment to referring image segmentation by leveraging its pretrained knowledge.

## Abstract

AbstractReferring Image Segmentation (RIS) is a cross-modal task that aims to segment an instance described by a natural language expression. Recent methods leverage large-scale pretrained unimodal models as backbones along with fusion techniques for joint reasoning across modalities. However, the inherent cross-modal nature of RIS raises questions about the effectiveness of unimodal backbones. We propose RISCLIP, a novel framework that effectively leverages the cross-modal nature of CLIP for RIS. Observing CLIP’s inherent alignment between image and text features, we capitalize on this starting point and introduce simple but strong modules that enhance unimodal feature extraction and leverage rich alignment knowledge in CLIP’s image-text shared-embedding space. RISCLIP exhibits outstanding results on all three major RIS benchmarks and also outperforms previous CLIP-based methods, demonstrating the efficacy of our strategy in extending CLIP’s image-text alignment to RIS.