---
title: "LaMAR: Laplacian Pyramid for Multimodal Adaptive Super Resolution (Student Abstract)"
source: "https://www.semanticscholar.org/paper/1c6194ecfbebbe175a6a3ff942ef43be77eea2c9"
categories: ['audio-multimodal-generation-and-modeling', 'multimodal-time-series-foundation-models']
tags: ['super-resolution', 'multimodal', 'laplacian-pyramid', 'image-translation']
venue: "AAAI 2024"
tldr: "Uses a Laplacian pyramid for multimodal adaptive super-resolution, enhancing low-res non-visual imagery with RGB guidance."
---

# LaMAR: Laplacian Pyramid for Multimodal Adaptive Super Resolution (Student Abstract)

**Source**: [https://www.semanticscholar.org/paper/1c6194ecfbebbe175a6a3ff942ef43be77eea2c9](https://www.semanticscholar.org/paper/1c6194ecfbebbe175a6a3ff942ef43be77eea2c9)

**TLDR**: Uses a Laplacian pyramid for multimodal adaptive super-resolution, enhancing low-res non-visual imagery with RGB guidance.

## Abstract

Recent advances in image-to-image translation involve the integration of non-visual imagery in deep models. Non-visual sensors, although more costly, often produce low-resolution images. To combat this, methods using RGB images to enhance the resolution of these modalities have been introduced. Fusing these modalities to achieve high-resolution results demands models with millions of parameters and extended inference times. We present LaMAR, a lightweight model. It employs Laplacian image pyramids combined with a low-resolution thermal image for Guided Thermal Super Resolution. By decomposing the RGB image into a Laplacian pyramid, LaMAR preserves image details and avoids high-resolution feature map computations, ensuring efficiency. With faster inference times and fewer parameters, our model demonstrates state-of-the-art results.