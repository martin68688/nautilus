---
title: "Single Image Rolling Shutter Removal with Diffusion Models"
source: "https://www.semanticscholar.org/paper/c63e9e2e5aeb2f7ded9bab13aefa8f9b254b827b"
categories: ['vision-transformer-semi-supervised-learning', 'protein-molecule-ai-design-models']
tags: ['rolling-shutter', 'diffusion-models', 'single-image', 'correction']
venue: "AAAI 2024"
tldr: "Presents a diffusion model-based method for correcting rolling shutter artifacts from a single image."
---

# Single Image Rolling Shutter Removal with Diffusion Models

**Source**: [https://www.semanticscholar.org/paper/c63e9e2e5aeb2f7ded9bab13aefa8f9b254b827b](https://www.semanticscholar.org/paper/c63e9e2e5aeb2f7ded9bab13aefa8f9b254b827b)

**TLDR**: Presents a diffusion model-based method for correcting rolling shutter artifacts from a single image.

## Abstract

We present RS-Diffusion, the first Diffusion Models-based method for single-frame Rolling Shutter (RS) correction. RS artifacts compromise visual quality of frames due to the row-wise exposure of CMOS sensors. Most previous methods have focused on multi-frame approaches, using temporal information from consecutive frames for the motion rectification. However, few approaches address the more challenging but important single frame RS correction. In this work, we present an ``image-to-motion" framework via diffusion techniques, with a designed patch-attention module. In addition, we present the RS-Real dataset, comprised of captured RS frames alongside their corresponding Global Shutter (GS) ground-truth pairs. The GS frames are corrected from the RS ones, guided by the corresponding Inertial Measurement Unit (IMU) gyroscope data acquired during capture. Experiments show that RS-Diffusion surpasses previous single-frame RS methods, demonstrates the potential of diffusion-based approaches, and provides a valuable dataset for further research.