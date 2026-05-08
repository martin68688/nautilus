---
title: "MuST: Robust Image Watermarking for Multi-Source Tracing"
source: "https://www.semanticscholar.org/paper/5ee6562ed0c2a4edd1b3ecf81eaef618d6cd87e6"
categories: ['llm-safety-adversarial-robustness', 'brain-inspired-spiking-neural-networks']
tags: ['image-watermarking', 'multi-source-tracing', 'robustness']
venue: "AAAI 2024"
tldr: "Introduces a robust image watermarking method for tracing the sources of unauthorized materials in multi-source composite images."
---

# MuST: Robust Image Watermarking for Multi-Source Tracing

**Source**: [https://www.semanticscholar.org/paper/5ee6562ed0c2a4edd1b3ecf81eaef618d6cd87e6](https://www.semanticscholar.org/paper/5ee6562ed0c2a4edd1b3ecf81eaef618d6cd87e6)

**TLDR**: Introduces a robust image watermarking method for tracing the sources of unauthorized materials in multi-source composite images.

## Abstract

In recent years, with the popularity of social media applications, massive digital images are available online, which brings great convenience to image recreation. However, the use of unauthorized image materials in multi-source composite images is still inadequately regulated, which may cause significant loss and discouragement to the copyright owners of the source image materials. Ideally, deep watermarking techniques could provide a solution for protecting these copyrights based on their encoder-noise-decoder training strategy. Yet existing image watermarking schemes, which are mostly designed for single images, cannot well address the copyright protection requirements in this scenario, since the multi-source image composing process commonly includes distortions that are not well investigated in previous methods, e.g., the extreme downsizing.
To meet such demands, we propose MuST, a multi-source tracing robust watermarking scheme, whose architecture includes a multi-source image detector and minimum external rectangle operation for multiple watermark resynchronization and extraction. Furthermore, we constructed an image material dataset covering common image categories and designed the simulation model of the multi-source image composing process as the noise layer. Experiments demonstrate the excellent performance of MuST in tracing sources of image materials from the composite images compared with SOTA watermarking methods, which could maintain the extraction accuracy above 98% to trace the sources of at least 3 different image materials while keeping the average PSNR of watermarked image materials higher than 42.51 dB. We released our code on https://github.com/MrCrims/MuST