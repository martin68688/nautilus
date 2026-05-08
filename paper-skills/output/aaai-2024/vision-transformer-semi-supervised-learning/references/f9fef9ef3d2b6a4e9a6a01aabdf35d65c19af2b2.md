---
title: "IINet: Implicit Intra-inter Information Fusion for Real-Time Stereo Matching"
source: "https://www.semanticscholar.org/paper/f9fef9ef3d2b6a4e9a6a01aabdf35d65c19af2b2"
categories: ['vision-transformer-semi-supervised-learning', 'fair-division-matching-mechanism-design']
tags: ['stereo-matching', '3d-cnn', 'efficiency', 'real-time']
venue: "AAAI 2024"
tldr: "An efficient stereo matching network fuses implicit intra- and inter-information to balance accuracy and speed."
---

# IINet: Implicit Intra-inter Information Fusion for Real-Time Stereo Matching

**Source**: [https://www.semanticscholar.org/paper/f9fef9ef3d2b6a4e9a6a01aabdf35d65c19af2b2](https://www.semanticscholar.org/paper/f9fef9ef3d2b6a4e9a6a01aabdf35d65c19af2b2)

**TLDR**: An efficient stereo matching network fuses implicit intra- and inter-information to balance accuracy and speed.

## Abstract

Recently, there has been a growing interest in 3D CNN-based stereo matching methods due to their remarkable accuracy. However, the high complexity of 3D convolution makes it challenging to strike a balance between accuracy and speed. Notably, explicit 3D volumes contain considerable redundancy. In this study, we delve into more compact 2D implicit network to eliminate redundancy and boost real-time performance. However, simply replacing explicit 3D networks with 2D implicit networks causes issues that can lead to performance degradation, including the loss of structural information, the quality decline of inter-image information, as well as the inaccurate regression caused by low-level features. To address these issues, we first integrate intra-image information to fuse with inter-image information, facilitating propagation guided by structural cues. Subsequently, we introduce the Fast Multi-scale Score Volume (FMSV) and Confidence Based Filtering (CBF) to efficiently acquire accurate multi-scale, noise-free inter-image information. Furthermore, combined with the Residual Context-aware Upsampler (RCU), our Intra-Inter Fusing network is meticulously designed to enhance information transmission on both feature-level and disparity-level, thereby enabling accurate and robust regression. Experimental results affirm the superiority of our network in terms of both speed and accuracy compared to all other fast methods.