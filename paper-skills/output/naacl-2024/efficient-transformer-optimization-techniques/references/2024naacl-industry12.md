---
title: "Conformer-Based Speech Recognition On Extreme Edge-Computing Devices"
source: "https://aclanthology.org/2024.naacl-industry.12/"
categories: ['efficient-transformer-optimization-techniques', 'human-llm-opinion-dynamics-moderation', 'speech-language-processing-multilingual']
tags: ['speech-recognition', 'edge-computing', 'conformer', 'on-device', 'privacy']
venue: "NAACL 2024"
tldr: "Implements efficient Conformer-based automatic speech recognition on extreme edge-computing devices to enable on-device ASR with privacy benefits."
---

# Conformer-Based Speech Recognition On Extreme Edge-Computing Devices

**Source**: [https://aclanthology.org/2024.naacl-industry.12/](https://aclanthology.org/2024.naacl-industry.12/)

**TLDR**: Implements efficient Conformer-based automatic speech recognition on extreme edge-computing devices to enable on-device ASR with privacy benefits.

## Abstract

AbstractWith increasingly more powerful compute capabilities and resources in today’s devices, traditionally compute-intensive automatic speech recognition (ASR) has been moving from the cloud to devices to better protect user privacy. However, it is still challenging to implement on-device ASR on resource-constrained devices, such as smartphones, smart wearables, and other small home automation devices. In this paper, we propose a series of model architecture adaptions, neural network graph transformations, and numerical optimizations to fit an advanced Conformer based end-to-end streaming ASR system on resource-constrained devices without accuracy degradation. We achieve over 5.26 times faster than realtime (0.19 RTF) speech recognition on small wearables while minimizing energy consumption and achieving state-of-the-art accuracy. The proposed methods are widely applicable to other transformer-based server-free AI applications. In addition, we provide a complete theory on optimal pre-normalizers that numerically stabilize layer normalization in any Lp-norm using any floating point precision.