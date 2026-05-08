---
title: "Conditional Backdoor Attack via JPEG Compression"
source: "https://www.semanticscholar.org/paper/d0cfd1ae33fbb52eadd05f49b4dda3c8270bbd71"
categories: ['llm-safety-adversarial-robustness']
tags: ['backdoor-attack', 'adversarial-robustness', 'jpeg-compression']
venue: "AAAI 2024"
tldr: "Proposes a conditional backdoor attack using JPEG compression as a stealthy trigger."
---

# Conditional Backdoor Attack via JPEG Compression

**Source**: [https://www.semanticscholar.org/paper/d0cfd1ae33fbb52eadd05f49b4dda3c8270bbd71](https://www.semanticscholar.org/paper/d0cfd1ae33fbb52eadd05f49b4dda3c8270bbd71)

**TLDR**: Proposes a conditional backdoor attack using JPEG compression as a stealthy trigger.

## Abstract

Deep neural network (DNN) models have been proven vulnerable to backdoor attacks. One trend of backdoor attacks is developing more invisible and dynamic triggers to make attacks stealthier. However, these invisible and dynamic triggers can be inadvertently mitigated by some widely used passive denoising operations, such as image compression, making the efforts under this trend questionable. Another trend is to exploit the full potential of backdoor attacks by proposing new triggering paradigms, such as hibernated or opportunistic backdoors. In line with these trends, our work investigates the first conditional backdoor attack, where the backdoor is activated by a specific condition rather than pre-defined triggers. Specifically, we take the JPEG compression as our condition and jointly optimize the compression operator and the target model's loss function, which can force the target model to accurately learn the JPEG compression behavior as the triggering condition. In this case, besides the conditional triggering feature, our attack is also stealthy and robust to denoising operations. Extensive experiments on the MNIST, GTSRB and CelebA verify our attack's effectiveness, stealthiness and resistance to existing backdoor defenses and denoising operations. As a new triggering paradigm, the conditional backdoor attack brings a new angle for assessing the vulnerability of DNN models, and conditioned over JPEG compression magnifies its threat due to the universal usage of JPEG.