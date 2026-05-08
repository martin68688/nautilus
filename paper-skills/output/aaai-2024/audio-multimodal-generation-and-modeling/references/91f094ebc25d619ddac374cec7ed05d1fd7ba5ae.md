---
title: "Generative Model Perception Rectification Algorithm for Trade-Off between Diversity and Quality"
source: "https://www.semanticscholar.org/paper/91f094ebc25d619ddac374cec7ed05d1fd7ba5ae"
categories: ['brain-inspired-spiking-neural-networks', 'audio-multimodal-generation-and-modeling']
tags: ['generative-models', 'diversity-quality-tradeoff', 'perception-rectification']
venue: "AAAI 2024"
tldr: "Introduces a perception rectification algorithm for generative models to achieve a trade-off between diversity and quality."
---

# Generative Model Perception Rectification Algorithm for Trade-Off between Diversity and Quality

**Source**: [https://www.semanticscholar.org/paper/91f094ebc25d619ddac374cec7ed05d1fd7ba5ae](https://www.semanticscholar.org/paper/91f094ebc25d619ddac374cec7ed05d1fd7ba5ae)

**TLDR**: Introduces a perception rectification algorithm for generative models to achieve a trade-off between diversity and quality.

## Abstract

How to balance the diversity and quality of results from generative models through perception rectification poses a significant challenge. Abnormal perception in generative models is typically caused by two factors: inadequate model structure and imbalanced data distribution. In response to this issue, we propose the dynamic model perception rectification algorithm (DMPRA) for generalized generative models. The core idea is to gain a comprehensive perception of the data in the generative model by appropriately highlighting the low-density samples in the perception space, also known as the minor group samples. The entire process can be summarized as "search-evaluation-adjustment". To identify low-density regions in the data manifold within the perception space of generative models, we introduce a filtering method based on extended neighborhood sampling. Based on the informational value of samples from low-density regions, our proposed mechanism generates informative weights to assess the significance of these samples in correcting the models' perception. By using dynamic adjustment, DMPRA ensures simultaneous enhancement of diversity and quality in the presence of imbalanced data distribution. Experimental results indicate that the algorithm has effectively improved Generative Adversarial Nets (GANs), Normalizing Flows (Flows), Variational Auto-Encoders (VAEs), and Diffusion Models (Diffusion).