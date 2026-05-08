---
title: "DEED: Dynamic Early Exit on Decoder for Accelerating Encoder-Decoder Transformer Models"
source: "https://aclanthology.org/2024.findings-naacl.9/"
categories: ['efficient-transformer-optimization-techniques', 'clinical-nlp-biomedical-applications']
tags: ['inference-acceleration', 'dynamic-early-exit', 'encoder-decoder']
venue: "NAACL 2024"
tldr: "Proposes dynamic early exit on the decoder to accelerate inference for encoder-decoder transformer models."
---

# DEED: Dynamic Early Exit on Decoder for Accelerating Encoder-Decoder Transformer Models

**Source**: [https://aclanthology.org/2024.findings-naacl.9/](https://aclanthology.org/2024.findings-naacl.9/)

**TLDR**: Proposes dynamic early exit on the decoder to accelerate inference for encoder-decoder transformer models.

## Abstract

AbstractEncoder-decoder transformer models have achieved great success on various vision-language (VL) and language tasks, but they suffer from high inference latency. Typically, the decoder takes up most of the latency because of the auto-regressive decoding. To accelerate the inference, we propose an approach of performing Dynamic Early Exit on Decoder (DEED). We build a multi-exit encoder-decoder transformer model which is trained with deep supervision so that each of its decoder layers is capable of generating plausible predictions. In addition, we leverage simple yet practical techniques, including shared generation head and adaptation modules, to keep accuracy when exiting at shallow decoder layers. Based on the multi-exit model, we perform step-level dynamic early exit during inference, where the model may decide to use fewer decoder layers based on its confidence of the current layer at each individual decoding step. Considering different number of decoder layers may be used at different decoding steps, we compute deeper-layer decoder features of previous decoding steps just-in-time, which ensures the features from different decoding steps are semantically aligned. We evaluate our approach with three state-of-the-art encoder-decoder transformer models on various VL and language tasks. We show our approach can reduce overall inference latency by 20%-74% with comparable or even higher accuracy compared to baselines.