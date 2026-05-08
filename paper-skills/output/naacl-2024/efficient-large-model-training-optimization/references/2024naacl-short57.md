---
title: "Efficient Sample-Specific Encoder Perturbations"
source: "https://aclanthology.org/2024.naacl-short.57/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques']
tags: ['parameter-efficient', 'alignment', 'detoxification']
venue: "NAACL 2024"
tldr: "Proposes a lightweight method to control language model behavior by perturbing encoder parameters for specific attributes."
---

# Efficient Sample-Specific Encoder Perturbations

**Source**: [https://aclanthology.org/2024.naacl-short.57/](https://aclanthology.org/2024.naacl-short.57/)

**TLDR**: Proposes a lightweight method to control language model behavior by perturbing encoder parameters for specific attributes.

## Abstract

AbstractEncoder-decoder foundation models have displayed state-of-the-art performance on a range of autoregressive sequence tasks. This paper proposes a simple and lightweight modification to such systems to control the behaviour according to a specific attribute of interest. This paper proposes a novel inference-efficient approach to modifying the behaviour of an encoder-decoder system according to a specific attribute of interest. Specifically, we show that a small proxy network can be used to find a sample-by-sample perturbation of the encoder output of a frozen foundation model to trigger the decoder to generate improved decodings. This work explores a specific realization of this framework focused on improving the COMET performance of Flan-T5 on Machine Translation and the WER of Whisper foundation models on Speech Recognition. Results display consistent improvements in performance evaluated through COMET and WER respectively. Furthermore, experiments also show that the proxies are robust to the exact nature of the data used to train them and can extend to other domains.