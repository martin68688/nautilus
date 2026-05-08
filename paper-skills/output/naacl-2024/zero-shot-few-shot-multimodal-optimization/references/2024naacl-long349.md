---
title: "When Does Monolingual Data Help Multilingual Translation: The Role of Domain and Model Scale"
source: "https://aclanthology.org/2024.naacl-long.349/"
categories: ['zero-shot-few-shot-multimodal-optimization', 'speech-language-processing-multilingual']
tags: ['multilingual', 'monolingual-data', 'domain-adaptation']
venue: "NAACL 2024"
tldr: "Investigates when and how monolingual data helps multilingual translation, focusing on the roles of domain and model scale."
---

# When Does Monolingual Data Help Multilingual Translation: The Role of Domain and Model Scale

**Source**: [https://aclanthology.org/2024.naacl-long.349/](https://aclanthology.org/2024.naacl-long.349/)

**TLDR**: Investigates when and how monolingual data helps multilingual translation, focusing on the roles of domain and model scale.

## Abstract

AbstractMultilingual machine translation (MMT), trained on a mixture of parallel and monolingual data, is key for improving translation in low-resource language pairs. However, the literature offers conflicting results on the performance of different methods of including monolingual data. To resolve this, we examine how denoising autoencoding (DAE) and backtranslation (BT) impact MMT under different data conditions and model scales. Unlike prior studies, we use a realistic dataset of 100 translation directions and consider many domain combinations of monolingual and test data. We find that monolingual data generally helps MMT, but models are surprisingly brittle to domain mismatches, especially at smaller model scales. BT is beneficial when the parallel, monolingual, and test data sources are similar but can be detrimental otherwise, while DAE is less effective than previously reported. Next, we analyze the impact of scale (from 90M to 1.6B parameters) and find it is important for both methods, particularly DAE. As scale increases, DAE transitions from underperforming the parallel-only baseline at 90M to converging with BT performance at 1.6B, and even surpassing it in low-resource. These results offer new insights into how to best use monolingual data in MMT.