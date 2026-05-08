---
title: "X-LLaVA: Optimizing Bilingual Large Vision-Language Alignment"
source: "https://aclanthology.org/2024.findings-naacl.158/"
categories: ['zero-shot-multimodal-large-language-models', 'text-guided-multimodal-generation']
tags: ['multimodal-models', 'vision-language-alignment', 'bilingual']
venue: "NAACL 2024"
tldr: "Optimizes bilingual large vision-language alignment to reduce training data costs for multimodal models."
---

# X-LLaVA: Optimizing Bilingual Large Vision-Language Alignment

**Source**: [https://aclanthology.org/2024.findings-naacl.158/](https://aclanthology.org/2024.findings-naacl.158/)

**TLDR**: Optimizes bilingual large vision-language alignment to reduce training data costs for multimodal models.

## Abstract

AbstractThe impressive development of large language models (LLMs) is expanding into the realm of large multimodal models (LMMs), which incorporate multiple types of data beyond text. However, the nature of multimodal models leads to significant expenses in the creation of training data. Furthermore, constructing multilingual data for LMMs presents its own set of challenges due to language diversity and complexity. Therefore, in this study, we propose two cost-effective methods to solve this problem: (1) vocabulary expansion and pretraining of multilingual LLM for specific languages, and (2) automatic and elaborate construction of multimodal datasets using GPT4-V. Based on these methods, we constructed a 91K English-Korean-Chinese multilingual, multimodal training dataset. Additionally, we developed a bilingual multimodal model that exhibits excellent performance in both Korean and English, surpassing existing approaches.