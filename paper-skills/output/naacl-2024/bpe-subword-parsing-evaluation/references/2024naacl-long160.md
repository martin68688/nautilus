---
title: "Mitigating Language-Level Performance Disparity in mPLMs via Teacher Language Selection and Cross-lingual Self-Distillation"
source: "https://aclanthology.org/2024.naacl-long.160/"
categories: ['bpe-subword-parsing-evaluation', 'zero-shot-few-shot-multimodal-optimization']
tags: ['multilingual', 'distillation', 'performance-disparity', 'teacher-selection']
venue: "NAACL 2024"
tldr: "Mitigates language-level performance gaps in mPLMs via teacher language selection and cross-lingual self-distillation."
---

# Mitigating Language-Level Performance Disparity in mPLMs via Teacher Language Selection and Cross-lingual Self-Distillation

**Source**: [https://aclanthology.org/2024.naacl-long.160/](https://aclanthology.org/2024.naacl-long.160/)

**TLDR**: Mitigates language-level performance gaps in mPLMs via teacher language selection and cross-lingual self-distillation.

## Abstract

AbstractLarge-scale multilingual Pretrained Language Models (mPLMs) yield impressive performance on cross-language tasks, yet significant performance disparities exist across different languages within the same mPLM. Previous studies endeavored to narrow these disparities by supervise fine-tuning the mPLMs with multilingual data.However, obtaining labeled multilingual data is time-consuming, and fine-tuning mPLM with limited labeled multilingual data merely encapsulates the knowledge specific to the labeled data.Therefore, we introduce **ALSACE** to leverage the learned knowledge from the well-performing languages to guide under-performing ones within the same mPLM, eliminating the need for additional labeled multilingual data. Experiments show that ALSACE effectively mitigates language-level performance disparity across various mPLMs while showing the competitive performance on different multilingual NLU tasks, ranging from full resource to limited resource settings. The code for our approach is available at https://github.com/pkunlp-icler/ALSACE.