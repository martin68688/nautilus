---
title: "Teaching Llama a New Language Through Cross-Lingual Knowledge Transfer"
source: "https://aclanthology.org/2024.findings-naacl.210/"
categories: ['efficient-large-model-training-optimization', 'speech-language-processing-multilingual']
tags: ['cross-lingual-transfer', 'low-resource-language', 'model-adaptation']
venue: "NAACL 2024"
tldr: "Explores cost-efficient adaptation of Llama 2 to a lower-resource language (Estonian) via cross-lingual instruction-tuning and monolingual pretraining."
---

# Teaching Llama a New Language Through Cross-Lingual Knowledge Transfer

**Source**: [https://aclanthology.org/2024.findings-naacl.210/](https://aclanthology.org/2024.findings-naacl.210/)

**TLDR**: Explores cost-efficient adaptation of Llama 2 to a lower-resource language (Estonian) via cross-lingual instruction-tuning and monolingual pretraining.

## Abstract

AbstractThis paper explores cost-efficient methods to adapt pretrained Large Language Models (LLMs) to new lower-resource languages, with a specific focus on Estonian. Leveraging the Llama 2 model, we investigate the impact of combining cross-lingual instruction-tuning with additional monolingual pretraining. Our results demonstrate that even a relatively small amount of additional monolingual pretraining followed by cross-lingual instruction-tuning significantly enhances results on Estonian. Furthermore, we showcase cross-lingual knowledge transfer from high-quality English instructions to Estonian, resulting in improvements in commonsense reasoning and multi-turn conversation capabilities. Our best model, named Llammas, represents the first open-source instruction-following LLM for Estonian. Additionally, we publish Alpaca-est, the first general task instruction dataset for Estonia. These contributions mark the initial progress in the direction of developing open-source LLMs for Estonian.