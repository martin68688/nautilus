---
title: "Leveraging Generative Large Language Models with Visual Instruction and Demonstration Retrieval for Multimodal Sarcasm Detection"
source: "https://aclanthology.org/2024.naacl-long.97/"
categories: ['llm-ranking-benchmarking-adaptation', 'sentiment-emotion-analysis-multimodal-llms']
tags: ['multimodal', 'sarcasm-detection', 'instruction-retrieval']
venue: "NAACL 2024"
tldr: "Uses LLMs with visual instructions and retrieved examples for multimodal sarcasm detection."
---

# Leveraging Generative Large Language Models with Visual Instruction and Demonstration Retrieval for Multimodal Sarcasm Detection

**Source**: [https://aclanthology.org/2024.naacl-long.97/](https://aclanthology.org/2024.naacl-long.97/)

**TLDR**: Uses LLMs with visual instructions and retrieved examples for multimodal sarcasm detection.

## Abstract

AbstractMultimodal sarcasm detection aims to identify sarcasm in the given image-text pairs and has wide applications in the multimodal domains. Previous works primarily design complex network structures to fuse the image-text modality features for classification. However, such complicated structures may risk overfitting on in-domain data, reducing the performance in out-of-distribution (OOD) scenarios. Additionally, existing methods typically do not fully utilize cross-modal features, limiting their performance on in-domain datasets. Therefore, to build a more reliable multimodal sarcasm detection model, we propose a generative multimodal sarcasm model consisting of a designed instruction template and a demonstration retrieval module based on the large language model. Moreover, to assess the generalization of current methods, we introduce an OOD test set, RedEval. Experimental results demonstrate that our method is effective and achieves state-of-the-art (SOTA) performance on the in-domain MMSD2.0 and OOD RedEval datasets.