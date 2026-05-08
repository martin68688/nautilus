---
title: "MaCSC: Towards Multimodal-augmented Pre-trained Language Models via Conceptual Prototypes and Self-balancing Calibration"
source: "https://aclanthology.org/2024.naacl-long.446/"
categories: ['zero-shot-few-shot-multimodal-optimization', 'legal-ai-nlp-applications']
tags: ['multimodal', 'pre-trained-models', 'conceptual-prototypes']
venue: "NAACL 2024"
tldr: "Proposes MaCSC, a method to augment pre-trained language models with multimodal semantics via conceptual prototypes and self-balancing calibration."
---

# MaCSC: Towards Multimodal-augmented Pre-trained Language Models via Conceptual Prototypes and Self-balancing Calibration

**Source**: [https://aclanthology.org/2024.naacl-long.446/](https://aclanthology.org/2024.naacl-long.446/)

**TLDR**: Proposes MaCSC, a method to augment pre-trained language models with multimodal semantics via conceptual prototypes and self-balancing calibration.

## Abstract

AbstractPre-trained language models (PLMs) that rely solely on textual data may exhibit limitations in multimodal semantics comprehension. Existing solutions attempt to alleviate this issue by incorporating explicit image retrieval or generation techniques.However, these methods: (1) focus exclusively on the static image modality; (2) inevitably encounter modality gaps and noise; (3) indiscriminately treat all modalities.In this paper, we propose a novel multimodal-augmented framework termed MaCSC, which can infuse multimodal semantics into PLMs and facilitate a self-balancing calibration of information allocation.Specifically, MaCSC obtains modal-specific conceptual prototypes from contrastive pre-training models (e.g., CLIP),and aggregates the intra- and inter-modal semantics of the conceptual prototype to enhance PLMs.In addition, we utilize a novel self-balancing contrastive loss to achieve multi-scale self-balancing calibration of multimodal information during fine-tuning PLMs.Experimental results show that MaCSC consistently improves the performance of PLMs across various architectures and scales, and outperforms competitive baselines on multiple NLP tasks.