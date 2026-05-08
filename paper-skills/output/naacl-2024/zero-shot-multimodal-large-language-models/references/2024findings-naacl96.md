---
title: "MCAD: Multi-teacher Cross-modal Alignment Distillation for efficient image-text retrieval"
source: "https://aclanthology.org/2024.findings-naacl.96/"
categories: ['zero-shot-multimodal-large-language-models', 'knowledge-graph-image-text-matching']
tags: ['cross-modal', 'distillation', 'image-text-retrieval', 'efficient-inference']
venue: "NAACL 2024"
tldr: "Multi-teacher distillation for efficient cross-modal alignment in image-text retrieval."
---

# MCAD: Multi-teacher Cross-modal Alignment Distillation for efficient image-text retrieval

**Source**: [https://aclanthology.org/2024.findings-naacl.96/](https://aclanthology.org/2024.findings-naacl.96/)

**TLDR**: Multi-teacher distillation for efficient cross-modal alignment in image-text retrieval.

## Abstract

AbstractDue to the success of large-scale visual-language pretraining (VLP) models and the widespread use of image-text retrieval in industry areas, it is now critically necessary to reduce the model size and streamline their mobile-device deployment. Single- and dual-stream model structures are commonly used in image-text retrieval with the goal of closing the semantic gap between textual and visual modalities. While single-stream models use deep feature fusion to achieve more accurate cross-model alignment, dual-stream models are better at offline indexing and fast inference. We propose a Multi-teacher Cross-modality Alignment Distillation (MCAD) technique to integrate the advantages of single- and dual-stream models. By incorporating the fused single-stream features into the image and text features of the dual-stream model, we formulate new modified teacher similarity distributions and features. Then, we conduct both distribution and feature distillation to boost the capability of the student dual-stream model, achieving high retrieval performance without increasing inference complexity. Extensive experiments demonstrate the remarkable performance and high efficiency of MCAD on image-text retrieval tasks. Furthermore, we implement a lightweight CLIP model on Snapdragon/Dimensity chips with only ~100M running memory and ~8.0ms search latency, achieving the mobile-device application of VLP models.