---
title: "KDMCSE: Knowledge Distillation Multimodal Sentence Embeddings with Adaptive Angular margin Contrastive Learning"
source: "https://aclanthology.org/2024.naacl-long.42/"
categories: ['contrastive-and-generative-representation-learning', 'llm-alignment-safety-detoxification', 'text-guided-multimodal-generation']
tags: ['multimodal-embeddings', 'contrastive-learning', 'knowledge-distillation', 'angular-margin']
venue: "NAACL 2024"
tldr: "Improves multimodal sentence embeddings via knowledge distillation and adaptive angular margin contrastive learning to handle noisy negatives."
---

# KDMCSE: Knowledge Distillation Multimodal Sentence Embeddings with Adaptive Angular margin Contrastive Learning

**Source**: [https://aclanthology.org/2024.naacl-long.42/](https://aclanthology.org/2024.naacl-long.42/)

**TLDR**: Improves multimodal sentence embeddings via knowledge distillation and adaptive angular margin contrastive learning to handle noisy negatives.

## Abstract

AbstractPrevious work on multimodal sentence embedding has proposed multimodal contrastive learning and achieved promising results. However, by taking the rest of the batch as negative samples without reviewing when forming contrastive pairs, those studies encountered many suspicious and noisy negative examples, significantly affecting the methods’ overall performance. In this work, we propose KDMCSE (Knowledge Distillation Multimodal contrastive learning of Sentence Embeddings), a novel approach that enhances the discrimination and generalizability of multimodal representation and inherits the knowledge from the teacher model to learn the difference between positive and negative instances and via that, can detect noisy and wrong negative samples effectively before they are calculated in the contrastive objective. Furthermore, to overcome the limitation of modeling the variation within negative pairs, we introduce a new contrastive objective, AdapACSE (Adaptive Angular Margin Supervised Contrastive Learning for Multimodal sentence embeddings), that enhances the discriminative representation by strengthening the margin within the angular space while capturing varying semantics within the negative. Experimental results on widely used Semantic Textual Similarity (STS) benchmarks demonstrate the effectiveness of our approach.