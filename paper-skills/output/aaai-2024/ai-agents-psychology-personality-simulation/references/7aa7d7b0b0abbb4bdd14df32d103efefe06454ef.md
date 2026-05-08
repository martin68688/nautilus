---
title: "Measuring Cross-Modal Interactions in Multimodal Models"
source: "https://www.semanticscholar.org/paper/7aa7d7b0b0abbb4bdd14df32d103efefe06454ef"
categories: ['explainable-ai-and-human-collaboration', 'ai-agents-psychology-personality-simulation']
tags: ['multimodal-ai', 'explainable-ai', 'cross-modal-interactions', 'healthcare']
venue: "AAAI 2024"
tldr: "Proposes a method to measure cross-modal interactions in multimodal AI models to improve explainability for healthcare applications."
---

# Measuring Cross-Modal Interactions in Multimodal Models

**Source**: [https://www.semanticscholar.org/paper/7aa7d7b0b0abbb4bdd14df32d103efefe06454ef](https://www.semanticscholar.org/paper/7aa7d7b0b0abbb4bdd14df32d103efefe06454ef)

**TLDR**: Proposes a method to measure cross-modal interactions in multimodal AI models to improve explainability for healthcare applications.

## Abstract

Integrating AI in healthcare can greatly improve patient care and system efficiency. However, the lack of explainability in AI systems (XAI) hinders their clinical adoption, especially in multimodal decision-making that combines various data sources. The majority of existing XAI methods focus on unimodal models, which fail to capture cross-modal interactions that are crucial for understanding the combined impact of multiple data sources. Existing methods for quantifying cross-modal interactions are limited to two modalities, rely on labelled data, and depend on model performance, which is problematic in healthcare, where XAI must handle multiple data sources and provide individualised explanations. This paper introduces InterSHAP, a cross-modal interaction score that addresses the limitations of existing approaches. InterSHAP uses the Shapley interaction index to precisely separate and quantify the contributions of the individual modalities and their interactions without approximations. By integrating an open-source implementation with the SHAP package, we enhance reproducibility and ease of use. We show that InterSHAP accurately measures the presence of cross-modal interactions, can handle multiple modalities, and provides detailed explanations at a local level for individual data points. Furthermore, we apply InterSHAP to real medical multimodal datasets, and demonstrate its practical applicability for individualised explanations.