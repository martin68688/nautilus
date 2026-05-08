---
title: "Explaining Text Similarity in Transformer Models"
source: "https://aclanthology.org/2024.naacl-long.435/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques']
tags: ['transformer', 'interpretability', 'similarity', 'attention']
venue: "NAACL 2024"
tldr: "Analyzes how Transformer models compute text similarity, focusing on attention mechanisms in unsupervised tasks like retrieval."
---

# Explaining Text Similarity in Transformer Models

**Source**: [https://aclanthology.org/2024.naacl-long.435/](https://aclanthology.org/2024.naacl-long.435/)

**TLDR**: Analyzes how Transformer models compute text similarity, focusing on attention mechanisms in unsupervised tasks like retrieval.

## Abstract

AbstractAs Transformers have become state-of-the-art models for natural language processing (NLP) tasks, the need to understand and explain their predictions is increasingly apparent. Especially in unsupervised applications, such as information retrieval tasks, similarity models built on top of foundation model representations have been widely applied. However, their inner prediction mechanisms have mostly remained opaque. Recent advances in explainable AI have made it possible to mitigate these limitations by leveraging improved explanations for Transformers through layer-wise relevance propagation (LRP). Using BiLRP, an extension developed for computing second-order explanations in bilinear similarity models, we investigate which feature interactions drive similarity in NLP models. We validate the resulting explanations and demonstrate their utility in three corpus-level use cases, analyzing grammatical interactions, multilingual semantics, and biomedical text retrieval. Our findings contribute to a deeper understanding of different semantic similarity tasks and models, highlighting how novel explainable AI methods enable in-depth analyses and corpus-level insights.