---
title: "Towards Translating Objective Product Attributes Into Customer Language"
source: "https://aclanthology.org/2024.naacl-industry.20/"
categories: ['llm-knowledge-reasoning-retrieval', 'clinical-nlp-biomedical-applications']
tags: ['information-retrieval', 'product-attributes', 'translation', 'customer-language']
venue: "NAACL 2024"
tldr: "Translating objective product catalog attributes into the subjective language used by customers for better search."
---

# Towards Translating Objective Product Attributes Into Customer Language

**Source**: [https://aclanthology.org/2024.naacl-industry.20/](https://aclanthology.org/2024.naacl-industry.20/)

**TLDR**: Translating objective product catalog attributes into the subjective language used by customers for better search.

## Abstract

AbstractWhen customers search online for a product they are not familiar with, their needs are often expressed through subjective product attributes, such as ”picture quality” for a TV or ”easy to clean” for a sofa. In contrast, the product catalog in online stores includes objective attributes such as ”screen resolution” or ”material”. In this work, we aim to find a link between the objective product catalog and the subjective needs of the customers, to help customers better understand the product space using their own words. We apply correlation-based methods to the store’s product catalog and product reviews in order to find the best potential links between objective and subjective attributes; next, Large Language Models (LLMs) reduce spurious correlations by incorporating common sense and world knowledge (e.g., picture quality is indeed affected by screen resolution, and 8k is the best one). We curate a dataset for this task and show that our combined approach outperforms correlation-only and causation-only approaches.