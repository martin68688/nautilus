---
title: "Beyond Surface Similarity: Detecting Subtle Semantic Shifts in Financial Narratives"
source: "https://aclanthology.org/2024.findings-naacl.168/"
categories: ['financial-risk-narrative-analysis', 'zero-shot-few-shot-multimodal-optimization']
tags: ['semantic-similarity', 'financial-nlp', 'temporal-analysis']
venue: "NAACL 2024"
tldr: "Introduces a financial domain task to measure nuanced semantic shifts in narratives from the same company over time."
---

# Beyond Surface Similarity: Detecting Subtle Semantic Shifts in Financial Narratives

**Source**: [https://aclanthology.org/2024.findings-naacl.168/](https://aclanthology.org/2024.findings-naacl.168/)

**TLDR**: Introduces a financial domain task to measure nuanced semantic shifts in narratives from the same company over time.

## Abstract

AbstractIn this paper, we introduce the Financial-STS task, a financial domain-specific NLP task designed to measure the nuanced semantic similarity between pairs of financial narratives. These narratives originate from the financial statements of the same company but correspond to different periods, such as year-over-year comparisons. Measuring the subtle semantic differences between these paired narratives enables market stakeholders to gauge changes over time in the company’s financial and operational situations, which is critical for financial decision-making. We find that existing pretrained embedding models and LLM embeddings fall short in discerning these subtle financial narrative shifts. To address this gap, we propose an LLM-augmented pipeline specifically designed for the Financial-STS task. Evaluation on a human-annotated dataset demonstrates that our proposed method outperforms existing methods trained on classic STS tasks and generic LLM embeddings.