---
title: "Semi-Supervised Dialogue Abstractive Summarization via High-Quality Pseudolabel Selection"
source: "https://aclanthology.org/2024.naacl-long.333/"
categories: ['llm-evaluation-summarization-argument-extraction']
tags: ['summarization', 'dialogue', 'semi-supervised']
venue: "NAACL 2024"
tldr: "Proposes a method to select high-quality pseudo-summaries for semi-supervised dialogue summarization to reduce reliance on human-labeled data."
---

# Semi-Supervised Dialogue Abstractive Summarization via High-Quality Pseudolabel Selection

**Source**: [https://aclanthology.org/2024.naacl-long.333/](https://aclanthology.org/2024.naacl-long.333/)

**TLDR**: Proposes a method to select high-quality pseudo-summaries for semi-supervised dialogue summarization to reduce reliance on human-labeled data.

## Abstract

AbstractSemi-supervised dialogue summarization (SSDS) leverages model-generated summaries to reduce reliance on human-labeled data and improve the performance of summarization models. While addressing label noise, previous works on semi-supervised learning primarily focus on natural language understanding tasks, assuming each sample has a unique label. However, these methods are not directly applicable to SSDS, as it is a generative task, and each dialogue can be summarized in different ways. In this work, we propose a novel scoring approach, SiCF, which encapsulates three primary dimensions of summarization model quality: Semantic invariance (indicative of model confidence), Coverage (factual recall), and Faithfulness (factual precision). Using the SiCF score, we select unlabeled dialogues with high-quality generated summaries to train summarization models. Comprehensive experiments on three public datasets demonstrate the effectiveness of SiCF scores in uncertainty estimation and semi-supervised learning for dialogue summarization tasks. Our code is available at https://github.com/amazon-science/summarization-sicf-score.