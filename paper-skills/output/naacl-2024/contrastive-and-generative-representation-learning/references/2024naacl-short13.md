---
title: "SKICSE: Sentence Knowable Information Prompted by LLMs Improves Contrastive Sentence Embeddings"
source: "https://aclanthology.org/2024.naacl-short.13/"
categories: ['contrastive-and-generative-representation-learning']
tags: ['sentence-embeddings', 'contrastive-learning', 'llm-prompting', 'knowledge-augmentation']
venue: "NAACL 2024"
tldr: "Improving contrastive sentence embeddings by using LLM-generated knowable information as prompts."
---

# SKICSE: Sentence Knowable Information Prompted by LLMs Improves Contrastive Sentence Embeddings

**Source**: [https://aclanthology.org/2024.naacl-short.13/](https://aclanthology.org/2024.naacl-short.13/)

**TLDR**: Improving contrastive sentence embeddings by using LLM-generated knowable information as prompts.

## Abstract

AbstractContrastive learning, which utilizes positive pairs and in-batch negatives to optimize the loss objective, has been proven to be an effective method for learning sentence embeddings. However, we argue that the previous methods of constructing positive pairs only through dropout perturbation or entailment relation are limited. Since there is more sentence knowable information (SKI) to be mined, such as sentence external knowledge, semantic analysis, and grammatical description. In this work, we first hand-craft a simple and effective prompt template that is able to obtain the knowable information of input sentences from LLMs (e.g., LLaMA). Then we combine the original sentence and its knowable information to form a positive pair for contrastive learning. We evaluate our method on standard semantic textual similarity (STS) tasks. Experimental results show that our unsupervised and supervised models using BERTbase achieve an average of 78.65% and 82.45% Spearman’s correlation respectively, a 2.40% and 0.88% improvement compared to SimCSE. Our model outperforms the previous state-of-the-art model PromptBERT in both unsupervised and supervised settings and specifically yields a new state-of-the-art performance in supervised setting.