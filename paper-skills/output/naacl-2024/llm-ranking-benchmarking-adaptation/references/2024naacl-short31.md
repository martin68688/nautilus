---
title: "Beyond Yes and No: Improving Zero-Shot LLM Rankers via Scoring Fine-Grained Relevance Labels"
source: "https://aclanthology.org/2024.naacl-short.31/"
categories: ['llm-ranking-benchmarking-adaptation', 'llm-knowledge-reasoning-retrieval']
tags: ['zero-shot-ranking', 'llm-ranker', 'relevance-labeling']
venue: "NAACL 2024"
tldr: "Improves zero-shot LLM rankers by prompting them to score fine-grained relevance labels beyond binary choices."
---

# Beyond Yes and No: Improving Zero-Shot LLM Rankers via Scoring Fine-Grained Relevance Labels

**Source**: [https://aclanthology.org/2024.naacl-short.31/](https://aclanthology.org/2024.naacl-short.31/)

**TLDR**: Improves zero-shot LLM rankers by prompting them to score fine-grained relevance labels beyond binary choices.

## Abstract

AbstractZero-shot text rankers powered by recent LLMs achieve remarkable ranking performance by simply prompting. Existing prompts for pointwise LLM rankers mostly ask the model to choose from binary relevance labels like “Yes” and “No”. However, the lack of intermediate relevance label options may cause the LLM to provide noisy or biased answers for documents that are partially relevant to the query. We propose to incorporate fine-grained relevance labels into the prompt for LLM rankers, enabling them to better differentiate among documents with different levels of relevance to the query and thus derive a more accurate ranking. We study two variants of the prompt template, coupled with different numbers of relevance levels. Our experiments on 8 BEIR data sets show that adding fine-grained relevance labels significantly improves the performance of LLM rankers.