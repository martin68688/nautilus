---
title: "Improving Multi-lingual Alignment Through Soft Contrastive Learning"
source: "https://aclanthology.org/2024.naacl-srw.16/"
categories: ['knowledge-graph-and-information-extraction']
tags: ['multilingual-alignment', 'contrastive-learning', 'sentence-embeddings']
venue: "NAACL 2024"
tldr: "Improves multilingual sentence representation alignment through soft contrastive learning based on monolingual embedding similarity."
---

# Improving Multi-lingual Alignment Through Soft Contrastive Learning

**Source**: [https://aclanthology.org/2024.naacl-srw.16/](https://aclanthology.org/2024.naacl-srw.16/)

**TLDR**: Improves multilingual sentence representation alignment through soft contrastive learning based on monolingual embedding similarity.

## Abstract

AbstractMaking decent multi-lingual sentence representations is critical to achieve high performances in cross-lingual downstream tasks. In this work, we propose a novel method to align multi-lingual embeddings based on the similarity of sentences measured by a pre-trained mono-lingual embedding model. Given translation sentence pairs, we train a multi-lingual model in a way that the similarity between cross-lingual embeddings follows the similarity of sentences measured at the mono-lingual teacher model. Our method can be considered as contrastive learning with soft labels defined as the similarity between sentences. Our experimental results on five languages show that our contrastive loss with soft labels far outperforms conventional constrastive loss with hard labels in various benchmarks for bitext mining tasks and STS tasks. In addition, our method outperforms existing multi-lingual embeddings including LaBSE, for Tatoeba dataset.