---
title: "Towards Better Generalization in Open-Domain Question Answering by Mitigating Context Memorization"
source: "https://aclanthology.org/2024.findings-naacl.48/"
categories: ['llm-knowledge-reasoning-retrieval', 'large-language-model-evaluation-augmentation', 'knowledge-conflict-diagnostic-temporal-reasoning']
tags: ['open-domain-qa', 'context-memorization', 'knowledge-updates']
venue: "NAACL 2024"
tldr: "Mitigates context memorization in open-domain QA for better generalization to evolving knowledge."
---

# Towards Better Generalization in Open-Domain Question Answering by Mitigating Context Memorization

**Source**: [https://aclanthology.org/2024.findings-naacl.48/](https://aclanthology.org/2024.findings-naacl.48/)

**TLDR**: Mitigates context memorization in open-domain QA for better generalization to evolving knowledge.

## Abstract

AbstractOpen-domain Question Answering (OpenQA) aims at answering factual questions with an external large-scale knowledge corpus. However, real-world knowledge is not static; it updates and evolves continually. Such a dynamic characteristic of knowledge poses a vital challenge for these models, as the trained models need to constantly adapt to the latest information to make sure that the answers remain accurate. In addition, it is still unclear how well an OpenQA model can transfer to completely new knowledge domains. In this paper, we investigate the generalization performance of a retrieval-augmented QA model in two specific scenarios: 1) adapting to updated versions of the same knowledge corpus; 2) switching to completely different knowledge domains. We observe that the generalization challenges of OpenQA models stem from the reader’s over-reliance on memorizing the knowledge from the external corpus, which hinders the model from generalizing to a new knowledge corpus. We introduce Corpus-Invariant Tuning (CIT), a simple but effective training strategy, to mitigate the knowledge over-memorization by controlling the likelihood of retrieved contexts during training. Extensive experimental results on multiple OpenQA benchmarks show that CIT achieves significantly better generalizability without compromising the model’s performance in its original corpus and domain.