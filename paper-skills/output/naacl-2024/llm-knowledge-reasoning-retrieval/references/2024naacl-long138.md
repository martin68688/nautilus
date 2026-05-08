---
title: "MILL: Mutual Verification with Large Language Models for Zero-Shot Query Expansion"
source: "https://aclanthology.org/2024.naacl-long.138/"
categories: ['llm-knowledge-reasoning-retrieval', 'zero-shot-few-shot-multimodal-optimization']
tags: ['query-expansion', 'zero-shot', 'retrieval']
venue: "NAACL 2024"
tldr: "MILL is a zero-shot query expansion method that uses large language models for mutual verification to improve query representation."
---

# MILL: Mutual Verification with Large Language Models for Zero-Shot Query Expansion

**Source**: [https://aclanthology.org/2024.naacl-long.138/](https://aclanthology.org/2024.naacl-long.138/)

**TLDR**: MILL is a zero-shot query expansion method that uses large language models for mutual verification to improve query representation.

## Abstract

AbstractQuery expansion, pivotal in search engines, enhances the representation of user information needs with additional terms. While existing methods expand queries using retrieved or generated contextual documents, each approach has notable limitations. Retrieval-based methods often fail to accurately capture search intent, particularly with brief or ambiguous queries. Generation-based methods, utilizing large language models (LLMs), generally lack corpus-specific knowledge and entail high fine-tuning costs. To address these gaps, we propose a novel zero-shot query expansion framework utilizing LLMs for mutual verification. Specifically, we first design a query-query-document generation method, leveraging LLMs’ zero-shot reasoning ability to produce diverse sub-queries and corresponding documents. Then, a mutual verification process synergizes generated and retrieved documents for optimal expansion. Our proposed method is fully zero-shot, and extensive experiments on three public benchmark datasets are conducted to demonstrate its effectiveness over existing methods. Our code is available online at https://github.com/Applied-Machine-Learning-Lab/MILL to ease reproduction.