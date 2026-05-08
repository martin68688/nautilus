---
title: "GenDecider: Integrating “None of the Candidates” Judgments in Zero-Shot Entity Linking Re-ranking"
source: "https://aclanthology.org/2024.naacl-short.22/"
categories: ['llm-knowledge-reasoning-retrieval', 'llm-ranking-benchmarking-adaptation']
tags: ['entity-linking', 'zero-shot', 're-ranking']
venue: "NAACL 2024"
tldr: "Presents a novel re-ranking approach for zero-shot entity linking that can detect when the correct entity is not among the candidates."
---

# GenDecider: Integrating “None of the Candidates” Judgments in Zero-Shot Entity Linking Re-ranking

**Source**: [https://aclanthology.org/2024.naacl-short.22/](https://aclanthology.org/2024.naacl-short.22/)

**TLDR**: Presents a novel re-ranking approach for zero-shot entity linking that can detect when the correct entity is not among the candidates.

## Abstract

AbstractWe introduce GenDecider, a novel re-ranking approach for Zero-Shot Entity Linking (ZSEL), built on the Llama model. It innovatively detects scenarios where the correct entity is not among the retrieved candidates, a common oversight in existing re-ranking methods. By autoregressively generating outputs based on the context of the entity mention and the candidate entities, GenDecider significantly enhances disambiguation, improving the accuracy and reliability of ZSEL systems, as demonstrated on the benchmark ZESHEL dataset. Our code is available at https://github.com/kangISU/GenDecider.