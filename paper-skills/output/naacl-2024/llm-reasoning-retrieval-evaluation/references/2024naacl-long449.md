---
title: "IterCQR: Iterative Conversational Query Reformulation with Retrieval Guidance"
source: "https://aclanthology.org/2024.naacl-long.449/"
categories: ['llm-reasoning-retrieval-evaluation', 'large-language-model-evaluation-augmentation']
tags: ['conversational-search', 'query-reformulation', 'retrieval']
venue: "NAACL 2024"
tldr: "Presents an iterative method for conversational query reformulation guided by retrieval to improve search effectiveness."
---

# IterCQR: Iterative Conversational Query Reformulation with Retrieval Guidance

**Source**: [https://aclanthology.org/2024.naacl-long.449/](https://aclanthology.org/2024.naacl-long.449/)

**TLDR**: Presents an iterative method for conversational query reformulation guided by retrieval to improve search effectiveness.

## Abstract

AbstractConversational search aims to retrieve passages containing essential information to answer queries in a multi-turn conversation. In conversational search, reformulating context-dependent conversational queries into stand-alone forms is imperative to effectively utilize off-the-shelf retrievers. Previous methodologies for conversational query reformulation frequently depend on human-annotated rewrites.However, these manually crafted queries often result in sub-optimal retrieval performance and require high collection costs.To address these challenges, we propose **Iter**ative **C**onversational **Q**uery **R**eformulation (**IterCQR**), a methodology that conducts query reformulation without relying on human rewrites. IterCQR iteratively trains the conversational query reformulation (CQR) model by directly leveraging information retrieval (IR) signals as a reward.Our IterCQR training guides the CQR model such that generated queries contain necessary information from the previous dialogue context.Our proposed method shows state-of-the-art performance on two widely-used datasets, demonstrating its effectiveness on both sparse and dense retrievers. Moreover, IterCQR exhibits superior performance in challenging settings such as generalization on unseen datasets and low-resource scenarios.