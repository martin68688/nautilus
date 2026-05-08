---
title: "HIL: Hybrid Isotropy Learning for Zero-shot Performance in Dense retrieval"
source: "https://aclanthology.org/2024.naacl-long.437/"
categories: ['llm-knowledge-reasoning-retrieval', 'knowledge-graph-and-information-extraction']
tags: ['dense-retrieval', 'zero-shot', 'isotropy']
venue: "NAACL 2024"
tldr: "Proposes Hybrid Isotropy Learning (HIL) to improve the zero-shot performance of dense retrieval models like ColBERT."
---

# HIL: Hybrid Isotropy Learning for Zero-shot Performance in Dense retrieval

**Source**: [https://aclanthology.org/2024.naacl-long.437/](https://aclanthology.org/2024.naacl-long.437/)

**TLDR**: Proposes Hybrid Isotropy Learning (HIL) to improve the zero-shot performance of dense retrieval models like ColBERT.

## Abstract

AbstractAdvancements in dense retrieval models have brought ColBERT to prominence in Information Retrieval (IR) with its advanced interaction techniques.However, ColBERT is reported to frequently underperform in zero-shot scenarios, where traditional techniques such as BM25 still exceed it.Addressing this, we propose to balance representation isotropy and anisotropy for zero-shot model performance, based on our observations that isotropy can enhance cosine similarity computations and anisotropy may aid in generalizing to unseen data.Striking a balance between these isotropic and anisotropic qualities stands as a critical objective to refine model efficacy.Based on this, we present ours, a Hybrid Isotropy Learning (HIL) architecture that integrates isotropic and anisotropic representations.Our experiments with the BEIR benchmark show that our model significantly outperforms the baseline ColBERT model, highlighting the importance of harmonized isotropy in improving zero-shot retrieval performance.