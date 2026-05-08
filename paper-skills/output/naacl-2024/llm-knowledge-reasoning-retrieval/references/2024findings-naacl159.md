---
title: "Why So Gullible? Enhancing the Robustness of Retrieval-Augmented Models against Counterfactual Noise"
source: "https://aclanthology.org/2024.findings-naacl.159/"
categories: ['llm-knowledge-reasoning-retrieval', 'llm-alignment-safety-detoxification']
tags: ['retrieval-augmented', 'robustness', 'counterfactual-noise']
venue: "NAACL 2024"
tldr: "Enhances the robustness of retrieval-augmented LMs against misleading information in retrieved documents."
---

# Why So Gullible? Enhancing the Robustness of Retrieval-Augmented Models against Counterfactual Noise

**Source**: [https://aclanthology.org/2024.findings-naacl.159/](https://aclanthology.org/2024.findings-naacl.159/)

**TLDR**: Enhances the robustness of retrieval-augmented LMs against misleading information in retrieved documents.

## Abstract

AbstractMost existing retrieval-augmented language models (LMs) assume a naive dichotomy within a retrieved document set: query-relevance and irrelevance. Our work investigates a more challenging scenario in which even the “relevant” documents may contain misleading or incorrect information, causing conflict among the retrieved documents and thereby negatively influencing model decisions as noise. We observe that existing LMs are highly brittle to the presence of conflicting information in both the fine-tuning and in-context few-shot learning scenarios. We propose approaches for handling knowledge conflicts among retrieved documents by explicitly fine-tuning a discriminator or prompting GPT-3.5 to elicit its discriminative capability. Our empirical results on open-domain QA show that these approaches significantly enhance model robustness. We also provide our findings on incorporating the fine-tuned discriminator’s decision into the in-context learning process, proposing a way to exploit the benefits of two disparate learning schemes. Alongside our findings, we provide MacNoise, a machine-generated, conflict-induced dataset to further encourage research in this direction.