---
title: "Towards Improved Multi-Source Attribution for Long-Form Answer Generation"
source: "https://aclanthology.org/2024.naacl-long.216/"
categories: ['llm-knowledge-reasoning-retrieval', 'llm-attribution-verification']
tags: ['attribution', 'long-form-generation', 'hallucination-reduction', 'retrieval-augmented']
venue: "NAACL 2024"
tldr: "Aims to improve multi-source attribution in long-form answer generation by LLMs to reduce hallucinations and increase verifiability."
---

# Towards Improved Multi-Source Attribution for Long-Form Answer Generation

**Source**: [https://aclanthology.org/2024.naacl-long.216/](https://aclanthology.org/2024.naacl-long.216/)

**TLDR**: Aims to improve multi-source attribution in long-form answer generation by LLMs to reduce hallucinations and increase verifiability.

## Abstract

AbstractTeaching large language models (LLMs) to generate text with attribution to evidence sources can reduce hallucinations, improve verifiability in question answering systems (QA), and increase reliability of retrieval augmented LLMs. Despite gaining increasing popularity for usage in QA systems and search engines, current LLMs struggle with attribution for long-form responses which require reasoning over multiple evidence sources. To address this, in this paper we aim to improve the attribution capability of LLMs for long-form answer generation to multiple sources, with multiple citations per sentence. However, data for training multi-source attributable QA systems is difficult and expensive to annotate, and therefore scarce. To overcome this challenge, we transform existing QA datasets for this task (MultiAttr), and empirically demonstrate, on a wide range of attribution benchmark datasets, that fine-tuning on MultiAttr provides significant improvements over training only on the target QA domain. Lastly, to fill a gap in existing benchmarks, we present a multi-source attribution dataset containing multi-paragraph answers, PolitiICite, based on PolitiFact articles that discuss events closely related to implementation statuses of election promises.